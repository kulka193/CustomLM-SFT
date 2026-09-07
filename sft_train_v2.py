#!/usr/bin/env python3
import argparse, json, math, os, random

import numpy as np
import torch
import torch.nn.functional as F
import tiktoken
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model_moe import MoETransformer

IGNORE_INDEX = -100


class IndexedSFTDataset(Dataset):
    """One Dataset item == one complete SFT example."""
    def __init__(self, data_dir, split):
        self.tokens = np.memmap(os.path.join(data_dir, f"{split}_tokens.bin"),
                                dtype=np.int32, mode="r")
        self.labels = np.memmap(os.path.join(data_dir, f"{split}_labels.bin"),
                                dtype=np.int32, mode="r")
        self.offsets = np.load(os.path.join(data_dir, f"{split}_offsets.npy"),
                               mmap_mode="r")
        if len(self.tokens) != len(self.labels):
            raise ValueError("token/label file length mismatch")

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, i):
        start, end = map(int, self.offsets[i])
        # copies avoid read-only memmap tensor issues
        t = torch.from_numpy(np.asarray(self.tokens[start:end], dtype=np.int64).copy())
        l = torch.from_numpy(np.asarray(self.labels[start:end], dtype=np.int64).copy())
        return t, l


class SFTCollator:
    """Causal shift + dynamic RIGHT padding.

    Stored:
      tokens = [prompt, prompt..., answer..., EOT]
      labels = [-100,-100..., answer..., EOT]

    Model training:
      X = tokens[:-1]
      Y = labels[1:]

    Thus the logit after the last prompt token predicts the FIRST answer token.
    """
    def __init__(self, eot_token, max_seq_length, pad_to_multiple_of=8):
        self.eot = int(eot_token)
        self.max_seq = int(max_seq_length)
        self.multiple = pad_to_multiple_of

    def __call__(self, examples):
        xs, ys = [], []
        for tokens, semantic_labels in examples:
            assert len(tokens) == len(semantic_labels)
            if len(tokens) < 2:
                continue
            x = tokens[:-1]
            y = semantic_labels[1:]
            if len(x) > self.max_seq - 1:
                raise RuntimeError("prepared example exceeds max_seq_length")
            if torch.any(y != IGNORE_INDEX):
                xs.append(x); ys.append(y)

        if not xs:
            raise RuntimeError("batch has no supervised tokens")

        T = max(len(x) for x in xs)
        if self.multiple:
            m = int(self.multiple)
            T = min(self.max_seq - 1, ((T + m - 1) // m) * m)

        # Your model has no padding attention mask. RIGHT padding is safe here:
        # causal real-token positions cannot attend to future pad positions.
        X = torch.full((len(xs), T), self.eot, dtype=torch.long)
        Y = torch.full((len(xs), T), IGNORE_INDEX, dtype=torch.long)
        for i, (x, y) in enumerate(zip(xs, ys)):
            n = len(x)
            X[i, :n] = x
            Y[i, :n] = y
        return {"input_ids": X, "labels": Y}


def build_model(cfg):
    m = cfg["model_config"]
    return MoETransformer(
        vocab_size=m["vocab_size"], d_model=m["d_model"], num_heads=m["num_heads"],
        d_ff=m["d_ff"], num_layers=m["num_layers"], num_experts=m["num_experts"],
        max_seq_len=m["block_size"], top_k=m.get("top_k", 2), dropout=m["dropout"]
    )


def build_optimizer(model, tc):
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    decay = [p for _, p in named if p.dim() >= 2]
    no_decay = [p for _, p in named if p.dim() < 2]
    return AdamW(
        [{"params": decay, "weight_decay": tc["weight_decay"]},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=tc["lr"], betas=(tc["beta1"], tc["beta2"])
    )


def build_scheduler(optimizer, tc):
    max_iters = int(tc["max_iters"])
    warmup = min(int(tc["warmup_iters"]), max_iters - 1)
    lr, min_lr = float(tc["lr"]), float(tc["min_lr"])
    if warmup <= 0:
        return CosineAnnealingLR(optimizer, T_max=max_iters, eta_min=min_lr)
    warm = LinearLR(optimizer, start_factor=max(min_lr/lr, 0.01),
                    end_factor=1.0, total_iters=warmup)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, max_iters-warmup), eta_min=min_lr)
    return SequentialLR(optimizer, [warm, cosine], milestones=[warmup])


def load_base(model, path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=True)


@torch.no_grad()
def evaluate(model, loader, accelerator, eval_iters):
    model.eval()
    loss_sum = torch.zeros((), device=accelerator.device)
    token_count = torch.zeros((), device=accelerator.device)
    for i, batch in enumerate(loader):
        logits, _ = model(batch["input_ids"])
        y = batch["labels"]
        loss_sum += F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1),
            ignore_index=IGNORE_INDEX, reduction="sum"
        )
        token_count += (y != IGNORE_INDEX).sum()
        if i + 1 >= eval_iters:
            break
    loss_sum = accelerator.reduce(loss_sum, reduction="sum")
    token_count = accelerator.reduce(token_count, reduction="sum")
    out = (loss_sum / token_count.clamp_min(1)).item()
    model.train()
    return out


def save_ckpt(accelerator, model, optimizer, scheduler, step, sup_tokens, cfg, path):
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        torch.save({
            "model": accelerator.unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "iter": step,
            "supervised_tokens": sup_tokens,
            "config": cfg,
        }, path)
        accelerator.print(f"saved {path}")


def main(config_path, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    with open(config_path) as f:
        cfg = json.load(f)
    tc, dc, cc, dist = (cfg["training_config"], cfg["data_config"],
                        cfg["checkpoint_config"], cfg["distributed_config"])
    os.makedirs(cc["output_dir"], exist_ok=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=tc["gradient_accumulation_steps"],
        mixed_precision=dist.get("mixed_precision", "bf16"),
        kwargs_handlers=[DistributedDataParallelKwargs(
            find_unused_parameters=dist.get("find_unused_parameters", False))]
    )

    enc = tiktoken.get_encoding("gpt2")
    collate = SFTCollator(enc.eot_token, dc["max_seq_length"], dc.get("pad_to_multiple_of", 8))
    train_ds = IndexedSFTDataset(dc["data_dir"], "train")
    val_ds = IndexedSFTDataset(dc["data_dir"], "val")
    train_loader = DataLoader(
        train_ds, batch_size=tc["batch_size"], shuffle=True,
        num_workers=dc.get("num_workers", 2), pin_memory=True,
        collate_fn=collate, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=tc.get("eval_batch_size", tc["batch_size"]), shuffle=False,
        num_workers=dc.get("num_workers", 2), pin_memory=True,
        collate_fn=collate, drop_last=False
    )

    model = build_model(cfg)
    freeze_n = int(tc.get("freeze_layers", 0))
    for i in range(min(freeze_n, len(model.blocks))):
        for p in model.blocks[i].parameters(): p.requires_grad = False

    optimizer = build_optimizer(model, tc)
    scheduler = build_scheduler(optimizer, tc)
    step, total_sup = 0, 0

    if cc.get("resume_from"):
        state = torch.load(cc["resume_from"], map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        step = int(state.get("iter", 0))
        total_sup = int(state.get("supervised_tokens", 0))
    else:
        load_base(model, cc["base_model_path"])

    accelerator.print(
        f"train examples={len(train_ds):,}, val examples={len(val_ds):,}, "
        f"trainable params={sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n"
        f"LR={tc['lr']:.2e}, minLR={tc['min_lr']:.2e}, warmup={tc['warmup_iters']}"
    )

    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )
    model.train(); optimizer.zero_grad(set_to_none=True)

    max_iters = int(tc["max_iters"])
    max_sup = int(tc.get("max_sup_tokens", 0))
    log_every = int(tc.get("log_interval", 200))
    save_every = int(tc.get("save_interval", 5000))
    aux_w = float(tc.get("aux_loss_weight", 1e-3))
    recent_ce = recent_total = 0.0; recent_n = 0
    pbar = tqdm(
        total=max_iters,
        initial=step,
        desc="SFT",
        unit="step",
        disable=not accelerator.is_local_main_process,
    )
    while step < max_iters:
        for batch in tqdm(train_loader):  # reshuffles on every new epoch
            with accelerator.accumulate(model):
                x, y = batch["input_ids"], batch["labels"]
                logits, aux_loss = model(x)
                ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                                     ignore_index=IGNORE_INDEX)
                loss = ce + (aux_w * aux_loss if aux_loss is not None else 0.0)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), tc["grad_clip"])
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if accelerator.sync_gradients:
                    scheduler.step()

                # Count globally across GPUs.
                local_n = (y != IGNORE_INDEX).sum().to(accelerator.device)
                total_sup += int(accelerator.reduce(local_n, reduction="sum").item())
                recent_ce += ce.detach().float().item()
                recent_total += loss.detach().float().item()
                recent_n += 1

            if accelerator.sync_gradients:
                step += 1
                pbar.update(1)
                pbar.set_postfix(
                    ce=f"{ce.detach().float().item():.4f}",
                    loss=f"{loss.detach().float().item():.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                    sup_tokens=f"{total_sup:,}",
                )
                if step % log_every == 0:
                    accelerator.print(
                        f"step={step:,} ce={recent_ce/recent_n:.4f} "
                        f"loss={recent_total/recent_n:.4f} "
                        f"lr={optimizer.param_groups[0]['lr']:.3e} sup_tokens={total_sup:,}"
                    )
                    recent_ce = recent_total = 0.0; recent_n = 0
                    v = evaluate(model, val_loader, accelerator, int(tc.get("eval_iters", 50)))
                    accelerator.print(f"  val_response_ce={v:.4f}, val_ppl={math.exp(min(v,20)):.2f}")
                    pbar.set_postfix(
                            ce=f"{ce.detach().float().item():.4f}",
                            val_ce=f"{v:.4f}",
                            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                            sup_tokens=f"{total_sup:,}",
                        )
                if step % save_every == 0:
                    save_ckpt(accelerator, model, optimizer, scheduler, step, total_sup, cfg,
                              os.path.join(cc["output_dir"], f"sft_ckpt_1b_{step:07d}.pt"))

                if step >= max_iters or (max_sup > 0 and total_sup >= max_sup):
                    break
        if step >= max_iters or (max_sup > 0 and total_sup >= max_sup):
            break
    pbar.close()
    save_ckpt(accelerator, model, optimizer, scheduler, step, total_sup, cfg,
              os.path.join(cc["output_dir"], "sft_ckpt_1b_final.pt"))
    accelerator.print(f"done: steps={step:,}, supervised_tokens={total_sup:,}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="sft_config.json")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args(); main(a.config, a.seed)
