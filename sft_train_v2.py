#!/usr/bin/env python3
import argparse, json, math, os, random
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
import tiktoken
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm

from model_moe import MoETransformer

IGNORE_INDEX = -100


class IndexedSFTDataset(Dataset):
    """In-memory index over inspectable JSONL SFT examples."""
    def __init__(self, data_dir, split):
        path = os.path.join(data_dir, f"{split}_examples.jsonl")
        self.examples = []
        with open(path, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    example = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc

                for field in ("prompt", "response", "source"):
                    if not isinstance(example.get(field), str) or not example[field].strip():
                        raise ValueError(
                            f"{path}:{line_number} has an invalid {field!r} field"
                        )
                self.examples.append({
                    "prompt": example["prompt"],
                    "response": example["response"],
                    "source": example["source"],
                })

        if not self.examples:
            raise RuntimeError(f"No examples found in {path}")
        self.sources = [example["source"] for example in self.examples]
        self.source_counts = Counter(self.sources)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def build_source_sampler(dataset, data_config, seed):
    """Sample sources by configured percentage, then rows uniformly per source."""
    source_mix = data_config.get("source_mix_percent")
    if not source_mix:
        return None, {}

    configured_sources = set(source_mix)
    prepared_sources = set(dataset.source_counts)
    missing = sorted(prepared_sources - configured_sources)
    if missing:
        raise ValueError(
            "source_mix_percent is missing prepared sources: " + ", ".join(missing)
        )

    percentages = {}
    for source, value in source_mix.items():
        percentage = float(value)
        if percentage < 0:
            raise ValueError(
                f"source_mix_percent[{source!r}] must be non-negative"
            )
        if percentage > 0 and source not in prepared_sources:
            raise ValueError(
                f"source_mix_percent gives {source!r} {percentage:g}%, but that "
                "source has no prepared examples"
            )
        percentages[source] = percentage

    total_percentage = sum(percentages.values())
    if not math.isclose(total_percentage, 100.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(
            f"source_mix_percent must sum to 100, got {total_percentage:g}"
        )

    per_source_weight = {}
    expected_shares = {}
    for source, count in dataset.source_counts.items():
        percentage = percentages[source]
        if percentage == 0:
            continue
        # WeightedRandomSampler assigns weight per row. Dividing a source's
        # probability mass equally across its rows makes the source-level draw
        # probability match the configured percentage.
        per_source_weight[source] = (percentage / 100.0) / count
        expected_shares[source] = percentage / 100.0

    sample_weights = torch.tensor(
        [per_source_weight.get(source, 0.0) for source in dataset.sources],
        dtype=torch.double,
    )
    if not torch.any(sample_weights > 0):
        raise ValueError("source_mix_percent gives zero probability to all examples")
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True,
        generator=generator,
    )
    return sampler, expected_shares


class SFTCollator:
    """Tokenize complete JSONL examples, causally shift, and right-pad."""

    def __init__(
        self, encoding, max_seq_length, system_prompt_dropout=0.0,
        pad_to_multiple_of=8
    ):
        if not 0.0 <= system_prompt_dropout <= 1.0:
            raise ValueError("system_prompt_dropout must be between 0 and 1")
        self.enc = encoding
        self.eot = int(encoding.eot_token)
        self.max_seq = int(max_seq_length)
        self.system_prompt_dropout = float(system_prompt_dropout)
        self.multiple = pad_to_multiple_of

    @staticmethod
    def remove_system_prompt(prompt: str) -> str:
        system_header = "### SYSTEM:\n"
        instruction_header = "\n\n### Instruction:\n"
        if not prompt.startswith(system_header):
            return prompt
        marker = prompt.find(instruction_header, len(system_header))
        if marker < 0:
            return prompt
        return "### Instruction:\n" + prompt[marker + len(instruction_header):]

    def __call__(self, examples):
        xs, ys = [], []
        for example in examples:
            prompt = example["prompt"]
            response = example["response"].strip()
            if (
                self.system_prompt_dropout > 0.0
                and random.random() < self.system_prompt_dropout
            ):
                prompt = self.remove_system_prompt(prompt)

            # Prevent tiktoken crash on unexpected special substrings
            prompt_tokens = self.enc.encode_ordinary(prompt)
            response_tokens = self.enc.encode_ordinary(response)
            prompt_len = len(prompt_tokens)
            if not response_tokens:
                continue
            full_text = prompt + response
            full_tokens = self.enc.encode_ordinary(full_text)
            # Check if the prefix matches exactly. If there's a BPE boundary merge
            # at the junction, full_tokens[:prompt_len] won't match prompt_tokens
            if full_tokens[:prompt_len] != prompt_tokens:
                tokens = prompt_tokens + self.enc.encode_ordinary(response) + [self.eot]
                labels = [IGNORE_INDEX] * prompt_len + tokens[prompt_len:]
            else:
                tokens = full_tokens + [self.eot]
                labels = [IGNORE_INDEX] * prompt_len + full_tokens[prompt_len:] + [self.eot]
            # Shift causally:
            # X: predicts next token starting from the last prompt token
            # Y: target label for next token
            if len(tokens) > self.max_seq:
                continue
            xs.append(torch.tensor(tokens[:-1], dtype=torch.long))
            ys.append(torch.tensor(labels[1:], dtype=torch.long))

        if not xs:
            raise RuntimeError(
                "Batch has no usable examples; responses may be empty or exceed max_seq_length."
            )

        max_len = max(len(x) for x in xs)
        T = max_len
        if self.multiple:
            m = int(self.multiple)
            T = ((max_len + m - 1) // m) * m
            T = min(T, self.max_seq - 1)

        batch_size = len(xs)
        X = torch.full((batch_size, T), self.eot, dtype=torch.long)
        Y = torch.full((batch_size, T), IGNORE_INDEX, dtype=torch.long)
        valid_mask = torch.zeros((batch_size, T), dtype=torch.bool)

        for i, (x, y) in enumerate(zip(xs, ys)):
            # Safeguard slice length up to T
            n = min(len(x), T)
            X[i, :n] = x[:n]
            Y[i, :n] = y[:n]
            valid_mask[i, :n] = True
        return {"input_ids": X, "labels": Y, "valid_mask": valid_mask}

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
        logits, _ = model(
            batch["input_ids"], valid_mask=batch["valid_mask"]
        )
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


def save_ckpt(
    accelerator, model, optimizer, scheduler, step,
    stage_sup_tokens, lifetime_sup_tokens, cfg, path
):
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        torch.save({
            "model": accelerator.unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "iter": step,
            "stage_supervised_tokens": stage_sup_tokens,
            "lifetime_supervised_tokens": lifetime_sup_tokens,
            # Backward compatibility for existing checkpoint readers.
            "supervised_tokens": lifetime_sup_tokens,
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

    wandb_run = None
    wandb_cfg = cfg.get("wandb_config", {})
    if accelerator.is_main_process and wandb_cfg.get("enabled", False):
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                "W&B logging is enabled but wandb is not installed. "
                "Run: pip install wandb"
            ) from exc
        wandb.login()
        wandb_run = wandb.init(
            project=wandb_cfg.get("project", "customlm-sft"),
            config=tc,
        )

    enc = tiktoken.get_encoding("gpt2")
    train_collate = SFTCollator(
        enc, dc["max_seq_length"], system_prompt_dropout=0.10,
        pad_to_multiple_of=dc.get("pad_to_multiple_of", 8)
    )
    val_collate = SFTCollator(
        enc, dc["max_seq_length"], system_prompt_dropout=0.0,
        pad_to_multiple_of=dc.get("pad_to_multiple_of", 8)
    )
    train_ds = IndexedSFTDataset(dc["data_dir"], "train")
    val_ds = IndexedSFTDataset(dc["data_dir"], "val")
    train_sampler, expected_source_shares = build_source_sampler(
        train_ds, dc, seed
    )
    num_workers = int(dc.get("num_workers", 2))
    train_loader = DataLoader(
        train_ds, batch_size=tc["batch_size"], sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=num_workers, pin_memory=True,
        collate_fn=train_collate, drop_last=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=tc.get("eval_batch_size", tc["batch_size"]), shuffle=False,
        num_workers=num_workers, pin_memory=True,
        collate_fn=val_collate, drop_last=False,
        persistent_workers=num_workers > 0,
    )
    if expected_source_shares:
        accelerator.print(
            "configured source example shares: "
            + ", ".join(
                f"{source}={share:.1%}"
                for source, share in expected_source_shares.items()
            )
        )

    model = build_model(cfg)
    freeze_n = int(tc.get("freeze_layers", 0))
    for i in range(min(freeze_n, len(model.blocks))):
        for p in model.blocks[i].parameters(): p.requires_grad = False

    step, stage_sup, lifetime_sup = 0, 0, 0
    optimizer = build_optimizer(model, tc)
    if cc.get("resume_from"):
        state = torch.load(cc["resume_from"], map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        for param_group in optimizer.param_groups:
            param_group["lr"] = tc["lr"]
            param_group["initial_lr"] = tc["lr"]
        # Build a NEW scheduler AFTER optimizer loading/LR override
        scheduler = build_scheduler(optimizer, tc)
        lifetime_sup = int(state.get(
            "lifetime_supervised_tokens", state.get("supervised_tokens", 0)
        ))
    else:
        load_base(model, cc["base_model_path"])
        optimizer = build_optimizer(model, tc)
        scheduler = build_scheduler(optimizer, tc)
    accelerator.print(
        f"train examples={len(train_ds):,}, val examples={len(val_ds):,}, "
        f"trainable params={sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n"
        f"LR={tc['lr']:.2e}, minLR={tc['min_lr']:.2e}, warmup={tc['warmup_iters']}\n"
        f"stage_step={step}, stage_supervised_tokens={stage_sup}, "
        f"lifetime_supervised_tokens={lifetime_sup}"
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
    token_progress = max_sup > 0
    pbar = tqdm(
        total=max_sup if token_progress else max_iters,
        initial=stage_sup if token_progress else step,
        desc="SFT stage",
        unit="tok" if token_progress else "step",
        disable=not accelerator.is_local_main_process,
    )
    while step < max_iters:
        for batch in train_loader:  # reshuffles on every new epoch
            with accelerator.accumulate(model):
                x, y = batch["input_ids"], batch["labels"]
                logits, aux_loss = model(x, valid_mask=batch["valid_mask"])
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
                batch_sup = int(accelerator.reduce(local_n, reduction="sum").item())
                stage_sup += batch_sup
                lifetime_sup += batch_sup
                if token_progress:
                    pbar.update(batch_sup)
                recent_ce += ce.detach().float().item()
                recent_total += loss.detach().float().item()
                recent_n += 1

            if accelerator.sync_gradients:
                step += 1
                if step % log_every == 0:
                    train_ce = recent_ce / recent_n
                    train_loss = recent_total / recent_n
                    learning_rate = optimizer.param_groups[0]["lr"]
                    accelerator.print(
                        f"step={step:,} ce={train_ce:.4f} "
                        f"loss={train_loss:.4f} "
                        f"lr={learning_rate:.3e} "
                        f"stage_sup_tokens={stage_sup:,} lifetime_sup_tokens={lifetime_sup:,}"
                    )
                    v = evaluate(model, val_loader, accelerator, int(tc.get("eval_iters", 50)))
                    val_ppl = math.exp(min(v, 20))
                    accelerator.print(f"  val_response_ce={v:.4f}, val_ppl={val_ppl:.2f}")

                    if wandb_run is not None:
                        wandb_run.log({
                            "train/response_ce": train_ce,
                            "train/loss": train_loss,
                            "train/learning_rate": learning_rate,
                            "train/stage_supervised_tokens": stage_sup,
                            "train/lifetime_supervised_tokens": lifetime_sup,
                            "eval/response_ce": v,
                            "eval/perplexity": val_ppl,
                        }, step=step)

                    recent_ce = recent_total = 0.0; recent_n = 0

                    router_stats = model.get_router_stats()
                    if accelerator.is_main_process and router_stats:
                        avg_cv = sum(s["cv"] for _, s in router_stats) / len(router_stats)
                        worst_layer, worst = max(router_stats, key=lambda x: x[1]["cv"])
                        accelerator.print(  #average coefficient_of_variation-> std deviation expert usage / mean expert usage 
                                            f"avg_cv={avg_cv:.2f} "  
                                            f"worst=Layer{worst_layer} "
                                            f"min={worst['min']*100:.1f}% "
                                            f"max={worst['max']*100:.1f}% "
                                            f"cv={worst['cv']:.2f}")
                if step % save_every == 0:
                    save_ckpt(accelerator, model, optimizer, scheduler, step,
                              stage_sup, lifetime_sup, cfg,
                              os.path.join(cc["output_dir"], f"sft_ckpt_1c_{step:07d}.pt"))

                if not token_progress:
                    pbar.update(1)
                if step >= max_iters or (max_sup > 0 and stage_sup >= max_sup):
                    break
        if step >= max_iters or (max_sup > 0 and stage_sup >= max_sup):
            break 
    pbar.close()
    save_ckpt(accelerator, model, optimizer, scheduler, step,
              stage_sup, lifetime_sup, cfg,
              os.path.join(cc["output_dir"], "sft_ckpt_1c_final.pt"))
    accelerator.print(
        f"done: stage_steps={step:,}, stage_supervised_tokens={stage_sup:,}, "
        f"lifetime_supervised_tokens={lifetime_sup:,}"
    )
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="sft_config.json")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args(); main(a.config, a.seed)
