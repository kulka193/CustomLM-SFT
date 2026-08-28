import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dpo_prepare import build_preference_datasets, pad_tensor_rows, preference_collate_fn
from model_moe import MoETransformer


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_checkpoint_state(path: str) -> dict:
    raw = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(raw, dict) and "model" in raw:
        return raw["model"]
    return raw


def load_model_from_checkpoint(model_cfg: dict, checkpoint_path: str) -> MoETransformer:
    model = MoETransformer(
            vocab_size=model_cfg["vocab_size"], d_model=model_cfg["d_model"],
            num_heads=model_cfg["num_heads"], d_ff=model_cfg["d_ff"],
            num_layers=model_cfg["num_layers"], num_experts=model_cfg["num_experts"],
            max_seq_len=model_cfg["block_size"], top_k=2, dropout=model_cfg.get("dropout", 0.1))
    state = load_checkpoint_state(checkpoint_path)
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch. Missing={missing}, unexpected={unexpected}")
    return model


def build_optimizer(model: MoETransformer, lr: float, weight_decay: float, beta1: float, beta2: float) -> AdamW:
    param_dict = {name: param for name, param in model.named_parameters() if param.requires_grad}
    decay = [param for _, param in param_dict.items() if param.dim() >= 2]
    no_decay = [param for _, param in param_dict.items() if param.dim() < 2]
    return AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(beta1, beta2),
    )


def maybe_trim_sequence(prompt_ids: list[int], response_ids: list[int], block_size: int) -> tuple[list[int], list[int]]:
    if len(prompt_ids) + len(response_ids) <= block_size:
        return prompt_ids, response_ids

    overflow = len(prompt_ids) + len(response_ids) - block_size
    if overflow >= len(prompt_ids):
        prompt_ids = prompt_ids[-1:]
    else:
        prompt_ids = prompt_ids[overflow:]
    return prompt_ids, response_ids


def build_logprob_tensors(
    prompts: list[list[int]],
    responses: list[list[int]],
    pad_id: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs = []
    targets = []
    masks = []

    for prompt_ids, response_ids in zip(prompts, responses):
        prompt_ids, response_ids = maybe_trim_sequence(prompt_ids, response_ids, block_size)
        full_ids = prompt_ids + response_ids
        if len(full_ids) < 2:
            full_ids = full_ids + [pad_id]
            response_ids = response_ids + [pad_id]

        input_ids = full_ids[:-1]
        target_ids = full_ids[1:]
        mask = torch.zeros(len(target_ids), dtype=torch.float32)
        start = max(len(prompt_ids) - 1, 0)
        if start < len(mask):
            mask[start:] = 1.0

        inputs.append(input_ids)
        targets.append(target_ids)
        masks.append(mask.tolist())

    input_batch, _ = pad_tensor_rows(inputs, pad_id)
    target_batch, _ = pad_tensor_rows(targets, pad_id)
    max_len = input_batch.size(1)
    mask_batch = torch.zeros((len(masks), max_len), dtype=torch.float32)
    for i, mask in enumerate(masks):
        mask_batch[i, : len(mask)] = torch.tensor(mask, dtype=torch.float32)

    return input_batch, target_batch, mask_batch


def reduce_masked_logprobs(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, length_normalize: bool) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    token_log_probs = token_log_probs * mask
    if length_normalize:
        denom = mask.sum(dim=1).clamp_min(1.0)
        return token_log_probs.sum(dim=1) / denom
    return token_log_probs.sum(dim=1)


def compute_response_logprobs(
    model: MoETransformer,
    prompts: list[list[int]],
    responses: list[list[int]],
    pad_id: int,
    block_size: int,
    length_normalize: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_batch, target_batch, mask_batch = build_logprob_tensors(prompts, responses, pad_id, block_size)
    device = next(model.parameters()).device
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    mask_batch = mask_batch.to(device)

    logits, aux_loss = model(input_batch)
    reduced = reduce_masked_logprobs(logits, target_batch, mask_batch, length_normalize=length_normalize)
    return reduced, aux_loss


def calculate_dpo_loss(
    policy_chosen_logprob: torch.Tensor,
    policy_rejected_logprob: torch.Tensor,
    ref_chosen_logprob: torch.Tensor,
    ref_rejected_logprob: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = beta * (
        (policy_chosen_logprob - policy_rejected_logprob)
        - (ref_chosen_logprob - ref_rejected_logprob)
    )
    loss = -F.logsigmoid(logits).mean()
    reward_accuracy = (logits > 0).float().mean()
    reward_margin = logits.mean()
    chosen_margin = (policy_chosen_logprob - policy_rejected_logprob).mean()
    return loss, reward_accuracy, reward_margin, chosen_margin


@torch.no_grad()
def evaluate(
    policy_model: MoETransformer,
    reference_model: MoETransformer,
    val_loader: DataLoader | None,
    pad_id: int,
    block_size: int,
    beta: float,
    length_normalize: bool,
) -> dict:
    if val_loader is None:
        return {"val_loss": None, "val_reward_accuracy": None, "val_reward_margin": None}

    policy_model.eval()
    reference_model.eval()
    losses = []
    accuracies = []
    margins = []

    for batch in val_loader:
        pi_chosen, _ = compute_response_logprobs(
            policy_model, batch["prompt_ids"], batch["chosen_ids"], pad_id, block_size, length_normalize
        )
        pi_rejected, _ = compute_response_logprobs(
            policy_model, batch["prompt_ids"], batch["rejected_ids"], pad_id, block_size, length_normalize
        )
        ref_chosen, _ = compute_response_logprobs(
            reference_model, batch["prompt_ids"], batch["chosen_ids"], pad_id, block_size, length_normalize
        )
        ref_rejected, _ = compute_response_logprobs(
            reference_model, batch["prompt_ids"], batch["rejected_ids"], pad_id, block_size, length_normalize
        )

        loss, reward_accuracy, reward_margin, _ = calculate_dpo_loss(
            pi_chosen, pi_rejected, ref_chosen, ref_rejected, beta
        )
        losses.append(loss.item())
        accuracies.append(reward_accuracy.item())
        margins.append(reward_margin.item())

    policy_model.train()
    return {
        "val_loss": float(np.mean(losses)) if losses else None,
        "val_reward_accuracy": float(np.mean(accuracies)) if accuracies else None,
        "val_reward_margin": float(np.mean(margins)) if margins else None,
    }


def train(config_path: str = "rlhf_config.json") -> dict:
    cfg = load_json(config_path)
    model_cfg = cfg["model_config"]
    train_cfg = cfg["dpo_config"]
    ckpt_cfg = cfg["checkpoint_config"]
    runtime_cfg = cfg["runtime_config"]

    seed_everything(runtime_cfg.get("seed", 42))
    os.makedirs(ckpt_cfg["policy_output_dir"], exist_ok=True)

    train_ds, val_ds, enc = build_preference_datasets(cfg)
    pad_id = enc.eot_token

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        collate_fn=preference_collate_fn,
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=train_cfg["batch_size"],
            shuffle=False,
            collate_fn=preference_collate_fn,
        )
        if val_ds
        else None
    )

    accelerator = Accelerator(
        mixed_precision=runtime_cfg.get("mixed_precision", "bf16"),
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
    )

    policy_model = load_model_from_checkpoint(model_cfg, ckpt_cfg["sft_checkpoint_path"])
    if ckpt_cfg.get("resume_policy_from"):
        resume_state = load_checkpoint_state(ckpt_cfg["resume_policy_from"])
        policy_model.load_state_dict(resume_state, strict=True)

    reference_model = load_model_from_checkpoint(model_cfg, ckpt_cfg["sft_checkpoint_path"])
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad = False

    optimizer = build_optimizer(
        policy_model,
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        beta1=train_cfg["beta1"],
        beta2=train_cfg["beta2"],
    )

    if val_loader is not None:
        policy_model, optimizer, train_loader, val_loader = accelerator.prepare(
            policy_model, optimizer, train_loader, val_loader
        )
    else:
        policy_model, optimizer, train_loader = accelerator.prepare(policy_model, optimizer, train_loader)
    reference_model.to(accelerator.device)

    global_step = 0
    history = []

    for epoch in range(train_cfg["epochs"]):
        progress = tqdm(
            train_loader,
            desc=f"dpo-epoch-{epoch + 1}",
            disable=not accelerator.is_local_main_process,
        )

        for batch in progress:
            with accelerator.accumulate(policy_model):
                policy_chosen, aux_chosen = compute_response_logprobs(
                    policy_model,
                    batch["prompt_ids"],
                    batch["chosen_ids"],
                    pad_id,
                    model_cfg["block_size"],
                    train_cfg["length_normalize"],
                )
                policy_rejected, aux_rejected = compute_response_logprobs(
                    policy_model,
                    batch["prompt_ids"],
                    batch["rejected_ids"],
                    pad_id,
                    model_cfg["block_size"],
                    train_cfg["length_normalize"],
                )

                with torch.no_grad():
                    ref_chosen, _ = compute_response_logprobs(
                        reference_model,
                        batch["prompt_ids"],
                        batch["chosen_ids"],
                        pad_id,
                        model_cfg["block_size"],
                        train_cfg["length_normalize"],
                    )
                    ref_rejected, _ = compute_response_logprobs(
                        reference_model,
                        batch["prompt_ids"],
                        batch["rejected_ids"],
                        pad_id,
                        model_cfg["block_size"],
                        train_cfg["length_normalize"],
                    )

                pref_loss, reward_accuracy, reward_margin, chosen_margin = calculate_dpo_loss(
                    policy_chosen,
                    policy_rejected,
                    ref_chosen,
                    ref_rejected,
                    beta=train_cfg["beta"],
                )
                aux_loss = 0.5 * (aux_chosen + aux_rejected)
                loss = pref_loss + train_cfg["aux_loss_weight"] * aux_loss

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(policy_model.parameters(), train_cfg["grad_clip"])
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if accelerator.is_local_main_process and global_step % train_cfg["log_interval"] == 0:
                    progress.set_postfix(
                        loss=float(loss.item()),
                        pref=float(pref_loss.item()),
                        acc=float(reward_accuracy.item()),
                        margin=float(reward_margin.item()),
                    )
                    history.append(
                        {
                            "step": global_step,
                            "loss": float(loss.item()),
                            "pref_loss": float(pref_loss.item()),
                            "reward_accuracy": float(reward_accuracy.item()),
                            "reward_margin": float(reward_margin.item()),
                            "policy_margin": float(chosen_margin.item()),
                        }
                    )

                if accelerator.is_main_process and global_step % train_cfg["save_interval"] == 0:
                    save_path = os.path.join(ckpt_cfg["policy_output_dir"], f"dpo_policy_step_{global_step}.pt")
                    accelerator.save(
                        {
                            "model": accelerator.unwrap_model(policy_model).state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "step": global_step,
                            "config": cfg,
                            "stage": "dpo_policy",
                        },
                        save_path,
                    )

        if accelerator.is_main_process:
            metrics = evaluate(
                accelerator.unwrap_model(policy_model),
                reference_model,
                val_loader,
                pad_id,
                model_cfg["block_size"],
                train_cfg["beta"],
                train_cfg["length_normalize"],
            )
            print(
                f"[dpo] epoch={epoch + 1} "
                f"val_loss={metrics['val_loss']} "
                f"val_reward_accuracy={metrics['val_reward_accuracy']} "
                f"val_reward_margin={metrics['val_reward_margin']}"
            )

    final_path = os.path.join(ckpt_cfg["policy_output_dir"], "dpo_policy_final.pt")
    if accelerator.is_main_process:
        accelerator.save(
            {
                "model": accelerator.unwrap_model(policy_model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": global_step,
                "config": cfg,
                "history_tail": history[-100:],
                "stage": "dpo_policy",
            },
            final_path,
        )
        if history:
            with open(os.path.join(ckpt_cfg["policy_output_dir"], "dpo_history.json"), "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)
        print(f"Saved DPO policy checkpoint to {final_path}")

    accelerator.wait_for_everyone()
    return {
        "policy_checkpoint": final_path,
        "steps": global_step,
        "history_tail": history[-10:],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DPO on the local MoE SFT checkpoint")
    parser.add_argument("--config", default="rlhf_config.json", help="Path to rlhf_config.json")
    args = parser.parse_args()

    results = train(config_path=args.config)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
