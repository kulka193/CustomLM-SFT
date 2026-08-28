"""
DPO preference-data preparation pipeline.

This reuses the existing SFT dataset loaders and converts single-response
instruction data into pairwise preference JSONL records:

    {"prompt": str, "chosen": str, "rejected": str, "source": str}

The default behavior is intentionally simple:
- chosen   = the original high-quality SFT response
- rejected = either a mismatched response from another example, a truncated
             response, or a lightly corrupted numeric/text variant

This gives you a small synthetic preference set that matches the repo's prompt
format and can be replaced later with human rankings or stronger model-judged
pairs.
"""

import argparse
import json
import os
import random
import re

import tiktoken
import torch
from torch.utils.data import Dataset

from sft_prepare import DATASET_LOADERS, END_TOKEN_STR, build_sft_prompt


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_prompt_from_record(record: dict) -> str:
    prompt = record.get("prompt", "").strip()
    if prompt:
        return prompt

    instruction = record.get("instruction", "").strip()
    context = record.get("context", "").strip()
    system = record.get(
        "system",
        "You are a helpful, precise, and honest AI assistant.",
    ).strip()
    return build_sft_prompt(system, instruction, context)


def write_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_jsonl(path: str, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_preference_records(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            rec = json.loads(line)
            prompt = build_prompt_from_record(rec)
            chosen = rec.get("chosen", "").strip()
            rejected = rec.get("rejected", "").strip()
            if not prompt or not chosen or not rejected:
                raise ValueError(
                    f"{path}:{line_num} must contain prompt/instruction and chosen/rejected text"
                )

            records.append(
                {
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                    "source": rec.get("source", "unknown"),
                    "rejected_strategy": rec.get("rejected_strategy", "unknown"),
                }
            )

    if not records:
        raise ValueError(f"No preference rows found in {path}")
    return records


def maybe_trim_sequence(prompt_ids: list[int], response_ids: list[int], block_size: int) -> tuple[list[int], list[int]]:
    if len(prompt_ids) + len(response_ids) <= block_size:
        return prompt_ids, response_ids

    overflow = len(prompt_ids) + len(response_ids) - block_size
    if overflow >= len(prompt_ids):
        prompt_ids = prompt_ids[-1:]
    else:
        prompt_ids = prompt_ids[overflow:]
    return prompt_ids, response_ids


class PreferenceDataset(Dataset):
    def __init__(
        self,
        records: list[dict],
        enc: tiktoken.Encoding,
        max_prompt_length: int,
        max_response_length: int,
        block_size: int,
    ) -> None:
        self.samples = []
        end_tokens = enc.encode_ordinary(END_TOKEN_STR) + [enc.eot_token]

        for rec in records:
            raw_prompt_ids = enc.encode_ordinary(rec["prompt"])[-max_prompt_length:]
            chosen_ids = enc.encode_ordinary(rec["chosen"])[:max_response_length] + end_tokens
            rejected_ids = enc.encode_ordinary(rec["rejected"])[:max_response_length] + end_tokens
            max_response_span = max(len(chosen_ids), len(rejected_ids))
            prompt_budget = max(block_size - max_response_span, 1)
            prompt_ids = raw_prompt_ids[-prompt_budget:]
            self.samples.append(
                {
                    "prompt_ids": prompt_ids,
                    "chosen_ids": chosen_ids,
                    "rejected_ids": rejected_ids,
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


def preference_collate_fn(samples: list[dict]) -> dict:
    return {
        "prompt_ids": [sample["prompt_ids"] for sample in samples],
        "chosen_ids": [sample["chosen_ids"] for sample in samples],
        "rejected_ids": [sample["rejected_ids"] for sample in samples],
    }


def build_preference_datasets(cfg: dict) -> tuple[PreferenceDataset, PreferenceDataset | None, tiktoken.Encoding]:
    model_cfg = cfg["model_config"]
    data_cfg = cfg["data_config"]
    enc = tiktoken.get_encoding("gpt2")

    train_records = load_preference_records(data_cfg["train_file"])
    val_records = load_preference_records(data_cfg["val_file"]) if data_cfg.get("val_file") else None

    train_ds = PreferenceDataset(
        train_records,
        enc,
        data_cfg["max_prompt_length"],
        data_cfg["max_response_length"],
        model_cfg["block_size"],
    )
    val_ds = (
        PreferenceDataset(
            val_records,
            enc,
            data_cfg["max_prompt_length"],
            data_cfg["max_response_length"],
            model_cfg["block_size"],
        )
        if val_records
        else None
    )
    return train_ds, val_ds, enc


def pad_tensor_rows(rows: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(row) for row in rows)
    batch = torch.full((len(rows), max_len), pad_id, dtype=torch.long)
    lengths = torch.tensor([len(row) for row in rows], dtype=torch.long)
    for i, row in enumerate(rows):
        batch[i, : len(row)] = torch.tensor(row, dtype=torch.long)
    return batch, lengths


def truncate_response(text: str, min_chars: int = 40) -> str:
    text = text.strip()
    if len(text) <= min_chars:
        return "I am not sure about the correct answer."
    cut = max(min_chars, len(text) // 3)
    return text[:cut].rstrip() + " ..."


def swap_last_number(text: str) -> str | None:
    matches = list(re.finditer(r"-?\d+(\.\d+)?", text))
    if not matches:
        return None

    last = matches[-1]
    value = last.group(0)
    if "." in value:
        replacement = f"{float(value) + 1.0:.1f}"
    else:
        replacement = str(int(value) + 1)
    return text[: last.start()] + replacement + text[last.end() :]


def generic_bad_response(source: str) -> str:
    if "code" in source:
        return "I am not sure, but you can try writing a quick script and see if it works."
    if "math" in source or "gsm" in source:
        return "The answer is probably 0, but I am not certain."
    return "I am not fully sure, but maybe that is the right idea."


def build_rejected_response(
    example: dict,
    same_source_pool: list[dict],
    global_pool: list[dict],
    rng: random.Random,
) -> tuple[str, str]:
    chosen = example["response"].strip()
    source = example["source"]

    strategies = ["swap", "truncate", "generic", "numeric"]
    rng.shuffle(strategies)

    for strategy in strategies:
        if strategy == "swap":
            candidate_pool = same_source_pool if len(same_source_pool) > 1 else global_pool
            for _ in range(8):
                candidate = rng.choice(candidate_pool)
                candidate_response = candidate["response"].strip()
                if candidate_response and candidate_response != chosen:
                    return candidate_response, "swap_response"

        elif strategy == "truncate":
            rejected = truncate_response(chosen)
            if rejected != chosen:
                return rejected, "truncate"

        elif strategy == "numeric":
            rejected = swap_last_number(chosen)
            if rejected and rejected != chosen:
                return rejected, "numeric_perturb"

        elif strategy == "generic":
            rejected = generic_bad_response(source)
            if rejected != chosen:
                return rejected, "generic_low_quality"

    return "I do not know the answer.", "fallback_generic"


def load_preference_source_datasets(dataset_mix: dict[str, int], cache_dir: str) -> tuple[dict[str, list[dict]], dict[str, int]]:
    raw_datasets: dict[str, list[dict]] = {}
    counts: dict[str, int] = {}

    for name, requested in dataset_mix.items():
        if requested <= 0:
            continue
        if name not in DATASET_LOADERS:
            print(f"[WARN] Unknown dataset '{name}', skipping.")
            continue

        print(f"  Loading {name} ...")
        records = DATASET_LOADERS[name](cache_dir)
        for rec in records:
            rec["source"] = name
        raw_datasets[name] = records
        counts[name] = min(requested, len(records))
        print(f"    -> {len(records):,} available, using {counts[name]:,}")

    if not raw_datasets:
        raise ValueError("No preference datasets were loaded. Check preference_dataset_mix in rlhf_config.json.")

    return raw_datasets, counts


def build_preference_records(
    raw_datasets: dict[str, list[dict]],
    counts: dict[str, int],
    val_examples_per_dataset: int,
    seed: int,
) -> tuple[list[dict], list[dict], dict]:
    rng = random.Random(seed)
    train_records = []
    val_records = []

    global_pool = []
    for dataset_records in raw_datasets.values():
        global_pool.extend(dataset_records)

    stats = {}
    for source, dataset_records in raw_datasets.items():
        used = counts[source]
        train_pool = dataset_records[:used]
        held_out_pool = dataset_records[used:]
        if not held_out_pool:
            held_out_pool = dataset_records

        source_train = []
        source_val = []

        for ex in train_pool:
            rejected, strategy = build_rejected_response(ex, train_pool, global_pool, rng)
            source_train.append(
                {
                    "prompt": ex["prompt"],
                    "chosen": ex["response"],
                    "rejected": rejected,
                    "source": source,
                    "rejected_strategy": strategy,
                }
            )

        val_sample_n = min(val_examples_per_dataset, len(held_out_pool))
        val_examples = rng.sample(held_out_pool, val_sample_n)
        for ex in val_examples:
            rejected, strategy = build_rejected_response(ex, held_out_pool, global_pool, rng)
            source_val.append(
                {
                    "prompt": ex["prompt"],
                    "chosen": ex["response"],
                    "rejected": rejected,
                    "source": source,
                    "rejected_strategy": strategy,
                }
            )

        train_records.extend(source_train)
        val_records.extend(source_val)
        stats[source] = {
            "train_pairs": len(source_train),
            "val_pairs": len(source_val),
        }

    rng.shuffle(train_records)
    rng.shuffle(val_records)
    return train_records, val_records, stats


def prepare_preferences(config_path: str = "rlhf_config.json", seed: int = 42) -> None:
    cfg = load_json(config_path)
    pref_cfg = cfg["preference_data_config"]
    data_cfg = cfg["data_config"]

    output_dir = pref_cfg["output_dir"]
    cache_dir = os.path.join(output_dir, "hf_cache")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    print("[1/4] Loading source datasets ...")
    raw_datasets, counts = load_preference_source_datasets(pref_cfg["dataset_mix"], cache_dir)

    print("[2/4] Building synthetic preference pairs ...")
    train_records, val_records, stats = build_preference_records(
        raw_datasets=raw_datasets,
        counts=counts,
        val_examples_per_dataset=pref_cfg.get("val_examples_per_dataset", 25),
        seed=seed,
    )

    print("[3/4] Writing JSONL files ...")
    train_path = os.path.join(output_dir, "train_preferences.jsonl")
    val_path = os.path.join(output_dir, "val_preferences.jsonl")
    write_jsonl(train_path, train_records)
    write_jsonl(val_path, val_records)

    print("[4/4] Writing metadata ...")
    metadata = {
        "train_pairs": len(train_records),
        "val_pairs": len(val_records),
        "dataset_mix": pref_cfg["dataset_mix"],
        "stats_per_dataset": stats,
        "config_snapshot": cfg,
    }
    write_json(os.path.join(output_dir, "metadata.json"), metadata)

    print("\nPreference data preparation complete.")
    print(f"  Train pairs : {len(train_records):,}")
    print(f"  Val pairs   : {len(val_records):,}")
    print(f"  Output dir  : {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare synthetic DPO preference data")
    parser.add_argument("--config", default="rlhf_config.json", help="Path to rlhf_config.json")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    prepare_preferences(config_path=args.config, seed=args.seed)
