#!/usr/bin/env python3
import argparse, json, os, random, re
from collections import defaultdict

import numpy as np
import tiktoken
from tqdm import tqdm

# Reuse your existing HF dataset download/field parsing code.
# This file must sit beside your current sft_prepare.py.
import sft_prepare as legacy

IGNORE_INDEX = -100

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Answer the user's request directly, accurately, and concisely."
)

NO_INPUT_TEMPLATE = (
    "### SYSTEM:\n{system}\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n"
)
WITH_INPUT_TEMPLATE = (
    "### SYSTEM:\n{system}\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)


def recover_semantic_fields(old_example: dict) -> dict | None:
    """Remove the legacy dataset-specific SYSTEM prompt.

    Your current loaders all emit the same Alpaca-ish structure, so we recover
    only Instruction/Input and then rebuild the prompt with ONE system prompt.
    """
    prompt = old_example["prompt"]
    response = old_example["response"].strip()
    if not response:
        return None

    m = re.search(
        r"### Instruction:\n(.*?)(?:\n\n### Input:\n(.*?))?\n\n### Response:\n\Z",
        prompt,
        flags=re.S,
    )
    if not m:
        return None

    instruction = (m.group(1) or "").strip()
    input_text = (m.group(2) or "").strip()
    if not instruction:
        return None

    return {
        "instruction": instruction,
        "input": input_text,
        "response": response,
    }


def build_prompt(system: str, instruction: str, input_text: str = "") -> str:
    if input_text:
        return WITH_INPUT_TEMPLATE.format(
            system=system.strip(), instruction=instruction.strip(), input=input_text.strip()
        )
    return NO_INPUT_TEMPLATE.format(
        system=system.strip(), instruction=instruction.strip()
    )


def encode_example(record, enc, system_prompt, max_seq_length, long_policy="skip"):
    """Encode one whole example. Never truncate away the instruction.

    Default policy is 'skip'. Optional 'truncate_input' trims ONLY the separate
    context/Input field; the system prompt, full instruction, Response boundary,
    full answer, and EOT are always preserved.
    """
    instruction = record["instruction"].strip()
    input_text = record.get("input", "").strip()
    response = record["response"].strip()

    response_tokens = enc.encode_ordinary(response)
    eot = [enc.eot_token]
    if not response_tokens:
        return None

    prompt = build_prompt(system_prompt, instruction, input_text)
    prompt_tokens = enc.encode_ordinary(prompt)

    if len(prompt_tokens) + len(response_tokens) + 1 > max_seq_length:
        if long_policy != "truncate_input" or not input_text:
            return None

        # Rebuild token-wise so ONLY Input is shortened.
        prefix = (
            f"### SYSTEM:\n{system_prompt.strip()}\n\n"
            f"### Instruction:\n{instruction}\n\n"
            "### Input:\n"
        )
        suffix = "\n\n### Response:\n"
        prefix_t = enc.encode_ordinary(prefix)
        suffix_t = enc.encode_ordinary(suffix)
        input_t = enc.encode_ordinary(input_text)

        input_budget = (
            max_seq_length
            - len(prefix_t)
            - len(suffix_t)
            - len(response_tokens)
            - 1
        )
        if input_budget < 0:
            return None  # full instruction + full response cannot fit

        input_t = input_t[:input_budget]
        prompt_tokens = prefix_t + input_t + suffix_t
        prompt = enc.decode(prompt_tokens)

    tokens = prompt_tokens + response_tokens + eot
    labels = [IGNORE_INDEX] * len(prompt_tokens) + response_tokens + eot

    assert len(tokens) == len(labels)
    assert len(tokens) <= max_seq_length
    return tokens, labels, prompt


def write_split(name, records, data_dir, enc, system_prompt, max_seq, long_policy):
    encoded = []
    human = []
    skipped = 0
    by_source = defaultdict(lambda: {"kept": 0, "skipped": 0, "supervised_tokens": 0})

    for r in tqdm(records, desc=f"Encoding {name}"):
        item = encode_example(r, enc, system_prompt, max_seq, long_policy)
        src = r["source"]
        if item is None:
            skipped += 1
            by_source[src]["skipped"] += 1
            continue
        tokens, labels, prompt = item
        encoded.append((tokens, labels))
        by_source[src]["kept"] += 1
        by_source[src]["supervised_tokens"] += sum(x != IGNORE_INDEX for x in labels)
        if name == "val":
            human.append({"prompt": prompt, "response": r["response"], "source": src})

    if not encoded:
        raise RuntimeError(f"No usable examples in {name} split")

    total = sum(len(t) for t, _ in encoded)
    offsets = np.empty((len(encoded), 2), dtype=np.int64)
    tok = np.memmap(os.path.join(data_dir, f"{name}_tokens.bin"), dtype=np.int32,
                    mode="w+", shape=(total,))
    lab = np.memmap(os.path.join(data_dir, f"{name}_labels.bin"), dtype=np.int32,
                    mode="w+", shape=(total,))

    cursor = 0
    supervised = 0
    for i, (tokens, labels) in enumerate(encoded):
        end = cursor + len(tokens)
        tok[cursor:end] = np.asarray(tokens, dtype=np.int32)
        lab[cursor:end] = np.asarray(labels, dtype=np.int32)
        offsets[i] = (cursor, end)
        supervised += sum(x != IGNORE_INDEX for x in labels)
        cursor = end
    tok.flush(); lab.flush()
    np.save(os.path.join(data_dir, f"{name}_offsets.npy"), offsets)

    if name == "val":
        with open(os.path.join(data_dir, "val_examples.jsonl"), "w", encoding="utf-8") as f:
            for x in human:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")

    lens = offsets[:, 1] - offsets[:, 0]
    return {
        "examples": len(encoded), "skipped": skipped, "tokens": int(total),
        "supervised_tokens": int(supervised), "mean_length": float(lens.mean()),
        "max_length": int(lens.max()), "by_source": dict(by_source),
    }


def prepare(config_path, seed):
    random.seed(seed); np.random.seed(seed)
    with open(config_path, "r") as f:
        cfg = json.load(f)

    dc = cfg["data_config"]
    data_dir = dc["data_dir"]
    os.makedirs(data_dir, exist_ok=True)
    cache_dir = os.path.join(data_dir, "hf_cache")
    os.makedirs(cache_dir, exist_ok=True)

    system_prompt = dc.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
    long_policy = dc.get("long_example_policy", "skip")
    if long_policy not in {"skip", "truncate_input"}:
        raise ValueError("long_example_policy must be 'skip' or 'truncate_input'")

    val_split = float(dc.get("val_split", 0.02))
    enc = tiktoken.get_encoding("gpt2")
    train_records, val_records = [], []
    source_selection = {}

    for k, (name, requested_train) in enumerate(dc["dataset_mix"].items()):
        if requested_train <= 0:
            continue
        loader = legacy.DATASET_LOADERS.get(name)
        if loader is None:
            print(f"[WARN] unknown dataset {name}; skipping")
            continue

        print(f"Loading {name} ...")
        old_examples = loader(cache_dir)
        records = []
        for ex in old_examples:
            r = recover_semantic_fields(ex)
            if r:
                records.append(r)

        rng = random.Random(seed + 1009 * (k + 1))
        rng.shuffle(records)
        n_train = min(int(requested_train), len(records))
        desired_val = max(1, round(n_train * val_split / max(1.0 - val_split, 1e-8)))
        n_val = min(desired_val, max(0, len(records) - n_train))

        tr = records[:n_train]
        va = records[n_train:n_train + n_val]
        for r in tr:
            r["source"] = name
        for r in va:
            r["source"] = name
        train_records.extend(tr)
        val_records.extend(va)
        source_selection[name] = {
            "available": len(records), "train": len(tr), "val": len(va)
        }
        print(f"  available={len(records):,}, train={len(tr):,}, val={len(va):,}")

    random.Random(seed).shuffle(train_records)
    random.Random(seed + 1).shuffle(val_records)

    if not val_records:
        # Only a fallback. Prefer requesting fewer train examples so validation is disjoint.
        val_records = [dict(x) for x in train_records[-min(256, len(train_records)):]]
        val_disjoint = False
        print("[WARN] validation fallback overlaps training")
    else:
        val_disjoint = True

    train_stats = write_split("train", train_records, data_dir, enc, system_prompt,
                              int(dc["max_seq_length"]), long_policy)
    val_stats = write_split("val", val_records, data_dir, enc, system_prompt,
                            int(dc["max_seq_length"]), long_policy)

    metadata = {
        "seed": seed, "tokenizer": "gpt2", "eot_token": enc.eot_token,
        "ignore_index": IGNORE_INDEX, "system_prompt": system_prompt,
        "long_example_policy": long_policy, "val_is_disjoint": val_disjoint,
        "source_selection": source_selection,
        "train": train_stats, "val": val_stats, "config_snapshot": cfg,
    }
    with open(os.path.join(data_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(json.dumps({"train": train_stats, "val": val_stats}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="sft_config_v2.json")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    prepare(a.config, a.seed)
