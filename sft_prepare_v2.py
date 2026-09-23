#!/usr/bin/env python3
import argparse, json, os, random

# Reuse the HF dataset download and field parsing code.
import sft_data_loader as legacy

IGNORE_INDEX = -100


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

def build_sft_prompt(system: str, instruction: str, input_text: str = "") -> str:
    system = system.strip()
    instruction = instruction.strip()
    input_text = input_text.strip()
    if input_text:
        return WITH_INPUT_TEMPLATE.format(
            system=system,
            instruction=instruction,
            input=input_text,
        )
    return NO_INPUT_TEMPLATE.format(system=system, instruction=instruction)

def write_jsonl(path, examples):
    with open(path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")


def prepare(config_path, seed):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    dc = cfg["data_config"]
    data_dir = dc["data_dir"]
    os.makedirs(data_dir, exist_ok=True)
    if os.name == "nt":
        # Some dataset-generated Arrow names exceed MAX_PATH under the repo.
        cache_dir = os.path.join(os.path.expanduser("~"), ".sft_hf_cache")
    else:
        cache_dir = os.path.join(data_dir, "hf_cache")
    os.makedirs(cache_dir, exist_ok=True)

    val_split = float(dc.get("val_split", 0.02))
    if not 0.0 < val_split < 1.0:
        raise ValueError("val_split must be between 0 and 1")

    train_examples = []
    val_examples = []
    source_counts = {}

    for index, (source, requested_train) in enumerate(dc["dataset_mix"].items()):
        requested_train = int(requested_train)
        if requested_train <= 0:
            continue

        loader = legacy.DATASET_LOADERS.get(source)
        if loader is None:
            raise ValueError(f"No dataset loader registered for {source!r}")

        print(f"Loading {source} ...")
        loaded = loader(cache_dir)
        examples = []
        skipped = 0
        duplicates = 0
        seen = set()
        for example in loaded:
            system = example.get("system", "")
            instruction = example.get("instruction", "")
            input_text = example.get("input", "")
            response = example.get("response", "")
            if not all(
                isinstance(value, str)
                for value in (system, instruction, input_text, response)
            ):
                skipped += 1
                continue

            system = system.strip()
            instruction = instruction.strip()
            input_text = input_text.strip()
            response = response.strip()
            if not system or not instruction or not response:
                skipped += 1
                continue

            prompt = build_sft_prompt(system, instruction, input_text)
            key = (prompt, response)
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            examples.append({
                "prompt": prompt,
                "response": response,
                "source": source,
            })

        rng = random.Random(seed + 1009 * (index + 1))
        rng.shuffle(examples)
        target_train = min(requested_train, len(examples))
        desired_val = max(1, round(target_train * val_split / (1.0 - val_split)))
        if len(examples) >= 2:
            n_val = min(desired_val, len(examples) - 1)
            n_train = min(requested_train, len(examples) - n_val)
        else:
            n_train = target_train
            n_val = 0

        train_examples.extend(examples[:n_train])
        val_examples.extend(examples[n_train:n_train + n_val])
        source_counts[source] = {
            "available": len(examples),
            "train": n_train,
            "val": n_val,
            "skipped": skipped,
            "duplicates": duplicates,
        }
        print(
            f"  available={len(examples):,}, train={n_train:,}, "
            f"val={n_val:,}, skipped={skipped:,}, duplicates={duplicates:,}"
        )

    if not train_examples:
        raise RuntimeError("No training examples were prepared")
    if not val_examples:
        raise RuntimeError(
            "No validation examples were prepared; reduce requested dataset sizes"
        )

    random.Random(seed).shuffle(train_examples)
    random.Random(seed + 1).shuffle(val_examples)

    train_path = os.path.join(data_dir, "train_examples.jsonl")
    val_path = os.path.join(data_dir, "val_examples.jsonl")
    write_jsonl(train_path, train_examples)
    write_jsonl(val_path, val_examples)

    print(f"Wrote {len(train_examples):,} examples to {train_path}")
    print(f"Wrote {len(val_examples):,} examples to {val_path}")
    print(json.dumps(source_counts, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="sft_config.json")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    prepare(a.config, a.seed)
