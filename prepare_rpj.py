# saves the RedPajama-Data-1T dataset to a binary file for training.
# follows the same pattern as prepare_fineweb_edu.py and prepare_bookcorpus.py.
#
# Output files: train_rpj.bin, val_rpj.bin
# Compatible with get_batch() in model_moe.py which reads via np.memmap with dtype=np.uint16
#
# RedPajama-1T has 6 subsets (the 'book' config is defunct due to copyright):

#   github        ~59B  tokens
#   wikipedia     ~24B  tokens
#   stackexchange ~20B  tokens
# SUBSET SELECTION:
#   Edit SUBSETS below to control which slices are concatenated.
#   Defaults to the 4 smaller, higher-quality subsets (~127B tokens total) to keep
#   disk/RAM requirements manageable. Add 'github', 'c4', or 'common_crawl' as needed.
#
# EXPECTED OUTPUT SIZES (default subsets):
#   train_rpj.bin  ~250GB  (~127B tokens)
#   val_rpj.bin    ~125MB  (~64M  tokens)
#
# Verify eot_token density after writing:
#   import numpy as np
#   eot = 50256
#   for name, path in [('RPJ','train_rpj.bin'), ('FWeb','train_fweb.bin')]:
#       data = np.memmap(path, dtype=np.uint16, mode='r')
#       density = (data == eot).sum() / len(data)
#       print(f"{name}: eot every ~{1/density:.0f} tokens")

import os
from tqdm import tqdm
import numpy as np
import tiktoken
import aiohttp
import json
import ast
from datasets import load_dataset, concatenate_datasets
from datasets.builder import DatasetGenerationError

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
os.environ["RED_PAJAMA_DATA_DIR"] = "./cache_root/redpajama"
# Subsets to include. Remove or comment out entries to reduce dataset size.
# 'book' is defunct (copyright takedown) — do not add it.
SUBSETS = [
    "github",           # ~59B  tokens — code; uncomment if you want code data
    "wikipedia",       # ~24B tokens  — high quality, encyclopedic
    "stackexchange",   # ~20B tokens  — high quality, Q&A
]

# Val fraction — 0.0005 matches OWT / FineWeb / BookCorpus convention
VAL_FRACTION = 0.0005
SEED = 2357
SELECT_FRACTION = 0.2
# Number of workers for .map() tokenization (~num_cpu_cores // 2)
num_proc = 8

# Number of workers for load_dataset() — network + disk bound
num_proc_load_dataset = num_proc

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
enc = tiktoken.get_encoding("gpt2")


def is_english(example):
    """
    Filter strictly english-only examples and drop rest
    """
    # Deserialize if meta is a raw JSON string (e.g. common_crawl subset)
    meta = example["meta"]
    meta_dict = {}  # Initialize to empty dic
    if isinstance(meta, str): #which it most likely is
        if meta == "":
            raise Exception(f"meta field is neither string nor dict. Full meta:\n {meta}")
        meta_dict = ast.literal_eval(meta)
        if not isinstance(meta_dict, dict):
            raise Exception("Could not evaluate string as dict object")
    elif isinstance(meta, dict):
        meta_dict = meta
    
    # meta_dict is now a dict; keep if no language tag or language is English
    lang = meta_dict.get("language", "")
    if isinstance(lang, list): # prolly from github, so we will just pass
        return True
    return lang == "en"

# ---------------------------------------------------------------------------
# Tokenization function (same as prepare_fineweb_edu.py)
# ---------------------------------------------------------------------------
def process(example):
    ids = enc.encode_ordinary(example['text'])  # ignores any special tokens
    ids.append(enc.eot_token)                   # one eot_token per document
    out = {'ids': ids, 'len': len(ids)}
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':

    # ------------------------------------------------------------------
    # Step 1: Load each requested subset and concatenate into one dataset
    # ------------------------------------------------------------------
    subset_datasets = []
    for subset in SUBSETS:
        print(f"Loading subset: {subset} ...")
        '''
        ds = load_dataset(
            "togethercomputer/RedPajama-Data-1T",
            subset,
            split="train",          # RPJ only ships a 'train' split
            cache_dir="./cache_root",
            trust_remote_code=True,
            streaming=True,        # set True + adjust write loop if RAM is tight
        )
        '''
        try:
            ds = load_dataset("togethercomputer/RedPajama-Data-1T", subset, cache_dir="./cache_root", split="train", data_files=os.path.join("./cache_root/redpajama", subset, "*.jsonl"))
        except DatasetGenerationError as e:
            continue
        subset_datasets.append(ds)
        print(f"  {subset}: {len(ds):,} documents loaded")
    print(f"\nConcatenating {len(subset_datasets)} subset(s)...")
    full_dataset = concatenate_datasets(subset_datasets)
    print(f"Total documents: {len(full_dataset):,}")
    # Drop non-English documents
    print("\nFiltering non-English documents...")
    full_dataset = full_dataset.filter(
        is_english,
        desc="filtering non-English documents",
        num_proc=num_proc,
    )
    print(f"Documents after language filter: {len(full_dataset):,}")
    n_select = int(len(full_dataset) * SELECT_FRACTION)
    print(f"\nShuffling and selecting {SELECT_FRACTION*100:.0f}% "
          f"({n_select:,} / {len(full_dataset):,} documents)...")
    full_dataset = (
        full_dataset
        .shuffle(seed=SEED)
        .select(range(n_select))
    )
    print(f"Documents after sampling: {len(full_dataset):,}")

    # ------------------------------------------------------------------
    # Step 2: Train / val split
    # ------------------------------------------------------------------
    # test_size=VAL_FRACTION mirrors the OWT / FineWeb / BookCorpus convention
    split_dataset = full_dataset.train_test_split(
        test_size=VAL_FRACTION,
        seed=SEED,
        shuffle=True,
    )
    split_dataset['val'] = split_dataset.pop('test')  # rename test -> val

    print(f"\nSplit sizes:")
    for split_name, dset in split_dataset.items():
        print(f"  {split_name}: {len(dset):,} documents")

    # ------------------------------------------------------------------
    # Step 3: Tokenize with GPT-2 BPE (same tokenizer as OWT / FineWeb)
    # ------------------------------------------------------------------
    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    # ------------------------------------------------------------------
    # Step 4: Write flat binary files (np.uint16 memmap)
    # ------------------------------------------------------------------
    output_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)

    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(output_dir, f'{split}_rpj.bin')
        dtype = np.uint16  # safe: enc.max_token_value == 50256 < 2**16 == 65536
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))

        # Cap total_batches to dataset size — val split may have far fewer rows
        # than 1024, which would cause an IndexError in dset.shard()
        total_batches = min(1024, len(dset))

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            batch = dset.shard(
                num_shards=total_batches,
                index=batch_idx,
                contiguous=True,
            ).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            arr[idx: idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)

        arr.flush()
        print(f'{split}_rpj.bin — {arr_len:,} tokens written to {filename}')