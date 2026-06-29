# saves the bookcorpus dataset to a binary file for training. follows prepare.py pattern
# Output files: train_book.bin, val_book.bin
# Compatible with get_batch() in model_moe.py which reads via np.memmap with dtype=np.uint16
#
# NOTE: BookCorpus is sentence-level (one sentence per example). Naively appending
# eot_token per sentence causes the model to see <|endoftext|> every ~12 tokens vs
# every ~500 tokens in OWT/FineWeb, causing premature EOS prediction at inference.
# Fix: sentences are grouped into chunks of SENTENCES_PER_CHUNK via batched .map()
# before tokenizing, so eot_token density matches OWT/FineWeb (once per chunk).

import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset, Dataset  # huggingface datasets

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 8

# number of workers in load_dataset() call
# best number might be different from num_proc above as it also depends on NW speed.
# it is better than 1 usually though
num_proc_load_dataset = num_proc

enc = tiktoken.get_encoding("gpt2")

# Number of sentences to concatenate into one document chunk before tokenizing.
# At ~15 tokens/sentence avg, 500 sentences ~ 7,500 tokens per chunk.
# This keeps eot_token density comparable to OWT (~1 per 500 tokens).
SENTENCES_PER_CHUNK = 500

def group_into_chunks(batch):
    """
    Batched .map() function — receives a batch of SENTENCES_PER_CHUNK sentences
    and joins them into a single document-sized chunk.
    Each chunk gets one eot_token during tokenization, matching OWT/FineWeb density.
    Called with batched=True and batch_size=SENTENCES_PER_CHUNK.
    """
    joined = ' '.join([s.strip() for s in batch['text']])
    return {'text': [joined]}

if __name__ == '__main__':
    # bookcorpus/bookcorpus contains ~74M sentences / ~1B tokens
    # only has a 'train' split by default, so we create our own val split
    dataset = load_dataset(
        "bookcorpus/bookcorpus",
        split="train",
        num_proc=num_proc_load_dataset,
        cache_dir="./cache_root",
        trust_remote_code=True,
    )

    # Step 1: group sentences into chunks via batched .map() BEFORE splitting train/val.
    # batched=True with batch_size=SENTENCES_PER_CHUNK passes exactly 500 sentences
    # per call to group_into_chunks(), which joins them into one chunk and returns it.
    # This is fully streaming/memory-safe — no list() or Dataset.from_list() needed.
    # ~74M sentences / 500 per chunk = ~148k chunks total.
    print(f"Grouping sentences into chunks of {SENTENCES_PER_CHUNK} via batched map...")
    chunked_dataset = dataset.map(
        group_into_chunks,
        batched=True,
        batch_size=SENTENCES_PER_CHUNK,
        desc="grouping sentences into chunks",
        num_proc=1,  # must be 1 — parallel workers would break chunk boundaries
    )
    print(f"Total chunks: {len(chunked_dataset):,}")

    # Step 2: train/val split on chunks (not raw sentences)
    # test_size=0.0005 gives ~74 chunks for val, rest for train
    split_dataset = chunked_dataset.train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test')  # rename test split to val

    # this results in:
    # DatasetDict({
    #     train: Dataset({ features: ['text'], num_rows: ~147,926 })
    #     val:   Dataset({ features: ['text'], num_rows: ~74      })
    # })
    # Each chunk is ~500 sentences joined by spaces — one eot_token per chunk.

    # Step 3: tokenize chunks using gpt2 bpe (same tokenizer as OWT and FineWeb-Edu runs)
    def process(example):
        ids = enc.encode_ordinary(example['text'])  # encode_ordinary ignores any special tokens
        ids.append(enc.eot_token)                   # one eot per chunk, not per sentence
        out = {'ids': ids, 'len': len(ids)}
        return out

    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    output_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)

    # concatenate all the ids in each dataset into one large file we can use for training
    # dtype=np.uint16 is safe since enc.max_token_value == 50256 < 2**16 == 65536
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(output_dir, f'{split}_fweb.bin')
        dtype = np.uint16
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))

        # cap total_batches to dataset size — val split only has ~74 chunks
        # so sharding into 256 pieces would cause an IndexError
        total_batches = min(256, len(dset))

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            # Write into mmap
            arr[idx: idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()

        print(f'{split}_book.bin — {arr_len:,} tokens written to {filename}')

    # Expected output sizes:
    # train_book.bin ~ 1GB  (~1B tokens across ~148k chunks)
    # val_book.bin   ~ <1MB (~500K tokens across ~74 chunks)
    #
    # Verify eot_token density matches OWT/FineWeb before training:
    #   import numpy as np
    #   eot = 50256
    #   for name, path in [('OWT','train.bin'), ('FWeb','train_fweb.bin'), ('Book','train_book.bin')]:
    #       data = np.memmap(path, dtype=np.uint16, mode='r')
    #       density = (data == eot).sum() / len(data)
    #       print(f"{name}: eot every ~{1/density:.0f} tokens")  # all should be ~500-700
