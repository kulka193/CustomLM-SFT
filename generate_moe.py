import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import torch
import tiktoken
from model_moe import MoETransformer
import torch.nn.functional as F

warnings.filterwarnings("ignore")

'''
{"vocab_size": 50304,
 "d_model": 384,
 "d_ff": 768,
 "num_layers": 16,
 "num_experts": 16,
 "block_size": 2048,
 "batch_size": 4,
 "max_iters": 600000,
 "num_heads": 16,
 "log_interval": 50000,
 "eval_iters": 100,
 "gradient_acc_steps": 4,
 "lr": 0.001
}
'''

# GPT-2 uses byte-level BPE. Multi-byte UTF-8 characters are split
# across multiple tokens.
#
#  447, 250, 251 -> 0xE2 0x80 0x9C -> U+201C left double quote  -> 22  (")
#  447, 250, 252 -> 0xE2 0x80 0x9D -> U+201D right double quote -> 22  (")
#  447, 250, 223 -> 0xE2 0x80 0x98 -> U+2018 left single quote  -> 6   (')
#  447, 250, 224 -> 0xE2 0x80 0x99 -> U+2019 right single quote -> 6   (')
#  447, 250, 218 -> 0xE2 0x80 0x93 -> U+2013 en dash            -> 12  (-)
#  447, 250, 219 -> 0xE2 0x80 0x94 -> U+2014 em dash            -> 438 (--)
#  447, 250, 241 -> 0xE2 0x80 0xA6 -> U+2026 ellipsis           -> 986 (...)
_TOKEN_SEQ_REPLACEMENTS = [
    ([447, 250], [1]),
    ([447, 251], [1]),
    ([447, 252], [11]),
    ([447, 247], [705])
]

def replace_token_sequences(tokens: list) -> list:
    for seq, replacement in _TOKEN_SEQ_REPLACEMENTS:
        seq_len = len(seq)
        result = []
        i = 0
        while i < len(tokens):
            if tokens[i:i + seq_len] == seq:
                result.extend(replacement)
                i += seq_len
            else:
                result.append(tokens[i])
                i += 1
        tokens = result
    return tokens


class ModelMOEConfig:
    DEFAULT_CONFIG_PATH = "moe_config.json"

    def __init__(self, config_dict: dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

    def __repr__(self):
        attrs = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"ModelMOEConfig({attrs})"

    @classmethod
    def from_json(cls, path: str = None) -> "ModelMOEConfig":
        """Load config from a JSON file and return a ModelMOEConfig instance.

        Args:
            path: Path to the JSON config file.
                  Defaults to 'moe_config.json' in the same directory as this script.

        Returns:
            A ModelMOEConfig instance with all JSON fields set as attributes.

        Raises:
            FileNotFoundError: If the config file does not exist at the given path.
            ValueError: If the file contains invalid JSON.
        """
        if path is None:
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), cls.DEFAULT_CONFIG_PATH)

        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: '{path}'")

        try:
            with open(path, "r", encoding="utf-8") as f:
                config_dict = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file '{path}': {e}") from e

        return cls(config_dict)


def parse_arguments():
    """Parse command line arguments for inference."""
    parser = argparse.ArgumentParser(
        description='Generate text using a trained GPT model'
    )
    
    # Required arguments
    parser.add_argument(
        'model_path',
        type=str,
        help='Path to the trained model checkpoint'
    )

    # Prompt is optional — mutually exclusive with --from-val
    parser.add_argument(
        'prompt',
        type=str,
        nargs='?',
        default=None,
        help='Input prompt for text generation. Omit if using --from-val'
    )

    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to the MoE model config JSON'
    )

    parser.add_argument(
        '--val-suffix',
        type=str,
        default='owt',
        help='Suffix for val.bin filename (default: "")'
    )

    # Val.bin sampling
    parser.add_argument(
        '--from-val',
        action='store_true',
        help='Use a random segment from val.bin as the prompt instead of text input\n' \
        '\t utb2 -> Utltratextbook2;\n\t owt -> OpenWebText;\n\t fweb -> Fineweb;\n\t fmath4 -> finemathplus4'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='./',
        help='Directory containing val.bin, used with --from-val (default: ./)'
    )
    parser.add_argument(
        '--prompt-len',
        type=int,
        default=64,
        help='Number of seed tokens to sample from val.bin (default: 64)'
    )

    parser.add_argument(
        '--load-moe',
        action='store_true',
        required=False,
        help='Load Mixture of experts based Transformer'
    )
    
    # Generation parameters
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=500,
        help='Maximum number of tokens to generate (default: 500)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.95,
        help='Sampling temperature (default: 0.95)'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=0,
        help='Top-k sampling parameter (default: 0, disabled)'
    )
    parser.add_argument(
        '--top-p',
        type=float,
        default=0.0,
        help='Top-p (nucleus) sampling parameter (default: 0.0, disabled)'
    )
    parser.add_argument(
        '--repetition-penalty',
        type=float,
        default=1.0,
        help='Repetition penalty (default: 1.0, disabled)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to run inference on (default: cuda)'
    )
    
    return parser.parse_args()

def load_model(model_path, config, device, map_to="cpu"):
    """Load trained model from checkpoint."""
    model = MoETransformer(
        config.vocab_size, config.d_model, config.num_heads,
        config.d_ff, config.num_layers, config.num_experts,
        max_seq_len=config.block_size, top_k=2, dropout=0.0  # no dropout at inference
    )
    model.load_state_dict(
        torch.load(model_path, weights_only=True, map_location=map_to),
        strict=True
    )
    return model

def get_model_size_mb(model):
    """Calculate model size in megabytes."""
    total_size = sum(p.numel() * p.element_size() for p in model.parameters())
    return total_size / (1024 * 1024)


def format_expert_percentages(percentages, columns=4):
    percentage_strings = [
        f"E{i:02d}: {pct:5.1f}%" for i, pct in enumerate(percentages)
    ]
    rows = [
        "  " + " | ".join(percentage_strings[i:i + columns])
        for i in range(0, len(percentage_strings), columns)
    ]
    return "\n".join(rows)


def print_expert_usage(model):
    if not hasattr(model, "get_expert_usage_percentages"):
        return

    aggregate_percentages, layer_percentages = model.get_expert_usage_percentages()
    if aggregate_percentages is None:
        print("Expert utilization: no MoE routing data collected")
        return

    print("Expert utilization across generated inference calls:")
    print(format_expert_percentages(aggregate_percentages))
    print("Per MoE layer:")
    for layer_idx, percentages in layer_percentages:
        print(f"  Layer {layer_idx}:")
        print(format_expert_percentages(percentages))


def get_val_prompt(data_dir, suffix, prompt_len, device):
    """Sample a random segment from val.bin to use as generation seed."""
    val_path = os.path.join(data_dir, f'val_{suffix}.bin')
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"val_{suffix}.bin not found at: '{val_path}'")

    data = np.memmap(val_path, dtype=np.uint16, mode='r')

    max_start = len(data) - prompt_len - 1
    if max_start <= 0:
        raise ValueError(
            f"val_{suffix}.bin has {len(data)} tokens, too small for prompt_len={prompt_len}"
        )

    start_idx = torch.randint(0, max_start, (1,)).item()
    token_ids = data[start_idx : start_idx + prompt_len].astype(np.int64)
    input_tokens = torch.tensor(token_ids, dtype=torch.long, device=device)[None, ...]
    return input_tokens

@torch.no_grad()
def generate_tokens(
    model,
    tokens,
    max_new_tokens,
    block_size,
    temperature=1.0,
    top_k=0,
    top_p=0.0,
    repetition_penalty=1.0,
    eot_token_id=None,
):
    """
    Generate tokens autoregressively from the model.
    """
    model.eval()
    
    for _ in range(max_new_tokens):
        # Crop tokens to block size
        input_tokens = tokens[:, -block_size:]

        # MoETransformer returns (logits, aux_loss) — aux_loss ignored at inference
        logits, _ = model(input_tokens)
        logits = logits[:, -1, :]  # (1, vocab_size)

        if temperature <= 0:
            # Greedy decoding
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_token], dim=1)
            yield next_token
            if eot_token_id is not None and next_token.item() == eot_token_id:
                break
            continue
        if repetition_penalty != 1.0:
            context = input_tokens[0]   # (T,) — recent context
            for token_id in context.unique():
                if logits[0, token_id] > 0:
                    logits[0, token_id] /= repetition_penalty   # reduce positive logits
                else:
                    logits[0, token_id] *= repetition_penalty   # increase negative logits

        logits = logits / temperature

        # Top-k filtering
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')

        # Top-p (nucleus) filtering — fixed scatter
        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift right: always keep at least one token
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False

            # Scatter mask back to vocab-ordered logits
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = -float('Inf')

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat([tokens, next_token], dim=1)
        yield next_token

        if eot_token_id is not None and next_token.item() == eot_token_id:
            break


def main():
    """Main inference function."""
    args = parse_arguments()

    # Validate mutual exclusion of prompt and --from-val
    if args.from_val and args.prompt:
        print("Error: Cannot use both 'prompt' and '--from-val' together.")
        sys.exit(1)
    if not args.from_val and not args.prompt:
        print("Error: Must provide either a 'prompt' argument or '--from-val'.")
        sys.exit(1)

    device = torch.device(args.device)
    enc = tiktoken.get_encoding("gpt2")
    decode = lambda l: enc.decode(l)

    # Load config and model
    print(f"Loading model from {args.model_path}...")
    config = ModelMOEConfig.from_json(args.config)
    model = load_model(args.model_path, config, device, map_to=args.device)
    model.to(device)

    model_size = get_model_size_mb(model)
    print(f"Model size: {model_size:.2f} MB")
    print(f"Generation parameters:")
    print(f"  Max tokens:  {args.max_tokens}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Top-k:       {args.top_k}")
    print(f"  Top-p:       {args.top_p}")
    print(f"  Rep. penalty:{args.repetition_penalty}")
    print("=" * 50)

    # Build input tokens from prompt or val.bin
    if args.from_val:
        print(f"Sampling {args.prompt_len} tokens from val.bin in '{args.data_dir}'...")
        input_tokens = get_val_prompt(args.data_dir, args.val_suffix, args.prompt_len, device)
        seed_tokens = replace_token_sequences(input_tokens[0].tolist())
        print(f"[Seed from val.bin]:")
        print(decode(seed_tokens), end="", flush=True)
    else:
        start_ids = enc.encode_ordinary(args.prompt)
        input_tokens = torch.tensor(
            start_ids, dtype=torch.long, device=device
        )[None, ...]
        print(decode(replace_token_sequences(start_ids)), end="", flush=True)
    print("\n" + "=" * 50 + "\nModel's generated tokens\n" + "=" * 50)
    # Generate tokens
    start_time = time.time()
    token_count = 0
    all_tokens = []
    if hasattr(model, "set_expert_usage_tracking"):
        model.reset_expert_usage()
        model.set_expert_usage_tracking(True)

    generator = generate_tokens(
        model=model,
        tokens=input_tokens,
        max_new_tokens=args.max_tokens,
        block_size=config.block_size,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        eot_token_id=enc.eot_token,
    )
    
    prev_len = 0
    for token in generator:
        all_tokens.append(token.item())
        token_count += 1

        # Replace multi-token UTF-8 byte sequences with single ASCII tokens
        # before decoding - handles curly quotes, dashes, ellipsis cleanly
        clean_tokens = replace_token_sequences(all_tokens)
        full_decoded = decode(clean_tokens)

        # Print only the newly added suffix since last iteration
        print(full_decoded[prev_len:], end="", flush=True)
        prev_len = len(full_decoded)

    if hasattr(model, "set_expert_usage_tracking"):
        model.set_expert_usage_tracking(False)
    
    total_duration = time.time() - start_time
    tokens_per_second = token_count / total_duration if total_duration > 0 else 0
    #print("\n"," ".join([str(token) for token in all_tokens]))
    print("\n" + "=" * 50)
    print(f"Generated {token_count} tokens in {total_duration:.2f} seconds")
    print(f"Speed: {tokens_per_second:.2f} tokens/second")
    print_expert_usage(model)

if __name__ == "__main__":
    main()
