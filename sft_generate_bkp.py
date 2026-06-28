"""
sft_generate.py — Interactive inference for SFT-tuned MoE Transformer
======================================================================
Loads a checkpoint produced by sft_train.ipynb and generates responses
to instruction prompts using the exact Alpaca prompt format that
sft_prepare.py used during data preparation.

Prompt format (mirrors sft_prepare.py exactly)
-----------------------------------------------
  Without context:
      Below is an instruction that describes a task. Write a response that
      appropriately completes the request.

      ### Instruction:
      <your instruction>

      ### Response:

  With context (--context "..."):
      <your context>

      ### Instruction:
      <your instruction>

      ### Response:

Usage examples
--------------
  # Single instruction, streamed output
  python sft_generate.py ./sft_checkpoints/sft_ckpt_final.pt \\
      --instruction "Explain what a transformer neural network is."

  # With additional context
  python sft_generate.py ./sft_checkpoints/sft_ckpt_final.pt \\
      --instruction "Summarise the following text." \\
      --context "The mitochondria is the powerhouse of the cell..."

  # Interactive REPL (no --instruction flag)
  python sft_generate.py ./sft_checkpoints/sft_ckpt_final.pt --interactive

  # Eval file mode — run model on held-out examples from sft_prepare.py
  python sft_generate.py ./sft_checkpoints/sft_ckpt_final.pt \\
      --eval-file ./sft_data/val_examples.jsonl

  # Eval file, filter to a single dataset source
  python sft_generate.py ./sft_checkpoints/sft_ckpt_final.pt \\
      --eval-file ./sft_data/val_examples.jsonl --eval-source gsm8k

  # Tune sampling
  python sft_generate.py ./sft_checkpoints/sft_ckpt_final.pt \\
      --instruction "Write a haiku about rain." \\
      --temperature 0.9 --top-p 0.95 --top-k 50 --max-tokens 200

  # Greedy decoding (temperature=0)
  python sft_generate.py ./sft_checkpoints/sft_ckpt_final.pt \\
      --instruction "What is 12 times 12?" \\
      --temperature 0

  # Run on CPU
  python sft_generate.py ./sft_checkpoints/sft_ckpt_final.pt \\
      --instruction "Hello!" --device cpu
"""

import argparse
import json
import os
import sys
import time
import warnings

import torch
import torch.nn.functional as F
import tiktoken
from sft_prepare import ALPACA_NO_INPUT, ALPACA_WITH_INPUT
from model_moe import MoETransformer

warnings.filterwarnings("ignore")


# The boundary string the model was trained to generate responses after.
# Used to cleanly slice the response out of the full decoded sequence.
RESPONSE_BOUNDARY = "### Response:\n"

# ── GPT-2 multi-byte UTF-8 token sequence replacements ───────────────────────
# Inherited from generate_moe.py — keeps curly quotes, dashes, ellipsis clean.
_TOKEN_SEQ_REPLACEMENTS = [
    ([447, 250], [1]),
    ([447, 251], [1]),
    ([447, 252], [11]),
    ([447, 247], [705]),
]
END_TOKEN_STR    = "\n### END"



def replace_token_sequences(tokens: list) -> list:
    """Replace known multi-token UTF-8 byte sequences with clean ASCII tokens."""
    for seq, replacement in _TOKEN_SEQ_REPLACEMENTS:
        seq_len = len(seq)
        result = []
        i = 0
        while i < len(tokens):
            if tokens[i : i + seq_len] == seq:
                result.extend(replacement)
                i += seq_len
            else:
                result.append(tokens[i])
                i += 1
        tokens = result
    return tokens


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_sft_model(
    checkpoint_path: str,
    config_path: str,
    device: torch.device,
) -> tuple[MoETransformer, dict]:
    """
    Load an SFT checkpoint 

    Returns
    -------
    model  : MoETransformer  ready for inference (eval mode, on device)
    config : dict            the model_config sub-dict used to build the model
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # ── Determine whether this is a wrapped or raw state_dict ────────────────
    if isinstance(raw, dict) and "model" in raw:
        state_dict    = raw["model"]
        saved_cfg     = raw.get("config", None)
        saved_iter    = raw.get("iter",   "unknown")
    else:
        # Raw state_dict (e.g. saved with accelerator.save(model.state_dict()))
        state_dict    = raw
        saved_cfg     = None
        saved_iter    = "unknown"

    # ── Resolve model architecture config ────────────────────────────────────
    if saved_cfg is not None and "model_config" in saved_cfg:
        mc = saved_cfg["model_config"]
        print(f"  Config source : embedded in checkpoint (iter={saved_iter})")
    else:
        # Fall back to the config file on disk
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"No config embedded in checkpoint and '{config_path}' not found. "
                "Pass --config pointing to your sft_config.json."
            )
        with open(config_path, "r") as f:
            disk_cfg = json.load(f)
        mc = disk_cfg["model_config"]
        print(f"  Config source : {config_path}")

    print(
        f"  Architecture  : d_model={mc['d_model']}  layers={mc['num_layers']}  "
        f"experts={mc['num_experts']}  heads={mc['num_heads']}  d_ff={mc['d_ff']}"
    )

    # ── Build model ───────────────────────────────────────────────────────────
    model = MoETransformer(
        vocab_size  = mc["vocab_size"],
        d_model     = mc["d_model"],
        num_heads   = mc["num_heads"],
        d_ff        = mc["d_ff"],
        num_layers  = mc["num_layers"],
        num_experts = mc["num_experts"],
        max_seq_len = mc["block_size"],
        top_k       = 2,
        dropout     = 0.0,  # always 0 at inference
    )

    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing:
        print(f"  [WARN] Missing keys   : {missing}")
    if unexpected:
        print(f"  [WARN] Unexpected keys: {unexpected}")

    model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    size_mb      = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 ** 2
    print(f"  Parameters    : {total_params:,}  ({size_mb:.1f} MB)")

    return model, mc


# ─────────────────────────────────────────────────────────────────────────────
# Prompt building
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt(instruction: str, context: str = "") -> str:
    """
    Build the Alpaca-format prompt string that matches what sft_prepare.py
    wrote into the training binary files.

    Parameters
    ----------
    instruction : str   The user's instruction / question.
    context     : str   Optional additional context / input text.
                        When non-empty, uses the ALPACA_WITH_INPUT template.
    """
    instruction = instruction.strip()
    context     = context.strip()
    if context:
        return ALPACA_WITH_INPUT.format(instruction=instruction, input=context)
    return ALPACA_NO_INPUT.format(instruction=instruction)


# ─────────────────────────────────────────────────────────────────────────────
# Token generation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_tokens(
    model: MoETransformer,
    tokens: torch.Tensor,
    max_new_tokens: int,
    block_size: int,
    temperature: float = 0.8,
    top_k: int = 0,
    top_p: float = 0.0,
    repetition_penalty: float = 1.0,
    eot_token_id: int | None = None,
    end_sequence=None
):
    """
    Autoregressive token generator — yields one token tensor at a time so the
    caller can stream output to the terminal as it is produced.

    Sampling pipeline (applied in order):
        1. Repetition penalty  — discourages repeating tokens already in context
        2. Temperature scaling — controls sharpness of the distribution
        3. Top-k filtering     — keeps only the k highest-probability tokens
        4. Top-p filtering     — nucleus sampling; keeps the smallest set whose
                                 cumulative probability exceeds p
        5. Multinomial sample  — draw one token from the filtered distribution

    Special cases:
        temperature <= 0  → greedy argmax (no sampling at all)
        top_k == 0        → top-k disabled
        top_p == 0.0      → top-p disabled
        repetition_penalty == 1.0 → penalty disabled
    """
    recent_tokens = []   # sliding window to detect ### END sequence
    end_seq_len   = len(end_sequence) if end_sequence else 0
    model.eval()

    for _ in range(max_new_tokens):
        # Crop to the model's maximum context window
        input_tokens = tokens[:, -block_size:]

        # MoETransformer returns (logits, aux_loss); aux_loss is ignored here
        logits, _ = model(input_tokens)
        logits = logits[:, -1, :]  # (1, vocab_size) — last position only
        
        # ── Repetition penalty ────────────────────────────────────────────────
        if repetition_penalty != 1.0:
            for token_id in input_tokens[0].unique():
                if logits[0, token_id] > 0:
                    logits[0, token_id] /= repetition_penalty
                else:
                    logits[0, token_id] *= repetition_penalty

        # ── Greedy shortcut ───────────────────────────────────────────────────
        if temperature <= 0:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_token], dim=1)
            yield next_token
            tok_id = next_token.item()
            if eot_token_id is not None and tok_id == eot_token_id:
                break
            if end_sequence:
                recent_tokens.append(tok_id)
            if len(recent_tokens) > end_seq_len:
                recent_tokens.pop(0)
            if recent_tokens == end_sequence:
                break
            continue

        # ── Temperature ───────────────────────────────────────────────────────
        logits = logits / temperature

        # ── Top-k ─────────────────────────────────────────────────────────────
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")

        # ── Top-p (nucleus) ───────────────────────────────────────────────────
        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(
                F.softmax(sorted_logits, dim=-1), dim=-1
            )
            # Remove tokens whose cumulative probability exceeds the threshold.
            # Shift right by one so the token that pushes us over p is kept.
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0]  = False
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float("-inf")
        # ── Sample ────────────────────────────────────────────────────────────
        probs      = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        tokens     = torch.cat([tokens, next_token], dim=1)
        yield next_token
        tok_id = next_token.item()
        if eot_token_id is not None and tok_id == eot_token_id:
            break
        if end_sequence:
            recent_tokens.append(tok_id)
            if len(recent_tokens) > end_seq_len:
                recent_tokens.pop(0)
            if recent_tokens == end_sequence:
                break


# Single-turn generation (one instruction → one response)


def run_single(
    model: MoETransformer,
    enc: tiktoken.Encoding,
    block_size: int,
    instruction: str,
    context: str,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    """
    Format the prompt, stream the response to stdout, then print timing stats.
    """
    prompt       = build_prompt(instruction, context)
    prompt_ids   = enc.encode_ordinary(prompt)
    prompt_len   = len(prompt_ids)
    input_tensor = torch.tensor(
        prompt_ids, dtype=torch.long, device=device
    ).unsqueeze(0)  # (1, prompt_len)

    # Print the prompt header so the user can see what was sent
    print("\n" + "=" * 70)
    print(f"INSTRUCTION : {instruction}")
    if context:
        print(f"CONTEXT     : {context[:120]}{'...' if len(context) > 120 else ''}")
    print("=" * 70)
    print("RESPONSE    : ", end="", flush=True)

    start_time   = time.time()
    token_count  = 0
    all_new_toks = []   # raw generated token ids (before replacement)
    prev_len     = 0    # character length of last decoded string

    for next_tok in generate_tokens(
        model               = model,
        tokens              = input_tensor,
        max_new_tokens      = args.max_tokens,
        block_size          = block_size,
        temperature         = args.temperature,
        top_k               = args.top_k,
        top_p               = args.top_p,
        repetition_penalty  = args.repetition_penalty,
        eot_token_id        = enc.eot_token,
    ):
        tok_id = next_tok.item()

        # Stop streaming at EOT — don't print the token itself
        if tok_id == enc.eot_token:
            break

        all_new_toks.append(tok_id)
        token_count += 1

        # Apply multi-byte UTF-8 sequence replacements before decoding so
        # curly quotes, em-dashes, and ellipses render cleanly in the terminal.
        clean = replace_token_sequences(all_new_toks)
        decoded = enc.decode(clean)
        decoded = decoded.replace("\n### END", "").rstrip()
        # Print only the newly added suffix since the last iteration
        print(decoded[prev_len:], end="", flush=True)
        prev_len = len(decoded)

    elapsed = time.time() - start_time
    tps     = token_count / elapsed if elapsed > 0 else 0.0
    print(f"\n{'-' * 70}")
    print(f"  {token_count} tokens  |  {elapsed:.2f}s  |  {tps:.1f} tok/s")


# ─────────────────────────────────────────────────────────────────────────────
# Eval-file mode — batch inference against val_examples.jsonl
# ─────────────────────────────────────────────────────────────────────────────

def run_eval_file(
    model: MoETransformer,
    enc: tiktoken.Encoding,
    block_size: int,
    eval_file: str,
    args: argparse.Namespace,
    device: torch.device,
    source_filter: str | None = None,
) -> None:
    """
    Load val_examples.jsonl produced by sft_prepare.py and run the model on
    every example, printing the model's response alongside the ground-truth
    reference so you can eyeball quality across datasets.

    Each line of the JSONL must be:
        {"prompt": str, "response": str, "source": str}

    Parameters
    ----------
    eval_file     : path to val_examples.jsonl
    source_filter : if set, only run examples whose "source" field matches
                    this string (e.g. "gsm8k", "code_alpaca")
    """
    if not os.path.exists(eval_file):
        print(f"[ERROR] Eval file not found: {eval_file}")
        print("  Run sft_prepare.py first to generate val_examples.jsonl.")
        return

    # ── Load examples ─────────────────────────────────────────────────────────
    examples: list[dict] = []
    with open(eval_file, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] Skipping malformed line {line_no}: {e}")
                continue
            if source_filter and rec.get("source", "") != source_filter:
                continue
            examples.append(rec)
    import random
    random.shuffle(examples)
    if not examples:
        msg = f"No examples found in {eval_file}"
        if source_filter:
            msg += f" with source='{source_filter}'"
        print(f"[WARN] {msg}")
        return

    # ── Count examples per source for the header ──────────────────────────────
    source_counts: dict[str, int] = {}
    for ex in examples:
        src = ex.get("source", "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1

    print("\n" + "=" * 70)
    print(f"EVAL FILE : {eval_file}")
    print(f"Examples  : {len(examples):,}")
    for src, cnt in sorted(source_counts.items()):
        print(f"  {src:<30} {cnt:>4} examples")
    if source_filter:
        print(f"Filter    : source == '{source_filter}'")
    print("=" * 70)

    # ── Run inference on each example ─────────────────────────────────────────
    total_tokens = 0
    total_time   = 0.0

    for idx, ex in enumerate(examples, 1):
        prompt    = ex["prompt"]
        reference = ex["response"]
        source    = ex.get("source", "unknown")

        prompt_ids   = enc.encode_ordinary(prompt)
        input_tensor = torch.tensor(
            prompt_ids, dtype=torch.long, device=device
        ).unsqueeze(0)

        print(f"\n{'─' * 70}")
        print(f"[{idx}/{len(examples)}]  source={source}")

        # Extract the instruction text from the prompt for display.
        # Template is: {input}\n\n### Instruction:\n{instruction}\n\n### Response:\n
        # For no-input prompts: ### Instruction:\n{instruction}\n\n### Response:\n
        instr_marker = "### Instruction:\n"
        resp_marker  = "### Response:\n"
        if instr_marker in prompt:
            instr_start   = prompt.index(instr_marker) + len(instr_marker)
            instr_end     = prompt.index(resp_marker) if resp_marker in prompt else len(prompt)
            instr_display = prompt[instr_start:instr_end].strip()
            # If there is a context block before ### Instruction, show it too
            context_block = prompt[:prompt.index(instr_marker)].strip()
            if context_block:
                instr_display = f"[CTX] {context_block[:80]}{'...' if len(context_block) > 80 else ''}\n      {instr_display}"
        else:
            instr_display = prompt.strip()

        # Truncate long instructions for display only
        #if len(instr_display) > 200:
        #    instr_display = instr_display[:200] + " ..."
        print(f"INSTRUCTION : {instr_display}")

        # ── Generate ──────────────────────────────────────────────────────────
        start_time   = time.time()
        token_count  = 0
        all_new_toks: list[int] = []
        prev_len     = 0

        print("MODEL       : ", end="", flush=True)
        for next_tok in generate_tokens(
            model              = model,
            tokens             = input_tensor,
            max_new_tokens     = args.max_tokens,
            block_size         = block_size,
            temperature        = args.temperature,
            top_k              = args.top_k,
            top_p              = args.top_p,
            repetition_penalty = args.repetition_penalty,
            eot_token_id       = enc.eot_token,
        ):
            tok_id = next_tok.item()
            if tok_id == enc.eot_token:
                break
            all_new_toks.append(tok_id)
            token_count += 1
            clean   = replace_token_sequences(all_new_toks)
            decoded = enc.decode(clean)
            decoded = decoded.replace("\n### END", "").rstrip()
            print(decoded[prev_len:], end="", flush=True)
            prev_len = len(decoded)

        elapsed       = time.time() - start_time
        total_tokens += token_count
        total_time   += elapsed

        # ── Ground truth reference ────────────────────────────────────────────
        ref_display = reference.strip()
        if len(ref_display) > 400:
            ref_display = ref_display[:400] + " ..."
        print(f"\nREFERENCE   : {ref_display}")
        print(f"  [{token_count} tokens | {elapsed:.2f}s | {token_count/elapsed:.1f} tok/s]")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    avg_tps = total_tokens / total_time if total_time > 0 else 0.0
    print(f"EVAL COMPLETE")
    print(f"  Examples evaluated : {len(examples):,}")
    print(f"  Total tokens gen.  : {total_tokens:,}")
    print(f"  Total time         : {total_time:.1f}s")
    print(f"  Avg throughput     : {avg_tps:.1f} tok/s")
    print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# Interactive REPL
# ─────────────────────────────────────────────────────────────────────────────

def run_interactive(
    model: MoETransformer,
    enc: tiktoken.Encoding,
    block_size: int,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    """
    Read–Eval–Print loop: accepts instructions from stdin one at a time.

    Special commands
    ----------------
    /context <text>   Set a persistent context/input that is prepended to
                      every subsequent instruction (like a system prompt).
    /clear            Clear the persistent context.
    /params           Print current sampling parameters.
    /quit  or  /exit  Exit the REPL.
    """
    print("\n" + "=" * 70)
    print("SFT Interactive Mode  (type /quit to exit, /help for commands)")
    print("=" * 70)

    persistent_context = ""

    while True:
        try:
            raw = input("\n>>> Instruction: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not raw:
            continue

        # ── Special commands ──────────────────────────────────────────────────
        if raw.startswith("/"):
            cmd = raw.lower()

            if cmd in ("/quit", "/exit"):
                print("Goodbye.")
                break

            elif cmd == "/help":
                print(
                    "  /context <text>  — set persistent context for all prompts\n"
                    "  /clear           — clear persistent context\n"
                    "  /params          — show current sampling parameters\n"
                    "  /quit            — exit"
                )

            elif raw.lower().startswith("/context "):
                persistent_context = raw[len("/context "):].strip()
                print(f"  Context set: '{persistent_context[:80]}...'")

            elif cmd == "/clear":
                persistent_context = ""
                print("  Context cleared.")

            elif cmd == "/params":
                print(
                    f"  temperature      = {args.temperature}\n"
                    f"  top_k            = {args.top_k}\n"
                    f"  top_p            = {args.top_p}\n"
                    f"  repetition_penalty = {args.repetition_penalty}\n"
                    f"  max_tokens       = {args.max_tokens}"
                )
            else:
                print(f"  Unknown command: {raw}")
            continue

        # ── Normal instruction ────────────────────────────────────────────────
        run_single(
            model       = model,
            enc         = enc,
            block_size  = block_size,
            instruction = raw,
            context     = persistent_context,
            args        = args,
            device      = device,
        )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate responses from an SFT-tuned MoE Transformer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Positional ────────────────────────────────────────────────────────────
    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to the SFT checkpoint (.pt) produced by sft_train.ipynb.",
    )

    # ── Prompt ────────────────────────────────────────────────────────────────
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument(
        "--instruction", "-i",
        type=str,
        default=None,
        metavar="TEXT",
        help="Instruction text to send to the model (single-turn mode).",
    )
    prompt_group.add_argument(
        "--interactive",
        action="store_true",
        help="Launch an interactive REPL instead of a single-turn generation.",
    )
    prompt_group.add_argument(
        "--eval-file",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Path to val_examples.jsonl produced by sft_prepare.py. "
            "Runs the model on every example and prints model response "
            "vs ground-truth reference side by side."
        ),
    )

    parser.add_argument(
        "--eval-source",
        type=str,
        default=None,
        metavar="NAME",
        help=(
            "When using --eval-file, only evaluate examples whose 'source' "
            "field matches this string (e.g. gsm8k, code_alpaca, ultrachat). "
            "Omit to evaluate all sources."
        ),
    )

    parser.add_argument(
        "--context", "-c",
        type=str,
        default="",
        metavar="TEXT",
        help=(
            "Optional context / input text paired with --instruction. "
            "Triggers the ALPACA_WITH_INPUT template (same as sft_prepare.py). "
            "Ignored in --interactive mode (use /context command instead)."
        ),
    )

    # ── Config ────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--config",
        type=str,
        default="sft_config.json",
        metavar="PATH",
        help=(
            "Path to sft_config.json. Only needed when the checkpoint does not "
            "carry an embedded config dict (default: sft_config.json)."
        ),
    )

    # ── Sampling ─────────────────────────────────────────────────────────────
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        metavar="N",
        help="Maximum number of new tokens to generate (default: 512).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.,
        metavar="T",
        help=(
            "Sampling temperature. "
            "Higher → more random, lower → more focused. "
            "0 = greedy argmax (default: 0)."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        metavar="K",
        help=(
            "Keep only the top-K most probable tokens before sampling. "
            "0 = disabled (default: 0)."
        ),
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.,
        metavar="P",
        help=(
            "Nucleus sampling: keep the smallest set of tokens whose cumulative "
            "probability exceeds P. 0.0 = disabled (default: 0.)."
        ),
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        metavar="RP",
        help=(
            "Penalise tokens that already appear in the context window. "
            "1.0 = disabled, >1.0 = stronger penalty (default: 1.0)."
        ),
    )

    # ── Hardware ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run inference on (default: cuda).",
    )

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_arguments()

    # ── Validate mode ─────────────────────────────────────────────────────────
    if not args.interactive and not args.instruction and not args.eval_file:
        print(
            "Error: provide one of:\n"
            "  --instruction <text>   single-turn generation\n"
            "  --interactive          interactive REPL\n"
            "  --eval-file <path>     batch eval against val_examples.jsonl\n"
            "Run with --help for full usage."
        )
        sys.exit(1)

    # ── Device ────────────────────────────────────────────────────────────────
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available, falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)

    # ── Tokeniser ─────────────────────────────────────────────────────────────
    enc = tiktoken.get_encoding("gpt2")
    end_token_ids = set(enc.encode_ordinary(END_TOKEN_STR))
    # ── Model ─────────────────────────────────────────────────────────────────
    model, mc = load_sft_model(args.checkpoint, args.config, device)
    block_size = mc["block_size"]

    print(f"\n  Sampling params:")
    print(f"    temperature        = {args.temperature}")
    print(f"    top_k              = {args.top_k}")
    print(f"    top_p              = {args.top_p}")
    print(f"    repetition_penalty = {args.repetition_penalty}")
    print(f"    max_tokens         = {args.max_tokens}")
    print(f"    device             = {args.device}")

    # ── Dispatch ──────────────────────────────────────────────────────────────
    if args.interactive:
        run_interactive(model, enc, block_size, args, device)
    elif args.eval_file:
        run_eval_file(
            model         = model,
            enc           = enc,
            block_size    = block_size,
            eval_file     = args.eval_file,
            args          = args,
            device        = device,
            source_filter = args.eval_source,
        )
    else:
        run_single(
            model       = model,
            enc         = enc,
            block_size  = block_size,
            instruction = args.instruction,
            context     = args.context,
            args        = args,
            device      = device,
        )


if __name__ == "__main__":
    main()
