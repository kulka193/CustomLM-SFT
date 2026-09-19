#!/usr/bin/env python3
"""Strip training-only state from a PyTorch checkpoint.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch


def _torch_load_cpu(path: str | Path) -> Any:
    """Load a checkpoint on CPU using the safest available torch.load options."""
    load_kwargs = {"map_location": "cpu", "weights_only": True}
    try:
        return torch.load(path, mmap=True, **load_kwargs)
    except TypeError:
        # Older PyTorch builds may not support mmap and/or weights_only.
        try:
            return torch.load(path, **load_kwargs)
        except TypeError:
            return torch.load(path, map_location="cpu")


def _looks_like_state_dict(obj: Any) -> bool:
    return isinstance(obj, dict) and obj and all(
        isinstance(k, str) and torch.is_tensor(v) for k, v in obj.items()
    )


def _extract_model_state(checkpoint: Any) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Return the model state_dict and lightweight metadata to preserve."""
    metadata: dict[str, Any] = {}

    if _looks_like_state_dict(checkpoint):
        return checkpoint, metadata

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Expected a checkpoint dict or raw state_dict, "
            f"got {type(checkpoint).__name__}."
        )

    for key in ("model", "state_dict", "model_state_dict"):
        state = checkpoint.get(key)
        if _looks_like_state_dict(state):
            for meta_key in ("iter", "step", "supervised_tokens", "config"):
                if meta_key in checkpoint:
                    metadata[meta_key] = checkpoint[meta_key]
            return state, metadata

    available = ", ".join(sorted(str(k) for k in checkpoint.keys()))
    raise KeyError(
        "Could not find model weights under 'model', 'state_dict', or "
        f"'model_state_dict'. Available checkpoint keys: {available}"
    )


def _clean_state_dict(state_dict: dict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    """Detach tensors from autograd and move them to CPU for a lean save."""
    clean = OrderedDict()
    for name, tensor in state_dict.items():
        if not torch.is_tensor(tensor):
            raise TypeError(f"State entry {name!r} is not a tensor.")
        clean[name] = tensor.detach().cpu()
    return clean


def strip_model(model_path: str | Path, output_path: str | Path | None = None) -> Path:
    """Create a model-only ``.pt`` file from a training checkpoint.

    Parameters
    ----------
    model_path:
        Path to the full checkpoint, e.g. one containing ``model``,
        ``optimizer``, ``scheduler``, and training metadata.
    output_path:
        Optional destination. If omitted, writes next to the source as
        ``<source_stem>_model_only.pt``.

    Returns
    -------
    pathlib.Path
        The path to the stripped checkpoint.
    """
    src = Path(model_path).expanduser()
    if not src.exists():
        raise FileNotFoundError(src)

    dst = (
        Path(output_path).expanduser()
        if output_path is not None
        else src.with_name(f"{src.stem}_model_only.pt")
    )
    dst.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = _torch_load_cpu(src)
    state_dict, metadata = _extract_model_state(checkpoint)
    clean_state = _clean_state_dict(state_dict)

    stripped = {"model": clean_state, **metadata}
    torch.save(stripped, dst)
    return dst


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strip optimizer/scheduler state from a PyTorch model checkpoint."
    )
    parser.add_argument("model_path", help="Path to the full .pt checkpoint.")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output .pt path. Defaults to '<input>_model_only.pt'.",
    )
    args = parser.parse_args()

    out = strip_model(args.model_path, args.output)
    print(f"Saved model-only checkpoint to: {out}")


if __name__ == "__main__":
    main()
