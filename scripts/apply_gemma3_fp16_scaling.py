#!/usr/bin/env python3
"""
Apply Gemma3 FP16 residual stream scaling to a HuggingFace model.

This implements the weight-only transform used by ANEMLL's Gemma3 converter:
  1) embed_tokens.weight *= alpha
  2) post_attention_layernorm.weight = alpha * (1 + w) - 1
  3) post_feedforward_layernorm.weight = alpha * (1 + w) - 1

This shrinks the residual stream without changing model behavior (RMSNorm is
scale-invariant and the final norm cancels the global scale).

Usage:
  python scripts/apply_gemma3_fp16_scaling.py \
    --model-id google/gemma-3-1b-it \
    --alpha 0.82 \
    --output runs/gemma3_1b_fp16scaled
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import torch


GEMMA3_SCALING_FACTORS = {
    "gemma-3-270m": 0.48,
    "gemma-3-1b": 0.82,
    "gemma-3-4b-it-qat": 0.1875,
    "gemma-3-4b": 0.5,
    "gemma-3n-e2b": 0.5,
    "gemma-3n-e4b": 0.5,
}


def _resolve_alpha(alpha: str, model_id: str) -> Optional[float]:
    if alpha.lower() == "auto":
        model_lower = model_id.lower()
        for pattern, val in GEMMA3_SCALING_FACTORS.items():
            if pattern in model_lower:
                return float(val)
        return None
    try:
        return float(alpha)
    except ValueError:
        return None


def _get_gemma3_components(model):
    # Handle both text-only and multimodal (language_model) layouts
    if hasattr(model, "model"):
        if hasattr(model.model, "language_model"):
            lm = model.model.language_model
            return getattr(lm, "embed_tokens", None), getattr(lm, "layers", None)
        return getattr(model.model, "embed_tokens", None), getattr(model.model, "layers", None)
    return None, None


def _load_tokenizer(model_id: str, trust_remote_code: bool = True):
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            use_fast=False,
            fix_mistral_regex=True,
        )
    except TypeError:
        # Older transformers may not support fix_mistral_regex.
        # Use slow tokenizer path to avoid the known fast-tokenizer regex issue.
        return AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            use_fast=False,
        )


@torch.no_grad()
def apply_fp16_scaling(model, alpha: float, verbose: bool = True) -> int:
    if alpha <= 0 or alpha > 1:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")

    embed, layers = _get_gemma3_components(model)
    if embed is None or layers is None:
        raise RuntimeError("Could not find Gemma3 embed_tokens or layers in model")

    if hasattr(embed, "weight"):
        embed.weight.mul_(alpha)
        if verbose:
            print(f"  Scaled embed_tokens.weight by {alpha}")

    scaled_norms = 0
    for layer in layers:
        for norm_name in ("post_attention_layernorm", "post_feedforward_layernorm"):
            norm = getattr(layer, norm_name, None)
            if norm is None or not hasattr(norm, "weight"):
                continue
            # Gemma3 RMSNorm gain is (1 + weight)
            norm.weight.data = alpha * (1.0 + norm.weight.data) - 1.0
            scaled_norms += 1

    if verbose:
        print(f"  Transformed {scaled_norms} post-norm weights across {len(layers)} layers")

    return scaled_norms


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply Gemma3 FP16 residual stream scaling (embed + post-norms)."
    )
    parser.add_argument("--model-id", type=str, required=True, help="HF model ID or local path")
    parser.add_argument("--alpha", type=str, default="auto",
                        help="Scaling factor (float or 'auto'). Recommended: 0.82 for 1B")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output directory")
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "float16", "bfloat16"],
                        help="Load dtype (default: float32)")
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimal output")

    args = parser.parse_args()
    verbose = not args.quiet

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    alpha = _resolve_alpha(args.alpha, args.model_id)
    if alpha is None:
        raise SystemExit(f"Invalid alpha '{args.alpha}' and no auto scale matched {args.model_id}")

    if verbose:
        print("=" * 60)
        print("Gemma3 FP16 Residual Scaling")
        print("=" * 60)
        print(f"Model:  {args.model_id}")
        print(f"Alpha:  {alpha}")
        print(f"Dtype:  {args.dtype}")
        print(f"Output: {args.output}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            dtype=dtype,
            trust_remote_code=True,
        )
    except TypeError:
        # Backward compatibility with older transformers.
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
    # Keep script portable: no accelerate dependency via device_map.
    model.to("cpu")

    model_type = str(getattr(model.config, "model_type", "")).lower()
    if "gemma3" not in model_type:
        raise SystemExit(f"Model does not look like Gemma3 (model_type={model_type!r})")

    if verbose:
        print("\nApplying scaling...")
    apply_fp16_scaling(model, alpha=alpha, verbose=verbose)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\nSaving scaled model to {output_dir}...")
    model.save_pretrained(output_dir)

    try:
        tokenizer = _load_tokenizer(args.model_id, trust_remote_code=True)
        tokenizer.save_pretrained(output_dir)
        if verbose:
            print("  Saved tokenizer")
    except Exception as e:
        if verbose:
            print(f"  Warning: Could not save tokenizer: {e}")

    meta = {
        "source_model": args.model_id,
        "alpha": alpha,
        "notes": "Gemma3 residual stream scaling (embed + post-norms)",
    }
    with open(output_dir / "fp16_scale_config.json", "w") as f:
        json.dump(meta, f, indent=2)

    if verbose:
        print("\nDone.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
