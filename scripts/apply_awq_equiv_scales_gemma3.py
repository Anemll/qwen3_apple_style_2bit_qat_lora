#!/usr/bin/env python3
"""
Apply AWQ-equivalent scale transforms to a Gemma3 model using iMatrix stats.

Gemma3-specific differences vs Qwen3:
  - MLP uses pre_feedforward_layernorm (not post_attention_layernorm)
  - RMSNorm gain is (1 + weight), so inverse-scaling must be:
      w_new = (1 + w_old) / s - 1

Transforms (per layer):
  1) input_layernorm → q_proj, k_proj, v_proj
  2) v_proj → o_proj (GQA-aware fold/expand)
  3) pre_feedforward_layernorm → gate_proj, up_proj
  4) up_proj → down_proj

Usage:
  python scripts/compute_imatrix.py --model google/gemma-3-1b-it --out runs/imatrix.pt
  python scripts/apply_awq_equiv_scales_gemma3.py \
      --model-id google/gemma-3-1b-it \
      --imatrix runs/imatrix.pt \
      --alpha 0.5 \
      --output runs/gemma3_1b_awqscaled
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn


def compute_awq_scales(
    sigma2: torch.Tensor,
    alpha: float = 0.5,
    min_scale: float = 0.1,
    max_scale: float = 10.0,
) -> torch.Tensor:
    """Compute AWQ scaling factors from activation stats (diag-iMatrix)."""
    sigma2 = sigma2.float()
    rms = torch.sqrt(sigma2 + 1e-8)
    s = rms / (rms.mean() + 1e-8)
    s = s ** alpha
    s = s.clamp(min=min_scale, max=max_scale)
    s = s / (s.mean() + 1e-8)
    s = s.clamp(min=min_scale, max=max_scale)
    return s


def _scale_gemma3_rmsnorm(norm: nn.Module, scales: torch.Tensor) -> None:
    """Apply inverse scaling to Gemma3 RMSNorm with gain (1 + weight)."""
    if not hasattr(norm, "weight") or norm.weight is None:
        raise ValueError("Norm module has no weight")
    scales = scales.to(norm.weight.device)
    if scales.numel() != norm.weight.numel():
        raise ValueError(f"scale_ln_fcs: scales len {scales.numel()} != ln.weight len {norm.weight.numel()}")

    # Gemma3 gain is (1 + w). We want output / s.
    norm.weight.data = (1.0 + norm.weight.data) / scales - 1.0
    if hasattr(norm, "bias") and norm.bias is not None:
        norm.bias.data /= scales


def scale_ln_fcs_gemma3(
    ln: nn.Module,
    fcs: list,
    scales: torch.Tensor,
) -> None:
    """Gemma3-aware scale_ln_fcs using (1 + w) RMSNorm gain."""
    _scale_gemma3_rmsnorm(ln, scales)
    for fc in fcs:
        if fc.weight.shape[1] != scales.numel():
            raise ValueError(f"scale_ln_fcs: fc.in_features={fc.weight.shape[1]} != scales len {scales.numel()}")
        fc.weight.data *= scales.view(1, -1)


def scale_fc_fc(
    prev_fc: nn.Module,
    next_fcs: list,
    scales: torch.Tensor,
) -> None:
    """Apply AWQ scale_fc_fc: Linear → Linear(s)."""
    scales = scales.to(prev_fc.weight.device)
    if prev_fc.weight.shape[0] != scales.numel():
        raise ValueError(f"scale_fc_fc: prev_fc.out_features={prev_fc.weight.shape[0]} != scales len {scales.numel()}")

    prev_fc.weight.data /= scales.view(-1, 1)
    if prev_fc.bias is not None:
        prev_fc.bias.data /= scales

    for fc in next_fcs:
        if fc.weight.shape[1] != scales.numel():
            raise ValueError(f"scale_fc_fc: next_fc.in_features={fc.weight.shape[1]} != scales len {scales.numel()}")
        fc.weight.data *= scales.view(1, -1)


def fold_gqa_scales(
    scales_full: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Fold head-space scales to KV-head space by averaging repeated heads."""
    num_groups = num_heads // num_kv_heads
    s = scales_full.view(num_kv_heads, num_groups, head_dim)
    return s.mean(dim=1).reshape(num_kv_heads * head_dim)


def expand_gqa_scales(
    scales_kv: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Expand KV-head scales to head-space by repetition."""
    num_groups = num_heads // num_kv_heads
    s = scales_kv.view(num_kv_heads, 1, head_dim)
    s = s.repeat(1, num_groups, 1)
    return s.reshape(num_heads * head_dim)


def _get_layers_and_prefix(model) -> tuple[list, str]:
    # Handle text-only and multimodal layouts
    if hasattr(model, "model"):
        if hasattr(model.model, "language_model"):
            layers = model.model.language_model.layers
            return layers, "model.language_model.layers"
        if hasattr(model.model, "layers"):
            layers = model.model.layers
            return layers, "model.layers"
    raise RuntimeError("Could not resolve model layers for Gemma3")


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


def apply_awq_to_gemma3(
    model: nn.Module,
    imatrix: Dict[str, torch.Tensor],
    alpha: float = 0.5,
    verbose: bool = True,
) -> Dict[str, object]:
    sigma2_dict = imatrix.get("sigma2", imatrix)

    config = model.config
    model_type = str(getattr(config, "model_type", "")).lower()
    if "gemma3" not in model_type:
        raise ValueError(f"Expected Gemma3 model_type, got: {model_type!r}")
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = getattr(config, "head_dim", config.hidden_size // num_heads)

    layers, prefix_base = _get_layers_and_prefix(model)
    num_layers = len(layers)

    if verbose:
        print(f"\nApplying AWQ transforms (alpha={alpha})")
        print(f"  num_heads={num_heads}, num_kv_heads={num_kv_heads}, head_dim={head_dim}")
        print(f"  num_layers={num_layers}")

    stats = {
        "alpha": alpha,
        "num_layers": num_layers,
        "transforms_applied": 0,
    }

    for layer_idx, layer in enumerate(layers):
        prefix = f"{prefix_base}.{layer_idx}"

        attn = getattr(layer, "self_attn", None) or getattr(layer, "attention", None)
        mlp = getattr(layer, "mlp", None) or getattr(layer, "feed_forward", None)

        if attn is None or mlp is None:
            if verbose and layer_idx == 0:
                print("  [WARN] Missing attn/mlp modules; skipping AWQ for this layer")
            continue

        # === Transform 1: input_layernorm → q_proj, k_proj, v_proj ===
        attn_key = None
        for pattern in [
            f"{prefix}.self_attn.q_proj",
            f"{prefix}.self_attn.k_proj",
            f"{prefix}.self_attn.v_proj",
            f"{prefix}.attention.q_proj",
            f"{prefix}.attention.k_proj",
            f"{prefix}.attention.v_proj",
        ]:
            if pattern in sigma2_dict:
                attn_key = pattern
                break

        if attn_key and attn_key in sigma2_dict:
            sigma2 = sigma2_dict[attn_key]
            scales = compute_awq_scales(sigma2, alpha)

            scale_ln_fcs_gemma3(
                ln=layer.input_layernorm,
                fcs=[attn.q_proj, attn.k_proj, attn.v_proj],
                scales=scales,
            )
            stats["transforms_applied"] += 1
            if verbose and layer_idx == 0:
                print(f"  Layer {layer_idx}: input_layernorm → qkv (scales {scales.min():.3f}..{scales.max():.3f})")

        # === Transform 2: v_proj → o_proj (GQA-aware) ===
        o_proj_key = None
        for pattern in [
            f"{prefix}.self_attn.o_proj",
            f"{prefix}.attention.o_proj",
        ]:
            if pattern in sigma2_dict:
                o_proj_key = pattern
                break

        if o_proj_key in sigma2_dict:
            sigma2_full = sigma2_dict[o_proj_key]
            device = attn.v_proj.weight.device

            if sigma2_full.numel() != num_heads * head_dim:
                raise ValueError(f"{o_proj_key}: sigma2 len {sigma2_full.numel()} != num_heads*head_dim {num_heads*head_dim}")

            if num_kv_heads < num_heads:
                sigma2_kv = fold_gqa_scales(sigma2_full, num_heads, num_kv_heads, head_dim)
                s_kv = compute_awq_scales(sigma2_kv, alpha).to(device)
                s_full = expand_gqa_scales(s_kv, num_heads, num_kv_heads, head_dim).to(device)
            else:
                s_full = compute_awq_scales(sigma2_full, alpha).to(device)
                s_kv = s_full

            # v_proj output (row-wise)
            attn.v_proj.weight.data /= s_kv.view(-1, 1)
            if attn.v_proj.bias is not None:
                attn.v_proj.bias.data /= s_kv

            # o_proj input (column-wise)
            attn.o_proj.weight.data *= s_full.view(1, -1)

            stats["transforms_applied"] += 1
            if verbose and layer_idx == 0:
                print(f"  Layer {layer_idx}: v_proj → o_proj (GQA fold/expand)")

        # === Transform 3: pre_feedforward_layernorm → gate_proj, up_proj ===
        mlp_key = None
        for pattern in [
            f"{prefix}.mlp.gate_proj",
            f"{prefix}.mlp.up_proj",
        ]:
            if pattern in sigma2_dict:
                mlp_key = pattern
                break

        if mlp_key and mlp_key in sigma2_dict:
            sigma2 = sigma2_dict[mlp_key]
            scales = compute_awq_scales(sigma2, alpha)

            ln_ff = getattr(layer, "pre_feedforward_layernorm", None)
            if ln_ff is None:
                # Fallback for unexpected naming
                ln_ff = getattr(layer, "post_attention_layernorm", None)
            if ln_ff is None:
                raise ValueError("Missing pre_feedforward_layernorm/post_attention_layernorm on Gemma3 layer")

            scale_ln_fcs_gemma3(
                ln=ln_ff,
                fcs=[mlp.gate_proj, mlp.up_proj],
                scales=scales,
            )
            stats["transforms_applied"] += 1
            if verbose and layer_idx == 0:
                print(f"  Layer {layer_idx}: pre_ffn_ln → gate+up (scales {scales.min():.3f}..{scales.max():.3f})")

        # === Transform 4: up_proj → down_proj ===
        down_key = f"{prefix}.mlp.down_proj"
        if down_key in sigma2_dict:
            sigma2 = sigma2_dict[down_key]
            scales = compute_awq_scales(sigma2, alpha)

            scale_fc_fc(
                prev_fc=mlp.up_proj,
                next_fcs=[mlp.down_proj],
                scales=scales,
            )
            stats["transforms_applied"] += 1
            if verbose and layer_idx == 0:
                print(f"  Layer {layer_idx}: up_proj → down_proj (scales {scales.min():.3f}..{scales.max():.3f})")

    if verbose:
        print(f"\nApplied {stats['transforms_applied']} transforms across {num_layers} layers")

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply AWQ-equivalent scale transforms to Gemma3 model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model-id", type=str, required=True, help="HF model ID or local path")
    parser.add_argument("--imatrix", type=str, required=True, help="Path to iMatrix .pt file")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="AWQ scaling strength: 0.5=moderate (recommended)")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output directory")
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "float16", "bfloat16"],
                        help="Model dtype (default: float32)")
    parser.add_argument("--device", type=str, default="cpu", help="Device for processing")
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimal output")

    args = parser.parse_args()
    verbose = not args.quiet

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    if verbose:
        print("=" * 60)
        print("AWQ-Equivalent Scale Transform for Gemma3")
        print("=" * 60)
        print(f"Model:    {args.model_id}")
        print(f"iMatrix:  {args.imatrix}")
        print(f"Alpha:    {args.alpha}")
        print(f"Output:   {args.output}")
        print(f"Dtype:    {args.dtype}")

    if verbose:
        print(f"\nLoading iMatrix from {args.imatrix}...")
    imatrix = torch.load(args.imatrix, map_location="cpu", weights_only=False)
    sigma2_dict = imatrix.get("sigma2", imatrix)
    if verbose:
        print(f"  Found {len(sigma2_dict)} layer entries")

    if verbose:
        print(f"\nLoading model {args.model_id}...")
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
    model.to(args.device)

    if verbose:
        print(f"  Loaded with dtype={dtype}, device={args.device}")

    stats = apply_awq_to_gemma3(
        model=model,
        imatrix=imatrix,
        alpha=args.alpha,
        verbose=verbose,
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\nSaving AWQ-scaled model to {output_dir}...")
    model.save_pretrained(output_dir)

    try:
        tokenizer = _load_tokenizer(args.model_id, trust_remote_code=True)
        tokenizer.save_pretrained(output_dir)
        if verbose:
            print("  Saved tokenizer")
    except Exception as e:
        if verbose:
            print(f"  Warning: Could not save tokenizer: {e}")

    awq_config = {
        "source_model": args.model_id,
        "imatrix_path": args.imatrix,
        "alpha": args.alpha,
        "transforms_applied": stats["transforms_applied"],
        "num_layers": stats["num_layers"],
    }
    with open(output_dir / "awq_config.json", "w") as f:
        json.dump(awq_config, f, indent=2)

    if verbose:
        print("\nDone.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
