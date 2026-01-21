#!/usr/bin/env python3
"""
Apply AWQ-equivalent scale transforms to a Qwen3 model.

AWQ (Activation-aware Weight Quantization) protects salient channels by applying
an equivalent transformation: scale weights by s, inverse-scale the preceding
layer so the block's function is preserved.

This script applies the same transformations as llm-awq's scale_ln_fcs and
scale_fc_fc, creating an "AWQ-scaled FP" model that can then be quantized
with standard methods (like V2 init) while preserving important channels.

AWQ PAIRINGS FOR QWEN3:
======================

  1. input_layernorm → (q_proj, k_proj, v_proj)
     - RMSNorm → Linear (scale_ln_fcs)
     - ln.weight /= s, fc.weight *= s (column-wise)

  2. v_proj → o_proj
     - Linear → Linear (scale_fc_fc)
     - v_proj.weight /= s (row-wise), o_proj.weight *= s (column-wise)
     - With GQA: o_proj has 2048 input channels (16 heads × 128)
                 v_proj has 1024 output channels (8 KV heads × 128)
                 Need to fold scales back with repeat_kv inverse

  3. post_attention_layernorm → (gate_proj, up_proj)
     - RMSNorm → Linear (scale_ln_fcs)

  4. up_proj → down_proj
     - Linear → Linear (scale_fc_fc)
     - up_proj.weight /= s (row-wise), down_proj.weight *= s (column-wise)

USAGE:
======

  # Step 1: Create iMatrix (importance statistics)
  python scripts/compute_imatrix.py \\
      --model Qwen/Qwen3-0.6B \\
      --tokens 100000 --seq-len 512 \\
      --out runs/imatrix.pt

  # Step 2: Apply AWQ-equivalent scaling
  python scripts/apply_awq_equiv_scales.py \\
      --model-id Qwen/Qwen3-0.6B \\
      --imatrix runs/imatrix.pt \\
      --alpha 0.5 \\
      --output runs/awq_scaled_model

  # Step 3: Initialize V2 from AWQ-scaled model
  python scripts/init_model_v2.py \\
      --model-id runs/awq_scaled_model \\
      --output runs/v2_awq \\
      --config q4a4

THEORY:
=======

  For a linear layer y = W @ x, AWQ does:
    y = (W * s) @ (x / s) = W @ x  (equivalent!)

  The scaling s is computed from activation statistics (iMatrix σ²):
    s = (σ² / mean(σ²))^alpha
    s = clamp(s, 0.1, 10.0)  # prevent extremes

  With alpha=0.5:
    - High σ² → s > 1 → weight columns scaled up → smaller Q values → less error
    - Low σ²  → s < 1 → weight columns scaled down → larger Q values → more error
    - Net effect: quantization error is concentrated in less important channels

Author: ANEMLL Team
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))


def compute_awq_scales(
    sigma2: torch.Tensor,
    alpha: float = 0.5,
    min_scale: float = 0.1,
    max_scale: float = 10.0,
) -> torch.Tensor:
    """Compute AWQ scaling factors from activation second moments (diag).

    iMatrix gives sigma2 = E[x^2]. We convert to RMS magnitude first:
        rms = sqrt(E[x^2])
    Then:
        s = (rms / mean(rms)) ** alpha
    Clamp and re-normalize to keep mean(s)≈1.

    Returns:
        s: [features] scale factors (mean-normalized).
    """
    sigma2 = sigma2.float()
    rms = torch.sqrt(sigma2 + 1e-8)
    s = rms / (rms.mean() + 1e-8)
    s = s ** alpha
    s = s.clamp(min=min_scale, max=max_scale)
    s = s / (s.mean() + 1e-8)
    s = s.clamp(min=min_scale, max=max_scale)
    return s


def scale_ln_fcs(
    ln: nn.Module,  # RMSNorm or LayerNorm
    fcs: list,      # List of Linear layers following the norm
    scales: torch.Tensor,  # [in_features] scaling factors
) -> None:
    """Apply AWQ scale_ln_fcs: RMSNorm → Linear(s).

    Transforms: ln.weight /= s, fc.weight *= s (column-wise)
    Net effect: x_out = ln(x) → x_out * s → fc input scaled by s
                fc output = W @ (x_out * s) = (W * s) @ x_out
    So we scale fc columns by s and divide ln weight by s to compensate.
    """
    scales = scales.to(ln.weight.device)

    # RMSNorm: output = x * (weight / rms(x))
    # To make output effectively x * (weight / rms(x)) / s,
    # we divide weight by s
    ln.weight.data /= scales

    # For each following fc: multiply input columns by s
    # W @ (x/s * s) = W @ x, but we want W_new @ (x/s) = W @ x
    # So W_new = W * s (column-wise, i.e., multiply each input channel)
    for fc in fcs:
        fc.weight.data *= scales.view(1, -1)


def scale_fc_fc(
    prev_fc: nn.Module,  # Previous Linear (output being scaled)
    next_fcs: list,      # Next Linear(s) (input being inverse-scaled)
    scales: torch.Tensor,  # [features] scaling factors
) -> None:
    """Apply AWQ scale_fc_fc: Linear → Linear(s).

    Transforms: prev_fc.weight /= s (row-wise, output channels)
                prev_fc.bias /= s (if exists)
                next_fc.weight *= s (column-wise, input channels)
    """
    scales = scales.to(prev_fc.weight.device)

    # Previous fc: divide output channels by s
    # This makes fc output = (W/s) @ x = (W @ x) / s
    prev_fc.weight.data /= scales.view(-1, 1)
    if prev_fc.bias is not None:
        prev_fc.bias.data /= scales

    # Next fc: multiply input columns by s
    # This makes fc output = (W*s) @ (input/s) = W @ input
    for fc in next_fcs:
        fc.weight.data *= scales.view(1, -1)


def fold_gqa_scales(
    scales: torch.Tensor,  # [o_proj_in_features] = [num_heads * head_dim]
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Fold GQA-expanded scales back to KV head dimension.

    o_proj input has num_heads * head_dim channels (after repeat_kv expansion).
    v_proj output has num_kv_heads * head_dim channels.

    We need to map scales from o_proj input space back to v_proj output space,
    averaging across the repeated heads.

    Args:
        scales: [num_heads * head_dim] scales from o_proj input statistics
        num_heads: Number of query heads
        num_kv_heads: Number of KV heads (< num_heads for GQA)
        head_dim: Dimension per head

    Returns:
        folded_scales: [num_kv_heads * head_dim] scales for v_proj output
    """
    num_groups = num_heads // num_kv_heads  # How many query heads share each KV head

    # Reshape: [num_heads * head_dim] → [num_kv_heads, num_groups, head_dim]
    scales = scales.view(num_kv_heads, num_groups, head_dim)

    # Average across groups (or could use max - AWQ typically uses mean)
    folded = scales.mean(dim=1)  # [num_kv_heads, head_dim]

    return folded.flatten()  # [num_kv_heads * head_dim]


def expand_gqa_scales(
    scales_kv: torch.Tensor,  # [num_kv_heads * head_dim]
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Expand KV-head scales to head-space scales used by o_proj input.

    For GQA, KV channels are repeated across num_groups = num_heads/num_kv_heads.
    To keep the transform equivalent, the scale applied on the expanded head
    channels MUST be constant across the repeated groups.
    """
    if num_kv_heads == num_heads:
        return scales_kv
    num_groups = num_heads // num_kv_heads
    s = scales_kv.view(num_kv_heads, 1, head_dim).repeat(1, num_groups, 1)
    return s.reshape(num_heads * head_dim)


def apply_awq_to_qwen3(
    model: nn.Module,
    imatrix: Dict[str, torch.Tensor],
    alpha: float = 0.5,
    verbose: bool = True,
    ymatrix: Optional[Dict[str, torch.Tensor]] = None,
    alpha_row: float = 0.0,
    row_min_scale: float = 0.5,
    row_max_scale: float = 2.0,
) -> Dict[str, any]:
    """Apply AWQ-equivalent transforms to a Qwen3 model.

    Args:
        model: Qwen3 model (from transformers)
        imatrix: Dict with 'sigma2' containing layer name → variance tensor
        alpha: AWQ scaling strength
        verbose: Print progress

    Returns:
        Stats dict with applied transforms
    """
    sigma2_dict = imatrix.get('sigma2', imatrix)

    # Get model config
    config = model.config
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = getattr(config, 'head_dim', config.hidden_size // num_heads)
    num_layers = config.num_hidden_layers

    if verbose:
        print(f"\nApplying AWQ transforms (alpha={alpha})")
        print(f"  num_heads={num_heads}, num_kv_heads={num_kv_heads}, head_dim={head_dim}")
        print(f"  num_layers={num_layers}")

    stats = {
        'alpha': alpha,
        'num_layers': num_layers,
        'transforms_applied': 0,
    }

    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        prefix = f"model.layers.{layer_idx}"

        # === Transform 1: input_layernorm → q_proj, k_proj, v_proj ===
        # Find iMatrix key for this layer's attention input
        attn_key = None
        for pattern in [
            f"{prefix}.self_attn.q_proj",
            f"{prefix}.self_attn.k_proj",
            f"{prefix}.self_attn.v_proj",
        ]:
            if pattern in sigma2_dict:
                attn_key = pattern
                break

        if attn_key and attn_key in sigma2_dict:
            sigma2 = sigma2_dict[attn_key]
            scales = compute_awq_scales(sigma2, alpha)

            scale_ln_fcs(
                ln=layer.input_layernorm,
                fcs=[
                    layer.self_attn.q_proj,
                    layer.self_attn.k_proj,
                    layer.self_attn.v_proj,
                ],
                scales=scales,
            )
            stats['transforms_applied'] += 1

            if verbose and layer_idx == 0:
                print(f"  Layer {layer_idx}: input_layernorm → qkv (scales range: {scales.min():.3f} - {scales.max():.3f})")

        # === Transform 2: v_proj → o_proj (with GQA handling) ===
        # This is tricky because of GQA:
        #   - v_proj output: num_kv_heads * head_dim (e.g., 8 * 128 = 1024)
        #   - o_proj input:  num_heads * head_dim (e.g., 16 * 128 = 2048)
        # We can't use scale_fc_fc directly because the dimensions don't match.
        # Instead, we handle them separately:
        #   - v_proj: apply folded scales to output rows
        #   - o_proj: apply full scales to input columns
        o_proj_key = f"{prefix}.self_attn.o_proj"
        if o_proj_key in sigma2_dict:
            sigma2 = sigma2_dict[o_proj_key]

            # o_proj input has num_heads * head_dim channels
            scales_full = compute_awq_scales(sigma2, alpha).to(layer.self_attn.v_proj.weight.device)

            if num_kv_heads < num_heads:
                # GQA: fold scales from o_proj dimension to v_proj dimension
                scales_vo = fold_gqa_scales(scales_full, num_heads, num_kv_heads, head_dim)
            else:
                # MHA: no folding needed
                scales_vo = scales_full

            # Apply to v_proj output (row-wise, using folded scales)
            # v_proj.weight shape: [num_kv_heads * head_dim, hidden_size]
            layer.self_attn.v_proj.weight.data /= scales_vo.view(-1, 1)
            if layer.self_attn.v_proj.bias is not None:
                layer.self_attn.v_proj.bias.data /= scales_vo

            # Apply to o_proj input (column-wise, using full scales)
            # o_proj.weight shape: [hidden_size, num_heads * head_dim]
            layer.self_attn.o_proj.weight.data *= scales_full.view(1, -1)

            stats['transforms_applied'] += 1

            if verbose and layer_idx == 0:
                print(f"  Layer {layer_idx}: v_proj → o_proj (GQA: {num_kv_heads}→{num_heads}, scales_vo={scales_vo.shape[0]}, scales_full={scales_full.shape[0]})")

        # === Transform 3: post_attention_layernorm → gate_proj, up_proj ===
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

            scale_ln_fcs(
                ln=layer.post_attention_layernorm,
                fcs=[
                    layer.mlp.gate_proj,
                    layer.mlp.up_proj,
                ],
                scales=scales,
            )
            stats['transforms_applied'] += 1

            if verbose and layer_idx == 0:
                print(f"  Layer {layer_idx}: post_attn_ln → gate+up (scales range: {scales.min():.3f} - {scales.max():.3f})")

        # === Transform 4: up_proj → down_proj ===
        down_key = f"{prefix}.mlp.down_proj"
        if down_key in sigma2_dict:
            sigma2 = sigma2_dict[down_key]
            scales = compute_awq_scales(sigma2, alpha)

            scale_fc_fc(
                prev_fc=layer.mlp.up_proj,
                next_fcs=[layer.mlp.down_proj],
                scales=scales,
            )
            stats['transforms_applied'] += 1

            if verbose and layer_idx == 0:
                print(f"  Layer {layer_idx}: up_proj → down_proj (scales range: {scales.min():.3f} - {scales.max():.3f})")

        # === Pass B (optional): Row-smoothing using Y-matrix (output second moments) ===
        # Apply ONLY on safe pairs with a unique linear consumer:
        #   - v_proj (rows, kv-space)  -> o_proj (cols, head-space expanded for GQA)
        #   - up_proj (rows) -> down_proj (cols)
        if ymatrix is not None and alpha_row > 0.0:
            sigma2_out_dict = ymatrix.get("sigma2_out", ymatrix)

            # B1: up_proj -> down_proj (use up_proj OUTPUT stats if available)
            up_key = f"{prefix}.mlp.up_proj"
            if up_key in sigma2_out_dict:
                sigma2_up = sigma2_out_dict[up_key].to(layer.mlp.up_proj.weight.device)
                s_up = compute_awq_scales(sigma2_up, alpha_row, min_scale=row_min_scale, max_scale=row_max_scale)
                # row-scale up_proj, col-scale down_proj
                scale_fc_fc(prev_fc=layer.mlp.up_proj, next_fcs=[layer.mlp.down_proj], scales=s_up)
                stats['transforms_applied'] += 1
                if verbose and layer_idx == 0:
                    print(f"  Layer {layer_idx}: up→down (PassB) scales[{s_up.min():.3f},{s_up.max():.3f}]")

            # B2: v_proj -> o_proj (use v_proj OUTPUT stats if available)
            v_key = f"{prefix}.self_attn.v_proj"
            if v_key in sigma2_out_dict:
                sigma2_v = sigma2_out_dict[v_key].to(layer.self_attn.v_proj.weight.device)
                s_kv = compute_awq_scales(sigma2_v, alpha_row, min_scale=row_min_scale, max_scale=row_max_scale)
                # v_proj rows (kv-space)
                layer.self_attn.v_proj.weight.data /= s_kv.view(-1, 1)
                if layer.self_attn.v_proj.bias is not None:
                    layer.self_attn.v_proj.bias.data /= s_kv
                # o_proj cols (head-space)
                s_full = expand_gqa_scales(s_kv, num_heads, num_kv_heads, head_dim).to(layer.self_attn.o_proj.weight.device)
                layer.self_attn.o_proj.weight.data *= s_full.view(1, -1)
                stats['transforms_applied'] += 1
                if verbose and layer_idx == 0:
                    print(f"  Layer {layer_idx}: v→o (PassB) kv[{s_kv.min():.3f},{s_kv.max():.3f}] head[{s_full.min():.3f},{s_full.max():.3f}]")


    if verbose:
        print(f"\nApplied {stats['transforms_applied']} transforms across {num_layers} layers")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Apply AWQ-equivalent scale transforms to Qwen3 model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--model-id', type=str, default='Qwen/Qwen3-0.6B',
                        help='HuggingFace model ID or local path')
    parser.add_argument('--imatrix', type=str, required=True,
                        help='Path to iMatrix .pt file (from compute_imatrix.py)')
    parser.add_argument('--ymatrix', type=str, default=None,
                        help='Optional path to Y-matrix .pt file (from compute_ymatrix.py) for Pass B row-smoothing')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Pass A (column) AWQ strength. 0.25–0.5 typical.')
    parser.add_argument('--alpha-row', type=float, default=0.0,
                        help='Pass B (row-smoothing) strength using Y-matrix (0 disables)')
    parser.add_argument('--row-min-scale', type=float, default=0.5,
                        help='Min clamp for Pass B row scales')
    parser.add_argument('--row-max-scale', type=float, default=2.0,
                        help='Max clamp for Pass B row scales')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output directory for AWQ-scaled model')
    parser.add_argument('--dtype', type=str, default='float32',
                        choices=['float32', 'float16', 'bfloat16'],
                        help='Model dtype (default: float32 for accuracy)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device for processing (cpu recommended for large models)')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Minimal output')

    args = parser.parse_args()

    # Resolve dtype
    dtype_map = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    verbose = not args.quiet

    if verbose:
        print("=" * 60)
        print("AWQ-Equivalent Scale Transform for Qwen3")
        print("=" * 60)
        print(f"Model:    {args.model_id}")
        print(f"iMatrix:  {args.imatrix}")
        print(f"Alpha:    {args.alpha}")
        print(f"Output:   {args.output}")
        print(f"Dtype:    {args.dtype}")

    # Load iMatrix
    if verbose:
        print(f"\nLoading iMatrix from {args.imatrix}...")
    imatrix = torch.load(args.imatrix, map_location='cpu', weights_only=False)
    ymatrix = None
    if args.ymatrix:
        ymatrix = torch.load(args.ymatrix, map_location='cpu')

    sigma2_dict = imatrix.get('sigma2', imatrix)
    if verbose:
        print(f"  Found {len(sigma2_dict)} layer entries")

    # Load model
    if verbose:
        print(f"\nLoading model {args.model_id}...")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.to(args.device)

    if verbose:
        print(f"  Loaded with dtype={dtype}, device={args.device}")

    # Apply AWQ transforms
    stats = apply_awq_to_qwen3(
        model=model,
        imatrix=imatrix,
        alpha=args.alpha,
        verbose=verbose,
        ymatrix=ymatrix,
        alpha_row=args.alpha_row,
        row_min_scale=args.row_min_scale,
        row_max_scale=args.row_max_scale,
    )

    # Save transformed model
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\nSaving AWQ-scaled model to {output_dir}...")

    model.save_pretrained(output_dir)

    # Also save tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
        tokenizer.save_pretrained(output_dir)
        if verbose:
            print("  Saved tokenizer")
    except Exception as e:
        if verbose:
            print(f"  Warning: Could not save tokenizer: {e}")

    # Save AWQ config
    awq_config = {
        'source_model': args.model_id,
        'imatrix_path': args.imatrix,
        'alpha': args.alpha,
        'transforms_applied': stats['transforms_applied'],
        'num_layers': stats['num_layers'],
    }
    with open(output_dir / 'awq_config.json', 'w') as f:
        json.dump(awq_config, f, indent=2)

    if verbose:
        print(f"\nSaved:")
        print(f"  {output_dir}/")
        print(f"    model.safetensors (or pytorch_model.bin)")
        print(f"    config.json")
        print(f"    awq_config.json")
        print(f"\nNext step: Use this as --model-id for init_model_v2.py")
        print(f"  python scripts/init_model_v2.py \\")
        print(f"      --model-id {output_dir} \\")
        print(f"      --output runs/v2_awq \\")
        print(f"      --config q4a4")

    print("\nDone!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
