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
    """Compute AWQ scaling factors from activation statistics (diag-iMatrix).

    Notes:
      - iMatrix stores sigma2 = E[x^2]. We convert to RMS(x) = sqrt(E[x^2]) first.
      - We normalize to mean=1 so this is a *pure reparameterization* (no global gain shift).
      - We clamp to avoid extreme per-channel scaling (stability / FP16 friendliness).

    Args:
        sigma2: [features] activation second moment from iMatrix (E[x_i^2])
        alpha: Scaling strength (0.5 = moderate, recommended)
        min_scale: Minimum scale factor (prevents near-zero)
        max_scale: Maximum scale factor (prevents explosion)

    Returns:
        s: [features] scaling factors, mean-normalized (≈1.0)
    """
    sigma2 = sigma2.float()
    # Convert to RMS-like magnitude; eps avoids sqrt(0)
    rms = torch.sqrt(sigma2 + 1e-8)

    # Normalize to mean=1
    s = rms / (rms.mean() + 1e-8)

    # Apply power
    s = s ** alpha

    # Clamp + re-normalize (keeps overall gain stable)
    s = s.clamp(min=min_scale, max=max_scale)
    s = s / (s.mean() + 1e-8)
    s = s.clamp(min=min_scale, max=max_scale)

    return s


def scale_ln_fcs(
    ln: nn.Module,  # RMSNorm or LayerNorm
    fcs: list,      # List of Linear layers following the norm
    scales: torch.Tensor,  # [in_features] scaling factors
) -> None:
    """Apply AWQ scale_ln_fcs: RMSNorm/LayerNorm → Linear(s).

    Equivalent transform:
      x_norm = ln(x)
      y = fc(x_norm)

      Apply scales s on channels of x_norm:
        x_norm' = x_norm / s
        fc' = fc with input columns scaled by s

      So: fc'(x_norm') = (W*s) @ (x_norm/s) = W @ x_norm

    IMPORTANT:
      - For LayerNorm with bias, bias must also be divided by s (otherwise not equivalent).
      - For RMSNorm (Qwen3), there is typically no bias; this is a safe no-op.
    """
    if not hasattr(ln, "weight") or ln.weight is None:
        raise ValueError("Norm module has no weight")

    scales = scales.to(ln.weight.device)
    if scales.numel() != ln.weight.numel():
        raise ValueError(f"scale_ln_fcs: scales len {scales.numel()} != ln.weight len {ln.weight.numel()}")

    # Divide norm affine params by s (so ln output is divided by s)
    ln.weight.data /= scales
    if hasattr(ln, "bias") and ln.bias is not None:
        ln.bias.data /= scales

    # Multiply following fc columns by s
    for fc in fcs:
        if fc.weight.shape[1] != scales.numel():
            raise ValueError(f"scale_ln_fcs: fc.in_features={fc.weight.shape[1]} != scales len {scales.numel()}")
        fc.weight.data *= scales.view(1, -1)

def scale_fc_fc(
    prev_fc: nn.Module,  # Previous Linear (output being scaled)
    next_fcs: list,      # Next Linear(s) (input being inverse-scaled)
    scales: torch.Tensor,  # [features] scaling factors
) -> None:
    """Apply AWQ scale_fc_fc: Linear → Linear(s).

    Equivalent transform between two consecutive linears:
      h = prev_fc(x)
      y = next_fc(h)

    Apply per-channel scales s on h:
      h' = h / s    (implemented by dividing prev_fc rows (+bias) by s)
      next_fc' = next_fc with input columns scaled by s

      next_fc'(h') = (W_next*s) @ (h/s) = W_next @ h

    Args:
      scales must have length == prev_fc.out_features == next_fc.in_features.
    """
    scales = scales.to(prev_fc.weight.device)

    if prev_fc.weight.shape[0] != scales.numel():
        raise ValueError(f"scale_fc_fc: prev_fc.out_features={prev_fc.weight.shape[0]} != scales len {scales.numel()}")

    # Previous fc: divide output channels by s
    prev_fc.weight.data /= scales.view(-1, 1)
    if prev_fc.bias is not None:
        prev_fc.bias.data /= scales

    # Next fc(s): multiply input columns by s
    for fc in next_fcs:
        if fc.weight.shape[1] != scales.numel():
            raise ValueError(f"scale_fc_fc: next_fc.in_features={fc.weight.shape[1]} != scales len {scales.numel()}")
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
    """Expand KV-space scales to full head-space by repeating.

    This is the inverse of fold_gqa_scales. It repeats each KV head's scale
    across all query heads that share it (GQA groups).

    CRITICAL: This ensures the AWQ transform is function-preserving under GQA.
    The scale on o_proj's input must be constant across repeated groups for
    each KV head, matching how repeat_kv works.

    Args:
        scales_kv: [num_kv_heads * head_dim] scales for KV heads
        num_heads: Number of query heads
        num_kv_heads: Number of KV heads
        head_dim: Dimension per head

    Returns:
        scales_full: [num_heads * head_dim] scales repeated to match o_proj input
    """
    num_groups = num_heads // num_kv_heads

    # Reshape: [num_kv_heads * head_dim] → [num_kv_heads, 1, head_dim]
    s = scales_kv.view(num_kv_heads, 1, head_dim)

    # Repeat across groups: [num_kv_heads, num_groups, head_dim]
    s = s.repeat(1, num_groups, 1)

    # Flatten back to [num_heads * head_dim]
    return s.reshape(num_heads * head_dim)


def apply_awq_to_qwen3(
    model: nn.Module,
    imatrix: Dict[str, torch.Tensor],
    alpha: float = 0.5,
    verbose: bool = True,
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
        #
        # CRITICAL for function-preserving transform:
        #   The scale on o_proj's input must be constant across the repeated groups
        #   for each KV head, matching how repeat_kv works.
        #
        # Correct approach:
        #   1. Fold sigma2 (not scales!) to KV-space first
        #   2. Compute s_kv from the folded sigma2
        #   3. Expand s_kv to s_full by repeating (not independent computation!)
        #
        # This ensures: repeat_kv(v / s_kv) = repeat_kv(v) / s_full
        # And then: o_proj @ (x * s_full) / s_full cancels exactly.
        o_proj_key = f"{prefix}.self_attn.o_proj"
        if o_proj_key in sigma2_dict:
            sigma2_full = sigma2_dict[o_proj_key]  # [num_heads * head_dim]
            device = layer.self_attn.v_proj.weight.device

            # Shape sanity checks
            if sigma2_full.numel() != num_heads * head_dim:
                raise ValueError(f"{o_proj_key}: sigma2 len {sigma2_full.numel()} != num_heads*head_dim {num_heads*head_dim}")
            if layer.self_attn.o_proj.weight.shape[1] != num_heads * head_dim:
                raise ValueError(f"{o_proj_key}: o_proj.in_features {layer.self_attn.o_proj.weight.shape[1]} != num_heads*head_dim {num_heads*head_dim}")

            if num_kv_heads < num_heads:
                # GQA: fold sigma2 to KV-space FIRST, then compute scales
                sigma2_kv = fold_gqa_scales(sigma2_full, num_heads, num_kv_heads, head_dim)
                s_kv = compute_awq_scales(sigma2_kv, alpha).to(device)
                # Expand s_kv back to head-space by repeating (NOT independent computation!)
                s_full = expand_gqa_scales(s_kv, num_heads, num_kv_heads, head_dim).to(device)
            else:
                # MHA: no folding needed
                s_full = compute_awq_scales(sigma2_full, alpha).to(device)
                s_kv = s_full

            # Apply to v_proj output (row-wise, using KV-space scales)
            # v_proj.weight shape: [num_kv_heads * head_dim, hidden_size]
            layer.self_attn.v_proj.weight.data /= s_kv.view(-1, 1)
            if layer.self_attn.v_proj.bias is not None:
                layer.self_attn.v_proj.bias.data /= s_kv

            # Apply to o_proj input (column-wise, using repeated scales)
            # o_proj.weight shape: [hidden_size, num_heads * head_dim]
            # s_full is s_kv repeated to match repeat_kv behavior
            layer.self_attn.o_proj.weight.data *= s_full.view(1, -1)

            stats['transforms_applied'] += 1

            if verbose and layer_idx == 0:
                print(f"  Layer {layer_idx}: v_proj → o_proj (GQA: fold σ²→s_kv({s_kv.shape[0]})→expand s_full({s_full.shape[0]}))")

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
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='AWQ scaling strength: 0.5=moderate (recommended), 1.0=strong')
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
