#!/usr/bin/env python3
"""
Snap rank_magnitude tensors to FP16-representable values (stored as FP32).

This pre-quantizes magnitudes so training sees the same values as FP16 inference,
without actually converting to FP16 dtype.

Usage:
    python scripts/snap_mags_fp16.py checkpoint.pt --output checkpoint_magssnapped.pt
    python scripts/snap_mags_fp16.py checkpoint.pt --mlp-only  # Only snap MLP layers
"""

import argparse
import torch
from pathlib import Path


def snap_to_fp16(x: torch.Tensor) -> torch.Tensor:
    """Round tensor to FP16-representable values, keep as FP32."""
    return x.half().float()


def load_state_dict(path):
    """Load checkpoint, handle nested state_dict."""
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        return ckpt['model_state_dict'], True
    return ckpt, False


def main():
    parser = argparse.ArgumentParser(description='Snap rank_magnitude to FP16-representable values')
    parser.add_argument('checkpoint', type=str, help='Input checkpoint')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path (default: input with _magssnapped suffix)')
    parser.add_argument('--mlp-only', action='store_true',
                        help='Only snap MLP layers (gate_proj, up_proj, down_proj)')
    args = parser.parse_args()

    if args.output is None:
        p = Path(args.checkpoint)
        args.output = str(p.parent / f"{p.stem}_magssnapped{p.suffix}")

    print("=" * 60)
    print("SNAP RANK_MAGNITUDE TO FP16 VALUES")
    print("=" * 60)
    print(f"Input:    {args.checkpoint}")
    print(f"Output:   {args.output}")
    print(f"MLP only: {args.mlp_only}")

    # Load
    print("\nLoading checkpoint...")
    state_dict, was_nested = load_state_dict(args.checkpoint)

    # Find and snap rank_magnitude keys
    mlp_patterns = ['gate_proj', 'up_proj', 'down_proj']

    snapped = 0
    skipped = 0
    max_diff = 0.0
    total_diff = 0.0

    for key in list(state_dict.keys()):
        if not key.endswith('.rank_magnitude'):
            continue

        # Check if MLP layer
        is_mlp = any(p in key for p in mlp_patterns)

        if args.mlp_only and not is_mlp:
            skipped += 1
            continue

        val = state_dict[key]
        snapped_val = snap_to_fp16(val)

        # Compute difference
        diff = (val - snapped_val).abs()
        max_diff = max(max_diff, diff.max().item())
        total_diff += diff.mean().item()

        # Replace
        state_dict[key] = snapped_val
        snapped += 1

    print(f"\nSnapped {snapped} rank_magnitude tensors")
    if skipped > 0:
        print(f"Skipped {skipped} attention layers (--mlp-only)")
    print(f"Max difference:  {max_diff:.6f}")
    print(f"Mean difference: {total_diff / max(snapped, 1):.6f}")

    # Sample
    rm_keys = [k for k in state_dict.keys() if k.endswith('.rank_magnitude')]
    if rm_keys:
        sample = state_dict[rm_keys[0]]
        print(f"\nSample ({rm_keys[0]}):")
        print(f"  dtype: {sample.dtype}")
        print(f"  range: [{sample.min():.4f}, {sample.max():.4f}]")
        print(f"  FP16 exact: {sample.half().float().equal(sample)}")

    # Save
    print(f"\nSaving to {args.output}...")
    torch.save(state_dict, args.output)
    print("Done!")

    print("\n" + "=" * 60)
    print("NEXT: Use this checkpoint for training with --freeze-mags")
    print("=" * 60)


if __name__ == '__main__':
    main()
