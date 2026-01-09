#!/usr/bin/env python3
"""
Check if rank_magnitude values are FP16-representable.

Loads a checkpoint and compares rank_magnitude to CPU-snapped FP16 values.
If differences are zero, the checkpoint was properly snapped.

Usage:
    python scripts/check_mags_fp16.py checkpoint.pt
    python scripts/check_mags_fp16.py checkpoint.pt --top 20
"""

import argparse
import torch


def main():
    parser = argparse.ArgumentParser(description='Check rank_magnitude FP16 precision')
    parser.add_argument('checkpoint', type=str, help='Checkpoint file (.pt)')
    parser.add_argument('--top', type=int, default=10, help='Show top N differences (default: 10)')
    args = parser.parse_args()

    print("=" * 60)
    print("CHECK RANK_MAGNITUDE FP16 PRECISION")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")

    # Load checkpoint on CPU
    print("\nLoading checkpoint on CPU...")
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)

    # Handle nested state_dict
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
        print("  Found nested 'model_state_dict'")
    else:
        state_dict = ckpt

    # Find rank_magnitude keys
    mag_keys = [k for k in state_dict.keys() if k.endswith('.rank_magnitude')]
    print(f"  Found {len(mag_keys)} rank_magnitude tensors")

    if not mag_keys:
        print("\nNo rank_magnitude tensors found. Is this a V2 checkpoint?")
        return

    # Compute differences
    diffs = []
    total_snapped = 0
    total_unsnapped = 0

    for key in mag_keys:
        val = state_dict[key].float()  # Ensure FP32 for comparison
        snapped = val.cpu().half().float()  # CPU FP16 snap

        diff = (val - snapped).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        # Check if already snapped (diff should be exactly 0)
        is_snapped = max_diff == 0.0

        if is_snapped:
            total_snapped += 1
        else:
            total_unsnapped += 1
            diffs.append((max_diff, mean_diff, key, val.min().item(), val.max().item()))

    # Sort by max diff
    diffs.sort(reverse=True, key=lambda x: x[0])

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Already FP16-snapped: {total_snapped}/{len(mag_keys)}")
    print(f"NOT snapped:          {total_unsnapped}/{len(mag_keys)}")

    if total_unsnapped == 0:
        print("\n✓ All rank_magnitude tensors are FP16-representable!")
        print("  This checkpoint was properly snapped with --freeze-mags.")
    else:
        print(f"\n✗ {total_unsnapped} tensors have FP16 rounding differences")
        print("  Consider using --freeze-mags during training.")

        # Show top differences
        print(f"\n" + "=" * 60)
        print(f"TOP {min(args.top, len(diffs))} LARGEST DIFFERENCES")
        print("=" * 60)

        for i, (max_diff, mean_diff, key, vmin, vmax) in enumerate(diffs[:args.top]):
            # Extract layer info
            layer_name = key.replace('.rank_magnitude', '')
            is_mlp = any(p in key for p in ['gate_proj', 'up_proj', 'down_proj'])
            layer_type = "MLP" if is_mlp else "Attn"

            print(f"{i+1:2d}. {max_diff:.6f} | {layer_type:4s} | range=[{vmin:.1f}, {vmax:.1f}] | {layer_name}")

        # Stats by layer type
        mlp_diffs = [d for d in diffs if any(p in d[2] for p in ['gate_proj', 'up_proj', 'down_proj'])]
        attn_diffs = [d for d in diffs if d not in mlp_diffs]

        print(f"\n" + "=" * 60)
        print("BY LAYER TYPE")
        print("=" * 60)
        if mlp_diffs:
            avg_mlp = sum(d[0] for d in mlp_diffs) / len(mlp_diffs)
            max_mlp = max(d[0] for d in mlp_diffs)
            print(f"MLP layers:  {len(mlp_diffs)} unsnapped, max_diff={max_mlp:.6f}, avg={avg_mlp:.6f}")
        else:
            print(f"MLP layers:  All snapped ✓")

        if attn_diffs:
            avg_attn = sum(d[0] for d in attn_diffs) / len(attn_diffs)
            max_attn = max(d[0] for d in attn_diffs)
            print(f"Attn layers: {len(attn_diffs)} unsnapped, max_diff={max_attn:.6f}, avg={avg_attn:.6f}")
        else:
            print(f"Attn layers: All snapped ✓")

        # Show top magnitudes by max value (largest values lose most precision)
        sorted_by_max = sorted(diffs, key=lambda x: x[4], reverse=True)  # Sort by vmax
        print(f"\n" + "=" * 60)
        print(f"TOP {min(args.top, len(sorted_by_max))} LARGEST MAGNITUDES (most affected by FP16)")
        print("=" * 60)
        for i, (max_diff, mean_diff, key, vmin, vmax) in enumerate(sorted_by_max[:args.top]):
            layer_name = key.replace('.rank_magnitude', '')
            is_mlp = any(p in key for p in ['gate_proj', 'up_proj', 'down_proj'])
            layer_type = "MLP" if is_mlp else "Attn"
            print(f"{i+1:2d}. max={vmax:6.1f} | diff={max_diff:.6f} | {layer_type:4s} | {layer_name}")

    # Always show top magnitude values (regardless of snapping)
    all_mags = []
    for key in mag_keys:
        val = state_dict[key].float()
        vmin, vmax = val.min().item(), val.max().item()
        snapped = val.cpu().half().float()
        max_diff = (val - snapped).abs().max().item()
        all_mags.append((vmax, max_diff, key, vmin))

    all_mags.sort(reverse=True, key=lambda x: x[0])  # Sort by max value

    print(f"\n" + "=" * 60)
    print(f"TOP {min(args.top, len(all_mags))} RANK_MAGNITUDE VALUES")
    print("=" * 60)
    for i, (vmax, max_diff, key, vmin) in enumerate(all_mags[:args.top]):
        layer_name = key.replace('.rank_magnitude', '')
        is_mlp = any(p in key for p in ['gate_proj', 'up_proj', 'down_proj'])
        layer_type = "MLP" if is_mlp else "Attn"
        snap_status = "✓" if max_diff == 0 else f"Δ{max_diff:.4f}"
        print(f"{i+1:2d}. max={vmax:6.1f} | {snap_status:8s} | {layer_type:4s} | {layer_name}")

    # Sample values
    print(f"\n" + "=" * 60)
    print("SAMPLE VALUES")
    print("=" * 60)
    sample_key = mag_keys[0]
    sample = state_dict[sample_key]
    print(f"Key: {sample_key}")
    print(f"  dtype: {sample.dtype}")
    print(f"  shape: {sample.shape}")
    print(f"  range: [{sample.min():.4f}, {sample.max():.4f}]")
    print(f"  first 5: {sample.flatten()[:5].tolist()}")


if __name__ == '__main__':
    main()
