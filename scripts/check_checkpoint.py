#!/usr/bin/env python3
"""
Checkpoint health diagnostic tool.

Checks for:
1. Non-finite values (NaN/Inf) in parameters
2. Extreme magnitudes that could cause fp16 overflow
3. Scale/magnitude issues that cause long-context NaN

Usage:
    python scripts/check_checkpoint.py checkpoint.pt
    python scripts/check_checkpoint.py checkpoint.pt --verbose
    python scripts/check_checkpoint.py checkpoint.pt --top 50
"""

import argparse
import sys
from pathlib import Path

import torch


def check_checkpoint(checkpoint_path: str, top_n: int = 20, verbose: bool = False):
    """Check checkpoint for non-finite values and extreme magnitudes."""

    print(f"\n{'='*60}")
    print("CHECKPOINT HEALTH CHECK")
    print(f"{'='*60}")
    print(f"Checkpoint: {checkpoint_path}")

    # Load checkpoint
    sd = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Handle nested state dict
    if 'model_state_dict' in sd:
        sd = sd['model_state_dict']

    # Categories of interest
    suspects = ["rank_magnitude", "scale_A", "scale_B", "_Q", "lut", "_indices", "magnitude"]

    # Track issues
    non_finite = []
    big_values = []
    fp16_overflow_risk = []
    stats = {
        'total_params': 0,
        'total_tensors': 0,
        'finite_tensors': 0,
        'non_finite_tensors': 0,
    }

    print(f"\nScanning {len(sd)} keys...")

    for key, value in sd.items():
        if not torch.is_tensor(value) or value.numel() == 0:
            continue

        stats['total_tensors'] += 1
        stats['total_params'] += value.numel()

        # Check for non-finite values (convert to float for int tensors)
        value_float = value.float()
        finite_mask = torch.isfinite(value_float)
        if not finite_mask.all():
            non_finite.append({
                'key': key,
                'dtype': value.dtype,
                'shape': tuple(value.shape),
                'nan_count': (~finite_mask).sum().item(),
                'total': value.numel(),
            })
            stats['non_finite_tensors'] += 1
        else:
            stats['finite_tensors'] += 1

        # Track magnitudes for suspect tensors
        is_suspect = any(s in key for s in suspects)
        if is_suspect or verbose:
            max_val = value_float.abs().max().item()
            min_val = value_float.abs().min().item()
            mean_val = value_float.abs().mean().item()

            big_values.append({
                'key': key,
                'max': max_val,
                'min': min_val,
                'mean': mean_val,
                'dtype': value.dtype,
                'shape': tuple(value.shape),
                'is_suspect': is_suspect,
            })

            # Check fp16 overflow risk (fp16 max ~65504)
            if max_val > 65000:
                fp16_overflow_risk.append({
                    'key': key,
                    'max': max_val,
                    'dtype': value.dtype,
                })

    # Report: Non-finite tensors
    print(f"\n{'-'*60}")
    print("NON-FINITE VALUES")
    print(f"{'-'*60}")

    if non_finite:
        print(f"\n❌ Found {len(non_finite)} tensor(s) with NaN/Inf:")
        for item in non_finite[:20]:
            pct = 100 * item['nan_count'] / item['total']
            print(f"  {item['key']}")
            print(f"    dtype={item['dtype']}, shape={item['shape']}")
            print(f"    non-finite: {item['nan_count']:,} / {item['total']:,} ({pct:.1f}%)")
        if len(non_finite) > 20:
            print(f"  ... and {len(non_finite) - 20} more")
    else:
        print("\n✅ All tensors are finite (no NaN/Inf)")

    # Report: Extreme magnitudes
    print(f"\n{'-'*60}")
    print(f"TOP {top_n} MAGNITUDES (suspect tensors)")
    print(f"{'-'*60}")

    # Sort by max value
    big_values.sort(key=lambda x: x['max'], reverse=True)

    print(f"\n{'Max':>14}  {'Mean':>12}  {'Dtype':>14}  Key")
    print(f"{'-'*14}  {'-'*12}  {'-'*14}  {'-'*40}")

    for item in big_values[:top_n]:
        flag = "⚠️ " if item['max'] > 65000 else "   "
        suspect_mark = "*" if item['is_suspect'] else " "
        dtype_str = str(item['dtype']).replace('torch.', '')
        print(f"{item['max']:14.6g}  {item['mean']:12.6g}  {dtype_str:>14}  {flag}{suspect_mark}{item['key']}")

    # Report: FP16 overflow risk
    print(f"\n{'-'*60}")
    print("FP16 OVERFLOW RISK (max > 65000)")
    print(f"{'-'*60}")

    if fp16_overflow_risk:
        print(f"\n⚠️  Found {len(fp16_overflow_risk)} tensor(s) at risk of fp16 overflow:")
        for item in fp16_overflow_risk[:20]:
            print(f"  {item['key']}: max={item['max']:.6g}")
        if len(fp16_overflow_risk) > 20:
            print(f"  ... and {len(fp16_overflow_risk) - 20} more")
        print("\n  These tensors may cause NaN when cast to fp16 or during")
        print("  fp16 matmul. Consider clamping scales/magnitudes during training.")
    else:
        print("\n✅ No tensors exceed fp16 safe range")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total tensors:      {stats['total_tensors']:,}")
    print(f"Total parameters:   {stats['total_params']:,}")
    print(f"Finite tensors:     {stats['finite_tensors']:,}")
    print(f"Non-finite tensors: {stats['non_finite_tensors']:,}")

    # Diagnosis
    print(f"\n{'-'*60}")
    print("DIAGNOSIS")
    print(f"{'-'*60}")

    if non_finite:
        print("\n❌ CRITICAL: Checkpoint contains NaN/Inf values!")
        print("   Root cause: Training diverged. Check:")
        print("   - Learning rate too high")
        print("   - Missing gradient clipping")
        print("   - Unconstrained scale/magnitude growth")
        print("\n   Fix: Add guardrails during training:")
        print("   - Clamp rank_magnitude to [-10, 10]")
        print("   - Clamp scale_A/scale_B")
        print("   - Add NaN detection + checkpoint rollback")
        return 1

    if fp16_overflow_risk:
        print("\n⚠️  WARNING: Some values exceed fp16 safe range!")
        print("   This will cause NaN during fp16 inference.")
        print("\n   Likely cause: Scales or magnitudes grew too large during training.")
        print("   This is common when training at short context (128) but")
        print("   evaluating at long context (1024).")
        print("\n   Fix options:")
        print("   1. Use bf16 or fp32 for evaluation (workaround)")
        print("   2. Clamp magnitudes during training (proper fix)")
        print("   3. Freeze magnitudes after initialization (--freeze-mags)")
        print("   4. Train with occasional long-context probes")
        return 2

    # Check for suspiciously large but not overflow values
    max_magnitude = big_values[0]['max'] if big_values else 0
    if max_magnitude > 1000:
        print(f"\n⚠️  CAUTION: Largest magnitude is {max_magnitude:.2f}")
        print("   While not overflowing, large magnitudes can cause")
        print("   numerical instability at long context lengths.")
        print("\n   Consider constraining scales/magnitudes during training.")
        return 0

    print("\n✅ Checkpoint looks healthy!")
    print("   If you still see NaN during evaluation, check:")
    print("   - Long context overflow (try --max-length 128)")
    print("   - dtype issues (try --dtype bf16 or fp32)")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Check checkpoint for NaN/Inf and extreme values',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('checkpoint', help='Path to checkpoint file (.pt)')
    parser.add_argument('--top', type=int, default=20,
                        help='Show top N magnitudes (default: 20)')
    parser.add_argument('--verbose', action='store_true',
                        help='Show all tensors, not just suspects')

    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return 1

    return check_checkpoint(args.checkpoint, args.top, args.verbose)


if __name__ == '__main__':
    sys.exit(main())
