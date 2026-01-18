#!/usr/bin/env python3
"""
Compare weights between two checkpoints.

Usage:
  python scripts/compare_checkpoints.py ckpt1.pt ckpt2.pt
  python scripts/compare_checkpoints.py ckpt1.pt ckpt2.pt --ignore-lora
  python scripts/compare_checkpoints.py ckpt1.pt ckpt2.pt --verbose
  python scripts/compare_checkpoints.py ckpt1.pt ckpt2.pt --filter "mlp"
"""

import argparse
import sys
from pathlib import Path

import torch


def load_state_dict(path: str) -> dict:
    """Load state dict, unwrapping if needed."""
    path = Path(path)

    # Handle directory (look for checkpoint file)
    if path.is_dir():
        for name in ['model_state_dict.pt', 'checkpoint_fp32.pt', 'recovery_best.pt']:
            candidate = path / name
            if candidate.exists():
                path = candidate
                break
        else:
            # Find any .pt file
            pt_files = list(path.glob('*.pt'))
            if pt_files:
                path = pt_files[0]
            else:
                raise FileNotFoundError(f"No .pt file found in {path}")

    print(f"Loading: {path}")
    state_dict = torch.load(path, map_location='cpu', weights_only=False)

    # Unwrap if needed
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']

    return state_dict, path


def filter_keys(keys: set, ignore_lora: bool = False, filter_pattern: str = None) -> set:
    """Filter keys based on options."""
    result = keys

    if ignore_lora:
        result = {k for k in result if 'lora_' not in k}

    if filter_pattern:
        result = {k for k in result if filter_pattern in k}

    return result


def compare_tensors(t1: torch.Tensor, t2: torch.Tensor) -> dict:
    """Compare two tensors and return statistics."""
    if t1.shape != t2.shape:
        return {
            'match': False,
            'reason': f'shape mismatch: {t1.shape} vs {t2.shape}',
        }

    if t1.dtype != t2.dtype:
        # Cast to float for comparison
        t1 = t1.float()
        t2 = t2.float()

    # Exact match
    if torch.equal(t1, t2):
        return {'match': True, 'exact': True}

    # Compute differences
    diff = (t1.float() - t2.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    # Relative difference (avoid div by zero)
    t1_abs = t1.float().abs()
    rel_diff = diff / (t1_abs + 1e-8)
    max_rel_diff = rel_diff.max().item()

    return {
        'match': False,
        'exact': False,
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'max_rel_diff': max_rel_diff,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Compare weights between two checkpoints',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('ckpt1', help='First checkpoint path')
    parser.add_argument('ckpt2', help='Second checkpoint path')
    parser.add_argument('--ignore-lora', action='store_true',
                        help='Ignore LoRA weights in comparison')
    parser.add_argument('--filter', type=str, default=None,
                        help='Only compare keys containing this pattern')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed per-key differences')
    parser.add_argument('--threshold', type=float, default=1e-6,
                        help='Threshold for considering tensors different (default: 1e-6)')
    parser.add_argument('--top-k', type=int, default=10,
                        help='Show top K largest differences (default: 10)')

    args = parser.parse_args()

    # Load checkpoints
    print("=" * 60)
    print("CHECKPOINT COMPARISON")
    print("=" * 60)

    sd1, path1 = load_state_dict(args.ckpt1)
    sd2, path2 = load_state_dict(args.ckpt2)

    print(f"\nCheckpoint 1: {path1}")
    print(f"  Keys: {len(sd1)}")
    print(f"Checkpoint 2: {path2}")
    print(f"  Keys: {len(sd2)}")

    # Get key sets
    keys1 = set(sd1.keys())
    keys2 = set(sd2.keys())

    # Filter keys
    keys1_filtered = filter_keys(keys1, args.ignore_lora, args.filter)
    keys2_filtered = filter_keys(keys2, args.ignore_lora, args.filter)

    if args.ignore_lora:
        lora_keys1 = len([k for k in keys1 if 'lora_' in k])
        lora_keys2 = len([k for k in keys2 if 'lora_' in k])
        print(f"\n[--ignore-lora] Ignoring {lora_keys1}/{lora_keys2} LoRA keys")

    if args.filter:
        print(f"\n[--filter '{args.filter}'] Comparing {len(keys1_filtered)}/{len(keys2_filtered)} matching keys")

    # Find key differences
    only_in_1 = keys1_filtered - keys2_filtered
    only_in_2 = keys2_filtered - keys1_filtered
    common_keys = keys1_filtered & keys2_filtered

    print(f"\nKey analysis:")
    print(f"  Only in ckpt1: {len(only_in_1)}")
    print(f"  Only in ckpt2: {len(only_in_2)}")
    print(f"  Common:        {len(common_keys)}")

    # LUT-specific analysis
    lut_keys1 = {k for k in keys1 if 'lut' in k.lower()}
    lut_keys2 = {k for k in keys2 if 'lut' in k.lower()}
    common_lut = lut_keys1 & lut_keys2
    lut_only_1 = lut_keys1 - lut_keys2
    lut_only_2 = lut_keys2 - lut_keys1

    print(f"\nLUT analysis:")
    print(f"  LUTs in ckpt1: {len(lut_keys1)}")
    print(f"  LUTs in ckpt2: {len(lut_keys2)}")
    print(f"  Common LUTs:   {len(common_lut)}")
    if lut_only_1:
        print(f"  Only in ckpt1: {len(lut_only_1)}")
        for k in sorted(lut_only_1)[:3]:
            print(f"    {k}")
        if len(lut_only_1) > 3:
            print(f"    ... and {len(lut_only_1) - 3} more")
    if lut_only_2:
        print(f"  Only in ckpt2: {len(lut_only_2)}")
        for k in sorted(lut_only_2)[:3]:
            print(f"    {k}")
        if len(lut_only_2) > 3:
            print(f"    ... and {len(lut_only_2) - 3} more")

    if only_in_1 and args.verbose:
        print(f"\n  Keys only in ckpt1:")
        for k in sorted(only_in_1)[:5]:
            print(f"    {k}")
        if len(only_in_1) > 5:
            print(f"    ... and {len(only_in_1) - 5} more")

    if only_in_2 and args.verbose:
        print(f"\n  Keys only in ckpt2:")
        for k in sorted(only_in_2)[:5]:
            print(f"    {k}")
        if len(only_in_2) > 5:
            print(f"    ... and {len(only_in_2) - 5} more")

    # Compare common keys
    print(f"\nComparing {len(common_keys)} common keys...")

    exact_match = []
    close_match = []  # Below threshold
    different = []  # Above threshold
    shape_mismatch = []

    differences = []  # (key, max_diff, mean_diff)

    for key in sorted(common_keys):
        t1 = sd1[key]
        t2 = sd2[key]

        result = compare_tensors(t1, t2)

        if result.get('reason'):
            shape_mismatch.append((key, result['reason']))
        elif result.get('exact'):
            exact_match.append(key)
        elif result['max_diff'] < args.threshold:
            close_match.append(key)
        else:
            different.append(key)
            differences.append((key, result['max_diff'], result['mean_diff']))

    # Summary
    print(f"\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Exact match:     {len(exact_match):4d} keys")
    print(f"Close match:     {len(close_match):4d} keys (max_diff < {args.threshold})")
    print(f"Different:       {len(different):4d} keys")
    print(f"Shape mismatch:  {len(shape_mismatch):4d} keys")

    # LUT-specific comparison for common LUTs
    if common_lut:
        lut_exact = [k for k in exact_match if 'lut' in k.lower()]
        lut_diff = [(k, d, m) for k, d, m in differences if 'lut' in k.lower()]
        print(f"\nLUT comparison ({len(common_lut)} common):")
        print(f"  Exact match:   {len(lut_exact)}")
        print(f"  Different:     {len(lut_diff)}")
        if lut_diff:
            print(f"  Top LUT differences:")
            lut_diff_sorted = sorted(lut_diff, key=lambda x: x[1], reverse=True)
            for k, max_d, mean_d in lut_diff_sorted[:5]:
                print(f"    {max_d:.6e}  {k}")

            # Print actual values for LUT with highest difference
            if lut_diff_sorted:
                worst_lut_key, worst_max_diff, worst_mean_diff = lut_diff_sorted[0]
                print(f"\n  Detailed values for LUT with highest difference:")
                print(f"  Key: {worst_lut_key}")
                print(f"  Max diff: {worst_max_diff:.6e}, Mean diff: {worst_mean_diff:.6e}")

                lut1 = sd1[worst_lut_key]
                lut2 = sd2[worst_lut_key]

                print(f"\n  Checkpoint 1 ({path1.name}):")
                print(f"    Shape: {lut1.shape}, dtype: {lut1.dtype}")
                print(f"    Values: {lut1}")

                print(f"\n  Checkpoint 2 ({path2.name}):")
                print(f"    Shape: {lut2.shape}, dtype: {lut2.dtype}")
                print(f"    Values: {lut2}")

                print(f"\n  Absolute difference:")
                diff_tensor = (lut1.float() - lut2.float()).abs()
                print(f"    {diff_tensor}")

    # Overall verdict
    if len(different) == 0 and len(shape_mismatch) == 0:
        GREEN = "\033[92m"
        RESET = "\033[0m"
        print(f"\n{GREEN}✓ Checkpoints are IDENTICAL (within threshold){RESET}")
    else:
        RED = "\033[91m"
        RESET = "\033[0m"
        print(f"\n{RED}✗ Checkpoints are DIFFERENT{RESET}")

    # Show top differences
    if differences:
        print(f"\nTop {min(args.top_k, len(differences))} largest differences:")
        differences.sort(key=lambda x: x[1], reverse=True)
        for key, max_diff, mean_diff in differences[:args.top_k]:
            print(f"  {max_diff:.6e} (mean={mean_diff:.6e})  {key}")

    # Show shape mismatches
    if shape_mismatch:
        print(f"\nShape mismatches:")
        for key, reason in shape_mismatch[:5]:
            print(f"  {key}: {reason}")
        if len(shape_mismatch) > 5:
            print(f"  ... and {len(shape_mismatch) - 5} more")

    # Verbose: show all different keys
    if args.verbose and different:
        print(f"\nAll different keys ({len(different)}):")
        for key, max_diff, mean_diff in differences:
            print(f"  {key}: max={max_diff:.6e}, mean={mean_diff:.6e}")

    return 0 if (len(different) == 0 and len(shape_mismatch) == 0) else 1


if __name__ == '__main__':
    sys.exit(main())
