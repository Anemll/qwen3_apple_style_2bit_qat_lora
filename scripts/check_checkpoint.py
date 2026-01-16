#!/usr/bin/env python3
"""
Checkpoint health diagnostic tool.

Checks for:
1. Non-finite values (NaN/Inf) in parameters
2. Extreme magnitudes that could cause fp16 overflow
3. Scale/magnitude issues that cause long-context NaN
4. LUT training stats (_lut_raw_deltas): count, rebuild, FP16-validate, diff vs stored

Usage:
    python scripts/check_checkpoint.py checkpoint.pt
    python scripts/check_checkpoint.py checkpoint.pt --verbose
    python scripts/check_checkpoint.py checkpoint.pt --top 50
"""

import argparse
import math
import sys
from pathlib import Path

import torch


# =============================================================================
# LUT Training Analysis Helpers
# =============================================================================

def _lut_prefix_from_raw_key(k: str) -> str:
    """Extract layer prefix from _lut_raw_deltas key."""
    return k[: -len("._lut_raw_deltas")]


def _compute_min_delta_fp16(max_abs: float) -> float:
    """Conservative FP16-safe minimum delta to avoid 0/duplicates."""
    max_abs = float(max(max_abs, 1e-6))
    exp = math.ceil(math.log2(max_abs))
    ulp_at_max = 2 ** (exp - 10)  # FP16 mantissa ~10 bits
    return max(2.0 * ulp_at_max, 1e-4)


@torch.no_grad()
def _build_lut16_from_raw_deltas(raw: torch.Tensor, max_abs: float) -> torch.Tensor:
    """Rebuild LUT16 from _lut_raw_deltas (same logic as ane_qat_linear_v2.py)."""
    raw = raw.float().cpu()
    max_abs = float(max_abs)
    min_delta = _compute_min_delta_fp16(max_abs)
    remaining = max_abs - 8.0 * min_delta
    w = torch.softmax(raw, dim=0)

    if remaining <= 0:
        deltas = torch.full((8,), max_abs / 8.0, dtype=torch.float32)
    else:
        deltas = min_delta + remaining * w

    positive = torch.cumsum(deltas, dim=0)
    negative = -positive.flip(0)
    return torch.cat([negative, positive], dim=0)


@torch.no_grad()
def _lut16_fp16_qc(lut: torch.Tensor) -> dict:
    """Quality check: validate LUT is FP16-safe (unique, strictly increasing, no zero)."""
    lut_fp16 = lut.to(torch.float16)
    return {
        "unique_fp16": int(torch.unique(lut_fp16).numel()),
        "has_zero": bool((lut_fp16 == 0).any().item()),
        "strict_inc": bool((lut_fp16[1:] > lut_fp16[:-1]).all().item()),
        "min_abs": float(lut_fp16.abs().min().item()),
        "max_abs": float(lut_fp16.abs().max().item()),
    }


@torch.no_grad()
def check_q_indices_consistency(sd: dict, top_n: int = 5) -> dict:
    """Check if _Q == lut[_indices] for all layers that have both.

    This is a critical QC check. If _Q and lut[_indices] differ, utilities
    that read _Q directly will see stale quantization values.

    Returns:
        Dict with 'consistent', 'mismatches' (list), 'max_diff', 'checked'
    """
    # Find all layers that have both _Q and _indices
    q_keys = [k for k in sd.keys() if k.endswith('._Q')]
    mismatches = []
    checked = 0

    for q_key in sorted(q_keys):
        # Derive related keys
        base = q_key[:-3]  # Remove '._Q'
        indices_key = f"{base}._indices"
        lut_key = f"{base}.lut"

        if indices_key not in sd or lut_key not in sd:
            continue

        checked += 1
        Q = sd[q_key].float()
        indices = sd[indices_key].long()
        lut = sd[lut_key].float()

        # Compute what _Q should be
        Q_expected = lut[indices]

        # Check difference
        diff = (Q - Q_expected).abs().max().item()
        if diff > 1e-3:  # FP16-ish tolerance
            mismatches.append({
                'layer': base,
                'max_diff': diff,
                'mean_diff': (Q - Q_expected).abs().mean().item(),
            })

    result = {
        'checked': checked,
        'consistent': len(mismatches) == 0,
        'mismatches': mismatches,
        'max_diff': max((m['max_diff'] for m in mismatches), default=0),
    }

    if checked == 0:
        return result

    print()
    print("=" * 60)
    print("Q-INDICES CONSISTENCY CHECK")
    print("=" * 60)
    print(f"Layers with _Q + _indices + lut: {checked}")

    if mismatches:
        print(f"\n⚠️  {len(mismatches)} layers have _Q != lut[_indices]:")
        print(f"   Max difference: {result['max_diff']:.6f}")
        print()
        for m in sorted(mismatches, key=lambda x: -x['max_diff'])[:top_n]:
            print(f"  {m['layer']}: max_diff={m['max_diff']:.6f}, mean_diff={m['mean_diff']:.6f}")
        if len(mismatches) > top_n:
            print(f"  ... and {len(mismatches) - top_n} more")
        print()
        print("   This means utilities reading _Q will see STALE values!")
        print("   Fix: Run sync_q_from_indices_all() or re-bake the checkpoint.")
    else:
        print(f"\n✅ All layers have _Q == lut[_indices] (within tolerance)")

    return result


@torch.no_grad()
def summarize_lut_raw_deltas(sd: dict, top_n: int = 5) -> None:
    """Summarize _lut_raw_deltas tensors: count, rebuild, validate, diff vs stored."""
    raw_keys = [k for k in sd.keys() if k.endswith("._lut_raw_deltas")]
    if not raw_keys:
        return

    print()
    print("=" * 60)
    print("LUT TRAINING STATS (_lut_raw_deltas)")
    print("=" * 60)
    print(f"Found _lut_raw_deltas tensors: {len(raw_keys)}")

    rows = []
    diffs = []

    for k in sorted(raw_keys):
        prefix = _lut_prefix_from_raw_key(k)
        raw = sd[k]
        lut_key = f"{prefix}.lut"
        stored_lut = sd.get(lut_key, None)

        # Infer max_abs from stored lut if present, else default 1.0
        if stored_lut is not None and stored_lut.numel() == 16:
            max_abs = float(stored_lut.abs().max().item())
        else:
            max_abs = 1.0

        rebuilt = _build_lut16_from_raw_deltas(raw, max_abs=max_abs)
        qc = _lut16_fp16_qc(rebuilt)

        d = None
        if stored_lut is not None and stored_lut.numel() == 16:
            d = float((stored_lut.float().cpu() - rebuilt).abs().max().item())
            diffs.append((d, prefix, qc))

        rows.append((prefix, max_abs, float(raw.abs().max().item()), qc))

    # Count FP16-valid LUTs
    ok = [r for r in rows if (r[3]["unique_fp16"] == 16 and (not r[3]["has_zero"]) and r[3]["strict_inc"])]
    print(f"FP16-valid rebuilt LUTs: {len(ok)}/{len(rows)} (unique=16, strict inc, no zero)")

    if diffs:
        diffs.sort(key=lambda x: x[0], reverse=True)
        max_diff = diffs[0][0]
        avg_diff = sum(d for d, _, _ in diffs) / len(diffs)
        print(f"Max |stored_lut - rebuilt|: {max_diff:.6f}")
        print(f"Avg |stored_lut - rebuilt|: {avg_diff:.6f}")
        print()
        print("Top LUT mismatches:")
        for d, prefix, qc in diffs[:top_n]:
            print(f"  {prefix}: diff={d:.6f}, max_abs={qc['max_abs']:.3f}, "
                  f"min_abs={qc['min_abs']:.6f}, unique_fp16={qc['unique_fp16']}")
    else:
        print("No per-layer '.lut' buffers found (cannot diff stored vs rebuilt).")


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

    # LUT training stats (if any _lut_raw_deltas present)
    summarize_lut_raw_deltas(sd, top_n=top_n)

    # Q-indices consistency check
    q_consistency = check_q_indices_consistency(sd, top_n=top_n)

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

    # Check Q-indices consistency
    if q_consistency.get('checked', 0) > 0 and not q_consistency.get('consistent', True):
        print(f"\n⚠️  WARNING: _Q and lut[_indices] are inconsistent!")
        print(f"   {len(q_consistency['mismatches'])} layers have stale _Q values.")
        print("   Utilities reading _Q directly will see outdated quantization.")
        print("\n   Fix: Run sync_q_from_indices_all() or re-bake the checkpoint.")
        return 0  # Not critical, but worth noting

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
    parser.add_argument('-b', '--base-dir', metavar='FOLDER',
                        help='Base folder for checkpoint path')
    parser.add_argument('--top', type=int, default=20,
                        help='Show top N magnitudes (default: 20)')
    parser.add_argument('--verbose', action='store_true',
                        help='Show all tensors, not just suspects')

    args = parser.parse_args()

    # Apply base directory if specified
    if args.base_dir:
        args.checkpoint = str(Path(args.base_dir) / args.checkpoint)

    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return 1

    return check_checkpoint(args.checkpoint, args.top, args.verbose)


if __name__ == '__main__':
    sys.exit(main())
