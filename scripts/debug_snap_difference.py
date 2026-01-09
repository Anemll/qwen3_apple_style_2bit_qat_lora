#!/usr/bin/env python3
"""
Compare original vs snapped checkpoint to diagnose divergence.

Usage:
    python scripts/debug_snap_difference.py /path/to/original.pt /path/to/snapped.pt

    # With default paths (SR-008-32B):
    python scripts/debug_snap_difference.py
"""

import argparse
import torch
from pathlib import Path


def load_checkpoint(path):
    """Load checkpoint, handle nested state_dict."""
    state_dict = torch.load(path, map_location='cpu')
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    return state_dict


def compare_keys(orig, snap):
    """Compare key sets between checkpoints."""
    orig_keys = set(orig.keys())
    snap_keys = set(snap.keys())

    only_orig = orig_keys - snap_keys
    only_snap = snap_keys - orig_keys
    common = orig_keys & snap_keys

    print("=" * 60)
    print("KEY COMPARISON")
    print("=" * 60)
    print(f"Keys in original: {len(orig_keys)}")
    print(f"Keys in snapped:  {len(snap_keys)}")
    print(f"Keys in common:   {len(common)}")
    print(f"Only in original: {len(only_orig)}")
    print(f"Only in snapped:  {len(only_snap)}")

    # Check for _Q buffers
    orig_Q = [k for k in orig_keys if '._Q' in k]
    snap_Q = [k for k in snap_keys if '._Q' in k]
    print(f"\n_Q buffers in original: {len(orig_Q)}")
    print(f"_Q buffers in snapped:  {len(snap_Q)}")

    # Check for _scales_baked_flag - show both COUNT and VALUES
    orig_baked_keys = [k for k in orig_keys if '_scales_baked' in k]
    snap_baked_keys = [k for k in snap_keys if '_scales_baked' in k]

    # Count how many buffers have value=1 (actually baked) vs value=0 (not baked)
    def count_baked_values(state_dict, keys):
        baked_true = 0
        baked_false = 0
        for k in keys:
            val = state_dict[k].item() if state_dict[k].numel() == 1 else state_dict[k][0].item()
            if val == 1:
                baked_true += 1
            else:
                baked_false += 1
        return baked_true, baked_false

    print(f"\n_scales_baked_flag buffers:")
    print(f"  Original: {len(orig_baked_keys)} buffers exist", end="")
    if orig_baked_keys:
        orig_true, orig_false = count_baked_values(orig, orig_baked_keys)
        print(f" → {orig_true} baked (value=1), {orig_false} not baked (value=0)")
    else:
        print(" (none)")

    print(f"  Snapped:  {len(snap_baked_keys)} buffers exist", end="")
    if snap_baked_keys:
        snap_true, snap_false = count_baked_values(snap, snap_baked_keys)
        print(f" → {snap_true} baked (value=1), {snap_false} not baked (value=0)")
    else:
        print(" (none)")

    return common


def compare_dtypes(orig, snap, common_keys):
    """Compare data types between checkpoints."""
    print("\n" + "=" * 60)
    print("DTYPE COMPARISON")
    print("=" * 60)

    dtype_changes = {}
    for key in common_keys:
        orig_dtype = orig[key].dtype
        snap_dtype = snap[key].dtype
        if orig_dtype != snap_dtype:
            change = f"{orig_dtype} -> {snap_dtype}"
            if change not in dtype_changes:
                dtype_changes[change] = []
            dtype_changes[change].append(key)

    if dtype_changes:
        for change, keys in dtype_changes.items():
            print(f"{change}: {len(keys)} tensors")
            # Show first 3 examples
            for k in keys[:3]:
                print(f"    {k}")
            if len(keys) > 3:
                print(f"    ... and {len(keys) - 3} more")
    else:
        print("All common tensors have same dtype")


def analyze_layer(orig, snap, layer_prefix):
    """Analyze a single layer in detail."""
    print(f"\n=== Layer: {layer_prefix} ===")

    suffixes = ['.weight', '.scale_A', '.scale_B', '.rank_magnitude', '._Q', '._scales_baked_flag']

    for suffix in suffixes:
        key = f"{layer_prefix}{suffix}"
        in_orig = key in orig
        in_snap = key in snap

        status = ""
        if in_orig and in_snap:
            o, s = orig[key], snap[key]
            # Compare original vs snapped (both as FP32 on CPU)
            o_f32 = o.cpu().float()
            s_f32 = s.cpu().float()
            diff = (o_f32 - s_f32).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()

            # Also show what CPU FP16 snap of original would give
            o_cpu_snap = o_f32.half().float()
            cpu_snap_diff = (o_cpu_snap - s_f32).abs().max().item()

            status = f"max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, dtype={o.dtype}->{s.dtype}"
            if cpu_snap_diff > 0:
                status += f" [CPU_SNAP_DIFF={cpu_snap_diff:.6f}]"
        elif in_orig:
            status = "ONLY IN ORIGINAL"
        elif in_snap:
            status = f"ONLY IN SNAPPED (dtype={snap[key].dtype})"
        else:
            status = "NOT IN EITHER"

        print(f"  {suffix:25s} | {status}")

    # Check rank_magnitude baking
    rm_key = f"{layer_prefix}.rank_magnitude"
    if rm_key in snap:
        rm = snap[rm_key]
        is_baked = torch.allclose(rm.float(), torch.ones_like(rm.float()), atol=1e-5)
        if is_baked:
            print(f"  [BAKED] rank_magnitude is all ones")
        else:
            print(f"  [NOT BAKED] rank_magnitude range: [{rm.min():.4f}, {rm.max():.4f}]")


def compute_w_eff_diff(orig, snap, layer_prefix):
    """Compute W_eff = Q * scales difference for a layer."""
    keys_needed = [
        f"{layer_prefix}._Q",
        f"{layer_prefix}.scale_A",
        f"{layer_prefix}.scale_B",
        f"{layer_prefix}.rank_magnitude",
    ]

    # Check all keys exist
    for key in keys_needed:
        if key not in orig or key not in snap:
            return None

    def compute_w_eff(state_dict, prefix):
        # All values as FP32 on CPU
        Q = state_dict[f"{prefix}._Q"].cpu().float()
        A = state_dict[f"{prefix}.scale_A"].cpu().float()
        B = state_dict[f"{prefix}.scale_B"].cpu().float()
        g = state_dict[f"{prefix}.rank_magnitude"].cpu().float()

        # Compute full scales: (A * g) @ B
        scales = (A * g) @ B
        return Q * scales

    # Compare both as FP32
    orig_w = compute_w_eff(orig, layer_prefix)
    snap_w = compute_w_eff(snap, layer_prefix)

    diff = (orig_w - snap_w).abs()
    rel_diff = diff / (orig_w.abs() + 1e-8)

    return {
        'max_diff': diff.max().item(),
        'mean_diff': diff.mean().item(),
        'rel_max': rel_diff.max().item(),
        'rel_mean': rel_diff.mean().item(),
    }


def find_largest_diffs(orig, snap, common_keys, top_n=20):
    """Find tensors with largest differences."""
    print("\n" + "=" * 60)
    print(f"TOP {top_n} LARGEST DIFFERENCES")
    print("=" * 60)

    diffs = []
    for key in common_keys:
        if orig[key].shape != snap[key].shape:
            continue
        # Compare original FP32 vs snapped (as FP32)
        o_f32 = orig[key].cpu().float()
        s_f32 = snap[key].cpu().float()
        diff = (o_f32 - s_f32).abs().max().item()
        diffs.append((key, diff))

    diffs.sort(key=lambda x: -x[1])

    for key, diff in diffs[:top_n]:
        print(f"{diff:.6f}: {key}")


def analyze_lut_fp16(orig, snap):
    """Check if LUT values are FP16-representable."""
    print("\n" + "=" * 60)
    print("LUT FP16 PRECISION CHECK")
    print("=" * 60)

    for name, state_dict in [("Original", orig), ("Snapped", snap)]:
        lut_keys = [k for k in state_dict.keys() if k.endswith('.lut')]
        if not lut_keys:
            print(f"{name}: No LUT tensors found")
            continue

        lut_snapped = 0
        lut_unsnapped = 0
        lut_diffs = []

        for key in lut_keys:
            val = state_dict[key].float()
            snapped = val.cpu().half().float()
            diff = (val - snapped).abs()
            max_diff = diff.max().item()

            if max_diff == 0.0:
                lut_snapped += 1
            else:
                lut_unsnapped += 1
                lut_diffs.append((max_diff, key, val.shape[0]))

        print(f"\n{name}:")
        print(f"  Found {len(lut_keys)} LUT tensors")
        print(f"  FP16-snapped: {lut_snapped}/{len(lut_keys)}")
        print(f"  NOT snapped:  {lut_unsnapped}/{len(lut_keys)}")

        if lut_unsnapped > 0:
            lut_diffs.sort(reverse=True, key=lambda x: x[0])
            print(f"  Top differences:")
            for i, (diff, key, lut_size) in enumerate(lut_diffs[:3]):
                print(f"    {i+1}. diff={diff:.6f} | LUT{lut_size} | {key}")

        # Show sample LUT
        if lut_keys:
            sample_key = lut_keys[0]
            sample_lut = state_dict[sample_key]
            print(f"  Sample LUT ({sample_key.split('.')[-2]}):")
            print(f"    dtype: {sample_lut.dtype}, shape: {sample_lut.shape}")
            print(f"    values: {sample_lut.flatten().tolist()}")

            # Check if all LUTs identical
            first_lut = state_dict[lut_keys[0]].float()
            all_identical = all(
                torch.allclose(first_lut, state_dict[k].float())
                for k in lut_keys[1:]
            )
            print(f"    All LUTs identical: {'✓ Yes' if all_identical else '✗ No'}")


def analyze_attention_vs_mlp(orig, snap, common_keys):
    """Compare attention vs MLP layer differences."""
    print("\n" + "=" * 60)
    print("ATTENTION vs MLP ANALYSIS")
    print("=" * 60)

    attn_diffs = []
    mlp_diffs = []

    for key in common_keys:
        if orig[key].shape != snap[key].shape:
            continue
        # Compare original FP32 vs snapped (as FP32)
        o_f32 = orig[key].cpu().float()
        s_f32 = snap[key].cpu().float()
        diff = (o_f32 - s_f32).abs().max().item()

        if 'q_proj' in key or 'k_proj' in key or 'v_proj' in key or 'o_proj' in key:
            attn_diffs.append(diff)
        elif 'gate_proj' in key or 'up_proj' in key or 'down_proj' in key:
            mlp_diffs.append(diff)

    if attn_diffs:
        print(f"Attention layers: avg_max_diff={sum(attn_diffs)/len(attn_diffs):.6f} (n={len(attn_diffs)})")
    if mlp_diffs:
        print(f"MLP layers:       avg_max_diff={sum(mlp_diffs)/len(mlp_diffs):.6f} (n={len(mlp_diffs)})")


def main():
    parser = argparse.ArgumentParser(description='Compare original vs snapped checkpoint')
    parser.add_argument('original', type=str, nargs='?',
                        default='/Users/anemll/Downloads/SR-008-32B/best_recovery_lora200.pt',
                        help='Path to original checkpoint')
    parser.add_argument('snapped', type=str, nargs='?',
                        default='/Users/anemll/Downloads/SR-008-32B/best_recovery_lora200_snapped.pt',
                        help='Path to snapped checkpoint')
    parser.add_argument('--layers', type=str, nargs='*',
                        default=['model.layers.0.mlp.gate_proj', 'model.layers.0.self_attn.q_proj'],
                        help='Layers to analyze in detail')
    args = parser.parse_args()

    print("=" * 60)
    print("SNAP DIFFERENCE DIAGNOSTIC")
    print("=" * 60)
    print(f"Original: {args.original}")
    print(f"Snapped:  {args.snapped}")

    # Load checkpoints
    print("\nLoading checkpoints...")
    orig = load_checkpoint(args.original)
    snap = load_checkpoint(args.snapped)
    print("Done.")

    # Compare keys
    common_keys = compare_keys(orig, snap)

    # Compare dtypes
    compare_dtypes(orig, snap, common_keys)

    # Analyze specific layers
    for layer in args.layers:
        analyze_layer(orig, snap, layer)

        # Compute W_eff difference
        w_eff = compute_w_eff_diff(orig, snap, layer)
        if w_eff:
            print(f"  W_eff diff: max={w_eff['max_diff']:.6f}, mean={w_eff['mean_diff']:.6f}, "
                  f"rel_max={w_eff['rel_max']:.4%}, rel_mean={w_eff['rel_mean']:.4%}")

    # Find largest differences
    find_largest_diffs(orig, snap, common_keys)

    # Attention vs MLP
    analyze_attention_vs_mlp(orig, snap, common_keys)

    # LUT FP16 precision check
    analyze_lut_fp16(orig, snap)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Summary checks
    orig_Q_count = len([k for k in orig.keys() if '._Q' in k])
    snap_Q_count = len([k for k in snap.keys() if '._Q' in k])

    orig_fp16 = sum(1 for k in orig.keys() if orig[k].dtype == torch.float16)
    snap_fp16 = sum(1 for k in snap.keys() if snap[k].dtype == torch.float16)

    print(f"_Q buffers: orig={orig_Q_count}, snap={snap_Q_count}")
    print(f"FP16 tensors: orig={orig_fp16}, snap={snap_fp16}")

    # Check if scales are baked
    baked_count = 0
    for key in snap.keys():
        if '.rank_magnitude' in key:
            rm = snap[key]
            if torch.allclose(rm.float(), torch.ones_like(rm.float()), atol=1e-5):
                baked_count += 1

    total_rm = len([k for k in snap.keys() if '.rank_magnitude' in k])
    print(f"Baked scales: {baked_count}/{total_rm} layers")

    if snap_fp16 > orig_fp16:
        print("\n[!] Snapped checkpoint has FP16 tensors - FP16 precision mismatch likely cause")
    if baked_count > 0:
        print(f"[!] {baked_count} layers have baked scales - scale computation path differs")


if __name__ == '__main__':
    main()
