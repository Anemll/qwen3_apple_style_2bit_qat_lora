#!/usr/bin/env python3
"""
Bake learned LUT values from _lut_raw_deltas into .lut buffer.

After LUT training, the learned values are stored in _lut_raw_deltas parameters,
but the .lut buffer remains unchanged. This script computes the actual LUT
from the deltas and updates the .lut buffer for proper inference.

Usage:
    python scripts/bake_lut.py source.pt dest.pt
    python scripts/bake_lut.py source.pt dest.pt --verbose
    python scripts/bake_lut.py source.pt --inplace  # Overwrites original
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Import repair function from ane_qat_linear_v2
sys.path.insert(0, str(Path(__file__).parent.parent))
from qat_lora.ane_qat_linear_v2 import repair_lut_duplicates_symmetric, verify_lut_fp16


def build_symmetric_lut16(raw_deltas: torch.Tensor, max_abs: float, min_delta: float) -> torch.Tensor:
    """Build symmetric 16-value LUT from raw delta logits.

    Same logic as in ane_qat_linear_v2.py.
    """
    half_size = 8

    # Constrained simplex: softmax gives weights, scale by remaining budget
    weights = F.softmax(raw_deltas, dim=0)
    remaining = max_abs - half_size * min_delta
    deltas = min_delta + remaining * weights

    # Build positive half via cumsum
    positive_lut = torch.cumsum(deltas, dim=0)

    # Build negative half (mirror)
    negative_lut = -positive_lut.flip(0)

    # Concatenate: [-max, ..., -ε, +ε, ..., +max]
    lut = torch.cat([negative_lut, positive_lut], dim=0)

    return lut


def compute_min_delta(max_abs: float) -> float:
    """Compute minimum delta to ensure FP16-safe values."""
    return max(1e-4, max_abs * 0.001)


def bake_lut_checkpoint(
    checkpoint_path: str,
    output_path: str = None,
    inplace: bool = False,
    verbose: bool = False,
) -> dict:
    """Bake LUT values from _lut_raw_deltas into .lut buffers.

    Args:
        checkpoint_path: Input checkpoint with _lut_raw_deltas
        output_path: Output checkpoint path (required unless inplace)
        inplace: Overwrite input checkpoint
        verbose: Print detailed info

    Returns:
        Dict with bake statistics
    """
    if not inplace and not output_path:
        raise ValueError("Must specify output_path or use --inplace")

    if inplace:
        output_path = checkpoint_path

    print(f"\n{'='*60}")
    print("BAKE LUT FROM DELTAS")
    print(f"{'='*60}")
    print(f"Input:  {checkpoint_path}")
    print(f"Output: {output_path}")

    # Load checkpoint
    sd = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Handle nested state dict
    is_nested = 'model_state_dict' in sd
    if is_nested:
        state_dict = sd['model_state_dict']
    else:
        state_dict = sd

    # Find all _lut_raw_deltas keys
    delta_keys = [k for k in state_dict.keys() if '_lut_raw_deltas' in k]

    if not delta_keys:
        print("\n⚠️  No _lut_raw_deltas found in checkpoint")
        print("   Nothing to bake - checkpoint may already be baked or not LUT-trained")
        return {'baked': 0, 'skipped': 0}

    print(f"\nFound {len(delta_keys)} _lut_raw_deltas tensors")

    # Process each delta
    stats = {
        'baked': 0,
        'skipped': 0,
        'errors': 0,
        'q_refreshed': 0,
        'layers': [],
    }

    for delta_key in sorted(delta_keys):
        # Derive the .lut key from the delta key
        # _lut_raw_deltas -> lut
        base_key = delta_key.replace('_lut_raw_deltas', '')
        lut_key = base_key + 'lut'

        # Also need max_abs and min_delta
        # These are stored as _lut_max_abs and _lut_min_delta (non-tensor attributes)
        # If not in state_dict, use defaults
        max_abs_key = base_key + '_lut_max_abs'
        min_delta_key = base_key + '_lut_min_delta'

        raw_deltas = state_dict[delta_key]

        if lut_key not in state_dict:
            if verbose:
                print(f"  SKIP: {delta_key} (no matching .lut)")
            stats['skipped'] += 1
            continue

        # Get max_abs from state_dict or use default
        # Note: max_abs is typically a Python float, not saved in state_dict
        # We need to infer it from the current LUT or use default
        current_lut = state_dict[lut_key]
        max_abs = current_lut.abs().max().item()
        if max_abs < 0.1:
            max_abs = 2.0  # Default

        min_delta = compute_min_delta(max_abs)

        # Build the learned LUT from raw deltas
        learned_lut = build_symmetric_lut16(raw_deltas, max_abs, min_delta)

        # CRITICAL: Apply FP16 snap + repair to get the exact LUT that will be used at inference
        # This ensures _Q == lut[_indices] in the saved checkpoint
        baked_lut = repair_lut_duplicates_symmetric(learned_lut, max_abs)

        # Validate the repaired LUT
        if not verify_lut_fp16(baked_lut):
            print(f"  WARNING: {delta_key} LUT failed FP16 validation after repair")
            stats['errors'] += 1

        # Compute difference for stats (vs old stored LUT)
        old_lut = state_dict[lut_key]
        diff = (baked_lut - old_lut).abs().max().item()

        if verbose:
            layer_name = delta_key.replace('._lut_raw_deltas', '')
            print(f"\n  {layer_name}")
            print(f"    max_abs={max_abs:.4f}, max_diff={diff:.6f}")
            print(f"    {'idx':>3}  {'old':>10}  {'baked':>10}  {'diff':>10}")
            print(f"    {'-'*3}  {'-'*10}  {'-'*10}  {'-'*10}")
            for i in range(len(old_lut)):
                o = old_lut[i].item()
                n = baked_lut[i].item()
                d = n - o
                flag = "*" if abs(d) > 0.001 else " "
                print(f"    {i:3d}  {o:10.6f}  {n:10.6f}  {d:+10.6f} {flag}")

        # Update the .lut buffer with the REPAIRED LUT
        state_dict[lut_key] = baked_lut.to(old_lut.dtype)

        # CRITICAL: Compute _Q from the SAME repaired LUT that we just saved
        # This guarantees _Q == lut[_indices] in the saved checkpoint
        indices_key = base_key + '_indices'
        q_key = base_key + '_Q'
        if indices_key in state_dict:
            indices = state_dict[indices_key]
            Q_new = baked_lut[indices.long()]
            # Match dtype of existing _Q or use LUT dtype
            if q_key in state_dict:
                Q_new = Q_new.to(state_dict[q_key].dtype)
            else:
                Q_new = Q_new.to(old_lut.dtype)
            state_dict[q_key] = Q_new
            stats['q_refreshed'] += 1
            if verbose:
                print(f"    _Q refreshed from baked_lut[_indices]")

        # Remove _lut_raw_deltas (no longer needed)
        del state_dict[delta_key]

        stats['baked'] += 1
        stats['layers'].append({
            'name': delta_key.replace('._lut_raw_deltas', ''),
            'max_diff': diff,
        })

    # Save updated checkpoint
    if is_nested:
        sd['model_state_dict'] = state_dict
        torch.save(sd, output_path)
    else:
        torch.save(state_dict, output_path)

    # Summary
    print(f"\n{'-'*60}")
    print("RESULTS")
    print(f"{'-'*60}")
    print(f"LUTs baked:   {stats['baked']}")
    print(f"_Q refreshed: {stats['q_refreshed']}")
    print(f"Skipped:      {stats['skipped']}")
    if stats['errors'] > 0:
        print(f"Errors:       {stats['errors']} (LUT validation failed)")

    if stats['layers']:
        max_diff = max(l['max_diff'] for l in stats['layers'])
        avg_diff = sum(l['max_diff'] for l in stats['layers']) / len(stats['layers'])
        print(f"Max LUT diff: {max_diff:.6f}")
        print(f"Avg LUT diff: {avg_diff:.6f}")

    print(f"\n✅ Saved baked checkpoint to: {output_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Bake learned LUT values from _lut_raw_deltas into .lut buffer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('source', help='Input checkpoint with _lut_raw_deltas')
    parser.add_argument('dest', nargs='?', default=None,
                        help='Output checkpoint path (optional if --inplace)')
    parser.add_argument('--inplace', action='store_true',
                        help='Overwrite input checkpoint')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed per-layer info')

    args = parser.parse_args()

    if not Path(args.source).exists():
        print(f"Error: Checkpoint not found: {args.source}")
        return 1

    if not args.dest and not args.inplace:
        print("Error: Must specify dest or --inplace")
        return 1

    try:
        stats = bake_lut_checkpoint(
            args.source,
            args.dest,
            args.inplace,
            args.verbose,
        )
        return 0 if stats['baked'] > 0 or stats['skipped'] == 0 else 1
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
