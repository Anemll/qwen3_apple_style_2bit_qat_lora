#!/usr/bin/env python3
"""
Sync _Q buffers from lut[_indices] in a checkpoint file.

Use this to fix checkpoints where _Q has become stale (e.g., after LUT changes).
Works directly on checkpoint files without loading the full model.

Usage:
    python scripts/sync_q_checkpoint.py checkpoint.pt                    # Dry run
    python scripts/sync_q_checkpoint.py checkpoint.pt --inplace          # Fix in place
    python scripts/sync_q_checkpoint.py checkpoint.pt -o fixed.pt        # Save to new file
    python scripts/sync_q_checkpoint.py checkpoint.pt --verbose          # Show details
"""

import argparse
import sys
from pathlib import Path

import torch


@torch.no_grad()
def sync_q_in_state_dict(state_dict: dict, verbose: bool = False) -> dict:
    """Sync _Q = lut[_indices] for all layers in a state dict.

    Returns:
        Dict with 'synced', 'skipped', 'max_diff', 'layers'
    """
    stats = {
        'synced': 0,
        'skipped': 0,
        'max_diff': 0.0,
        'layers': [],
    }

    # Find all _indices keys
    indices_keys = [k for k in state_dict.keys() if k.endswith('._indices')]

    for indices_key in sorted(indices_keys):
        base = indices_key[:-9]  # Remove '._indices'
        lut_key = f"{base}.lut"
        q_key = f"{base}._Q"

        # Need both lut and indices
        if lut_key not in state_dict:
            if verbose:
                print(f"  SKIP {base}: no .lut")
            stats['skipped'] += 1
            continue

        indices = state_dict[indices_key].long()
        lut = state_dict[lut_key]

        # Compute new Q
        Q_new = lut[indices]

        # Check if _Q exists and compute diff
        diff = 0.0
        if q_key in state_dict:
            Q_old = state_dict[q_key]
            diff = (Q_new.float() - Q_old.float()).abs().max().item()
            # Preserve dtype
            Q_new = Q_new.to(Q_old.dtype)
        else:
            # No existing _Q, use lut dtype
            Q_new = Q_new.to(lut.dtype)

        # Update
        state_dict[q_key] = Q_new
        stats['synced'] += 1
        stats['max_diff'] = max(stats['max_diff'], diff)
        stats['layers'].append({
            'name': base,
            'diff': diff,
        })

        if verbose:
            flag = "*" if diff > 1e-3 else ""
            print(f"  {base}: diff={diff:.6f} {flag}")

    return stats


def sync_q_checkpoint(
    checkpoint_path: str,
    output_path: str = None,
    inplace: bool = False,
    verbose: bool = False,
) -> dict:
    """Sync _Q from lut[_indices] in a checkpoint file.

    Args:
        checkpoint_path: Input checkpoint
        output_path: Output path (required unless inplace or dry run)
        inplace: Overwrite input file
        verbose: Print per-layer info

    Returns:
        Dict with sync statistics
    """
    dry_run = not inplace and not output_path

    if inplace:
        output_path = checkpoint_path

    print(f"\n{'='*60}")
    print("SYNC _Q FROM LUT[_INDICES]")
    print(f"{'='*60}")
    print(f"Input:  {checkpoint_path}")
    if dry_run:
        print("Mode:   DRY RUN (no changes saved)")
    else:
        print(f"Output: {output_path}")

    # Load checkpoint
    sd = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Handle nested state dict
    is_nested = 'model_state_dict' in sd
    if is_nested:
        state_dict = sd['model_state_dict']
    else:
        state_dict = sd

    # Count what we have
    indices_count = len([k for k in state_dict.keys() if k.endswith('._indices')])
    q_count = len([k for k in state_dict.keys() if k.endswith('._Q')])
    lut_count = len([k for k in state_dict.keys() if k.endswith('.lut')])

    print(f"\nFound: {indices_count} _indices, {q_count} _Q, {lut_count} .lut")

    if indices_count == 0:
        print("\nNo _indices found - nothing to sync")
        return {'synced': 0, 'skipped': 0}

    # Sync
    print("\nSyncing...")
    stats = sync_q_in_state_dict(state_dict, verbose=verbose)

    # Summary
    print(f"\n{'-'*60}")
    print("RESULTS")
    print(f"{'-'*60}")
    print(f"Layers synced: {stats['synced']}")
    print(f"Layers skipped: {stats['skipped']}")
    print(f"Max diff:      {stats['max_diff']:.6f}")

    # Count layers with significant diff
    changed = [l for l in stats['layers'] if l['diff'] > 1e-3]
    if changed:
        print(f"\n{len(changed)} layers had diff > 1e-3:")
        for l in sorted(changed, key=lambda x: -x['diff'])[:5]:
            print(f"  {l['name']}: {l['diff']:.6f}")
        if len(changed) > 5:
            print(f"  ... and {len(changed) - 5} more")
    else:
        print("\nAll layers already consistent (diff <= 1e-3)")

    # Save if not dry run
    if not dry_run:
        if is_nested:
            sd['model_state_dict'] = state_dict
            torch.save(sd, output_path)
        else:
            torch.save(state_dict, output_path)
        print(f"\n✅ Saved to: {output_path}")
    else:
        print("\n(Dry run - no changes saved)")
        print("Use --inplace or -o <path> to save changes")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Sync _Q buffers from lut[_indices] in checkpoint',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('checkpoint', help='Input checkpoint file')
    parser.add_argument('-o', '--output', default=None,
                        help='Output checkpoint path')
    parser.add_argument('--inplace', action='store_true',
                        help='Overwrite input checkpoint')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Show per-layer details')

    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return 1

    try:
        stats = sync_q_checkpoint(
            args.checkpoint,
            args.output,
            args.inplace,
            args.verbose,
        )
        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
