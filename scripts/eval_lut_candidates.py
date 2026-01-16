#!/usr/bin/env python3
"""
Evaluate LUT candidate checkpoints with fast screening pipeline.

Pipeline:
1. Layer-only distortion: MAE per layer (from candidates_summary.json)
2. PPL first N chunks (--max-chunks) for fast screening
3. Full PPL for top-K candidates (optional)

This script reads candidates from apply_lut_candidates.py output and ranks
them using fast perplexity estimation.

Usage:
    # Fast screening with 20 chunks
    python scripts/eval_lut_candidates.py ./lut_candidates/ --max-chunks 20

    # Full PPL for top 3 candidates
    python scripts/eval_lut_candidates.py ./lut_candidates/ --max-chunks 20 --full-ppl --top-k 3

    # Custom config
    python scripts/eval_lut_candidates.py ./lut_candidates/ --config q2a4 --max-chunks 32

    # With specific device
    python scripts/eval_lut_candidates.py ./lut_candidates/ --device mps --dtype fp16
"""

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

REPO_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_DIR))


@dataclass
class CandidateResult:
    """Result for a single candidate."""
    name: str
    checkpoint_path: str
    avg_mae: float
    max_mae: float
    ppl_fast: Optional[float] = None
    ppl_full: Optional[float] = None
    time_fast: Optional[float] = None
    time_full: Optional[float] = None


def load_candidates_summary(candidates_dir: Path) -> Dict:
    """Load candidates_summary.json from output directory."""
    summary_path = candidates_dir / 'candidates_summary.json'
    if not summary_path.exists():
        raise FileNotFoundError(f"candidates_summary.json not found in {candidates_dir}")

    with open(summary_path) as f:
        return json.load(f)


def run_perplexity(
    checkpoint_path: str,
    max_chunks: Optional[int] = None,
    config: Optional[str] = None,
    model_id: str = 'Qwen/Qwen3-0.6B',
    device: str = 'auto',
    dtype: str = 'auto',
    batch_size: int = 0,
    seq_len: int = 512,
) -> Dict:
    """Run measure_perplexity.py and parse results."""
    cmd = [
        sys.executable,
        str(REPO_DIR / 'scripts' / 'measure_perplexity.py'),
        checkpoint_path,
        '--model', model_id,
        '--device', device,
        '--dtype', dtype,
    ]

    if config:
        cmd.extend(['--config', config])

    if max_chunks:
        cmd.extend(['--max-chunks', str(max_chunks)])

    if batch_size > 0:
        cmd.extend(['--batch-size', str(batch_size)])
        cmd.extend(['--seq-len', str(seq_len)])

    # Run and capture output
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Parse perplexity from output
    ppl = None
    time_s = None

    for line in result.stdout.split('\n'):
        if 'Perplexity:' in line:
            # Extract number after "Perplexity:"
            try:
                ppl = float(line.split('Perplexity:')[1].strip().split()[0])
            except (ValueError, IndexError):
                pass
        if 'Time:' in line:
            try:
                time_s = float(line.split('Time:')[1].strip().split('s')[0])
            except (ValueError, IndexError):
                pass

    return {
        'perplexity': ppl,
        'time': time_s,
        'stdout': result.stdout,
        'stderr': result.stderr,
        'returncode': result.returncode,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate LUT candidate checkpoints',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument('candidates_dir', help='Directory containing candidate checkpoints')
    parser.add_argument('--max-chunks', type=int, default=20,
                        help='Chunks for fast PPL screening (default: 20)')
    parser.add_argument('--full-ppl', action='store_true',
                        help='Run full PPL for top candidates')
    parser.add_argument('--top-k', type=int, default=3,
                        help='Number of top candidates for full eval (default: 3)')
    parser.add_argument('--config', default=None,
                        help='Config preset (q2a4, q4a4, etc.)')
    parser.add_argument('--model-id', default='Qwen/Qwen3-0.6B',
                        help='Base model name')
    parser.add_argument('--device', default='auto',
                        help='Device (auto, mps, cuda, cpu)')
    parser.add_argument('--dtype', default='auto',
                        help='Dtype (auto, fp16, bf16, fp32)')
    parser.add_argument('--batch-size', type=int, default=0,
                        help='Batch size for PPL (0=sliding window)')
    parser.add_argument('--seq-len', type=int, default=512,
                        help='Sequence length for batched mode')
    parser.add_argument('--output', '-o', default=None,
                        help='Output results JSON path')

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("EVALUATE LUT CANDIDATES")
    print("=" * 60)
    print(f"Candidates dir: {args.candidates_dir}")
    print(f"Max chunks:     {args.max_chunks}")
    print(f"Full PPL:       {args.full_ppl}")
    print(f"Top K:          {args.top_k}")
    print()

    candidates_dir = Path(args.candidates_dir)

    # Load summary
    try:
        summary = load_candidates_summary(candidates_dir)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Run apply_lut_candidates.py first to generate candidates.")
        return 1

    candidates = summary.get('candidates', [])
    if not candidates:
        print("ERROR: No candidates found in summary")
        return 1

    print(f"Found {len(candidates)} candidates")
    print()

    # Sort by avg_mae (pre-existing distortion metric)
    candidates.sort(key=lambda x: x.get('avg_mae', float('inf')))

    # Phase 1: Show MAE ranking
    print("=" * 60)
    print("PHASE 1: MAE RANKING (from apply_lut_candidates)")
    print("=" * 60)
    for i, c in enumerate(candidates[:10], 1):
        print(f"  {i}. {c['name']}: avg_MAE={c['avg_mae']:.6f}, max_MAE={c['max_mae']:.6f}")
    print()

    # Phase 2: Fast PPL screening
    print("=" * 60)
    print(f"PHASE 2: FAST PPL SCREENING (max_chunks={args.max_chunks})")
    print("=" * 60)

    results: List[CandidateResult] = []

    for i, c in enumerate(candidates, 1):
        name = c['name']
        ckpt_path = c['checkpoint_path']

        # Check if checkpoint exists
        if not Path(ckpt_path).exists():
            print(f"  {i}/{len(candidates)} {name}: SKIP (checkpoint not found)")
            continue

        print(f"  {i}/{len(candidates)} {name}...", end=' ', flush=True)

        ppl_result = run_perplexity(
            checkpoint_path=ckpt_path,
            max_chunks=args.max_chunks,
            config=args.config,
            model_id=args.model_id,
            device=args.device,
            dtype=args.dtype,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
        )

        if ppl_result['perplexity'] is not None:
            print(f"PPL={ppl_result['perplexity']:.2f} ({ppl_result['time']:.1f}s)")
            results.append(CandidateResult(
                name=name,
                checkpoint_path=ckpt_path,
                avg_mae=c['avg_mae'],
                max_mae=c['max_mae'],
                ppl_fast=ppl_result['perplexity'],
                time_fast=ppl_result['time'],
            ))
        else:
            print(f"ERROR (returncode={ppl_result['returncode']})")
            if ppl_result['stderr']:
                print(f"    stderr: {ppl_result['stderr'][:200]}")

    # Sort by fast PPL
    results.sort(key=lambda x: x.ppl_fast if x.ppl_fast else float('inf'))

    print()
    print("=" * 60)
    print("FAST PPL RANKING")
    print("=" * 60)
    for i, r in enumerate(results[:10], 1):
        print(f"  {i}. {r.name}: PPL={r.ppl_fast:.2f}, MAE={r.avg_mae:.6f}")

    # Phase 3: Full PPL for top K (optional)
    if args.full_ppl and results:
        print()
        print("=" * 60)
        print(f"PHASE 3: FULL PPL (top {args.top_k})")
        print("=" * 60)

        top_k = results[:args.top_k]
        for i, r in enumerate(top_k, 1):
            print(f"  {i}/{len(top_k)} {r.name}...", end=' ', flush=True)

            ppl_result = run_perplexity(
                checkpoint_path=r.checkpoint_path,
                max_chunks=None,  # Full PPL
                config=args.config,
                model_id=args.model_id,
                device=args.device,
                dtype=args.dtype,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
            )

            if ppl_result['perplexity'] is not None:
                r.ppl_full = ppl_result['perplexity']
                r.time_full = ppl_result['time']
                print(f"PPL={r.ppl_full:.2f} (full, {r.time_full:.1f}s)")
            else:
                print("ERROR")

        # Re-sort by full PPL for top K
        top_k_sorted = sorted(top_k, key=lambda x: x.ppl_full if x.ppl_full else float('inf'))

        print()
        print("=" * 60)
        print("FULL PPL RANKING")
        print("=" * 60)
        for i, r in enumerate(top_k_sorted, 1):
            fast_str = f"fast={r.ppl_fast:.2f}" if r.ppl_fast else "fast=?"
            full_str = f"full={r.ppl_full:.2f}" if r.ppl_full else "full=?"
            print(f"  {i}. {r.name}: {full_str}, {fast_str}, MAE={r.avg_mae:.6f}")

    # Final summary
    print()
    print("=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)

    if results:
        best = results[0]
        print(f"Best candidate: {best.name}")
        print(f"  Checkpoint:   {best.checkpoint_path}")
        print(f"  Fast PPL:     {best.ppl_fast:.2f}")
        print(f"  Avg MAE:      {best.avg_mae:.6f}")
        if best.ppl_full:
            print(f"  Full PPL:     {best.ppl_full:.2f}")
    else:
        print("No valid results to recommend")

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'candidates_dir': str(candidates_dir),
            'max_chunks': args.max_chunks,
            'results': [
                {
                    'name': r.name,
                    'checkpoint_path': r.checkpoint_path,
                    'avg_mae': r.avg_mae,
                    'max_mae': r.max_mae,
                    'ppl_fast': r.ppl_fast,
                    'ppl_full': r.ppl_full,
                    'time_fast': r.time_fast,
                    'time_full': r.time_full,
                }
                for r in results
            ],
        }
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved: {output_path}")

    print("\nDone!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
