#!/usr/bin/env python3
"""
Analyze LUT evolution from initial to trained checkpoint.

Compares LUT values between two checkpoints to understand:
1. How LUTs change during training
2. Which layers change most
3. Patterns that could inform better initialization

Usage:
    # Direct checkpoint paths
    python scripts/analyze_lut_evolution.py initial.pt trained.pt

    # Compare vs linear initialization
    python scripts/analyze_lut_evolution.py --initial-linear trained.pt

    # Use Google Drive run folder (-b flag)
    python scripts/analyze_lut_evolution.py -b srLUT-004b_all_alpaca --step1 200 --step2 2000
    python scripts/analyze_lut_evolution.py -b srLUT-004b_all_alpaca --initial-linear --step 2000
    python scripts/analyze_lut_evolution.py -b srLUT-004b_all_alpaca --best  # Use baked_best.pt

    # Quick summary only
    python scripts/analyze_lut_evolution.py trained.pt --summary
"""

import argparse
import sys
from pathlib import Path
import torch
import numpy as np
from collections import defaultdict
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def detect_platform():
    """Detect the platform we're running on."""
    if os.path.exists("/content"):
        return "colab"
    elif sys.platform == "darwin":
        return "macos"
    else:
        return "linux"


def find_gdrive_root():
    """Find the Google Drive root directory for qwen3_runs."""
    # Check environment variable first
    env_base = os.environ.get("GDRIVE_BASE")
    if env_base and Path(env_base).exists():
        return Path(env_base)

    platform = detect_platform()

    if platform == "colab":
        # Colab: /content/drive/MyDrive/qwen3_runs
        gdrive_base = Path("/content/drive/MyDrive")
        if not gdrive_base.exists():
            print("WARNING: Google Drive not mounted in Colab!")
            print("  Run: from google.colab import drive; drive.mount('/content/drive')")
            return None
        return gdrive_base / "qwen3_runs"

    elif platform == "macos":
        # macOS: CloudStorage mount
        candidates = [
            os.path.expanduser("~/Library/CloudStorage/GoogleDrive-realanemll@gmail.com/My Drive/qwen3_runs"),
            os.path.expanduser("~/Library/CloudStorage/GoogleDrive/My Drive/qwen3_runs"),
            os.path.expanduser("~/Google Drive/My Drive/qwen3_runs"),
        ]
        for path in candidates:
            if Path(path).exists():
                return Path(path)
        return None

    else:
        # Linux: typical mount points
        candidates = [
            os.path.expanduser("~/gdrive/qwen3_runs"),
            os.path.expanduser("~/google-drive/qwen3_runs"),
            "/mnt/gdrive/qwen3_runs",
        ]
        for path in candidates:
            if Path(path).exists():
                return Path(path)
        return None


def resolve_checkpoint_path(run_name: str, step: int = None, best: bool = False,
                           gdrive_root: Path = None, local_root: str = "runs") -> Path:
    """Resolve a checkpoint path from run name and step."""
    # Try Google Drive first
    if gdrive_root is None:
        gdrive_root = find_gdrive_root()

    search_paths = []
    if gdrive_root:
        search_paths.append(gdrive_root / run_name)
    search_paths.append(Path(local_root) / run_name)

    for base_dir in search_paths:
        if not base_dir.exists():
            continue

        if best:
            # Look for baked_best*.pt
            candidates = list(base_dir.glob("baked_best*.pt"))
            if candidates:
                return candidates[0]
            # Fall back to best_state_dict.pt
            best_path = base_dir / "best_state_dict.pt"
            if best_path.exists():
                return best_path
        elif step is not None:
            # Look for checkpoint at specific step
            ckpt_path = base_dir / f"checkpoint_step{step}.pt"
            if ckpt_path.exists():
                return ckpt_path
            # Try baked version
            baked_candidates = list(base_dir.glob(f"baked_step{step}*.pt"))
            if baked_candidates:
                return baked_candidates[0]

    # If not found, return expected path for error message
    if gdrive_root:
        return gdrive_root / run_name / (f"checkpoint_step{step}.pt" if step else "baked_best.pt")
    return Path(local_root) / run_name / (f"checkpoint_step{step}.pt" if step else "baked_best.pt")


def load_luts_from_checkpoint(ckpt_path: str) -> dict:
    """Extract all LUT tensors from a checkpoint."""
    print(f"Loading: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']

    luts = {}
    for key, value in state_dict.items():
        if '.lut' in key and isinstance(value, torch.Tensor):
            luts[key] = value.float().cpu()

    print(f"  Found {len(luts)} LUT tensors")
    return luts


def generate_linear_luts(lut_size: int = 16, max_abs: float = 2.0) -> torch.Tensor:
    """Generate default linear LUT initialization."""
    return torch.linspace(-max_abs, max_abs, lut_size)


def analyze_single_lut(initial: torch.Tensor, trained: torch.Tensor, name: str) -> dict:
    """Analyze changes in a single LUT."""
    diff = trained - initial

    # Basic stats
    stats = {
        'name': name,
        'lut_size': len(initial),
        'initial_range': (initial.min().item(), initial.max().item()),
        'trained_range': (trained.min().item(), trained.max().item()),
        'mean_shift': diff.mean().item(),
        'max_abs_change': diff.abs().max().item(),
        'rms_change': (diff ** 2).mean().sqrt().item(),
        'initial_spacing': (initial[1:] - initial[:-1]).mean().item(),
        'trained_spacing': (trained[1:] - trained[:-1]).mean().item(),
    }

    # Analyze spacing changes (are bins more/less uniform?)
    initial_spacing = initial[1:] - initial[:-1]
    trained_spacing = trained[1:] - trained[:-1]
    stats['spacing_std_initial'] = initial_spacing.std().item()
    stats['spacing_std_trained'] = trained_spacing.std().item()

    # Check if spacing became more non-uniform (adaptive quantization)
    stats['spacing_ratio'] = stats['spacing_std_trained'] / max(stats['spacing_std_initial'], 1e-6)

    # Per-bin changes
    stats['per_bin_change'] = diff.tolist()
    stats['initial_values'] = initial.tolist()
    stats['trained_values'] = trained.tolist()

    return stats


def categorize_layer(name: str) -> tuple:
    """Extract layer info from name."""
    # Example: model.layers.0.mlp.gate_proj.lut
    parts = name.split('.')

    layer_idx = None
    layer_type = None  # mlp or self_attn
    proj_type = None   # gate_proj, up_proj, down_proj, q_proj, k_proj, v_proj, o_proj

    for i, p in enumerate(parts):
        if p == 'layers' and i + 1 < len(parts):
            try:
                layer_idx = int(parts[i + 1])
            except ValueError:
                pass
        if p == 'mlp':
            layer_type = 'mlp'
        if p == 'self_attn':
            layer_type = 'attn'
        if p in ['gate_proj', 'up_proj', 'down_proj', 'q_proj', 'k_proj', 'v_proj', 'o_proj']:
            proj_type = p

    return layer_idx, layer_type, proj_type


def print_summary(all_stats: list, top_n: int = 10):
    """Print summary of LUT changes."""
    print("\n" + "=" * 70)
    print("LUT EVOLUTION SUMMARY")
    print("=" * 70)

    # Overall stats
    total_luts = len(all_stats)
    mean_rms = np.mean([s['rms_change'] for s in all_stats])
    max_change = max(s['max_abs_change'] for s in all_stats)

    print(f"\nTotal LUTs analyzed: {total_luts}")
    print(f"Mean RMS change: {mean_rms:.6f}")
    print(f"Max absolute change: {max_change:.6f}")

    # Top changers
    sorted_by_rms = sorted(all_stats, key=lambda x: x['rms_change'], reverse=True)
    print(f"\n--- Top {top_n} Most Changed LUTs ---")
    for i, s in enumerate(sorted_by_rms[:top_n]):
        print(f"  {i+1}. {s['name']}")
        print(f"     RMS: {s['rms_change']:.6f}, Max: {s['max_abs_change']:.6f}, Mean shift: {s['mean_shift']:+.6f}")

    # Least changers
    print(f"\n--- Top {top_n} Least Changed LUTs ---")
    for i, s in enumerate(sorted_by_rms[-top_n:]):
        print(f"  {i+1}. {s['name']}")
        print(f"     RMS: {s['rms_change']:.6f}, Max: {s['max_abs_change']:.6f}")

    # By layer type
    mlp_stats = [s for s in all_stats if 'mlp' in s['name']]
    attn_stats = [s for s in all_stats if 'self_attn' in s['name']]

    print(f"\n--- By Layer Type ---")
    if mlp_stats:
        print(f"  MLP ({len(mlp_stats)} LUTs):")
        print(f"    Mean RMS: {np.mean([s['rms_change'] for s in mlp_stats]):.6f}")
    if attn_stats:
        print(f"  Attention ({len(attn_stats)} LUTs):")
        print(f"    Mean RMS: {np.mean([s['rms_change'] for s in attn_stats]):.6f}")

    # By projection type
    proj_types = defaultdict(list)
    for s in all_stats:
        _, _, proj = categorize_layer(s['name'])
        if proj:
            proj_types[proj].append(s)

    print(f"\n--- By Projection Type ---")
    for proj, stats in sorted(proj_types.items()):
        mean_rms = np.mean([s['rms_change'] for s in stats])
        print(f"  {proj} ({len(stats)} LUTs): Mean RMS = {mean_rms:.6f}")

    # By layer depth
    layer_changes = defaultdict(list)
    for s in all_stats:
        layer_idx, _, _ = categorize_layer(s['name'])
        if layer_idx is not None:
            layer_changes[layer_idx].append(s['rms_change'])

    print(f"\n--- By Layer Depth ---")
    early = [v for k, vals in layer_changes.items() for v in vals if k < 10]
    mid = [v for k, vals in layer_changes.items() for v in vals if 10 <= k < 20]
    late = [v for k, vals in layer_changes.items() for v in vals if k >= 20]

    if early:
        print(f"  Early layers (0-9): Mean RMS = {np.mean(early):.6f}")
    if mid:
        print(f"  Middle layers (10-19): Mean RMS = {np.mean(mid):.6f}")
    if late:
        print(f"  Late layers (20+): Mean RMS = {np.mean(late):.6f}")

    # Spacing analysis
    print(f"\n--- Spacing Analysis ---")
    more_uniform = sum(1 for s in all_stats if s['spacing_ratio'] < 0.9)
    less_uniform = sum(1 for s in all_stats if s['spacing_ratio'] > 1.1)
    print(f"  LUTs with MORE uniform spacing after training: {more_uniform}")
    print(f"  LUTs with LESS uniform spacing after training: {less_uniform}")
    print(f"  (ratio < 0.9 = more uniform, > 1.1 = less uniform)")


def analyze_bin_patterns(all_stats: list):
    """Analyze per-bin change patterns across all LUTs."""
    print("\n" + "=" * 70)
    print("PER-BIN ANALYSIS")
    print("=" * 70)

    # Aggregate per-bin changes
    lut_size = all_stats[0]['lut_size'] if all_stats else 16
    bin_changes = [[] for _ in range(lut_size)]

    for s in all_stats:
        for i, change in enumerate(s['per_bin_change']):
            if i < lut_size:
                bin_changes[i].append(change)

    print(f"\nPer-bin average changes (LUT size={lut_size}):")
    print("  Bin | Mean Change | Std Dev  | Typical Direction")
    print("  " + "-" * 55)

    for i, changes in enumerate(bin_changes):
        if changes:
            mean = np.mean(changes)
            std = np.std(changes)
            direction = "↑" if mean > 0.001 else "↓" if mean < -0.001 else "→"
            print(f"  {i:3d} | {mean:+10.6f} | {std:8.6f} | {direction}")

    # Suggest optimal initial values
    print(f"\n--- Suggested Optimized Initial LUT ---")
    print("  Based on average trained values:")
    avg_trained = []
    for s in all_stats:
        avg_trained.append(s['trained_values'])

    if avg_trained:
        avg_trained = np.mean(avg_trained, axis=0)
        print(f"  lut = torch.tensor([")
        for i in range(0, len(avg_trained), 4):
            chunk = avg_trained[i:i+4]
            print(f"      " + ", ".join(f"{v:+.6f}" for v in chunk) + ",")
        print(f"  ])")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze LUT evolution between checkpoints',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Direct checkpoint paths
  python scripts/analyze_lut_evolution.py initial.pt trained.pt

  # Use Google Drive run folder
  python scripts/analyze_lut_evolution.py -b srLUT-004b_all_alpaca --step1 200 --step2 2000

  # Compare trained checkpoint vs linear init
  python scripts/analyze_lut_evolution.py -b srLUT-004b_all_alpaca --initial-linear --step 2000
  python scripts/analyze_lut_evolution.py -b srLUT-004b_all_alpaca --initial-linear --best

  # Quick summary
  python scripts/analyze_lut_evolution.py trained.pt --summary
"""
    )

    # Checkpoint specification (direct paths)
    parser.add_argument('checkpoint1', nargs='?', help='Initial checkpoint (or trained if --initial-linear)')
    parser.add_argument('checkpoint2', nargs='?', help='Trained checkpoint')

    # Google Drive run folder mode
    parser.add_argument('-b', '-B', '--base-dir', metavar='RUN_NAME',
                       help='Run name in Google Drive (e.g., srLUT-004b_all_alpaca)')
    parser.add_argument('--step1', type=int, help='Initial checkpoint step (with -b)')
    parser.add_argument('--step2', type=int, help='Trained checkpoint step (with -b)')
    parser.add_argument('--step', type=int, help='Single step for --initial-linear mode (with -b)')
    parser.add_argument('--best', action='store_true',
                       help='Use baked_best.pt as the trained checkpoint (with -b)')

    # Comparison options
    parser.add_argument('--initial-linear', action='store_true',
                       help='Compare against linear initialization instead of checkpoint')
    parser.add_argument('--max-abs', type=float, default=2.0,
                       help='Max abs value for linear init (default: 2.0)')
    parser.add_argument('--lut-size', type=int, default=16,
                       help='LUT size for linear init (default: 16)')

    # Output options
    parser.add_argument('--summary', action='store_true',
                       help='Only print summary, skip detailed analysis')
    parser.add_argument('--top-n', type=int, default=10,
                       help='Number of top/bottom LUTs to show (default: 10)')
    parser.add_argument('--output', '-o', type=str,
                       help='Save detailed analysis to JSON file')

    # Path options
    parser.add_argument('--drive-root', help='Google Drive root (auto-detect if not specified)')
    parser.add_argument('--local-root', default='runs', help='Local runs directory (default: runs)')

    args = parser.parse_args()

    # Resolve checkpoint paths
    gdrive_root = Path(args.drive_root) if args.drive_root else find_gdrive_root()

    if args.base_dir:
        # Google Drive mode
        run_name = args.base_dir
        if run_name.startswith("runs/"):
            run_name = run_name[5:]

        print(f"Run: {run_name}")
        if gdrive_root:
            print(f"Google Drive: {gdrive_root / run_name}")

        if args.initial_linear:
            # Compare vs linear init
            step = args.step or args.step2
            if args.best:
                ckpt_path = resolve_checkpoint_path(run_name, best=True, gdrive_root=gdrive_root, local_root=args.local_root)
            elif step:
                ckpt_path = resolve_checkpoint_path(run_name, step=step, gdrive_root=gdrive_root, local_root=args.local_root)
            else:
                parser.error("With -b and --initial-linear, specify --step or --best")

            trained_luts = load_luts_from_checkpoint(str(ckpt_path))
            print(f"\nComparing against linear initialization (max_abs={args.max_abs})")
            linear_lut = generate_linear_luts(args.lut_size, args.max_abs)
            initial_luts = {k: linear_lut.clone() for k in trained_luts.keys()}
        else:
            # Compare two checkpoints
            if args.best:
                ckpt2_path = resolve_checkpoint_path(run_name, best=True, gdrive_root=gdrive_root, local_root=args.local_root)
            elif args.step2:
                ckpt2_path = resolve_checkpoint_path(run_name, step=args.step2, gdrive_root=gdrive_root, local_root=args.local_root)
            else:
                parser.error("With -b, specify --step1/--step2 or --best")

            if args.step1:
                ckpt1_path = resolve_checkpoint_path(run_name, step=args.step1, gdrive_root=gdrive_root, local_root=args.local_root)
            else:
                parser.error("With -b (non --initial-linear), specify --step1")

            initial_luts = load_luts_from_checkpoint(str(ckpt1_path))
            trained_luts = load_luts_from_checkpoint(str(ckpt2_path))
    else:
        # Direct checkpoint path mode
        if args.initial_linear:
            if not args.checkpoint1:
                parser.error("checkpoint1 is required")
            trained_luts = load_luts_from_checkpoint(args.checkpoint1)
            print(f"\nComparing against linear initialization (max_abs={args.max_abs})")
            linear_lut = generate_linear_luts(args.lut_size, args.max_abs)
            initial_luts = {k: linear_lut.clone() for k in trained_luts.keys()}
        else:
            if not args.checkpoint1 or not args.checkpoint2:
                parser.error("checkpoint1 and checkpoint2 are required unless -b or --initial-linear is used")
            initial_luts = load_luts_from_checkpoint(args.checkpoint1)
            trained_luts = load_luts_from_checkpoint(args.checkpoint2)

    # Find common keys
    common_keys = set(initial_luts.keys()) & set(trained_luts.keys())
    print(f"\nComparing {len(common_keys)} common LUT tensors")

    if not common_keys:
        print("ERROR: No common LUT keys found!")
        return 1

    # Analyze each LUT
    all_stats = []
    for key in sorted(common_keys):
        stats = analyze_single_lut(initial_luts[key], trained_luts[key], key)
        all_stats.append(stats)

    # Print summary
    print_summary(all_stats, args.top_n)

    # Detailed per-bin analysis
    if not args.summary:
        analyze_bin_patterns(all_stats)

    # Save to JSON if requested
    if args.output:
        import json
        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        output_data = {
            'summary': {
                'total_luts': len(all_stats),
                'mean_rms_change': float(np.mean([s['rms_change'] for s in all_stats])),
                'max_abs_change': float(max(s['max_abs_change'] for s in all_stats)),
            },
            'per_lut': [{k: convert(v) for k, v in s.items()} for s in all_stats],
        }

        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nDetailed analysis saved to: {args.output}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
