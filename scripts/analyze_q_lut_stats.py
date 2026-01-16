#!/usr/bin/env python3
"""
Analyze per-layer Q/LUT distribution statistics for V2 QAT checkpoints.

Computes statistics in two spaces for each AnemllQATLinearV2 layer:
1. Normalized Q space (Q_eff = W_eff / S) - what LUT actually represents
2. Effective weight space (W_eff = Q * S) - what the model uses

Outputs CSV or JSON with metrics useful for LUT design decisions:
- LUT health: unique values, monotonicity, range
- Saturation: percentage of Q hitting LUT endpoints
- Distribution: percentiles, mean, std
- Shape: tail_ratio, center_ratio for LUT family selection

Usage:
    # Basic usage
    python scripts/analyze_q_lut_stats.py checkpoint.pt --output stats.csv

    # With config preset
    python scripts/analyze_q_lut_stats.py checkpoint.pt --config q2a4 -o stats.json

    # MLP only
    python scripts/analyze_q_lut_stats.py checkpoint.pt --scope mlp --output mlp_stats.csv

    # Verbose output
    python scripts/analyze_q_lut_stats.py checkpoint.pt --verbose
"""

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass, asdict, fields
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Dict, Optional

REPO_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_DIR))

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


# =============================================================================
# CONFIG PRESETS (from train_v2_simple.py)
# =============================================================================

CONFIG_PRESETS = {
    'q2a4': {'mlp_lut': 4, 'mlp_rank': 32, 'attn_lut': 16, 'attn_rank': 8},
    'q4a4': {'mlp_lut': 16, 'mlp_rank': 4, 'attn_lut': 16, 'attn_rank': 4},
    'q4a4_r32': {'mlp_lut': 16, 'mlp_rank': 32, 'attn_lut': 16, 'attn_rank': 32},
    'q4_r32': {'mlp_lut': 16, 'mlp_rank': 32, 'attn_lut': 16, 'attn_rank': 32},
    'q2a2': {'mlp_lut': 4, 'mlp_rank': 32, 'attn_lut': 4, 'attn_rank': 32},
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class LayerQStats:
    """Per-layer Q/LUT statistics."""
    layer_idx: int
    layer_name: str
    kind: str  # 'mlp.gate', 'mlp.up', 'mlp.down', 'attn.q', 'attn.k', 'attn.v', 'attn.o'
    out_features: int
    in_features: int

    # LUT health
    lut_size: int
    lut_unique_fp16: int
    lut_monotonic_fp16: bool
    lut_min_abs: float
    lut_max_abs: float

    # Saturation (percentage of Q at LUT endpoints)
    pct_Q_at_min: float
    pct_Q_at_max: float
    pct_Qeff_outside_lut: float

    # Q_eff distribution (|W_eff / S_safe|)
    qeff_abs_p50: float
    qeff_abs_p90: float
    qeff_abs_p95: float
    qeff_abs_p99: float
    qeff_abs_p99_9: float
    qeff_abs_max: float
    qeff_mean_abs: float
    qeff_std: float

    # W_eff distribution
    weff_abs_p50: float
    weff_abs_p90: float
    weff_abs_p95: float
    weff_abs_p99: float
    weff_abs_p99_9: float
    weff_abs_max: float

    # Shape metrics (for LUT family selection)
    tail_ratio: float  # p99 / p90 - high = heavy tail
    center_ratio: float  # p50 / p90 - low = mass near zero

    # Critical: Is max_abs too small? (Q_eff_target = W_eff/S_safe distribution)
    qeff_abs_p999: float  # 99.9th percentile of |Q_eff_target|
    p999_over_maxabs: float  # qeff_abs_p999 / lut_max_abs - if >1.0, widen max_abs
    pct_small_S: float  # % of entries where |S| < eps (division artifact risk)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def parse_layer_kind(name: str) -> Tuple[int, str]:
    """Extract layer index and kind from module path.

    Examples:
        'model.layers.5.mlp.gate_proj' -> (5, 'mlp.gate')
        'model.layers.12.self_attn.q_proj' -> (12, 'attn.q')
    """
    # Extract layer index
    layer_match = re.search(r'\.layers\.(\d+)\.', name)
    layer_idx = int(layer_match.group(1)) if layer_match else -1

    # Map projection name to kind
    kind_map = {
        'gate_proj': 'mlp.gate',
        'up_proj': 'mlp.up',
        'down_proj': 'mlp.down',
        'q_proj': 'attn.q',
        'k_proj': 'attn.k',
        'v_proj': 'attn.v',
        'o_proj': 'attn.o',
    }
    for suffix, kind in kind_map.items():
        if name.endswith(suffix):
            return layer_idx, kind
    return layer_idx, 'unknown'


def matches_scope(name: str, scope: str) -> bool:
    """Check if module name matches the target scope."""
    if scope == 'all':
        return True
    elif scope == 'mlp':
        return '.mlp.' in name
    elif scope == 'attn':
        return '.self_attn.' in name or '_attn.' in name
    return True


def load_config_json(ckpt_dir: Path) -> dict:
    """Load config.json from checkpoint directory if it exists."""
    for name in ['config.json', 'v2_config.json']:
        config_path = ckpt_dir / name
        if config_path.exists():
            with open(config_path) as f:
                return json.load(f)
    return {}


@torch.no_grad()
def compute_layer_stats(
    module,  # AnemllQATLinearV2
    name: str,
    eps: float = 1e-4,
) -> LayerQStats:
    """Compute Q/LUT statistics for a single V2 layer.

    Key computations:
    1. lut_fp16 = module.lut.half().float()  # FP16-snapped LUT
    2. Q_current = module._Q (or lut[indices])
    3. S = module._compute_full_scales()
    4. S_safe = S where |S| >= eps else sign(S) * eps
    5. W_eff = Q_current * S
    6. Q_eff = W_eff / S_safe (target Q values)
    """
    layer_idx, kind = parse_layer_kind(name)

    # Get LUT and compute FP16 version
    # CRITICAL: use get_lut() which handles trainable LUT case
    lut = module.get_lut().float().cpu() if hasattr(module, 'get_lut') else module.lut.float().cpu()
    lut_fp16 = lut.half().float()
    lut_unique_fp16 = int(torch.unique(lut_fp16).numel())
    lut_monotonic_fp16 = bool((lut_fp16[1:] > lut_fp16[:-1]).all().item())
    lut_min_abs = float(lut_fp16.abs().min().item())
    lut_max_abs = float(lut_fp16.abs().max().item())

    # Get Q_current
    if module._Q is not None:
        Q_current = module._Q.float().cpu()
    elif module._indices is not None:
        # CRITICAL: move _indices to CPU before indexing CPU lut
        Q_current = lut[module._indices.long().cpu()]
    else:
        raise ValueError(f"Layer {name} has no _Q or _indices")

    # Compute scales
    S = module._compute_full_scales().float().cpu()

    # Safe scales (avoid division by zero)
    sign = torch.sign(S)
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    small_mask = S.abs() < eps
    S_safe = torch.where(small_mask, sign * eps, S)

    # Compute W_eff and Q_eff
    W_eff = Q_current * S
    Q_eff = W_eff / S_safe

    # Flatten for stats
    Q_flat = Q_current.flatten()
    Q_eff_flat = Q_eff.flatten()
    Q_eff_abs = Q_eff_flat.abs()
    W_eff_abs = W_eff.flatten().abs()

    # Saturation stats (using original LUT range)
    lut_min = lut.min().item()
    lut_max = lut.max().item()
    pct_Q_at_min = float((Q_flat == lut_min).float().mean().item() * 100)
    pct_Q_at_max = float((Q_flat == lut_max).float().mean().item() * 100)
    pct_outside = float(((Q_eff_flat < lut_min) | (Q_eff_flat > lut_max)).float().mean().item() * 100)

    # Percentile helper
    def percentiles(t: torch.Tensor, ps: List[float]) -> List[float]:
        return [float(torch.quantile(t.float(), p / 100).item()) for p in ps]

    qeff_ps = percentiles(Q_eff_abs, [50, 90, 95, 99, 99.9])
    weff_ps = percentiles(W_eff_abs, [50, 90, 95, 99, 99.9])

    # Shape metrics
    tail_ratio = qeff_ps[3] / qeff_ps[1] if qeff_ps[1] > 0 else float('inf')
    center_ratio = qeff_ps[0] / qeff_ps[1] if qeff_ps[1] > 0 else 0

    # Critical: Is max_abs too small?
    # p999_over_maxabs > 1.0 means the 99.9th percentile of Q_eff exceeds current max_abs
    qeff_abs_p999 = qeff_ps[4]  # 99.9th percentile
    p999_over_maxabs = qeff_abs_p999 / lut_max_abs if lut_max_abs > 0 else float('inf')

    # Percentage of entries where |S| < eps (division artifacts)
    pct_small_S = float(small_mask.float().mean().item() * 100)

    return LayerQStats(
        layer_idx=layer_idx,
        layer_name=name,
        kind=kind,
        out_features=module.out_features,
        in_features=module.in_features,
        lut_size=lut.numel(),
        lut_unique_fp16=lut_unique_fp16,
        lut_monotonic_fp16=lut_monotonic_fp16,
        lut_min_abs=lut_min_abs,
        lut_max_abs=lut_max_abs,
        pct_Q_at_min=pct_Q_at_min,
        pct_Q_at_max=pct_Q_at_max,
        pct_Qeff_outside_lut=pct_outside,
        qeff_abs_p50=qeff_ps[0],
        qeff_abs_p90=qeff_ps[1],
        qeff_abs_p95=qeff_ps[2],
        qeff_abs_p99=qeff_ps[3],
        qeff_abs_p99_9=qeff_ps[4],
        qeff_abs_max=float(Q_eff_abs.max().item()),
        qeff_mean_abs=float(Q_eff_abs.mean().item()),
        qeff_std=float(Q_eff_abs.std().item()),
        weff_abs_p50=weff_ps[0],
        weff_abs_p90=weff_ps[1],
        weff_abs_p95=weff_ps[2],
        weff_abs_p99=weff_ps[3],
        weff_abs_p99_9=weff_ps[4],
        weff_abs_max=float(W_eff_abs.max().item()),
        tail_ratio=tail_ratio,
        center_ratio=center_ratio,
        qeff_abs_p999=qeff_abs_p999,
        p999_over_maxabs=p999_over_maxabs,
        pct_small_S=pct_small_S,
    )


def suggest_lut_family(stats: LayerQStats) -> str:
    """Suggest LUT family based on layer statistics.

    Returns one of: 'uniform', 'dense-center', 'heavy-tail', 'quantile'

    Key insight: endpoint saturation (pct_Q_at_min/max) measures stored _Q values,
    NOT whether the initializer needed values outside max_abs. Use p999_over_maxabs
    to determine if max_abs should be widened.
    """
    # CRITICAL: p999_over_maxabs > 1.0 means Q_eff p99.9 exceeds current LUT range
    # This is the proper signal to widen max_abs
    if stats.p999_over_maxabs > 1.05:
        return 'heavy-tail (widen max_abs)'

    # Significant values outside current LUT range
    if stats.pct_Qeff_outside_lut > 0.5:
        return 'heavy-tail (widen max_abs)'

    # High tail ratio -> heavy-tail LUT
    if stats.tail_ratio > 2.0:
        return 'heavy-tail'

    # Low center ratio (mass near zero) -> dense-center
    if stats.center_ratio < 0.3:
        return 'dense-center'

    # Default: uniform is fine
    return 'uniform'


def save_csv(stats_list: List[LayerQStats], output_path: Path):
    """Save statistics to CSV file."""
    if not stats_list:
        return

    fieldnames = [f.name for f in fields(LayerQStats)]

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for stats in stats_list:
            writer.writerow(asdict(stats))


def save_json(
    stats_list: List[LayerQStats],
    output_path: Path,
    checkpoint: str,
    model_id: str,
    scope: str,
):
    """Save statistics to JSON file with metadata."""
    # Compute summary
    mlp_layers = [s for s in stats_list if s.kind.startswith('mlp')]
    attn_layers = [s for s in stats_list if s.kind.startswith('attn')]

    # Legacy saturation metrics
    avg_saturation = sum(s.pct_Q_at_min + s.pct_Q_at_max for s in stats_list) / len(stats_list) if stats_list else 0
    high_sat_layers = [s for s in stats_list if s.pct_Q_at_min > 1 or s.pct_Q_at_max > 1]

    # Critical p999_over_maxabs metrics
    p999_ratios = [s.p999_over_maxabs for s in stats_list] if stats_list else [0]
    need_widen = [s for s in stats_list if s.p999_over_maxabs > 1.0]
    small_s_layers = [s for s in stats_list if s.pct_small_S > 1.0]

    output = {
        'metadata': {
            'checkpoint': str(checkpoint),
            'model_id': model_id,
            'scope': scope,
            'timestamp': datetime.now().isoformat(),
        },
        'summary': {
            'total_layers': len(stats_list),
            'mlp_layers': len(mlp_layers),
            'attn_layers': len(attn_layers),
            # Critical: p999_over_maxabs is the proper signal for widening max_abs
            'p999_over_maxabs_max': round(max(p999_ratios), 4),
            'p999_over_maxabs_avg': round(sum(p999_ratios) / len(p999_ratios), 4) if p999_ratios else 0,
            'layers_need_widen_maxabs': len(need_widen),
            'layers_with_small_S': len(small_s_layers),
            # Legacy metrics (less informative)
            'avg_endpoint_saturation_pct': round(avg_saturation, 2),
            'layers_with_high_endpoint_saturation': len(high_sat_layers),
        },
        'layers': [asdict(s) for s in stats_list],
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Analyze per-layer Q/LUT distribution statistics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument('checkpoint', help='V2 checkpoint path (.pt file)')
    parser.add_argument('--config', default=None,
                        help='Config preset (q2a4, q4a4, etc.) or path to config.json')
    parser.add_argument('--scope', choices=['mlp', 'attn', 'all'], default='all',
                        help='Layer scope to analyze (default: all)')
    parser.add_argument('--model-id', default='Qwen/Qwen3-0.6B',
                        help='Base model name (default: Qwen/Qwen3-0.6B)')
    parser.add_argument('--output', '-o', default=None,
                        help='Output path (CSV or JSON based on extension)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print per-layer details')
    parser.add_argument('--eps', type=float, default=1e-4,
                        help='Epsilon for safe division (default: 1e-4)')

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("ANALYZE Q/LUT STATS")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Scope:      {args.scope}")
    print()

    # Load config
    ckpt_path = Path(args.checkpoint)
    ckpt_config = load_config_json(ckpt_path.parent)

    # Determine config preset
    if args.config and args.config in CONFIG_PRESETS:
        preset = CONFIG_PRESETS[args.config]
        print(f"Config:     {args.config} (preset)")
    elif args.config and Path(args.config).exists():
        with open(args.config) as f:
            ckpt_config = json.load(f)
        preset = CONFIG_PRESETS.get(ckpt_config.get('config', 'q4a4_r32'), CONFIG_PRESETS['q4a4_r32'])
        print(f"Config:     {args.config} (file)")
    elif 'config' in ckpt_config:
        preset_name = ckpt_config.get('config', 'q4a4_r32')
        preset = CONFIG_PRESETS.get(preset_name, CONFIG_PRESETS['q4a4_r32'])
        print(f"Config:     {preset_name} (from checkpoint)")
    else:
        preset = CONFIG_PRESETS['q4a4_r32']
        print("Config:     q4a4_r32 (default)")

    mlp_lut = preset['mlp_lut']
    mlp_rank = preset['mlp_rank']
    attn_lut = preset['attn_lut']
    attn_rank = preset['attn_rank']
    print(f"  MLP:      LUT{mlp_lut} rank={mlp_rank}")
    print(f"  Attn:     LUT{attn_lut} rank={attn_rank}")
    print()

    # Import V2 classes
    from qat_lora import AnemllQuantConfigV2, replace_linear_with_anemll_v2
    from qat_lora.ane_qat_linear_v2 import AnemllQATLinearV2

    # Load base model
    print(f"Loading base model: {args.model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )

    # Create V2 configs
    mlp_config = AnemllQuantConfigV2(
        lut_size=mlp_lut,
        scale_rank=mlp_rank,
        force_positive_scales=False,
        magnitude_activation='identity',
    )
    attn_config = AnemllQuantConfigV2(
        lut_size=attn_lut,
        scale_rank=attn_rank,
        force_positive_scales=False,
        magnitude_activation='identity',
    )

    # Replace with V2 layers
    replace_linear_with_anemll_v2(
        model,
        mlp_config=mlp_config,
        attn_config=attn_config,
        quantize_attn=True,
        quantize_lm_head=False,
        skip_init=True,
    )

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    state_dict = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    model.load_state_dict(state_dict, strict=False)

    # Manually load _Q buffers
    q_loaded = 0
    for name, m in model.named_modules():
        if isinstance(m, AnemllQATLinearV2):
            q_key = f"{name}._Q"
            if q_key in state_dict and m._Q is None:
                m.register_buffer("_Q", state_dict[q_key])
                q_loaded += 1
    if q_loaded > 0:
        print(f"  Loaded {q_loaded} _Q buffers")

    # Analyze layers
    print("\nAnalyzing layers...")
    stats_list: List[LayerQStats] = []

    for name, module in model.named_modules():
        if not isinstance(module, AnemllQATLinearV2):
            continue
        if not matches_scope(name, args.scope):
            continue

        # Check for _Q or _indices
        if module._Q is None and module._indices is None:
            print(f"  WARNING: {name} has no _Q or _indices, skipping")
            continue

        try:
            stats = compute_layer_stats(module, name, eps=args.eps)
            stats_list.append(stats)

            if args.verbose:
                suggestion = suggest_lut_family(stats)
                sat_str = f"sat={stats.pct_Q_at_min:.1f}%/{stats.pct_Q_at_max:.1f}%"
                shape_str = f"tail={stats.tail_ratio:.2f} center={stats.center_ratio:.2f}"
                # Critical metrics for max_abs decision
                p999_flag = "⚠️" if stats.p999_over_maxabs > 1.0 else ""
                small_s_flag = "⚠️" if stats.pct_small_S > 1.0 else ""
                print(f"  {name}:")
                print(f"    {sat_str}, outside={stats.pct_Qeff_outside_lut:.2f}%")
                print(f"    p999/max_abs={stats.p999_over_maxabs:.3f}{p999_flag}, small_S={stats.pct_small_S:.2f}%{small_s_flag}")
                print(f"    {shape_str} -> {suggestion}")
        except Exception as e:
            print(f"  ERROR: {name}: {e}")

    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Layers analyzed:    {len(stats_list)}")

    mlp_stats = [s for s in stats_list if s.kind.startswith('mlp')]
    attn_stats = [s for s in stats_list if s.kind.startswith('attn')]
    print(f"  MLP:              {len(mlp_stats)}")
    print(f"  Attention:        {len(attn_stats)}")

    # Critical: p999_over_maxabs summary (the true signal for widening max_abs)
    print(f"\n--- Critical Max_abs Diagnostic ---")
    if stats_list:
        p999_ratios = [s.p999_over_maxabs for s in stats_list]
        max_p999_ratio = max(p999_ratios)
        avg_p999_ratio = sum(p999_ratios) / len(p999_ratios)
        need_widen = [s for s in stats_list if s.p999_over_maxabs > 1.0]

        print(f"p999/max_abs (aggregate): max={max_p999_ratio:.3f}, avg={avg_p999_ratio:.3f}")
        if need_widen:
            print(f"⚠️  {len(need_widen)} layers have p999_over_maxabs > 1.0 (should widen max_abs):")
            for s in sorted(need_widen, key=lambda x: -x.p999_over_maxabs)[:5]:
                print(f"    {s.layer_name}: p999/max_abs={s.p999_over_maxabs:.3f}, p999={s.qeff_abs_p999:.4f}, max_abs={s.lut_max_abs:.3f}")
            if len(need_widen) > 5:
                print(f"    ... and {len(need_widen) - 5} more")
        else:
            print(f"✅ All layers have p999_over_maxabs <= 1.0 (max_abs is adequate)")

    # pct_small_S summary (division artifact risk)
    if stats_list:
        small_s_layers = [s for s in stats_list if s.pct_small_S > 1.0]
        if small_s_layers:
            print(f"\n⚠️  {len(small_s_layers)} layers have >1% small |S| entries (quantile LUT may be polluted):")
            for s in sorted(small_s_layers, key=lambda x: -x.pct_small_S)[:3]:
                print(f"    {s.layer_name}: pct_small_S={s.pct_small_S:.2f}%")
        else:
            print(f"\n✅ All layers have <1% small |S| entries (quantile LUTs safe)")

    # Legacy saturation summary (less informative but still useful)
    print(f"\n--- Legacy Saturation Metrics ---")
    high_sat = [s for s in stats_list if s.pct_Q_at_min > 1 or s.pct_Q_at_max > 1]
    outside_lut = [s for s in stats_list if s.pct_Qeff_outside_lut > 0.5]

    if high_sat:
        print(f"NOTE: {len(high_sat)} layers have >1% endpoint saturation (stored _Q at LUT min/max):")
        print(f"      This does NOT mean max_abs is too small. Check p999_over_maxabs instead.")
        for s in high_sat[:3]:
            print(f"    {s.layer_name}: min={s.pct_Q_at_min:.1f}% max={s.pct_Q_at_max:.1f}%")
        if len(high_sat) > 3:
            print(f"    ... and {len(high_sat) - 3} more")
    else:
        print(f"Endpoint saturation: All layers <1%")

    if outside_lut:
        print(f"\nWARNING: {len(outside_lut)} layers have >0.5% Q_eff outside LUT:")
        for s in outside_lut[:5]:
            print(f"  {s.layer_name}: {s.pct_Qeff_outside_lut:.2f}%")
        if len(outside_lut) > 5:
            print(f"  ... and {len(outside_lut) - 5} more")

    # LUT family suggestions
    suggestions = {}
    for s in stats_list:
        suggestion = suggest_lut_family(s)
        suggestions[suggestion] = suggestions.get(suggestion, 0) + 1

    print(f"\nLUT family suggestions:")
    for family, count in sorted(suggestions.items(), key=lambda x: -x[1]):
        print(f"  {family}: {count} layers")

    # Save output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix.lower() == '.json':
            save_json(stats_list, output_path, args.checkpoint, args.model_id, args.scope)
        else:
            save_csv(stats_list, output_path)

        print(f"\nOutput saved: {output_path}")

    print("\nDone!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
