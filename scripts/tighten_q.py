#!/usr/bin/env python3
"""
Tighten _Q buffers in V2 QAT checkpoints.

Recalculates _Q to be consistent with current scales (A/B) and magnitudes (G).
This "tightens" quantization after phases that change A/B (e.g., postfreeze_AB).

Algorithm:
    1. S = module._compute_full_scales()  # [out, in]
    2. Q_target = W_ref / S
    3. Snap Q_target to nearest LUT values
    4. Update module._Q

W_ref Source:
    --ref baseline (default): Use fresh HF weights (recommended, avoids polluted weights)
    --ref checkpoint: Use module.weight from checkpoint (with sanity checks)

Usage:
    # MLP only (baseline W_ref - recommended)
    python scripts/tighten_q.py \
        --v2-checkpoint runs/SR-011_q4_a4_r32_postfreeze_AB/v2_q4a4_r32_fp32_*.pt \
        --output runs/SR-011_q4_a4_r32_postfreeze_AB/tightQ_mlp.pt \
        --ref baseline \
        --scope mlp \
        --clamp-q

    # Dry run (QC only)
    python scripts/tighten_q.py \
        --v2-checkpoint runs/SR-011_q4_a4_r32_postfreeze_AB/v2_q4a4_r32_fp32_*.pt \
        --output /dev/null \
        --scope all \
        --dry-run
"""

import argparse
import json
import os
import sys
from pathlib import Path

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
# HELPER FUNCTIONS
# =============================================================================

def quantize_chunked(Q_target: torch.Tensor, lut: torch.Tensor, chunk_size: int = 4096) -> torch.Tensor:
    """Memory-safe nearest-LUT assignment.

    Args:
        Q_target: Target values to quantize [out, in]
        lut: LUT values [lut_size]
        chunk_size: Process this many elements at once

    Returns:
        Indices tensor with same shape as Q_target
    """
    flat = Q_target.flatten()
    indices = torch.empty_like(flat, dtype=torch.long)

    for i in range(0, flat.numel(), chunk_size):
        end = min(i + chunk_size, flat.numel())
        chunk = flat[i:end]
        # dist[j, k] = |chunk[j] - lut[k]|
        dist = (chunk.unsqueeze(-1) - lut).abs()
        indices[i:end] = dist.argmin(dim=-1)

    return indices.view_as(Q_target)


def load_baseline_weights(model_id: str) -> dict:
    """Load fresh HF weights for W_ref (not quantized).

    Returns:
        Mapping: module path -> weight tensor
    """
    print(f"  Loading baseline weights from {model_id}...")
    baseline = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )

    W_ref_map = {}
    for name, module in baseline.named_modules():
        if hasattr(module, 'weight') and isinstance(module.weight, nn.Parameter):
            W_ref_map[name] = module.weight.data.clone()

    del baseline
    return W_ref_map


def map_v2_to_baseline_name(v2_name: str) -> str:
    """Map V2 module path to baseline module path.

    For Qwen: paths are identical (no transformation needed).
    """
    return v2_name  # Identity mapping


def sanity_check_weight(weight: torch.Tensor, lut_size: int) -> tuple:
    """Check if weight looks like fresh FP weights (not quantized garbage).

    Returns:
        (ok: bool, warnings: list[str])
    """
    unique = weight.unique().numel()
    range_ok = weight.abs().max() > 1.0  # Not already normalized
    unique_ok = unique > lut_size * 10   # Not already quantized

    warnings = []
    if not range_ok:
        warnings.append(f"weight range looks normalized (max={weight.abs().max():.3f})")
    if not unique_ok:
        warnings.append(f"weight has few unique values ({unique} vs lut_size={lut_size})")

    return len(warnings) == 0, warnings


def matches_scope(name: str, scope: str, skip_k_proj: bool, layer_indices: list) -> bool:
    """Check if module name matches the target scope.

    Args:
        name: Module name (e.g., 'model.layers.5.mlp.gate_proj')
        scope: 'mlp', 'attn', or 'all'
        skip_k_proj: Skip k_proj modules
        layer_indices: List of layer indices to process (None = all)

    Returns:
        True if module should be processed
    """
    # Check layer index if specified
    if layer_indices is not None:
        # Extract layer index from name like 'model.layers.5.mlp.gate_proj'
        parts = name.split('.')
        layer_idx = None
        for i, part in enumerate(parts):
            if part == 'layers' and i + 1 < len(parts):
                try:
                    layer_idx = int(parts[i + 1])
                except ValueError:
                    pass
        if layer_idx is not None and layer_idx not in layer_indices:
            return False

    # Check k_proj
    if skip_k_proj and 'k_proj' in name:
        return False

    # Check scope
    if scope == 'all':
        return True
    elif scope == 'mlp':
        return '.mlp.' in name or name.endswith('.mlp')
    elif scope == 'attn':
        return '.self_attn.' in name or '_attn.' in name or 'attention' in name.lower()

    return True


def load_config_json(ckpt_dir: Path) -> dict:
    """Load config.json from checkpoint directory if it exists."""
    config_path = ckpt_dir / 'config.json'
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return {}


# =============================================================================
# CORE ALGORITHM
# =============================================================================

@torch.no_grad()
def tighten_q_layer(
    module,  # AnemllQATLinearV2
    W_ref: torch.Tensor,
    eps: float = 1e-4,
    clamp_q: bool = True,
    chunk_size: int = 4096,
) -> dict:
    """Tighten _Q for a single V2 layer.

    Args:
        module: V2 layer with _Q to tighten
        W_ref: Reference weights (baseline HF, not checkpoint!)
        eps: Floor for scale magnitude
        clamp_q: Clamp Q_target to LUT range
        chunk_size: Chunk size for memory-safe quantization

    Returns:
        QC metrics dictionary
    """
    # 1. Get full scales using module's own method
    S = module._compute_full_scales()  # [out, in]

    # Ensure W_ref is on the same device as S
    W_ref = W_ref.to(S.device)

    # Validate shapes match
    assert S.shape == W_ref.shape, f"S.shape {S.shape} != W_ref.shape {W_ref.shape}"

    # 2. Guard against small/zero scales (fixed sign handling)
    sign = torch.sign(S)
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)  # Fix S==0 case
    small_mask = S.abs() < eps
    S_safe = torch.where(small_mask, sign * eps, S)
    num_small = small_mask.sum().item()

    # 3. Compute Q_target = W_ref / S
    Q_target = W_ref / S_safe

    # 4. Clamp to LUT range
    lut = module.lut
    lut_min, lut_max = lut.min(), lut.max()
    if clamp_q:
        Q_target = Q_target.clamp(lut_min, lut_max)

    # 5. Memory-safe nearest-LUT assignment (chunked)
    indices_new = quantize_chunked(Q_target, lut, chunk_size)
    _Q_new = lut[indices_new]

    # 6. Preserve original _Q shape (may be [out, in] or [out, in, 1, 1])
    _Q_old = module._Q
    original_shape = _Q_old.shape
    _Q_new_shaped = _Q_new.view(original_shape)
    indices_new_shaped = indices_new.view(original_shape)

    # 7. QC metrics
    changed_mask = (_Q_old != _Q_new_shaped)
    num_changed = changed_mask.sum().item()
    total = _Q_old.numel()

    # Reconstruction error (MSE - Mean Squared Error)
    W_eff_old = _Q_old.view(W_ref.shape) * S_safe
    W_eff_new = _Q_new.view(W_ref.shape) * S_safe
    mse_old = ((W_ref - W_eff_old) ** 2).mean().item()
    mse_new = ((W_ref - W_eff_new) ** 2).mean().item()

    # Saturation metrics
    flat_Q = _Q_new.flatten()
    pct_at_min = (flat_Q == lut_min).float().mean().item() * 100
    pct_at_max = (flat_Q == lut_max).float().mean().item() * 100
    unique_Q = flat_Q.unique().numel()

    # 8. Update buffers
    module._Q.copy_(_Q_new_shaped)

    # Only update _indices if it exists, is not None, and has matching numel
    if hasattr(module, '_indices') and module._indices is not None:
        if module._indices.numel() == indices_new_shaped.numel():
            module._indices.copy_(indices_new_shaped.view(module._indices.shape))

    return {
        'num_changed': num_changed,
        'total': total,
        'pct_changed': 100 * num_changed / total,
        'num_small_S': num_small,
        'mse_old': mse_old,
        'mse_new': mse_new,
        'mse_delta': mse_new - mse_old,
        'pct_at_lut_min': pct_at_min,
        'pct_at_lut_max': pct_at_max,
        'unique_Q': unique_Q,
        'lut_size': lut.numel(),
    }


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Tighten _Q buffers in V2 QAT checkpoints',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required
    parser.add_argument('--v2-checkpoint', type=str, required=True,
                        help='Input V2 checkpoint path')
    parser.add_argument('--output', type=str, required=True,
                        help='Output checkpoint path')

    # W_ref source
    parser.add_argument('--ref', type=str, default='baseline', choices=['baseline', 'checkpoint'],
                        help='W_ref source: baseline (default, recommended) or checkpoint')

    # Scope
    parser.add_argument('--scope', type=str, default='all', choices=['mlp', 'attn', 'all'],
                        help='Layers to process: mlp, attn, or all (default: all)')
    parser.add_argument('--layers', type=str, default=None,
                        help='Comma-separated layer indices to process (optional)')
    parser.add_argument('--skip-k-proj', action='store_true',
                        help='Skip k_proj modules')

    # Algorithm params
    parser.add_argument('--eps', type=float, default=1e-4,
                        help='Denominator floor for S (default: 1e-4)')
    parser.add_argument('--clamp-q', action='store_true',
                        help='Clamp Q_target to LUT range before assignment')
    parser.add_argument('--chunk-size', type=int, default=4096,
                        help='Chunk size for quantization (default: 4096)')

    # Output options
    parser.add_argument('--dry-run', action='store_true',
                        help='Only compute QC, don\'t write checkpoint')
    parser.add_argument('--force', action='store_true',
                        help='Skip sanity checks on W_ref')
    parser.add_argument('--qc-output', type=str, default=None,
                        help='Path to save QC JSON (optional)')

    # Model config
    parser.add_argument('--model-id', type=str, default='Qwen/Qwen3-0.6B',
                        help='Base model name (default: Qwen/Qwen3-0.6B)')
    parser.add_argument('--config', type=str, default=None,
                        choices=['q2a4', 'q4a4', 'q4a4_r32', 'q4_r32', 'q2a2'],
                        help='Config preset (auto-detect from config.json if not specified)')
    parser.add_argument('--mlp-lut', type=int, default=None,
                        help='Override MLP LUT size')
    parser.add_argument('--mlp-rank', type=int, default=None,
                        help='Override MLP scale rank')
    parser.add_argument('--attn-lut', type=int, default=None,
                        help='Override Attention LUT size')
    parser.add_argument('--attn-rank', type=int, default=None,
                        help='Override Attention scale rank')

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("TIGHTEN Q")
    print("=" * 60)
    print(f"V2 checkpoint: {args.v2_checkpoint}")
    print(f"Output: {args.output}")
    print(f"W_ref source: {args.ref}")
    print(f"Scope: {args.scope}")
    print()

    # Parse layer indices
    layer_indices = None
    if args.layers:
        layer_indices = [int(x.strip()) for x in args.layers.split(',')]
        print(f"Layer indices: {layer_indices}")

    # Load config from checkpoint dir
    ckpt_path = Path(args.v2_checkpoint)
    ckpt_config = load_config_json(ckpt_path.parent)

    # Determine config preset
    if args.config:
        preset = CONFIG_PRESETS[args.config]
    elif 'config' in ckpt_config:
        preset_name = ckpt_config.get('config', 'q4a4_r32')
        preset = CONFIG_PRESETS.get(preset_name, CONFIG_PRESETS['q4a4_r32'])
    else:
        preset = CONFIG_PRESETS['q4a4_r32']

    # Apply overrides
    mlp_lut = args.mlp_lut or preset['mlp_lut']
    mlp_rank = args.mlp_rank or preset['mlp_rank']
    attn_lut = args.attn_lut or preset['attn_lut']
    attn_rank = args.attn_rank or preset['attn_rank']

    print(f"Config: MLP lut={mlp_lut} rank={mlp_rank}, Attn lut={attn_lut} rank={attn_rank}")
    print()

    # Import V2 classes
    from qat_lora import AnemllQuantConfigV2, replace_linear_with_anemll_v2
    from qat_lora.ane_qat_linear_v2 import AnemllQATLinearV2

    # Load baseline HF weights for W_ref (if --ref baseline)
    W_ref_map = None
    if args.ref == 'baseline':
        W_ref_map = load_baseline_weights(args.model_id)
        print(f"  Loaded {len(W_ref_map)} weight tensors from baseline")

    # Load base model + convert to V2
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )

    # Create V2 configs (group_size only matters for init, not for loaded checkpoint)
    v2_mlp_config = AnemllQuantConfigV2(
        lut_size=mlp_lut,
        scale_rank=mlp_rank,
        group_size=32,  # V2 has no grouping in scale_B
        force_positive_scales=False,
        magnitude_activation='identity',
    )
    v2_attn_config = AnemllQuantConfigV2(
        lut_size=attn_lut,
        scale_rank=attn_rank,
        group_size=32,  # V2 has no grouping in scale_B
        force_positive_scales=False,
        magnitude_activation='identity',
    )

    # Replace with V2 layers
    replace_linear_with_anemll_v2(
        model,
        mlp_config=v2_mlp_config,
        attn_config=v2_attn_config,
        quantize_attn=True,
        quantize_lm_head=False,
        skip_init=True,
    )

    # Load checkpoint FULLY (DO NOT filter .weight keys - would clobber embeddings/norms!)
    print(f"Loading checkpoint: {args.v2_checkpoint}")
    state_dict = torch.load(args.v2_checkpoint, map_location='cpu', weights_only=False)
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    model.load_state_dict(state_dict, strict=False)

    # Manually load _Q buffers (None buffers don't load automatically)
    q_loaded = 0
    for name, m in model.named_modules():
        if isinstance(m, AnemllQATLinearV2):
            q_key = f"{name}._Q"
            if q_key in state_dict and m._Q is None:
                m.register_buffer("_Q", state_dict[q_key])
                q_loaded += 1
    if q_loaded > 0:
        print(f"  Manually loaded {q_loaded} _Q buffers")

    # Validate _Q exists for layers we want to tighten
    # (don't auto-freeze - it uses wrong W_ref!)
    missing_Q_layers = []
    for name, module in model.named_modules():
        if isinstance(module, AnemllQATLinearV2):
            if matches_scope(name, args.scope, args.skip_k_proj, layer_indices):
                if module._Q is None:
                    missing_Q_layers.append(name)

    if missing_Q_layers:
        print(f"\nERROR: {len(missing_Q_layers)} layers have _Q=None:")
        for n in missing_Q_layers[:5]:
            print(f"  {n}")
        if len(missing_Q_layers) > 5:
            print(f"  ... and {len(missing_Q_layers) - 5} more")
        print(f"\nRun freeze_Q on your checkpoint first, or use a checkpoint that contains _Q.")
        sys.exit(1)

    # Tighten _Q for matching layers
    print("\nTightening _Q...")
    qc_results = {}
    worse_count = 0
    processed = 0

    for name, module in model.named_modules():
        if isinstance(module, AnemllQATLinearV2):
            if not matches_scope(name, args.scope, args.skip_k_proj, layer_indices):
                continue

            # Get W_ref for this layer
            if args.ref == 'baseline':
                baseline_name = map_v2_to_baseline_name(name)
                W_ref = W_ref_map.get(baseline_name)
                if W_ref is None:
                    print(f"  WARNING: No baseline weight for {name}, skipping")
                    continue
            else:
                # Use checkpoint weight with sanity check
                W_ref = module.weight.data
                ok, warnings = sanity_check_weight(W_ref, module.lut.numel())
                if not ok and not args.force:
                    print(f"  WARNING: {name}: {warnings}")
                    print(f"           Use --force to proceed anyway, or use --ref baseline")
                    continue

            # Tighten
            qc = tighten_q_layer(
                module,
                W_ref,
                eps=args.eps,
                clamp_q=args.clamp_q,
                chunk_size=args.chunk_size,
            )
            qc_results[name] = qc
            processed += 1

            if qc['mse_delta'] > 1e-12:  # MSE got worse
                worse_count += 1

            sat_str = f"sat={qc['pct_at_lut_min']:.1f}%/{qc['pct_at_lut_max']:.1f}%"
            delta_str = f"{qc['mse_delta']:+.2e}" if qc['mse_delta'] != 0 else "0"
            print(f"  {name}: {qc['pct_changed']:.1f}% changed, "
                  f"MSE {qc['mse_old']:.2e}->{qc['mse_new']:.2e} ({delta_str}), {sat_str}")

    # Summary & QC aggregation
    if qc_results:
        total_changed = sum(r['num_changed'] for r in qc_results.values())
        total_params = sum(r['total'] for r in qc_results.values())
        avg_mse_delta = sum(r['mse_delta'] for r in qc_results.values()) / len(qc_results)

        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Layers tightened: {len(qc_results)}")
        print(f"Total Q changed:  {total_changed:,} / {total_params:,} ({100*total_changed/total_params:.2f}%)")
        print(f"Avg MSE delta:    {avg_mse_delta:+.2e}")

        # Warn if high saturation
        high_sat_layers = [n for n, r in qc_results.items()
                          if r['pct_at_lut_min'] > 5 or r['pct_at_lut_max'] > 5]
        if high_sat_layers:
            print(f"\nWARNING: {len(high_sat_layers)} layers have >5% saturation")

        # Abort if >60% of layers got worse MSE
        if worse_count > len(qc_results) * 0.6:
            print(f"\nERROR: {worse_count}/{len(qc_results)} layers have worse MSE!")
            print(f"       Something is wrong. Check W_ref source and scale computation.")
            if not args.force:
                sys.exit(1)

    else:
        print("\nNo layers were tightened!")
        total_changed = 0
        total_params = 0
        avg_mse_delta = 0

    # Save checkpoint
    if not args.dry_run and args.output != '/dev/null':
        print(f"\nSaving checkpoint: {args.output}")
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.output)
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  Saved: {args.output} ({size_mb:.1f} MB)")
    elif args.dry_run:
        print("\n(Dry run - no checkpoint saved)")

    # Save QC JSON
    if args.qc_output:
        qc_data = {
            'v2_checkpoint': str(args.v2_checkpoint),
            'ref': args.ref,
            'scope': args.scope,
            'eps': args.eps,
            'clamp_q': args.clamp_q,
            'total_changed': total_changed,
            'total_params': total_params,
            'pct_changed': 100 * total_changed / total_params if total_params > 0 else 0,
            'avg_mse_delta': avg_mse_delta,
            'worse_count': worse_count,
            'layers': qc_results,
        }
        with open(args.qc_output, 'w') as f:
            json.dump(qc_data, f, indent=2)
        print(f"QC saved: {args.qc_output}")

    print("\nDone!")


if __name__ == '__main__':
    main()
