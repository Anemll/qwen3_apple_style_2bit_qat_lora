#!/usr/bin/env python3
"""
Select best LUT for each layer by testing all candidates and picking optimal per-layer.

Unlike apply_lut_candidates.py which applies the same LUT globally, this script:
1. Tests each LUT candidate on each layer individually
2. Selects the best LUT per layer (by weighted MSE or MAE)
3. Generates a hybrid checkpoint with per-layer optimal LUTs

LUT Families:
    E: Original - existing LUT from checkpoint (baseline to beat!)
    F: Predefined - from init_model_v2.py (fp4_dense, uniform)
    G: K-means - weighted Lloyd's algorithm (often beats hand-crafted grids!)
    A: Uniform - linspace(-M, M, 16)
    B: Dense-center - asinh, tanh, sqrt
    C: Heavy-tail - geometric, power2, power3
    D: Quantile - data-driven from layer's |Q_eff|

Selection Metrics:
    --metric weighted_mse   (default) - S^2-weighted MSE (better PPL correlation)
    --metric mae            - Mean absolute error (original)
    --metric activation_mse - Activation-weighted MSE (requires --imatrix)
    --metric iActMSE        - BEST: iMatrix-weighted activation MSE (requires --imatrix)
                              iActMSE = Σ σ²[i] * S² * (Q_target - Q_quant)²
                              This is the theoretically correct metric for PPL optimization.

Note:
    - checkpoint.pt provides: scales (scale_A, scale_B), current LUT, _Q/_indices
    - OR use --from-scratch to initialize scales via SVD (no checkpoint needed)
    - Original FP32 model (--model-id) provides: target weights for optimization
    - Goal: minimize ||W_orig - Q_new * S|| using best LUT per layer

Usage:
    # FROM SCRATCH - Initialize scales via SVD and find optimal LUTs (recommended)
    python scripts/select_best_lut_per_layer.py --from-scratch -o hybrid.pt --families E,F,G,A,B,C,D

    # From checkpoint - load existing scales, optimize LUTs
    python scripts/select_best_lut_per_layer.py checkpoint.pt -o hybrid.pt --families E,F,G,A,B,C,D

    # With auto max_abs from layer p99.9
    python scripts/select_best_lut_per_layer.py --from-scratch -o hybrid.pt --auto-max-abs

    # Use weighted MSE (default, better PPL correlation)
    python scripts/select_best_lut_per_layer.py checkpoint.pt -o hybrid.pt --metric weighted_mse

    # BEST: Use iActMSE metric (requires imatrix first)
    python scripts/compute_imatrix.py --output imatrix.pt
    python scripts/select_best_lut_per_layer.py checkpoint.pt -o hybrid.pt \
        --metric iActMSE --imatrix imatrix.pt --families E,G

    # With stats output
    python scripts/select_best_lut_per_layer.py checkpoint.pt -o hybrid.pt --output-stats stats.json

    # Dry run (no checkpoint saved)
    python scripts/select_best_lut_per_layer.py checkpoint.pt --dry-run
"""

import argparse
import gc
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

REPO_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_DIR))

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

# Import LUT generation and application functions from apply_lut_candidates
from scripts.apply_lut_candidates import (
    CONFIG_PRESETS,
    snap_to_fp16,
    validate_lut_fp16,
    generate_lut_family_A,
    generate_lut_family_B,
    generate_lut_family_C,
    generate_lut_family_D,
    generate_lut_family_G,  # Weighted k-means (asymmetric, eval-only)
    generate_lut_family_H_symkmeans,  # Symmetric k-means (trainable-compatible)
    compute_weighted_mse,   # S^2-weighted MSE
    compute_activation_weighted_mse,  # Activation-weighted MSE (Option A)
    compute_iActMSE,  # iMatrix-weighted activation MSE (CORRECT metric)
    quantize_chunked,
    matches_scope,
    compute_layer_weff_and_qeff,
    apply_lut_candidate,
    bake_lut_into_module,
)


def get_predefined_luts(lut_size: int) -> List[Tuple[str, torch.Tensor]]:
    """Get predefined LUTs from init_model_v2.py (Family F)."""
    try:
        from scripts.init_model_v2 import make_lut_candidates
        predefined = make_lut_candidates(lut_size=lut_size)
        return [(f'F_{name}', lut) for name, lut in predefined.items()]
    except ImportError:
        print("WARNING: Could not import init_model_v2.py, skipping Family F")
        return []


def generate_all_candidates(
    lut_size: int,
    max_abs: float,
    families: List[str],
    original_lut: Optional[torch.Tensor] = None,
    Q_eff_target: Optional[torch.Tensor] = None,
    S: Optional[torch.Tensor] = None,
    eps: float = 1e-4,
    scale_to_max_abs: bool = False,
    sigma2: Optional[torch.Tensor] = None,  # [in_features] iMatrix variance
) -> List[Tuple[str, torch.Tensor]]:
    """Generate all LUT candidates for a given LUT size.

    Args:
        lut_size: Number of LUT entries
        max_abs: Maximum absolute LUT value
        families: List of family codes to generate
        original_lut: Original LUT from checkpoint (for Family E)
        Q_eff_target: Target Q values [out, in] = W_ref / S
        S: Scale matrix [out, in]
        eps: Threshold for small scales
        scale_to_max_abs: If True, scale k-means LUTs to fill [0, max_abs]
        sigma2: Optional [in_features] iMatrix variance for iActMSE-optimal
                k-means and quantile generation. If provided, D/G families
                will use σ² × S² weighting.
    """
    candidates = []

    # Family E: Original LUT (baseline)
    if 'E' in families and original_lut is not None:
        candidates.append(('E_original', original_lut.clone()))

    # Family F: Predefined LUTs from init_model_v2.py
    if 'F' in families:
        candidates.extend(get_predefined_luts(lut_size))

    # Family A: Uniform
    if 'A' in families:
        candidates.extend(generate_lut_family_A(lut_size, max_abs))

    # Family B: Dense-center
    if 'B' in families:
        candidates.extend(generate_lut_family_B(lut_size, max_abs))

    # Family C: Heavy-tail
    if 'C' in families:
        candidates.extend(generate_lut_family_C(lut_size, max_abs))

    # Family D: Quantile (requires layer data)
    if 'D' in families and Q_eff_target is not None and S is not None:
        # With sigma2: uses iMatrix-weighted quantiles (σ² × S²)
        # Without sigma2: uses unweighted quantiles (original behavior)
        name, lut = generate_lut_family_D(lut_size, Q_eff_target, S, max_abs, eps, sigma2=sigma2)
        candidates.append((name, lut))

    # Family G: Weighted k-means (requires layer data) - asymmetric, eval-only
    if 'G' in families and Q_eff_target is not None and S is not None:
        # With sigma2: uses iMatrix × S² weighting (iActMSE-optimal)
        # Without sigma2: uses S²-only weighting (original behavior)
        name, lut = generate_lut_family_G(
            lut_size, Q_eff_target, S, max_abs, eps,
            scale_to_max_abs=scale_to_max_abs,
            sigma2=sigma2,
        )
        candidates.append((name, lut))

    # Family H: Symmetric k-means (requires layer data) - trainable-compatible
    if 'H' in families and Q_eff_target is not None and S is not None:
        # H expects 1D tensors, flatten and filter valid entries
        S_flat = S.flatten().abs()
        Q_flat = Q_eff_target.flatten()
        valid_mask = S_flat >= eps
        if valid_mask.sum() > 100:  # Need enough valid entries
            name, lut, stats = generate_lut_family_H_symkmeans(
                lut_size, Q_flat[valid_mask], S_flat[valid_mask], max_abs, eps
            )
            candidates.append((name, lut))

    return candidates


def process_single_layer(args_tuple: Tuple) -> Dict[str, Any]:
    """
    Process a single layer - standalone function for multiprocessing.

    CRITICAL: Uses W_ref (original FP32 weights) as target, not W_eff_old.
    This is the key insight: we want to optimize Q_target = W_ref / S,
    not preserve the existing quantized values.

    Args:
        args_tuple: (name, layer_data, config) where:
            - name: layer name string
            - layer_data: dict with lut, _Q, _indices, scale_A, scale_B, rank_magnitude, W_orig
            - config: dict with families, max_abs, auto_max_abs, metric, eps, activation_var, verbose

    Returns:
        Dict with best_name, best_score, best_lut, best_indices, all_results, etc.
    """
    name, layer_data, config = args_tuple

    # Unpack layer data
    lut = layer_data['lut']
    _Q = layer_data['_Q']
    _indices = layer_data['_indices']
    scale_A = layer_data['scale_A']
    scale_B = layer_data['scale_B']
    rank_magnitude = layer_data['rank_magnitude']
    W_orig = layer_data['W_orig']  # W_ref: Original FP32 weights (THE TARGET)

    # Unpack config
    families = config['families']
    max_abs = config['max_abs']
    auto_max_abs = config['auto_max_abs']
    metric = config['metric']
    eps = config['eps']
    activation_var = config.get('activation_var')  # σ² from iMatrix [in_features]
    verbose = config.get('verbose', False)
    scale_to_max_abs = config.get('scale_to_max_abs', False)

    lut_size = lut.numel()

    # Compute scales S = (scale_A * rank_magnitude) @ scale_B
    # rank_magnitude is (rank,), scale_A is (out, rank), scale_B is (rank, in)
    # Result: S is (out, in)
    A_scaled = scale_A * rank_magnitude  # [out, rank] * [rank] = [out, rank]
    S = A_scaled @ scale_B  # [out, rank] @ [rank, in] = [out, in]

    # Current Q values (for fallback/comparison only)
    if _Q is not None:
        Q_current = _Q.float()
    elif _indices is not None:
        Q_current = lut[_indices].float()
    else:
        return {'name': name, 'error': 'No _Q or _indices'}

    # CRITICAL: Target is W_ref / S (original FP32 weights), NOT W_eff_old
    # This is what makes optimization meaningful - we're trying to best
    # approximate the original weights, not preserve existing quantization.
    if W_orig is not None:
        S_safe = torch.where(S.abs() < eps, torch.full_like(S, eps), S)
        Q_target = W_orig / S_safe  # Target Q values (what we want to quantize)
        W_ref = W_orig  # Reference weights for scoring
    else:
        # Fallback: no original weights available, can only preserve existing
        Q_target = Q_current
        W_ref = Q_current * S

    # Compute per-layer max_abs if requested
    if auto_max_abs:
        valid_mask = S.abs() >= eps
        Q_valid = Q_target.abs()[valid_mask] if valid_mask.sum() > 0 else Q_target.abs()
        layer_max_abs = torch.quantile(Q_valid.flatten().float(), 0.999).item()
        layer_max_abs = max(layer_max_abs, 0.1)
    else:
        layer_max_abs = max_abs

    # Generate all candidates for this layer
    # Pass sigma2 (activation_var) for iMatrix-weighted k-means/quantile
    original_lut = lut.float().clone()
    candidates = generate_all_candidates(
        lut_size=lut_size,
        max_abs=layer_max_abs,
        families=families,
        original_lut=original_lut,
        Q_eff_target=Q_target,  # Target from W_ref / S
        S=S,
        eps=eps,
        scale_to_max_abs=scale_to_max_abs,
        sigma2=activation_var,  # Pass iMatrix for weighted k-means/quantile
    )

    # Test each candidate
    best_name = None
    best_score = float('inf')
    best_lut = None
    best_indices = None
    all_results = {}

    for cand_name, lut_new in candidates:
        # Validate LUT
        v = validate_lut_fp16(lut_new)
        if not v['valid']:
            continue

        # Quantize Q_target to LUT
        indices_new = quantize_chunked(Q_target, lut_new)
        Q_quant = lut_new[indices_new]
        W_new = Q_quant * S

        # Compute score based on selected metric
        # ALL metrics should compare to W_ref (original weights), not W_eff_old
        if metric == 'iActMSE' and activation_var is not None:
            # CORRECT metric: iActMSE = Σ σ²[i] * S² * (Q_target - Q_quant)²
            score = compute_iActMSE(Q_target, Q_quant, S, activation_var, normalize=True)
        elif metric == 'activation_mse' and activation_var is not None:
            # Legacy: activation-weighted MSE on W (same as iActMSE but computed differently)
            score = compute_activation_weighted_mse(W_ref, W_new, activation_var)
        elif metric == 'weighted_mse':
            # S²-weighted MSE (no iMatrix)
            score = compute_weighted_mse(W_ref, W_new, S, eps=eps)
        else:
            # MAE fallback
            score = (W_ref - W_new).abs().mean().item()

        all_results[cand_name] = score

        if score < best_score:
            best_score = score
            best_name = cand_name
            best_lut = lut_new.clone()
            best_indices = indices_new.clone()

    # Compute original score for comparison
    original_score = all_results.get('E_original', float('inf'))
    improved = best_name != 'E_original' and best_score < original_score

    # Get LUT range for winning candidate (always show, not just G_kmeans)
    lut_range = None
    if best_lut is not None:
        lut_range = (best_lut.min().item(), best_lut.max().item())

    return {
        'name': name,
        'best_name': best_name,
        'best_score': best_score,
        'best_lut': best_lut,
        'best_indices': best_indices,
        'original_score': original_score,
        'improved': improved,
        'all_results': all_results,
        'lut_range': lut_range,
        'metric': metric,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description='Select best LUT per layer from all candidates',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument('checkpoint', nargs='?', default=None,
                        help='Input V2 checkpoint (provides scales, LUT structure). Optional if --from-scratch is used.')
    parser.add_argument('--from-scratch', action='store_true',
                        help='Initialize scales via SVD from base model (no checkpoint needed)')
    parser.add_argument('-b', '--base-dir', metavar='FOLDER',
                        help='Base folder for checkpoint and output paths')
    parser.add_argument('-o', '--output', default=None,
                        help='Output hybrid checkpoint path')
    parser.add_argument('--output-stats', default=None,
                        help='Output JSON with per-layer selection stats')
    parser.add_argument('--config', default='q4_r32',
                        help='Config preset (q2a4, q4a4, q4_r32, etc.) - default: q4_r32 (LUT16, rank32)')
    parser.add_argument('--model-id', default='Qwen/Qwen3-0.6B',
                        help='Base model for FP32 target weights (SVD target)')
    parser.add_argument('--scope', choices=['mlp', 'attn', 'all'], default='all',
                        help='Layer scope to modify (default: all)')
    parser.add_argument('--families', default='E,F,A,B,C,D',
                        help='Comma-separated family codes: '
                             'E=original, F=predefined, A=uniform, B=dense-center, C=heavy-tail, '
                             'D=quantile, G=kmeans (eval-only), H=symmetric-kmeans (trainable)')
    parser.add_argument('--max-abs', type=float, default=2.0,
                        help='Max absolute LUT value for generated candidates (default: 2.0 for FP4 range)')
    parser.add_argument('--auto-max-abs', action='store_true',
                        help='Use per-layer max_abs from p99.9 of |Q_eff| (recommended)')
    parser.add_argument('--metric', choices=['weighted_mse', 'mae', 'activation_mse', 'iActMSE'], default='weighted_mse',
                        help='Selection metric: weighted_mse (S^2), mae, activation_mse, or iActMSE (BEST - requires --imatrix). '
                             'iActMSE = Σ σ²[i] * S² * (Q_target - Q_quant)² is the theoretically correct metric.')
    parser.add_argument('--imatrix', default=None,
                        help='Path to importance matrix (.pt file) for activation_mse metric. Use compute_imatrix.py or calibrate_activation_stats.py to generate.')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show selections without saving checkpoint')
    parser.add_argument('--verbose', action='store_true',
                        help='Print per-layer details')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of parallel workers (default: 1, sequential)')
    parser.add_argument('--eps', type=float, default=1e-4,
                        help='Epsilon for safe division (default: 1e-4)')
    parser.add_argument('--group-size', type=int, default=16,
                        help='Group size for scale initialization (default: 16, must match init_model_v2.py)')
    parser.add_argument('--scale-lut', action='store_true',
                        help='Scale G/H k-means LUTs to fill [0, max_abs] range (new behavior). '
                             'Without this flag, LUTs use data-adaptive range (old behavior).')
    parser.add_argument('--no-tighten', action='store_true',
                        help='Skip tightening Q after FP16 snap (for --from-scratch mode)')

    args = parser.parse_args()

    # Validate: need checkpoint or --from-scratch
    if not args.checkpoint and not args.from_scratch:
        parser.error("Must specify checkpoint or --from-scratch")

    # Apply base directory if specified
    if args.base_dir:
        base = Path(args.base_dir)
        if args.checkpoint:
            args.checkpoint = str(base / args.checkpoint)
        if args.output:
            args.output = str(base / args.output)
        if args.output_stats:
            args.output_stats = str(base / args.output_stats)

    return args


def main():
    args = parse_args()

    print("=" * 60)
    print("SELECT BEST LUT PER LAYER")
    print("=" * 60)
    if args.from_scratch:
        print(f"Mode:        FROM SCRATCH (SVD initialization)")
    else:
        print(f"Checkpoint:  {args.checkpoint}")
    print(f"Model:       {args.model_id}")
    print(f"Output:      {args.output or '(dry run)'}")
    print(f"Families:    {args.families}")
    print(f"Scope:       {args.scope}")
    print(f"Metric:      {args.metric}")
    print(f"Max abs:     {'auto (p99.9)' if args.auto_max_abs else args.max_abs}")
    print(f"Group size:  {args.group_size}")
    print(f"Scale LUT:   {'enabled (new)' if args.scale_lut else 'disabled (old/data-adaptive)'}")
    if args.from_scratch:
        print(f"Tighten Q:   {'disabled (--no-tighten)' if args.no_tighten else 'enabled'}")
    if args.imatrix:
        print(f"iMatrix:     {args.imatrix}")
    print()

    # Validate metric requirements
    if args.metric in ('activation_mse', 'iActMSE') and not args.imatrix:
        print(f"ERROR: --metric {args.metric} requires --imatrix")
        print("Run: python scripts/compute_imatrix.py --output imatrix.pt")
        return 1

    # Load importance matrix if provided
    activation_variances = None
    if args.imatrix:
        print(f"Loading importance matrix: {args.imatrix}")
        cache_data = torch.load(args.imatrix, map_location='cpu', weights_only=False)
        # Support both formats: 'variances' (old) and 'sigma2' (new from compute_imatrix.py)
        activation_variances = cache_data.get('variances') or cache_data.get('sigma2', {})
        print(f"  Loaded σ² for {len(activation_variances)} layers")
        if 'metadata' in cache_data:
            meta = cache_data['metadata']
            print(f"  Source: {meta.get('model_id', '?')}, tokens={meta.get('num_tokens', '?')}")
        elif 'num_samples' in cache_data:
            print(f"  Calibration: {cache_data.get('num_samples', '?')} samples, {cache_data.get('dataset', '?')}")
        print()

    # Parse families
    families = [f.strip().upper() for f in args.families.split(',')]
    print(f"Testing LUT families: {families}")

    # Load config (default: q4_r32 = LUT16, rank32)
    if args.config in CONFIG_PRESETS:
        preset = CONFIG_PRESETS[args.config]
        config_name = args.config
    else:
        preset = CONFIG_PRESETS['q4_r32']
        config_name = 'q4_r32'
        print(f"Warning: Unknown config '{args.config}', using default 'q4_r32'")

    mlp_lut = preset['mlp_lut']
    mlp_rank = preset['mlp_rank']
    attn_lut = preset['attn_lut']
    attn_rank = preset['attn_rank']
    print(f"Config: {config_name} -> MLP LUT{mlp_lut} rank={mlp_rank}, Attn LUT{attn_lut} rank={attn_rank}, group_size={args.group_size}")
    print()

    # Import V2 classes
    from qat_lora import AnemllQuantConfigV2, replace_linear_with_anemll_v2
    from qat_lora.ane_qat_linear_v2 import AnemllQATLinearV2

    # Load base model FIRST to get original FP32 weights
    print(f"Loading base model: {args.model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )

    # Extract original FP32 weights BEFORE replacing with V2 modules
    print("Extracting original FP32 weights...")
    original_weights = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Store original weight for later use
            original_weights[name] = module.weight.data.float().cpu().clone()
    print(f"  Extracted {len(original_weights)} linear layer weights")

    # Create V2 configs (must match init_model_v2.py settings)
    mlp_config = AnemllQuantConfigV2(
        lut_size=mlp_lut,
        scale_rank=mlp_rank,
        group_size=args.group_size,
        learnable_lut=False,
        force_positive_scales=False,
        magnitude_activation='identity',
    )
    attn_config = AnemllQuantConfigV2(
        lut_size=attn_lut,
        scale_rank=attn_rank,
        group_size=args.group_size,
        learnable_lut=False,
        force_positive_scales=False,
        magnitude_activation='identity',
    )

    # skip_init=False for from-scratch (do SVD), skip_init=True when loading checkpoint
    skip_init = not args.from_scratch
    if args.from_scratch:
        print("Initializing scales via SVD (this may take a moment)...")

    replace_linear_with_anemll_v2(
        model,
        mlp_config=mlp_config,
        attn_config=attn_config,
        quantize_attn=True,
        quantize_lm_head=False,
        skip_init=skip_init,
    )

    # Freeze Q (quantize weights to LUT indices) for from-scratch mode
    # This must happen AFTER replace_linear_with_anemll_v2 and BEFORE loading checkpoint
    if args.from_scratch:
        from qat_lora.ane_qat_linear_v2 import freeze_Q_all
        from scripts.tighten_q import tighten_q_layer

        print("Freezing Q (quantizing weights to LUT indices)...")

        # Print LUT values once (all layers use the same default LUT)
        first_lut_printed = False
        for name, module in model.named_modules():
            if isinstance(module, AnemllQATLinearV2) and not first_lut_printed:
                lut = module.lut.cpu()
                lut_size = lut.numel()
                lut_str = ", ".join([f"{v:+.4f}" for v in lut.tolist()])
                print(f"  Default LUT ({lut_size} entries): [{lut_str}]")
                first_lut_printed = True
                break

        if args.verbose:
            # Verbose mode: show LUT info and MSE per layer
            frozen = 0
            for name, module in model.named_modules():
                if isinstance(module, AnemllQATLinearV2):
                    # Get original weight for MSE computation
                    W_ref = original_weights.get(name)
                    if W_ref is None:
                        W_ref = original_weights.get(f"{name}.weight")

                    # Freeze Q
                    module.freeze_Q()
                    frozen += 1

                    # Compute W_effective after freeze
                    S = module._compute_full_scales()
                    W_eff = module._Q * S

                    # LUT info
                    lut_min = module.lut.min().item()
                    lut_max = module.lut.max().item()

                    # Compute MSE vs original weight
                    if W_ref is not None:
                        W_ref_dev = W_ref.to(W_eff.device)
                        mse = ((W_eff - W_ref_dev) ** 2).mean().item()
                        mae = (W_eff - W_ref_dev).abs().mean().item()
                        # Short layer name for display
                        short_name = name.replace('model.layers.', '').replace('.self_attn.', '.').replace('.mlp.', '.')
                        print(f"  {short_name:40s} LUT=[{lut_min:+.4f}, {lut_max:+.4f}] MSE={mse:.2e} MAE={mae:.2e}")
                    else:
                        short_name = name.replace('model.layers.', '').replace('.self_attn.', '.').replace('.mlp.', '.')
                        print(f"  {short_name:40s} LUT=[{lut_min:+.4f}, {lut_max:+.4f}] (no ref weight)")
        else:
            frozen = freeze_Q_all(model)
        print(f"  Froze Q for {frozen} layers")

        # Step 2: Snap magnitudes to FP16 (same as init_model_v2.py)
        print("Snapping magnitudes to FP16...")
        mags_snapped = 0
        with torch.no_grad():
            for name, module in model.named_modules():
                if isinstance(module, AnemllQATLinearV2):
                    if hasattr(module, 'rank_magnitude') and module.rank_magnitude is not None:
                        module.rank_magnitude.data = module.rank_magnitude.data.to(torch.float16).to(torch.float32)
                        mags_snapped += 1
        print(f"  Snapped {mags_snapped} layers")

        # Step 3: Tighten Q (recalculate to match snapped scales)
        if args.no_tighten:
            print("Skipping Tighten Q (--no-tighten)")
        else:
            print("Tightening Q (recalculating to match snapped scales)...")
            tightened = 0
            tighten_mse_old_sum = 0.0
            tighten_mse_new_sum = 0.0
            for name, module in model.named_modules():
                if isinstance(module, AnemllQATLinearV2) and module._Q is not None:
                    # Get W_ref for this layer
                    W_ref = original_weights.get(name)
                    if W_ref is None:
                        # Try alternate key format
                        W_ref = original_weights.get(f"{name}.weight")
                    if W_ref is not None:
                        result = tighten_q_layer(module, W_ref, clamp_q=True)
                        tighten_mse_old_sum += result['mse_old']
                        tighten_mse_new_sum += result['mse_new']
                        tightened += 1
            print(f"  Tightened Q for {tightened} layers")
            print(f"  Total MSE (sum): before={tighten_mse_old_sum:.6e}, after={tighten_mse_new_sum:.6e}")

    # Load checkpoint (if provided)
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        state_dict = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        model.load_state_dict(state_dict, strict=False)

        # Manually load _Q buffers
        for name, m in model.named_modules():
            if isinstance(m, AnemllQATLinearV2):
                q_key = f"{name}._Q"
                if q_key in state_dict and m._Q is None:
                    m.register_buffer("_Q", state_dict[q_key])
    elif not args.from_scratch:
        print("Using SVD-initialized scales (no checkpoint)")

    # Count layers first for progress display
    # After freeze_Q_all() (from-scratch) or checkpoint load, all layers have _Q/_indices
    layer_list = []
    for name, module in model.named_modules():
        if not isinstance(module, AnemllQATLinearV2):
            continue
        if not matches_scope(name, args.scope):
            continue
        if module._Q is None and module._indices is None:
            continue
        layer_list.append((name, module))

    total_layers = len(layer_list)
    print(f"\nFound {total_layers} layers to process")

    if total_layers == 0:
        print("\nERROR: No layers found to process!")
        print("  Check --scope setting and model configuration.")
        return 1

    if args.workers > 1:
        print(f"Using {args.workers} parallel workers")

    # Process each layer
    print("\n" + "=" * 60)
    print("PER-LAYER LUT SELECTION")
    print("=" * 60)

    layer_selections = {}
    family_counts = defaultdict(int)
    improved_layers = 0
    total_score = 0.0  # Sum of all best scores
    total_original_score = 0.0  # Sum of all original scores

    # Helper function to find activation variance for a layer
    def get_activation_var(name):
        if activation_variances is None:
            return None
        # Exact match
        if name in activation_variances:
            return activation_variances[name].float()
        # Try without .linear suffix
        base_name = name.replace('.linear', '')
        if base_name in activation_variances:
            return activation_variances[base_name].float()
        # Fuzzy match
        name_suffix = '.'.join(name.split('.')[-3:])
        for cache_key in activation_variances.keys():
            if cache_key.endswith(name_suffix):
                return activation_variances[cache_key].float()
        return None

    # Helper function to find original weight key
    def get_orig_weight_key(name):
        if name in original_weights:
            return name
        base = name.replace('.linear', '')
        if base in original_weights:
            return base
        with_linear = name + '.linear'
        if with_linear in original_weights:
            return with_linear
        return None

    # Prepare layer data for processing
    def prepare_layer_data(name, module):
        """Extract serializable data from module for parallel processing."""
        orig_key = get_orig_weight_key(name)
        W_orig = original_weights.get(orig_key) if orig_key else None

        # Use detach() before clone() for tensors that may require_grad
        # This is needed for multiprocessing serialization
        return {
            'lut': module.lut.detach().float().cpu().clone(),
            '_Q': module._Q.detach().float().cpu().clone() if module._Q is not None else None,
            '_indices': module._indices.detach().cpu().clone() if module._indices is not None else None,
            'scale_A': module.scale_A.detach().float().cpu().clone(),
            'scale_B': module.scale_B.detach().float().cpu().clone(),
            'rank_magnitude': module.rank_magnitude.detach().float().cpu().clone(),
            'W_orig': W_orig.detach().float().cpu().clone() if W_orig is not None else None,
        }

    # Shared config for all layers
    shared_config = {
        'families': families,
        'max_abs': args.max_abs,
        'auto_max_abs': args.auto_max_abs,
        'metric': args.metric,
        'eps': args.eps,
        'verbose': args.verbose,
        'scale_to_max_abs': args.scale_lut,  # If True, scale G k-means LUTs to fill [0, max_abs]
    }

    # Process layers - sequential or parallel
    start_time = time.time()

    if args.workers <= 1:
        # Sequential processing (original behavior)
        for layer_idx, (name, module) in enumerate(layer_list, 1):
            # Progress indicator with ETA
            pct = 100 * layer_idx / total_layers
            elapsed = time.time() - start_time
            if layer_idx > 1:
                eta_sec = elapsed / (layer_idx - 1) * (total_layers - layer_idx + 1)
                eta_str = f"{int(eta_sec // 60)}m {int(eta_sec % 60):02d}s"
            else:
                eta_str = "..."
            # Extract layer number and proj name: model.layers.2.self_attn.q_proj -> q_proj.2
            parts = name.split('.')
            layer_num = parts[2] if len(parts) > 2 and parts[2].isdigit() else '?'
            proj_name = parts[-1]
            short_name = f"{proj_name}.{layer_num}".ljust(12)
            print(f"\r{' ' * 100}\r  Processing: {layer_idx}/{total_layers} ({pct:.0f}%) - {short_name} ETA: {eta_str}", end='', flush=True)

            # Prepare data
            layer_data = prepare_layer_data(name, module)
            config = shared_config.copy()
            config['activation_var'] = get_activation_var(name)

            # Process layer
            result = process_single_layer((name, layer_data, config))

            # Handle result
            if 'error' in result:
                print(f"\n    ERROR: {result['error']}")
                continue

            best_name = result['best_name']
            best_score = result['best_score']
            best_lut = result['best_lut']
            best_indices = result['best_indices']
            original_score = result['original_score']
            improved = result['improved']
            all_results = result['all_results']
            lut_range = result['lut_range']

            # Debug: show ALL candidate scores for first layer
            if args.verbose and layer_idx == 1:
                print()
                print(f"    ALL CANDIDATE SCORES for {name}:")
                for cn, sc in sorted(all_results.items(), key=lambda x: x[1]):
                    marker = " <-- BEST" if cn == best_name else ""
                    print(f"      {cn}: {sc:.8f}{marker}")

            # Record selection
            layer_selections[name] = {
                'best': best_name,
                'score': best_score,
                'metric': args.metric,
                'original_score': original_score,
                'improved': improved,
                'all_candidates': all_results,
            }

            # Count family and track totals
            family_code = best_name[0] if best_name else '?'
            family_counts[family_code] += 1
            total_score += best_score
            total_original_score += original_score

            if improved:
                improved_layers += 1

            if args.verbose:
                status = "IMPROVED" if improved else "kept"
                metric_label = args.metric.upper()
                if lut_range:
                    print(f"  -> {short_name}: {best_name} ({metric_label}={best_score:.2e}, orig={original_score:.2e}) LUT=[{lut_range[0]:.4f}, {lut_range[1]:.4f}] [{status}]")
                else:
                    print(f"  -> {short_name}: {best_name} ({metric_label}={best_score:.2e}, orig={original_score:.2e}) [{status}]")

            # Apply best LUT to module
            if best_lut is not None and best_indices is not None and not args.dry_run:
                bake_lut_into_module(module, best_lut, best_indices, validate=True)

            # Memory cleanup
            del layer_data
            orig_key = get_orig_weight_key(name)
            if orig_key and orig_key in original_weights:
                del original_weights[orig_key]
            if layer_idx % 10 == 0:
                gc.collect()

    else:
        # Parallel processing with ThreadPoolExecutor (avoids spawn overhead)
        # Note: Using threads instead of processes because:
        # 1. ProcessPoolExecutor with spawn requires re-importing all modules
        # 2. Our workload releases GIL during torch tensor operations
        # 3. Much faster startup and lower memory overhead
        from concurrent.futures import ThreadPoolExecutor

        print(f"\n  Preparing {total_layers} layers for parallel processing...")
        sys.stdout.flush()

        # Prepare all layer data upfront
        tasks = []
        module_map = {}  # name -> module reference for applying results later

        for i, (name, module) in enumerate(layer_list):
            if (i + 1) % 50 == 0:
                print(f"\r    Prepared {i + 1}/{total_layers} layers...", end='', flush=True)
            layer_data = prepare_layer_data(name, module)
            config = shared_config.copy()
            config['activation_var'] = get_activation_var(name)
            tasks.append((name, layer_data, config))
            module_map[name] = module

        print(f"\r    Prepared {total_layers}/{total_layers} layers.       ")
        print(f"  Starting {args.workers} workers (ThreadPoolExecutor)...")
        sys.stdout.flush()

        # Process in parallel
        results = {}
        completed = 0

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            # Submit all tasks
            future_to_name = {executor.submit(process_single_layer, task): task[0] for task in tasks}

            # Collect results as they complete
            parallel_start = time.time()
            for future in as_completed(future_to_name):
                name = future_to_name[future]
                completed += 1
                pct = 100 * completed / total_layers
                # ETA calculation
                elapsed = time.time() - parallel_start
                if completed > 0:
                    eta_sec = elapsed / completed * (total_layers - completed)
                    eta_str = f"{int(eta_sec // 60)}m {int(eta_sec % 60):02d}s"
                else:
                    eta_str = "..."
                # Extract layer number and proj name: model.layers.2.self_attn.q_proj -> q_proj.2
                parts = name.split('.')
                layer_num = parts[2] if len(parts) > 2 and parts[2].isdigit() else '?'
                proj_name = parts[-1]
                short_name = f"{proj_name}.{layer_num}".ljust(12)
                print(f"\r{' ' * 100}\r  Completed: {completed}/{total_layers} ({pct:.0f}%) - {short_name} ETA: {eta_str}", end='', flush=True)

                try:
                    result = future.result()
                    results[name] = result
                except Exception as e:
                    print(f"\n    ERROR processing {name}: {e}")
                    results[name] = {'name': name, 'error': str(e)}

        print()  # New line after progress

        # Apply results to modules (must be sequential)
        print("  Applying results to model...")
        for name, module in layer_list:
            result = results.get(name)
            if not result or 'error' in result:
                continue

            best_name = result['best_name']
            best_score = result['best_score']
            best_lut = result['best_lut']
            best_indices = result['best_indices']
            original_score = result['original_score']
            improved = result['improved']
            all_results = result['all_results']
            lut_range = result['lut_range']

            # Record selection
            layer_selections[name] = {
                'best': best_name,
                'score': best_score,
                'metric': args.metric,
                'original_score': original_score,
                'improved': improved,
                'all_candidates': all_results,
            }

            # Count family and track totals
            family_code = best_name[0] if best_name else '?'
            family_counts[family_code] += 1
            total_score += best_score
            total_original_score += original_score

            if improved:
                improved_layers += 1

            if args.verbose:
                status = "IMPROVED" if improved else "kept"
                metric_label = args.metric.upper()
                # Extract layer number: model.layers.2.self_attn.q_proj -> q_proj.2
                parts = name.split('.')
                layer_num = parts[2] if len(parts) > 2 and parts[2].isdigit() else '?'
                proj_name = parts[-1]
                short_name = f"{proj_name}.{layer_num}".ljust(12)
                if lut_range:
                    print(f"    {short_name}: {best_name} ({metric_label}={best_score:.2e}, orig={original_score:.2e}) LUT=[{lut_range[0]:.4f}, {lut_range[1]:.4f}] [{status}]")
                else:
                    print(f"    {short_name}: {best_name} ({metric_label}={best_score:.2e}, orig={original_score:.2e}) [{status}]")

            # Apply best LUT to module
            if best_lut is not None and best_indices is not None and not args.dry_run:
                bake_lut_into_module(module, best_lut, best_indices, validate=True)

        # Cleanup
        del tasks, results, original_weights
        gc.collect()

    # Summary - clear progress line first
    print()  # New line after progress
    print("\n" + "=" * 60)
    print("SELECTION SUMMARY")
    print("=" * 60)
    print(f"Total layers:    {total_layers}")
    print(f"Improved layers: {improved_layers} ({100*improved_layers/total_layers:.1f}%)")
    print()
    print(f"Total {args.metric.upper()}:")
    print(f"  Original:  {total_original_score:.6e}")
    print(f"  Optimized: {total_score:.6e}")
    if total_original_score > 0:
        reduction = (total_original_score - total_score) / total_original_score * 100
        print(f"  Reduction: {reduction:.2f}%")
    print()
    print("LUT Family Distribution:")
    for family in sorted(family_counts.keys()):
        count = family_counts[family]
        pct = 100 * count / total_layers if total_layers > 0 else 0
        family_desc = {
            'E': 'Original (baseline)',
            'F': 'Predefined (fp4_dense, uniform)',
            'G': 'K-means (weighted Lloyd)',
            'A': 'Uniform',
            'B': 'Dense-center',
            'C': 'Heavy-tail',
            'D': 'Quantile',
        }.get(family, 'Unknown')
        print(f"  {family}: {count} layers ({pct:.1f}%) - {family_desc}")

    # Save stats
    if args.output_stats:
        stats_path = Path(args.output_stats)
        stats_data = {
            'timestamp': datetime.now().isoformat(),
            'checkpoint': str(args.checkpoint),
            'families': families,
            'scope': args.scope,
            'metric': args.metric,
            'max_abs': 'auto' if args.auto_max_abs else args.max_abs,
            'imatrix': str(args.imatrix) if args.imatrix else None,
            'config': args.config,
            'group_size': args.group_size,
            'total_layers': total_layers,
            'improved_layers': improved_layers,
            'total_score': total_score,
            'total_original_score': total_original_score,
            'score_reduction_pct': (total_original_score - total_score) / total_original_score * 100 if total_original_score > 0 else 0,
            'family_counts': dict(family_counts),
            'layer_selections': layer_selections,
        }
        with open(stats_path, 'w') as f:
            json.dump(stats_data, f, indent=2)
        print(f"\nStats saved: {stats_path}")

    # Save hybrid checkpoint
    if args.output and not args.dry_run:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), output_path)
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"\nHybrid checkpoint saved: {output_path} ({size_mb:.1f} MB)")
    elif args.dry_run:
        print("\n(Dry run - no checkpoint saved)")

    print("\nDone!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
