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
    --metric activation_mse - Activation-weighted MSE (BEST - requires calibration)

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

    # BEST: Use activation-weighted MSE (requires calibration first)
    python scripts/calibrate_activation_stats.py --output activation_stats.pt
    python scripts/select_best_lut_per_layer.py checkpoint.pt -o hybrid.pt \
        --metric activation_mse --activation-cache activation_stats.pt

    # With stats output
    python scripts/select_best_lut_per_layer.py checkpoint.pt -o hybrid.pt --output-stats stats.json

    # Dry run (no checkpoint saved)
    python scripts/select_best_lut_per_layer.py checkpoint.pt --dry-run
"""

import argparse
import gc
import json
import sys
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
    generate_lut_family_G,  # Weighted k-means
    compute_weighted_mse,   # S^2-weighted MSE
    compute_activation_weighted_mse,  # Activation-weighted MSE (Option A)
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
) -> List[Tuple[str, torch.Tensor]]:
    """Generate all LUT candidates for a given LUT size."""
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
        name, lut = generate_lut_family_D(lut_size, Q_eff_target, S, max_abs, eps)
        candidates.append((name, lut))

    # Family G: Weighted k-means (requires layer data)
    if 'G' in families and Q_eff_target is not None and S is not None:
        name, lut = generate_lut_family_G(lut_size, Q_eff_target, S, max_abs, eps)
        candidates.append((name, lut))

    return candidates


def process_single_layer(args_tuple: Tuple) -> Dict[str, Any]:
    """
    Process a single layer - standalone function for multiprocessing.

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
    W_orig = layer_data['W_orig']

    # Unpack config
    families = config['families']
    max_abs = config['max_abs']
    auto_max_abs = config['auto_max_abs']
    metric = config['metric']
    eps = config['eps']
    activation_var = config.get('activation_var')
    verbose = config.get('verbose', False)

    lut_size = lut.numel()

    # Compute scales S = (scale_A * rank_magnitude) @ scale_B
    # rank_magnitude is (rank,), scale_A is (out, rank), scale_B is (rank, in)
    # Result: S is (out, in)
    A_scaled = scale_A * rank_magnitude  # [out, rank] * [rank] = [out, rank]
    S = A_scaled @ scale_B  # [out, rank] @ [rank, in] = [out, in]

    # Current Q values
    if _Q is not None:
        Q_current = _Q.float()
    elif _indices is not None:
        Q_current = lut[_indices].float()
    else:
        return {'name': name, 'error': 'No _Q or _indices'}

    # Current W_eff
    W_eff_old = Q_current * S

    # Target: original FP32 weights (continuous, not snapped)
    if W_orig is not None:
        S_safe = torch.where(S.abs() < eps, torch.full_like(S, eps), S)
        Q_eff_target = W_orig / S_safe
        W_eff_target = W_orig
    else:
        # Fallback to snapped values
        Q_eff_target = Q_current
        W_eff_target = W_eff_old

    # Compute per-layer max_abs if requested
    if auto_max_abs:
        valid_mask = S.abs() >= eps
        Q_valid = Q_eff_target.abs()[valid_mask] if valid_mask.sum() > 0 else Q_eff_target.abs()
        layer_max_abs = torch.quantile(Q_valid.flatten().float(), 0.999).item()
        layer_max_abs = max(layer_max_abs, 0.1)
    else:
        layer_max_abs = max_abs

    # Generate all candidates for this layer
    original_lut = lut.float().clone()
    candidates = generate_all_candidates(
        lut_size=lut_size,
        max_abs=layer_max_abs,
        families=families,
        original_lut=original_lut,
        Q_eff_target=Q_eff_target,
        S=S,
        eps=eps,
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

        indices_new, stats = apply_lut_candidate(
            W_eff_old, S, Q_eff_target, lut_new,
            eps=eps,
        )

        # Compute score based on selected metric
        Q_new = lut_new[indices_new]
        W_eff_new = Q_new * S

        if metric == 'weighted_mse':
            score = compute_weighted_mse(W_eff_target, W_eff_new, S, eps=eps)
        elif metric == 'activation_mse' and activation_var is not None:
            score = compute_activation_weighted_mse(W_eff_target, W_eff_new, activation_var)
        else:  # mae or fallback
            if metric == 'activation_mse' and activation_var is None:
                score = compute_weighted_mse(W_eff_target, W_eff_new, S, eps=eps)
            else:
                score = stats['mae']

        all_results[cand_name] = score

        if score < best_score:
            best_score = score
            best_name = cand_name
            best_lut = lut_new.clone()
            best_indices = indices_new.clone()

    # Compute original score for comparison
    original_score = all_results.get('E_original', float('inf'))
    improved = best_name != 'E_original' and best_score < original_score

    # Get LUT range for G_kmeans
    lut_range = None
    if best_name == 'G_kmeans' and best_lut is not None:
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
    parser.add_argument('--config', default=None,
                        help='Config preset (q2a4, q4a4, etc.)')
    parser.add_argument('--model-id', default='Qwen/Qwen3-0.6B',
                        help='Base model for FP32 target weights (SVD target)')
    parser.add_argument('--scope', choices=['mlp', 'attn', 'all'], default='all',
                        help='Layer scope to modify (default: all)')
    parser.add_argument('--families', default='E,F,G,A,B,C,D',
                        help='Comma-separated family codes (default: E,F,G,A,B,C,D)')
    parser.add_argument('--max-abs', type=float, default=2.0,
                        help='Max absolute LUT value for generated candidates (default: 2.0 for FP4 range)')
    parser.add_argument('--auto-max-abs', action='store_true',
                        help='Use per-layer max_abs from p99.9 of |Q_eff| (recommended)')
    parser.add_argument('--metric', choices=['weighted_mse', 'mae', 'activation_mse'], default='weighted_mse',
                        help='Selection metric: weighted_mse (S^2), mae, or activation_mse (best, requires --activation-cache)')
    parser.add_argument('--activation-cache', default=None,
                        help='Path to activation_stats.pt from calibrate_activation_stats.py (enables activation_mse)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show selections without saving checkpoint')
    parser.add_argument('--verbose', action='store_true',
                        help='Print per-layer details')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of parallel workers (default: 1, sequential)')
    parser.add_argument('--eps', type=float, default=1e-4,
                        help='Epsilon for safe division (default: 1e-4)')

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
    if args.activation_cache:
        print(f"Act cache:   {args.activation_cache}")
    print()

    # Validate metric requirements
    if args.metric == 'activation_mse' and not args.activation_cache:
        print("ERROR: --metric activation_mse requires --activation-cache")
        print("Run: python scripts/calibrate_activation_stats.py --output activation_stats.pt")
        return 1

    # Load activation cache if provided
    activation_variances = None
    if args.activation_cache:
        print(f"Loading activation cache: {args.activation_cache}")
        cache_data = torch.load(args.activation_cache, map_location='cpu', weights_only=False)
        activation_variances = cache_data.get('variances', {})
        print(f"  Loaded variances for {len(activation_variances)} layers")
        print(f"  Calibration: {cache_data.get('num_samples', '?')} samples, {cache_data.get('dataset', '?')}")
        print()

    # Parse families
    families = [f.strip().upper() for f in args.families.split(',')]
    print(f"Testing LUT families: {families}")

    # Load config
    if args.config and args.config in CONFIG_PRESETS:
        preset = CONFIG_PRESETS[args.config]
    else:
        preset = CONFIG_PRESETS['q4_r32']

    mlp_lut = preset['mlp_lut']
    mlp_rank = preset['mlp_rank']
    attn_lut = preset['attn_lut']
    attn_rank = preset['attn_rank']
    print(f"Config: MLP LUT{mlp_lut} rank={mlp_rank}, Attn LUT{attn_lut} rank={attn_rank}")
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
        print("Freezing Q (quantizing weights to LUT indices)...")
        frozen = freeze_Q_all(model)
        print(f"  Froze Q for {frozen} layers")

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
    }

    # Process layers - sequential or parallel
    if args.workers <= 1:
        # Sequential processing (original behavior)
        for layer_idx, (name, module) in enumerate(layer_list, 1):
            # Progress indicator
            pct = 100 * layer_idx / total_layers
            short_name = name.split('.')[-1]
            print(f"\r{' ' * 80}\r  Processing: {layer_idx}/{total_layers} ({pct:.0f}%) - {short_name}", end='', flush=True)

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

            # Count family
            family_code = best_name[0] if best_name else '?'
            family_counts[family_code] += 1

            if improved:
                improved_layers += 1

            if args.verbose:
                status = "IMPROVED" if improved else "kept"
                metric_label = args.metric.upper()
                if lut_range:
                    print(f"  -> {best_name} ({metric_label}={best_score:.2e}, orig={original_score:.2e}) LUT=[{lut_range[0]:.4f}, {lut_range[1]:.4f}] [{status}]")
                else:
                    print(f"  -> {best_name} ({metric_label}={best_score:.2e}, orig={original_score:.2e}) [{status}]")

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
            for future in as_completed(future_to_name):
                name = future_to_name[future]
                completed += 1
                pct = 100 * completed / total_layers
                short_name = name.split('.')[-1]
                print(f"\r{' ' * 80}\r  Completed: {completed}/{total_layers} ({pct:.0f}%) - {short_name}", end='', flush=True)

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

            # Count family
            family_code = best_name[0] if best_name else '?'
            family_counts[family_code] += 1

            if improved:
                improved_layers += 1

            if args.verbose:
                status = "IMPROVED" if improved else "kept"
                metric_label = args.metric.upper()
                if lut_range:
                    print(f"    {name.split('.')[-1]}: {best_name} ({metric_label}={best_score:.2e}, orig={original_score:.2e}) LUT=[{lut_range[0]:.4f}, {lut_range[1]:.4f}] [{status}]")
                else:
                    print(f"    {name.split('.')[-1]}: {best_name} ({metric_label}={best_score:.2e}, orig={original_score:.2e}) [{status}]")

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
            'total_layers': total_layers,
            'improved_layers': improved_layers,
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
