#!/usr/bin/env python3
"""
Generate LUT candidates from 4 families and apply them to create checkpoint variants.

LUT Families:
    A: Uniform - linspace(-M, M, 16), standard baseline
    B: Dense-center - more values near zero (asinh, tanh, sqrt grids)
    C: Heavy-tail - more values at extremes (geometric, power2, power3)
    D: Quantile - data-driven from layer's own |Q_eff|

For each candidate:
    1. Keep S (scales) fixed
    2. Recompute indices = nearest(lut_new, Q_eff_target)
    3. Compute distortion MAE = mean(|W_eff_old - W_eff_new|)
    4. Bake into checkpoint variant

Usage:
    # Generate all families
    python scripts/apply_lut_candidates.py checkpoint.pt -o ./lut_candidates/

    # Specific families only
    python scripts/apply_lut_candidates.py checkpoint.pt -o ./lut_candidates/ --families A,D

    # Different max_abs values
    python scripts/apply_lut_candidates.py checkpoint.pt -o ./lut_candidates/ --max-abs 1.5

    # Dry run (compute MAE without saving)
    python scripts/apply_lut_candidates.py checkpoint.pt -o ./lut_candidates/ --dry-run

    # MLP only
    python scripts/apply_lut_candidates.py checkpoint.pt -o ./lut_candidates/ --scope mlp
"""

import argparse
import json
import math
import re
import sys
from copy import deepcopy
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Dict, Optional

REPO_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_DIR))

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


# =============================================================================
# CONFIG PRESETS
# =============================================================================

CONFIG_PRESETS = {
    'q2a4': {'mlp_lut': 4, 'mlp_rank': 32, 'attn_lut': 16, 'attn_rank': 8},
    'q4a4': {'mlp_lut': 16, 'mlp_rank': 4, 'attn_lut': 16, 'attn_rank': 4},
    'q4a4_r32': {'mlp_lut': 16, 'mlp_rank': 32, 'attn_lut': 16, 'attn_rank': 32},
    'q4_r32': {'mlp_lut': 16, 'mlp_rank': 32, 'attn_lut': 16, 'attn_rank': 32},
    'q2a2': {'mlp_lut': 4, 'mlp_rank': 32, 'attn_lut': 4, 'attn_rank': 32},
}


# =============================================================================
# LUT GENERATION FAMILIES
# =============================================================================

def snap_to_fp16(lut: torch.Tensor, require_no_zero: bool = True) -> torch.Tensor:
    """Snap LUT values to FP16 and ensure strict monotonicity.

    Uses FP16 bit manipulation to guarantee distinct values after snapping.
    This is more robust than torch.nextafter() which operates in FP32 space.

    Args:
        lut: LUT tensor (assumed symmetric around 0)
        require_no_zero: If True, ensure no exact zero values

    Returns:
        FP16-snapped, strictly monotonic LUT with guaranteed distinct FP16 values
    """
    import struct

    def float_to_fp16_bits(f: float) -> int:
        """Convert float to FP16 bit representation."""
        return struct.unpack('H', struct.pack('e', f))[0]

    def fp16_bits_to_float(bits: int) -> float:
        """Convert FP16 bits back to float."""
        return struct.unpack('e', struct.pack('H', bits))[0]

    def next_fp16(f: float) -> float:
        """Get the next larger distinct FP16 value."""
        bits = float_to_fp16_bits(f)
        if f >= 0:
            return fp16_bits_to_float(bits + 1)
        else:
            return fp16_bits_to_float(bits - 1)

    def snap_single_to_fp16(f: float) -> float:
        """Snap a single float to nearest FP16."""
        return float(torch.tensor(f, dtype=torch.float32).half().item())

    # Initial FP16 snap
    lut = lut.half().float()
    half = len(lut) // 2

    # Work on positive half only, then mirror
    positive = lut[half:].clone()

    # Ensure no zero in positive half if required
    if require_no_zero and positive[0] == 0:
        # Use smallest positive FP16 subnormal: 2^-24 ≈ 5.96e-8
        # Or use 2^-14 ≈ 6.1e-5 for smallest normal
        positive[0] = torch.tensor(2**-14, dtype=torch.float32)

    # Ensure strict monotonicity using FP16 bit increment
    for i in range(1, len(positive)):
        prev_val = positive[i - 1].item()
        curr_val = positive[i].item()

        # Snap current to FP16
        curr_fp16 = snap_single_to_fp16(curr_val)

        # If not strictly greater, increment in FP16 space
        while curr_fp16 <= prev_val:
            curr_fp16 = next_fp16(curr_fp16)

        positive[i] = torch.tensor(curr_fp16, dtype=torch.float32)

    # Mirror to negative half
    negative = -positive.flip(0)
    lut = torch.cat([negative, positive])

    return lut


def validate_lut_fp16(lut: torch.Tensor) -> dict:
    """Validate LUT is FP16-safe: unique, strictly increasing, optionally no zero."""
    lut_fp16 = lut.half()
    return {
        'unique_fp16': int(torch.unique(lut_fp16).numel()),
        'expected_unique': len(lut),
        'strict_inc': bool((lut_fp16[1:] > lut_fp16[:-1]).all().item()),
        'has_zero': bool((lut_fp16 == 0).any().item()),
        'min_abs': float(lut_fp16.abs().min().item()),
        'max_abs': float(lut_fp16.abs().max().item()),
        'valid': int(torch.unique(lut_fp16).numel()) == len(lut) and (lut_fp16[1:] > lut_fp16[:-1]).all().item(),
    }


def generate_lut_family_A(lut_size: int, max_abs: float) -> List[Tuple[str, torch.Tensor]]:
    """Family A: Uniform linspace(-M, M, lut_size).

    Variations with different max_abs values.
    """
    candidates = []
    for m in [max_abs, max_abs * 1.5, max_abs * 2.0]:
        lut = torch.linspace(-m, m, lut_size, dtype=torch.float32)
        lut = snap_to_fp16(lut)
        candidates.append((f'A_uniform_M{m:.1f}', lut))
    return candidates


def generate_lut_family_B(lut_size: int, max_abs: float) -> List[Tuple[str, torch.Tensor]]:
    """Family B: Dense-center (more values near zero).

    Good for layers where most Q_eff values are small.
    """
    candidates = []
    half = lut_size // 2

    # B1: Asinh spacing - smooth compression near zero
    t = torch.linspace(0, 1, half)
    asinh_scale = 2.0
    positive = max_abs * torch.sinh(asinh_scale * t) / torch.sinh(torch.tensor(asinh_scale))
    lut = torch.cat([-positive.flip(0), positive])
    lut = snap_to_fp16(lut)
    candidates.append(('B_asinh_s2', lut))

    # B2: Tanh spacing
    tanh_scale = 1.5
    t = torch.linspace(0, 1, half)
    positive = max_abs * torch.tanh(tanh_scale * t) / torch.tanh(torch.tensor(tanh_scale))
    lut = torch.cat([-positive.flip(0), positive])
    lut = snap_to_fp16(lut)
    candidates.append(('B_tanh_s1.5', lut))

    # B3: Sqrt spacing (even denser near zero)
    t = torch.linspace(0, 1, half)
    positive = max_abs * torch.sqrt(t)
    positive[0] = max_abs * 0.01  # Avoid exact zero
    lut = torch.cat([-positive.flip(0), positive])
    lut = snap_to_fp16(lut)
    candidates.append(('B_sqrt', lut))

    return candidates


def generate_lut_family_C(lut_size: int, max_abs: float) -> List[Tuple[str, torch.Tensor]]:
    """Family C: Heavy-tail (more values at extremes).

    Good for layers with outliers or high saturation.
    """
    candidates = []
    half = lut_size // 2

    # C1: Geometric progression
    min_val = max_abs * 0.001  # Avoid exact zero
    positive = torch.logspace(math.log10(min_val), math.log10(max_abs), half)
    lut = torch.cat([-positive.flip(0), positive])
    lut = snap_to_fp16(lut)
    candidates.append(('C_geometric', lut))

    # C2: Power-2 (quadratic)
    t = torch.linspace(0, 1, half)
    positive = max_abs * (t ** 2)
    positive[0] = max_abs * 0.01  # Ensure nonzero minimum
    lut = torch.cat([-positive.flip(0), positive])
    lut = snap_to_fp16(lut)
    candidates.append(('C_power2', lut))

    # C3: Power-3 (cubic, even more heavy-tail)
    t = torch.linspace(0, 1, half)
    positive = max_abs * (t ** 3)
    positive[0] = max_abs * 0.001
    lut = torch.cat([-positive.flip(0), positive])
    lut = snap_to_fp16(lut)
    candidates.append(('C_power3', lut))

    return candidates


def generate_lut_family_D(
    lut_size: int,
    Q_eff_target: torch.Tensor,
    S: torch.Tensor,
    max_abs: float,
    eps: float = 1e-4,
    max_samples: int = 10_000_000,
) -> Tuple[str, torch.Tensor]:
    """Family D: Quantile LUT from layer's own |Q_eff|.

    Data-driven: places LUT values at data quantiles.
    CRITICAL: Excludes entries where |S| < eps to avoid division artifacts.

    Args:
        lut_size: Number of LUT entries
        Q_eff_target: Target Q values [out, in]
        S: Scale matrix [out, in] - used to mask small-scale entries
        max_abs: Maximum absolute value for LUT
        eps: Threshold for small scales
        max_samples: Maximum samples for quantile computation (avoids OOM)
    """
    # Mask out entries where S is too small (division artifacts)
    S_flat = S.flatten().abs()
    Q_flat = Q_eff_target.abs().flatten().float()

    valid_mask = S_flat >= eps
    Q_valid = Q_flat[valid_mask]

    if Q_valid.numel() < 100:
        # Fallback: not enough valid entries, use all
        print(f"    WARNING: Only {Q_valid.numel()} valid Q_eff entries (S >= {eps}), using all")
        Q_valid = Q_flat

    # Sample if tensor is too large (torch.quantile has limits)
    # Use strided sampling (works on TPU, avoids randperm issues)
    if Q_valid.numel() > max_samples:
        stride = Q_valid.numel() // max_samples
        Q_valid = Q_valid[::stride][:max_samples]

    half = lut_size // 2

    # Use percentiles to determine positive half
    # Good spread: 50, 70, 85, 92, 96, 98, 99.5, 99.9 (for 8 positive values)
    if half == 8:
        quantiles = torch.tensor([50, 70, 85, 92, 96, 98, 99.5, 99.9])
    else:
        # Generic spread for other sizes
        quantiles = torch.linspace(50, 99.9, half)

    positive = torch.quantile(Q_valid, quantiles / 100)

    # Clamp to max_abs
    positive = positive.clamp(max=max_abs)

    # Ensure minimum value is not zero
    min_val = max(positive[0].item(), max_abs * 0.001)
    positive[0] = torch.tensor(min_val, dtype=torch.float32)

    # Build full LUT and snap
    lut = torch.cat([-positive.flip(0), positive])
    lut = snap_to_fp16(lut)

    return ('D_quantile', lut)


def generate_lut_family_G(
    lut_size: int,
    Q_eff_target: torch.Tensor,
    S: torch.Tensor,
    max_abs: float,
    eps: float = 1e-4,
    max_iters: int = 20,
    max_samples: int = 1_000_000,
) -> Tuple[str, torch.Tensor]:
    """Family G: Weighted k-means codebook (Lloyd's algorithm).

    Unlike quantile LUTs, this directly minimizes weighted MSE distortion.
    Weights = S^2 (so errors in high-scale entries matter more).

    This often beats hand-crafted grids because it adapts to the actual
    distribution while optimizing the distortion objective.

    Args:
        lut_size: Number of LUT entries
        Q_eff_target: Target Q values [out, in]
        S: Scale matrix [out, in] - used as weights (S^2)
        max_abs: Maximum absolute value for LUT
        eps: Threshold for small scales
        max_iters: Maximum Lloyd iterations
        max_samples: Maximum samples to use (for speed)
    """
    half = lut_size // 2

    # Flatten and get valid entries
    Q_flat = Q_eff_target.flatten().float()
    S_flat = S.flatten().abs().float()
    weights = S_flat ** 2  # Weight by S^2 for MSE objective

    # Filter valid entries
    valid_mask = S_flat >= eps
    Q_valid = Q_flat[valid_mask]
    W_valid = weights[valid_mask]

    if Q_valid.numel() < 100:
        Q_valid = Q_flat
        W_valid = weights

    # Sample if too large
    if Q_valid.numel() > max_samples:
        stride = Q_valid.numel() // max_samples
        Q_valid = Q_valid[::stride][:max_samples]
        W_valid = W_valid[::stride][:max_samples]

    # Work on absolute values (symmetric LUT)
    Q_abs = Q_valid.abs()

    # Initialize centroids with quantiles (good starting point)
    quantiles = torch.linspace(0, 1, half + 2)[1:-1]  # Exclude 0 and 1
    centroids = torch.quantile(Q_abs, quantiles)

    # Lloyd's algorithm (weighted k-means)
    for _ in range(max_iters):
        # Assignment: find nearest centroid for each point
        dists = (Q_abs.unsqueeze(-1) - centroids.unsqueeze(0)).abs()
        assignments = dists.argmin(dim=-1)

        # Update: weighted mean for each cluster
        new_centroids = torch.zeros_like(centroids)
        for k in range(half):
            mask = assignments == k
            if mask.sum() > 0:
                new_centroids[k] = (Q_abs[mask] * W_valid[mask]).sum() / W_valid[mask].sum()
            else:
                new_centroids[k] = centroids[k]  # Keep old if empty cluster

        # Check convergence
        if (new_centroids - centroids).abs().max() < 1e-6:
            break
        centroids = new_centroids

    # Sort centroids (required for monotonic LUT)
    centroids, _ = centroids.sort()

    # Ensure minimum value and max_abs constraint
    centroids = centroids.clamp(min=max_abs * 0.001, max=max_abs)

    # Ensure strictly increasing
    for i in range(1, half):
        if centroids[i] <= centroids[i - 1]:
            centroids[i] = centroids[i - 1] + max_abs * 0.01

    # Build symmetric LUT
    lut = torch.cat([-centroids.flip(0), centroids])
    lut = snap_to_fp16(lut)

    return ('G_kmeans', lut)


def compute_weighted_mse(
    W_eff_old: torch.Tensor,
    W_eff_new: torch.Tensor,
    S: torch.Tensor,
    eps: float = 1e-4,
) -> float:
    """Compute S^2-weighted MSE (better proxy for PPL than MAE).

    For a linear layer y = W x, the output MSE depends on:
    - The weight error ΔW
    - The input activation covariance Σ_x

    If we approximate Σ_x ∝ I (identity), then we just use MSE.
    If we weight by S^2, we emphasize entries with larger scales
    (which typically correspond to more important weight directions).

    Returns:
        Weighted MSE value (lower is better)
    """
    valid_mask = S.abs() >= eps
    error_sq = (W_eff_old - W_eff_new) ** 2
    weights = S ** 2

    if valid_mask.sum() > 0:
        weighted_error = (error_sq * weights)[valid_mask]
        return weighted_error.mean().item()
    else:
        return (error_sq * weights).mean().item()


def compute_activation_weighted_mse(
    W_eff_old: torch.Tensor,
    W_eff_new: torch.Tensor,
    input_var: torch.Tensor,
    eps: float = 1e-8,
) -> float:
    """Compute activation-weighted MSE (Option A - best cost/benefit).

    For a linear layer y = W x, the expected output MSE is:
        E[||ΔW x||²] = tr(ΔW Σ_x ΔW^T)

    If we approximate Σ_x with diagonal (per-feature variance), then:
        score = Σ_j Var(x_j) * ||ΔW[:,j]||²

    This weights errors by how much each input feature varies during inference,
    which correlates better with PPL than raw weight MAE.

    Args:
        W_eff_old: Original effective weights [out_features, in_features]
        W_eff_new: New effective weights [out_features, in_features]
        input_var: Per-input-feature variance [in_features] from calibration
        eps: Small value to avoid division by zero

    Returns:
        Activation-weighted MSE score (lower is better)
    """
    # Weight error per element
    delta_W = W_eff_old - W_eff_new  # [out, in]

    # Squared column norms: ||ΔW[:,j]||² for each input feature j
    col_sq_norms = (delta_W ** 2).sum(dim=0)  # [in_features]

    # Ensure input_var matches shape
    if input_var.shape[0] != col_sq_norms.shape[0]:
        # Handle shape mismatch (shouldn't happen if calibration is correct)
        return (delta_W ** 2).mean().item()

    # Weight by input variance
    # score = Σ_j Var(x_j) * ||ΔW[:,j]||²
    weighted_score = (input_var * col_sq_norms).sum()

    # Normalize by total variance to make scores comparable across layers
    total_var = input_var.sum() + eps
    normalized_score = weighted_score / total_var

    return normalized_score.item()


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def quantize_chunked(Q_target: torch.Tensor, lut: torch.Tensor, chunk_size: int = 8192) -> torch.Tensor:
    """Memory-safe nearest-LUT assignment."""
    flat = Q_target.flatten()
    indices = torch.empty_like(flat, dtype=torch.long)

    for i in range(0, flat.numel(), chunk_size):
        end = min(i + chunk_size, flat.numel())
        chunk = flat[i:end]
        dist = (chunk.unsqueeze(-1) - lut).abs()
        indices[i:end] = dist.argmin(dim=-1)

    return indices.view_as(Q_target)


def matches_scope(name: str, scope: str) -> bool:
    """Check if module name matches the target scope."""
    if scope == 'all':
        return True
    elif scope == 'mlp':
        return '.mlp.' in name
    elif scope == 'attn':
        return '.self_attn.' in name or '_attn.' in name
    return True


def parse_layer_kind(name: str) -> Tuple[int, str]:
    """Extract layer index and kind from module path."""
    layer_match = re.search(r'\.layers\.(\d+)\.', name)
    layer_idx = int(layer_match.group(1)) if layer_match else -1

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


def load_config_json(ckpt_dir: Path) -> dict:
    """Load config.json from checkpoint directory if it exists."""
    for name in ['config.json', 'v2_config.json']:
        config_path = ckpt_dir / name
        if config_path.exists():
            with open(config_path) as f:
                return json.load(f)
    return {}


@torch.no_grad()
def compute_layer_weff_and_qeff(
    module,  # AnemllQATLinearV2
    eps: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute W_eff_old, S, and Q_eff_target for a layer.

    Returns:
        W_eff_old: Current effective weights [out, in]
        S: Full scales [out, in]
        Q_eff_target: Target Q values [out, in]
    """
    # Get current Q - CRITICAL: use get_lut() which handles trainable LUT case
    # If LUT training is enabled, get_lut() builds LUT from _lut_raw_deltas
    lut = module.get_lut().float().cpu() if hasattr(module, 'get_lut') else module.lut.float().cpu()
    if module._Q is not None:
        Q_current = module._Q.float().cpu()
    elif module._indices is not None:
        # CRITICAL: move _indices to CPU before indexing CPU lut
        Q_current = lut[module._indices.long().cpu()]
    else:
        raise ValueError("Layer has no _Q or _indices")

    # Compute scales
    S = module._compute_full_scales().float().cpu()

    # Safe scales - handle S==0 correctly (torch.sign(0)==0)
    # Use torch.copysign to preserve sign, defaulting to +eps for zero
    S_safe = torch.where(
        S.abs() < eps,
        torch.copysign(torch.full_like(S, eps), S + eps),  # +eps ensures 0 -> +eps
        S
    )

    # Current effective weights
    W_eff_old = Q_current * S

    # Target Q values
    Q_eff_target = W_eff_old / S_safe

    return W_eff_old, S, Q_eff_target


@torch.no_grad()
def apply_lut_candidate(
    W_eff_old: torch.Tensor,
    S: torch.Tensor,
    Q_eff_target: torch.Tensor,
    lut_new: torch.Tensor,
    eps: float = 1e-4,
    chunk_size: int = 8192,
) -> Tuple[torch.Tensor, Dict]:
    """Apply new LUT to layer, keeping S fixed.

    CRITICAL: Uses same valid_mask (|S| >= eps) for all error metrics to ensure
    consistency with quantile LUT generation.

    Returns:
        indices_new: New quantization indices
        stats: Dict with mae, p99_error, pct_outside_lut (all computed on valid entries)
    """
    # Find nearest LUT indices
    indices_new = quantize_chunked(Q_eff_target, lut_new, chunk_size)

    # Compute new W_eff
    Q_new = lut_new[indices_new]
    W_eff_new = Q_new * S

    # CRITICAL: Use same valid_mask as quantile LUT generation
    # This ensures consistent ranking across candidates
    valid_mask = S.abs() >= eps
    n_valid = valid_mask.sum().item()

    # Compute error metrics on VALID entries only (where |S| >= eps)
    error = (W_eff_old - W_eff_new).abs()
    error_valid = error[valid_mask]

    if n_valid > 0:
        mae = error_valid.mean().item()
        p99_error = torch.quantile(error_valid.flatten().float(), 0.99).item()
    else:
        # Fallback: use all entries if no valid ones
        mae = error.mean().item()
        p99_error = torch.quantile(error.flatten().float(), 0.99).item()

    # Check saturation: how much Q_eff_target is outside LUT range (on valid entries)
    lut_min, lut_max = lut_new.min().item(), lut_new.max().item()
    Q_flat = Q_eff_target.flatten()
    valid_flat = valid_mask.flatten()

    if n_valid > 0:
        Q_valid = Q_flat[valid_flat]
        pct_outside = ((Q_valid < lut_min) | (Q_valid > lut_max)).float().mean().item() * 100
    else:
        pct_outside = ((Q_flat < lut_min) | (Q_flat > lut_max)).float().mean().item() * 100

    # Also compute max error for sanity checking
    max_error = error_valid.max().item() if n_valid > 0 else error.max().item()

    stats = {
        'mae': mae,
        'p99_error': p99_error,
        'max_error': max_error,
        'pct_outside_lut': pct_outside,
        'n_valid': n_valid,
        'n_total': S.numel(),
        'pct_valid': n_valid / S.numel() * 100 if S.numel() > 0 else 0,
    }

    return indices_new, stats


@torch.no_grad()
def bake_lut_into_module(
    module,  # AnemllQATLinearV2
    lut_new: torch.Tensor,
    indices_new: torch.Tensor,
    validate: bool = True,
):
    """Update module with new LUT and indices.

    CRITICAL: Also sets _use_indices=True so forward() uses _indices.

    Args:
        module: AnemllQATLinearV2 module to update
        lut_new: New LUT tensor
        indices_new: New quantization indices
        validate: If True, run post-bake validation assertions
    """
    device = module.lut.device
    dtype = module.lut.dtype
    lut_size = lut_new.numel()

    # Update LUT
    if isinstance(module.lut, nn.Parameter):
        module.lut.data = lut_new.to(dtype).to(device)
    else:
        module.lut = lut_new.to(dtype).to(device)

    # Update indices (int16 for memory efficiency)
    if module._indices is not None:
        idx_dtype = module._indices.dtype
        module._indices = indices_new.to(idx_dtype).to(device)
    else:
        module.register_buffer("_indices", indices_new.to(torch.int16).to(device))

    # CRITICAL: Set _use_indices so forward() uses the new indices
    module._use_indices = True

    # Update _Q buffer (for inference path that uses _Q directly)
    Q_new = lut_new[indices_new.long()]
    if module._Q is not None:
        module._Q = Q_new.to(dtype).to(device)
    else:
        module.register_buffer("_Q", Q_new.to(dtype).to(device))

    # VALIDATION: Ensure everything is consistent
    if validate:
        # 1. _indices must exist
        assert module._indices is not None, "BAKE FAIL: _indices is None after bake"

        # 2. _indices must be on same device as lut
        assert module._indices.device == module.lut.device, \
            f"BAKE FAIL: _indices on {module._indices.device} but lut on {module.lut.device}"

        # 3. _indices must be in valid range [0, lut_size)
        idx_min = module._indices.min().item()
        idx_max = module._indices.max().item()
        assert idx_min >= 0, f"BAKE FAIL: _indices.min()={idx_min} < 0"
        assert idx_max < lut_size, f"BAKE FAIL: _indices.max()={idx_max} >= lut_size={lut_size}"

        # 4. Reconstructed Q must match stored Q_new (within numerical tolerance)
        Q_reconstructed = module.lut[module._indices.long()]
        Q_stored = module._Q
        if Q_stored is not None:
            max_diff = (Q_reconstructed - Q_stored).abs().max().item()
            assert max_diff < 1e-5, f"BAKE FAIL: Q mismatch, max_diff={max_diff}"


@torch.no_grad()
def verify_weff_reconstruction(
    module,  # AnemllQATLinearV2
    W_eff_old: torch.Tensor,
    eps: float = 1e-4,
) -> Dict:
    """End-to-end sanity check: verify W_eff reconstruction after baking.

    This is THE critical test. If this fails, the LUT candidate comparison is invalid.

    Returns:
        Dict with max_error, p99_error, mean_error, pass (bool)
    """
    # Recompute W_eff_new from baked module state
    # CRITICAL: use get_lut() which handles trainable LUT case
    lut = module.get_lut().float().cpu() if hasattr(module, 'get_lut') else module.lut.float().cpu()
    if module._Q is not None:
        Q_new = module._Q.float().cpu()
    elif module._indices is not None:
        # CRITICAL: move _indices to CPU before indexing CPU lut
        Q_new = lut[module._indices.long().cpu()]
    else:
        return {'pass': False, 'error': 'No _Q or _indices after bake'}

    S = module._compute_full_scales().float().cpu()
    W_eff_new = Q_new * S

    # Use same valid_mask
    valid_mask = S.abs() >= eps
    W_eff_old_cpu = W_eff_old.float().cpu()

    error = (W_eff_old_cpu - W_eff_new).abs()
    error_valid = error[valid_mask] if valid_mask.any() else error

    max_error = error_valid.max().item()
    p99_error = torch.quantile(error_valid.flatten().float(), 0.99).item()
    mean_error = error_valid.mean().item()

    # The errors should be small (quantization noise level)
    # If max_error is huge (>1.0), something is very wrong
    passed = max_error < 1.0  # Very generous threshold

    return {
        'max_error': max_error,
        'p99_error': p99_error,
        'mean_error': mean_error,
        'pass': passed,
    }


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate and apply LUT candidates to V2 checkpoints',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument('checkpoint', help='Input V2 checkpoint path')
    parser.add_argument('-b', '--base-dir', metavar='FOLDER',
                        help='Base folder for checkpoint and output paths')
    parser.add_argument('--output-dir', '-o', required=True,
                        help='Output directory for checkpoint variants')
    parser.add_argument('--config', default=None,
                        help='Config preset (q2a4, q4a4, etc.) or path')
    parser.add_argument('--model-id', default='Qwen/Qwen3-0.6B',
                        help='Base model name')
    parser.add_argument('--scope', choices=['mlp', 'attn', 'all'], default='all',
                        help='Layer scope to modify (default: all)')
    parser.add_argument('--families', default='A,B,C,D',
                        help='Comma-separated family codes: A=uniform, B=dense-center, C=heavy-tail, D=quantile')
    parser.add_argument('--max-abs', type=float, default=1.0,
                        help='Max absolute LUT value (default: 1.0)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Compute MAE without saving checkpoints')
    parser.add_argument('--verbose', action='store_true',
                        help='Print per-layer MAE')
    parser.add_argument('--eps', type=float, default=1e-4,
                        help='Epsilon for safe division (default: 1e-4)')
    parser.add_argument('--chunk-size', type=int, default=8192,
                        help='Chunk size for quantization (default: 8192)')

    args = parser.parse_args()

    # Apply base directory if specified
    if args.base_dir:
        base = Path(args.base_dir)
        args.checkpoint = str(base / args.checkpoint)
        args.output_dir = str(base / args.output_dir)

    return args


def main():
    args = parse_args()

    print("=" * 60)
    print("APPLY LUT CANDIDATES")
    print("=" * 60)
    print(f"Checkpoint:  {args.checkpoint}")
    print(f"Output dir:  {args.output_dir}")
    print(f"Families:    {args.families}")
    print(f"Scope:       {args.scope}")
    print(f"Max abs:     {args.max_abs}")
    print()

    # Parse families
    families = [f.strip().upper() for f in args.families.split(',')]
    print(f"Generating LUT families: {families}")

    # Load config
    ckpt_path = Path(args.checkpoint)
    ckpt_config = load_config_json(ckpt_path.parent)

    if args.config and args.config in CONFIG_PRESETS:
        preset = CONFIG_PRESETS[args.config]
    elif 'config' in ckpt_config:
        preset_name = ckpt_config.get('config', 'q4a4_r32')
        preset = CONFIG_PRESETS.get(preset_name, CONFIG_PRESETS['q4a4_r32'])
    else:
        preset = CONFIG_PRESETS['q4a4_r32']

    mlp_lut = preset['mlp_lut']
    mlp_rank = preset['mlp_rank']
    attn_lut = preset['attn_lut']
    attn_rank = preset['attn_rank']
    print(f"Config: MLP LUT{mlp_lut} rank={mlp_rank}, Attn LUT{attn_lut} rank={attn_rank}")
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
    for name, m in model.named_modules():
        if isinstance(m, AnemllQATLinearV2):
            q_key = f"{name}._Q"
            if q_key in state_dict and m._Q is None:
                m.register_buffer("_Q", state_dict[q_key])

    # Collect layer data for quantile LUT (family D)
    print("\nCollecting layer data...")
    layer_data = {}  # name -> (W_eff_old, S, Q_eff_target, lut_size)

    for name, module in model.named_modules():
        if not isinstance(module, AnemllQATLinearV2):
            continue
        if not matches_scope(name, args.scope):
            continue
        if module._Q is None and module._indices is None:
            print(f"  WARNING: {name} has no _Q or _indices, skipping")
            continue

        W_eff_old, S, Q_eff_target = compute_layer_weff_and_qeff(module, eps=args.eps)
        layer_data[name] = (W_eff_old, S, Q_eff_target, module.lut.numel())

    print(f"  Collected data for {len(layer_data)} layers")

    # Determine LUT sizes
    lut_sizes = set(data[3] for data in layer_data.values())
    print(f"  LUT sizes: {sorted(lut_sizes)}")

    # Generate candidates for each LUT size
    print("\nGenerating LUT candidates...")
    candidates_by_size = {}  # lut_size -> [(name, lut_tensor), ...]

    for lut_size in lut_sizes:
        candidates = []

        if 'A' in families:
            for name, lut in generate_lut_family_A(lut_size, args.max_abs):
                v = validate_lut_fp16(lut)
                if not v['valid']:
                    print(f"    WARNING: {name} FP16 invalid: unique={v['unique_fp16']}")
                candidates.append((name, lut))

        if 'B' in families:
            for name, lut in generate_lut_family_B(lut_size, args.max_abs):
                v = validate_lut_fp16(lut)
                if not v['valid']:
                    print(f"    WARNING: {name} FP16 invalid: unique={v['unique_fp16']}")
                candidates.append((name, lut))

        if 'C' in families:
            for name, lut in generate_lut_family_C(lut_size, args.max_abs):
                v = validate_lut_fp16(lut)
                if not v['valid']:
                    print(f"    WARNING: {name} FP16 invalid: unique={v['unique_fp16']}")
                candidates.append((name, lut))

        candidates_by_size[lut_size] = candidates
        print(f"  LUT{lut_size}: {len(candidates)} candidates from families {[f for f in 'ABC' if f in families]}")

    # For family D (quantile), we need to generate per-layer
    # For simplicity, use aggregated Q_eff from all layers of same LUT size
    if 'D' in families:
        print("  Generating quantile LUTs (family D)...")
        for lut_size in lut_sizes:
            # Aggregate Q_eff AND S from all layers with this LUT size
            all_Q_eff = []
            all_S = []
            for layer_name, (W_eff_old, S, Q_eff_target, ls) in layer_data.items():
                if ls == lut_size:
                    all_Q_eff.append(Q_eff_target.flatten())
                    all_S.append(S.flatten())

            if all_Q_eff:
                combined_Q_eff = torch.cat(all_Q_eff)
                combined_S = torch.cat(all_S)
                cand_name, lut = generate_lut_family_D(
                    lut_size, combined_Q_eff, combined_S, args.max_abs, eps=args.eps
                )
                # Validate FP16
                validation = validate_lut_fp16(lut)
                if not validation['valid']:
                    print(f"    WARNING: {cand_name} FP16 validation failed: {validation}")
                candidates_by_size[lut_size].append((cand_name, lut))
                print(f"    LUT{lut_size}: Added {cand_name} (unique={validation['unique_fp16']})")

    # Evaluate candidates
    print("\nEvaluating candidates...")
    results = []  # List of (candidate_name, lut_size, avg_mae, max_mae, avg_p99, avg_outside, per_layer_stats, lut_new)

    for lut_size, candidates in candidates_by_size.items():
        layers_with_size = [(n, d) for n, d in layer_data.items() if d[3] == lut_size]

        for cand_name, lut_new in candidates:
            per_layer_stats = {}
            total_mae = 0
            max_mae = 0
            total_p99 = 0
            total_outside = 0

            for name, (W_eff_old, S, Q_eff_target, _) in layers_with_size:
                indices_new, stats = apply_lut_candidate(
                    W_eff_old, S, Q_eff_target, lut_new,
                    eps=args.eps, chunk_size=args.chunk_size
                )
                per_layer_stats[name] = stats
                total_mae += stats['mae']
                max_mae = max(max_mae, stats['mae'])
                total_p99 += stats['p99_error']
                total_outside += stats['pct_outside_lut']

                if args.verbose:
                    print(f"    {cand_name} | {name}: MAE={stats['mae']:.6f}, "
                          f"p99={stats['p99_error']:.6f}, max={stats['max_error']:.6f}, "
                          f"outside={stats['pct_outside_lut']:.2f}%, valid={stats['pct_valid']:.1f}%")

            n_layers = len(layers_with_size)
            avg_mae = total_mae / n_layers if n_layers else 0
            avg_p99 = total_p99 / n_layers if n_layers else 0
            avg_outside = total_outside / n_layers if n_layers else 0

            results.append((cand_name, lut_size, avg_mae, max_mae, avg_p99, avg_outside, per_layer_stats, lut_new))

            # Sanity assertion: warn if high outside percentage
            outside_warn = " ⚠️" if avg_outside > 1.0 else ""
            print(f"  {cand_name} (LUT{lut_size}): MAE={avg_mae:.6f}, p99={avg_p99:.6f}, "
                  f"outside={avg_outside:.2f}%{outside_warn}")

    # Sort by avg_mae
    results.sort(key=lambda x: x[2])

    # Summary
    print()
    print("=" * 60)
    print("CANDIDATE RANKING (by avg MAE)")
    print("=" * 60)
    for i, (cand_name, lut_size, avg_mae, max_mae, avg_p99, avg_outside, _, _) in enumerate(results[:10], 1):
        outside_warn = " ⚠️" if avg_outside > 1.0 else ""
        print(f"  {i}. {cand_name} (LUT{lut_size}): MAE={avg_mae:.6f}, p99={avg_p99:.6f}, "
              f"outside={avg_outside:.1f}%{outside_warn}")

    # Save checkpoints
    if not args.dry_run:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nSaving checkpoint variants to {output_dir}/...")

        summary = {
            'source_checkpoint': str(args.checkpoint),
            'timestamp': datetime.now().isoformat(),
            'scope': args.scope,
            'max_abs': args.max_abs,
            'candidates': [],
        }

        for cand_name, lut_size, avg_mae, max_mae, avg_p99, avg_outside, per_layer_stats, lut_new in results:
            # Reload model fresh for each candidate
            model_copy = AutoModelForCausalLM.from_pretrained(
                args.model_id,
                torch_dtype=torch.float32,
                trust_remote_code=True,
            )
            replace_linear_with_anemll_v2(
                model_copy,
                mlp_config=mlp_config,
                attn_config=attn_config,
                quantize_attn=True,
                quantize_lm_head=False,
                skip_init=True,
            )
            model_copy.load_state_dict(state_dict, strict=False)

            # Load _Q buffers
            for name, m in model_copy.named_modules():
                if isinstance(m, AnemllQATLinearV2):
                    q_key = f"{name}._Q"
                    if q_key in state_dict and m._Q is None:
                        m.register_buffer("_Q", state_dict[q_key])

            # Apply LUT to matching layers
            for name, module in model_copy.named_modules():
                if not isinstance(module, AnemllQATLinearV2):
                    continue
                if not matches_scope(name, args.scope):
                    continue
                if module.lut.numel() != lut_size:
                    continue
                if module._Q is None and module._indices is None:
                    continue

                W_eff_old, S, Q_eff_target = compute_layer_weff_and_qeff(module, eps=args.eps)
                indices_new, _ = apply_lut_candidate(
                    W_eff_old, S, Q_eff_target, lut_new,
                    eps=args.eps, chunk_size=args.chunk_size
                )
                bake_lut_into_module(module, lut_new, indices_new, validate=True)

                # End-to-end verification (sample first layer only for speed)
                if args.verbose and name == list(layer_data.keys())[0]:
                    verify_result = verify_weff_reconstruction(module, W_eff_old, eps=args.eps)
                    if not verify_result['pass']:
                        print(f"    ⚠️ VERIFY FAIL {name}: max_err={verify_result['max_error']:.6f}")
                    else:
                        print(f"    ✅ VERIFY OK {name}: max_err={verify_result['max_error']:.6f}")

            # Save checkpoint
            ckpt_path = output_dir / f"{cand_name}.pt"
            torch.save(model_copy.state_dict(), ckpt_path)
            size_mb = ckpt_path.stat().st_size / (1024 * 1024)
            print(f"  Saved: {ckpt_path.name} ({size_mb:.1f} MB)")

            summary['candidates'].append({
                'name': cand_name,
                'lut_size': lut_size,
                'checkpoint_path': str(ckpt_path),
                'avg_mae': round(avg_mae, 8),
                'max_mae': round(max_mae, 8),
                'avg_p99_error': round(avg_p99, 8),
                'avg_pct_outside_lut': round(avg_outside, 4),
                'per_layer_stats': {
                    k: {sk: round(sv, 8) for sk, sv in v.items()}
                    for k, v in list(per_layer_stats.items())[:5]
                },  # Sample first 5 layers
            })

            del model_copy

        # Save summary
        summary_path = output_dir / 'candidates_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved: {summary_path}")

    else:
        print("\n(Dry run - no checkpoints saved)")

    print("\nDone!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
