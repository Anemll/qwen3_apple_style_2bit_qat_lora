"""
Anemll-style QATLinear V2 - ANE-Friendly Rank-by-Rank Forward Pass.

Key changes from V1:
- Rank-by-rank forward: y = Σₖ gₖ · (aₖ ⊙ (Q (bₖ ⊙ x)))
- Per-rank normalization: A_dir (unit columns), B_dir (unit rows), magnitude
- Q is frozen buffer (computed once at init/snap)
- No A @ B materialization in forward pass

Target formula:
    y = Σₖ gₖ · (A_dir[:, k] ⊙ (Q (B_dir[k, :] ⊙ x)))

Where:
- A_dir: [out, rank] unit-norm columns
- B_dir: [rank, in] unit-norm rows
- gₖ = rank_magnitude[k] (the ONLY magnitude)
- Q = lut[indices] frozen buffer in [-1, 1]
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# CONFIGURATION (same as V1)
# =============================================================================

@dataclass
class AnemllQuantConfigV2:
    """Configuration for Anemll-style LUT quantization V2."""

    lut_size: int = 16  # Number of LUT entries (4-bit = 16, 2-bit = 4)
    scale_rank: int = 4  # Rank for low-rank scales
    group_size: int = 128  # Group size for scale initialization (same as V1)
    lut_include_zero: bool = False  # Whether LUT includes 0
    learnable_lut: bool = False  # Whether LUT values are trainable

    # Scale constraints (to avoid clamp(A@B) materialization in forward)
    # NOTE: When Q (LUT indices) is frozen, letting scales go negative can flip effective weight signs.
    # For the 'freeze Q + train scales' pipeline, keeping scales nonnegative is the safest choice.
    force_positive_scales: bool = True  # enforce S >= 0 by construction
    positive_scale_method: str = "abs"  # "abs" or "softplus"
    magnitude_activation: str = "softplus"  # "identity" | "abs" | "softplus"
    magnitude_eps: float = 1e-6

    # FP16-safe epsilon for normalization (1e-8 underflows in FP16)
    norm_eps: float = 1e-6

    # STE-FP16: Apply FP16 rounding in forward with FP32 gradients (for ANE matching)
    use_ste_fp16: bool = False

    # === QRANKLUT: Quantize A_dir/B_dir with k-means LUT ===
    # When enabled, A_dir and B_dir are quantized using per-layer k-means LUTs.
    # Values are in [-1, 1] since A_dir/B_dir are unit-normalized.
    rank_lut_size: int = 0  # 0 = disabled, 64 = 6-bit, 256 = 8-bit
    rank_lut_learnable: bool = True  # LUT values are trainable
    rank_lut_frozen_idx: bool = True  # Indices frozen (start frozen, can unfreeze)
    rank_lut_signed: bool = True  # Codebook in [-1, 1] (vs [0, 1])

    @property
    def lut_bits(self) -> int:
        return int(math.ceil(math.log2(self.lut_size)))

    @property
    def rank_lut_bits(self) -> int:
        """Bits for rank LUT (0 if disabled)."""
        if self.rank_lut_size <= 0:
            return 0
        return int(math.ceil(math.log2(self.rank_lut_size)))


# =============================================================================
# LUT UTILITIES
# =============================================================================

def make_lut(
    lut_size: int,
    device: torch.device,
    dtype: torch.dtype,
    include_zero: bool = False,
) -> torch.Tensor:
    """Create monotonic LUT in [-1, 1].

    IMPORTANT: LUT values are created in FP16 first, then upcast to target dtype.
    This ensures all values are FP16-representable, preventing precision loss
    when snapping to FP16 for ANE export.
    """
    if lut_size < 2:
        raise ValueError("lut_size must be >= 2")
    if not include_zero or (lut_size % 2 == 1):
        # Create in FP16 first, then cast to target dtype for FP16-representable values
        lut_fp16 = torch.linspace(-1.0, 1.0, steps=lut_size, device='cpu', dtype=torch.float16)
        return lut_fp16.to(device=device, dtype=dtype)

    # Even size with zero: non-uniform but includes 0
    # Create in FP16 first for FP16-representable values
    neg = torch.linspace(-1.0, 0.0, steps=lut_size // 2 + 1, device='cpu', dtype=torch.float16)
    pos = torch.linspace(0.0, 1.0, steps=lut_size // 2, device='cpu', dtype=torch.float16)
    lut_fp16 = torch.cat([neg[:-1], pos], dim=0)
    return lut_fp16.to(device=device, dtype=dtype)


def quantize_to_lut_indices(
    normalized: torch.Tensor,
    lut_size: int,
    include_zero: bool = False,
    lut: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Map normalized values [-1, 1] to nearest LUT indices.

    Args:
        normalized: Tensor of values in [-1, 1] to quantize
        lut_size: Number of LUT entries (used for uniform fallback)
        include_zero: Whether LUT includes exact zero (for uniform fallback)
        lut: Optional actual LUT tensor. If provided, finds nearest LUT value
             for each input. If None, assumes uniform spacing.

    Returns:
        Tensor of LUT indices (same shape as normalized)
    """
    # If actual LUT provided, find nearest value in LUT
    if lut is not None:
        # Clamp to LUT's range (supports fp4_x LUTs with values >1)
        lut_f = lut.float()
        lut_min, lut_max = lut_f.min().item(), lut_f.max().item()
        x = normalized.clamp(lut_min, lut_max)

        # lut: [lut_size], x: [*, in] -> distances: [*, in, lut_size]
        # Use broadcasting: x[..., None] - lut[None, :]
        lut_f = lut_f.to(x.device)
        orig_shape = x.shape
        x_flat = x.flatten()  # [N]

        # Compute distance to each LUT value: [N, lut_size]
        distances = (x_flat.unsqueeze(1) - lut_f.unsqueeze(0)).abs()

        # Find index of minimum distance
        indices = distances.argmin(dim=1)  # [N]
        return indices.view(orig_shape).long()

    # Legacy fallback: uniform LUT assumption (clamp to [-1, 1])
    x = normalized.clamp(-1.0, 1.0)

    # Fallback: uniform LUT assumption (legacy behavior)
    if not include_zero or (lut_size % 2 == 1):
        # Uniform LUT
        step = 2.0 / (lut_size - 1)
        return torch.round((x + 1.0) / step).long().clamp(0, lut_size - 1)

    # Non-uniform with zero
    half = lut_size // 2
    step_neg = 1.0 / half
    step_pos = 1.0 / max(1, half - 1)

    y_neg = (x + 1.0) / step_neg
    y_pos = (x / step_pos) + half
    y = torch.where(x < 0, y_neg, y_pos)
    return torch.round(y).long().clamp(0, lut_size - 1)


# =============================================================================
# QRANKLUT: K-MEANS LUT FOR A_dir/B_dir QUANTIZATION
# =============================================================================

def kmeans_1d(
    data: torch.Tensor,
    k: int,
    max_iters: int = 100,
    tol: float = 1e-6,
) -> torch.Tensor:
    """1D k-means clustering (Lloyd-Max algorithm).

    Clusters 1D data into k centroids. Much faster than sklearn for 1D case.

    Args:
        data: 1D tensor of values to cluster
        k: Number of centroids (LUT size)
        max_iters: Maximum iterations
        tol: Convergence tolerance

    Returns:
        Sorted centroids tensor of shape [k]
    """
    data = data.flatten().float()
    n = data.numel()

    if n == 0:
        return torch.linspace(-1.0, 1.0, k, device=data.device, dtype=data.dtype)

    # Initialize centroids with quantile-based spacing
    # This is faster and more stable than random init for 1D
    quantiles = torch.linspace(0, 1, k, device=data.device)
    sorted_data = data.sort().values
    indices = (quantiles * (n - 1)).long().clamp(0, n - 1)
    centroids = sorted_data[indices].clone()

    for _ in range(max_iters):
        # Assignment: find nearest centroid for each point
        # distances: [n, k]
        distances = (data.unsqueeze(1) - centroids.unsqueeze(0)).abs()
        assignments = distances.argmin(dim=1)

        # Update: compute mean of each cluster
        new_centroids = torch.zeros_like(centroids)
        counts = torch.zeros(k, device=data.device)

        for j in range(k):
            mask = (assignments == j)
            count = mask.sum()
            if count > 0:
                new_centroids[j] = data[mask].mean()
                counts[j] = count
            else:
                # Empty cluster: keep old centroid
                new_centroids[j] = centroids[j]

        # Check convergence
        if (new_centroids - centroids).abs().max() < tol:
            centroids = new_centroids
            break

        centroids = new_centroids

    # Sort centroids (monotonic for fast bucketize)
    return centroids.sort().values


def assign_to_lut(
    data: torch.Tensor,
    lut: torch.Tensor,
) -> torch.Tensor:
    """Assign values to nearest LUT entry (for k-means LUT).

    Uses bucketize for sorted LUT (fast O(n log k)).

    Args:
        data: Tensor of values to quantize
        lut: Sorted 1D LUT tensor of shape [k]

    Returns:
        Index tensor of same shape as data
    """
    k = lut.numel()
    if k <= 1:
        return torch.zeros_like(data, dtype=torch.long)

    # Compute midpoints between consecutive centroids
    # boundaries[j] = (lut[j] + lut[j+1]) / 2
    boundaries = (lut[:-1] + lut[1:]) / 2  # [k-1]

    # bucketize: find which bucket each value falls into
    # Returns index i such that boundaries[i-1] < x <= boundaries[i]
    indices = torch.bucketize(data, boundaries)  # [0, k-1]

    return indices.clamp(0, k - 1)


def qranklut_fake_quant(
    x: torch.Tensor,
    lut: torch.Tensor,
    frozen_idx: bool = True,
) -> torch.Tensor:
    """Apply qranklut fake quantization with STE.

    Args:
        x: Input tensor (any shape)
        lut: Sorted 1D LUT tensor [k]
        frozen_idx: If True, indices are frozen (gradients only to LUT values)
                   If False, use STE for index selection

    Returns:
        Quantized tensor with same shape as x, gradients flow through STE
    """
    # Assign to nearest LUT entry
    idx = assign_to_lut(x, lut)

    # Dequantize
    x_q = lut[idx]

    # STE: forward uses quantized, backward passes through unchanged
    return x + (x_q - x).detach()


# =============================================================================
# TRAINABLE LUT UTILITIES
# =============================================================================

def compute_min_delta(lut_max_abs: float) -> float:
    """Minimum normalized delta to prevent FP16 collapse/duplicates.

    For LUT16: 8 positive values from ε to max_abs.
    Each delta must be large enough that:
    1. ε = first delta doesn't snap to 0 in FP16
    2. Adjacent values don't collide after FP16 snap
    """
    # FP16 ULP at scale max_abs
    exp = math.ceil(math.log2(max(lut_max_abs, 1e-6)))
    ulp_at_max = 2 ** (exp - 10)

    # Conservative: 2x ULP, minimum 1e-4 for safety
    return max(2 * ulp_at_max, 1e-4)


def snap_to_fp16_ste(x: torch.Tensor) -> torch.Tensor:
    """Snap to FP16 in forward, pass gradients through in backward."""
    x_fp16 = x.to(torch.float16).to(x.dtype)
    return x + (x_fp16 - x).detach()


def build_symmetric_lut16(
    raw_deltas: torch.Tensor,
    max_abs: float,
    min_delta: float,
) -> torch.Tensor:
    """Build globally increasing symmetric LUT16 using constrained simplex.

    CRITICAL: Uses floor + softmax to guarantee min_delta AFTER normalization.
    This prevents FP16 collapse to 0 or duplicates.

    Args:
        raw_deltas: Shape (8,) - learnable (unnormalized logits)
        max_abs: Maximum absolute value (e.g., 1.0, 2.0)
        min_delta: Minimum per-value delta (must be FP16-safe)

    Returns:
        Full LUT of shape (16,) - globally increasing, symmetric, no 0
        Order: [-max_abs, ..., -ε, +ε, ..., +max_abs]
    """
    n_deltas = raw_deltas.shape[0]  # 8

    # Constrained simplex: delta_i = min_delta + (remaining) * softmax(raw)[i]
    # This guarantees each delta >= min_delta even after normalization
    remaining = max_abs - n_deltas * min_delta  # Space for learnable distribution
    weights = F.softmax(raw_deltas, dim=0)       # Sum to 1
    pos_deltas_norm = min_delta + remaining * weights  # Each >= min_delta, sum = max_abs

    # Positive side: cumsum gives 8 values ending at max_abs
    positive_lut = torch.cumsum(pos_deltas_norm, dim=0)
    # positive_lut[0] >= min_delta (never 0 in FP16)
    # positive_lut[7] = max_abs (exactly)

    # Negative side: mirror in increasing order (from -max_abs to -ε)
    negative_lut = -positive_lut.flip(0)

    # Full symmetric LUT: globally increasing
    full_lut = torch.cat([negative_lut, positive_lut])

    return snap_to_fp16_ste(full_lut)


def repair_lut_duplicates_symmetric(
    lut: torch.Tensor,
    max_abs: float,
) -> torch.Tensor:
    """Fix FP16 duplicates while preserving symmetry. CPU-only.

    Strategy: endpoint-first to prevent repair from overshooting.
    """
    lut = lut.detach().cpu().clone()
    half = len(lut) // 2  # 8

    # Extract positive side (indices 8-15)
    positive = lut[half:].to(torch.float16)

    # STEP 1: Fix endpoint FIRST
    max_abs_fp16 = torch.tensor(max_abs, dtype=torch.float16)
    positive[-1] = max_abs_fp16

    # STEP 2: Repair left→right, but never exceed endpoint
    # Work backwards to compute maximum allowed values
    max_allowed = torch.zeros_like(positive)
    max_allowed[-1] = max_abs_fp16
    for i in range(len(positive) - 2, -1, -1):
        # Each value must be strictly less than next
        max_allowed[i] = torch.nextafter(
            max_allowed[i + 1],
            torch.tensor(float('-inf'), dtype=torch.float16)
        )

    # STEP 3: Repair left→right ensuring strictly increasing AND within cap
    for i in range(len(positive)):
        if i == 0:
            # First value: just ensure it's positive and within cap
            if positive[i] <= 0:
                positive[i] = torch.tensor(1e-4, dtype=torch.float16)
            if positive[i] > max_allowed[i]:
                positive[i] = max_allowed[i]
        else:
            # Must be > previous AND <= max_allowed
            if positive[i] <= positive[i - 1]:
                positive[i] = torch.nextafter(
                    positive[i - 1],
                    torch.tensor(float('inf'), dtype=torch.float16)
                )
            if positive[i] > max_allowed[i]:
                positive[i] = max_allowed[i]

    # STEP 4: Mirror to negative side (in increasing order)
    negative = -positive.flip(0)

    # Reconstruct full LUT
    return torch.cat([negative, positive]).to(torch.float32)


def compute_nearest_indices(q: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    """Find nearest LUT index for each Q value. CPU-only, chunked."""
    q_flat = q.flatten()
    indices = torch.zeros(q_flat.shape[0], dtype=torch.int16)

    chunk_size = 100000
    for start in range(0, len(q_flat), chunk_size):
        end = min(start + chunk_size, len(q_flat))
        chunk = q_flat[start:end]
        distances = (chunk.unsqueeze(1) - lut.unsqueeze(0)).abs()
        indices[start:end] = distances.argmin(dim=1).to(torch.int16)

    return indices.view(q.shape)


def verify_lut_fp16(lut: torch.Tensor) -> bool:
    """Verify LUT is valid after FP16 snap."""
    lut_fp16 = lut.to(torch.float16)
    # Check no zeros
    if (lut_fp16 == 0).any():
        return False
    # Check strictly increasing
    if not (lut_fp16[1:] > lut_fp16[:-1]).all():
        return False
    # Check unique
    if len(torch.unique(lut_fp16)) != len(lut_fp16):
        return False
    return True


# =============================================================================
# STE-FP16 UTILITIES
# =============================================================================

# FP16 max value - values beyond this become inf when cast to FP16
FP16_MAX = 65504.0

# Global flag for debugging STE-FP16 saturation events
_STE_FP16_DEBUG = False


def set_ste_fp16_debug(enabled: bool):
    """Enable/disable debug logging for STE-FP16 saturation events."""
    global _STE_FP16_DEBUG
    _STE_FP16_DEBUG = enabled


def ste_fp16(x: torch.Tensor, saturate: bool = True) -> torch.Tensor:
    """Straight-through estimator for FP16 rounding.

    Forward: Rounds x to FP16 precision (matches ANE behavior).
    Backward: Identity gradient (passes through as if no rounding).

    This allows training with FP32 master weights while the forward pass
    matches what ANE will actually execute in FP16.

    Args:
        x: Input tensor (any dtype)
        saturate: If True, clamp values to FP16 range before casting to prevent
                  inf values. This is critical for TPU/XLA where FP32 values
                  can exceed FP16 max (~65504). Default True.

    Returns:
        Tensor with FP16-rounded values but same dtype as input.
        Gradients flow through unchanged.
    """
    if x.dtype == torch.float16:
        return x  # Already FP16, no rounding needed

    if saturate:
        # Clamp to FP16 range to prevent inf on overflow
        # This is a valid emulation: ANE would also saturate/clip on overflow
        if _STE_FP16_DEBUG:
            overflow_mask = x.abs() > FP16_MAX
            if overflow_mask.any():
                overflow_count = overflow_mask.sum().item()
                max_val = x.abs().max().item()
                print(f"[STE_FP16 SATURATE] {overflow_count} values clamped, max|x|={max_val:.1f}")
        x_safe = x.clamp(-FP16_MAX, FP16_MAX)
        x16 = x_safe.to(torch.float16)
    else:
        x16 = x.to(torch.float16)

    # STE: forward uses x16 values, backward pretends this is identity
    return x + (x16.to(x.dtype) - x).detach()


# =============================================================================
# ANEMLL QAT LINEAR V2 - RANK-BY-RANK
# =============================================================================

class AnemllQATLinearV2(nn.Module):
    """
    Linear layer with Anemll-style LUT quantization - V2 (ANE-friendly).

    Key differences from V1:
    - rank_magnitude: [rank] - the ONLY magnitude (not multiplied by norms)
    - scale_A: [out, rank] - unit-norm columns (A_dir)
    - scale_B: [rank, in] - unit-norm rows (B_dir), NO padding
    - _Q: [out, in] - frozen LUT values buffer
    - _indices: [out, in] - frozen quantization indices buffer

    Forward pass: y = Σₖ gₖ · (aₖ ⊙ (Q (bₖ ⊙ x)))
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        config: Optional[AnemllQuantConfigV2] = None,
        # LoRA params (kept for compatibility)
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        # Internal: skip SVD init (used by from_linear with skip_init=True)
        _skip_scale_init: bool = False,
        # Custom LUT: override default uniform LUT with custom values
        custom_lut: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self._skip_scale_init = _skip_scale_init
        self.in_features = in_features
        self.out_features = out_features
        self.config = config or AnemllQuantConfigV2()

        # Scale rank
        self.scale_rank = self.config.scale_rank
        if self.scale_rank <= 0:
            raise ValueError("V2 requires scale_rank > 0 for rank-by-rank forward")

        # Base weights
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None

        # Scale parameters - unit-norm with separate magnitude
        # A_dir: [out, rank] - unit-norm columns
        # B_dir: [rank, in] - unit-norm rows (NO padding!)
        # rank_magnitude: [rank] - the ONLY magnitude
        self.scale_A = nn.Parameter(torch.empty(out_features, self.scale_rank))
        self.scale_B = nn.Parameter(torch.empty(self.scale_rank, in_features))
        self.rank_magnitude = nn.Parameter(torch.ones(self.scale_rank))

        # LUT - use custom_lut if provided, otherwise create default uniform LUT
        if custom_lut is not None:
            # Use custom LUT (e.g., power2, power3, inv_mu50 from per-tensor search)
            lut = custom_lut.clone().to(torch.float32)
            if lut.size(0) != self.config.lut_size:
                raise ValueError(
                    f"custom_lut size {lut.size(0)} doesn't match config.lut_size {self.config.lut_size}"
                )
        else:
            # Default: uniform LUT via make_lut
            lut = make_lut(
                self.config.lut_size,
                device=torch.device("cpu"),
                dtype=torch.float32,
                include_zero=self.config.lut_include_zero,
            )
        if self.config.learnable_lut:
            self.lut = nn.Parameter(lut)
        else:
            self.register_buffer("lut", lut)

        # Frozen Q buffer (computed once by freeze_Q())
        self.register_buffer("_Q", None)
        # _indices buffer for LUT training (must be persistent for checkpoint save/load)
        self.register_buffer("_indices", None, persistent=True)

        # Cached full weights for inference
        self.register_buffer("_cached_weight_q", None)

        # Enable/disable fake quantization (for backward compat)
        self.enable_fake_quant = True

        # Store lut_bits for conversion pipeline
        self.lut_bits = self.config.lut_bits

        # === TRAINABLE LUT: Per-layer trainable LUT for LUT training ===
        # _lut_raw_deltas: [8] - learnable logits for constrained simplex (LUT16 only)
        # _lut_max_abs: Maximum absolute LUT value
        # _lut_min_delta: Minimum normalized delta (post-softmax)
        # _lut_trainable: Whether LUT training is enabled for this layer
        # _use_indices: Whether to use _indices for forward (vs _Q)
        self._lut_raw_deltas: Optional[nn.Parameter] = None
        self._lut_max_abs: float = 1.0
        self._lut_min_delta: float = 0.001
        self._lut_trainable: bool = False
        self._use_indices: bool = False

        # === QRANKLUT: Per-layer LUTs for A_dir/B_dir quantization ===
        # A_lut: [rank_lut_size] - LUT for A_dir quantization
        # B_lut: [rank_lut_size] - LUT for B_dir quantization
        # _A_idx: [out, rank] - quantization indices for A_dir
        # _B_idx: [rank, in] - quantization indices for B_dir
        self._rank_lut_enabled = False  # Enabled after init_rank_lut()
        rank_lut_size = self.config.rank_lut_size

        if rank_lut_size > 0:
            # Initialize with uniform LUT (will be replaced by k-means)
            init_lut = torch.linspace(-1.0, 1.0, rank_lut_size)
            if self.config.rank_lut_learnable:
                self.A_lut = nn.Parameter(init_lut.clone())
                self.B_lut = nn.Parameter(init_lut.clone())
            else:
                self.register_buffer("A_lut", init_lut.clone())
                self.register_buffer("B_lut", init_lut.clone())
            # Index buffers (computed by init_rank_lut)
            self.register_buffer("_A_idx", None)
            self.register_buffer("_B_idx", None)
        else:
            self.A_lut = None
            self.B_lut = None
            self._A_idx = None
            self._B_idx = None

        # LoRA (kept for compatibility)
        self.lora_r = int(lora_r)
        self.lora_alpha = float(lora_alpha)
        self.lora_dropout = float(lora_dropout)

        if self.lora_r > 0:
            self.lora_A = nn.Parameter(torch.zeros(self.lora_r, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, self.lora_r))
            self.scaling = self.lora_alpha / self.lora_r
            self.lora_drop = nn.Dropout(p=self.lora_dropout)
        else:
            self.lora_A = None
            self.lora_B = None
            self.scaling = 0.0
            self.lora_drop = None

        # Use batched forward (for CoreML) or loop (for PyTorch training)
        # Auto-enable batched for high rank to reduce STE calls (4*rank -> 4)
        self.use_batched_forward = self.scale_rank >= 16

        # V2 defaults to factored inference (rank-by-rank) for ANE compatibility
        # Set to False for faster PyTorch inference (single matmul)
        self.use_factored_inference = True

        # Flag: scales are already baked (magnitude in A, B is unit, g=1)
        # When True, _get_normalized_scales() returns raw params without re-normalizing
        # Registered as buffer so it's saved/loaded with the model
        self.register_buffer("_scales_baked_flag", torch.tensor(0, dtype=torch.int8))
        # Python cache to avoid .item() calls on XLA (causes graph breaks!)
        # Initialize to False for fresh models (scales are never baked at init)
        self._scales_baked_python: bool = False

        # Register hook to sync Python cache from tensor buffer after load_state_dict
        self.register_load_state_dict_post_hook(self._sync_scales_baked_from_buffer)

        self.reset_parameters()

    @property
    def _scales_baked(self) -> bool:
        """Check if scales are already baked (magnitude in A, g=1).

        Uses Python cache to avoid .item() calls which cause XLA graph breaks.
        The cache is:
        - Initialized to False in __init__ for fresh models
        - Synced from tensor buffer by _sync_scales_baked_from_buffer hook after load_state_dict
        - Updated by setter whenever the flag changes
        """
        return self._scales_baked_python

    @_scales_baked.setter
    def _scales_baked(self, value: bool):
        """Set the scales_baked flag."""
        # Update Python cache (fast path for XLA)
        self._scales_baked_python = value
        # Update tensor buffer (for save/load persistence)
        if self._scales_baked_flag is not None:
            self._scales_baked_flag.fill_(1 if value else 0)

    @staticmethod
    def _sync_scales_baked_from_buffer(module, incompatible_keys):
        """Hook called after load_state_dict to sync Python cache from tensor buffer.

        This ensures the Python cache is correct after loading a checkpoint,
        avoiding .item() calls in the forward path on XLA.
        """
        if hasattr(module, '_scales_baked_flag') and module._scales_baked_flag is not None:
            # Read tensor value ONCE here (during load, not in forward)
            module._scales_baked_python = bool(module._scales_baked_flag.item())

    def mark_scales_baked(self, force: bool = False) -> bool:
        """Detect and mark scales as baked (for old checkpoints).

        Call this ONCE after loading a snapped checkpoint that doesn't have
        _scales_baked_flag set. This avoids torch.allclose() in the hot path.

        Args:
            force: If True, mark as baked without checking. Otherwise auto-detect.

        Returns:
            True if scales were marked as baked.
        """
        if force:
            self._scales_baked = True
            return True
        # Auto-detect: if rank_magnitude is all ones, scales are baked
        if self.rank_magnitude is not None:
            g = self.rank_magnitude
            if g.numel() > 0 and torch.allclose(g, torch.ones_like(g), atol=1e-5):
                self._scales_baked = True
                return True
        return False

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

        # Initialize scales from weight statistics
        # Note: skip if called from from_linear() which handles this separately
        if not getattr(self, '_skip_scale_init', False):
            self._init_scales_from_weight()

        if self.lora_r > 0:
            nn.init.normal_(self.lora_A, std=0.02)
            nn.init.zeros_(self.lora_B)

    @torch.no_grad()
    def _init_scales_from_weight(
        self,
        importance: Optional[torch.Tensor] = None,
        awq_alpha: float = 0.0,
    ):
        """Initialize scale parameters with unit-norm + magnitude decomposition.

        Uses V1's group-based initialization:
        1. Compute per-group max-abs scales
        2. Expand to per-weight
        3. SVD to get unit-norm A, B and magnitudes G

        Result:
        - scale_A = u[:, :r] (unit-norm columns)
        - scale_B = vh[:r, :] (unit-norm rows)
        - rank_magnitude = s[:r] (singular values = magnitudes G)

        Args:
            importance: NOT USED (AWQ not implemented). Kept for future compatibility.
            awq_alpha: NOT USED (AWQ not implemented). Kept for future compatibility.

        NOTE: AWQ-style weighting is NOT IMPLEMENTED. AWQ requires:
        1. Scale weights by importance factor
        2. Apply INVERSE factor to inputs via RMSNorm.weight modification
        Without step 2, AWQ breaks the model. See TODO in code for details.
        """
        w = self.weight.float()
        group_size = self.config.group_size

        # Pad input dimension to multiple of group_size
        pad = (group_size - self.in_features % group_size) % group_size
        if pad > 0:
            w = F.pad(w, (0, pad))

        padded_in = w.size(1)
        num_groups = padded_in // group_size

        # Step 1: Compute per-group max-abs scales [out, num_groups]
        grouped = w.view(self.out_features, num_groups, group_size)
        scales_per_group = grouped.abs().amax(dim=2).clamp(min=1e-8)

        # Step 2: Expand to per-weight [out, padded_in]
        scales_per_weight = scales_per_group.repeat_interleave(group_size, dim=1)

        # Trim to actual input size (V2 doesn't use padding in forward)
        scales_per_weight = scales_per_weight[:, :self.in_features]

        # Step 2b: AWQ-style importance weighting (NOT IMPLEMENTED)
        # WARNING: AWQ requires BOTH weight scaling AND inverse scaling in RMSNorm.
        # Without the RMSNorm compensation, this BREAKS the model output.
        # The code below is DISABLED - keeping for future implementation reference.
        if importance is not None and awq_alpha > 0:
            # TODO: Implement proper AWQ with RMSNorm.weight modification
            # For now, just skip and warn (don't break the model)
            import warnings
            warnings.warn(
                f"AWQ weighting (alpha={awq_alpha}) is NOT IMPLEMENTED correctly. "
                f"AWQ requires inverse scaling in RMSNorm, which is not done. "
                f"Ignoring AWQ parameters to avoid breaking the model.",
                UserWarning
            )
            # DO NOT apply AWQ weighting - it breaks inference without RMSNorm compensation
            # The code below is what WOULD be done if RMSNorm was also modified:
            # imp = importance.to(w.device).float()
            # imp = imp / (imp.mean() + 1e-8)
            # imp = imp.clamp(0.1, 10.0)
            # awq_factor = imp ** awq_alpha
            # awq_factor = awq_factor / (awq_factor.mean() + 1e-8)
            # scales_per_weight = scales_per_weight * awq_factor.view(1, -1)

        # Step 3: SVD: scales ≈ u @ diag(s) @ vh
        u, s, vh = torch.linalg.svd(scales_per_weight, full_matrices=False)
        r = self.scale_rank

        # u columns and vh rows are already unit-norm from SVD!
        self.scale_A.data = u[:, :r].to(self.weight.dtype)
        self.scale_B.data = vh[:r, :].to(self.weight.dtype)

        # Floor small singular values to prevent dead ranks
        # Minimum is 1% of largest singular value (or 0.01 absolute minimum)
        s_r = s[:r]
        min_magnitude = max(0.01 * s_r[0].item(), 0.01)
        s_floored = s_r.clamp(min=min_magnitude)
        self.rank_magnitude.data = s_floored.to(self.weight.dtype)

    def _get_normalized_scales(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return normalized scale directions + per-rank magnitude.

        This is where we encode the effect of the old:
            (A @ B).clamp(min=eps)
        without ever materializing A@B in the forward pass.

        If `config.force_positive_scales` is enabled (recommended when Q is frozen):
        - force the *factors* to be nonnegative (abs/softplus) so S = (A*g)@B stays >= 0
        - force g >= 0 (softplus/abs) so rank magnitudes can't flip sign

        Returns:
            A_dir: [out, rank] unit-norm columns (or baked A if _scales_baked)
            B_dir: [rank, in] unit-norm rows
            g: [rank] nonnegative magnitudes (if configured)
        """
        # Fast path: scales already baked (after snap_for_ane/snap_for_export)
        # Return raw params without re-normalizing
        # Note: _scales_baked uses Python cache to avoid XLA graph breaks
        if self._scales_baked:
            return self.scale_A, self.scale_B, self.rank_magnitude

        # NOTE: Auto-detection of baked scales removed to avoid XLA graph breaks.
        # torch.allclose() forces tensor sync which breaks XLA compilation.
        # If loading old snapped checkpoints, call mark_scales_baked() explicitly.

        # Use config-driven eps (1e-6 is FP16-safe, 1e-8 underflows in FP16)
        eps = getattr(self.config, 'norm_eps', 1e-6)

        # Start from raw params
        A = self.scale_A
        B = self.scale_B

        # Optional: force nonnegative factors so the implied scale matrix stays >= 0.
        if getattr(self.config, "force_positive_scales", False):
            method = getattr(self.config, "positive_scale_method", "abs")
            if method == "softplus":
                A = F.softplus(A)
                B = F.softplus(B)
            elif method == "abs":
                A = A.abs()
                B = B.abs()
            else:
                raise ValueError(f"Unknown positive_scale_method: {method}")

        # Normalize columns of A (directions)
        A_norms = A.norm(dim=0, keepdim=True).clamp(min=eps)
        A_dir = A / A_norms

        # Normalize rows of B (directions)
        B_norms = B.norm(dim=1, keepdim=True).clamp(min=eps)
        B_dir = B / B_norms

        # === QRANKLUT: Apply LUT quantization to A_dir/B_dir ===
        if getattr(self, '_rank_lut_enabled', False) and self.A_lut is not None:
            if self._A_idx is not None and self._B_idx is not None:
                # Frozen mode: use cached indices (fast lookup)
                A_dir = self.A_lut[self._A_idx]
                B_dir = self.B_lut[self._B_idx]
            else:
                # STE mode: compute indices on-the-fly with gradient passthrough
                frozen = getattr(self.config, 'rank_lut_frozen_idx', True)
                A_dir = qranklut_fake_quant(A_dir, self.A_lut, frozen_idx=frozen)
                B_dir = qranklut_fake_quant(B_dir, self.B_lut, frozen_idx=frozen)

        # Per-rank magnitude (optionally forced nonnegative)
        g = self.rank_magnitude
        act = getattr(self.config, "magnitude_activation", "identity")
        g_eps = float(getattr(self.config, "magnitude_eps", 0.0))
        if act == "softplus":
            g = F.softplus(g) + g_eps
        elif act == "abs":
            g = g.abs() + g_eps
        elif act == "identity":
            if g_eps:
                g = g + g_eps
        else:
            raise ValueError(f"Unknown magnitude_activation: {act}")

        # STE-FP16: Round rank_magnitude to FP16-representable values
        # This ensures rank_magnitude stays FP16-friendly during training,
        # avoiding large rounding errors when snapping to FP16 for ANE.
        if getattr(self.config, 'use_ste_fp16', False):
            g = ste_fp16(g)

        return A_dir, B_dir, g

    def _compute_full_scales(self) -> torch.Tensor:
        """Reconstruct full scales matrix [out, in].

        Only used at init/snap time, NOT in forward pass.
        """
        A_dir, B_dir, g = self._get_normalized_scales()
        # S = A_dir @ diag(g) @ B_dir = (A_dir * g) @ B_dir
        A_scaled = A_dir * g  # [out, rank]
        S = (A_scaled @ B_dir)  # [out, in]

        # When force_positive_scales=True, A, B, g are all >= 0 by construction
        # so S is guaranteed >= 0.
        # When force_positive_scales=False (V1 compatibility), scales can be negative
        # and we should NOT clamp them.
        return S

    @torch.no_grad()
    def freeze_Q(self):
        """Compute Q = lut[indices] once. Call before training scales.

        This freezes the quantization indices and Q values.
        After calling this, only scale_A, scale_B, rank_magnitude are trained.
        """
        device = self.weight.device

        # Full scales needed ONLY here, at init/snap time
        scales = self._compute_full_scales()  # [out, in]
        normalized = self.weight / scales

        # Quantize to LUT indices (on CPU for compatibility, then move)
        # Pass actual LUT values for non-uniform LUTs (power2, power3, inv_mu50, etc.)
        indices_cpu = quantize_to_lut_indices(
            normalized.cpu(),
            lut_size=self.lut.size(0),
            include_zero=self.config.lut_include_zero,
            lut=self.lut.cpu(),  # Use actual LUT values for nearest-neighbor search
        )

        # Store indices on the same device as weights
        self._indices = indices_cpu.to(device)

        # Store Q = lut[indices] in [-1, 1]
        # Index on CPU then move (more compatible with MPS)
        self._Q = self.lut.cpu()[indices_cpu].to(device)

        # Freeze weight (we're training scales only)
        self.weight.requires_grad = False

    # =========================================================================
    # TRAINABLE LUT METHODS
    # =========================================================================

    def convert_q_to_indices_cpu(
        self,
        max_error_threshold: float = 0.01,
        allow_bad_qc: bool = False,
    ) -> bool:
        """Convert _Q (values) to _indices (int16 LUT indices). CPU-only with QC.

        FAIL-CLOSED by default: returns False if max_error > threshold unless
        allow_bad_qc=True is explicitly passed.

        Args:
            max_error_threshold: Maximum allowed reconstruction error
            allow_bad_qc: If True, proceed even with high error (dangerous)

        Returns:
            True if conversion succeeded, False if aborted due to QC failure
        """
        if self._indices is not None:
            self._use_indices = True
            return True  # Already have indices

        if self._Q is None:
            return True  # Nothing to convert

        # Ensure we're on CPU
        q_cpu = self._Q.detach().cpu()
        lut_cpu = self.lut.detach().cpu()

        # Find nearest indices
        indices = compute_nearest_indices(q_cpu, lut_cpu)

        # Validate range
        assert (indices >= 0).all() and (indices <= 15).all(), \
            f"Invalid LUT indices: min={indices.min()}, max={indices.max()}"

        # QC: compute reconstruction error
        q_reconstructed = lut_cpu[indices.long()]
        max_error = (q_cpu - q_reconstructed).abs().max().item()
        mean_error = (q_cpu - q_reconstructed).abs().mean().item()

        print(f"    _Q→_indices QC: max_error={max_error:.6f}, mean_error={mean_error:.6f}")

        if max_error > max_error_threshold:
            print(f"    ERROR: max error {max_error:.6f} > threshold {max_error_threshold}")
            print(f"    _Q values are NOT on LUT grid. Model weights will change!")
            if not allow_bad_qc:
                print(f"    ABORTING. Use --allow-bad-q2idx to force conversion.")
                return False  # FAIL-CLOSED
            else:
                print(f"    WARNING: Proceeding anyway (--allow-bad-q2idx). Expect loss spike!")

        # Store indices (keep on CPU, will be moved with model.to(device))
        self._indices = indices.to(self._Q.device)
        self._use_indices = True
        return True

    def enable_lut_training(
        self,
        max_abs: float = 1.0,
        allow_bad_qc: bool = False,
    ) -> bool:
        """Enable trainable LUT for this layer. Must be called on CPU.

        Args:
            max_abs: Maximum absolute LUT value (e.g., 1.0, 2.0)
            allow_bad_qc: Allow _Q→_indices conversion even with high error

        Returns:
            True if LUT training enabled, False if failed (QC or unsupported)
        """
        if self._lut_trainable:
            return True  # Already enabled

        lut = self.lut
        if lut is None:
            return False

        # HARD CHECK: Must be on CPU (index conversion is CPU-only)
        if self._Q is not None and self._Q.device.type != 'cpu':
            raise RuntimeError(
                f"enable_lut_training() must be called BEFORE model.to(device). "
                f"_Q is on {self._Q.device}, expected CPU. "
                f"Move enable_lut_training_all() call before model.to(device)."
            )

        lut_size = lut.shape[0]
        if lut_size != 16:
            print(f"  Skipping LUT training for lut_size={lut_size} (only 16 supported)")
            return False

        # Convert _Q to _indices with QC check (CPU-only, FAIL-CLOSED)
        if not self.convert_q_to_indices_cpu(allow_bad_qc=allow_bad_qc):
            return False

        self._lut_max_abs = max_abs
        self._lut_min_delta = compute_min_delta(max_abs)

        # Initialize from current positive LUT values
        # Current LUT is globally increasing: [-max, ..., -ε, +ε, ..., +max]
        # Positive half is indices 8-15 (8 values from +ε to +max)
        with torch.no_grad():
            lut_cpu = lut.detach().cpu()
            half_size = lut_size // 2  # 8

            # Get positive values
            positive_vals = lut_cpu[half_size:]  # [+ε, ..., +max]

            # Compute 8 deltas: first value, then differences
            deltas = torch.zeros(half_size)
            deltas[0] = positive_vals[0].abs()  # First positive value (ε)
            deltas[1:] = positive_vals[1:] - positive_vals[:-1]

            # Clamp to min_delta floor
            deltas = deltas.clamp(min=self._lut_min_delta)

            # Compute raw logits that would produce these deltas via constrained simplex
            # delta_i = min_delta + remaining * softmax(raw)[i]
            # So: softmax(raw)[i] = (delta_i - min_delta) / remaining
            remaining = max_abs - half_size * self._lut_min_delta
            if remaining <= 0:
                # Edge case: min_delta too large, use uniform
                raw = torch.zeros(half_size)
            else:
                weights = (deltas - self._lut_min_delta) / remaining
                weights = weights.clamp(min=1e-6)  # Avoid log(0)
                # Invert softmax: raw = log(weights) + const (const cancels in softmax)
                raw = torch.log(weights)

            self._lut_raw_deltas = nn.Parameter(raw.to(lut.device))

        self._lut_trainable = True
        return True

    def get_lut(self) -> torch.Tensor:
        """Get current LUT (trainable or frozen). Device-consistent.

        If LUT training is enabled, builds LUT from trainable parameters.
        Otherwise returns the frozen LUT buffer.
        """
        if self._lut_trainable and self._lut_raw_deltas is not None:
            return build_symmetric_lut16(
                self._lut_raw_deltas,
                self._lut_max_abs,
                self._lut_min_delta,
            )
        else:
            return self.lut

    @torch.no_grad()
    def init_rank_lut(self, verbose: bool = False):
        """Initialize qranklut with k-means clustering from current A_dir/B_dir.

        This method:
        1. Computes normalized A_dir, B_dir from current scale_A, scale_B
        2. Runs k-means to find optimal LUT values for each
        3. Assigns indices to each element
        4. Enables qranklut mode (_rank_lut_enabled = True)

        Call this after loading a trained checkpoint to add qranklut compression.
        """
        if self.A_lut is None or self.config.rank_lut_size <= 0:
            if verbose:
                print(f"  [skip] rank_lut_size = {self.config.rank_lut_size}")
            return

        device = self.weight.device
        k = self.config.rank_lut_size

        # Get normalized directions (without qranklut applied)
        # Temporarily disable qranklut to get raw normalized values
        was_enabled = self._rank_lut_enabled
        self._rank_lut_enabled = False

        eps = getattr(self.config, 'norm_eps', 1e-6)
        A = self.scale_A
        B = self.scale_B

        # Apply positivity if configured
        if getattr(self.config, "force_positive_scales", False):
            method = getattr(self.config, "positive_scale_method", "abs")
            if method == "softplus":
                A = F.softplus(A)
                B = F.softplus(B)
            elif method == "abs":
                A = A.abs()
                B = B.abs()

        # Normalize
        A_norms = A.norm(dim=0, keepdim=True).clamp(min=eps)
        A_dir = (A / A_norms).float()

        B_norms = B.norm(dim=1, keepdim=True).clamp(min=eps)
        B_dir = (B / B_norms).float()

        # Run k-means on A_dir
        A_lut = kmeans_1d(A_dir.flatten(), k)
        A_idx = assign_to_lut(A_dir, A_lut)

        # Run k-means on B_dir
        B_lut = kmeans_1d(B_dir.flatten(), k)
        B_idx = assign_to_lut(B_dir, B_lut)

        # Store LUT values
        if isinstance(self.A_lut, nn.Parameter):
            self.A_lut.data = A_lut.to(device)
            self.B_lut.data = B_lut.to(device)
        else:
            self.A_lut = A_lut.to(device)
            self.B_lut = B_lut.to(device)

        # Store indices
        self._A_idx = A_idx.to(device)
        self._B_idx = B_idx.to(device)

        # Enable qranklut
        self._rank_lut_enabled = True

        if verbose:
            # Compute reconstruction error
            A_recon = A_lut[A_idx]
            B_recon = B_lut[B_idx]
            A_err = (A_dir - A_recon).abs().mean().item()
            B_err = (B_dir - B_recon).abs().mean().item()
            print(f"  A_lut range: [{A_lut.min():.4f}, {A_lut.max():.4f}]")
            print(f"  B_lut range: [{B_lut.min():.4f}, {B_lut.max():.4f}]")
            print(f"  Reconstruction MAE: A={A_err:.6f}, B={B_err:.6f}")

    @torch.no_grad()
    def freeze_rank_lut(self):
        """Freeze rank LUT indices (keep LUT values trainable).

        After calling this:
        - _A_idx, _B_idx are frozen (no gradient)
        - A_lut, B_lut can still be trained (if rank_lut_learnable=True)
        """
        if self._A_idx is not None:
            self._A_idx = self._A_idx.detach()
        if self._B_idx is not None:
            self._B_idx = self._B_idx.detach()

    def enable_recovery_lora(self, r: int, alpha: float = None, dropout: float = 0.0):
        """Enable recovery LoRA adapters for post-QAT knowledge recovery.

        This method initializes LoRA parameters to recover accuracy lost during
        quantization. Call this AFTER loading a QAT checkpoint and BEFORE
        creating the optimizer (see critical ordering below).

        CRITICAL ORDER for DDP/optimizer compatibility:
            1. model = load_model(...)
            2. model.load_state_dict(checkpoint)
            3. enable_recovery_lora_all(model, r=8)  # <-- enables LoRA
            4. freeze_for_recovery_training(model)
            5. optimizer = AdamW([p for p in model.parameters() if p.requires_grad])
            6. model = DDP(model) / FSDP(model) if distributed

        Args:
            r: LoRA rank (recommended: start with 8, increase to 16/32 if needed)
            alpha: LoRA alpha (default: r). Controls scaling = alpha / r
            dropout: LoRA dropout (default: 0.0)

        Note:
            - LoRA params are created in FP32 for optimizer stability
            - Forward pass casts to input dtype automatically
            - Initialize: A with small random (std=0.02), B with zeros
        """
        if self.lora_r > 0:
            return  # Already enabled

        alpha = alpha if alpha is not None else float(r)
        self.lora_r = int(r)
        self.lora_alpha = float(alpha)
        self.scaling = alpha / r
        self.lora_dropout = float(dropout)

        # IMPORTANT: Keep LoRA in FP32 even if base is FP16 (optimizer stability)
        device = self.weight.device
        self.lora_A = nn.Parameter(
            torch.zeros(self.lora_r, self.in_features, device=device, dtype=torch.float32)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(self.out_features, self.lora_r, device=device, dtype=torch.float32)
        )
        self.lora_drop = nn.Dropout(p=dropout) if dropout > 0 else None

        # Initialize: A with small random, B with zeros (standard LoRA init)
        nn.init.normal_(self.lora_A, std=0.02)
        nn.init.zeros_(self.lora_B)

    def disable_recovery_lora(self):
        """Disable recovery LoRA adapters (for inference without LoRA)."""
        self.lora_r = 0
        self.lora_A = None
        self.lora_B = None
        self.lora_drop = None
        self.scaling = 0.0

    @torch.no_grad()
    def unfreeze_rank_lut_for_training(self):
        """Clear cached indices to enable STE-based training.

        After calling this:
        - Indices are recomputed on-the-fly in forward pass
        - Gradients flow through STE to A_lut, B_lut, and potentially scale_A/scale_B
        """
        self._A_idx = None
        self._B_idx = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Rank-by-rank forward: y = Σₖ gₖ · (aₖ ⊙ (Q (bₖ ⊙ x)))

        No A @ B materialization in this path.
        """
        # Fast path for cached full weights (single matmul inference)
        # Skip if use_factored_inference=True (for ANE export testing)
        # NOTE: This path still needs to add LoRA if enabled
        if self._cached_weight_q is not None and not self.use_factored_inference:
            w_q = self._cached_weight_q.to(x.dtype)
            bias = self.bias.to(x.dtype) if self.bias is not None else None
            y = F.linear(x, w_q, bias)

            # Add LoRA if enabled (don't skip!)
            if self.lora_r > 0 and self.lora_A is not None:
                x_d = self.lora_drop(x) if self.lora_drop is not None else x
                lora_A = self.lora_A.to(x.dtype)
                lora_B = self.lora_B.to(x.dtype)
                hidden = x_d @ lora_A.t()
                y = y + (hidden @ lora_B.t()) * self.scaling

            return y

        # Get normalized scales
        A_dir, B_dir, g = self._get_normalized_scales()

        # Get Q: either via trainable LUT lookup, frozen buffer, or on-the-fly
        if self._lut_trainable and self._use_indices and self._indices is not None:
            # Trainable LUT mode: use get_lut() with indices lookup
            # This allows gradients to flow through the LUT
            lut = self.get_lut()
            Q = lut[self._indices.long()]
        elif self._use_indices and self._indices is not None:
            # Frozen indices with frozen LUT
            Q = self.lut[self._indices.long()]
        elif self._Q is not None:
            # Frozen Q buffer (pre-computed lut[indices])
            Q = self._Q
        else:
            # Fallback: compute Q on the fly (for backward compat / testing)
            scales = self._compute_full_scales()
            normalized = self.weight / scales
            indices = quantize_to_lut_indices(
                normalized, self.lut.size(0), self.config.lut_include_zero,
                lut=self.lut,  # Use actual LUT values
            )
            Q = self.lut[indices]

        Q = Q.to(x.dtype)

        # STE-FP16: Apply to Q buffer (frozen LUT values snapped to FP16)
        use_ste = getattr(self.config, 'use_ste_fp16', False)
        if use_ste:
            Q = ste_fp16(Q)

        # For LoRA training: V2 base doesn't need gradients, only LoRA does
        # This saves massive memory by not building computation graph for V2
        # Also use loop-based forward to avoid materializing [batch,seq,rank,out] tensors
        use_no_grad_base = self.lora_r > 0 and self.training

        if use_no_grad_base:
            with torch.no_grad():
                # ALWAYS use loop-based forward for LoRA training - much more memory efficient
                # _forward_batched creates [batch,seq,rank,in] and [batch,seq,rank,out] tensors
                # _forward_loop only keeps [batch,seq,out] accumulator
                y = self._forward_loop(x, A_dir, B_dir, g, Q)
                if self.bias is not None:
                    bias = self.bias.to(x.dtype)
                    if use_ste:
                        bias = ste_fp16(bias)
                    y = y + bias
            # Detach to ensure no gradients flow back through V2
            y = y.detach()
        else:
            # Choose forward implementation
            if self.use_batched_forward:
                y = self._forward_batched(x, A_dir, B_dir, g, Q)
            else:
                y = self._forward_loop(x, A_dir, B_dir, g, Q)
            # Add bias
            if self.bias is not None:
                bias = self.bias.to(x.dtype)
                if use_ste:
                    bias = ste_fp16(bias)
                y = y + bias

        # Add LoRA if enabled (in-place for memory efficiency)
        if self.lora_r > 0:
            x_d = self.lora_drop(x) if self.lora_drop is not None else x
            lora_A = self.lora_A.to(x.dtype)
            lora_B = self.lora_B.to(x.dtype)

            # STE-FP16: Apply to LoRA weights (snapped to FP16)
            if use_ste:
                lora_A = ste_fp16(lora_A)
                lora_B = ste_fp16(lora_B)

            # In-place addition: y += ... is more memory efficient
            hidden = x_d @ lora_A.t()  # [*, lora_r] - small intermediate
            if use_ste:
                hidden = ste_fp16(hidden)
            lora_out = (hidden @ lora_B.t()) * self.scaling
            if use_ste:
                lora_out = ste_fp16(lora_out)
            y += lora_out

        return y

    def _forward_loop(
        self,
        x: torch.Tensor,
        A_dir: torch.Tensor,
        B_dir: torch.Tensor,
        g: torch.Tensor,
        Q: torch.Tensor,
    ) -> torch.Tensor:
        """Loop-based rank-by-rank forward (better for training).

        When use_ste_fp16=True, applies FP16 rounding at each step with
        straight-through gradients. This makes the forward pass match ANE's
        FP16 behavior while keeping FP32 gradients for stable training.
        """
        use_ste = getattr(self.config, 'use_ste_fp16', False)
        y = torch.zeros(*x.shape[:-1], self.out_features, device=x.device, dtype=x.dtype)

        for k in range(self.scale_rank):
            b_k = B_dir[k, :].to(x.dtype)  # [in]
            a_k = (g[k] * A_dir[:, k]).to(x.dtype)  # [out]

            # Apply STE-FP16 to scale vectors
            if use_ste:
                b_k = ste_fp16(b_k)
                a_k = ste_fp16(a_k)

            # Q @ (b_k * x)
            scaled_x = x * b_k  # [..., in]
            if use_ste:
                scaled_x = ste_fp16(scaled_x)

            Qx = F.linear(scaled_x, Q)  # [..., out]
            if use_ste:
                Qx = ste_fp16(Qx)

            # Accumulate with a_k weighting
            y = y + a_k * Qx  # [..., out]
            if use_ste:
                y = ste_fp16(y)

        return y

    def _forward_batched(
        self,
        x: torch.Tensor,
        A_dir: torch.Tensor,
        B_dir: torch.Tensor,
        g: torch.Tensor,
        Q: torch.Tensor,
    ) -> torch.Tensor:
        """Batched rank-by-rank forward (fewer ops for CoreML).

        When use_ste_fp16=True, applies FP16 rounding at key steps with
        straight-through gradients.
        """
        use_ste = getattr(self.config, 'use_ste_fp16', False)

        # Stack all b_k: [rank, in]
        B = B_dir.to(x.dtype)

        # Stack all a_k with magnitudes: [rank, out]
        A = (A_dir * g).T.to(x.dtype)  # [rank, out]

        # Apply STE-FP16 to scale matrices
        if use_ste:
            B = ste_fp16(B)
            A = ste_fp16(A)

        # Compute all scaled inputs at once
        # x: [..., in], B: [rank, in] -> X_rank: [..., rank, in]
        X_rank = x.unsqueeze(-2) * B  # [..., rank, in]
        if use_ste:
            X_rank = ste_fp16(X_rank)

        # Apply Q to each rank's scaled input
        # Q: [out, in] -> Y_rank: [..., rank, out]
        Y_rank = torch.einsum('...ri,oi->...ro', X_rank, Q)  # [..., rank, out]
        if use_ste:
            Y_rank = ste_fp16(Y_rank)

        # Weight by A and sum over ranks
        # A: [rank, out] -> y: [..., out]
        y = (Y_rank * A).sum(dim=-2)  # [..., out]
        if use_ste:
            y = ste_fp16(y)

        return y

    @torch.no_grad()
    def freeze_for_inference(self):
        """Cache full W_eff = Q * scales for fast inference."""
        if self._Q is None:
            self.freeze_Q()

        # Compute full scales
        scales = self._compute_full_scales()

        # W_eff = Q * scales
        W_eff = self._Q * scales

        # Cache
        self._cached_weight_q = W_eff.to(self.weight.dtype)

    @torch.no_grad()
    def unfreeze_for_training(self):
        """Clear cached weights for training."""
        self._cached_weight_q = None
        self._scales_baked = False  # Re-enable normalization

    @torch.no_grad()
    def snap_for_export(self):
        """Bake normalization into params. No norms/divides in inference graph.

        After calling this:
        - scale_A contains A_dir * g (no longer unit-norm)
        - scale_B contains B_dir (unit-norm)
        - rank_magnitude is all ones
        - _cached_weight_q contains full W_eff
        - _scales_baked is True (skip normalization in forward)
        """
        A_dir, B_dir, g = self._get_normalized_scales()

        # Bake magnitude into A
        self.scale_A.data = (A_dir * g).to(self.weight.dtype)
        self.scale_B.data = B_dir.to(self.weight.dtype)
        self.rank_magnitude.data = torch.ones_like(g)

        # Mark scales as baked (skip normalization in forward)
        self._scales_baked = True

        # Cache full W_eff
        self.freeze_for_inference()

    @torch.no_grad()
    def snap_for_ane(self, recompute_indices: bool = True):
        """Snap for ANE export in FP16 precision.

        ANE runs in FP16, so we need to:
        1. Recompute LUT in FP16
        2. Recompute quantization indices in FP16 (optional)
        3. Compute scales in FP16
        4. Store Q = lut[indices] in FP16

        IMPORTANT: All FP16 conversions use CPU for consistent rounding.
        MPS/TPU/XLA may have different FP16 rounding behavior than CPU.

        Args:
            recompute_indices: If True, recompute indices in FP16 precision.
                              If False, keep existing indices but convert Q to FP16.
        """
        device = self.weight.device
        fp16 = torch.float16

        # Helper: CPU-based FP16 snap for consistent rounding across devices
        def cpu_fp16(t: torch.Tensor) -> torch.Tensor:
            """Snap tensor to FP16 via CPU (MPS/TPU have different rounding)."""
            return t.cpu().half().to(device)

        # 1. Use existing LUT, just convert to FP16 (don't recreate!)
        if hasattr(self, 'lut') and self.lut is not None:
            lut_fp16 = cpu_fp16(self.lut)
        else:
            # Fallback: create new LUT (shouldn't happen in trained model)
            lut_fp16 = make_lut(
                self.config.lut_size,
                device=device,
                dtype=fp16,
                include_zero=self.config.lut_include_zero,
            )

        if recompute_indices:
            # 2. Compute scales in FP16 (on CPU for consistent rounding)
            A_dir, B_dir, g = self._get_normalized_scales()
            A_dir = cpu_fp16(A_dir)
            B_dir = cpu_fp16(B_dir)
            g = cpu_fp16(g)

            # Compute full scales in FP16 (no clamping for V2)
            A_scaled = A_dir * g
            scales_fp16 = (A_scaled @ B_dir)

            # 3. Recompute indices in FP16
            weight_fp16 = cpu_fp16(self.weight)
            normalized = weight_fp16 / scales_fp16

            indices_fp16 = quantize_to_lut_indices(
                normalized,
                lut_size=self.config.lut_size,
                include_zero=self.config.lut_include_zero,
                lut=lut_fp16,  # Use actual LUT values for nearest-neighbor search
            )

            # Store indices
            self._indices = indices_fp16

            # Bake scales into scale_A, scale_B
            self.scale_A.data = A_scaled
            self.scale_B.data = B_dir
            self.rank_magnitude.data = torch.ones_like(g)

            # Mark scales as baked (skip normalization in forward)
            self._scales_baked = True

            # Compute Q = lut[indices] in FP16
            self._Q = lut_fp16[indices_fp16]
        else:
            # No recompute - just convert existing _Q to FP16
            if self._Q is not None:
                self._Q = cpu_fp16(self._Q)
            elif self._indices is not None:
                # Have indices but no _Q - compute from indices
                self._Q = lut_fp16[self._indices]
            else:
                # Neither _Q nor _indices - need to compute (fallback)
                self.freeze_Q()
                self._Q = cpu_fp16(self._Q)

            # Convert scales to FP16 via CPU (don't bake)
            self.scale_A.data = cpu_fp16(self.scale_A.data)
            self.scale_B.data = cpu_fp16(self.scale_B.data)
            self.rank_magnitude.data = cpu_fp16(self.rank_magnitude.data)

        # Update LUT buffer
        if hasattr(self, 'lut') and isinstance(self.lut, torch.Tensor):
            if isinstance(self.lut, nn.Parameter):
                self.lut.data = lut_fp16
            else:
                self.lut = lut_fp16

        # Convert weight and bias to FP16 via CPU
        self.weight.data = cpu_fp16(self.weight.data)
        if self.bias is not None:
            self.bias.data = cpu_fp16(self.bias.data)

        # Convert LoRA adapters to FP16 (if present)
        # Note: For ANE deployment, consider using resnap_with_lora() first
        # to merge LoRA into weights and avoid extra matmuls at inference.
        if self.lora_r > 0 and self.lora_A is not None:
            self.lora_A.data = cpu_fp16(self.lora_A.data)
            self.lora_B.data = cpu_fp16(self.lora_B.data)

        # Clear cached weight (force factored forward)
        self._cached_weight_q = None

    @torch.no_grad()
    def convert_to_fp16(self):
        """Convert all tensors to FP16 for FP16 training pipeline.

        Use this BEFORE freeze_Q() to ensure indices are computed in FP16.
        This ensures no precision mismatch between training and ANE inference.

        IMPORTANT: All FP16 conversions use CPU for consistent rounding.
        MPS/TPU/XLA may have different FP16 rounding behavior than CPU.
        """
        fp16 = torch.float16
        device = self.weight.device

        # Helper: CPU-based FP16 snap for consistent rounding across devices
        def cpu_fp16(t: torch.Tensor) -> torch.Tensor:
            """Snap tensor to FP16 via CPU (MPS/TPU have different rounding)."""
            return t.cpu().half().to(device)

        # Convert weight and bias via CPU
        self.weight.data = cpu_fp16(self.weight.data)
        if self.bias is not None:
            self.bias.data = cpu_fp16(self.bias.data)

        # Convert scales via CPU
        self.scale_A.data = cpu_fp16(self.scale_A.data)
        self.scale_B.data = cpu_fp16(self.scale_B.data)
        self.rank_magnitude.data = cpu_fp16(self.rank_magnitude.data)

        # Recompute LUT in FP16
        lut_fp16 = make_lut(
            self.config.lut_size,
            device=device,
            dtype=fp16,
            include_zero=self.config.lut_include_zero,
        )

        # Update LUT buffer
        if isinstance(self.lut, nn.Parameter):
            self.lut.data = lut_fp16
        else:
            self.lut = lut_fp16

        # Recompute Q in FP16 (if already frozen)
        if self._Q is not None:
            self._Q = self.lut[self._indices]

        # Convert LoRA adapters to FP16 via CPU (if present)
        if self.lora_r > 0 and self.lora_A is not None:
            self.lora_A.data = cpu_fp16(self.lora_A.data)
            self.lora_B.data = cpu_fp16(self.lora_B.data)

        # Clear cached weight
        self._cached_weight_q = None

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        config: Optional[AnemllQuantConfigV2] = None,
        skip_init: bool = False,
        custom_lut: Optional[torch.Tensor] = None,
    ) -> "AnemllQATLinearV2":
        """Create AnemllQATLinearV2 from existing nn.Linear.

        Args:
            linear: Source nn.Linear to convert
            config: Quantization config
            skip_init: If True, skip SVD-based scale initialization (caller will load state_dict)
            custom_lut: Optional custom LUT tensor to use instead of default uniform LUT
        """
        config = config or AnemllQuantConfigV2()

        # Create layer, skipping SVD init in __init__ (we'll do it after copying weights)
        layer = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            config=config,
            _skip_scale_init=True,  # Skip SVD in reset_parameters
            custom_lut=custom_lut,  # Pass through custom LUT
        )

        # Copy weights (handle dtype/device mismatch on TPU/XLA)
        with torch.no_grad():
            layer.weight.copy_(linear.weight)
            if linear.bias is not None:
                layer.bias.copy_(linear.bias)

        # Initialize scales from the actual weights (slow SVD - skip if loading state_dict)
        if not skip_init:
            layer._init_scales_from_weight()

        return layer

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"rank={self.scale_rank}, lut={self.config.lut_size}, "
            f"Q_frozen={self._Q is not None}"
        )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def replace_linear_with_anemll_v2(
    model: nn.Module,
    mlp_config: AnemllQuantConfigV2,
    attn_config: Optional[AnemllQuantConfigV2] = None,
    quantize_attn: bool = True,
    quantize_lm_head: bool = False,
    verbose: bool = True,
    skip_init: bool = False,
    parallel_init: bool = True,
    num_workers: int = 0,
) -> int:
    """Replace MLP and optionally attention linears with AnemllQATLinearV2.

    Args:
        skip_init: If True, skip SVD-based scale initialization (use when loading state_dict after).
                   This makes layer replacement ~10x faster for multi-GPU/TPU where only rank 0
                   needs to do the full initialization.
        parallel_init: If True and skip_init=False, run SVD initialization in parallel.
        num_workers: Number of parallel workers (0 = auto-detect CPU count).
    """
    import re
    from concurrent.futures import ThreadPoolExecutor
    import os

    mlp_pattern = re.compile(r'\.mlp\.(gate_proj|up_proj|down_proj)$')
    attn_pattern = re.compile(r'\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$')
    lm_head_pattern = re.compile(r'^lm_head$')

    if attn_config is None:
        attn_config = mlp_config

    # First pass: collect layers to replace
    layers_to_replace = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if isinstance(module, AnemllQATLinearV2):
            continue

        is_mlp = mlp_pattern.search(name)
        is_attn = attn_pattern.search(name)
        is_lm_head = lm_head_pattern.search(name)

        if is_mlp:
            cfg = mlp_config
        elif is_attn and quantize_attn:
            cfg = attn_config
        elif is_lm_head and quantize_lm_head:
            cfg = mlp_config
        else:
            continue

        layers_to_replace.append((name, module, cfg))

    # Sequential SVD initialization (parallel disabled - GIL + BLAS conflicts make it slower)
    # Each SVD uses full BLAS parallelism internally, so sequential is actually faster
    new_modules = {}
    total_layers = len(layers_to_replace)
    if verbose and total_layers > 0:
        print(f"  Converting {total_layers} layers to V2...")
    for idx, (name, module, cfg) in enumerate(layers_to_replace):
        new_modules[name] = AnemllQATLinearV2.from_linear(module, config=cfg, skip_init=skip_init)
        # Snap rank_magnitude to FP16 (SVD init gives FP32 values)
        if not skip_init:
            with torch.no_grad():
                new_modules[name].rank_magnitude.data = new_modules[name].rank_magnitude.data.to(torch.float16).to(torch.float32)
        # Show progress every 20 layers or at end
        if verbose and (idx % 20 == 0 or idx == total_layers - 1):
            pct = 100 * (idx + 1) / total_layers
            short_name = name.split('.')[-2] + '.' + name.split('.')[-1] if '.' in name else name
            print(f"    [{idx + 1}/{total_layers}] ({pct:.0f}%) {short_name}", flush=True)

    # Build replacement list with parent references
    replacements = []
    named_modules_dict = dict(model.named_modules())
    for name, module, cfg in layers_to_replace:
        new_module = new_modules[name]
        parts = name.rsplit('.', 1)
        if len(parts) == 2:
            parent_name, attr = parts
            parent = named_modules_dict[parent_name]
        else:
            parent = model
            attr = name
        replacements.append((parent, attr, new_module, name))

    # Apply replacements
    replaced_names = []
    for parent, attr, new_module, name in replacements:
        setattr(parent, attr, new_module)
        replaced_names.append(name)

    if verbose:
        # Group by pattern (replace layer numbers with *)
        import re
        patterns = {}
        for name in replaced_names:
            # Replace layer numbers with * for grouping
            pattern = re.sub(r'\.layers\.(\d+)\.', '.layers.*.', name)
            if pattern not in patterns:
                patterns[pattern] = []
            patterns[pattern].append(name)

        # Print compact summary
        for pattern, names in sorted(patterns.items()):
            if len(names) > 1:
                print(f'  [replaced] {pattern} ({len(names)} layers)')
            else:
                print(f'  [replaced] {names[0]}')

        print(f'\nReplaced {len(replacements)} layers with V2', flush=True)

    return len(replacements)


def freeze_Q_all(model: nn.Module, verbose: bool = False) -> int:
    """Freeze Q for all AnemllQATLinearV2 layers."""
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, AnemllQATLinearV2):
            module.freeze_Q()
            count += 1
            if verbose:
                print(f'  [freeze_Q] {name}')
    return count


@torch.no_grad()
def sync_q_from_indices(
    module,  # AnemllQATLinearV2
    fp16: bool = True,
    prefer_trainable_lut: bool = True,
) -> bool:
    """Refresh _Q buffer from lut[_indices] to ensure consistency.

    This is the canonical way to ensure _Q reflects the current LUT and indices.
    Call this after changing LUT or indices to make utilities that read _Q
    see the updated quantization.

    Args:
        module: AnemllQATLinearV2 module
        fp16: If True, store _Q in FP16 (matches FP16-snapped behavior)
        prefer_trainable_lut: If True, use get_lut() which handles trainable LUT case

    Returns:
        True if sync succeeded, False if _indices not available
    """
    if module._indices is None:
        return False

    # Get current LUT
    if prefer_trainable_lut and hasattr(module, 'get_lut'):
        lut = module.get_lut()
    else:
        lut = module.lut

    # Compute Q from lut[indices]
    Q_new = lut[module._indices.long()]

    # Convert to target dtype
    target_dtype = torch.float16 if fp16 else lut.dtype
    Q_new = Q_new.to(target_dtype)

    # Update or create _Q buffer
    if module._Q is not None:
        module._Q.copy_(Q_new)
    else:
        module.register_buffer("_Q", Q_new)

    return True


def sync_q_from_indices_all(
    model: nn.Module,
    fp16: bool = True,
    prefer_trainable_lut: bool = True,
    verbose: bool = False,
) -> int:
    """Sync _Q from lut[_indices] for all V2 layers.

    Call this after baking LUTs or before saving checkpoints to ensure
    _Q is consistent with LUT and indices.

    Returns:
        Number of layers synced
    """
    count = 0
    skipped = 0
    for name, module in model.named_modules():
        if isinstance(module, AnemllQATLinearV2):
            if sync_q_from_indices(module, fp16=fp16, prefer_trainable_lut=prefer_trainable_lut):
                count += 1
                if verbose:
                    print(f'  [sync_q] {name}')
            else:
                skipped += 1
                if verbose:
                    print(f'  [sync_q] {name}: skipped (no _indices)')

    if verbose and count > 0:
        print(f'  Synced _Q for {count} layers')
        if skipped > 0:
            print(f'  Skipped {skipped} layers (no _indices)')

    return count


def enable_lut_training_all(
    model: nn.Module,
    scope: str = 'all',
    max_abs: float = 1.0,
    allow_bad_qc: bool = False,
    verbose: bool = True,
) -> int:
    """Enable LUT training for all V2 layers matching scope. Must be called on CPU.

    Args:
        model: The model containing AnemllQATLinearV2 layers
        scope: Which layers to enable LUT training for:
            - 'all': All layers
            - 'mlp': Only MLP layers (name contains 'mlp')
            - 'attn': Only attention layers (name contains 'self_attn')
        max_abs: Maximum absolute LUT value (default 1.0, try 2.0 for outliers)
        allow_bad_qc: If True, proceed with _Q→_indices conversion even when
            error exceeds threshold. Default False (fail-closed).
        verbose: Print progress messages

    Returns:
        Number of layers with LUT training enabled
    """
    count = 0
    failed = 0
    skipped = 0

    for name, module in model.named_modules():
        if not isinstance(module, AnemllQATLinearV2):
            continue

        # Check scope
        if scope == 'mlp' and 'mlp' not in name:
            continue
        if scope == 'attn' and 'self_attn' not in name:
            continue

        # Only enable for LUT16 (4-bit)
        if module.lut is None:
            if verbose:
                print(f"  Skipping {name}: no LUT")
            skipped += 1
            continue

        if module.lut.shape[0] != 16:
            if verbose:
                print(f"  Skipping {name}: lut_size={module.lut.shape[0]} (only 16 supported)")
            skipped += 1
            continue

        if verbose:
            print(f"  Enabling LUT training: {name}")

        if module.enable_lut_training(max_abs=max_abs, allow_bad_qc=allow_bad_qc):
            count += 1
        else:
            failed += 1
            if verbose:
                print(f"    FAILED: {name}")

    if verbose:
        print(f"  LUT training enabled: {count} layers")
        if failed > 0:
            print(f"  WARNING: {failed} layers failed LUT training enable (QC failures)")
        if skipped > 0:
            print(f"  Skipped: {skipped} layers (non-LUT16)")

        # Smoke test: print _indices info from first enabled layer
        if count > 0:
            for m in model.modules():
                if isinstance(m, AnemllQATLinearV2) and m._lut_trainable and m._indices is not None:
                    idx = m._indices
                    print(f"  [Smoke test] _indices: device={idx.device}, dtype={idx.dtype}, "
                          f"shape={tuple(idx.shape)}, range=[{idx.min().item()}, {idx.max().item()}]")
                    break

    return count


def freeze_model_for_inference_v2(model: nn.Module, verbose: bool = False) -> int:
    """Freeze all V2 layers for fast inference."""
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, AnemllQATLinearV2):
            module.freeze_for_inference()
            count += 1
            if verbose:
                print(f'  [frozen] {name}')
    return count


def unfreeze_model_for_training_v2(model: nn.Module) -> int:
    """Unfreeze all V2 layers for training."""
    count = 0
    for module in model.modules():
        if isinstance(module, AnemllQATLinearV2):
            module.unfreeze_for_training()
            count += 1
    return count


def set_factored_inference_v2(model: nn.Module, enabled: bool = True, verbose: bool = True) -> int:
    """Set factored inference mode for all V2 layers.

    Args:
        model: The model containing V2 layers
        enabled: If True, use rank-by-rank factored forward (ANE-friendly)
                 If False, use single matmul with cached weights (faster PyTorch)
        verbose: Print diagnostic information

    Returns:
        Number of layers updated
    """
    count = 0
    for module in model.modules():
        if isinstance(module, AnemllQATLinearV2):
            module.use_factored_inference = enabled
            count += 1

    if verbose:
        mode = "FACTORED (rank-by-rank)" if enabled else "SINGLE MATMUL (cached W_eff)"
        print(f"[V2 Inference Mode] {mode}")
        print(f"  Updated {count} layers")
        if enabled:
            print(f"  Forward: y = Σₖ gₖ · (aₖ ⊙ (Q @ (bₖ ⊙ x)))")
            print(f"  No A @ B materialization in forward pass")
        else:
            print(f"  Forward: y = W_eff @ x  (W_eff = Q * scales, precomputed)")
            print(f"  Faster but materializes full [out, in] weight")

    return count


def get_inference_mode_v2(model: nn.Module) -> dict:
    """Get inference mode statistics for V2 layers.

    Returns:
        Dictionary with counts of layers in each mode
    """
    factored = 0
    single_matmul = 0
    has_cached = 0
    has_Q = 0

    for module in model.modules():
        if isinstance(module, AnemllQATLinearV2):
            if module.use_factored_inference:
                factored += 1
            else:
                single_matmul += 1
            if module._cached_weight_q is not None:
                has_cached += 1
            if module._Q is not None:
                has_Q += 1

    total = factored + single_matmul
    return {
        'total': total,
        'factored': factored,
        'single_matmul': single_matmul,
        'has_cached_weights': has_cached,
        'has_frozen_Q': has_Q,
    }


def set_batched_forward_v2(
    model: nn.Module,
    enabled: bool = True,
    auto: bool = False,
    threshold: int = 16,
    verbose: bool = True,
) -> int:
    """Set batched forward mode for all V2 layers.

    Batched forward is more memory-efficient for high-rank configs because
    it makes 4 STE calls per layer (vs 4*rank for loop-based).

    Args:
        model: The model containing V2 layers
        enabled: If True, use batched forward. If False, use loop-based.
        auto: If True, auto-detect based on scale_rank >= threshold
        threshold: Rank threshold for auto mode (default 16)
        verbose: Print diagnostic information

    Returns:
        Number of layers updated
    """
    count = 0
    batched_count = 0

    for module in model.modules():
        if isinstance(module, AnemllQATLinearV2):
            if auto:
                use_batched = module.scale_rank >= threshold
            else:
                use_batched = enabled

            module.use_batched_forward = use_batched
            count += 1
            if use_batched:
                batched_count += 1

    if verbose:
        if auto:
            print(f"[V2 Forward Mode] AUTO (rank >= {threshold})")
        else:
            mode = "BATCHED" if enabled else "LOOP"
            print(f"[V2 Forward Mode] {mode}")
        print(f"  Total layers: {count}")
        print(f"  Batched: {batched_count}, Loop: {count - batched_count}")
        if batched_count > 0:
            print(f"  STE calls per layer: 4 (vs 4*rank for loop)")

    return count


def snap_model_for_ane_v2(
    model: nn.Module,
    recompute_indices: bool = True,
    verbose: bool = True,
) -> int:
    """Snap all V2 layers for ANE export in FP16 precision.

    This converts V2 quantized layers to FP16 and optionally recomputes
    quantization indices to match ANE's FP16 precision.

    IMPORTANT: This function ONLY touches AnemllQATLinearV2 layers.
    It does NOT modify:
    - embed_tokens (embedding lookup table - semantic precision matters)
    - lm_head (output projection - not quantized)
    - LayerNorm/RMSNorm (normalization layers)

    Keeping embed_tokens in full precision is critical because FP16 rounding
    causes ~0.02 max error which corrupts vocabulary embeddings.

    If LoRA adapters are present, they will be converted to FP16.
    For best ANE performance, consider calling resnap_with_lora() first
    to merge LoRA into the quantized weights (avoids extra matmuls).

    Args:
        model: The model containing V2 layers
        recompute_indices: If True, recompute indices in FP16 (recommended)
        verbose: Print diagnostic information

    Returns:
        Number of V2 layers snapped
    """
    count = 0
    lora_count = 0
    for name, module in model.named_modules():
        if isinstance(module, AnemllQATLinearV2):
            # Check for LoRA before snapping
            if module.lora_r > 0 and module.lora_A is not None:
                lora_count += 1
            module.snap_for_ane(recompute_indices=recompute_indices)
            count += 1
            if verbose and count <= 3:
                print(f"  [snap_ane] {name}")

    if verbose:
        print(f"\n[ANE FP16 Snap]")
        print(f"  Snapped {count} V2 layers to FP16")
        print(f"  Recomputed indices: {recompute_indices}")
        print(f"  V2 weights, scales, LUT, Q now in FP16")
        print(f"  embed_tokens, lm_head, norms: UNCHANGED (full precision)")
        if lora_count > 0:
            print(f"\n  [LoRA] {lora_count} layers have LoRA adapters (converted to FP16)")
            print(f"  [LoRA] For best ANE perf, consider: resnap_with_lora(model) first")
            print(f"         This merges LoRA into weights, avoiding 2 extra matmuls/layer")

    return count


def convert_model_to_fp16_v2(model: nn.Module, verbose: bool = True) -> int:
    """Convert all V2 layers to FP16 for FP16 training pipeline.

    Use this BEFORE freeze_Q_all() to ensure indices are computed in FP16.
    This is for training in FP16 from the start (not post-training conversion).

    Pipeline:
        1. Load model in FP16
        2. replace_linear_with_anemll_v2()
        3. convert_model_to_fp16_v2()  <-- ensures LUT is FP16
        4. freeze_Q_all()               <-- indices computed in FP16
        5. train_e2e(use_fp16=True)

    Args:
        model: The model containing V2 layers
        verbose: Print diagnostic information

    Returns:
        Number of layers converted
    """
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, AnemllQATLinearV2):
            module.convert_to_fp16()
            count += 1
            if verbose and count <= 3:
                print(f"  [fp16] {name}")

    if verbose:
        print(f"\n[FP16 Conversion]")
        print(f"  Converted {count} V2 layers to FP16")
        print(f"  LUT, weights, scales all in FP16")
        print(f"  Ready for FP16 training pipeline")

    return count


def load_v2_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: Optional[torch.device] = None,
    verbose: bool = True,
    prefer_indices: bool = True,
) -> dict:
    """Load a V2 checkpoint with proper handling of _Q and _indices buffers.

    PyTorch's load_state_dict doesn't load tensors into None buffers.
    This function pre-registers the buffers with correct shapes before loading.

    Args:
        model: Model with V2 layers (after replace_linear_with_anemll_v2)
        checkpoint_path: Path to the checkpoint file
        device: Device to load to (default: CPU)
        verbose: Print loading statistics
        prefer_indices: If True, set _use_indices=True when _indices are loaded.
                       This makes forward() use lut[_indices] instead of _Q directly,
                       so that any changes to .lut take effect. Set to True for
                       inference/evaluation, False for training continuation where
                       you want the original training path.

    Returns:
        Dictionary with loading statistics
    """
    device = device or torch.device('cpu')

    # Load checkpoint
    state_dict = torch.load(checkpoint_path, map_location='cpu')

    # Find V2 layers and pre-register buffers with correct shapes
    v2_layers = {}
    for name, module in model.named_modules():
        if isinstance(module, AnemllQATLinearV2):
            v2_layers[name] = module

    # Check for _Q and _indices in checkpoint and pre-register them
    q_loaded = 0
    indices_loaded = 0

    for name, module in v2_layers.items():
        # Look for _Q key
        q_key = f"{name}._Q"
        if q_key in state_dict:
            q_tensor = state_dict[q_key]
            # Re-register buffer with actual tensor (not None)
            module.register_buffer("_Q", q_tensor.to(device))
            q_loaded += 1

        # Look for _indices key
        indices_key = f"{name}._indices"
        if indices_key in state_dict:
            indices_tensor = state_dict[indices_key]
            module.register_buffer("_indices", indices_tensor.to(device))
            # Set _use_indices based on prefer_indices parameter
            # True: forward() uses lut[_indices] (inference - LUT changes take effect)
            # False: forward() uses _Q directly (training continuation - original path)
            if prefer_indices:
                module._use_indices = True
            indices_loaded += 1

    # Now load the rest of the state dict
    result = model.load_state_dict(state_dict, strict=False)

    # Move to device
    model.to(device)

    # Filter out expected missing keys (new buffers not in old checkpoints)
    expected_missing = {'_scales_baked_flag'}
    actual_missing = [k for k in result.missing_keys
                      if not any(exp in k for exp in expected_missing)]

    stats = {
        'v2_layers': len(v2_layers),
        'q_loaded': q_loaded,
        'indices_loaded': indices_loaded,
        'missing_keys': len(actual_missing),
        'missing_keys_expected': len(result.missing_keys) - len(actual_missing),
        'unexpected_keys': len(result.unexpected_keys),
    }

    if verbose:
        print(f"\n[V2 Checkpoint Loaded]")
        print(f"  V2 layers: {stats['v2_layers']}")
        print(f"  _Q loaded: {stats['q_loaded']}")
        print(f"  _indices loaded: {stats['indices_loaded']}")
        if indices_loaded > 0:
            mode = "lut[_indices]" if prefer_indices else "_Q direct"
            print(f"  _use_indices: {prefer_indices} ({mode})")
        print(f"  Missing keys: {stats['missing_keys']}")
        if stats['missing_keys_expected'] > 0:
            print(f"  Missing (expected, new buffers): {stats['missing_keys_expected']}")
        # Unexpected should be 0 now since we pre-loaded _Q and _indices
        remaining_unexpected = stats['unexpected_keys'] - (q_loaded + indices_loaded)
        print(f"  Unexpected keys (other): {max(0, remaining_unexpected)}")

        if stats['q_loaded'] == stats['v2_layers']:
            print(f"  Ready for inference (no freeze_Q needed)")
        else:
            print(f"  Call freeze_Q_all() before inference")

    return stats


# =============================================================================
# QRANKLUT MODEL-LEVEL HELPERS
# =============================================================================

def init_rank_lut_all(
    model: nn.Module,
    verbose: bool = True,
) -> dict:
    """Initialize qranklut (k-means LUT) for all V2 layers.

    This function:
    1. Iterates through all AnemllQATLinearV2 layers
    2. Runs k-means on A_dir and B_dir to find optimal LUT values
    3. Assigns indices and enables qranklut mode

    Call this after loading a trained checkpoint to add qranklut compression.

    Args:
        model: Model with V2 layers
        verbose: Print initialization statistics

    Returns:
        Dictionary with statistics
    """
    count = 0
    total_A_err = 0.0
    total_B_err = 0.0

    for name, module in model.named_modules():
        if isinstance(module, AnemllQATLinearV2):
            if module.A_lut is None or module.config.rank_lut_size <= 0:
                continue

            module.init_rank_lut(verbose=False)
            count += 1

            if verbose:
                # Compute reconstruction error for summary
                if module._A_idx is not None:
                    A_dir = module.scale_A.abs() if module.config.force_positive_scales else module.scale_A
                    A_norms = A_dir.norm(dim=0, keepdim=True).clamp(min=1e-6)
                    A_dir_norm = (A_dir / A_norms).float()
                    A_recon = module.A_lut[module._A_idx]
                    A_err = (A_dir_norm - A_recon).abs().mean().item()
                    total_A_err += A_err

                    B_dir = module.scale_B.abs() if module.config.force_positive_scales else module.scale_B
                    B_norms = B_dir.norm(dim=1, keepdim=True).clamp(min=1e-6)
                    B_dir_norm = (B_dir / B_norms).float()
                    B_recon = module.B_lut[module._B_idx]
                    B_err = (B_dir_norm - B_recon).abs().mean().item()
                    total_B_err += B_err

    stats = {
        'layers_initialized': count,
        'avg_A_mae': total_A_err / max(count, 1),
        'avg_B_mae': total_B_err / max(count, 1),
    }

    if verbose:
        print(f"\n[QRANKLUT Initialized]")
        print(f"  Layers: {count}")
        if count > 0:
            lut_size = model.modules().__next__().config.rank_lut_size if hasattr(model, 'modules') else 0
            # Find first V2 layer to get config
            for m in model.modules():
                if isinstance(m, AnemllQATLinearV2) and m.A_lut is not None:
                    lut_size = m.config.rank_lut_size
                    break
            print(f"  LUT size: {lut_size} ({int(math.log2(lut_size))}-bit)")
            print(f"  Avg MAE: A={stats['avg_A_mae']:.6f}, B={stats['avg_B_mae']:.6f}")

    return stats


def freeze_rank_lut_all(model: nn.Module, verbose: bool = True) -> int:
    """Freeze rank LUT indices for all V2 layers.

    After calling this:
    - Indices (_A_idx, _B_idx) are frozen
    - LUT values (A_lut, B_lut) can still be trained

    Args:
        model: Model with V2 layers
        verbose: Print summary

    Returns:
        Number of layers frozen
    """
    count = 0
    for module in model.modules():
        if isinstance(module, AnemllQATLinearV2):
            if module._A_idx is not None:
                module.freeze_rank_lut()
                count += 1

    if verbose:
        print(f"[QRANKLUT] Froze indices for {count} layers")

    return count


def enable_rank_lut_training_all(
    model: nn.Module,
    train_lut: bool = True,
    train_g: bool = True,
    train_scales: bool = False,
    freeze_base_q: bool = True,
    verbose: bool = True,
) -> dict:
    """Enable qranklut training mode for all V2 layers.

    This sets up the model for training LUT values and/or magnitudes.

    Args:
        model: Model with V2 layers
        train_lut: If True, A_lut and B_lut are trainable
        train_g: If True, rank_magnitude is trainable
        train_scales: If True, scale_A and scale_B are trainable
        freeze_base_q: If True, freeze base Q (weight indices)
        verbose: Print summary

    Returns:
        Dictionary with training configuration
    """
    lut_count = 0
    g_count = 0
    scales_count = 0

    for module in model.modules():
        if isinstance(module, AnemllQATLinearV2):
            # LUT training
            if module.A_lut is not None:
                if isinstance(module.A_lut, nn.Parameter):
                    module.A_lut.requires_grad = train_lut
                    module.B_lut.requires_grad = train_lut
                    if train_lut:
                        lut_count += 1

            # Magnitude training
            module.rank_magnitude.requires_grad = train_g
            if train_g:
                g_count += 1

            # Scale training
            module.scale_A.requires_grad = train_scales
            module.scale_B.requires_grad = train_scales
            if train_scales:
                scales_count += 1

            # Base Q freezing
            if freeze_base_q:
                module.weight.requires_grad = False

    stats = {
        'lut_trainable': lut_count,
        'g_trainable': g_count,
        'scales_trainable': scales_count,
    }

    if verbose:
        print(f"\n[QRANKLUT Training Mode]")
        print(f"  LUT trainable: {lut_count} layers")
        print(f"  Magnitude (g) trainable: {g_count} layers")
        print(f"  Scales trainable: {scales_count} layers")
        print(f"  Base Q frozen: {freeze_base_q}")

    return stats


def get_rank_lut_stats(model: nn.Module) -> dict:
    """Get qranklut statistics for all V2 layers.

    Returns:
        Dictionary with qranklut statistics
    """
    enabled = 0
    has_indices = 0
    total_lut_params = 0
    total_idx_elements = 0

    for module in model.modules():
        if isinstance(module, AnemllQATLinearV2):
            if getattr(module, '_rank_lut_enabled', False):
                enabled += 1
            if module._A_idx is not None:
                has_indices += 1
                total_idx_elements += module._A_idx.numel() + module._B_idx.numel()
            if module.A_lut is not None:
                total_lut_params += module.A_lut.numel() + module.B_lut.numel()

    return {
        'enabled': enabled,
        'has_indices': has_indices,
        'total_lut_params': total_lut_params,
        'total_idx_elements': total_idx_elements,
        'lut_memory_kb': total_lut_params * 4 / 1024,  # FP32
        'idx_memory_kb': total_idx_elements * 1 / 1024,  # uint8
    }


# =============================================================================
# RECOVERY LORA MODEL-LEVEL HELPERS
# =============================================================================

def enable_recovery_lora_all(
    model: nn.Module,
    r: int = 8,
    alpha: float = None,
    dropout: float = 0.0,
    skip_k_proj: bool = True,
    mlp_only: bool = False,
    verbose: bool = True,
) -> dict:
    """Enable recovery LoRA adapters on all V2 layers.

    This function enables LoRA on selected layers following Apple's adapter
    placement strategy (Q, V, O for attention; gate/up/down for MLP).

    CRITICAL ORDER for DDP/optimizer compatibility:
        1. model = load_model(...)
        2. model.load_state_dict(checkpoint)
        3. enable_recovery_lora_all(model, r=8)  # <-- enables LoRA
        4. freeze_for_recovery_training(model)
        5. optimizer = AdamW([p for p in model.parameters() if p.requires_grad])
        6. model = DDP(model) / FSDP(model) if distributed

    Args:
        model: Model with V2 layers
        r: LoRA rank (start with 8, increase to 16/32 if needed)
        alpha: LoRA alpha (default: r)
        dropout: LoRA dropout (default: 0.0)
        skip_k_proj: If True, skip K projection (Apple default)
        mlp_only: If True, only enable LoRA on MLP layers (recommended first)
        verbose: Print summary

    Returns:
        Dictionary with mask info for checkpoint:
        {
            'recovery_lora_mask': {'attn': [...], 'mlp': [...]},
            'recovery_lora_r': r,
            'recovery_lora_alpha': alpha,
            'layers_enabled': count
        }
    """
    alpha = alpha if alpha is not None else float(r)
    mask = {'attn': [], 'mlp': []}
    count = 0
    total_params = 0

    attn_names = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    mlp_names = ['gate_proj', 'up_proj', 'down_proj']

    for name, module in model.named_modules():
        if not isinstance(module, AnemllQATLinearV2):
            continue

        # Determine layer type
        is_attn = any(a in name for a in attn_names)
        is_mlp = any(m in name for m in mlp_names)
        is_k_proj = 'k_proj' in name

        # Apply mask logic
        if is_k_proj and skip_k_proj:
            continue
        if mlp_only and is_attn:
            continue

        # Enable LoRA
        module.enable_recovery_lora(r=r, alpha=alpha, dropout=dropout)
        count += 1

        # Track params
        if module.lora_A is not None:
            total_params += module.lora_A.numel() + module.lora_B.numel()

        # Track mask
        for a in attn_names:
            if a in name and a not in mask['attn']:
                mask['attn'].append(a)
        for m in mlp_names:
            if m in name and m not in mask['mlp']:
                mask['mlp'].append(m)

    result = {
        'recovery_lora_mask': mask,
        'recovery_lora_r': r,
        'recovery_lora_alpha': alpha,
        'layers_enabled': count,
        'total_lora_params': total_params,
    }

    if verbose:
        print(f"\n[Recovery LoRA Enabled]")
        print(f"  Rank: {r}, Alpha: {alpha}")
        print(f"  Layers: {count}")
        print(f"  LoRA params: {total_params:,} ({total_params * 4 / 1024 / 1024:.1f} MB FP32)")
        print(f"  Attention mask: {mask['attn']}")
        print(f"  MLP mask: {mask['mlp']}")
        if mlp_only:
            print(f"  Mode: MLP-only (attention LoRA disabled)")

    return result


def freeze_for_recovery_training(
    model: nn.Module,
    verbose: bool = True,
    train_norms: bool = False,
    train_embeddings: bool = False,
    train_lm_head: bool = False,
) -> dict:
    """Freeze all quantized params, keep only LoRA trainable.

    After calling this, only lora_A and lora_B will have requires_grad=True.
    All other parameters (weights, scales, Q, rank_magnitude, LUTs) are frozen.

    Optionally, specific components can be kept trainable:
    - train_norms: Keep layernorm weights trainable
    - train_embeddings: Keep embed_tokens trainable
    - train_lm_head: Keep lm_head trainable

    Args:
        model: Model with V2 layers (after enable_recovery_lora_all)
        verbose: Print summary
        train_norms: Keep layernorm weights trainable (default: False)
        train_embeddings: Keep embed_tokens trainable (default: False)
        train_lm_head: Keep lm_head trainable (default: False)

    Returns:
        Dictionary with frozen/trainable counts
    """
    frozen_count = 0
    trainable_count = 0
    lora_params = 0

    for name, module in model.named_modules():
        if isinstance(module, AnemllQATLinearV2):
            # Freeze quantized weights
            if module.weight is not None:
                module.weight.requires_grad = False
                frozen_count += 1

            # Freeze scales
            if module.scale_A is not None:
                module.scale_A.requires_grad = False
                frozen_count += 1
            if module.scale_B is not None:
                module.scale_B.requires_grad = False
                frozen_count += 1

            # Freeze rank_magnitude
            if module.rank_magnitude is not None:
                module.rank_magnitude.requires_grad = False
                frozen_count += 1

            # Freeze LUT (if parameter)
            if isinstance(module.lut, nn.Parameter):
                module.lut.requires_grad = False
                frozen_count += 1

            # Freeze rank LUTs (if parameters)
            if module.A_lut is not None and isinstance(module.A_lut, nn.Parameter):
                module.A_lut.requires_grad = False
                module.B_lut.requires_grad = False
                frozen_count += 2

            # LoRA stays trainable (if enabled)
            if module.lora_r > 0 and module.lora_A is not None:
                module.lora_A.requires_grad = True
                module.lora_B.requires_grad = True
                trainable_count += 2
                lora_params += module.lora_A.numel() + module.lora_B.numel()

    # Collect V2 module names so we can skip their params (already frozen above)
    v2_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, AnemllQATLinearV2):
            v2_module_names.add(name)

    # Also freeze non-V2 parameters (embeddings, layernorm, lm_head, etc.)
    # Unless explicitly kept trainable via flags
    other_frozen = 0
    embed_frozen = 0
    norm_frozen = 0
    lm_head_frozen = 0
    embed_kept_trainable = 0
    norm_kept_trainable = 0
    lm_head_kept_trainable = 0

    for name, param in model.named_parameters():
        # Skip V2 LoRA params (keep trainable)
        if 'lora_A' in name or 'lora_B' in name:
            continue

        # Skip params belonging to V2 layers (already frozen above)
        # Check if this param belongs to a V2 module
        is_v2_param = False
        for v2_name in v2_module_names:
            if name.startswith(v2_name + '.'):
                is_v2_param = True
                break
        if is_v2_param:
            continue

        # Check if this param should stay trainable based on flags
        is_embed = 'embed_tokens' in name
        is_norm = 'norm' in name.lower() or 'layernorm' in name.lower()
        is_lm_head = 'lm_head' in name

        # Keep trainable if flag is set
        if is_embed and train_embeddings:
            embed_kept_trainable += 1
            continue
        if is_norm and train_norms:
            norm_kept_trainable += 1
            continue
        if is_lm_head and train_lm_head:
            lm_head_kept_trainable += 1
            continue

        # Freeze everything else (embeddings, layernorm, lm_head, etc.)
        if param.requires_grad:
            param.requires_grad = False
            other_frozen += 1
            # Track what we're freezing for verbose output
            if is_embed:
                embed_frozen += 1
            elif 'norm' in name.lower() or 'layernorm' in name.lower():
                norm_frozen += 1
            elif 'lm_head' in name:
                lm_head_frozen += 1

    stats = {
        'v2_frozen': frozen_count,
        'lora_trainable': trainable_count,
        'other_frozen': other_frozen,
        'embed_frozen': embed_frozen,
        'norm_frozen': norm_frozen,
        'lm_head_frozen': lm_head_frozen,
        'embed_kept_trainable': embed_kept_trainable,
        'norm_kept_trainable': norm_kept_trainable,
        'lm_head_kept_trainable': lm_head_kept_trainable,
        'total_lora_params': lora_params,
    }

    if verbose:
        print(f"\n[Freeze for Recovery Training]")
        print(f"  V2 params frozen: {frozen_count}")
        print(f"  Other params frozen: {other_frozen}")
        if embed_frozen:
            print(f"    - embed_tokens: {embed_frozen}")
        if norm_frozen:
            print(f"    - layernorms: {norm_frozen}")
        if lm_head_frozen:
            print(f"    - lm_head: {lm_head_frozen}")
        print(f"  LoRA params trainable: {trainable_count}")
        print(f"  Total LoRA params: {lora_params:,}")
        # Show intentionally kept trainable params
        kept_trainable_total = embed_kept_trainable + norm_kept_trainable + lm_head_kept_trainable
        if kept_trainable_total > 0:
            print(f"  Extra params kept trainable: {kept_trainable_total}")
            if embed_kept_trainable:
                print(f"    - embed_tokens: {embed_kept_trainable} (--train-embeddings)")
            if norm_kept_trainable:
                print(f"    - layernorms: {norm_kept_trainable} (--train-norms)")
            if lm_head_kept_trainable:
                print(f"    - lm_head: {lm_head_kept_trainable} (--train-lm-head)")

    # Sanity check: verify only LoRA params (+ intentionally kept params) are trainable
    unexpected_trainable = []
    for name, param in model.named_parameters():
        if param.requires_grad and 'lora_' not in name:
            # Check if this param was intentionally kept trainable
            is_allowed = False
            if train_norms and ('norm' in name.lower() or 'layernorm' in name.lower()):
                is_allowed = True
            if train_embeddings and 'embed_tokens' in name:
                is_allowed = True
            if train_lm_head and 'lm_head' in name:
                is_allowed = True
            if not is_allowed:
                unexpected_trainable.append(name)

    if unexpected_trainable:
        import warnings
        warnings.warn(
            f"[freeze_for_recovery_training] WARNING: {len(unexpected_trainable)} unexpected params "
            f"are still trainable! First 5: {unexpected_trainable[:5]}"
        )
        stats['unexpected_trainable'] = unexpected_trainable
        if verbose:
            print(f"  ⚠️  WARNING: {len(unexpected_trainable)} unexpected params still trainable!")
            for n in unexpected_trainable[:5]:
                print(f"      {n}")
            if len(unexpected_trainable) > 5:
                print(f"      ... and {len(unexpected_trainable) - 5} more")

    return stats


def get_recovery_lora_params(model: nn.Module) -> int:
    """Get total number of trainable LoRA parameters.

    Args:
        model: Model with V2 layers

    Returns:
        Total number of trainable LoRA parameters
    """
    total = 0
    for module in model.modules():
        if isinstance(module, AnemllQATLinearV2):
            if module.lora_r > 0 and module.lora_A is not None:
                if module.lora_A.requires_grad:
                    total += module.lora_A.numel()
                if module.lora_B.requires_grad:
                    total += module.lora_B.numel()
    return total


def get_recovery_lora_stats(model: nn.Module) -> dict:
    """Get detailed LoRA statistics for all V2 layers.

    Args:
        model: Model with V2 layers

    Returns:
        Dictionary with LoRA statistics
    """
    enabled = 0
    total_params = 0
    layers_by_type = {'attn': 0, 'mlp': 0, 'other': 0}

    for name, module in model.named_modules():
        if isinstance(module, AnemllQATLinearV2):
            if module.lora_r > 0:
                enabled += 1
                total_params += module.lora_A.numel() + module.lora_B.numel()

                # Categorize
                if any(a in name for a in ['q_proj', 'k_proj', 'v_proj', 'o_proj']):
                    layers_by_type['attn'] += 1
                elif any(m in name for m in ['gate_proj', 'up_proj', 'down_proj']):
                    layers_by_type['mlp'] += 1
                else:
                    layers_by_type['other'] += 1

    return {
        'enabled': enabled,
        'total_params': total_params,
        'memory_mb': total_params * 4 / 1024 / 1024,  # FP32
        'layers_by_type': layers_by_type,
    }


def disable_recovery_lora_all(model: nn.Module, verbose: bool = True) -> int:
    """Disable recovery LoRA on all V2 layers.

    Args:
        model: Model with V2 layers
        verbose: Print summary

    Returns:
        Number of layers with LoRA disabled
    """
    count = 0
    for module in model.modules():
        if isinstance(module, AnemllQATLinearV2):
            if module.lora_r > 0:
                module.disable_recovery_lora()
                count += 1

    if verbose:
        print(f"[Recovery LoRA Disabled] {count} layers")

    return count


@torch.no_grad()
def resnap_with_lora(
    model: nn.Module,
    verbose: bool = True,
) -> dict:
    """Merge LoRA into effective weights and re-quantize.

    IMPORTANT: This is a NON-TRIVIAL operation with LUT quantization.
    It computes W_eff = W_q + LoRA_delta, then re-quantizes to new indices.

    Use this to export a LoRA-free model for ANE deployment when LoRA
    matmuls hurt inference performance.

    Pipeline:
        1. Compute effective FP weight with LoRA applied
        2. Recompute LUT indices
        3. Update _Q buffer
        4. Clear LoRA params

    Args:
        model: Model with V2 layers (with LoRA enabled)
        verbose: Print summary

    Returns:
        Dictionary with resnap statistics
    """
    count = 0
    total_error = 0.0

    for name, module in model.named_modules():
        if not isinstance(module, AnemllQATLinearV2):
            continue
        if module.lora_r <= 0 or module.lora_A is None:
            continue

        # Get current effective weight (Q * scales)
        if module._Q is not None:
            scales = module._compute_full_scales()
            W_base = module._Q * scales
        else:
            # No frozen Q - use fake_quant path (shouldn't happen in trained model)
            continue

        # Compute LoRA delta: LoRA_B @ LoRA_A * scaling
        lora_delta = (module.lora_B @ module.lora_A) * module.scaling

        # New effective weight
        W_eff = W_base + lora_delta.to(W_base.dtype)

        # Recompute scales from new effective weight
        # (Use existing scales - they're still valid, just re-quantize)
        scales_new = scales  # Keep existing scales

        # Normalize and re-quantize
        normalized = W_eff / scales_new
        indices_new = quantize_to_lut_indices(
            normalized,
            lut_size=module.lut.size(0),
            include_zero=module.config.lut_include_zero,
            lut=module.lut,  # Use actual LUT values
        )

        # Update Q buffer
        lut = module.lut.to(W_eff.dtype)
        Q_new = lut[indices_new]

        # Compute error
        W_reconstructed = Q_new * scales_new
        error = (W_eff - W_reconstructed).abs().mean() / W_eff.abs().mean()
        total_error += error.item()

        # Update buffers
        module._Q = Q_new
        module._indices = indices_new

        # Clear LoRA
        module.lora_r = 0
        module.lora_A = None
        module.lora_B = None
        module.lora_drop = None
        module.scaling = 0.0

        count += 1

        if verbose and count <= 3:
            print(f"  [resnap] {name}: rel_error={error.item():.6f}")

    stats = {
        'layers_resnapped': count,
        'avg_rel_error': total_error / max(count, 1),
    }

    if verbose:
        print(f"\n[Resnap with LoRA]")
        print(f"  Layers resnapped: {count}")
        print(f"  Avg rel error: {stats['avg_rel_error']:.6f}")
        print(f"  LoRA params cleared - model is now LoRA-free")

    return stats
