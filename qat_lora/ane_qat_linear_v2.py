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

    @property
    def lut_bits(self) -> int:
        return int(math.ceil(math.log2(self.lut_size)))


# =============================================================================
# LUT UTILITIES
# =============================================================================

def make_lut(
    lut_size: int,
    device: torch.device,
    dtype: torch.dtype,
    include_zero: bool = False,
) -> torch.Tensor:
    """Create monotonic LUT in [-1, 1]."""
    if lut_size < 2:
        raise ValueError("lut_size must be >= 2")
    if not include_zero or (lut_size % 2 == 1):
        return torch.linspace(-1.0, 1.0, steps=lut_size, device=device, dtype=dtype)

    # Even size with zero: non-uniform but includes 0
    neg = torch.linspace(-1.0, 0.0, steps=lut_size // 2 + 1, device=device, dtype=dtype)
    pos = torch.linspace(0.0, 1.0, steps=lut_size // 2, device=device, dtype=dtype)
    return torch.cat([neg[:-1], pos], dim=0)


def quantize_to_lut_indices(
    normalized: torch.Tensor,
    lut_size: int,
    include_zero: bool = False,
) -> torch.Tensor:
    """Map normalized values [-1, 1] to nearest LUT indices."""
    x = normalized.clamp(-1.0, 1.0)

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
# STE-FP16 UTILITIES
# =============================================================================

def ste_fp16(x: torch.Tensor) -> torch.Tensor:
    """Straight-through estimator for FP16 rounding.

    Forward: Rounds x to FP16 precision (matches ANE behavior).
    Backward: Identity gradient (passes through as if no rounding).

    This allows training with FP32 master weights while the forward pass
    matches what ANE will actually execute in FP16.

    Args:
        x: Input tensor (any dtype)

    Returns:
        Tensor with FP16-rounded values but same dtype as input.
        Gradients flow through unchanged.
    """
    if x.dtype == torch.float16:
        return x  # Already FP16, no rounding needed
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
    ):
        super().__init__()
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

        # LUT
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
        self.register_buffer("_indices", None)

        # Cached full weights for inference
        self.register_buffer("_cached_weight_q", None)

        # Enable/disable fake quantization (for backward compat)
        self.enable_fake_quant = True

        # Store lut_bits for conversion pipeline
        self.lut_bits = self.config.lut_bits

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
        self.use_batched_forward = False

        # V2 defaults to factored inference (rank-by-rank) for ANE compatibility
        # Set to False for faster PyTorch inference (single matmul)
        self.use_factored_inference = True

        # Flag: scales are already baked (magnitude in A, B is unit, g=1)
        # When True, _get_normalized_scales() returns raw params without re-normalizing
        self._scales_baked = False

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

        # Initialize scales from weight statistics
        self._init_scales_from_weight()

        if self.lora_r > 0:
            nn.init.normal_(self.lora_A, std=0.02)
            nn.init.zeros_(self.lora_B)

    @torch.no_grad()
    def _init_scales_from_weight(self):
        """Initialize scale parameters with unit-norm + magnitude decomposition.

        Uses SVD which gives orthonormal u and vh directly:
        - scale_A = u[:, :r] (unit-norm columns)
        - scale_B = vh[:r, :] (unit-norm rows)
        - rank_magnitude = s[:r] (singular values = magnitudes)
        """
        w = self.weight.float()

        # Compute per-weight scales (max-abs per output)
        # Simple approach: use abs(weight) as initial scale estimate
        scales_per_weight = w.abs().clamp(min=1e-8)

        # SVD: scales ≈ u @ diag(s) @ vh
        u, s, vh = torch.linalg.svd(scales_per_weight, full_matrices=False)
        r = self.scale_rank

        # u columns and vh rows are already unit-norm from SVD!
        self.scale_A.data = u[:, :r].to(self.weight.dtype)
        self.scale_B.data = vh[:r, :].to(self.weight.dtype)
        self.rank_magnitude.data = s[:r].to(self.weight.dtype)

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
        if getattr(self, '_scales_baked', False):
            return self.scale_A, self.scale_B, self.rank_magnitude

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
        # so S is guaranteed >= 0. No clamp needed.
        if getattr(self.config, "force_positive_scales", False):
            return S
        # Fallback clamp only when scales can go negative (use FP16-safe eps)
        eps = getattr(self.config, 'norm_eps', 1e-6)
        return S.clamp(min=eps)

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
        indices_cpu = quantize_to_lut_indices(
            normalized.cpu(),
            lut_size=self.lut.size(0),
            include_zero=self.config.lut_include_zero,
        )

        # Store indices on the same device as weights
        self._indices = indices_cpu.to(device)

        # Store Q = lut[indices] in [-1, 1]
        # Index on CPU then move (more compatible with MPS)
        self._Q = self.lut.cpu()[indices_cpu].to(device)

        # Freeze weight (we're training scales only)
        self.weight.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Rank-by-rank forward: y = Σₖ gₖ · (aₖ ⊙ (Q (bₖ ⊙ x)))

        No A @ B materialization in this path.
        """
        # Fast path for cached full weights (single matmul inference)
        # Skip if use_factored_inference=True (for ANE export testing)
        if self._cached_weight_q is not None and not self.use_factored_inference:
            w_q = self._cached_weight_q.to(x.dtype)
            bias = self.bias.to(x.dtype) if self.bias is not None else None
            return F.linear(x, w_q, bias)

        # Get normalized scales
        A_dir, B_dir, g = self._get_normalized_scales()

        # Get Q (frozen buffer or compute on-the-fly)
        if self._Q is not None:
            Q = self._Q
        else:
            # Fallback: compute Q on the fly (for backward compat / testing)
            scales = self._compute_full_scales()
            normalized = self.weight / scales
            indices = quantize_to_lut_indices(
                normalized, self.lut.size(0), self.config.lut_include_zero
            )
            Q = self.lut[indices]

        Q = Q.to(x.dtype)

        # Choose forward implementation
        if self.use_batched_forward:
            y = self._forward_batched(x, A_dir, B_dir, g, Q)
        else:
            y = self._forward_loop(x, A_dir, B_dir, g, Q)

        # Add bias
        if self.bias is not None:
            y = y + self.bias.to(x.dtype)

        # Add LoRA if enabled
        if self.lora_r > 0:
            x_d = self.lora_drop(x) if self.lora_drop is not None else x
            lora_A = self.lora_A.to(x.dtype)
            lora_B = self.lora_B.to(x.dtype)
            y = y + (x_d @ lora_A.t() @ lora_B.t()) * self.scaling

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

        Args:
            recompute_indices: If True, recompute indices in FP16 precision.
                              If False, keep existing indices but convert Q to FP16.
        """
        device = self.weight.device
        fp16 = torch.float16

        # 1. Recompute LUT in FP16
        lut_fp16 = make_lut(
            self.config.lut_size,
            device=device,
            dtype=fp16,
            include_zero=self.config.lut_include_zero,
        )

        if recompute_indices:
            # 2. Compute scales in FP16
            A_dir, B_dir, g = self._get_normalized_scales()
            A_dir = A_dir.to(fp16)
            B_dir = B_dir.to(fp16)
            g = g.to(fp16)

            # Compute full scales in FP16
            A_scaled = A_dir * g
            scales_fp16 = (A_scaled @ B_dir)
            if getattr(self.config, "force_positive_scales", False):
                scales_fp16 = scales_fp16  # Already >= 0
            else:
                scales_fp16 = scales_fp16.clamp(min=1e-8)

            # 3. Recompute indices in FP16
            weight_fp16 = self.weight.to(fp16)
            normalized = weight_fp16 / scales_fp16

            indices_fp16 = quantize_to_lut_indices(
                normalized,
                lut_size=self.config.lut_size,
                include_zero=self.config.lut_include_zero,
            )

            # Store indices
            self._indices = indices_fp16

            # Bake scales into scale_A, scale_B
            self.scale_A.data = A_scaled.to(fp16)
            self.scale_B.data = B_dir.to(fp16)
            self.rank_magnitude.data = torch.ones_like(g).to(fp16)

            # Mark scales as baked (skip normalization in forward)
            self._scales_baked = True
        else:
            # Just convert existing indices
            indices_fp16 = self._indices

        # 4. Compute Q = lut[indices] in FP16
        self._Q = lut_fp16[indices_fp16]

        # Update LUT buffer
        if hasattr(self, 'lut') and isinstance(self.lut, torch.Tensor):
            if isinstance(self.lut, nn.Parameter):
                self.lut.data = lut_fp16
            else:
                self.lut = lut_fp16

        # Convert weight and bias to FP16
        self.weight.data = self.weight.data.to(fp16)
        if self.bias is not None:
            self.bias.data = self.bias.data.to(fp16)

        # Clear cached weight (force factored forward)
        self._cached_weight_q = None

    @torch.no_grad()
    def convert_to_fp16(self):
        """Convert all tensors to FP16 for FP16 training pipeline.

        Use this BEFORE freeze_Q() to ensure indices are computed in FP16.
        This ensures no precision mismatch between training and ANE inference.
        """
        fp16 = torch.float16
        device = self.weight.device

        # Convert weight and bias
        self.weight.data = self.weight.data.to(fp16)
        if self.bias is not None:
            self.bias.data = self.bias.data.to(fp16)

        # Convert scales
        self.scale_A.data = self.scale_A.data.to(fp16)
        self.scale_B.data = self.scale_B.data.to(fp16)
        self.rank_magnitude.data = self.rank_magnitude.data.to(fp16)

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

        # Clear cached weight
        self._cached_weight_q = None

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        config: Optional[AnemllQuantConfigV2] = None,
    ) -> "AnemllQATLinearV2":
        """Create AnemllQATLinearV2 from existing nn.Linear."""
        config = config or AnemllQuantConfigV2()

        layer = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            config=config,
        )

        # Copy weights
        layer.weight.data = linear.weight.data.clone()
        if linear.bias is not None:
            layer.bias.data = linear.bias.data.clone()

        # Re-initialize scales from the actual weights
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
) -> int:
    """Replace MLP and optionally attention linears with AnemllQATLinearV2."""
    import re

    mlp_pattern = re.compile(r'\.mlp\.(gate_proj|up_proj|down_proj)$')
    attn_pattern = re.compile(r'\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$')
    lm_head_pattern = re.compile(r'^lm_head$')

    if attn_config is None:
        attn_config = mlp_config

    replacements = []

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

        new_module = AnemllQATLinearV2.from_linear(module, config=cfg)

        parts = name.rsplit('.', 1)
        if len(parts) == 2:
            parent_name, attr = parts
            parent = dict(model.named_modules())[parent_name]
        else:
            parent = model
            attr = name

        replacements.append((parent, attr, new_module, name))

    for parent, attr, new_module, name in replacements:
        setattr(parent, attr, new_module)
        if verbose:
            print(f'  [replaced] {name}')

    if verbose:
        print(f'\nReplaced {len(replacements)} layers with V2')

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


def snap_model_for_ane_v2(
    model: nn.Module,
    recompute_indices: bool = True,
    verbose: bool = True,
) -> int:
    """Snap all V2 layers for ANE export in FP16 precision.

    This converts the model to FP16 and recomputes quantization indices
    to match ANE's FP16 precision, avoiding BF16/FP16 mismatch issues.

    Args:
        model: The model containing V2 layers
        recompute_indices: If True, recompute indices in FP16 (recommended)
        verbose: Print diagnostic information

    Returns:
        Number of layers snapped
    """
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, AnemllQATLinearV2):
            module.snap_for_ane(recompute_indices=recompute_indices)
            count += 1
            if verbose and count <= 3:
                print(f"  [snap_ane] {name}")

    if verbose:
        print(f"\n[ANE FP16 Snap]")
        print(f"  Snapped {count} V2 layers to FP16")
        print(f"  Recomputed indices: {recompute_indices}")
        print(f"  All weights, scales, LUT, Q now in FP16")

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
) -> dict:
    """Load a V2 checkpoint with proper handling of _Q and _indices buffers.

    PyTorch's load_state_dict doesn't load tensors into None buffers.
    This function pre-registers the buffers with correct shapes before loading.

    Args:
        model: Model with V2 layers (after replace_linear_with_anemll_v2)
        checkpoint_path: Path to the checkpoint file
        device: Device to load to (default: CPU)
        verbose: Print loading statistics

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
            indices_loaded += 1

    # Now load the rest of the state dict
    result = model.load_state_dict(state_dict, strict=False)

    # Move to device
    model.to(device)

    stats = {
        'v2_layers': len(v2_layers),
        'q_loaded': q_loaded,
        'indices_loaded': indices_loaded,
        'missing_keys': len(result.missing_keys),
        'unexpected_keys': len(result.unexpected_keys),
    }

    if verbose:
        print(f"\n[V2 Checkpoint Loaded]")
        print(f"  V2 layers: {stats['v2_layers']}")
        print(f"  _Q loaded: {stats['q_loaded']}")
        print(f"  _indices loaded: {stats['indices_loaded']}")
        print(f"  Missing keys: {stats['missing_keys']}")
        # Unexpected should be 0 now since we pre-loaded _Q and _indices
        remaining_unexpected = stats['unexpected_keys'] - (q_loaded + indices_loaded)
        print(f"  Unexpected keys (other): {max(0, remaining_unexpected)}")

        if stats['q_loaded'] == stats['v2_layers']:
            print(f"  Ready for inference (no freeze_Q needed)")
        else:
            print(f"  Call freeze_Q_all() before inference")

    return stats
