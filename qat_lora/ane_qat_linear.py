"""
Anemll-style QATLinear with learnable per-weight scales (A @ B) and LUT.

Quantization formula:
    W_q[o, i] = LUT[idx[o, i]] * scales[o, i]

Key features:
- Scales are per-weight [out_features, in_features] for direct inference
- scales[o, i] = (A @ B)[o, i]  with low-rank A:[out, rank], B:[rank, in]
- Initialization: compute per-group, expand to per-weight, then SVD
- LUT is explicit and optionally learnable

Scale dimensions:
- Full scales: [out_features, in_features] - one scale per weight
- Low-rank:    A[out_features, rank] @ B[rank, in_features] - compressed

Example (out=512, in=1024, rank=4):
- Full scales: 512 × 1024 = 524,288 params
- Low-rank:    512 × 4 + 4 × 1024 = 6,144 params (85x compression)

During inference: direct element-wise scale * weight multiplication.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class AnemllQuantConfig:
    """Configuration for Anemll-style groupwise LUT quantization."""

    lut_size: int = 16  # Number of LUT entries (4-bit = 16, 2-bit = 4)
    group_size: int = 128  # Group size along input dimension
    scale_rank: int = 4  # Rank for A @ B scale approximation (0 = full scales)
    lut_include_zero: bool = False  # Whether LUT includes 0
    learnable_lut: bool = False  # Whether LUT values are trainable

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
    include_zero: bool,
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


def _round_ste(x: torch.Tensor) -> torch.Tensor:
    """Straight-through estimator for rounding."""
    return x + (torch.round(x) - x).detach()


# =============================================================================
# ANEMLL QAT LINEAR LAYER
# =============================================================================

class AnemllQATLinear(nn.Module):
    """
    Linear layer with Anemll-style groupwise LUT quantization.

    Learnable parameters:
    - weight: Full-precision weights (for gradient computation)
    - scale_A: [out_features, rank] scale factor
    - scale_B: [rank, num_groups] scale factor
    - lut: [lut_size] LUT values (optional)
    - lora_A, lora_B: LoRA adapters (optional)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        config: Optional[AnemllQuantConfig] = None,
        # LoRA params
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config or AnemllQuantConfig()

        # Compute group dimensions
        self.group_size = self.config.group_size
        self.pad = (-in_features) % self.group_size
        self.padded_in = in_features + self.pad
        self.num_groups = self.padded_in // self.group_size

        # Scale rank (0 means full scales, no low-rank)
        # For per-weight scales: A:[out, rank] @ B:[rank, padded_in]
        self.max_rank = min(out_features, self.padded_in)
        self.scale_rank = min(self.config.scale_rank, self.max_rank) if self.config.scale_rank > 0 else 0
        self.use_low_rank = self.scale_rank > 0 and self.scale_rank < self.max_rank

        # Base weights
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None

        # Scale parameters - per-weight: [out_features, padded_in_features]
        if self.use_low_rank:
            # Low-rank: scales = A @ B = [out, rank] @ [rank, padded_in]
            self.scale_A = nn.Parameter(torch.empty(out_features, self.scale_rank))
            self.scale_B = nn.Parameter(torch.empty(self.scale_rank, self.padded_in))
            self.register_buffer("_full_scales", None)  # Not used
        else:
            # Full per-weight scales
            self.register_parameter("scale_A", None)
            self.register_parameter("scale_B", None)
            self.full_scales = nn.Parameter(torch.empty(out_features, self.padded_in))

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

        # Enable/disable fake quantization
        self.enable_fake_quant = True

        # LoRA
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
        """Initialize scale parameters from weight statistics.

        Process:
        1. Compute per-group max-abs scales: [out, num_groups]
        2. Expand to per-weight: [out, padded_in] by repeating group_size times
        3. Apply SVD on full per-weight scale matrix
        """
        w = self.weight.float()

        # Pad if needed
        if self.pad > 0:
            w = F.pad(w, (0, self.pad))

        # Step 1: Compute per-group max-abs scales [out, num_groups]
        grouped = w.view(self.out_features, self.num_groups, self.group_size)
        scales_per_group = grouped.abs().amax(dim=2).clamp(min=1e-8)

        # Step 2: Expand to per-weight [out, padded_in]
        # Each group scale is repeated group_size times
        scales_per_weight = scales_per_group.repeat_interleave(self.group_size, dim=1)
        # scales_per_weight is now [out_features, padded_in]

        if self.use_low_rank:
            # Step 3: SVD on full per-weight scales
            u, s, vh = torch.linalg.svd(scales_per_weight, full_matrices=False)
            r = self.scale_rank
            self.scale_A.data = (u[:, :r] * s[:r]).to(self.weight.dtype)
            self.scale_B.data = vh[:r, :].to(self.weight.dtype)
        else:
            self.full_scales.data = scales_per_weight.to(self.weight.dtype)

    def get_scales(self) -> torch.Tensor:
        """Get the per-weight scale matrix [out_features, padded_in]."""
        if self.use_low_rank:
            return (self.scale_A @ self.scale_B).clamp(min=1e-8)
        else:
            return self.full_scales.clamp(min=1e-8)

    def _quant_rows(
        self,
        w_rows: torch.Tensor,
        scales_rows: torch.Tensor,
        lut: torch.Tensor,
    ) -> torch.Tensor:
        """Quantize a chunk of rows. Helper for fake_quant_weight.

        With per-weight scales, this is now element-wise:
        - w_rows: [num_rows, padded_in]
        - scales_rows: [num_rows, padded_in]
        """
        # --- Step A: Normalize to [-1, 1] using per-weight scales ---
        normalized = w_rows / scales_rows

        # --- Step B: Compute LUT indices ---
        lut_size = lut.size(0)
        if not self.config.lut_include_zero or (lut_size % 2 == 1):
            # Uniform LUT - memory-efficient index computation
            step = 2.0 / (lut_size - 1)
            clamped = normalized.clamp(-1.0, 1.0)
            indices_float = (clamped + 1.0) / step
            indices = _round_ste(indices_float).long().clamp(0, lut_size - 1)
        else:
            # Non-uniform LUT with zero
            indices = quantize_to_lut_indices(
                normalized.clamp(-1.0, 1.0),
                lut_size=lut_size,
                include_zero=True,
            )

        # --- Step C: Dequantize via LUT lookup ---
        # Direct element-wise: lut_value * scale
        dequant = lut[indices] * scales_rows
        return dequant

    def fake_quant_weight(self) -> torch.Tensor:
        """Apply fake quantization to weights with STE for gradient flow.

        Uses Straight-Through Estimator (STE):
        - Forward: returns quantized-dequantized weights
        - Backward: gradients flow directly to original weights
        """
        # --- Step 1: Prepare weights ---
        w = self.weight.float()

        # --- Step 2: Pad if needed ---
        if self.pad > 0:
            w = F.pad(w, (0, self.pad))

        # --- Step 3: Get per-weight scales [out, padded_in] ---
        scales = self.get_scales()

        # --- Step 4: Move LUT to same device/dtype ---
        lut = self.lut.to(device=w.device, dtype=w.dtype)

        # --- Step 5: Quantize-Dequantize ---
        # Process in chunks for large tensors (e.g., lm_head with 150k+ rows)
        chunk_size = 1024  # rows per chunk
        if w.numel() > 50_000_000:
            # Chunked processing for large tensors
            dequant_chunks = []
            for start in range(0, self.out_features, chunk_size):
                end = min(start + chunk_size, self.out_features)
                chunk = self._quant_rows(w[start:end], scales[start:end], lut)
                dequant_chunks.append(chunk)
            dequant = torch.cat(dequant_chunks, dim=0)
        else:
            # Process whole tensor at once for smaller tensors
            dequant = self._quant_rows(w, scales, lut)

        # --- Step 6: Remove padding ---
        if self.pad > 0:
            dequant = dequant[:, :self.in_features]

        # --- Step 7: Apply STE for gradient flow ---
        # When training weights: use STE so gradients flow to weights
        # When training scales only: let gradients flow through dequant to scales
        if self.weight.requires_grad:
            # STE: forward=dequant, backward=d/dw=1
            w_float = self.weight.float()
            w_q = w_float + (dequant - w_float).detach()
        else:
            # Scales training: gradients flow through dequant -> scales
            w_q = dequant

        return w_q.to(self.weight.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- Step 1: Get quantized weights ---
        # Use cached weights if available (inference mode)
        if hasattr(self, '_cached_weight_q') and self._cached_weight_q is not None:
            w_q = self._cached_weight_q
        elif self.enable_fake_quant:
            w_q = self.fake_quant_weight()
        else:
            w_q = self.weight

        # --- Step 2: Cast to input dtype ---
        w_q = w_q.to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None

        # --- Step 3: Linear transform ---
        y = F.linear(x, w_q, bias)

        # --- Step 4: Add LoRA residual (if enabled) ---
        if self.lora_r > 0:
            x_d = self.lora_drop(x) if self.lora_drop is not None else x
            lora_A = self.lora_A.to(x.dtype)
            lora_B = self.lora_B.to(x.dtype)
            y = y + (x_d @ lora_A.t() @ lora_B.t()) * self.scaling

        return y

    @torch.no_grad()
    def freeze_for_inference(self):
        """Precompute and cache quantized weights for fast inference.

        This precomputes: quantized_weight = LUT[indices] * (scale_A @ scale_B)
        and caches it to avoid recomputation on every forward pass.

        Call this before inference to speed up per-token generation.
        """
        if not self.enable_fake_quant:
            self._cached_weight_q = None
            return

        # Compute quantized weights once
        w_q = self.fake_quant_weight()

        # Store as buffer (not parameter) - no gradients needed
        self.register_buffer('_cached_weight_q', w_q.detach().clone())

        # Optionally delete original weight to save memory (commented for safety)
        # del self.weight

    def unfreeze_for_training(self):
        """Clear cached weights and re-enable training mode.

        Call this before resuming training after inference.
        """
        if hasattr(self, '_cached_weight_q'):
            del self._cached_weight_q
        self._cached_weight_q = None

    def enable_lora(self, r: int, alpha: float, dropout: float = 0.0):
        """Enable LoRA for this layer."""
        if self.lora_r > 0:
            return

        self.lora_r = int(r)
        self.lora_alpha = float(alpha)
        self.lora_dropout = float(dropout)

        self.lora_A = nn.Parameter(
            torch.zeros(self.lora_r, self.in_features, device=self.weight.device, dtype=self.weight.dtype)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(self.out_features, self.lora_r, device=self.weight.device, dtype=self.weight.dtype)
        )
        self.scaling = self.lora_alpha / self.lora_r
        self.lora_drop = nn.Dropout(p=self.lora_dropout)

        nn.init.normal_(self.lora_A, std=0.02)
        nn.init.zeros_(self.lora_B)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        config: Optional[AnemllQuantConfig] = None,
    ) -> "AnemllQATLinear":
        """Create AnemllQATLinear from existing nn.Linear."""
        config = config or AnemllQuantConfig()

        new = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            config=config,
        )

        with torch.no_grad():
            new.weight.copy_(linear.weight)
            if linear.bias is not None:
                new.bias.copy_(linear.bias)

        # Re-initialize scales from the actual weights
        new._init_scales_from_weight()

        # Move to same device as original linear
        new = new.to(device=linear.weight.device, dtype=linear.weight.dtype)

        return new

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"groups={self.num_groups}, rank={self.scale_rank}, "
            f"lut_size={self.config.lut_size}, lora_r={self.lora_r}"
        )

    @torch.no_grad()
    def compute_indices(self) -> torch.Tensor:
        """Compute quantization indices for current weights.

        Returns:
            indices: [out_features, in_features] LUT indices (int8 or int16)
        """
        w = self.weight.float()

        # Pad if needed
        if self.pad > 0:
            w = F.pad(w, (0, self.pad))

        # Get per-weight scales
        scales = self.get_scales()

        # Normalize to [-1, 1]
        normalized = w / scales

        # Compute indices
        lut_size = self.lut.size(0)
        if not self.config.lut_include_zero or (lut_size % 2 == 1):
            step = 2.0 / (lut_size - 1)
            clamped = normalized.clamp(-1.0, 1.0)
            indices = torch.round((clamped + 1.0) / step).long().clamp(0, lut_size - 1)
        else:
            indices = quantize_to_lut_indices(
                normalized.clamp(-1.0, 1.0),
                lut_size=lut_size,
                include_zero=True,
            )

        # Remove padding
        if self.pad > 0:
            indices = indices[:, :self.in_features]

        # Use smallest dtype that fits
        if lut_size <= 256:
            indices = indices.to(torch.uint8)
        else:
            indices = indices.to(torch.int16)

        return indices

    @torch.no_grad()
    def snap_weights_to_quantized(self, store_lut_values: bool = True) -> torch.Tensor:
        """Snap weights to their quantized values.

        Two modes:
        - store_lut_values=True: weight = LUT[idx] (normalized, in [-1,1])
          At inference: output = input @ (weight * scale).T
        - store_lut_values=False: weight = LUT[idx] * scale (full dequant)
          At inference: output = input @ weight.T (scales already applied)

        Args:
            store_lut_values: If True, store LUT[idx] as weights (keeps scales separate).
                              If False, store LUT[idx] * scale as weights.

        Returns:
            indices: [out_features, in_features] the indices used
        """
        w = self.weight.float()
        orig_dtype = self.weight.dtype

        # Pad if needed
        if self.pad > 0:
            w = F.pad(w, (0, self.pad))

        # Get per-weight scales [out, padded_in]
        scales = self.get_scales()

        # Normalize to [-1, 1]
        normalized = w / scales

        # Compute indices
        lut_size = self.lut.size(0)
        if not self.config.lut_include_zero or (lut_size % 2 == 1):
            step = 2.0 / (lut_size - 1)
            clamped = normalized.clamp(-1.0, 1.0)
            indices = torch.round((clamped + 1.0) / step).long().clamp(0, lut_size - 1)
        else:
            indices = quantize_to_lut_indices(
                normalized.clamp(-1.0, 1.0),
                lut_size=lut_size,
                include_zero=True,
            )

        # Get LUT values
        lut = self.lut.to(device=w.device, dtype=w.dtype)
        lut_values = lut[indices]  # [out, padded_in] in [-1, 1]

        if store_lut_values:
            # Store LUT[idx] as weight (normalized values)
            w_quantized = lut_values
        else:
            # Store LUT[idx] * scale as weight (full dequant)
            w_quantized = lut_values * scales

        # Remove padding
        if self.pad > 0:
            w_quantized = w_quantized[:, :self.in_features]
            indices = indices[:, :self.in_features]

        # Update weight parameter
        self.weight.data.copy_(w_quantized.to(orig_dtype))

        # Return indices (compact dtype)
        if lut_size <= 256:
            indices = indices.to(torch.uint8)
        else:
            indices = indices.to(torch.int16)

        return indices

    @torch.no_grad()
    def get_quantized_representation(self) -> dict:
        """Get the full quantized representation for export.

        Returns dict with:
            - indices: [out, in] uint8/int16 LUT indices
            - quantized_weights: [out, in] LUT[idx] values (in [-1, 1])
            - scales: [out, padded_in] full scales or {'scale_A', 'scale_B'} for low-rank
            - lut: [lut_size] LUT values
            - bias: [out] or None
            - config info
        """
        w = self.weight.float()

        # Pad if needed
        if self.pad > 0:
            w = F.pad(w, (0, self.pad))

        # Get scales
        scales = self.get_scales()

        # Normalize and compute indices
        normalized = w / scales
        lut_size = self.lut.size(0)
        if not self.config.lut_include_zero or (lut_size % 2 == 1):
            step = 2.0 / (lut_size - 1)
            clamped = normalized.clamp(-1.0, 1.0)
            indices = torch.round((clamped + 1.0) / step).long().clamp(0, lut_size - 1)
        else:
            indices = quantize_to_lut_indices(
                normalized.clamp(-1.0, 1.0),
                lut_size=lut_size,
                include_zero=True,
            )

        # Get quantized weights = LUT[idx]
        lut = self.lut.to(device=w.device, dtype=w.dtype)
        quantized_weights = lut[indices]

        # Remove padding
        if self.pad > 0:
            indices = indices[:, :self.in_features]
            quantized_weights = quantized_weights[:, :self.in_features]

        # Compact index dtype
        if lut_size <= 256:
            indices = indices.to(torch.uint8)
        else:
            indices = indices.to(torch.int16)

        # Scales for export
        if self.use_low_rank:
            scales_export = {
                'scale_A': self.scale_A.data.clone(),
                'scale_B': self.scale_B.data.clone(),
            }
        else:
            scales_export = self.full_scales.data.clone()

        return {
            'indices': indices,
            'quantized_weights': quantized_weights.to(self.weight.dtype),  # LUT[idx]
            'scales': scales_export,
            'lut': self.lut.clone(),
            'bias': self.bias.data.clone() if self.bias is not None else None,
            'in_features': self.in_features,
            'out_features': self.out_features,
            'group_size': self.group_size,
            'scale_rank': self.scale_rank,
            'lut_size': self.config.lut_size,
        }


# =============================================================================
# MODEL REPLACEMENT UTILITY
# =============================================================================

def replace_linear_with_anemll(
    model: nn.Module,
    mlp_config: AnemllQuantConfig,
    attn_config: Optional[AnemllQuantConfig] = None,
    quantize_attn: bool = True,
    quantize_lm_head: bool = False,
    verbose: bool = True,
) -> int:
    """Replace MLP and optionally attention linears with AnemllQATLinear.

    Args:
        model: The model to modify (in-place)
        mlp_config: Config for MLP layers (gate_proj, up_proj, down_proj)
        attn_config: Config for attention layers (q/k/v/o_proj). If None, uses mlp_config
        quantize_attn: Whether to quantize attention layers
        quantize_lm_head: Whether to quantize lm_head (usually False)
        verbose: Print replacement info

    Returns:
        Number of layers replaced
    """
    import re

    # --- Step 1: Define patterns ---
    mlp_pattern = re.compile(r'\.mlp\.(gate_proj|up_proj|down_proj)$')
    attn_pattern = re.compile(r'\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$')
    lm_head_pattern = re.compile(r'^lm_head$')

    if attn_config is None:
        attn_config = mlp_config

    # --- Step 2: Find layers to replace ---
    replacements = []

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if isinstance(module, AnemllQATLinear):
            continue

        # Check pattern
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

        # --- Step 3: Create replacement module ---
        new_module = AnemllQATLinear.from_linear(module, config=cfg)

        # --- Step 4: Find parent module ---
        parts = name.rsplit('.', 1)
        if len(parts) == 2:
            parent_name, attr = parts
            parent = dict(model.named_modules())[parent_name]
        else:
            parent = model
            attr = name

        replacements.append((parent, attr, new_module, name))

    # --- Step 5: Apply replacements ---
    for parent, attr, new_module, name in replacements:
        setattr(parent, attr, new_module)
        if verbose:
            print(f'  [replaced] {name}')

    if verbose:
        print(f'\nReplaced {len(replacements)} layers')

    return len(replacements)


def freeze_model_for_inference(model: nn.Module, verbose: bool = False) -> int:
    """Freeze all AnemllQATLinear layers for fast inference.

    This precomputes quantized weights for all layers, caching them
    to avoid recomputation on every forward pass.

    Args:
        model: Model containing AnemllQATLinear layers
        verbose: Print progress info

    Returns:
        Number of layers frozen
    """
    count = 0
    for name, module in model.named_modules():
        if type(module).__name__ == 'AnemllQATLinear':
            module.freeze_for_inference()
            count += 1
            if verbose:
                print(f'  [frozen] {name}')

    if verbose:
        print(f'\nFrozen {count} layers for inference')

    return count


def unfreeze_model_for_training(model: nn.Module, verbose: bool = False) -> int:
    """Unfreeze all AnemllQATLinear layers for training.

    This clears cached quantized weights, re-enabling dynamic
    fake quantization during training.

    Args:
        model: Model containing AnemllQATLinear layers
        verbose: Print progress info

    Returns:
        Number of layers unfrozen
    """
    count = 0
    for name, module in model.named_modules():
        if type(module).__name__ == 'AnemllQATLinear':
            module.unfreeze_for_training()
            count += 1
            if verbose:
                print(f'  [unfrozen] {name}')

    if verbose:
        print(f'\nUnfrozen {count} layers for training')

    return count


@torch.no_grad()
def compute_all_indices(model: nn.Module, verbose: bool = False) -> dict:
    """Compute quantization indices for all AnemllQATLinear layers.

    Args:
        model: Model containing AnemllQATLinear layers
        verbose: Print progress info

    Returns:
        Dict mapping layer names to indices tensors
    """
    indices_dict = {}
    count = 0

    for name, module in model.named_modules():
        if type(module).__name__ == 'AnemllQATLinear':
            indices = module.compute_indices()
            indices_dict[name] = indices
            count += 1
            if verbose:
                print(f'  [indices] {name}: {indices.shape}, dtype={indices.dtype}')

    if verbose:
        # Compute total size
        total_bytes = sum(idx.numel() * idx.element_size() for idx in indices_dict.values())
        print(f'\nComputed indices for {count} layers ({total_bytes / 1024 / 1024:.1f} MB)')

    return indices_dict


@torch.no_grad()
def snap_all_weights(
    model: nn.Module,
    store_lut_values: bool = True,
    verbose: bool = False,
) -> dict:
    """Snap all weights to their quantized values.

    Two modes:
    - store_lut_values=True: weight = LUT[idx] (normalized, in [-1,1])
      At inference: output = input @ (weight * scale).T
    - store_lut_values=False: weight = LUT[idx] * scale (full dequant)
      At inference: output = input @ weight.T (scales already applied)

    Args:
        model: Model containing AnemllQATLinear layers
        store_lut_values: If True, store LUT[idx] as weights (keeps scales separate)
        verbose: Print progress info

    Returns:
        Dict mapping layer names to indices tensors
    """
    indices_dict = {}
    count = 0
    total_error = 0.0

    for name, module in model.named_modules():
        if type(module).__name__ == 'AnemllQATLinear':
            # Compute error before snap
            w_before = module.weight.data.clone()
            indices = module.snap_weights_to_quantized(store_lut_values=store_lut_values)
            w_after = module.weight.data

            # Compute relative error
            error = (w_before - w_after).abs().mean() / w_before.abs().mean()
            total_error += error.item()

            indices_dict[name] = indices
            count += 1
            if verbose:
                w_range = f"[{w_after.min().item():.3f}, {w_after.max().item():.3f}]"
                print(f'  [snapped] {name}: rel_error={error.item():.6f}, range={w_range}')

    if verbose and count > 0:
        avg_error = total_error / count
        total_bytes = sum(idx.numel() * idx.element_size() for idx in indices_dict.values())
        mode = "LUT[idx]" if store_lut_values else "LUT[idx]*scale"
        print(f'\nSnapped {count} layers to {mode} (avg rel_error={avg_error:.6f}, indices={total_bytes / 1024 / 1024:.1f} MB)')

    return indices_dict


@torch.no_grad()
def export_quantized_model(model: nn.Module, verbose: bool = False) -> dict:
    """Export full quantized representation for all layers.

    Each layer dict contains:
    - indices: [out, in] uint8/int16 LUT indices
    - quantized_weights: [out, in] LUT[idx] values (in [-1, 1])
    - scales: full scales or {'scale_A', 'scale_B'} for low-rank
    - lut: [lut_size] LUT values
    - bias, in_features, out_features, group_size, scale_rank, lut_size

    This is useful for external tools like ANEMLL model converters.

    Args:
        model: Model containing AnemllQATLinear layers
        verbose: Print progress info

    Returns:
        Dict mapping layer names to quantized representation dicts
    """
    export_dict = {}
    count = 0
    total_idx_bytes = 0
    total_qw_bytes = 0
    total_scale_bytes = 0

    for name, module in model.named_modules():
        if type(module).__name__ == 'AnemllQATLinear':
            rep = module.get_quantized_representation()
            export_dict[name] = rep
            count += 1

            idx_size = rep['indices'].numel() * rep['indices'].element_size()
            qw_size = rep['quantized_weights'].numel() * rep['quantized_weights'].element_size()
            if isinstance(rep['scales'], dict):
                scale_size = (rep['scales']['scale_A'].numel() * rep['scales']['scale_A'].element_size() +
                              rep['scales']['scale_B'].numel() * rep['scales']['scale_B'].element_size())
            else:
                scale_size = rep['scales'].numel() * rep['scales'].element_size()

            total_idx_bytes += idx_size
            total_qw_bytes += qw_size
            total_scale_bytes += scale_size

            if verbose:
                qw_range = f"[{rep['quantized_weights'].min().item():.3f}, {rep['quantized_weights'].max().item():.3f}]"
                print(f'  [export] {name}: idx={idx_size/1024:.1f}KB, qw={qw_size/1024:.1f}KB {qw_range}, scales={scale_size/1024:.1f}KB')

    if verbose:
        print(f'\nExported {count} layers:')
        print(f'  indices:           {total_idx_bytes / 1024 / 1024:.2f} MB')
        print(f'  quantized_weights: {total_qw_bytes / 1024 / 1024:.2f} MB (LUT[idx])')
        print(f'  scales:            {total_scale_bytes / 1024 / 1024:.2f} MB')

    return export_dict
