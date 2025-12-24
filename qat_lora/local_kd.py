"""
Local reconstruction loss for progressive layer-by-layer QAT.

This module provides a loss function that compares the output of a fake-quantized
MLP to a frozen full-precision copy of the same MLP. Key optimizations:
- Token sampling BEFORE computing fp target (memory efficient)
- Uses attention_mask to sample only valid tokens (no padding)
- Mixed normalized + unnormalized loss for scale preservation
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalMLPReconstructionLoss(nn.Module):
    """
    Computes local reconstruction loss for a single MLP.
    Uses frozen fp weights to compute target, compares to fake-quant output.

    Key efficiency fix: Sample tokens BEFORE computing fp target.
    Key stability fix: Mixed normalized + unnormalized loss for scale preservation.

    The loss is:
        L = α * MSE(RMSNorm(y_q), RMSNorm(y_fp)) + (1-α) * MSE(y_q, y_fp)

    where α (norm_weight) ≈ 0.8 by default.
    """

    def __init__(
        self,
        frozen_weights: Dict[str, torch.Tensor],
        rms_norm_eps: float = 1e-6,
        norm_weight: float = 0.8,
        max_loss: float = 10.0,
    ):
        """
        Args:
            frozen_weights: Dict with 'gate', 'up', 'down' weight tensors
            rms_norm_eps: Epsilon for RMSNorm computation
            norm_weight: Weight α for normalized loss component (default 0.8)
            max_loss: Maximum loss value for clamping (prevents inf/explosion)
        """
        super().__init__()
        # Store frozen fp weights (no grad) as buffers for proper device handling
        self.register_buffer('gate_w', frozen_weights['gate'])
        self.register_buffer('up_w', frozen_weights['up'])
        self.register_buffer('down_w', frozen_weights['down'])
        self.rms_norm_eps = rms_norm_eps
        self.norm_weight = norm_weight
        self.max_loss = max_loss

    def _rms_norm(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMSNorm for directional loss (without learnable scale)."""
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.rms_norm_eps)

    def _sample_valid_tokens(
        self,
        attention_mask: torch.Tensor,
        num_tokens: int,
    ) -> torch.Tensor:
        """
        Sample token indices from valid (non-padding) positions only.

        Args:
            attention_mask: Valid token mask [B, T] (1=valid, 0=pad)
            num_tokens: Number of tokens to sample

        Returns:
            Indices tensor [num_samples] for gathering from flattened [B*T, H]
        """
        # Flatten batch dimension for simplicity
        valid_mask = attention_mask.view(-1).bool()  # [B*T]
        valid_indices = torch.where(valid_mask)[0]   # [num_valid]

        if len(valid_indices) <= num_tokens:
            return valid_indices

        # Random sample from valid positions
        perm = torch.randperm(len(valid_indices), device=attention_mask.device)
        return valid_indices[perm[:num_tokens]]

    def forward(
        self,
        mlp_input: torch.Tensor,
        mlp_output_q: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        num_tokens: int = 128,
    ) -> torch.Tensor:
        """
        Compute local reconstruction loss.

        Args:
            mlp_input: Input to MLP [B, T, H]
            mlp_output_q: Fake-quantized MLP output [B, T, H]
            attention_mask: Valid token mask [B, T] (1=valid, 0=pad). If None,
                            all tokens are considered valid.
            num_tokens: Number of tokens to sample for loss (default 128)

        Returns:
            Scalar loss tensor
        """
        B, T, H = mlp_input.shape

        # Create default attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones(B, T, device=mlp_input.device)

        # EFFICIENCY FIX: Sample tokens BEFORE computing fp target
        # Use reshape() instead of view() to handle non-contiguous tensors from hooks
        if T * B > num_tokens:
            sample_idx = self._sample_valid_tokens(attention_mask, num_tokens)

            # Flatten and gather (reshape handles non-contiguous tensors)
            mlp_input_flat = mlp_input.reshape(B * T, H)
            mlp_output_q_flat = mlp_output_q.reshape(B * T, H)

            mlp_input_sub = mlp_input_flat[sample_idx]       # [num_tokens, H]
            mlp_output_q_sub = mlp_output_q_flat[sample_idx]
        else:
            mlp_input_sub = mlp_input.reshape(-1, H)
            mlp_output_q_sub = mlp_output_q.reshape(-1, H)

        # STABILITY: Cast to float32 for loss computation to avoid bf16 overflow
        mlp_input_sub = mlp_input_sub.float()
        mlp_output_q_sub = mlp_output_q_sub.float()

        # Compute fp target ONLY on subset (no grad)
        with torch.no_grad():
            # Qwen MLP: down(silu(gate(x)) * up(x))
            # Use float32 for stability
            gate_w = self.gate_w.float()
            up_w = self.up_w.float()
            down_w = self.down_w.float()

            gate_out = F.linear(mlp_input_sub, gate_w)
            up_out = F.linear(mlp_input_sub, up_w)
            hidden = F.silu(gate_out) * up_out
            mlp_output_fp_sub = F.linear(hidden, down_w)

        # STABILITY: Check for NaN/inf in inputs
        if not torch.isfinite(mlp_output_q_sub).all():
            # Return a small constant loss to allow training to continue
            return torch.tensor(self.max_loss, device=mlp_input.device, dtype=mlp_input.dtype)

        # MIXED LOSS for scale preservation
        # L = α * MSE(RMSNorm(y_q), RMSNorm(y_fp)) + (1-α) * MSE(y_q, y_fp)
        y_q_norm = self._rms_norm(mlp_output_q_sub)
        y_fp_norm = self._rms_norm(mlp_output_fp_sub)

        loss_norm = F.mse_loss(y_q_norm, y_fp_norm)
        loss_raw = F.mse_loss(mlp_output_q_sub, mlp_output_fp_sub)

        loss = self.norm_weight * loss_norm + (1 - self.norm_weight) * loss_raw

        # STABILITY: Clamp loss to prevent explosion and check for NaN/inf
        if not torch.isfinite(loss):
            return torch.tensor(self.max_loss, device=mlp_input.device, dtype=mlp_input.dtype)

        return torch.clamp(loss, max=self.max_loss)
