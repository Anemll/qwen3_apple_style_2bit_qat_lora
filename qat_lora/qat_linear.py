"""
QATLinear: nn.Linear drop-in with Apple-style 2-bit weight fake-quantization,
plus optional LoRA.

Key principles:
- The *stored* weight is full precision (bf16/fp16/fp32), updated by the optimizer.
- The *forward* uses a fake-quantized view of that weight (4 levels), as Apple describes.
- The quantization scale uses a learnable parameter f, initialized carefully.
- For recovery, LoRA can be added while freezing the base weights.

This matches Apple's high-level recipe:
- QAT to train a model that tolerates int2 weight quantization
- freeze base weights
- train low-rank adapters to recover quality lost from compression artifacts
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .quantizer import QATQuantConfig, fake_quant_weight_nbit


class QATLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        qc: QATQuantConfig | None = None,
        # LoRA params (disabled by default)
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        qc = qc or QATQuantConfig(n_bits=2)

        # Persist bitwidth in state_dict so checkpoints cannot silently change bitwidth at inference time.
        self.register_buffer("_qat_nbits", torch.tensor(int(qc.n_bits), dtype=torch.int16), persistent=True)
        # z is a constant in this repo (0.5), but keep the config for scale/init routines outside this module.

        # Base (full-precision) weights
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None

        # Learnable scaling factor f (Apple)
        # We parameterize f with softplus to enforce positivity.
        self._f_param = nn.Parameter(torch.tensor(0.0))  # softplus(0)=0.693...

        # LoRA
        self.lora_r = int(lora_r)
        self.lora_alpha = float(lora_alpha)
        self.lora_dropout = float(lora_dropout)

        if self.lora_r > 0:
            # LoRA uses:  W + (alpha/r) * (B @ A)
            # Implemented as: x @ A^T @ B^T
            # Keep adapter params on the same device/dtype as base weights to avoid dtype mismatches.
            self.lora_A = nn.Parameter(
                torch.zeros(self.lora_r, in_features, device=self.weight.device, dtype=self.weight.dtype)
            )
            self.lora_B = nn.Parameter(
                torch.zeros(out_features, self.lora_r, device=self.weight.device, dtype=self.weight.dtype)
            )
            self.scaling = self.lora_alpha / self.lora_r
            self.lora_drop = nn.Dropout(p=self.lora_dropout)
        else:
            self.lora_A = None
            self.lora_B = None
            self.scaling = 0.0
            self.lora_drop = None

        self.reset_parameters()

    @property
    def n_bits(self) -> int:
        return int(self._qat_nbits.item())

    def reset_parameters(self):
        # Match nn.Linear default init approximately
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

        # LoRA init: A ~ N(0, 0.02), B = 0 is common (so adapter starts as no-op)
        if self.lora_r > 0:
            nn.init.normal_(self.lora_A, std=0.02)
            nn.init.zeros_(self.lora_B)

    def f(self) -> torch.Tensor:
        # softplus ensures positivity and smooth gradients
        return F.softplus(self._f_param) + 1e-8

    def set_f_init(self, f_init: float):
        """
        Initialize f to a chosen positive value.

        We store f via inverse-softplus so that softplus(_f_param) ~= f_init
        """
        y = torch.tensor(float(f_init)).clamp(min=1e-8)
        self._f_param.data = torch.log(torch.exp(y) - 1.0)

    def enable_lora(self, r: int, alpha: float, dropout: float):
        """
        Turn on LoRA for this layer. (Used in the recovery stage.)
        """
        if self.lora_r > 0:
            return  # already enabled

        self.lora_r = int(r)
        self.lora_alpha = float(alpha)
        self.lora_dropout = float(dropout)

        # Create on the same device/dtype as base weights to avoid dtype mismatches.
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apple-style fake-quantized weight in forward (2/4-bit), with STE.
        qc = QATQuantConfig(n_bits=self.n_bits)
        w_q = fake_quant_weight_nbit(self.weight, self.f(), qc)
        y = F.linear(x, w_q, self.bias)

        # Optional LoRA residual (kept in FP)
        if self.lora_r > 0:
            x_d = self.lora_drop(x) if self.lora_drop is not None else x
            y = y + (x_d @ self.lora_A.t() @ self.lora_B.t()) * self.scaling

        return y
