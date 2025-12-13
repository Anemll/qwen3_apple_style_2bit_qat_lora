"""
Model utilities:
- Replace nn.Linear modules with QATLinear (fake-quant weights)
- Initialize Apple-style learnable f parameters
- Freeze base weights & enable LoRA for recovery stage
- Layerwise grad scaling: inverse sqrt(neuron count)
"""

from __future__ import annotations

import re
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .qat_linear import QATLinear
from .quantizer import QATQuantConfig, init_f_from_weight


def replace_linear_with_qat(
    model: nn.Module,
    qc: Optional[QATQuantConfig] = None,
    include_regex: Optional[str] = None,
    exclude_regex: Optional[str] = None,
    verbose: bool = True,
) -> int:
    """
    In-place replace nn.Linear modules with QATLinear.

    include_regex:
      - if provided, only module names matching it are replaced.

    exclude_regex:
      - if provided, module names matching it are NOT replaced.

    Returns:
      number of modules replaced
    """
    qc = qc or QATQuantConfig()

    include_re = re.compile(include_regex) if include_regex else None
    exclude_re = re.compile(exclude_regex) if exclude_regex else None

    replaced = 0
    for module_name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue

        if include_re is not None and include_re.search(module_name) is None:
            continue
        if exclude_re is not None and exclude_re.search(module_name) is not None:
            continue

        # Find parent module and attribute name
        parent = model
        if module_name == "":
            continue
        parts = module_name.split(".")
        for p in parts[:-1]:
            parent = getattr(parent, p)
        attr = parts[-1]

        qat = QATLinear(
            in_features=module.in_features,
            out_features=module.out_features,
            bias=(module.bias is not None),
            qc=qc,
            lora_r=0,
        )
        # Copy weights
        qat.weight.data.copy_(module.weight.data)
        if module.bias is not None:
            qat.bias.data.copy_(module.bias.data)

        setattr(parent, attr, qat)
        replaced += 1
        if verbose:
            print(f"[replace] {module_name}: nn.Linear -> QATLinear")

    return replaced


@torch.no_grad()
def init_all_f(
    model: nn.Module,
    qc: Optional[QATQuantConfig] = None,
    method: str = "newton",
    newton_iters: int = 4,
    newton_samples: int = 65536,
    percentile: float = 99.5,
    verbose: bool = True,
) -> int:
    """
    Initialize f for every QATLinear weight tensor.

    Returns number of layers initialized.
    """
    qc = qc or QATQuantConfig()
    count = 0
    for name, m in model.named_modules():
        if isinstance(m, QATLinear):
            f_init = init_f_from_weight(
                m.weight,
                qc=qc,
                method=method,
                newton_iters=newton_iters,
                newton_samples=newton_samples,
                percentile=percentile,
            )
            m.set_f_init(f_init)
            count += 1
            if verbose:
                print(f"[init_f] {name}: f_init={f_init:.6f}")
    return count


def freeze_base_enable_lora(
    model: nn.Module,
    lora_r: int,
    lora_alpha: float,
    lora_dropout: float,
    include_regex: Optional[str] = None,
    exclude_regex: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[int, int]:
    """
    Freeze base model params and enable LoRA on selected QATLinear layers.

    Returns:
        (num_lora_layers_enabled, num_trainable_params)
    """
    include_re = re.compile(include_regex) if include_regex else None
    exclude_re = re.compile(exclude_regex) if exclude_regex else None

    # Freeze everything
    for p in model.parameters():
        p.requires_grad = False

    lora_layers = 0
    for name, m in model.named_modules():
        if isinstance(m, QATLinear):
            if include_re is not None and include_re.search(name) is None:
                continue
            if exclude_re is not None and exclude_re.search(name) is not None:
                continue

            m.enable_lora(r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
            # enable training for LoRA params only
            m.lora_A.requires_grad = True
            m.lora_B.requires_grad = True
            lora_layers += 1
            if verbose:
                print(f"[lora] enabled on {name}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return lora_layers, trainable


def apply_layerwise_grad_scaling(model: nn.Module, verbose: bool = True) -> int:
    """
    Apple reports improved stability with gradient scaling inversely proportional
    to sqrt(neuron count). We interpret "neuron count" for Linear as out_features.

    This registers backward hooks multiplying gradients by 1/sqrt(out_features)
    for:
      - QATLinear.weight
      - QATLinear._f_param
    """
    hooks = 0
    for name, m in model.named_modules():
        if isinstance(m, QATLinear):
            scale = 1.0 / math.sqrt(float(m.out_features))

            def _make_hook(s: float):
                def hook(grad: torch.Tensor):
                    return grad * s
                return hook

            m.weight.register_hook(_make_hook(scale))
            m._f_param.register_hook(_make_hook(scale))
            hooks += 2
            if verbose:
                print(f"[grad_scale] {name}: x{scale:.6f}")

    return hooks


def extract_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Extract only LoRA parameters from a model containing QATLinear layers.
    """
    sd = {}
    for name, m in model.named_modules():
        if isinstance(m, QATLinear) and m.lora_r > 0:
            sd[f"{name}.lora_A"] = m.lora_A.detach().cpu()
            sd[f"{name}.lora_B"] = m.lora_B.detach().cpu()
    return sd
