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


# ============================================================================
# Progressive Layer-by-Layer QAT Utilities
# ============================================================================

def freeze_all_except_layer(
    model: nn.Module,
    layer_idx: int,
    component: str = 'mlp',
    train_layernorm: bool = False,
    train_f_only: bool = False,
) -> int:
    """
    Freeze all parameters except specified layer's component.

    Args:
        model: The model to modify
        layer_idx: Which layer to keep trainable
        component: 'mlp' or 'attn'
        train_layernorm: Also train RMSNorm scale for this block (stabilizer)
        train_f_only: If True, only train _f_param (quantization scales), not weights.
                      This is more stable for ultra-low-bit quantization.

    Returns:
        Number of trainable parameters after freezing
    """
    # Freeze everything first
    for p in model.parameters():
        p.requires_grad = False

    # Patterns for target component
    if component == 'mlp':
        patterns = [f'model.layers.{layer_idx}.mlp.']
        if train_layernorm:
            patterns.append(f'model.layers.{layer_idx}.post_attention_layernorm.')
    else:  # attn
        patterns = [f'model.layers.{layer_idx}.self_attn.']
        if train_layernorm:
            patterns.append(f'model.layers.{layer_idx}.input_layernorm.')

    for name, p in model.named_parameters():
        if any(pat in name for pat in patterns):
            if train_f_only:
                # Only train _f_param (quantization scale) - more stable
                if name.endswith('_f_param'):
                    p.requires_grad = True
            else:
                # Train weight and _f_param (quantization scale)
                if name.endswith('.weight') or name.endswith('_f_param'):
                    p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable


def set_quantized_prefix(
    model: nn.Module,
    up_to_layer_idx: int,
    pass_type: str = 'mlp',
) -> None:
    """
    Enable fake-quant for finalized prefix, keep suffix fully fp.

    CRITICAL: Suffix layers (> up_to_layer_idx) must be FULLY fp (both MLP+attn)
    to provide stable downstream gradients.

    Args:
        model: The model to modify
        up_to_layer_idx: Current layer being trained
        pass_type: 'mlp' (Pass 1/3), 'attn' (Pass 2), or 'all' (E2E phase)
    """
    for name, m in model.named_modules():
        if not isinstance(m, QATLinear):
            continue
        match = re.search(r'model\.layers\.(\d+)', name)
        if not match:
            continue
        layer_i = int(match.group(1))
        is_mlp = '.mlp.' in name

        if layer_i > up_to_layer_idx:
            # SUFFIX: fully fp (both MLP and attn)
            m.enable_fake_quant = False
        elif layer_i < up_to_layer_idx:
            # PREFIX: fully quantized (finalized)
            m.enable_fake_quant = True
        else:
            # CURRENT LAYER (layer_i == up_to_layer_idx)
            if pass_type == 'mlp':
                m.enable_fake_quant = is_mlp  # Only MLP quantized
            elif pass_type == 'attn':
                m.enable_fake_quant = True  # Both quantized (MLP already done)
            else:  # 'all'
                m.enable_fake_quant = True


def get_frozen_mlp_copy(model: nn.Module, layer_idx: int) -> Dict[str, torch.Tensor]:
    """
    Create a frozen fp copy of a single MLP for local reconstruction loss.
    Returns dict of detached weight tensors.

    NOTE: In 4→2 curriculum, stage2 will use stage1-trained weights as target.
    This is intentional - global KD anchors to original teacher behavior.

    Args:
        model: The model (expects Qwen-style architecture)
        layer_idx: Which layer's MLP to copy

    Returns:
        Dict with 'gate', 'up', 'down' weight tensors (detached, cloned)
    """
    mlp = model.model.layers[layer_idx].mlp
    return {
        'gate': mlp.gate_proj.weight.detach().clone(),
        'up': mlp.up_proj.weight.detach().clone(),
        'down': mlp.down_proj.weight.detach().clone(),
    }


def infer_num_layers(model: nn.Module) -> int:
    """
    Infer the number of transformer layers in the model.

    Args:
        model: The model (expects model.model.layers structure)

    Returns:
        Number of layers
    """
    return len(model.model.layers)


def apply_train_f_only(model: nn.Module) -> int:
    """
    Freeze all parameters except _f_param (quantization scale) in QATLinear layers.
    Used for E2E quantizer-only tuning phase.

    Returns:
        Number of trainable f parameters
    """
    for p in model.parameters():
        p.requires_grad = False

    count = 0
    for name, p in model.named_parameters():
        if name.endswith('_f_param'):
            p.requires_grad = True
            count += 1

    return count
