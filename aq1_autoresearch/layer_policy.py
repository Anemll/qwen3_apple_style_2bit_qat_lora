#!/usr/bin/env python3
"""Helpers for per-layer AQ1 quantization policies."""

from __future__ import annotations

import math
import re
from dataclasses import replace
from typing import Any

import torch
import torch.nn as nn

from qat_lora.ane_qat_linear_v2 import AnemllQuantConfigV2, AnemllQATLinearV2


MLP_PATTERN = re.compile(r"\.mlp\.(gate_proj|up_proj|down_proj)$")
ATTN_PATTERN = re.compile(r"\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$")
LM_HEAD_PATTERN = re.compile(r"^lm_head$")


def layer_kind(name: str) -> str | None:
    if MLP_PATTERN.search(name):
        return "mlp"
    if ATTN_PATTERN.search(name):
        return "attn"
    if LM_HEAD_PATTERN.search(name):
        return "lm_head"
    return None


def is_quantized_linear_name(
    name: str,
    quantize_attn: bool = True,
    quantize_lm_head: bool = False,
) -> bool:
    kind = layer_kind(name)
    if kind == "mlp":
        return True
    if kind == "attn":
        return quantize_attn
    if kind == "lm_head":
        return quantize_lm_head
    return False


def summarize_layer_overrides(layer_overrides: dict[str, dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "num_layers": len(layer_overrides),
        "lut_bits": {},
        "scale_rank": {},
        "group_size": {},
    }
    for override in layer_overrides.values():
        lut_bits = override.get("lut_bits")
        scale_rank = override.get("scale_rank")
        group_size = override.get("group_size")
        if lut_bits is not None:
            summary["lut_bits"][lut_bits] = summary["lut_bits"].get(lut_bits, 0) + 1
        if scale_rank is not None:
            summary["scale_rank"][scale_rank] = summary["scale_rank"].get(scale_rank, 0) + 1
        if group_size is not None:
            summary["group_size"][group_size] = summary["group_size"].get(group_size, 0) + 1
    return summary


def read_layer_overrides_from_config(config: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    raw_overrides = (config or {}).get("layer_overrides") or {}
    normalized: dict[str, dict[str, Any]] = {}
    for name, override in raw_overrides.items():
        if not isinstance(override, dict):
            continue
        normalized[name] = {}
        if override.get("lut_bits") is not None:
            normalized[name]["lut_bits"] = int(override["lut_bits"])
        if override.get("scale_rank") is not None:
            normalized[name]["scale_rank"] = int(override["scale_rank"])
        if override.get("group_size") is not None:
            normalized[name]["group_size"] = int(override["group_size"])
    return normalized


def read_layer_overrides_from_state_dict(
    state_dict: dict[str, Any],
    default_group_size: int = 32,
) -> dict[str, dict[str, Any]]:
    overrides: dict[str, dict[str, Any]] = {}
    for key, value in state_dict.items():
        if not hasattr(value, "shape") or not key.endswith(".lut"):
            continue
        prefix = key[:-4]
        scale_a = state_dict.get(f"{prefix}.scale_A")
        if scale_a is None or not hasattr(scale_a, "shape"):
            continue
        lut_size = int(value.numel())
        overrides[prefix] = {
            "lut_bits": int(math.ceil(math.log2(lut_size))) if lut_size > 1 else 0,
            "scale_rank": int(scale_a.shape[1]),
            "group_size": int(default_group_size),
        }
    return overrides


def merge_layer_overrides(*override_maps: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for override_map in override_maps:
        for name, override in override_map.items():
            merged.setdefault(name, {}).update(override)
    return merged


def make_layer_config(
    base_config: AnemllQuantConfigV2,
    override: dict[str, Any] | None = None,
) -> AnemllQuantConfigV2:
    override = override or {}
    lut_bits = override.get("lut_bits")
    lut_size = override.get("lut_size")
    if lut_size is None and lut_bits is not None:
        lut_size = 2 ** int(lut_bits)

    return replace(
        base_config,
        lut_size=int(lut_size if lut_size is not None else base_config.lut_size),
        scale_rank=int(override.get("scale_rank", base_config.scale_rank)),
        group_size=int(override.get("group_size", base_config.group_size)),
    )


def replace_linear_with_layer_overrides(
    model: nn.Module,
    mlp_config: AnemllQuantConfigV2,
    attn_config: AnemllQuantConfigV2 | None = None,
    layer_overrides: dict[str, dict[str, Any]] | None = None,
    quantize_attn: bool = True,
    quantize_lm_head: bool = False,
    verbose: bool = True,
    skip_init: bool = False,
) -> int:
    attn_config = attn_config or mlp_config
    layer_overrides = layer_overrides or {}

    replacements: list[tuple[str, nn.Linear, AnemllQuantConfigV2, torch.Tensor | None]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or isinstance(module, AnemllQATLinearV2):
            continue
        if not is_quantized_linear_name(name, quantize_attn=quantize_attn, quantize_lm_head=quantize_lm_head):
            continue

        kind = layer_kind(name)
        base_config = attn_config if kind == "attn" else mlp_config
        override = layer_overrides.get(name)
        config = make_layer_config(base_config, override)
        custom_lut = override.get("custom_lut") if override else None
        replacements.append((name, module, config, custom_lut))

    if verbose and replacements:
        print(f"  Converting {len(replacements)} layers to V2...")

    named_modules = dict(model.named_modules())
    replaced = 0
    for idx, (name, linear_module, config, custom_lut) in enumerate(replacements):
        v2_layer = AnemllQATLinearV2.from_linear(
            linear_module,
            config=config,
            skip_init=skip_init,
            custom_lut=custom_lut,
        )
        if not skip_init:
            with torch.no_grad():
                v2_layer.rank_magnitude.data = v2_layer.rank_magnitude.data.to(torch.float16).to(torch.float32)

        parent_name, attr = name.rsplit(".", 1) if "." in name else ("", name)
        parent = named_modules[parent_name] if parent_name else model
        setattr(parent, attr, v2_layer)
        replaced += 1

        if verbose and (idx % 20 == 0 or idx == len(replacements) - 1):
            short_name = name.split(".")[-2] + "." + name.split(".")[-1] if "." in name else name
            print(
                f"    [{idx + 1}/{len(replacements)}] ({100 * (idx + 1) / len(replacements):.0f}%) "
                f"{short_name} lut={config.lut_size} rank={config.scale_rank}",
                flush=True,
            )

    if verbose and replaced:
        print(f"\nReplaced {replaced} layers with per-layer overrides", flush=True)
    return replaced
