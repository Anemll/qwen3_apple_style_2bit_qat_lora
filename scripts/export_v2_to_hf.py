#!/usr/bin/env python3
"""
Export a V2 QAT checkpoint (AnemllQATLinearV2) to a plain HF-compatible model.

Pipeline:
  1) Load base HF model (Gemma3, etc.)
  2) Replace Linear layers with AnemllQATLinearV2 (skip_init)
  3) Load V2 checkpoint (load_v2_checkpoint)
  4) Optional: snap to FP16 for ANE (snap_model_for_ane_v2)
  5) Bake W_eff = Q * scales into nn.Linear weights
  6) Save HF model + tokenizer

Usage:
  python scripts/export_v2_to_hf.py \
    --model-id google/gemma-3-1b-it \
    --checkpoint runs/gemma3_1b_qat_init/v2_initial.pt \
    --config runs/gemma3_1b_qat_init/config.json \
    --output runs/gemma3_1b_qat_baked \
    --snap-ane --recompute-indices
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import sys
import torch
import torch.nn as nn


def _load_v2_config(config_path: Path) -> Dict[str, object]:
    with config_path.open("r") as f:
        return json.load(f)


def _build_quant_configs(cfg: Dict[str, object]) -> Tuple[AnemllQuantConfigV2, AnemllQuantConfigV2]:
    from qat_lora.ane_qat_linear_v2 import AnemllQuantConfigV2

    # Required fields (from init_model_v2.py config.json)
    mlp_bits = int(cfg.get("mlp_lut_bits", cfg.get("lut_bits", 2)))
    attn_bits = int(cfg.get("attn_lut_bits", mlp_bits))
    mlp_rank = int(cfg.get("mlp_scale_rank", cfg.get("scale_rank", 32)))
    attn_rank = int(cfg.get("attn_scale_rank", mlp_rank))
    group_size = int(cfg.get("group_size", 32))

    force_pos = bool(cfg.get("force_positive_scales", False))
    mag_act = str(cfg.get("magnitude_activation", "identity"))
    mag_eps = float(cfg.get("magnitude_eps", 0.0)) if cfg.get("magnitude_eps") is not None else 0.0

    mlp_cfg = AnemllQuantConfigV2(
        lut_size=2 ** mlp_bits,
        scale_rank=mlp_rank,
        group_size=group_size,
        force_positive_scales=force_pos,
        positive_scale_method="abs",
        magnitude_activation=mag_act,
        magnitude_eps=mag_eps,
    )
    attn_cfg = AnemllQuantConfigV2(
        lut_size=2 ** attn_bits,
        scale_rank=attn_rank,
        group_size=group_size,
        force_positive_scales=force_pos,
        positive_scale_method="abs",
        magnitude_activation=mag_act,
        magnitude_eps=mag_eps,
    )
    return mlp_cfg, attn_cfg


def _replace_v2_with_linear(model: nn.Module, verbose: bool = True) -> int:
    """Replace all AnemllQATLinearV2 modules with nn.Linear using baked W_eff."""
    from qat_lora.ane_qat_linear_v2 import AnemllQATLinearV2

    replaced = 0

    for name, module in list(model.named_modules()):
        if not isinstance(module, AnemllQATLinearV2):
            continue

        # Compute effective weight: W_eff = Q * scales
        if module._Q is not None:
            q = module._Q
        elif module._indices is not None:
            q = module.lut[module._indices]
        else:
            raise RuntimeError(f"{name}: missing _Q and _indices; call snap_model_for_ane_v2 first")

        q = q.view(module.out_features, module.in_features)
        scales = module._compute_full_scales()
        w_eff = q * scales

        new_linear = nn.Linear(module.in_features, module.out_features, bias=module.bias is not None)
        new_linear = new_linear.to(device=module.weight.device, dtype=module.weight.dtype)
        new_linear.weight.data = w_eff.to(new_linear.weight.dtype)
        if module.bias is not None:
            new_linear.bias.data = module.bias.data.to(new_linear.bias.dtype)

        # Replace on parent
        parent = model
        if "." in name:
            parent_path, child_name = name.rsplit(".", 1)
            for part in parent_path.split("."):
                parent = getattr(parent, part)
        else:
            child_name = name
        setattr(parent, child_name, new_linear)
        replaced += 1

        if verbose and replaced <= 3:
            print(f"  [bake] {name} → nn.Linear")

    if verbose:
        print(f"  Replaced {replaced} V2 layers with nn.Linear")
    return replaced


def _load_tokenizer(model_id: str, trust_remote_code: bool = True):
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            use_fast=False,
            fix_mistral_regex=True,
        )
    except TypeError:
        # Older transformers may not support fix_mistral_regex.
        # Use slow tokenizer path to avoid the known fast-tokenizer regex issue.
        return AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            use_fast=False,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Export V2 QAT checkpoint to HF weights")
    parser.add_argument("--model-id", type=str, required=True, help="HF model ID or local path")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to V2 checkpoint (.pt)")
    parser.add_argument("--config", type=str, default=None, help="Path to V2 config.json (default: alongside checkpoint)")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output directory")
    parser.add_argument("--snap-ane", action="store_true", help="Snap to FP16 for ANE (recommended)")
    parser.add_argument("--recompute-indices", action="store_true",
                        help="Recompute LUT indices during FP16 snap (recommended)")
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "float16", "bfloat16"],
                        help="Base model dtype (default: float32)")
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimal output")

    args = parser.parse_args()
    verbose = not args.quiet

    repo_dir = Path(__file__).resolve().parent.parent
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))

    # Lazy import so CLI help works even if HF runtime deps are mismatched.
    from transformers import AutoModelForCausalLM
    from qat_lora.ane_qat_linear_v2 import (
        replace_linear_with_anemll_v2,
        load_v2_checkpoint,
        snap_model_for_ane_v2,
    )

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    config_path = Path(args.config) if args.config else ckpt_path.parent / "config.json"
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    if verbose:
        print("=" * 60)
        print("Export V2 QAT → HF Weights")
        print("=" * 60)
        print(f"Model:      {args.model_id}")
        print(f"Checkpoint: {ckpt_path}")
        print(f"Config:     {config_path}")
        print(f"Output:     {args.output}")
        print(f"Dtype:      {args.dtype}")
        print(f"Snap ANE:   {args.snap_ane} (recompute_indices={args.recompute_indices})")

    cfg = _load_v2_config(config_path)
    mlp_cfg, attn_cfg = _build_quant_configs(cfg)

    if verbose:
        print("\nLoading base model...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            dtype=dtype,
            trust_remote_code=True,
        )
    except TypeError:
        # Backward compatibility with older transformers.
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
    # Keep script portable: no accelerate dependency via device_map.
    model.to("cpu")

    if verbose:
        print("Replacing linears with V2 (skip_init)...")
    replace_linear_with_anemll_v2(
        model=model,
        mlp_config=mlp_cfg,
        attn_config=attn_cfg,
        quantize_attn=True,
        verbose=verbose,
        skip_init=True,
    )

    if verbose:
        print("Loading V2 checkpoint...")
    load_v2_checkpoint(model, str(ckpt_path), device=torch.device("cpu"), verbose=verbose, prefer_indices=True)

    if args.snap_ane:
        if verbose:
            print("\nSnapping to FP16 for ANE...")
        snap_model_for_ane_v2(model, recompute_indices=args.recompute_indices, verbose=verbose)

    if verbose:
        print("\nBaking W_eff into nn.Linear weights...")
    _replace_v2_with_linear(model, verbose=verbose)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\nSaving HF model to {output_dir}...")
    model.save_pretrained(output_dir)

    try:
        tokenizer = _load_tokenizer(args.model_id, trust_remote_code=True)
        tokenizer.save_pretrained(output_dir)
        if verbose:
            print("  Saved tokenizer")
    except Exception as e:
        if verbose:
            print(f"  Warning: Could not save tokenizer: {e}")

    export_meta = {
        "source_model": args.model_id,
        "checkpoint": str(ckpt_path),
        "config": str(config_path),
        "snap_ane": bool(args.snap_ane),
        "recompute_indices": bool(args.recompute_indices),
    }
    with open(output_dir / "export_meta.json", "w") as f:
        json.dump(export_meta, f, indent=2)

    if verbose:
        print("\nDone.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
