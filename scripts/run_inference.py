"""
Minimal inference script for a QAT + (optional) LoRA recovery checkpoint.

Loads:
1) Base HF model
2) Replaces Linear -> QATLinear (optionally skipping lm_head)
3) Loads a model-only state_dict checkpoint (e.g. qat_state_dict.pt or final_state_dict_ema.pt)
4) Optionally enables LoRA per-layer and loads LoRA weights (lora_only_state_dict.pt)

Example:
  python scripts/run_inference.py \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --qat_checkpoint runs/qwen3_0p6b_qat2b/final_state_dict_ema.pt \
    --lora_checkpoint runs/qwen3_0p6b_qat2b_lora_phase2_lr5e5/lora_only_state_dict.pt \
    --device mps \
    --prompt "Explain what quantization-aware training is."
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure local package imports work without installation.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from qat_lora.model_utils import replace_linear_with_qat
from qat_lora.qat_linear import QATLinear
from qat_lora.quantizer import QATQuantConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-0.6B")
    p.add_argument("--qat_checkpoint", type=str, required=True, help="Model-only state_dict (.pt).")
    p.add_argument("--lora_checkpoint", type=str, default=None, help="LoRA-only state_dict (.pt).")
    p.add_argument("--skip_lm_head", action="store_true")
    p.add_argument(
        "-q",
        "--quant_bits",
        type=int,
        default=2,
        choices=[2, 4],
        help="Bitwidth used when constructing QATLinear modules before loading a checkpoint. "
        "If the checkpoint contains per-layer _qat_nbits buffers, those will override this.",
    )
    p.add_argument("--lora_r", type=int, default=32, help="LoRA rank (needed to set scaling).")
    p.add_argument("--lora_alpha", type=float, default=32.0, help="LoRA alpha (needed to set scaling).")
    p.add_argument("--lora_dropout", type=float, default=0.0)

    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--dtype", type=str, default="auto", choices=["auto", "fp32", "bf16", "fp16"])

    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--plain_text", action="store_true", help="Do not apply the chat template; treat prompt as raw text.")
    p.add_argument(
        "--enable_thinking",
        type=str,
        default="false",
        choices=["true", "false"],
        help="Controls whether the chat template emits a think block.",
    )
    p.add_argument(
        "--show_special_tokens",
        action="store_true",
        help="Print decoded text with special tokens (default strips them).",
    )
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument(
        "--do_sample",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Whether to sample (true) or use greedy decoding (false).",
    )
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    return p.parse_args()


def _pick_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def _resolve_dtype(dtype: str, device: torch.device) -> torch.dtype:
    if dtype == "fp32":
        return torch.float32
    if dtype == "fp16":
        return torch.float16
    if dtype == "bf16":
        return torch.bfloat16
    # auto
    if device.type == "cuda":
        return torch.bfloat16
    if device.type == "mps":
        return torch.bfloat16
    return torch.float32


def _load_lora_into_model(
    model: torch.nn.Module,
    lora_sd: Dict[str, torch.Tensor],
    *,
    lora_alpha: float,
    lora_dropout: float,
    expected_r: Optional[int] = None,
) -> int:
    """
    Enable LoRA for layers present in lora_sd, infer r from lora_A shape, and load weights.
    """
    enabled = 0
    for name, module in model.named_modules():
        if not isinstance(module, QATLinear):
            continue
        kA = f"{name}.lora_A"
        kB = f"{name}.lora_B"
        if kA not in lora_sd or kB not in lora_sd:
            continue
        A = lora_sd[kA]
        B = lora_sd[kB]
        r = int(A.shape[0])
        if expected_r is not None and r != int(expected_r):
            raise RuntimeError(f"LoRA rank mismatch for {name}: checkpoint r={r} but --lora_r={expected_r}")
        if module.lora_r <= 0:
            # Default scaling must match the training config; take it from CLI.
            module.enable_lora(r=r, alpha=float(lora_alpha), dropout=float(lora_dropout))
        module.lora_A.data.copy_(A.to(module.lora_A.device, dtype=module.lora_A.dtype))
        module.lora_B.data.copy_(B.to(module.lora_B.device, dtype=module.lora_B.dtype))
        enabled += 1
    return enabled


@torch.no_grad()
def main():
    args = parse_args()
    device = _pick_device(args.device)
    dtype = _resolve_dtype(args.dtype, device)
    enable_thinking = args.enable_thinking.lower() == "true"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, dtype=torch.float32)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.float32)
    model.eval()

    # Rebuild QATLinear structure and load QAT weights
    qc = QATQuantConfig(n_bits=int(args.quant_bits))
    exclude = r"(^lm_head$)" if args.skip_lm_head else None
    replace_linear_with_qat(model, qc=qc, exclude_regex=exclude, verbose=False)

    qat_sd = torch.load(args.qat_checkpoint, map_location="cpu")
    missing, unexpected = model.load_state_dict(qat_sd, strict=False)
    if unexpected:
        print(f"[load] unexpected keys: {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}")
    if missing:
        print(f"[load] missing keys: {missing[:10]}{'...' if len(missing) > 10 else ''}")

    # Cast and move
    model = model.to(device=device, dtype=dtype)

    # Optional LoRA
    if args.lora_checkpoint:
        lora_sd = torch.load(args.lora_checkpoint, map_location="cpu")
        enabled = _load_lora_into_model(
            model,
            lora_sd,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            expected_r=args.lora_r,
        )
        print(f"[lora] loaded adapters for {enabled} layers from {args.lora_checkpoint}")

    # Build chat prompt
    if args.plain_text:
        inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    else:
        messages = [{"role": "user", "content": args.prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)

    do_sample = args.do_sample.lower() == "true"
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = args.temperature
        gen_kwargs["top_p"] = args.top_p
    else:
        gen_kwargs["temperature"] = 1.0
        gen_kwargs["top_p"] = 1.0

    out = model.generate(**gen_kwargs)
    decoded = tokenizer.decode(out[0], skip_special_tokens=False)
    if args.show_special_tokens:
        print(decoded)
    else:
        cleaned = tokenizer.decode(out[0], skip_special_tokens=True).strip()
        print(cleaned)


if __name__ == "__main__":
    main()
