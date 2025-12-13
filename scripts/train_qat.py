"""
Stage A: Apple-style 2-bit QAT on Qwen/Qwen3-0.6B (or any HF causal LM).

Key differences vs a "vanilla" Transformers Trainer script:
- We do NOT rely on HF Trainer AMP on MPS (which can be flaky depending on versions).
- Instead we run a small manual loop with torch.amp.autocast.

On CUDA:
- --amp_dtype fp16 => autocast fp16 + GradScaler
- --amp_dtype bf16 => autocast bf16 (no scaler)

On MPS:
- autocast works (fp16 is widely supported; bf16 support depends on your build)
- GradScaler is currently unreliable on MPS, so we do NOT use it.

We also implement Apple-specific QAT details:
- Replace nn.Linear with QATLinear (fake int2 weights with balanced levels)
- Initialize per-layer f using Newton-like clip estimator
- Optional layerwise grad scaling 1/sqrt(out_features)
- Optional EMA of weights

Apple also recommends weight_decay=0 for 2-bit QAT; this is the default here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import __version__ as transformers_version

# Ensure local package imports work without installation.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from qat_lora.quantizer import QATQuantConfig
from qat_lora.model_utils import replace_linear_with_qat, init_all_f, apply_layerwise_grad_scaling
from qat_lora.data import build_alpaca_messages, tokenize_chat_sft, DataCollatorForSFT
from qat_lora.mixed_precision import pick_device, MPConfig, resolve_amp_dtype, resolve_param_dtype
from qat_lora.train_loop import LoopConfig, train_sft_single_device


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-0.6B")
    p.add_argument("--dataset_name", type=str, default="tatsu-lab/alpaca")
    p.add_argument("--dataset_split", type=str, default="train")
    p.add_argument("--dataset_format", type=str, choices=["alpaca"], default="alpaca")
    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument("--max_length", type=int, default=1024)

    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)  # Apple recommends 0 for 2-bit QAT
    p.add_argument("--max_steps", type=int, default=2000)       # optimizer steps
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    # Device & mixed precision
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--amp_dtype", type=str, default="auto", choices=["auto", "no", "bf16", "fp16"])
    p.add_argument("--param_dtype", type=str, default="auto", choices=["auto", "fp32", "bf16", "fp16"])

    p.add_argument("--skip_lm_head", action="store_true", help="Do not quantize lm_head (recommended).")

    p.add_argument("--init_method", type=str, choices=["newton", "percentile"], default="newton")
    p.add_argument("--init_newton_iters", type=int, default=4)
    p.add_argument("--init_newton_samples", type=int, default=65536)
    p.add_argument("--init_percentile", type=float, default=99.5)

    p.add_argument("--enable_thinking", action="store_true", help="Qwen3 thinking mode in chat template. Default False.")
    p.add_argument("--grad_scale", action="store_true", help="Apply layerwise grad scaling 1/sqrt(out_features).")

    p.add_argument("--ema_decay", type=float, default=0.0, help="If >0, maintain EMA with this decay (e.g., 0.999).")

    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Qwen3 model card explicitly warns transformers<4.51.0 will throw KeyError: 'qwen3'
    if tuple(map(int, transformers_version.split(".")[:2])) < (4, 51):
        raise RuntimeError(
            f"Transformers {transformers_version} is too old for Qwen3. "
            "Please upgrade: pip install -U 'transformers>=4.51.0'."
        )

    device = pick_device(args.device)
    amp_dtype = resolve_amp_dtype(args.amp_dtype, device)
    param_dtype = resolve_param_dtype(args.param_dtype, device)

    print(f"[device] {device} | amp_dtype={amp_dtype} | param_dtype={param_dtype}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load on CPU first (more predictable), then .to(device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float32,  # load in fp32, then cast below for stability control
    )
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # Replace Linear -> QATLinear (on CPU)
    qc = QATQuantConfig()
    exclude = r"(^lm_head$)" if args.skip_lm_head else None

    print("Replacing Linear layers with QATLinear...")
    n_rep = replace_linear_with_qat(model, qc=qc, exclude_regex=exclude, verbose=False)
    print(f"Replaced {n_rep} Linear layers.")

    print("Initializing f parameters...")
    init_all_f(
        model,
        qc=qc,
        method=args.init_method,
        newton_iters=args.init_newton_iters,
        newton_samples=args.init_newton_samples,
        percentile=args.init_percentile,
        verbose=False,
    )

    if args.grad_scale:
        print("Registering layerwise grad scaling hooks...")
        apply_layerwise_grad_scaling(model, verbose=False)

    # Cast parameters (after QATLinear creation) to desired dtype.
    # NOTE: this affects memory and numerical stability.
    model = model.to(dtype=param_dtype)

    # Dataset
    ds = load_dataset(args.dataset_name, split=args.dataset_split)

    def map_fn(ex):
        msgs = build_alpaca_messages(ex)
        return tokenize_chat_sft(
            tokenizer=tokenizer,
            messages=msgs,
            max_length=args.max_length,
            enable_thinking=args.enable_thinking,
        )

    ds = ds.map(map_fn, remove_columns=ds.column_names)
    collator = DataCollatorForSFT(tokenizer)

    from torch.utils.data import DataLoader
    dl = DataLoader(
        ds,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collator,
        drop_last=True,
    )

    loop_cfg = LoopConfig(
        output_dir=str(out),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        max_grad_norm=args.max_grad_norm,
        amp_dtype=amp_dtype,
        ema_decay=args.ema_decay,
    )

    extra_state = {"stage": "qat", "args": vars(args)}
    train_sft_single_device(model, dl, device, loop_cfg, tokenizer=tokenizer, extra_state=extra_state)

    # Save a convenient "qat_state_dict.pt" name (as in the earlier version of this repo)
    torch.save(model.state_dict(), out / "qat_state_dict.pt")

    with open(out / "training_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Done. QAT checkpoint saved to: {out/'qat_state_dict.pt'}")


if __name__ == "__main__":
    main()
