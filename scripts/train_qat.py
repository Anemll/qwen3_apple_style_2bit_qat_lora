"""
Stage A: Apple-style 2-bit QAT on Qwen/Qwen3-0.6B.

This script:
- loads the HF model
- replaces nn.Linear with QATLinear (fake-quant weights)
- initializes f per layer (Newton-like clip estimator)
- optionally registers layerwise grad scaling hooks
- optionally runs EMA of weights
- trains using HF Trainer

Outputs:
- qat_state_dict.pt (includes learned f params)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers.utils import __version__ as transformers_version

# Ensure local package imports work without installation.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from qat_lora.quantizer import QATQuantConfig
from qat_lora.model_utils import replace_linear_with_qat, init_all_f, apply_layerwise_grad_scaling
from qat_lora.data import build_alpaca_messages, tokenize_chat_sft, DataCollatorForSFT
from qat_lora.ema import EMA
from qat_lora.callbacks import EMACallback


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--dataset_name", type=str, default="tatsu-lab/alpaca")
    p.add_argument("--dataset_split", type=str, default="train")
    p.add_argument("--dataset_format", type=str, choices=["alpaca"], default="alpaca")
    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument("--max_length", type=int, default=1024)

    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)  # Apple sets weight decay to 0 for 2-bit QAT
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=500)

    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")

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

    # Qwen3 requires transformers >= 4.51.0 (model card warns KeyError with older).
    # We'll do a light check.
    # NOTE: This is not a strict semantic version parser, just a guardrail.
    if tuple(map(int, transformers_version.split(".")[:2])) < (4, 51):
        raise RuntimeError(
            f"Transformers {transformers_version} is too old for Qwen3. "
            "Please pip install -U 'transformers>=4.51.0'."
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    # ensure pad token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None),
    )

    # Disable cache for training (gradient checkpointing compatibility)
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    qc = QATQuantConfig()

    exclude = None
    if args.skip_lm_head:
        # skip module exactly named lm_head
        exclude = r"(^lm_head$)"

    print("Replacing Linear layers with QATLinear...")
    n_rep = replace_linear_with_qat(model, qc=qc, exclude_regex=exclude, verbose=True)
    print(f"Replaced {n_rep} Linear layers.")

    print("Initializing f parameters...")
    n_init = init_all_f(
        model,
        qc=qc,
        method=args.init_method,
        newton_iters=args.init_newton_iters,
        newton_samples=args.init_newton_samples,
        percentile=args.init_percentile,
        verbose=False,
    )
    print(f"Initialized f for {n_init} QATLinear layers.")

    if args.grad_scale:
        print("Registering layerwise grad scaling hooks...")
        apply_layerwise_grad_scaling(model, verbose=False)

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

    training_args = TrainingArguments(
        output_dir=str(out),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        bf16=args.bf16,
        fp16=args.fp16,
        report_to=[],
        remove_unused_columns=False,
        save_total_limit=2,
    )

    callbacks = []
    if args.ema_decay and args.ema_decay > 0.0:
        ema = EMA(decay=args.ema_decay)
        callbacks.append(EMACallback(ema=ema, save_ema_path=str(out)))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=collator,
        callbacks=callbacks,
    )

    trainer.train()

    # Save QAT state dict (includes QATLinear._f_param)
    torch.save(model.state_dict(), out / "qat_state_dict.pt")

    # Save args
    with open(out / "training_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Save tokenizer for convenience
    tokenizer.save_pretrained(out)

    print(f"Done. QAT checkpoint saved to: {out/'qat_state_dict.pt'}")


if __name__ == "__main__":
    main()
