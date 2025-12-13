"""
Stage B: LoRA recovery adapters for a QAT model.

Flow:
1) Load base model
2) Replace Linear -> QATLinear (same as Stage A)
3) Load QAT state dict (weights + learned f)
4) Freeze base weights
5) Enable LoRA on QATLinear layers (train only LoRA A/B)
6) Run SFT loop (same dataset formatting)
7) Save:
   - lora_only_state_dict.pt
   - full_state_dict.pt

Mixed precision behavior is identical to Stage A.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import __version__ as transformers_version

# Ensure local package imports work without installation.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from qat_lora.quantizer import QATQuantConfig
from qat_lora.model_utils import replace_linear_with_qat, freeze_base_enable_lora, extract_lora_state_dict
from qat_lora.data import build_alpaca_messages, tokenize_chat_sft, DataCollatorForSFT
from qat_lora.mixed_precision import pick_device, resolve_amp_dtype, resolve_param_dtype
from qat_lora.train_loop import LoopConfig, train_sft_single_device


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-0.6B")
    p.add_argument("--qat_checkpoint", type=str, required=True)
    p.add_argument("--dataset_name", type=str, default="tatsu-lab/alpaca")
    p.add_argument("--dataset_split", type=str, default="train")
    p.add_argument("--dataset_format", type=str, choices=["alpaca"], default="alpaca")
    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument("--max_length", type=int, default=1024)

    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--warmup_steps", type=int, default=50)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    # Device & mixed precision
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--amp_dtype", type=str, default="auto", choices=["auto", "no", "bf16", "fp16"])
    p.add_argument("--param_dtype", type=str, default="auto", choices=["auto", "fp32", "bf16", "fp16"])

    p.add_argument("--skip_lm_head", action="store_true")
    p.add_argument("--enable_thinking", action="store_true")

    # LoRA config
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=float, default=32.0)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

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

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.float32)
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    qc = QATQuantConfig()
    exclude = r"(^lm_head$)" if args.skip_lm_head else None

    replace_linear_with_qat(model, qc=qc, exclude_regex=exclude, verbose=False)

    sd = torch.load(args.qat_checkpoint, map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"Loaded QAT checkpoint. missing={len(missing)} unexpected={len(unexpected)}")
    if unexpected:
        print("Unexpected keys (first 20):", unexpected[:20])

    # Cast base params to desired dtype BEFORE enabling LoRA
    model = model.to(dtype=param_dtype)

    # Freeze base + enable LoRA
    lora_layers, trainable = freeze_base_enable_lora(
        model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        verbose=False,
    )
    print(f"Enabled LoRA on {lora_layers} layers. Trainable params: {trainable:,}")

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
        ema_decay=0.0,
    )

    extra_state = {"stage": "lora_recovery", "args": vars(args)}
    train_sft_single_device(model, dl, device, loop_cfg, tokenizer=tokenizer, extra_state=extra_state)

    # Save LoRA only (recommended)
    lora_sd = extract_lora_state_dict(model)
    torch.save(lora_sd, out / "lora_only_state_dict.pt")

    # Save full (QAT + LoRA)
    torch.save(model.state_dict(), out / "full_state_dict.pt")

    with open(out / "training_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Done. Saved LoRA adapter to: {out/'lora_only_state_dict.pt'}")


if __name__ == "__main__":
    main()
