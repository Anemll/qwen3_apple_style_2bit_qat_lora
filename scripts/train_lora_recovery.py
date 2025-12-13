"""
Stage B: LoRA recovery adapters for a QAT model.

This script:
- loads the base HF model
- replaces nn.Linear with QATLinear (same as Stage A)
- loads the learned QAT state dict (weights + f)
- freezes base weights, enables LoRA on QATLinear layers
- trains only LoRA parameters
- saves:
    - lora_only_state_dict.pt (recommended)
    - full_state_dict.pt      (QAT+LoRA together)
"""

from __future__ import annotations

import argparse
import json
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
from qat_lora.model_utils import (
    replace_linear_with_qat,
    freeze_base_enable_lora,
    extract_lora_state_dict,
)
from qat_lora.data import build_alpaca_messages, tokenize_chat_sft, DataCollatorForSFT


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", type=str, required=True)
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

    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")

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
            "Please pip install -U 'transformers>=4.51.0'."
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None),
    )

    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    qc = QATQuantConfig()
    exclude = None
    if args.skip_lm_head:
        exclude = r"(^lm_head$)"

    # Rebuild the exact QATLinear structure and then load the QAT checkpoint.
    replace_linear_with_qat(model, qc=qc, exclude_regex=exclude, verbose=False)

    sd = torch.load(args.qat_checkpoint, map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"Loaded QAT checkpoint. missing={len(missing)} unexpected={len(unexpected)}")
    if unexpected:
        print("Unexpected keys (showing first 20):", unexpected[:20])

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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=collator,
    )

    trainer.train()

    # Save LoRA only
    lora_sd = extract_lora_state_dict(model)
    torch.save(lora_sd, out / "lora_only_state_dict.pt")

    # Save full (QAT + LoRA)
    torch.save(model.state_dict(), out / "full_state_dict.pt")

    with open(out / "training_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    tokenizer.save_pretrained(out)

    print(f"Done. Saved LoRA adapter to: {out/'lora_only_state_dict.pt'}")


if __name__ == "__main__":
    main()
