"""
Stage A: Apple-style 2-bit QAT on Qwen/Qwen3-0.6B (or any HF causal LM).

This script:
- loads a HF causal LM ("student")
- replaces nn.Linear with QATLinear (Apple-style fake-quant weights)
- initializes per-layer learnable scale factor f (Newton-like clip estimator)
- optionally applies layerwise grad scaling 1/sqrt(out_features)
- trains with a custom single-device loop (MPS-friendly mixed precision)
- optionally maintains EMA of weights during training

NEW: Optional **knowledge-distillation QAT (KD-QAT)**

If --teacher_model_name_or_path is provided, we freeze a "teacher" model and
train the quantized student to match the teacher's next-token distribution
via KL divergence:

    loss = distill_weight * KL( teacher || student ) + (1 - distill_weight) * CE(labels)

- If distill_weight=1.0 (default), training is pure KL distillation (no CE).
- distill_temperature is the softmax temperature used for KL.

This is a good fit when your goal is *preserving* the original model behavior
under aggressive 2-bit quantization (rather than learning new knowledge).

Dataset formats:
- alpaca: tatsu-lab/alpaca-style {instruction,input,output} -> chat template SFT masking
- text: plain text dataset with a configurable text field, suitable for general-text KD-QAT

Outputs in --output_dir:
- qat_state_dict.pt (plain model state_dict, convenience)
- final_state_dict.pt / final_state_dict_ema.pt (from the training loop)
- checkpoint_step{N}.pt and checkpoint_last.pt (full training state, if --save_steps < max_steps)
- loss.csv (logged loss points for plotting)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import __version__ as transformers_version

# Optional collator for plain-text LM datasets
try:
    from transformers import DataCollatorForLanguageModeling
except Exception:  # pragma: no cover
    DataCollatorForLanguageModeling = None  # type: ignore

# Ensure local package imports work without installation.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from qat_lora.data import DataCollatorForSFT, build_alpaca_messages, tokenize_chat_sft
from qat_lora.model_utils import apply_layerwise_grad_scaling, init_all_f, replace_linear_with_qat
from qat_lora.mixed_precision import pick_device, resolve_amp_dtype, resolve_param_dtype
from qat_lora.quantizer import QATQuantConfig
from qat_lora.train_loop import LoopConfig, train_sft_single_device


def _from_pretrained_fp32(model_name_or_path: str, trust_remote_code: bool = False):
    """
    Transformers is deprecating torch_dtype= in favor of dtype= in some versions.
    Support both to avoid warnings/breakage across installs.
    """
    # Some transformers builds accept dtype=, some accept torch_dtype=.
    # Some also accept trust_remote_code=; handle both gracefully.
    kwargs = {}
    if trust_remote_code:
        kwargs["trust_remote_code"] = True
    try:
        return AutoModelForCausalLM.from_pretrained(model_name_or_path, dtype=torch.float32, **kwargs)
    except TypeError:
        # Older transformers
        return AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float32, **kwargs)


def parse_args():
    p = argparse.ArgumentParser()

    # Model
    p.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-0.6B")

    # Distillation (optional)
    p.add_argument(
        "--teacher_model_name_or_path",
        type=str,
        default=None,
        help="If set, enable KD-QAT: run a frozen teacher and match its next-token distribution (KL).",
    )
    p.add_argument(
        "--distill_weight",
        type=float,
        default=1.0,
        help="Weight on KD loss. total = w*KL + (1-w)*CE. Use 1.0 for pure distillation.",
    )
    p.add_argument(
        "--distill_temperature",
        type=float,
        default=1.0,
        help="Temperature for distillation softmax (common: 1.0 or 2.0).",
    )

    # Dataset
    p.add_argument("--dataset_name", type=str, default="tatsu-lab/alpaca")
    p.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="Optional HF dataset config name (e.g. fineweb: sample-10BT, dolma: v1_6-sample).",
    )
    p.add_argument("--dataset_split", type=str, default="train")
    p.add_argument("--dataset_format", type=str, choices=["alpaca", "text"], default="alpaca")
    p.add_argument(
        "--dataset_text_field",
        type=str,
        default="text",
        help="For dataset_format=text, which column contains the text.",
    )
    p.add_argument(
        "--streaming",
        action="store_true",
        help="Stream the dataset (does not download the full dataset).",
    )
    p.add_argument(
        "--shuffle_buffer",
        type=int,
        default=10_000,
        help="For streaming datasets, shuffle buffer size (0 disables streaming shuffle).",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Pass trust_remote_code=True to HF loaders (needed for some datasets/models).",
    )

    p.add_argument("--output_dir", type=str, required=True)

    # Train
    p.add_argument("--max_length", type=int, default=1024)

    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)  # Apple recommends 0 for 2-bit QAT
    p.add_argument("--max_steps", type=int, default=2000)  # optimizer steps
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

    p.add_argument("--enable_thinking", action="store_true", help="Qwen3 thinking mode in chat template.")
    p.add_argument("--grad_scale", action="store_true", help="Apply layerwise grad scaling 1/sqrt(out_features).")
    p.add_argument("--ema_decay", type=float, default=0.0, help="If >0, maintain EMA with this decay (e.g., 0.999).")

    p.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to a training checkpoint .pt file. Use auto/latest/last to pick from output_dir.",
    )

    return p.parse_args()


def _build_dataset_and_collator(args, tokenizer):
    """Return (dataset, collator, shuffle_for_dataloader)."""
    ds_kwargs = {
        "split": args.dataset_split,
        "streaming": bool(args.streaming),
    }
    if args.dataset_config_name:
        ds_kwargs["name"] = args.dataset_config_name
    if args.trust_remote_code:
        ds_kwargs["trust_remote_code"] = True

    ds = load_dataset(args.dataset_name, **ds_kwargs)

    # Streaming shuffle must be done on the dataset itself (DataLoader shuffle won't work).
    if args.streaming and args.shuffle_buffer and args.shuffle_buffer > 0:
        try:
            ds = ds.shuffle(buffer_size=args.shuffle_buffer, seed=args.seed)
        except Exception as e:
            print(f"[warn] dataset.shuffle(...) failed in streaming mode: {e}. Continuing without shuffle.")

    if args.dataset_format == "alpaca":
        def map_fn(ex):
            msgs = build_alpaca_messages(ex)
            return tokenize_chat_sft(
                tokenizer=tokenizer,
                messages=msgs,
                max_length=args.max_length,
                enable_thinking=args.enable_thinking,
            )

        remove_cols = ds.column_names if hasattr(ds, "column_names") else None
        if remove_cols:
            ds = ds.map(map_fn, remove_columns=remove_cols)
        else:
            ds = ds.map(map_fn)

        collator = DataCollatorForSFT(tokenizer)
        shuffle = not args.streaming

        return ds, collator, shuffle

    if args.dataset_format == "text":
        if DataCollatorForLanguageModeling is None:
            raise RuntimeError(
                "DataCollatorForLanguageModeling is not available in your transformers install. "
                "Upgrade transformers (>=4.0) or switch dataset_format=alpaca."
            )

        text_field = args.dataset_text_field

        def map_fn(ex):
            txt = ex.get(text_field, None)
            if txt is None:
                raise KeyError(
                    f"dataset_format=text expects a '{text_field}' field. Available keys: {list(ex.keys())}"
                )
            # Plain causal LM tokenization. Labels are created by the collator.
            return tokenizer(txt, truncation=True, max_length=args.max_length)

        remove_cols = ds.column_names if hasattr(ds, "column_names") else None
        if remove_cols:
            ds = ds.map(map_fn, remove_columns=remove_cols)
        else:
            ds = ds.map(map_fn)

        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        shuffle = not args.streaming

        return ds, collator, shuffle

    raise ValueError(f"Unsupported dataset_format: {args.dataset_format}")


def _install_kd_forward(
    student_model: torch.nn.Module,
    teacher_model: torch.nn.Module,
    *,
    distill_weight: float,
    temperature: float,
):
    """
    Monkeypatch student_model.forward so that it returns a .loss computed from
    KL(teacher || student) (and optionally CE(labels)).

    This keeps checkpointing/resume intact (we do NOT wrap the model, we only replace forward).
    """
    if not (0.0 <= distill_weight <= 1.0):
        raise ValueError(f"distill_weight must be in [0,1]. Got {distill_weight}")
    if temperature <= 0:
        raise ValueError(f"distill_temperature must be >0. Got {temperature}")

    orig_forward = student_model.forward

    def forward_kd(
        input_ids=None,
        attention_mask=None,
        labels=None,
        **kwargs,
    ):
        # Make sure we get a dict-like output with .logits/.loss.
        kwargs.setdefault("return_dict", True)
        kwargs.setdefault("use_cache", False)

        # Student forward
        if distill_weight < 1.0 and labels is not None:
            student_out = orig_forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
            ce_loss = student_out.loss
        else:
            student_out = orig_forward(input_ids=input_ids, attention_mask=attention_mask, labels=None, **kwargs)
            ce_loss = None

        student_logits = student_out.logits

        # Teacher forward (frozen)
        with torch.no_grad():
            teacher_out = teacher_model(input_ids=input_ids, attention_mask=attention_mask, labels=None, **kwargs)
            teacher_logits = teacher_out.logits

        # Shift like standard causal LM loss: predict token t+1 from position t
        s = student_logits[:, :-1, :]
        t = teacher_logits[:, :-1, :]

        # Mask: follow labels (if provided) so KD matches the same token positions you'd train on.
        if labels is not None:
            y = labels[:, 1:]
            mask = (y != -100)
        elif attention_mask is not None:
            mask = attention_mask[:, 1:].bool()
        else:
            mask = torch.ones(s.shape[:2], dtype=torch.bool, device=s.device)

        # KL( teacher || student )
        T = float(temperature)
        log_p_s = F.log_softmax(s / T, dim=-1)
        p_t = F.softmax(t / T, dim=-1)

        # Per-token KL over vocab
        kd_per_tok = F.kl_div(log_p_s, p_t, reduction="none").sum(-1)  # [B, L-1]
        denom = mask.sum().clamp(min=1)
        kd_loss = (kd_per_tok * mask).sum() / denom
        kd_loss = kd_loss * (T * T)

        if ce_loss is not None and distill_weight < 1.0:
            loss = distill_weight * kd_loss + (1.0 - distill_weight) * ce_loss
        else:
            loss = kd_loss

        student_out.loss = loss
        return student_out

    student_model.forward = forward_kd  # type: ignore[attr-defined]


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

    # Load student on CPU first (more predictable), then cast and train on device.
    student = _from_pretrained_fp32(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
    student.config.use_cache = False
    if hasattr(student, "gradient_checkpointing_enable"):
        student.gradient_checkpointing_enable()

    # Replace with QATLinear + initialize f
    qc = QATQuantConfig()
    exclude = r"(^lm_head$)" if args.skip_lm_head else None
    replace_linear_with_qat(student, qc=qc, exclude_regex=exclude, verbose=False)
    init_all_f(
        student,
        qc=qc,
        method=args.init_method,
        newton_iters=args.init_newton_iters,
        newton_samples=args.init_newton_samples,
        percentile=args.init_percentile,
        verbose=False,
    )
    if args.grad_scale:
        apply_layerwise_grad_scaling(student, verbose=False)

    # Cast parameters (after QATLinear creation) to desired dtype.
    student = student.to(dtype=param_dtype)

    # Dataset + dataloader
    ds, collator, shuffle = _build_dataset_and_collator(args, tokenizer)

    from torch.utils.data import DataLoader

    dl = DataLoader(
        ds,
        batch_size=args.per_device_train_batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        drop_last=True,
    )

    # Optional KD-QAT: load teacher and patch student.forward to compute KL loss
    teacher: Optional[torch.nn.Module] = None
    if args.teacher_model_name_or_path:
        print(
            f"[kd] teacher={args.teacher_model_name_or_path} | weight={args.distill_weight} | T={args.distill_temperature}"
        )
        teacher = _from_pretrained_fp32(args.teacher_model_name_or_path, trust_remote_code=args.trust_remote_code)
        teacher.config.use_cache = False
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        teacher = teacher.to(device=device, dtype=param_dtype)

        _install_kd_forward(
            student,
            teacher,
            distill_weight=float(args.distill_weight),
            temperature=float(args.distill_temperature),
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

    extra_state = {
        "stage": "qat",
        "args": vars(args),
        "kd": {
            "enabled": bool(args.teacher_model_name_or_path),
            "teacher": args.teacher_model_name_or_path,
            "weight": float(args.distill_weight),
            "temperature": float(args.distill_temperature),
        },
    }

    train_sft_single_device(
        student,
        dl,
        device,
        loop_cfg,
        tokenizer=tokenizer,
        extra_state=extra_state,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )

    # Convenience name used by the rest of the repo.
    torch.save(student.state_dict(), out / "qat_state_dict.pt")
    with open(out / "training_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Done. QAT checkpoint saved to: {out/'qat_state_dict.pt'}")


if __name__ == "__main__":
    main()
