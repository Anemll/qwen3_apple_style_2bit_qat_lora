"""
A simple single-device SFT training loop with:
- gradient accumulation
- optional mixed precision via torch.amp.autocast
- optional GradScaler (CUDA fp16 only)
- optional EMA (Apple suggests EMA helps 2-bit QAT stability)

Why not Transformers Trainer?
- On MPS, Trainer/Accelerate mixed precision still frequently errors (e.g. "fp16 mixed precision requires a GPU (not 'mps')")
  depending on versions.
- A manual loop gives full control over autocast/scaler behavior and is easier to debug.

This loop is intentionally minimal and well-commented.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from .ema import EMA
from .mixed_precision import autocast_context, make_grad_scaler


@dataclass
class LoopConfig:
    output_dir: str
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 16
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    max_steps: int = 2000  # optimizer steps (not micro-steps)
    warmup_steps: int = 100
    logging_steps: int = 10
    save_steps: int = 500
    max_grad_norm: float = 1.0

    # Mixed precision
    amp_dtype: Optional[torch.dtype] = None

    # EMA
    ema_decay: float = 0.0


def save_checkpoint(model: torch.nn.Module, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / name)


def train_sft_single_device(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    cfg: LoopConfig,
    tokenizer=None,
    extra_state: Optional[dict] = None,
):
    """
    Train for cfg.max_steps * cfg.gradient_accumulation_steps micro-steps.

    We treat cfg.max_steps as "optimizer steps" (after accumulation),
    which matches the semantics of HF TrainingArguments.
    """
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model.to(device)
    model.train()

    # Optimizer over trainable params only
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=cfg.max_steps,
    )

    scaler = make_grad_scaler(device, cfg.amp_dtype)

    ema = None
    if cfg.ema_decay and cfg.ema_decay > 0.0:
        ema = EMA(decay=cfg.ema_decay)
        ema.init(model)

    micro_steps_total = cfg.max_steps * cfg.gradient_accumulation_steps
    data_iter = iter(dataloader)

    running_loss = 0.0
    pbar = tqdm(range(micro_steps_total), desc="train", dynamic_ncols=True)

    optimizer.zero_grad(set_to_none=True)

    opt_step = 0
    for micro_step in pbar:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        # Move to device
        batch = {k: v.to(device) for k, v in batch.items()}

        with autocast_context(device, cfg.amp_dtype):
            out = model(**batch)
            loss = out.loss
            loss = loss / cfg.gradient_accumulation_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        running_loss += float(loss.detach().cpu())

        # Optimizer step after accumulation
        if (micro_step + 1) % cfg.gradient_accumulation_steps == 0:
            if cfg.max_grad_norm and cfg.max_grad_norm > 0.0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            opt_step += 1
            if ema is not None:
                ema.update(model)

            if opt_step % cfg.logging_steps == 0:
                lr = scheduler.get_last_lr()[0]
                avg_loss = running_loss / cfg.logging_steps
                running_loss = 0.0
                pbar.set_postfix({"opt_step": opt_step, "loss": f"{avg_loss:.4f}", "lr": f"{lr:.2e}"})

            if opt_step % cfg.save_steps == 0:
                # Save raw (non-EMA) weights
                save_checkpoint(model, out_dir, f"checkpoint_step{opt_step}.pt")
                # Optionally also save EMA shadow weights
                if ema is not None:
                    backup = ema.apply_to(model)
                    save_checkpoint(model, out_dir, f"checkpoint_step{opt_step}_ema.pt")
                    ema.restore(model, backup)

    # Final save
    save_checkpoint(model, out_dir, "final_state_dict.pt")
    if ema is not None:
        backup = ema.apply_to(model)
        save_checkpoint(model, out_dir, "final_state_dict_ema.pt")
        ema.restore(model, backup)

    # Persist extra state for reproducibility
    if tokenizer is not None:
        tokenizer.save_pretrained(out_dir)

    if extra_state is None:
        extra_state = {}
    extra_state.update(
        {
            "device": str(device),
            "amp_dtype": str(cfg.amp_dtype),
            "max_steps": cfg.max_steps,
            "grad_accum": cfg.gradient_accumulation_steps,
            "batch_size": cfg.per_device_train_batch_size,
            "lr": cfg.learning_rate,
            "weight_decay": cfg.weight_decay,
            "ema_decay": cfg.ema_decay,
        }
    )
    with open(out_dir / "run_state.json", "w") as f:
        json.dump(extra_state, f, indent=2)

    return out_dir
