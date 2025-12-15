"""
Single-device SFT training loop with:
- gradient accumulation
- optional mixed precision via torch.amp.autocast
- optional GradScaler (CUDA fp16 only)
- optional EMA (Apple suggests EMA helps 2-bit QAT stability)
- **checkpointing + resume support**

Why not Transformers Trainer?
- On MPS, Trainer/Accelerate mixed precision still frequently errors depending on versions.
- A manual loop gives full control over autocast/scaler behavior and is easier to debug.

Important semantics:
- `max_steps` = **optimizer steps** (matches HF Trainer's `max_steps`)
- Each optimizer step runs `gradient_accumulation_steps` micro-batches.

Resume semantics:
- When resuming, we continue from the saved optimizer step up to `max_steps`.
  Example: if checkpoint.opt_step == 120 and max_steps == 500, we will run steps 121..500.
"""

from __future__ import annotations

import json
import random
import re
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from .ema import EMA
from .mixed_precision import autocast_context, make_grad_scaler


CHECKPOINT_FORMAT_VERSION = 1


@dataclass
class LoopConfig:
    output_dir: str
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 16
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    max_steps: int = 2000  # optimizer steps
    warmup_steps: int = 100
    logging_steps: int = 10  # optimizer steps
    save_steps: int = 500  # optimizer steps
    max_grad_norm: float = 1.0

    # Mixed precision
    amp_dtype: Optional[torch.dtype] = None

    # EMA
    ema_decay: float = 0.0


def _read_last_logged_step(loss_csv: Path) -> int:
    """
    Best-effort read the last logged step from a CSV file.
    Returns 0 if the file doesn't exist or can't be parsed.
    """
    try:
        if not loss_csv.exists() or loss_csv.stat().st_size == 0:
            return 0
        # Read tail; steps are written in ascending order.
        with open(loss_csv, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 8192), 0)
            tail = f.read().decode("utf-8", errors="ignore").splitlines()
        for line in reversed(tail):
            line = line.strip()
            if not line or line.startswith("step,"):
                continue
            # step,loss,lr
            step_str = line.split(",", 1)[0].strip()
            return int(float(step_str))
    except Exception:
        return 0
    return 0


def _append_loss_csv(loss_csv: Path, *, step: int, loss: float, lr: float):
    """
    Append a row to loss.csv, writing a header if needed.
    """
    loss_csv.parent.mkdir(parents=True, exist_ok=True)
    new_file = (not loss_csv.exists()) or loss_csv.stat().st_size == 0
    with open(loss_csv, "a", encoding="utf-8") as f:
        if new_file:
            f.write("step,loss,lr\n")
        f.write(f"{int(step)},{float(loss):.8f},{float(lr):.12g}\n")


def _atomic_save(obj: Any, path: Path):
    """
    Best-effort atomic torch.save:
    write to tmp file then rename.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def _move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device):
    """
    After optimizer.load_state_dict, state tensors can remain on CPU.
    Move them to the training device (cuda/mps) so steps don't error or run slowly.
    """
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


def _is_full_train_checkpoint(obj: Any) -> bool:
    return isinstance(obj, dict) and "model" in obj and "optimizer" in obj and "opt_step" in obj


def _parse_step_from_filename(path: Path) -> int:
    """
    Best-effort parse "...step123..." from filename.
    """
    m = re.search(r"step(\d+)", path.name)
    if m:
        return int(m.group(1))
    return 0


def save_training_checkpoint(
    *,
    out_dir: Path,
    opt_step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: Optional[torch.amp.GradScaler],
    ema: Optional[EMA],
    cfg: LoopConfig,
    device: torch.device,
):
    """
    Save a full training checkpoint that supports resume.

    We also write a `checkpoint_last.pt` pointer for easy auto-resume.
    """
    ckpt = {
        "format": "qat_lora_train_ckpt",
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "opt_step": int(opt_step),
        "cfg": asdict(cfg),
        "device_type": device.type,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "ema": ema.shadow if ema is not None else None,
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "python": random.getstate(),
        },
    }

    ckpt_path = out_dir / f"checkpoint_step{opt_step}.pt"
    _atomic_save(ckpt, ckpt_path)

    # Maintain a "last" checkpoint pointer
    last_path = out_dir / "checkpoint_last.pt"
    _atomic_save(ckpt, last_path)


def resolve_resume_checkpoint(resume_from_checkpoint: str, output_dir: str) -> Path:
    """
    Resolve resume_from_checkpoint into an actual checkpoint path.

    Supported forms:
    - explicit path to a checkpoint .pt file
    - a directory => we search for checkpoint_step*.pt inside it
    - "auto" / "latest" => search in output_dir for highest checkpoint_step*.pt, else checkpoint_last.pt
    - "last" => output_dir/checkpoint_last.pt if present, else same as "auto"
    """
    out_dir = Path(output_dir)
    token = resume_from_checkpoint.strip().lower()

    if token in {"auto", "latest"}:
        last = out_dir / "checkpoint_last.pt"
        if last.exists():
            return last

        candidates = sorted(out_dir.glob("checkpoint_step*.pt"))
        if not candidates:
            raise FileNotFoundError(f"No checkpoints found in {out_dir}")

        best = max(candidates, key=_parse_step_from_filename)
        return best

    if token in {"last"}:
        last = out_dir / "checkpoint_last.pt"
        if last.exists():
            return last
        return resolve_resume_checkpoint("auto", output_dir)

    p = Path(resume_from_checkpoint)
    if p.is_dir():
        candidates = sorted(p.glob("checkpoint_step*.pt"))
        if candidates:
            return max(candidates, key=_parse_step_from_filename)
        last = p / "checkpoint_last.pt"
        if last.exists():
            return last
        raise FileNotFoundError(f"No checkpoints found in directory {p}")

    if not p.exists():
        # allow relative to output_dir
        p2 = out_dir / resume_from_checkpoint
        if p2.exists():
            return p2
        raise FileNotFoundError(f"Checkpoint not found: {p}")

    return p


def load_training_checkpoint(
    *,
    ckpt_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: Optional[torch.amp.GradScaler],
    ema: Optional[EMA],
    device: torch.device,
) -> tuple[int, bool]:
    """
    Load a checkpoint and return:
      (saved_opt_step, is_full_checkpoint)

    Supports:
    - full training checkpoints created by this file (dict with model/optimizer/...)
    - "model-only" checkpoints (plain state_dict); in that case we resume weights only
      and restart optimizer/scheduler.

    NOTE: For full checkpoints we attempt to restore RNG state too for reproducibility.
    """
    obj = torch.load(ckpt_path, map_location="cpu")

    if _is_full_train_checkpoint(obj):
        missing, unexpected = model.load_state_dict(obj["model"], strict=False)
        if missing or unexpected:
            print(f"[resume] model.load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
            if unexpected:
                print("[resume] unexpected keys (first 10):", unexpected[:10])

        optimizer.load_state_dict(obj["optimizer"])
        _move_optimizer_state_to_device(optimizer, device)

        scheduler.load_state_dict(obj["scheduler"])

        if scaler is not None and obj.get("scaler") is not None:
            try:
                scaler.load_state_dict(obj["scaler"])
            except Exception as e:
                print(f"[resume] WARNING: failed to load GradScaler state: {e}")

        if ema is not None and obj.get("ema") is not None:
            ema.shadow = {k: v.to(device) for k, v in obj["ema"].items()}

        rng = obj.get("rng", {})
        try:
            if rng.get("torch") is not None:
                torch.set_rng_state(rng["torch"])
            if rng.get("cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng["cuda"])
            if rng.get("python") is not None:
                random.setstate(rng["python"])
        except Exception as e:
            print(f"[resume] WARNING: failed to restore RNG state: {e}")

        opt_step = int(obj.get("opt_step", 0))
        print(f"[resume] Loaded full checkpoint from {ckpt_path} at opt_step={opt_step}")
        return opt_step, True

    # Model-only checkpoint compatibility
    if isinstance(obj, dict) and all(isinstance(k, str) for k in obj.keys()):
        missing, unexpected = model.load_state_dict(obj, strict=False)
        if missing or unexpected:
            print(f"[resume] model-only state_dict: missing={len(missing)} unexpected={len(unexpected)}")
            if unexpected:
                print("[resume] unexpected keys (first 10):", unexpected[:10])

        opt_step = _parse_step_from_filename(ckpt_path)
        print(
            f"[resume] Loaded MODEL-ONLY checkpoint from {ckpt_path}. "
            f"Will resume weights but reinitialize optimizer/scheduler. Parsed opt_step={opt_step}."
        )
        return opt_step, False

    raise RuntimeError(f"Unrecognized checkpoint format: {ckpt_path}")


def train_sft_single_device(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    cfg: LoopConfig,
    tokenizer=None,
    extra_state: Optional[dict] = None,
    resume_from_checkpoint: Optional[str] = None,
):
    """
    Train for `cfg.max_steps` optimizer steps.

    Resume:
      - set resume_from_checkpoint to a checkpoint path,
        or to "auto"/"latest"/"last" to pick from output_dir.
    """
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model.to(device)
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    start_opt_step = 0
    resumed_full = False
    if resume_from_checkpoint:
        ckpt_path = resolve_resume_checkpoint(resume_from_checkpoint, cfg.output_dir)
        # Create scheduler/scaler/ema before loading: full checkpoints restore their state.
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

        start_opt_step, resumed_full = load_training_checkpoint(
            ckpt_path=ckpt_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            ema=ema,
            device=device,
        )
    else:
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

    # If we resumed from a model-only checkpoint, keep LR schedule aligned with opt_step.
    # (Optimizer/scheduler state cannot be restored in that case.)
    if resume_from_checkpoint and (not resumed_full) and start_opt_step > 0:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=cfg.warmup_steps,
            num_training_steps=cfg.max_steps,
            last_epoch=start_opt_step - 1,
        )

    if start_opt_step >= cfg.max_steps:
        print(f"[train] start_opt_step ({start_opt_step}) >= max_steps ({cfg.max_steps}). Nothing to do.")
        return out_dir

    loss_csv = out_dir / "loss.csv"
    last_logged_step = _read_last_logged_step(loss_csv)
    # If a previous run wrote loss.csv with larger step indices, and we're starting a new run
    # (or resuming from a checkpoint earlier than the last logged step), we'd never append new rows.
    # Rotate the existing file so the current run can write a fresh loss.csv.
    if last_logged_step > 0 and start_opt_step < last_logged_step:
        rotated = out_dir / f"loss_prev_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        try:
            loss_csv.replace(rotated)
            print(f"[loss.csv] Rotated existing {loss_csv.name} (last_step={last_logged_step}) -> {rotated.name}")
        except Exception as e:
            print(f"[loss.csv] WARNING: failed to rotate existing loss.csv: {e}")
        last_logged_step = 0

    micro_per_opt = cfg.gradient_accumulation_steps
    data_iter = iter(dataloader)
    running_loss = 0.0

    pbar = tqdm(
        range(start_opt_step + 1, cfg.max_steps + 1),
        desc="opt_step",
        dynamic_ncols=True,
        unit="step",
        initial=start_opt_step,
        total=cfg.max_steps,
    )

    optimizer.zero_grad(set_to_none=True)

    for opt_step in pbar:
        for _micro in range(micro_per_opt):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            batch = {k: v.to(device) for k, v in batch.items()}

            with autocast_context(device, cfg.amp_dtype):
                out = model(**batch)
                loss = out.loss / micro_per_opt

            # Fail fast on NaNs/Infs (common symptom of unstable low-precision runs).
            if not torch.isfinite(loss).all():
                # Best-effort save a model-only snapshot for debugging/repro.
                try:
                    _atomic_save(model.state_dict(), out_dir / f"nan_state_dict_step{opt_step}.pt")
                except Exception as e:  # pragma: no cover
                    print(f"[nan] WARNING: failed to save nan_state_dict_step{opt_step}.pt: {e}")
                raise RuntimeError(
                    f"Non-finite loss detected at opt_step={opt_step}. "
                    "Try lowering learning_rate, increasing warmup_steps, using --amp_dtype no, "
                    "and/or setting --param_dtype fp32."
                )

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            running_loss += float(loss.detach().cpu())

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

        if ema is not None:
            ema.update(model)

        if opt_step % cfg.logging_steps == 0:
            lr = scheduler.get_last_lr()[0]
            avg_loss = running_loss / cfg.logging_steps
            running_loss = 0.0
            pbar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{lr:.2e}"})
            if opt_step > last_logged_step:
                _append_loss_csv(loss_csv, step=opt_step, loss=avg_loss, lr=lr)
                last_logged_step = opt_step

        if opt_step % cfg.save_steps == 0:
            save_training_checkpoint(
                out_dir=out_dir,
                opt_step=opt_step,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                ema=ema,
                cfg=cfg,
                device=device,
            )

    _atomic_save(model.state_dict(), out_dir / "final_state_dict.pt")
    if ema is not None:
        backup = ema.apply_to(model)
        _atomic_save(model.state_dict(), out_dir / "final_state_dict_ema.pt")
        ema.restore(model, backup)

    if tokenizer is not None:
        tokenizer.save_pretrained(out_dir)

    if extra_state is None:
        extra_state = {}
    extra_state.update(
        {
            "device": str(device),
            "amp_dtype": str(cfg.amp_dtype),
            "max_steps": cfg.max_steps,
            "start_opt_step": start_opt_step,
            "grad_accum": cfg.gradient_accumulation_steps,
            "batch_size": cfg.per_device_train_batch_size,
            "effective_batch_size": cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps,
            "lr": cfg.learning_rate,
            "weight_decay": cfg.weight_decay,
            "ema_decay": cfg.ema_decay,
        }
    )
    with open(out_dir / "run_state.json", "w") as f:
        json.dump(extra_state, f, indent=2)

    return out_dir
