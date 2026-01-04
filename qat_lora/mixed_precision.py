"""
Device + mixed precision helpers.

Why this exists:
- Hugging Face Trainer/Accelerate still has rough edges for mixed precision on MPS
  (e.g. "fp16 mixed precision requires a GPU (not 'mps')" in some versions).
- PyTorch MPS supports autocast, but GradScaler is not yet reliable on MPS:
  GradScaler currently attempts to use float64 internally, which MPS doesn't support.

So we implement **manual autocast**:
- CUDA: autocast fp16/bf16; GradScaler for fp16 only
- MPS: autocast bf16/fp16; **no GradScaler**
- CPU: autocast bf16; no GradScaler

We also include small "can I even allocate this dtype on this device?" probes,
so users get a clear fallback path instead of a cryptic crash.
"""

from __future__ import annotations

import contextlib
import platform
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from packaging import version


@dataclass
class MPConfig:
    """
    Mixed precision config.

    amp_dtype:
      - "no":  no autocast
      - "bf16": autocast with bfloat16
      - "fp16": autocast with float16
      - "auto": pick a sensible default given the device

    param_dtype:
      - "fp32" / "bf16" / "fp16" / "auto"

    Notes:
      - On CUDA, the most stable setup is often:
          param_dtype=fp32, amp_dtype=bf16 or fp16
        (this mimics "AMP" behavior, keeping master weights in fp32)
      - On MPS, bf16 support depends on your PyTorch + macOS build.
        We attempt it, but will fall back to fp16 or fp32 if unsupported.
    """
    amp_dtype: str = "auto"
    param_dtype: str = "auto"


def _is_tpu_available() -> bool:
    """Check if TPU is available via torch_xla."""
    try:
        import torch_xla
        import torch_xla.core.xla_model as xm
        _ = xm.xla_device()
        return True
    except Exception:
        return False


def _get_tpu_device():
    """Get TPU device via torch_xla."""
    import torch_xla
    return torch_xla.device()


def pick_device(device_str: str = "auto") -> torch.device:
    """
    device_str: "auto" | "cuda" | "mps" | "cpu" | "tpu" | "xla"
    """
    device_str = device_str.lower()
    if device_str == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    if device_str == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Requested --device mps but torch.backends.mps.is_available() is False.")
        return torch.device("mps")
    if device_str in ("tpu", "xla"):
        if not _is_tpu_available():
            raise RuntimeError("Requested --device tpu/xla but torch_xla is not available.")
        return _get_tpu_device()
    if device_str == "cpu":
        return torch.device("cpu")
    if device_str == "auto":
        # Priority: CUDA > TPU > MPS > CPU
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _is_tpu_available():
            return _get_tpu_device()
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    raise ValueError(f"Unknown device: {device_str}")


def _can_alloc(device: torch.device, dtype: torch.dtype) -> bool:
    try:
        _ = torch.empty((1,), device=device, dtype=dtype)
        return True
    except Exception:
        return False


def resolve_param_dtype(requested: str, device: torch.device) -> torch.dtype:
    """
    Convert a string to a torch.dtype with a best-effort fallback.
    """
    requested = requested.lower()
    if requested == "fp32":
        return torch.float32
    if requested == "fp16":
        return torch.float16
    if requested == "bf16":
        return torch.bfloat16
    if requested == "auto":
        device_type = str(device).split(":")[0]  # Handle "xla:0" -> "xla"
        # CUDA: prefer bf16 if available, else fp16
        if device_type == "cuda":
            # On most modern NVIDIA GPUs, bf16 is available. We'll just pick bf16.
            return torch.bfloat16
        # TPU/XLA: bf16 is native and preferred
        if device_type == "xla":
            return torch.bfloat16
        # MPS: prefer fp16 (more widely supported) unless bf16 is allocatable
        if device_type == "mps":
            return torch.bfloat16 if _can_alloc(device, torch.bfloat16) else torch.float16
        # CPU: bf16 is generally okay on recent CPUs, but may be slow. Still a reasonable default.
        return torch.bfloat16
    raise ValueError(f"Unknown param_dtype: {requested}")


def resolve_amp_dtype(requested: str, device: torch.device) -> Optional[torch.dtype]:
    """
    Convert a string to a torch.dtype for autocast (or None for no autocast).
    """
    requested = requested.lower()
    if requested == "no":
        return None
    if requested == "fp16":
        return torch.float16
    if requested == "bf16":
        return torch.bfloat16
    if requested == "auto":
        if device.type == "cuda":
            return torch.bfloat16
        if device.type == "mps":
            # Prefer bf16 if available (PyTorch 2.6+ and macOS 14+).
            if _can_alloc(device, torch.bfloat16):
                return torch.bfloat16
            return torch.float16
        if device.type == "cpu":
            return torch.bfloat16
        return None
    raise ValueError(f"Unknown amp_dtype: {requested}")


def autocast_context(device: torch.device, amp_dtype: Optional[torch.dtype]):
    """
    Return a context manager for autocast if enabled, else a nullcontext.

    For MPS: torch.amp.autocast("mps", dtype=...) is supported in recent PyTorch,
    but if you request bf16 on an unsupported build, you may see a warning and/or fallback.
    """
    if amp_dtype is None:
        return contextlib.nullcontext()
    # torch.amp.autocast supports device_type string:
    return torch.amp.autocast(device_type=device.type, dtype=amp_dtype)


def make_grad_scaler(device: torch.device, amp_dtype: Optional[torch.dtype]) -> Optional[torch.amp.GradScaler]:
    """
    Return a GradScaler if we should use it.

    IMPORTANT:
      - CUDA + fp16: yes (standard AMP recipe)
      - CUDA + bf16: typically no
      - MPS: no (GradScaler currently unreliable on MPS due to float64 ops)
      - CPU: no
    """
    if device.type == "cuda" and amp_dtype == torch.float16:
        return torch.amp.GradScaler(device_type="cuda")
    return None
