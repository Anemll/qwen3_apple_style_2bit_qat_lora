#!/usr/bin/env python3
"""
ANE group-quant simulation on Qwen3-0.6B MLP weights.

What this script does:
- Loads `Qwen/Qwen3-0.6B` (or a local/cached model id).
- Finds MLP Linear weights (gate/up/down projections).
- Quantizes + dequantizes each tensor with groupwise uniform LUT levels.
- Prints per-tensor and overall reconstruction error.
- Runs a simple inference sanity check on the quantized/dequantized model.

Notes:
- This is a *simulation*: we replace weights with dequantized float weights.
- The LUT is defined in normalized space [-1, 1] with `lut_size` entries and can optionally include 0 (default: off).

Examples:
- Quick sanity (uses HF cache only): `python3 tests/dev/ANE-group-quant.py --local-files-only --limit 1 --run-baseline`
- Full MLP quant run: `python3 tests/dev/ANE-group-quant.py --local-files-only --group-size 128 --lut-size 16`
- Qwen thinking template: `python3 tests/dev/ANE-group-quant.py --local-files-only --prompt-format chat_think --max-new-tokens 128`
- Attention-only quant run: `python3 tests/dev/ANE-group-quant.py --local-files-only --attn-only --limit 4 --skip-inference`
- Save a quant bundle: `python3 tests/dev/ANE-group-quant.py --local-files-only --skip-inference --save-quantized-dir out/qwen3_quant_bundle`
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import dataclass
from typing import DefaultDict, Dict, Iterable, Tuple
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import save_file as safetensors_save_file


@dataclass(frozen=True)
class QuantStats:
    mae: float
    mse: float
    rmse: float
    rel_l2: float
    err_l2: float
    weight_l2: float


@dataclass(frozen=True)
class GMMSizeStats:
    num_groups: int
    max_rank: int
    effective_rank: int
    full_scales_elems: int
    full_scales_bytes: int
    ab_elems: int
    ab_bytes: int


@dataclass(frozen=True)
class StorageEstimate:
    weights_elems: int
    weights_elems_padded: int
    lut_bits: int
    index_bits_total: int
    scales_bytes: int
    lut_bytes: int
    total_bits: int
    packed_bytes_total: int


def _pick_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _pick_dtype(dtype: str) -> torch.dtype:
    dtype = dtype.lower()
    if dtype in {"fp32", "float32"}:
        return torch.float32
    if dtype in {"fp16", "float16"}:
        return torch.float16
    if dtype in {"bf16", "bfloat16"}:
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype}")


def _low_rank_approx(matrix: torch.Tensor, rank: int) -> torch.Tensor:
    if rank <= 0:
        return matrix
    max_rank = min(matrix.shape[0], matrix.shape[1])
    if rank >= max_rank:
        return matrix
    rank = min(rank, matrix.shape[0], matrix.shape[1])
    u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    u = u[:, :rank]
    s = s[:rank]
    vh = vh[:rank, :]
    return (u * s) @ vh


def _dtype_nbytes(dtype: torch.dtype) -> int:
    if dtype in {torch.float16, torch.bfloat16}:
        return 2
    if dtype in {torch.float32}:
        return 4
    return torch.finfo(dtype).bits // 8


def compute_gmm_size_stats(
    *,
    out_features: int,
    in_features: int,
    group_size: int,
    scale_rank: int,
    storage_dtype: torch.dtype,
) -> GMMSizeStats:
    if group_size <= 0:
        raise ValueError("group_size must be > 0")
    pad = (-in_features) % group_size
    padded_in = in_features + pad
    num_groups = padded_in // group_size
    max_rank = min(out_features, num_groups)
    effective_rank = 0 if scale_rank <= 0 else min(scale_rank, max_rank)

    nbytes = _dtype_nbytes(storage_dtype)
    full_scales_elems = out_features * num_groups
    full_scales_bytes = full_scales_elems * nbytes

    # "GMM vectors" here refer to storing low-rank factors for scales:
    # A: [out_features, r], B: [r, num_groups]  => r*(out_features + num_groups) elements.
    ab_elems = effective_rank * (out_features + num_groups)
    ab_bytes = ab_elems * nbytes

    return GMMSizeStats(
        num_groups=num_groups,
        max_rank=max_rank,
        effective_rank=effective_rank,
        full_scales_elems=full_scales_elems,
        full_scales_bytes=full_scales_bytes,
        ab_elems=ab_elems,
        ab_bytes=ab_bytes,
    )


def _lut_bits(lut_size: int) -> int:
    if lut_size < 2:
        raise ValueError("lut_size must be >= 2")
    return int(math.ceil(math.log2(lut_size)))


def make_lut(
    *,
    lut_size: int,
    device: torch.device,
    dtype: torch.dtype,
    include_zero: bool = True,
) -> torch.Tensor:
    """
    Create a monotonic LUT in [-1, 1].

    IMPORTANT: LUT values are created in FP16 first, then upcast to target dtype.
    This ensures all values are FP16-representable, preventing precision loss
    when snapping to FP16 for ANE export.

    If `include_zero=True` and `lut_size` is even, the LUT cannot be perfectly symmetric while including 0.
    We generate `lut_size//2` points in [0, 1] and `lut_size//2+1` points in [-1, 0], then drop the duplicate 0.
    """
    if lut_size < 2:
        raise ValueError("lut_size must be >= 2")
    if not include_zero or (lut_size % 2 == 1):
        # Create in FP16 first, then cast to target dtype for FP16-representable values
        lut_fp16 = torch.linspace(-1.0, 1.0, steps=lut_size, device='cpu', dtype=torch.float16)
        return lut_fp16.to(device=device, dtype=dtype)

    # Even size with zero: non-uniform but includes 0
    # Create in FP16 first for FP16-representable values
    neg = torch.linspace(-1.0, 0.0, steps=lut_size // 2 + 1, device='cpu', dtype=torch.float16)
    pos = torch.linspace(0.0, 1.0, steps=lut_size // 2, device='cpu', dtype=torch.float16)
    lut_fp16 = torch.cat([neg[:-1], pos], dim=0)
    return lut_fp16.to(device=device, dtype=dtype)


def quantize_to_lut_indices(
    normalized_values: torch.Tensor,
    *,
    lut_size: int,
    include_zero: bool,
) -> torch.Tensor:
    """
    Map normalized values in [-1, 1] to nearest LUT entry indices.

    Fast path:
    - Uniform LUT: arithmetic rounding.
    - Even-size `include_zero=True` LUT produced by `make_lut`: sign-split arithmetic rounding.
    """
    if lut_size < 2:
        raise ValueError("lut_size must be >= 2")
    x = normalized_values.clamp(-1.0, 1.0)

    # Uniform LUT covers all cases except even-size include_zero=True.
    if (not include_zero) or (lut_size % 2 == 1):
        step = 2.0 / (lut_size - 1)
        return torch.round((x + 1.0) / step).to(torch.int64).clamp_(0, lut_size - 1)

    # Even lut_size with include_zero=True: LUT has different spacing on negative vs positive side.
    # Use a single rounding step + torch.where to keep this reasonably fast on large tensors.
    half = lut_size // 2
    step_neg = 1.0 / half
    step_pos = 1.0 / max(1, (half - 1))

    y_neg = (x + 1.0) / step_neg  # [-1,0) -> [0, half]
    y_pos = (x / step_pos) + half  # [0,1]  -> [half, lut_size-1]
    y = torch.where(x < 0, y_neg, y_pos)
    return torch.round(y).to(torch.int64).clamp_(0, lut_size - 1)


def estimate_storage_bits(
    *,
    out_features: int,
    in_features: int,
    group_size: int,
    lut_size: int,
    scale_rank: int,
    assume_fp16_scales: bool = True,
    assume_fp16_lut: bool = True,
) -> StorageEstimate:
    if group_size <= 0:
        raise ValueError("group_size must be > 0")
    if lut_size < 2:
        raise ValueError("lut_size must be >= 2")

    weights_elems = out_features * in_features
    pad = (-in_features) % group_size
    padded_in = in_features + pad
    weights_elems_padded = out_features * padded_in

    lut_bits = _lut_bits(lut_size)
    # Packed to bytes (GGUF-like): per-row ceil(padded_in * lut_bits / 8).
    packed_cols = int(math.ceil((padded_in * lut_bits) / 8.0))
    packed_bytes_total = out_features * packed_cols
    index_bits_total = packed_bytes_total * 8

    # Scales: FP16 by request.
    scale_dtype = torch.float16 if assume_fp16_scales else torch.float32
    gmm = compute_gmm_size_stats(
        out_features=out_features,
        in_features=in_features,
        group_size=group_size,
        scale_rank=scale_rank,
        storage_dtype=scale_dtype,
    )
    # If rank is full (>= max_rank), treat as storing full scales (A,B would be larger and is a no-op anyway).
    if scale_rank <= 0 or gmm.effective_rank >= gmm.max_rank:
        scales_bytes = gmm.full_scales_bytes
    else:
        scales_bytes = gmm.ab_bytes

    # LUT values: assume FP16 storage; we count per-tensor (conservative).
    lut_bytes = lut_size * (2 if assume_fp16_lut else 4)

    total_bits = index_bits_total + 8 * (scales_bytes + lut_bytes)
    return StorageEstimate(
        weights_elems=weights_elems,
        weights_elems_padded=weights_elems_padded,
        lut_bits=lut_bits,
        index_bits_total=index_bits_total,
        scales_bytes=scales_bytes,
        lut_bytes=lut_bytes,
        total_bits=total_bits,
        packed_bytes_total=packed_bytes_total,
    )


def _svd_ab(matrix: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return low-rank factors A,B such that matrix ~= A @ B.
    A: [m, r], B: [r, n]
    """
    if rank <= 0:
        raise ValueError("rank must be > 0")
    u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    r = min(rank, u.shape[1])
    a = u[:, :r] * s[:r]
    b = vh[:r, :]
    return a, b


def _pack_indices_2d(indices: torch.Tensor, bits: int) -> torch.Tensor:
    """
    Pack a [rows, cols] tensor of integer indices into uint8 bytes using `bits` per value.
    Output shape: [rows, ceil(cols * bits / 8)]
    """
    if indices.ndim != 2:
        raise ValueError("indices must be 2D")
    if not (1 <= bits <= 8):
        raise ValueError("bits must be in [1,8]")
    rows, cols = indices.shape
    packed_cols = int(math.ceil((cols * bits) / 8.0))
    mask = (1 << bits) - 1

    idx = (indices.to(torch.int64) & mask).contiguous()
    offsets = torch.arange(cols, device=idx.device, dtype=torch.int64) * bits
    byte = offsets // 8
    shift = offsets % 8
    cross = (shift + bits) > 8
    cross_pos = torch.nonzero(cross, as_tuple=False).flatten()

    base = (torch.arange(rows, device=idx.device, dtype=torch.int64) * packed_cols).unsqueeze(1)
    flat_len = rows * packed_cols
    packed_flat = torch.zeros(flat_len, device=idx.device, dtype=torch.int32)

    tgt_low = (base + byte).reshape(-1)
    low = ((idx << shift) & 0xFF).to(torch.int32).reshape(-1)
    packed_flat.scatter_add_(0, tgt_low, low)

    if cross_pos.numel() > 0:
        shift_cross = shift[cross_pos]
        byte_cross = byte[cross_pos] + 1
        tgt_high = (base + byte_cross).reshape(rows, -1).reshape(-1)
        idx_cross = idx.index_select(1, cross_pos)
        high = (idx_cross >> (8 - shift_cross)).to(torch.int32).reshape(-1) & 0xFF
        # byte_cross always < packed_cols for any crossing field.
        packed_flat.scatter_add_(0, tgt_high, high)

    return packed_flat.view(rows, packed_cols).to(torch.uint8)


def quantize_to_payload(
    weight: torch.Tensor,
    *,
    group_size: int,
    lut_size: int,
    scale_rank: int,
    eps: float = 1e-8,
    store_scales_fp16: bool = True,
    lut_include_zero: bool = True,
    post_gmm_rescale: bool = False,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, object]]:
    """
    Quantize a 2D tensor into packed indices + (scales or A,B) plus metadata.
    Returns:
      - dequantized weight (same dtype as input)
      - tensors dict to be saved
      - JSON-serializable metadata dict
    """
    if weight.ndim != 2:
        raise ValueError(f"Expected 2D weight, got shape={tuple(weight.shape)}")
    if group_size <= 0:
        raise ValueError("group_size must be > 0")
    if lut_size < 2:
        raise ValueError("lut_size must be >= 2")

    orig_dtype = weight.dtype
    w = weight.float()
    out_features, in_features = w.shape

    pad = (-in_features) % group_size
    if pad:
        w = torch.nn.functional.pad(w, (0, pad))
    padded_in = w.shape[1]
    num_groups = padded_in // group_size

    grouped = w.view(out_features, num_groups, group_size)
    scales_full = grouped.abs().amax(dim=2).clamp_min(eps)  # [out, groups]

    max_rank = min(out_features, num_groups)
    eff_rank = 0 if scale_rank <= 0 else min(scale_rank, max_rank)
    use_ab = eff_rank > 0 and eff_rank < max_rank

    saved: Dict[str, torch.Tensor] = {}
    scale_dtype = torch.float16 if store_scales_fp16 else torch.float32
    if use_ab:
        a32, b32 = _svd_ab(scales_full, eff_rank)
        a = a32.to(scale_dtype).contiguous()
        b = b32.to(scale_dtype).contiguous()
        # Reconstruct using stored precision (GGUF-like: what you'd get at inference).
        approx_scales = (a.float() @ b.float()).clamp_min(eps)
        saved["A"] = a
        saved["B"] = b
        scale_rep = "ab"
    else:
        scales_saved = scales_full.to(scale_dtype).contiguous()
        approx_scales = scales_saved.float().clamp_min(eps)
        saved["scales"] = scales_saved
        scale_rep = "full"

    lut = make_lut(lut_size=lut_size, device=w.device, dtype=w.dtype, include_zero=lut_include_zero)
    # Quantize based on the reconstructed (saved) scale representation so indices match what the bundle encodes.
    normalized = grouped / approx_scales.unsqueeze(-1)
    indices_3d = quantize_to_lut_indices(normalized, lut_size=lut_size, include_zero=lut_include_zero).to(torch.int64)
    indices_2d = indices_3d.view(out_features, padded_in)

    dequant = lut[indices_3d] * approx_scales.unsqueeze(-1)
    dequant = dequant.view(out_features, padded_in)
    if pad:
        dequant = dequant[:, :in_features]
    dequant = dequant.contiguous().to(orig_dtype)

    bits = _lut_bits(lut_size)
    packed = _pack_indices_2d(indices_2d, bits)
    saved["qbytes"] = packed

    meta: Dict[str, object] = {
        "shape": [int(out_features), int(in_features)],
        "padded_in": int(padded_in),
        "pad": int(pad),
        "group_size": int(group_size),
        "num_groups": int(num_groups),
        "lut_size": int(lut_size),
        "lut_bits": int(bits),
        "lut_include_zero": bool(lut_include_zero),
        "post_gmm_rescale": bool(post_gmm_rescale),
        "packed_cols": int(packed.shape[1]),
        "scale_rep": scale_rep,
        "scale_rank": int(scale_rank),
        "effective_rank": int(eff_rank),
        "max_rank": int(max_rank),
        "scales_dtype": "fp16" if store_scales_fp16 else "fp32",
    }
    return dequant, saved, meta


def quant_dequant_groupwise_lut(
    weight: torch.Tensor,
    *,
    group_size: int,
    lut_size: int,
    scale_rank: int = 0,
    eps: float = 1e-8,
    lut_include_zero: bool = True,
    post_gmm_rescale: bool = False,
) -> Tuple[torch.Tensor, QuantStats]:
    """
    Quantize+dequantize a 2D weight matrix with per-(out, group) scaling and a uniform LUT.

    - Groups are formed along the input-feature dimension (dim=1).
    - Each group uses its own max-abs scale.
    - Values are normalized to [-1, 1], snapped to nearest LUT entry, then de-normalized.
    - Optionally approximates the scales matrix with a low-rank SVD (`scale_rank`).
    """
    if weight.ndim != 2:
        raise ValueError(f"Expected 2D weight, got shape={tuple(weight.shape)}")
    if group_size <= 0:
        raise ValueError("group_size must be > 0")
    if lut_size < 2:
        raise ValueError("lut_size must be >= 2")

    orig_dtype = weight.dtype
    w = weight.float()
    out_features, in_features = w.shape

    pad = (-in_features) % group_size
    if pad:
        w = torch.nn.functional.pad(w, (0, pad))
    padded_in = w.shape[1]
    num_groups = padded_in // group_size

    grouped = w.view(out_features, num_groups, group_size)
    lut = make_lut(lut_size=lut_size, device=w.device, dtype=w.dtype, include_zero=lut_include_zero)
    scales_full = grouped.abs().amax(dim=2).clamp_min(eps)  # [out, groups]

    max_rank = min(out_features, num_groups)
    eff_rank = 0 if scale_rank <= 0 else min(scale_rank, max_rank)
    use_ab = eff_rank > 0 and eff_rank < max_rank

    if use_ab:
        if post_gmm_rescale:
            # "Post" pass: use the *reconstructed* scale representation (as it would be stored: FP16 A,B),
            # then re-quantize based on that reconstructed scale (indices can change).
            a32, b32 = _svd_ab(scales_full, eff_rank)
            a = a32.to(torch.float16)
            b = b32.to(torch.float16)
            scales_hat = (a.float() @ b.float()).clamp_min(eps)
        else:
            # Default: low-rank scales as a float matrix (faster and typically higher fidelity than FP16 A,B storage).
            scales_hat = _low_rank_approx(scales_full, eff_rank).clamp_min(eps)
        normalized = grouped / scales_hat.unsqueeze(-1)
        indices = quantize_to_lut_indices(normalized, lut_size=lut_size, include_zero=lut_include_zero).to(torch.int64)
        dequant = lut[indices] * scales_hat.unsqueeze(-1)
    else:
        scales_hat = scales_full
        normalized = grouped / scales_hat.unsqueeze(-1)
        indices = quantize_to_lut_indices(normalized, lut_size=lut_size, include_zero=lut_include_zero).to(torch.int64)
        dequant = lut[indices] * scales_hat.unsqueeze(-1)

    dequant = dequant.view(out_features, padded_in)
    if pad:
        dequant = dequant[:, :in_features]
    dequant = dequant.contiguous()

    err = (weight.float() - dequant).float()
    mae = err.abs().mean().item()
    mse = (err * err).mean().item()
    rmse = math.sqrt(mse)
    weight_l2 = weight.float().norm().clamp_min(eps).item()
    err_l2 = err.norm().item()
    rel_l2 = err_l2 / weight_l2
    return dequant.to(orig_dtype), QuantStats(mae=mae, mse=mse, rmse=rmse, rel_l2=rel_l2, err_l2=err_l2, weight_l2=weight_l2)


def iter_weight_params(
    model: torch.nn.Module,
    *,
    name_regex: str,
) -> Iterable[Tuple[str, torch.nn.Parameter]]:
    pattern = re.compile(name_regex)
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim != 2:
            continue
        if pattern.search(name) is None:
            continue
        yield name, param


def _format_stats(stats: QuantStats) -> str:
    return f"mae={stats.mae:.6g} rmse={stats.rmse:.6g} rel_l2={stats.rel_l2:.6g}"


@torch.no_grad()
def run_inference(
    model,
    tokenizer,
    *,
    device: torch.device,
    prompt: str,
    system_prompt: str,
    prompt_format: str,
    strip_think: bool,
    max_new_tokens: int,
) -> str:
    prompt_format = prompt_format.lower()
    if prompt_format == "auto":
        if hasattr(tokenizer, "apply_chat_template"):
            prompt_format = "chat"
        else:
            prompt_format = "raw"

    if prompt_format == "raw":
        prompt_text = prompt
    elif prompt_format in {"chat", "chat_think"}:
        if not hasattr(tokenizer, "apply_chat_template"):
            raise ValueError("Tokenizer has no chat template; use --prompt-format raw")
        messages = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        elif prompt_format == "chat_think":
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "You are Qwen, a helpful assistant.\n"
                        "First think through the problem in <think>...</think>, then give a concise final answer."
                    ),
                }
            )
        messages.append({"role": "user", "content": prompt})
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if prompt_format == "chat_think":
            prompt_text = prompt_text + "<think>\n"
    else:
        raise ValueError(f"Unknown prompt_format={prompt_format!r}")

    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )[0]
    gen_ids = output_ids[inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    if strip_think:
        if "</think>" in text:
            text = text.split("</think>", 1)[1].strip()
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Groupwise LUT quant/dequant simulation on Qwen3 MLP weights")
    parser.add_argument("--model-id", default="Qwen/Qwen3-0.6B", help="HF model id or local path")
    parser.add_argument("--device", default="auto", help="auto|cpu|cuda|mps")
    parser.add_argument(
        "--inference-device",
        default="same",
        help="same|auto|cpu|cuda|mps (use 'cpu' as a workaround if MPS generation aborts)",
    )
    parser.add_argument("--dtype", default="fp16", help="fp16|bf16|fp32")
    parser.add_argument(
        "--attn-implementation",
        default="auto",
        choices=["auto", "eager", "sdpa", "flash_attention_2"],
        help="Attention backend; on MPS, prefer 'eager' to avoid MPSGraph matmul shape issues with GQA",
    )
    parser.add_argument("--local-files-only", action="store_true", help="Avoid network access for HF downloads")
    parser.add_argument("--group-size", type=int, default=128, help="Group size along in_features")
    parser.add_argument("--attn-group-size", type=int, default=0, help="Attention group size (0 => use --group-size)")
    parser.add_argument("--lut-size", type=int, default=16, help="Number of LUT entries (>=2)")
    parser.add_argument(
        "--lut-include-zero",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether LUT includes 0 (default: off). If enabled with even lut_size, LUT becomes non-uniform but still fast-mapped.",
    )
    parser.add_argument(
        "--scale-rank",
        type=int,
        default=0,
        help="If >0, low-rank SVD rank for the per-(out,group) scales matrix (use small values like 1/2/4; if >= num_groups it becomes full-rank/no-op)",
    )
    parser.add_argument(
        "--gmm-stats",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print per-tensor and per-layer storage stats for the scale low-rank factors (\"GMM vectors\")",
    )
    parser.add_argument(
        "--eff-bits",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Estimate effective bits/weight for quantized tensors (indices + scales + LUT) and print summary",
    )
    parser.add_argument(
        "--save-quantized-dir",
        default="",
        help="If set, write a GGUF-like quant bundle (quantized.safetensors + quantized.json + tokenizer/config) to this directory",
    )
    parser.add_argument(
        "--name-regex",
        default=r"\.mlp\.(gate_proj|up_proj|down_proj)\.weight$",
        help="Regex to select which weight tensors to quantize",
    )
    parser.add_argument(
        "--attn-name-regex",
        default=r"\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.weight$",
        help="Regex for attention weights (used by --quantize-attn / --attn-only)",
    )
    parser.add_argument("--quantize-attn", action="store_true", help="Also quantize attention projection weights")
    parser.add_argument("--attn-only", action="store_true", help="Quantize only attention projection weights")
    parser.add_argument(
        "--attn-lut-size",
        type=int,
        default=0,
        help="Attention LUT size (0 => use --lut-size)",
    )
    parser.add_argument(
        "--attn-lut-include-zero",
        default="inherit",
        choices=["inherit", "true", "false"],
        help="Attention LUT include-zero behavior: inherit|true|false",
    )
    parser.add_argument(
        "--post-gmm-rescale",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If using low-rank scales, re-quantize based on reconstructed low-rank scales (indices may change; matches saved FP16 A,B more closely)",
    )
    parser.add_argument(
        "--use-ab-scales",
        action="store_true",
        help="Alias for --post-gmm-rescale (use reconstructed FP16 A,B scales rather than float low-rank approximation)",
    )
    parser.add_argument(
        "--attn-scale-rank",
        type=int,
        default=-1,
        help="Attention scale SVD rank (-1 => use --scale-rank)",
    )
    parser.add_argument("--limit", type=int, default=0, help="If >0, quantize only first N matched tensors")
    parser.add_argument("--prompt", default="What is the capital of France?", help="Sanity-check prompt")
    parser.add_argument("--system-prompt", default="", help="Optional system prompt (chat formats only)")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--prompt-format",
        default="chat",
        choices=["raw", "chat", "chat_think"],
        help="Prompt formatting: raw (no template), chat template, or chat template + '<think>' prefix (Qwen thinking)",
    )
    parser.add_argument(
        "--strip-think",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove '<think>...</think>' from printed output (default: on)",
    )
    parser.add_argument("--use-chat-template", action="store_true", help="Deprecated: same as --prompt-format chat")
    parser.add_argument("--skip-inference", action="store_true", help="Only compute reconstruction errors (no generate)")
    parser.add_argument("--run-baseline", action="store_true", help="Run inference before quantization as well")
    args = parser.parse_args()
    if args.use_chat_template:
        args.prompt_format = "chat"
    if args.use_ab_scales:
        args.post_gmm_rescale = True
    if args.prompt_format == "chat_think" and args.max_new_tokens < 96 and not args.skip_inference:
        print("Note: --prompt-format chat_think often needs a larger --max-new-tokens (e.g. 128) to reach the final answer.")

    device = _pick_device(args.device)
    inference_device = device if args.inference_device == "same" else _pick_device(args.inference_device)
    dtype = _pick_dtype(args.dtype)
    attn_impl = args.attn_implementation
    if attn_impl == "auto" and device.type == "mps":
        attn_impl = "eager"

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, local_files_only=args.local_files_only, trust_remote_code=True)
    model_kwargs = dict(
        local_files_only=args.local_files_only,
        dtype=dtype,
        trust_remote_code=True,
    )
    if attn_impl != "auto":
        model_kwargs["attn_implementation"] = attn_impl
    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
    model.eval()
    model.to(device)
    load_s = time.time() - t0
    print(f"Loaded model={args.model_id} device={device.type} dtype={dtype} attn={attn_impl} in {load_s:.2f}s")

    if args.run_baseline and not args.skip_inference:
        if inference_device != device:
            model.to(inference_device)
        t_inf = time.time()
        baseline = run_inference(
            model,
            tokenizer,
            device=inference_device,
            prompt=args.prompt,
            system_prompt=args.system_prompt,
            prompt_format=args.prompt_format,
            strip_think=args.strip_think,
            max_new_tokens=args.max_new_tokens,
        )
        print(f"[baseline] {baseline} ({time.time() - t_inf:.2f}s)")
        if inference_device != device:
            model.to(device)

    if args.attn_only and args.quantize_attn:
        raise SystemExit("Use only one of --attn-only and --quantize-attn.")

    if args.attn_only:
        effective_name_regex = args.attn_name_regex
    elif args.quantize_attn:
        effective_name_regex = f"(?:{args.name_regex})|(?:{args.attn_name_regex})"
    else:
        effective_name_regex = args.name_regex

    names_and_params = list(iter_weight_params(model, name_regex=effective_name_regex))
    if args.limit and args.limit > 0:
        names_and_params = names_and_params[: args.limit]
    if not names_and_params:
        raise SystemExit(f"No tensors matched name_regex={effective_name_regex!r}")

    effective_attn_lut_size = args.lut_size if args.attn_lut_size in (0, None) else args.attn_lut_size
    effective_attn_scale_rank = args.scale_rank if args.attn_scale_rank in (-1, None) else args.attn_scale_rank
    effective_attn_group_size = args.group_size if args.attn_group_size in (0, None) else args.attn_group_size
    effective_attn_lut_include_zero = args.lut_include_zero
    if args.attn_lut_include_zero == "true":
        effective_attn_lut_include_zero = True
    elif args.attn_lut_include_zero == "false":
        effective_attn_lut_include_zero = False
    save_quant_dir = args.save_quantized_dir.strip()
    do_save_quant = bool(save_quant_dir)

    if (args.scale_rank > 0 or effective_attn_scale_rank > 0) and not args.gmm_stats:
        print("Note: pass --gmm-stats to print per-layer scale low-rank (A,B) size estimates.")

    # If scale-rank >= num_groups, the SVD is full-rank/no-op. Make it explicit since it's easy to misinterpret.
    group_counts: DefaultDict[str, int] = defaultdict(int)
    for name, p in names_and_params:
        in_features = int(p.shape[1])
        kind = "attn" if ".self_attn." in name else "mlp" if ".mlp." in name else "other"
        group_size = effective_attn_group_size if kind == "attn" else args.group_size
        pad = (-in_features) % group_size
        num_groups = (in_features + pad) // group_size
        group_counts[kind] = max(group_counts[kind], int(num_groups))

    if args.scale_rank > 0 and group_counts.get("mlp", 0) and args.scale_rank >= group_counts["mlp"]:
        print(
            f"Note: --scale-rank {args.scale_rank} >= max MLP num_groups {group_counts['mlp']}, "
            "so MLP scale SVD is full-rank (no effect). Try smaller (e.g. 1/2/4)."
        )
    if effective_attn_scale_rank > 0 and group_counts.get("attn", 0) and effective_attn_scale_rank >= group_counts["attn"]:
        print(
            f"Note: --attn-scale-rank {effective_attn_scale_rank} >= max attn num_groups {group_counts['attn']}, "
            "so attention scale SVD is full-rank (no effect). Try smaller (e.g. 1/2/4)."
        )

    print(
        f"Quantizing {len(names_and_params)} tensors: "
        f"mlp(group={args.group_size},lut={args.lut_size},scale_rank={args.scale_rank}) "
        f"attn(group={effective_attn_group_size},lut={effective_attn_lut_size},scale_rank={effective_attn_scale_rank})"
    )

    per_tensor: Dict[str, QuantStats] = {}
    gmm_total_bytes = 0
    scales_full_total_bytes = 0
    total_mae = 0.0
    total_mse = 0.0
    total_elems = 0
    total_err_l2_sq = 0.0
    total_weight_l2_sq = 0.0

    # Category stats: MLP vs attention.
    kind_elems: DefaultDict[str, int] = defaultdict(int)
    kind_mae_sum: DefaultDict[str, float] = defaultdict(float)
    kind_mse_sum: DefaultDict[str, float] = defaultdict(float)
    kind_err_l2_sq: DefaultDict[str, float] = defaultdict(float)
    kind_weight_l2_sq: DefaultDict[str, float] = defaultdict(float)
    kind_gmm_bytes: DefaultDict[str, int] = defaultdict(int)
    kind_scales_bytes: DefaultDict[str, int] = defaultdict(int)
    kind_storage_bits: DefaultDict[str, int] = defaultdict(int)
    kind_index_bits: DefaultDict[str, int] = defaultdict(int)
    kind_scale_bits: DefaultDict[str, int] = defaultdict(int)
    kind_lut_bits: DefaultDict[str, int] = defaultdict(int)
    kind_weights_elems: DefaultDict[str, int] = defaultdict(int)
    kind_weights_elems_padded: DefaultDict[str, int] = defaultdict(int)

    total_storage_bits = 0
    total_index_bits = 0
    total_scale_bits = 0
    total_lut_bits = 0
    total_weights_elems = 0
    total_weights_elems_padded = 0

    # Per-layer size stats printed incrementally (avoid needing a full end-of-run dump).
    current_layer_key: str = ""
    layer_scales_bytes: DefaultDict[str, int] = defaultdict(int)
    layer_gmm_bytes: DefaultDict[str, int] = defaultdict(int)

    save_tensors: Dict[str, torch.Tensor] = {}
    save_meta: Dict[str, object] = {
        "format": "anemll-groupwise-lut-quant",
        "format_version": 1,
        "model_id": args.model_id,
        "group_size": args.group_size,
        "mlp": {
            "group_size": args.group_size,
            "lut_size": args.lut_size,
            "lut_include_zero": bool(args.lut_include_zero),
            "scale_rank": args.scale_rank,
        },
        "attn": {
            "group_size": effective_attn_group_size,
            "lut_size": effective_attn_lut_size,
            "lut_include_zero": bool(effective_attn_lut_include_zero),
            "scale_rank": effective_attn_scale_rank,
        },
        "post_gmm_rescale": bool(args.post_gmm_rescale),
        "tensors": {},
    }

    t_q = time.time()
    for i, (name, param) in enumerate(names_and_params, start=1):
        kind = "attn" if ".self_attn." in name else "mlp" if ".mlp." in name else "other"
        lut_size = effective_attn_lut_size if kind == "attn" else args.lut_size
        lut_include_zero = effective_attn_lut_include_zero if kind == "attn" else args.lut_include_zero
        scale_rank = effective_attn_scale_rank if kind == "attn" else args.scale_rank
        group_size = effective_attn_group_size if kind == "attn" else args.group_size

        if args.eff_bits:
            out_features = int(param.shape[0])
            in_features = int(param.shape[1])
            est = estimate_storage_bits(
                out_features=out_features,
                in_features=in_features,
                group_size=group_size,
                lut_size=lut_size,
                scale_rank=scale_rank,
                assume_fp16_scales=True,
                assume_fp16_lut=True,
            )
            kind_storage_bits[kind] += est.total_bits
            kind_index_bits[kind] += est.index_bits_total
            kind_scale_bits[kind] += est.scales_bytes * 8
            kind_lut_bits[kind] += est.lut_bytes * 8
            kind_weights_elems[kind] += est.weights_elems
            kind_weights_elems_padded[kind] += est.weights_elems_padded

            total_storage_bits += est.total_bits
            total_index_bits += est.index_bits_total
            total_scale_bits += est.scales_bytes * 8
            total_lut_bits += est.lut_bytes * 8
            total_weights_elems += est.weights_elems
            total_weights_elems_padded += est.weights_elems_padded

        if args.gmm_stats:
            # Estimate storage for scale representation using the *original* tensor shape.
            out_features = int(param.shape[0])
            in_features = int(param.shape[1])
            gmm = compute_gmm_size_stats(
                out_features=out_features,
                in_features=in_features,
                group_size=group_size,
                scale_rank=scale_rank,
                storage_dtype=dtype,
            )
            layer_match = re.search(r"\.layers\.(\d+)\.", name)
            layer_key = f"layer_{layer_match.group(1)}" if layer_match else "layer_other"
            if current_layer_key and layer_key != current_layer_key:
                for k in ("mlp", "attn", "other"):
                    if layer_scales_bytes.get(k, 0) or layer_gmm_bytes.get(k, 0):
                        print(
                            f"{current_layer_key}.{k}: full_scales={layer_scales_bytes[k]/1024:.1f}KiB "
                            f"gmm(A,B)={layer_gmm_bytes[k]/1024:.1f}KiB"
                        )
                layer_scales_bytes = defaultdict(int)
                layer_gmm_bytes = defaultdict(int)
            current_layer_key = layer_key

            layer_gmm_bytes[kind] += gmm.ab_bytes
            layer_scales_bytes[kind] += gmm.full_scales_bytes
            kind_gmm_bytes[kind] += gmm.ab_bytes
            kind_scales_bytes[kind] += gmm.full_scales_bytes
            gmm_total_bytes += gmm.ab_bytes
            scales_full_total_bytes += gmm.full_scales_bytes

        if do_save_quant:
            dequant, payload_tensors, payload_meta = quantize_to_payload(
                param.data,
                group_size=group_size,
                lut_size=lut_size,
                scale_rank=scale_rank,
                store_scales_fp16=True,
                lut_include_zero=lut_include_zero,
                post_gmm_rescale=bool(args.post_gmm_rescale),
            )
            err = (param.data.float() - dequant.float()).float()
            mae = err.abs().mean().item()
            mse = (err * err).mean().item()
            rmse = math.sqrt(mse)
            weight_l2 = param.data.float().norm().clamp_min(1e-8).item()
            err_l2 = err.norm().item()
            rel_l2 = err_l2 / weight_l2
            stats = QuantStats(mae=mae, mse=mse, rmse=rmse, rel_l2=rel_l2, err_l2=err_l2, weight_l2=weight_l2)

            prefix = f"{name}"
            save_tensors[f"{prefix}.qbytes"] = payload_tensors["qbytes"].contiguous().cpu()
            if "scales" in payload_tensors:
                save_tensors[f"{prefix}.scales"] = payload_tensors["scales"].contiguous().cpu()
            else:
                save_tensors[f"{prefix}.A"] = payload_tensors["A"].contiguous().cpu()
                save_tensors[f"{prefix}.B"] = payload_tensors["B"].contiguous().cpu()
            save_meta["tensors"][name] = {"kind": kind, **payload_meta}
        else:
            dequant, stats = quant_dequant_groupwise_lut(
                param.data,
                group_size=group_size,
                lut_size=lut_size,
                scale_rank=scale_rank,
                lut_include_zero=lut_include_zero,
                post_gmm_rescale=bool(args.post_gmm_rescale),
            )

        param.data.copy_(dequant)
        per_tensor[name] = stats
        elems = param.numel()
        total_elems += elems
        total_mae += stats.mae * elems
        total_mse += stats.mse * elems
        total_err_l2_sq += stats.err_l2 * stats.err_l2
        total_weight_l2_sq += stats.weight_l2 * stats.weight_l2

        kind_elems[kind] += elems
        kind_mae_sum[kind] += stats.mae * elems
        kind_mse_sum[kind] += stats.mse * elems
        kind_err_l2_sq[kind] += stats.err_l2 * stats.err_l2
        kind_weight_l2_sq[kind] += stats.weight_l2 * stats.weight_l2

        line = f"[{i:3d}/{len(names_and_params):3d}] {name} {tuple(param.shape)} {_format_stats(stats)}"
        if args.gmm_stats:
            line += (
                f" | scales={gmm.full_scales_bytes/1024:.1f}KiB"
                f" gmm(A,B)={gmm.ab_bytes/1024:.1f}KiB"
                f" r={gmm.effective_rank}/{gmm.max_rank} groups={gmm.num_groups}"
                f" kind={kind}"
            )
        print(line)

    quant_s = time.time() - t_q
    overall_mae = total_mae / max(1, total_elems)
    overall_mse = total_mse / max(1, total_elems)
    overall_rmse = math.sqrt(overall_mse)
    overall_rel_l2 = math.sqrt(total_err_l2_sq) / max(1e-12, math.sqrt(total_weight_l2_sq))
    print(f"Overall: mae={overall_mae:.6g} rmse={overall_rmse:.6g} rel_l2={overall_rel_l2:.6g} (quant+dequant time {quant_s:.2f}s)")

    if args.eff_bits:
        # Printed before inference as requested.
        denom = max(1, total_weights_elems)
        eff_bpw = total_storage_bits / denom
        total_index_bytes = total_index_bits / 8.0
        total_scale_bytes = total_scale_bits / 8.0
        total_lut_bytes = total_lut_bits / 8.0
        total_bytes = total_storage_bits / 8.0
        print(
            f"Storage estimate: total={total_bytes/1024/1024:.3f}MiB "
            f"(indices={total_index_bytes/1024/1024:.3f}MiB scales={total_scale_bytes/1024/1024:.3f}MiB lut={total_lut_bytes/1024:.1f}KiB)"
        )
        print(
            "---------------------------------------------------\n"
            "Effective bits/weight (assume FP16 scales + FP16 LUT, LUT counted per tensor): "
            f"total={eff_bpw:.3f}b "
            f"(indices={total_index_bits/denom:.3f}b scales={total_scale_bits/denom:.3f}b lut={total_lut_bits/denom:.3f}b, "
            f"pad_overhead={(total_weights_elems_padded-total_weights_elems)/denom*100.0:.2f}%)"
        )
        for k in ("mlp", "attn", "other"):
            if kind_weights_elems.get(k, 0) == 0:
                continue
            k_denom = max(1, kind_weights_elems[k])
            k_eff = kind_storage_bits[k] / k_denom
            k_pad = (kind_weights_elems_padded[k] - kind_weights_elems[k]) / k_denom * 100.0
            k_bytes = kind_storage_bits[k] / 8.0
            k_index_bytes = kind_index_bits[k] / 8.0
            k_scale_bytes = kind_scale_bits[k] / 8.0
            k_lut_bytes = kind_lut_bits[k] / 8.0
            print(
                f"{k} storage: total={k_bytes/1024/1024:.3f}MiB "
                f"(indices={k_index_bytes/1024/1024:.3f}MiB scales={k_scale_bytes/1024/1024:.3f}MiB lut={k_lut_bytes/1024:.1f}KiB)"
            )
            print(
                f"{k} bits/weight: {k_eff:.3f}b "
                f"(indices={kind_index_bits[k]/k_denom:.3f}b scales={kind_scale_bits[k]/k_denom:.3f}b lut={kind_lut_bits[k]/k_denom:.3f}b, "
                f"pad_overhead={k_pad:.2f}%)"
            )

    if do_save_quant:
        out_dir = Path(save_quant_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if group_counts.get("mlp", 0) > 0:
            save_tensors["__lut__.mlp"] = make_lut(
                lut_size=args.lut_size, device=torch.device("cpu"), dtype=torch.float16, include_zero=bool(args.lut_include_zero)
            )
        if group_counts.get("attn", 0) > 0:
            save_tensors["__lut__.attn"] = make_lut(
                lut_size=effective_attn_lut_size,
                device=torch.device("cpu"),
                dtype=torch.float16,
                include_zero=bool(effective_attn_lut_include_zero),
            )

        meta_path = out_dir / "quantized.json"
        safetensors_path = out_dir / "quantized.safetensors"
        with meta_path.open("w") as f:
            json.dump(save_meta, f, indent=2, sort_keys=True)
        safetensors_save_file(save_tensors, str(safetensors_path), metadata={"format": str(save_meta["format"]), "version": "1"})

        try:
            tokenizer.save_pretrained(str(out_dir / "tokenizer"))
        except Exception as e:
            print(f"Warning: failed to save tokenizer: {e}")
        try:
            model.config.save_pretrained(str(out_dir / "model_config"))
        except Exception as e:
            print(f"Warning: failed to save model config: {e}")
        print(f"Saved quant bundle to {out_dir} ({safetensors_path.name}, {meta_path.name})")

    if args.gmm_stats:
        if current_layer_key:
            for k in ("mlp", "attn", "other"):
                if layer_scales_bytes.get(k, 0) or layer_gmm_bytes.get(k, 0):
                    print(
                        f"{current_layer_key}.{k}: full_scales={layer_scales_bytes[k]/1024:.1f}KiB "
                        f"gmm(A,B)={layer_gmm_bytes[k]/1024:.1f}KiB"
                    )
        print(
            f"Scale storage total: full_scales={scales_full_total_bytes/1024/1024:.2f}MiB "
            f"gmm(A,B)={gmm_total_bytes/1024/1024:.2f}MiB (dtype={dtype})"
        )

        if kind_scales_bytes:
            for k in ("mlp", "attn", "other"):
                if kind_elems.get(k, 0) == 0:
                    continue
                print(
                    f"{k} storage: full_scales={kind_scales_bytes[k]/1024/1024:.2f}MiB gmm(A,B)={kind_gmm_bytes[k]/1024/1024:.2f}MiB"
                )

    for k in ("mlp", "attn", "other"):
        if kind_elems.get(k, 0) == 0:
            continue
        k_mae = kind_mae_sum[k] / max(1, kind_elems[k])
        k_rmse = math.sqrt(kind_mse_sum[k] / max(1, kind_elems[k]))
        k_rel = math.sqrt(kind_err_l2_sq[k]) / max(1e-12, math.sqrt(kind_weight_l2_sq[k]))
        print(f"{k} error: mae={k_mae:.6g} rmse={k_rmse:.6g} rel_l2={k_rel:.6g} (elems={kind_elems[k]})")

    if not args.skip_inference:
        if inference_device != device:
            model.to(inference_device)
        t_inf = time.time()
        answer = run_inference(
            model,
            tokenizer,
            device=inference_device,
            prompt=args.prompt,
            system_prompt=args.system_prompt,
            prompt_format=args.prompt_format,
            strip_think=args.strip_think,
            max_new_tokens=args.max_new_tokens,
        )
        print(f"[quantized] {answer} ({time.time() - t_inf:.2f}s)")


if __name__ == "__main__":
    main()
