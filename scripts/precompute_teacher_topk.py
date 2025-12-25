#!/usr/bin/env python3
"""
Precompute teacher top-k logits for KD-QAT / KD-LoRA.

This script builds a cache that allows training WITHOUT:
  - running the teacher model each step
  - computing full-vocab logits for the student each step

Instead, training can:
  - run the student transformer to get hidden states
  - compute logits only on the cached candidate token ids (top-k [+ random negatives])
  - compute KL on that candidate set

Cache format (output_dir):
  meta.json
  shard_00000.pt, shard_00001.pt, ...

Each shard contains:
  - input_ids: [N, L] int32
  - attention_mask: [N, L] uint8
  - topk_idx: [N, L-1, K] int32
  - topk_logits: [N, L-1, K] float16  (raw teacher logits, not divided by T)

Important notes:
- We "pack" raw text into contiguous token blocks of length L to maximize token efficiency.
- The cache stores teacher logits for predicting the NEXT token (positions 0..L-2).

Examples:
python scripts/precompute_teacher_topk.py \
  --teacher_model_name_or_path Qwen/Qwen3-0.6B \
  --dataset_name allenai/c4 \
  --dataset_config_name en \
  --dataset_split train \
  --dataset_text_field text \
  --streaming \
  --shuffle_buffer 10000 \
  --max_length 64 \
  --topk 32 \
  --num_sequences 20000 \
  --batch_size 1 \
  --shard_size 1024 \
  --device mps \
  --dtype bf16 \
  --output_dir caches/c4_qwen3_topk_L64_K32

python scripts/precompute_teacher_topk.py \
  --teacher_model_name_or_path Qwen/Qwen3-0.6B \
  --dataset_name tatsu-lab/alpaca \
  --dataset_split train \
  --dataset_format alpaca_chat \
  --enable_thinking both \
  --max_length 128 \
  --topk 32 \
  --rand_neg 256 \
  --num_sequences 20000 \
  --batch_size 1 \
  --shard_size 512 \
  --device mps \
  --dtype bf16 \
  --output_dir caches/alpaca_chat_think_L128_K32_R256
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Iterator, List, Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import __version__ as transformers_version

# Ensure local package imports work without installation (for mixed_precision helpers).
sys.path.append(str(Path(__file__).resolve().parent.parent))

try:
    from qat_lora.mixed_precision import pick_device, resolve_param_dtype
except Exception:
    pick_device = None
    resolve_param_dtype = None

try:
    from qat_lora.data import build_alpaca_messages
except Exception:
    build_alpaca_messages = None


def _call_from_pretrained(fn, model_name_or_path: str, **kwargs):
    """
    Call Hugging Face `from_pretrained` with best-effort compatibility across versions.
    """
    token = kwargs.pop("token", None)

    def _call(call_kwargs: dict):
        if token is None:
            return fn(model_name_or_path, **call_kwargs)
        try:
            return fn(model_name_or_path, token=token, **call_kwargs)
        except TypeError:
            return fn(model_name_or_path, use_auth_token=token, **call_kwargs)

    try:
        return _call(dict(kwargs))
    except TypeError as e:
        # Some versions don't accept resume_download=...
        if "resume_download" in str(e) and "resume_download" in kwargs:
            call_kwargs = dict(kwargs)
            call_kwargs.pop("resume_download", None)
            return _call(call_kwargs)
        raise


def _hf_common_kwargs(args):
    kw = {}
    if getattr(args, "cache_dir", None):
        kw["cache_dir"] = args.cache_dir
    if getattr(args, "local_files_only", False):
        kw["local_files_only"] = True
    if getattr(args, "hf_token", None):
        kw["token"] = args.hf_token
    if getattr(args, "resume_download", False):
        # transformers forwards this to the hub; ignore if unsupported.
        kw["resume_download"] = True
    return kw


def _from_pretrained_fp32(model_name_or_path: str, *, hf_kwargs: Optional[dict] = None):
    """
    Transformers is deprecating torch_dtype= in favor of dtype= in some versions.
    Support both to avoid warnings/breakage across installs.
    """
    hf_kwargs = dict(hf_kwargs or {})
    try:
        return _call_from_pretrained(AutoModelForCausalLM.from_pretrained, model_name_or_path, dtype=torch.float32, **hf_kwargs)
    except TypeError:
        hf_kwargs.pop("resume_download", None)
        return _call_from_pretrained(
            AutoModelForCausalLM.from_pretrained, model_name_or_path, torch_dtype=torch.float32, **hf_kwargs
        )


def _pick_device_fallback(device: str) -> str:
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device


def _resolve_dtype_fallback(dtype: str, device: str) -> torch.dtype:
    if dtype in ("auto", None):
        # Prefer bf16 on cuda/mps if available; else fp16 on cuda; else fp32.
        if device in ("cuda", "mps"):
            return torch.bfloat16
        return torch.float32
    if dtype == "fp32":
        return torch.float32
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    raise ValueError(f"Unknown dtype: {dtype}")


def _chat_text_variants(tokenizer, messages: List[dict], thinking_mode: str) -> Iterator[str]:
    mode = (thinking_mode or "false").lower()
    if mode not in ("false", "true", "both"):
        raise ValueError(f"Unknown thinking mode: {thinking_mode}")
    flags = [False, True] if mode == "both" else [mode == "true"]
    for enable in flags:
        yield tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=enable,
        )


def _iter_text_samples(dataset_iter: Iterator[dict], args, tokenizer) -> Iterator[str]:
    fmt = args.dataset_format
    if fmt == "plain":
        field = args.dataset_text_field
        for ex in dataset_iter:
            txt = ex.get(field, None)
            if not isinstance(txt, str):
                continue
            txt = txt.strip()
            if not txt:
                continue
            yield txt
    elif fmt in ("alpaca", "alpaca_chat"):
        if build_alpaca_messages is None:
            raise RuntimeError("dataset_format=alpaca requires qat_lora.data to be importable.")
        for ex in dataset_iter:
            messages = build_alpaca_messages(ex)
            if not messages:
                continue
            for rendered in _chat_text_variants(tokenizer, messages, args.enable_thinking):
                if rendered:
                    yield rendered
    else:
        raise ValueError(f"Unknown dataset_format={fmt}")


class PackedTokenStream:
    """
    Packs text samples into fixed-length token blocks.
    """

    def __init__(
        self,
        text_iter: Iterator[str],
        tokenizer,
        max_length: int,
        add_eos: bool = True,
    ):
        self.text_iter = text_iter
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_eos = add_eos
        self._buf: List[int] = []

    def __iter__(self):
        eos = self.tokenizer.eos_token_id
        for txt in self.text_iter:
            ids = self.tokenizer(txt, add_special_tokens=False).input_ids
            if not ids:
                continue
            self._buf.extend(ids)
            if self.add_eos and eos is not None:
                self._buf.append(eos)

            # Yield as many full blocks as available.
            while len(self._buf) >= self.max_length:
                block = self._buf[: self.max_length]
                self._buf = self._buf[self.max_length :]
                yield block


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher_model_name_or_path", type=str, default="Qwen/Qwen3-0.6B")
    p.add_argument("--dataset_name", type=str, required=True)
    p.add_argument("--dataset_config_name", type=str, default=None)
    p.add_argument("--dataset_split", type=str, default="train")
    p.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Hugging Face cache directory (models/tokenizers/datasets). Useful on Colab to point at Drive.",
    )
    p.add_argument("--hf_token", type=str, default=None, help="HF token for gated models (or set HF_TOKEN env var).")
    p.add_argument("--local_files_only", action="store_true", help="Do not attempt to download anything from the Hub.")
    p.add_argument(
        "--resume_download",
        action="store_true",
        help="Resume partial Hugging Face downloads when possible (best-effort, depends on hub/transformers version).",
    )
    p.add_argument(
        "--dataset_format",
        type=str,
        default="plain",
        choices=["plain", "alpaca", "alpaca_chat"],
        help="plain=use dataset_text_field contents. alpaca/alpaca_chat render a chat template.",
    )
    p.add_argument("--dataset_text_field", type=str, default="text", help="Used when dataset_format=plain.")
    p.add_argument("--streaming", action="store_true")
    p.add_argument("--shuffle_buffer", type=int, default=0, help="If streaming, shuffle with this buffer size.")
    p.add_argument(
        "--enable_thinking",
        type=str,
        default="false",
        choices=["false", "true", "both"],
        help="When dataset_format uses chat templates, control whether thinking is disabled, enabled, or both per sample.",
    )

    p.add_argument("--max_length", type=int, default=64)
    p.add_argument("--topk", type=int, default=32)
    p.add_argument("--rand_neg", type=int, default=0,
                   help="If >0, also cache this many random negative token ids per position, plus their teacher logits.")
    p.add_argument("--num_sequences", type=int, default=20000, help="How many packed sequences to cache.")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--shard_size", type=int, default=1024, help="How many sequences per shard_*.pt.")

    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--dtype", type=str, default="auto", choices=["auto", "fp32", "bf16", "fp16"])

    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume writing into an existing output_dir (uses output_dir/progress.json).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing shard_*.pt files in output_dir.",
    )

    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    if args.hf_token is None:
        args.hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    progress_path = out / "progress.json"

    existing_shards = sorted(out.glob("shard_*.pt"))
    if args.resume and existing_shards and not progress_path.exists():
        raise RuntimeError(
            f"--resume requires {progress_path.name} when shard files already exist in: {out} "
            "(use --overwrite to start over)."
        )
    if existing_shards and not args.resume and not args.overwrite:
        print(
            f"[warn] output_dir already has {len(existing_shards)} shard files; "
            "this run will overwrite them (use --resume to continue, or --overwrite to silence)."
        )

    if tuple(map(int, transformers_version.split(".")[:2])) < (4, 51):
        # Not strictly required for precompute, but keeps versions aligned with Qwen3.
        print(
            f"[warn] transformers={transformers_version}. Qwen3 recommends transformers>=4.51.0. "
            "If you hit tokenizer/model errors, upgrade transformers."
        )

    if pick_device is not None:
        device = pick_device(args.device)
    else:
        device = _pick_device_fallback(args.device)

    if resolve_param_dtype is not None:
        param_dtype = resolve_param_dtype(args.dtype, device)
    else:
        param_dtype = _resolve_dtype_fallback(args.dtype, device)

    print(f"[device] {device} | dtype={param_dtype}")

    hf_kwargs = _hf_common_kwargs(args)
    try:
        tokenizer = _call_from_pretrained(AutoTokenizer.from_pretrained, args.teacher_model_name_or_path, **hf_kwargs)
    except KeyboardInterrupt:
        print("[interrupt] tokenizer download/load interrupted; re-run to resume/continue.")
        raise
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load teacher on CPU fp32 first, then cast.
    try:
        teacher = _from_pretrained_fp32(args.teacher_model_name_or_path, hf_kwargs=hf_kwargs)
    except KeyboardInterrupt:
        print("[interrupt] model download/load interrupted; re-run to resume/continue.")
        raise
    teacher.config.use_cache = False
    teacher.eval()
    teacher.to(device=device, dtype=param_dtype)

    # Dataset iterator
    ds_kwargs = {}
    if args.dataset_config_name:
        ds_kwargs["name"] = args.dataset_config_name
    if args.cache_dir:
        ds_kwargs["cache_dir"] = args.cache_dir

    if args.streaming:
        ds = load_dataset(args.dataset_name, **ds_kwargs, split=args.dataset_split, streaming=True)
        if args.shuffle_buffer and args.shuffle_buffer > 0:
            ds = ds.shuffle(buffer_size=args.shuffle_buffer, seed=args.seed)
        ds_iter = iter(ds)
    else:
        ds = load_dataset(args.dataset_name, **ds_kwargs, split=args.dataset_split)
        # For non-streaming, we can shuffle via Python iteration.
        ds = ds.shuffle(seed=args.seed)
        ds_iter = iter(ds)

    text_iter = _iter_text_samples(ds_iter, args, tokenizer)

    packed = PackedTokenStream(
        text_iter=text_iter,
        tokenizer=tokenizer,
        max_length=args.max_length,
        add_eos=True,
    )

    # Meta/progress
    meta_path = out / "meta.json"
    if not (args.resume and meta_path.exists()):
        meta = {
            "format": "qwen_kd_topk_cache_v1",
            "teacher_model": args.teacher_model_name_or_path,
            "tokenizer": args.teacher_model_name_or_path,
            "dataset_name": args.dataset_name,
            "dataset_config_name": args.dataset_config_name,
            "dataset_split": args.dataset_split,
            "dataset_format": args.dataset_format,
            "dataset_text_field": args.dataset_text_field,
            "chat_thinking": args.enable_thinking,
            "max_length": args.max_length,
            "topk": args.topk,
            "top_k": args.topk,  # backwards/for notebooks
            "rand_neg": int(args.rand_neg),
            "random_negatives": int(args.rand_neg),  # backwards/for notebooks
            "num_sequences": int(args.num_sequences),
            "total_sequences": int(args.num_sequences),  # backwards/for notebooks
            "vocab_size": int(getattr(teacher.config, "vocab_size", 0) or 0),
            "transformers_version": transformers_version,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
    else:
        try:
            prev = json.loads(meta_path.read_text())
        except Exception:
            prev = {}
        for k, v in {
            "teacher_model": args.teacher_model_name_or_path,
            "dataset_name": args.dataset_name,
            "dataset_split": args.dataset_split,
            "dataset_format": args.dataset_format,
            "max_length": args.max_length,
            "topk": args.topk,
            "rand_neg": int(args.rand_neg),
        }.items():
            if k in prev and prev.get(k) != v:
                raise RuntimeError(f"--resume but meta.json mismatch for {k}: {prev.get(k)} != {v}")

    # Accumulators for current shard
    shard_inputs: List[torch.Tensor] = []
    shard_attn: List[torch.Tensor] = []
    shard_topk_idx: List[torch.Tensor] = []
    shard_topk_logits: List[torch.Tensor] = []
    shard_rand_idx: List[torch.Tensor] = []
    shard_rand_logits: List[torch.Tensor] = []

    def flush_shard(shard_id: int):
        if not shard_inputs:
            return
        input_ids = torch.stack(shard_inputs, dim=0).to(torch.int32)
        attention_mask = torch.stack(shard_attn, dim=0).to(torch.uint8)
        topk_idx = torch.stack(shard_topk_idx, dim=0).to(torch.int32)
        topk_logits = torch.stack(shard_topk_logits, dim=0).to(torch.float16)

        shard_path = out / f"shard_{shard_id:05d}.pt"
        obj = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "topk_idx": topk_idx,
            "topk_logits": topk_logits,
        }
        if shard_rand_idx:
            obj["rand_idx"] = torch.stack(shard_rand_idx, dim=0).to(torch.int32)
            obj["rand_logits"] = torch.stack(shard_rand_logits, dim=0).to(torch.float16)
        torch.save(obj, shard_path)
        print(f"[write] {shard_path.name} | N={input_ids.shape[0]}")
        with open(progress_path, "w") as f:
            json.dump({"n_written": int(n_written), "next_shard_id": int(shard_id) + 1}, f, indent=2)
        shard_inputs.clear()
        shard_attn.clear()
        shard_topk_idx.clear()
        shard_topk_logits.clear()
        shard_rand_idx.clear()
        shard_rand_logits.clear()

    # Main loop
    shard_id = 0
    n_written = 0
    if args.resume and progress_path.exists():
        try:
            prog = json.loads(progress_path.read_text())
            shard_id = int(prog.get("next_shard_id", 0) or 0)
            n_written = int(prog.get("n_written", 0) or 0)
            print(f"[resume] shard_id={shard_id} | cached sequences={n_written}")
        except Exception as e:
            print(f"[warn] failed to read {progress_path.name}: {e}; starting from scratch.")

    # Build a small batcher from packed iterator
    batch: List[List[int]] = []

    try:
        with torch.inference_mode():
            for block in packed:
                batch.append(block)
                if len(batch) < args.batch_size:
                    continue

                # [B, L]
                input_ids = torch.tensor(batch, dtype=torch.long, device=device)
                attn = torch.ones_like(input_ids, dtype=torch.long, device=device)

                # Teacher forward (full vocab logits; we do this only ONCE in cache build)
                out_obj = teacher(input_ids=input_ids, attention_mask=attn, use_cache=False, return_dict=True)
                logits = out_obj.logits  # [B, L, V]
                logits = logits[:, :-1, :]  # [B, L-1, V]

                # Extract top-k
                topk_vals, topk_idx = torch.topk(logits, k=args.topk, dim=-1)

                rand_idx = None
                rand_vals = None
                if args.rand_neg and args.rand_neg > 0:
                    V = logits.shape[-1]
                    # Random token ids per position (uniform). Duplicates are extremely unlikely at typical vocab sizes.
                    rand_idx = torch.randint(
                        low=0,
                        high=V,
                        size=(logits.shape[0], logits.shape[1], int(args.rand_neg)),
                        device=logits.device,
                        dtype=torch.long,
                    )
                    rand_vals = logits.gather(dim=-1, index=rand_idx)

                # Move to CPU + compact dtypes
                input_ids_cpu = input_ids.detach().to("cpu", dtype=torch.int32)
                attn_cpu = attn.detach().to("cpu", dtype=torch.uint8)
                topk_idx_cpu = topk_idx.detach().to("cpu", dtype=torch.int32)
                topk_vals_cpu = topk_vals.detach().to("cpu", dtype=torch.float16)

                rand_idx_cpu = None
                rand_vals_cpu = None
                if rand_idx is not None:
                    rand_idx_cpu = rand_idx.detach().to("cpu", dtype=torch.int32)
                    rand_vals_cpu = rand_vals.detach().to("cpu", dtype=torch.float16)

                # Append each sequence in batch to shard buffers
                bsz = input_ids_cpu.shape[0]
                for i in range(bsz):
                    shard_inputs.append(input_ids_cpu[i])
                    shard_attn.append(attn_cpu[i])
                    shard_topk_idx.append(topk_idx_cpu[i])
                    shard_topk_logits.append(topk_vals_cpu[i])
                    if rand_idx_cpu is not None:
                        shard_rand_idx.append(rand_idx_cpu[i])
                        shard_rand_logits.append(rand_vals_cpu[i])

                n_written += bsz
                batch.clear()

                if len(shard_inputs) >= args.shard_size:
                    flush_shard(shard_id)
                    shard_id += 1

                if n_written >= args.num_sequences:
                    break
    except KeyboardInterrupt:
        # Save whatever we've accumulated so far, then exit cleanly.
        flush_shard(shard_id)
        print(f"[interrupt] stopping early; cached sequences={n_written} | out_dir={out}")
        raise

    # Flush remainder
    flush_shard(shard_id)
    with open(progress_path, "w") as f:
        json.dump({"n_written": int(n_written), "next_shard_id": int(shard_id) + 1, "done": True}, f, indent=2)

    print(f"Done. Cached sequences={n_written} | out_dir={out}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # Avoid a long stack trace on user cancel / Colab stop.
        sys.exit(130)
