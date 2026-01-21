#!/usr/bin/env python3
"""
compute_ymatrix.py (diag-only output stats)

Build a "Y-matrix" after your first conversion pass by collecting per-feature
second-moments of selected MODULE OUTPUTS:

  sigma2_out[name][o] = E[y_o^2]

This is useful for a *second, safe* AWQ-equivalent pass where you apply
ROW scaling on a producer Linear and inverse COLUMN scaling on its unique
consumer Linear (rows -> reverse by columns), but ONLY for graph locations
where the inverse is well-defined (e.g. v_proj -> o_proj, up_proj -> down_proj).

By default, this script collects outputs for:
  - model.layers.*.self_attn.v_proj
  - model.layers.*.mlp.up_proj

You can override with --targets-regex (comma-separated regex patterns).

Output .pt file:
  {
    "sigma2_out": { "<module_name>": FloatTensor[out_features], ... },
    "count":      { "<module_name>": int, ... },   # samples per feature
    "meta":       { ... }
  }

Example:
  python scripts/compute_ymatrix.py \
    --model Qwen/Qwen3-0.6B \
    --calib-mode random_ids --tokens 100000 --seq-len 512 --batch-size 1 \
    --out runs/ymatrix_qwen3_0.6b_random.pt \
    --trust-remote-code --verbose

Or compute on an AWQ-scaled base:
  python scripts/compute_ymatrix.py \
    --model runs/awq_scaled_model_a0.25 \
    --calib-mode pseudo_text --tokens 100000 --seq-len 512 --batch-size 1 \
    --out runs/ymatrix_awq_a0.25.pt --verbose
"""

from __future__ import annotations
import argparse
import datetime as _dt
import os
import re
import time
from typing import Dict, Optional, List, Tuple

import torch

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception as e:
    raise SystemExit(
        "This script requires transformers. Install with:\n"
        "  pip install transformers\n"
        f"Original import error: {e}"
    )


def _now_iso() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _resolve_dtype(s: str) -> torch.dtype:
    s = s.lower().strip()
    if s in ("fp16", "float16", "half"):
        return torch.float16
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp32", "float32", "float"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {s}")


def _resolve_device(s: str) -> torch.device:
    s = s.strip().lower()
    if s == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(s)


def _set_tokenizer_padding(tokenizer) -> None:
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})


def _allowed_token_ids(tokenizer) -> torch.Tensor:
    vocab = int(tokenizer.vocab_size)
    special = set(getattr(tokenizer, "all_special_ids", []) or [])
    if tokenizer.pad_token_id is not None:
        special.add(int(tokenizer.pad_token_id))
    allowed = [i for i in range(vocab) if i not in special]
    if not allowed:
        allowed = list(range(vocab))
    return torch.tensor(allowed, dtype=torch.long)


def _make_random_ids(
    allowed_ids: torch.Tensor,
    batch_size: int,
    seq_len: int,
    bos_id: Optional[int],
    eos_id: Optional[int],
    rng: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    idx = torch.randint(0, allowed_ids.numel(), (batch_size, seq_len), generator=rng, dtype=torch.long)
    input_ids = allowed_ids[idx]
    if bos_id is not None and seq_len >= 1:
        input_ids[:, 0] = int(bos_id)
    if eos_id is not None and seq_len >= 2:
        mask = torch.rand((batch_size,), generator=rng) < 0.10
        input_ids[mask, -1] = int(eos_id)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    return input_ids, attention_mask


_WORDS = [
    "system","user","assistant","token","matrix","importance","quant","linear","layer","model","weights","activation",
    "variance","scale","group","error","loss","optimize","entropy","calibration","python","torch","jax","compile",
    "tensor","dtype","float","bf16","fp16","int8","lut","rank","svd","attention","mlp","cache","latency","debug",
    "logits","softmax","north","south","east","west","city","desert","mountain","river","science","history",
    "0","1","2","3","4","5","6","7","8","9","(",")","{","}","[","]",":",";","=",",",".","-","_","/","*","+",
]


def _make_pseudorandom_text(batch_size: int, approx_chars: int, rng) -> List[str]:
    out = []
    for _ in range(batch_size):
        parts = []
        chars = 0
        while chars < approx_chars:
            mode = rng.random()
            if mode < 0.60:
                n = rng.randint(8, 20)
                words = [rng.choice(_WORDS) for _ in range(n)]
                line = " ".join(words).strip()
                if not line.endswith("."):
                    line += "."
            elif mode < 0.85:
                n = rng.randint(8, 18)
                toks = [rng.choice(_WORDS) for _ in range(n)]
                line = " ".join(toks)
                line = f"def f_{rng.randint(0,999)}({rng.choice(['x','y','z'])}): {line}"
            else:
                line = f'{{"k{rng.randint(0,99)}": "{rng.choice(_WORDS)}", "v": {rng.randint(0,9999)}, "ok": true}}'
            parts.append(line)
            chars += len(line) + 1
        out.append("\n".join(parts))
    return out


class Collector:
    def __init__(self):
        self.sums: Dict[str, torch.Tensor] = {}
        self.counts: Dict[str, int] = {}

    def add(self, name: str, y: torch.Tensor) -> None:
        if not isinstance(y, torch.Tensor) or (not y.is_floating_point()) or y.numel() == 0:
            return
        yf = y.float()
        out_features = yf.shape[-1]
        n = int(yf.numel() // out_features)
        s = (yf * yf).reshape(-1, out_features).sum(dim=0)  # [out_features]
        if name not in self.sums:
            self.sums[name] = s.detach().cpu()
            self.counts[name] = n
        else:
            self.sums[name] += s.detach().cpu()
            self.counts[name] += n

    def finalize(self) -> Dict[str, torch.Tensor]:
        out = {}
        for name, s in self.sums.items():
            c = max(int(self.counts.get(name, 0)), 1)
            out[name] = (s / c).to(dtype=torch.float32)
        return out


def _compile_targets(spec: str) -> List[re.Pattern]:
    # Comma-separated regexes
    pats = []
    for item in (spec or "").split(","):
        item = item.strip()
        if not item:
            continue
        pats.append(re.compile(item))
    return pats


def _matches_any(name: str, pats: List[re.Pattern]) -> bool:
    return any(p.search(name) for p in pats)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id or local path")
    ap.add_argument("--out", required=True, help="Output .pt path")
    ap.add_argument("--tokens", type=int, default=100_000, help="Total tokens to process (approx)")
    ap.add_argument("--seq-len", type=int, default=512, help="Sequence length per sample")
    ap.add_argument("--batch-size", type=int, default=1, help="Batch size")
    ap.add_argument("--calib-mode", choices=["random_ids", "pseudo_text", "textfile"], default="random_ids")
    ap.add_argument("--textfile", default=None)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--targets-regex", default=r".*\.self_attn\.v_proj$,\s*.*\.mlp\.up_proj$",
                    help="Comma-separated regexes for module names to collect OUTPUT stats from")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    device = _resolve_device(args.device)
    dtype = _resolve_dtype(args.dtype)

    targets = _compile_targets(args.targets_regex)
    if not targets:
        raise SystemExit("No valid --targets-regex patterns provided")

    if args.calib_mode == "textfile":
        if not args.textfile or not os.path.exists(args.textfile):
            raise SystemExit("--calib-mode textfile requires --textfile that exists")

    if args.verbose:
        print(f"[ymatrix] model={args.model}")
        print(f"[ymatrix] device={device} dtype={dtype}")
        print(f"[ymatrix] calib_mode={args.calib_mode} tokens≈{args.tokens} seq_len={args.seq_len} bs={args.batch_size}")
        print(f"[ymatrix] targets={args.targets_regex}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    _set_tokenizer_padding(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.to(device)

    collector = Collector()
    hooks = []

    def make_hook(name: str):
        def _hook(module, inputs, output):
            collector.add(name, output)
        return _hook

    # Register hooks only on matching modules
    for name, m in model.named_modules():
        if _matches_any(name, targets):
            # Use forward hook (captures module output)
            hooks.append(m.register_forward_hook(make_hook(name)))

    if args.verbose:
        print(f"[ymatrix] hooked modules: {len(hooks)}")

    py_rng = __import__("random").Random(args.seed)
    g = torch.Generator(device="cpu")
    g.manual_seed(args.seed)

    allowed_ids = None
    bos_id = getattr(tokenizer, "bos_token_id", None)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if args.calib_mode == "random_ids":
        allowed_ids = _allowed_token_ids(tokenizer)

    text_lines = None
    if args.calib_mode == "textfile":
        with open(args.textfile, "r", encoding="utf-8", errors="ignore") as f:
            text_lines = f.read().splitlines()
        if not text_lines:
            raise SystemExit(f"textfile is empty: {args.textfile}")

    tokens_target = int(args.tokens)
    step_tokens = args.batch_size * args.seq_len
    steps = max((tokens_target + step_tokens - 1) // step_tokens, 1)

    t0 = time.time()
    tokens_done = 0

    with torch.inference_mode():
        for step in range(steps):
            if args.calib_mode == "random_ids":
                input_ids, attn = _make_random_ids(allowed_ids, args.batch_size, args.seq_len, bos_id, eos_id, g)
            elif args.calib_mode == "pseudo_text":
                texts = _make_pseudorandom_text(args.batch_size, approx_chars=8 * args.seq_len, rng=py_rng)
                enc = tokenizer(texts, return_tensors="pt", padding="max_length", truncation=True,
                                max_length=args.seq_len, add_special_tokens=True)
                input_ids = enc["input_ids"]
                attn = enc.get("attention_mask", torch.ones_like(input_ids, dtype=torch.long))
            else:
                texts = []
                for _ in range(args.batch_size):
                    k = py_rng.randint(4, 12)
                    chunk = "\n".join(text_lines[py_rng.randrange(0, len(text_lines))] for _ in range(k))
                    texts.append(chunk)
                enc = tokenizer(texts, return_tensors="pt", padding="max_length", truncation=True,
                                max_length=args.seq_len, add_special_tokens=True)
                input_ids = enc["input_ids"]
                attn = enc.get("attention_mask", torch.ones_like(input_ids, dtype=torch.long))

            input_ids = input_ids.to(device)
            attn = attn.to(device)

            _ = model(input_ids=input_ids, attention_mask=attn, use_cache=False)

            tokens_done += int(input_ids.numel())
            if args.verbose and (step == 0 or (step + 1) % 20 == 0 or step + 1 == steps):
                dt = time.time() - t0
                tok_s = tokens_done / max(dt, 1e-9)
                print(f"[ymatrix] step {step+1}/{steps} tokens={tokens_done} ({tok_s:.1f} tok/s)")

    for h in hooks:
        h.remove()

    sigma2_out = collector.finalize()
    meta = {
        "created": _now_iso(),
        "model": args.model,
        "dtype": str(dtype),
        "device": str(device),
        "tokens_target": tokens_target,
        "tokens_processed": int(tokens_done),
        "seq_len": int(args.seq_len),
        "batch_size": int(args.batch_size),
        "calib_mode": args.calib_mode,
        "textfile": args.textfile,
        "seed": int(args.seed),
        "targets_regex": args.targets_regex,
        "hooked_modules": int(len(sigma2_out)),
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"sigma2_out": sigma2_out, "count": collector.counts, "meta": meta}, args.out)

    if args.verbose:
        print(f"[ymatrix] saved: {args.out}")
        print(f"[ymatrix] meta: {meta}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
