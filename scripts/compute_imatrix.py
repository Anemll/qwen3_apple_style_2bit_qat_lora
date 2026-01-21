#!/usr/bin/env python3
"""
compute_imatrix.py (diag-only)

Collects a diagonal "importance matrix" proxy per (Linear) tensor:
  sigma2[name][i] = E[x_i^2] over calibration tokens
where x is the *input activation* to that Linear.

This mirrors the common iMatrix simplification used for quantization scoring:
  iMSE ≈ sum_i sigma2[i] * ||ΔW[:, i]||^2

Output: a torch .pt file containing:
  {
    "sigma2": { "<module_name>": FloatTensor[in_features], ... },
    "count":  { "<module_name>": int, ... },   # number of samples per feature
    "meta":   { ... }                          # run metadata
  }

Example (pseudo-random tokens):
  python scripts/compute_imatrix.py \
    --model Qwen/Qwen3-0.6B \
    --calib-mode random_ids \
    --tokens 100000 --seq-len 512 --batch-size 1 \
    --out runs/imatrix_qwen3_0.6b_random.pt
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import os
import re
import time
from typing import Dict, Optional, Tuple

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


def _is_linear_like(m: torch.nn.Module) -> bool:
    # We want modules with a 2D float weight and float inputs (hook will verify).
    w = getattr(m, "weight", None)
    if w is None or not isinstance(w, torch.nn.Parameter):
        return False
    if w.dim() != 2:
        return False
    # Many HF models use torch.nn.Linear or variants; this is a permissive check.
    return True


def _compile_regex(pat: Optional[str]) -> Optional[re.Pattern]:
    if not pat:
        return None
    return re.compile(pat)


@dataclasses.dataclass
class Collector:
    sigma2_sums: Dict[str, torch.Tensor]
    counts: Dict[str, int]

    def __init__(self):
        self.sigma2_sums = {}
        self.counts = {}

    def add(self, name: str, x: torch.Tensor) -> None:
        # x: [..., in_features]
        if not x.is_floating_point():
            return
        if x.numel() == 0:
            return
        xf = x.float()
        in_features = xf.shape[-1]
        n = int(xf.numel() // in_features)  # number of samples per feature

        # sum over all leading dims
        s = (xf * xf).reshape(-1, in_features).sum(dim=0)  # [in_features]

        if name not in self.sigma2_sums:
            self.sigma2_sums[name] = s.detach().cpu()
            self.counts[name] = n
        else:
            self.sigma2_sums[name] += s.detach().cpu()
            self.counts[name] += n

    def finalize(self) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for name, s in self.sigma2_sums.items():
            c = max(int(self.counts.get(name, 0)), 1)
            out[name] = (s / c).to(dtype=torch.float32)
        return out


def _set_tokenizer_padding(tokenizer) -> None:
    # Ensure pad token exists for padding='max_length'
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            # Fallback to a common token; last resort
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})


def _allowed_token_ids(tokenizer) -> torch.Tensor:
    vocab = int(tokenizer.vocab_size)
    special = set(getattr(tokenizer, "all_special_ids", []) or [])
    if tokenizer.pad_token_id is not None:
        special.add(int(tokenizer.pad_token_id))
    # Keep BOS/EOS allowed (they're real tokens), but random streams usually don't need them.
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
    # Sample from allowed_ids uniformly
    idx = torch.randint(
        low=0,
        high=allowed_ids.numel(),
        size=(batch_size, seq_len),
        generator=rng,
        dtype=torch.long,
    )
    input_ids = allowed_ids[idx]
    if bos_id is not None and seq_len >= 1:
        input_ids[:, 0] = int(bos_id)
    # (Optional) add EOS sometimes
    if eos_id is not None and seq_len >= 2:
        # Set EOS in ~10% of sequences at the end (cheap variety)
        mask = torch.rand((batch_size,), generator=rng) < 0.10
        input_ids[mask, -1] = int(eos_id)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    return input_ids, attention_mask


_WORDS = [
    # A compact, high-entropy pool (general + code-ish)
    "system","user","assistant","analysis","result","token","matrix","importance","quant","linear","layer",
    "model","weights","activation","variance","scale","group","error","loss","optimize","entropy","calibration",
    "python","torch","jax","compile","kernel","tensor","dtype","float","bf16","fp16","int8","lut","rank","svd",
    "if","else","for","while","return","break","continue","try","except","class","def","import","from","with",
    "json","yaml","http","https","api","request","response","cache","stream","audio","video","time","date",
    "compute","memory","speed","throughput","latency","debug","trace","logits","softmax","attention","mlp",
    "north","south","east","west","city","desert","mountain","river","market","science","music","history",
    "0","1","2","3","4","5","6","7","8","9","(",")","{","}","[","]",":",";","=",",",".","-","_","/","*","+",
]


def _make_pseudorandom_text(batch_size: int, approx_chars: int, rng) -> list[str]:
    # Build pseudo-random "text" with a mix of natural-ish sentences and code-ish lines.
    out = []
    for _ in range(batch_size):
        parts = []
        chars = 0
        while chars < approx_chars:
            mode = rng.random()
            if mode < 0.60:
                # sentence
                n = rng.randint(8, 20)
                words = [rng.choice(_WORDS) for _ in range(n)]
                line = " ".join(words).strip()
                if not line.endswith("."):
                    line += "."
            elif mode < 0.85:
                # code-ish
                n = rng.randint(8, 18)
                toks = [rng.choice(_WORDS) for _ in range(n)]
                line = " ".join(toks)
                line = f"def f_{rng.randint(0,999)}({rng.choice(['x','y','z'])}): {line}"
            else:
                # json-ish
                line = f'{{"k{rng.randint(0,99)}": "{rng.choice(_WORDS)}", "v": {rng.randint(0,9999)}, "ok": true}}'
            parts.append(line)
            chars += len(line) + 1
        out.append("\n".join(parts))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id or local path (e.g., Qwen/Qwen3-0.6B)")
    ap.add_argument("--out", required=True, help="Output .pt path (e.g., runs/imatrix.pt)")
    ap.add_argument("--tokens", type=int, default=100_000, help="Total tokens to process (approx)")
    ap.add_argument("--seq-len", type=int, default=512, help="Sequence length (context) per sample")
    ap.add_argument("--batch-size", type=int, default=1, help="Batch size")
    ap.add_argument("--calib-mode", choices=["random_ids", "pseudo_text", "textfile"], default="random_ids",
                    help="Calibration source: random token ids, pseudo-random text, or a text file")
    ap.add_argument("--textfile", default=None, help="Used when --calib-mode textfile")
    ap.add_argument("--seed", type=int, default=12345, help="RNG seed")
    ap.add_argument("--device", default="auto", help="auto|cpu|cuda|mps|xla:0 etc.")
    ap.add_argument("--dtype", default="bf16", help="bf16|fp16|fp32 (model load dtype)")
    ap.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True to HF")
    ap.add_argument("--include", default=None, help="Regex to include module names")
    ap.add_argument("--exclude", default=None, help="Regex to exclude module names")
    ap.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = ap.parse_args()

    device = _resolve_device(args.device)
    dtype = _resolve_dtype(args.dtype)

    inc_re = _compile_regex(args.include)
    exc_re = _compile_regex(args.exclude)

    if args.calib_mode == "textfile":
        if not args.textfile or not os.path.exists(args.textfile):
            raise SystemExit("--calib-mode textfile requires --textfile that exists")

    if args.verbose:
        print(f"[imatrix] model={args.model}")
        print(f"[imatrix] device={device} dtype={dtype}")
        print(f"[imatrix] calib_mode={args.calib_mode} tokens≈{args.tokens} seq_len={args.seq_len} bs={args.batch_size}")

    # Load tokenizer + model
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

    # Prepare collector + hooks
    collector = Collector()
    hooks = []

    def make_hook(name: str):
        def _hook(module, inputs):
            if not inputs:
                return
            x = inputs[0]
            if not isinstance(x, torch.Tensor):
                return
            collector.add(name, x)
        return _hook

    # Register hooks on "linear-like" modules
    for name, m in model.named_modules():
        if not _is_linear_like(m):
            continue
        if inc_re and not inc_re.search(name):
            continue
        if exc_re and exc_re.search(name):
            continue
        # Keep names consistent with init_model_v2.py (it keys by module name)
        hooks.append(m.register_forward_pre_hook(make_hook(name)))

    if args.verbose:
        print(f"[imatrix] hooked modules: {len(hooks)}")

    # RNGs
    py_rng = __import__("random").Random(args.seed)
    g = torch.Generator(device="cpu")
    g.manual_seed(args.seed)

    allowed_ids = None
    bos_id = getattr(tokenizer, "bos_token_id", None)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if args.calib_mode == "random_ids":
        allowed_ids = _allowed_token_ids(tokenizer)

    # Read textfile once (can be large; we stream in chunks of lines)
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
                input_ids, attn = _make_random_ids(
                    allowed_ids=allowed_ids,
                    batch_size=args.batch_size,
                    seq_len=args.seq_len,
                    bos_id=bos_id,
                    eos_id=eos_id,
                    rng=g,
                )
            elif args.calib_mode == "pseudo_text":
                texts = _make_pseudorandom_text(args.batch_size, approx_chars=8 * args.seq_len, rng=py_rng)
                enc = tokenizer(
                    texts,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=args.seq_len,
                    add_special_tokens=True,
                )
                input_ids = enc["input_ids"]
                attn = enc.get("attention_mask", torch.ones_like(input_ids, dtype=torch.long))
            else:
                # textfile: sample random windows from lines
                texts = []
                for _ in range(args.batch_size):
                    # join a handful of random lines to get enough length
                    k = py_rng.randint(4, 12)
                    chunk = "\n".join(text_lines[py_rng.randrange(0, len(text_lines))] for _ in range(k))
                    texts.append(chunk)
                enc = tokenizer(
                    texts,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=args.seq_len,
                    add_special_tokens=True,
                )
                input_ids = enc["input_ids"]
                attn = enc.get("attention_mask", torch.ones_like(input_ids, dtype=torch.long))

            input_ids = input_ids.to(device)
            attn = attn.to(device)

            _ = model(input_ids=input_ids, attention_mask=attn, use_cache=False)

            tokens_done += int(input_ids.numel())
            if args.verbose and (step == 0 or (step + 1) % 20 == 0 or step + 1 == steps):
                dt = time.time() - t0
                tok_s = tokens_done / max(dt, 1e-9)
                print(f"[imatrix] step {step+1}/{steps} tokens={tokens_done} ({tok_s:.1f} tok/s)")

    # Remove hooks
    for h in hooks:
        h.remove()

    sigma2 = collector.finalize()
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
        "include": args.include,
        "exclude": args.exclude,
        "hooked_modules": int(len(sigma2)),
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"sigma2": sigma2, "count": collector.counts, "meta": meta}, args.out)

    if args.verbose:
        print(f"[imatrix] saved: {args.out}")
        print(f"[imatrix] meta: {meta}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
