"""
Stage A: Apple-style 2-bit QAT for Qwen/Qwen3-0.6B (or any HF causal LM)
+ optional KD (knowledge distillation) using a precomputed teacher top-k cache.

This script supports three training modes:

1) Standard SFT-QAT (no distillation):
   - Uses a dataset (e.g., alpaca-style chat) and trains the quantized model with CE loss.

2) Online KD-QAT (teacher model run during training):
   - (Not implemented in this minimal script; use cache mode below for MPS-friendly distillation.)

3) Cached KD-QAT (recommended for MPS):
   - Provide --kd_cache_dir pointing to a directory produced by scripts/precompute_teacher_topk*.py
   - The cache contains, for each token position, the teacher's top-k logits and token ids
     (and optionally random negatives).
   - Training computes student logits only on those candidate token ids and minimizes KL.

Why cached KD?
- Running a teacher model doubles memory and compute, which is painful on MPS.
- Full-vocab KL every step is expensive. Cached top-k is the usual approximation.

Known issue with candidate-set KD:
- The student can become "argmax-unstable" under full-vocab generation: a token outside the
  candidate set can become a runaway argmax, causing repetitive garbage ("opathyopathy...").

Fix implemented here:
- Add a small *full-vocab* hard teacher top-1 CE term on a tiny subset of positions
  (by default: the last non-pad prediction position per sequence):
    loss = KD(candidate-set) + hard_top1(candidate-set) + hard_full_top1(full-vocab)

The new option is:
  --hard_full_top1_weight 0.02-0.05

This term is cheap (one [B,V] matmul per batch) and strongly improves greedy decoding stability.

Extra stabilization options (paper-inspired / practical):
- --ov-freeze: freeze all attention v_proj and o_proj *weights* across layers (keep _f_param trainable)
- --freeze-last-mlp --freeze-last-mlp-layers N: freeze last N layers' MLP projection weights

Resume behavior:
- If you resume from a *full* training checkpoint while using freezing or --train_f_only,
  we automatically resume MODEL-ONLY (weights only), reinitializing optimizer/scheduler.
  This avoids shape/state mismatches when the set of trainable params changes.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import __version__ as transformers_version

# Ensure local package imports work without installation.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from qat_lora.data import DataCollatorForSFT, build_alpaca_messages, tokenize_chat_sft
from qat_lora.model_utils import apply_layerwise_grad_scaling, init_all_f, replace_linear_with_qat
from qat_lora.mixed_precision import pick_device, resolve_amp_dtype, resolve_param_dtype
from qat_lora.quantizer import QATQuantConfig
from qat_lora.train_loop import LoopConfig, train_sft_single_device, resolve_resume_checkpoint  # type: ignore


# -----------------------------
# Utilities
# -----------------------------

def _from_pretrained_fp32(model_name_or_path: str):
    """
    Transformers is deprecating torch_dtype= in favor of dtype= in some versions.
    Support both to avoid warnings/breakage across installs.
    """
    try:
        return AutoModelForCausalLM.from_pretrained(model_name_or_path, dtype=torch.float32)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float32)


def _get_base_transformer(causal_lm: torch.nn.Module) -> torch.nn.Module:
    """
    Return the module that produces hidden states (without computing lm_head logits).
    This is model-family dependent; we try a few common attributes.
    """
    for attr in ("model", "transformer", "gpt_neox", "decoder"):
        if hasattr(causal_lm, attr):
            m = getattr(causal_lm, attr)
            if isinstance(m, torch.nn.Module):
                return m
    if hasattr(causal_lm, "base_model") and isinstance(causal_lm.base_model, torch.nn.Module):
        return causal_lm.base_model
    raise AttributeError(
        "Could not locate base transformer module on this CausalLM "
        "(tried model/transformer/gpt_neox/decoder/base_model)."
    )


def _infer_num_layers(model: torch.nn.Module) -> int:
    # Prefer config, fallback to parsing parameter names.
    cfg = getattr(model, "config", None)
    if cfg is not None:
        for k in ("num_hidden_layers", "n_layer", "num_layers"):
            if hasattr(cfg, k):
                v = int(getattr(cfg, k))
                if v > 0:
                    return v
    # Parse model.layers.{i}.*
    max_i = -1
    pat = re.compile(r"\bmodel\.layers\.(\d+)\b")
    for name, _ in model.named_parameters():
        m = pat.search(name)
        if m:
            max_i = max(max_i, int(m.group(1)))
    return max_i + 1 if max_i >= 0 else 0


def _apply_train_f_only(model: torch.nn.Module) -> int:
    """
    Freeze everything except QAT f parameters (usually named *_f_param).
    Returns number of trainable parameters.
    """
    n = 0
    for name, p in model.named_parameters():
        if name.endswith("_f_param"):
            p.requires_grad = True
            n += p.numel()
        else:
            p.requires_grad = False
    return n


def _apply_ov_freeze(model: torch.nn.Module) -> int:
    """
    Freeze attention v_proj/o_proj *weights* across all layers.
    Keep *_f_param trainable.
    Returns number of frozen parameters (by elements).
    """
    frozen = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.endswith(".self_attn.v_proj.weight") or name.endswith(".self_attn.o_proj.weight"):
            p.requires_grad = False
            frozen += p.numel()
    return frozen


def _apply_freeze_last_mlp(model: torch.nn.Module, n_last_layers: int) -> int:
    """
    Freeze last N layers' MLP projection *weights*:
      gate_proj.weight, up_proj.weight, down_proj.weight
    Keep *_f_param trainable.
    Returns number of frozen elements.
    """
    num_layers = _infer_num_layers(model)
    if num_layers <= 0:
        print("[freeze] WARNING: could not infer num_layers; skipping freeze_last_mlp.")
        return 0
    n_last_layers = max(0, min(int(n_last_layers), num_layers))
    start = num_layers - n_last_layers
    frozen = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # example: model.layers.27.mlp.gate_proj.weight
        if not name.endswith(".weight"):
            continue
        if ".mlp." not in name:
            continue
        m = re.search(r"\bmodel\.layers\.(\d+)\b", name)
        if not m:
            continue
        layer_i = int(m.group(1))
        if layer_i >= start:
            p.requires_grad = False
            frozen += p.numel()
    return frozen


def _is_full_train_checkpoint(obj: object) -> bool:
    return isinstance(obj, dict) and ("model" in obj) and ("optimizer" in obj) and ("opt_step" in obj)


def _maybe_force_model_only_resume(
    *,
    output_dir: Path,
    resume_from_checkpoint: Optional[str],
    force_model_only: bool,
) -> Optional[str]:
    """
    If force_model_only is True and resume_from_checkpoint points to a *full* training checkpoint,
    extract model weights into a model-only file named with the opt_step, and return that path.

    Otherwise return resume_from_checkpoint unchanged.

    This mirrors the behavior you saw in logs:
      [resume] freeze/train_f_only enabled; using MODEL-ONLY resume ...
    """
    if not resume_from_checkpoint or not force_model_only:
        return resume_from_checkpoint

    try:
        ckpt_path = resolve_resume_checkpoint(resume_from_checkpoint, str(output_dir))
    except Exception:
        # If resolve fails, let the training loop error normally.
        return resume_from_checkpoint

    try:
        obj = torch.load(ckpt_path, map_location="cpu")
    except Exception:
        return resume_from_checkpoint

    if not _is_full_train_checkpoint(obj):
        return resume_from_checkpoint

    opt_step = int(obj.get("opt_step", 0))
    model_sd = obj.get("model", None)
    if not isinstance(model_sd, dict):
        return resume_from_checkpoint

    out_path = output_dir / f"resume_model_only_step{opt_step}.pt"
    if not out_path.exists():
        torch.save(model_sd, out_path)
    print(
        f"[resume] freeze/train_f_only enabled; using MODEL-ONLY resume from {out_path} "
        f"(extracted from {ckpt_path})"
    )
    return str(out_path)


# -----------------------------
# KD cache dataset
# -----------------------------

class TopKCacheIterableDataset(IterableDataset):
    """
    Stream samples from a KD cache directory.

    Expected files:
      - meta.json (optional but recommended)
      - shard_*.pt (or any *.pt) containing a dict of tensors:
          input_ids:      [N, L]
          attention_mask: [N, L]
          topk_idx:       [N, L-1, K]
          topk_logits:    [N, L-1, K]
          (optional)
          rand_idx:       [N, L-1, R]
          rand_logits:    [N, L-1, R]

    We yield per-example dicts with those keys.
    """

    def __init__(self, cache_dir: str, shuffle_files: bool = False, seed: int = 0):
        super().__init__()
        self.cache_dir = Path(cache_dir)
        if not self.cache_dir.exists():
            raise FileNotFoundError(f"KD cache dir not found: {self.cache_dir}")
        self.shuffle_files = bool(shuffle_files)
        self.seed = int(seed)

        self.meta = {}
        meta_path = self.cache_dir / "meta.json"
        if meta_path.exists():
            try:
                self.meta = json.loads(meta_path.read_text())
            except Exception:
                self.meta = {}

        # Prefer shard_*.pt, fallback to all .pt excluding checkpoints.
        shards = sorted(self.cache_dir.glob("shard_*.pt"))
        if not shards:
            shards = sorted([p for p in self.cache_dir.glob("*.pt") if not p.name.startswith("checkpoint_")])
        if not shards:
            raise FileNotFoundError(f"No shard_*.pt or *.pt files found in cache dir: {self.cache_dir}")
        self.shards = shards

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        rng = random.Random(self.seed + int(torch.initial_seed()) % 10_000_000)
        files = list(self.shards)
        if self.shuffle_files:
            rng.shuffle(files)

        for path in files:
            obj = torch.load(path, map_location="cpu")
            if not isinstance(obj, dict):
                continue

            # Required tensors
            input_ids = obj.get("input_ids")
            attn = obj.get("attention_mask")
            topk_idx = obj.get("topk_idx")
            topk_logits = obj.get("topk_logits")
            if input_ids is None or attn is None or topk_idx is None or topk_logits is None:
                raise ValueError(f"Cache shard missing required keys: {path}")

            rand_idx = obj.get("rand_idx", None)
            rand_logits = obj.get("rand_logits", None)

            n = input_ids.shape[0]
            for i in range(n):
                ex = {
                    "input_ids": input_ids[i],
                    "attention_mask": attn[i],
                    "topk_idx": topk_idx[i],
                    "topk_logits": topk_logits[i],
                }
                if rand_idx is not None and rand_logits is not None:
                    ex["rand_idx"] = rand_idx[i]
                    ex["rand_logits"] = rand_logits[i]
                yield ex


def _collate_cache_batch(examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    # Stack everything that exists in the first example.
    keys = examples[0].keys()
    batch: Dict[str, torch.Tensor] = {}
    for k in keys:
        batch[k] = torch.stack([ex[k] for ex in examples], dim=0)
    return batch


# -----------------------------
# KD cache forward
# -----------------------------

def _install_kd_cache_forward(
    model: torch.nn.Module,
    *,
    distill_temperature: float,
    distill_weight: float,
    hard_top1_weight: float = 0.0,
    hard_full_top1_weight: float = 0.0,
) -> None:
    """
    Monkeypatch model.forward so the training loop sees `.loss` computed from cached KD tensors.

    Batch must contain:
      - input_ids: [B, L]
      - attention_mask: [B, L]
      - topk_idx: [B, L-1, K]
      - topk_logits: [B, L-1, K] (raw teacher logits)
    Optionally:
      - rand_idx / rand_logits: [B, L-1, R] (random negatives sampled from vocab with teacher logits)

    Loss:
      KD KL on candidate set (topk [+ rand]) at temperature T (scaled by T^2)
      + hard_top1_weight * NLL(candidate-softmax, teacher top1)   (temp=1)
      + hard_full_top1_weight * CE(full-vocab, teacher top1)     (temp=1, last non-pad pos only)
    """
    if not hasattr(model, "lm_head"):
        raise AttributeError("Model has no lm_head; cannot compute logits for KD cache mode.")

    base = _get_base_transformer(model)
    lm_head = model.lm_head

    if not hasattr(lm_head, "weight"):
        raise AttributeError("lm_head has no .weight; KD cache mode expects a weight matrix.")

    T = float(distill_temperature)
    kd_w = float(distill_weight)

    def kd_forward(
        input_ids=None,
        attention_mask=None,
        topk_idx=None,
        topk_logits=None,
        rand_idx=None,
        rand_logits=None,
        **kwargs,
    ):
        if input_ids is None or topk_idx is None or topk_logits is None:
            raise ValueError("KD cache forward requires input_ids, topk_idx, topk_logits in the batch.")

        # Student hidden states [B, L, H]
        out = base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        hidden = out.last_hidden_state[:, :-1, :]  # [B, S=L-1, H]
        B, S, H = hidden.shape

        # Candidate ids/logits (teacher)
        cand_idx = topk_idx.to(torch.long)                # [B,S,K]
        cand_t_logits = topk_logits.float()               # [B,S,K]
        if rand_idx is not None and rand_logits is not None:
            cand_idx = torch.cat([cand_idx, rand_idx.to(torch.long)], dim=-1)          # [B,S,K+R]
            cand_t_logits = torch.cat([cand_t_logits, rand_logits.float()], dim=-1)    # [B,S,K+R]

        B, S, C = cand_idx.shape
        N = B * S

        # Flatten positions.
        h = hidden.reshape(N, H)            # [N,H]
        i = cand_idx.reshape(N, C)          # [N,C]

        # Gather lm_head weight rows: [N,C,H]
        W = lm_head.weight                  # [V,H] (ideally fp32/bf16/fp16)
        Wc = W[i]

        # Student logits on candidates: [B,S,C]
        # (Wc @ h) per row
        logits_c = torch.bmm(Wc, h.unsqueeze(-1)).squeeze(-1).reshape(B, S, C)

        # ---- KD KL on candidate set ----
        t = cand_t_logits / T
        s = logits_c.float() / T

        p_t = torch.softmax(t, dim=-1)
        log_p_t = torch.log_softmax(t, dim=-1)
        log_p_s_T = torch.log_softmax(s, dim=-1)

        kl = (p_t * (log_p_t - log_p_s_T)).sum(dim=-1)  # [B,S]

        if attention_mask is not None:
            m = attention_mask[:, 1:].float()  # prediction positions align with targets 1..L-1
            denom = m.sum().clamp(min=1.0)
            loss = (kl * m).sum() / denom
        else:
            m = None
            denom = None
            loss = kl.mean()

        loss = loss * (T * T) * kd_w

        # ---- Candidate-softmax hard top1 (cheap) ----
        if hard_top1_weight and hard_top1_weight > 0.0:
            # teacher top-1 is candidate index 0 (topk sorted)
            log_p_s_1 = torch.log_softmax(logits_c.float(), dim=-1)  # temp=1
            hard = -log_p_s_1[..., 0]  # [B,S]
            if m is not None:
                hard = (hard * m).sum() / denom
            else:
                hard = hard.mean()
            loss = loss + float(hard_top1_weight) * hard

        # ---- FULL-vocab hard top1 (the real argmax-stability fix) ----
        if hard_full_top1_weight and hard_full_top1_weight > 0.0:
            # teacher top-1 token id per position
            top1_ids = topk_idx[..., 0].to(torch.long)  # [B,S]

            if attention_mask is not None:
                # pick last non-pad prediction position per sequence:
                # lengths = number of valid tokens; last target index = lengths-1; last pred position = lengths-2
                lengths = attention_mask.to(torch.long).sum(dim=1)  # [B]
                valid = lengths >= 2
                if valid.any():
                    pos = (lengths - 2).clamp(min=0)  # [B]
                    b_idx = torch.arange(B, device=hidden.device)[valid]
                    pos_v = pos[valid]
                    h_sel = hidden[b_idx, pos_v, :]            # [Bv,H]
                    y_sel = top1_ids[b_idx, pos_v]            # [Bv]
                else:
                    h_sel = None
                    y_sel = None
            else:
                # all positions valid -> use last position for everyone
                h_sel = hidden[:, -1, :]
                y_sel = top1_ids[:, -1]

            if h_sel is not None and y_sel is not None and h_sel.numel() > 0:
                # Full vocab logits at selected positions: [Bv,V]
                # Compute in lm_head dtype to avoid copying the whole weight to fp32.
                h_in = h_sel.to(dtype=W.dtype)
                logits_full = F.linear(h_in, W)  # [Bv,V] in bf16/fp16/fp32
                hard_full = F.cross_entropy(logits_full.float(), y_sel)
                loss = loss + float(hard_full_top1_weight) * hard_full

        return SimpleNamespace(loss=loss)

    model.forward = kd_forward  # type: ignore[attr-defined]


# -----------------------------
# Args
# -----------------------------

def parse_args():
    p = argparse.ArgumentParser()

    # Model + output
    p.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-0.6B")
    p.add_argument("--output_dir", type=str, required=True)

    # Dataset (used only when NOT using kd_cache_dir)
    p.add_argument("--dataset_name", type=str, default="tatsu-lab/alpaca")
    p.add_argument("--dataset_split", type=str, default="train")
    p.add_argument("--dataset_format", type=str, choices=["alpaca"], default="alpaca")

    # Training
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    # Device & mixed precision
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--amp_dtype", type=str, default="auto", choices=["auto", "no", "bf16", "fp16"])
    p.add_argument("--param_dtype", type=str, default="auto", choices=["auto", "fp32", "bf16", "fp16"])

    # QAT config
    p.add_argument(
        "-q",
        "--quant_bits",
        type=int,
        default=2,
        choices=[2, 4],
        help="Weight quantization bits for QATLinear (2=default, 4=less aggressive).",
    )
    p.add_argument("--skip_lm_head", action="store_true", help="Do not quantize lm_head (recommended).")
    p.add_argument("--enable_thinking", action="store_true", help="Qwen3 thinking mode in chat template.")
    p.add_argument("--grad_scale", action="store_true", help="Apply layerwise grad scaling 1/sqrt(out_features).")
    p.add_argument("--ema_decay", type=float, default=0.0, help="If >0, maintain EMA with this decay (e.g., 0.999).")

    # Init f (only matters when starting from scratch; resuming overwrites anyway)
    p.add_argument("--init_method", type=str, choices=["newton", "percentile"], default="newton")
    p.add_argument("--init_newton_iters", type=int, default=4)
    p.add_argument("--init_newton_samples", type=int, default=65536)
    p.add_argument("--init_percentile", type=float, default=99.5)

    # Resume
    p.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to a training checkpoint .pt file. Use auto/latest/last to pick from output_dir.",
    )
    p.add_argument(
        "--init_model_state",
        type=str,
        default=None,
        help="Optional model.state_dict .pt to load before training (e.g., previous QAT run). "
        "Incompatible with --resume_from_checkpoint.",
    )

    # KD cache mode (optional)
    p.add_argument("--kd_cache_dir", type=str, default=None, help="Path to teacher top-k cache dir (meta.json + shards).")
    p.add_argument("--kd_cache_shuffle_files", action="store_true", help="Shuffle cache shard file order each epoch.")
    p.add_argument("--distill_temperature", type=float, default=2.0, help="KD temperature (typical: 1.0 or 2.0).")
    p.add_argument("--distill_weight", type=float, default=1.0, help="Weight for KD term (candidate-set KL).")

    p.add_argument(
        "--hard-top1-weight", "--hard_top1_weight",
        type=float, default=0.0, dest="hard_top1_weight",
        help="(kd_cache_dir) Candidate-softmax hard top-1 weight. Try 0.05.",
    )
    p.add_argument(
        "--hard-full-top1-weight", "--hard_full_top1_weight",
        type=float, default=0.0, dest="hard_full_top1_weight",
        help="(kd_cache_dir) FULL-vocab hard top-1 CE weight. Fixes argmax collapse. Try 0.02–0.05.",
    )

    # Freeze / trainable-set controls
    p.add_argument("--train_f_only", action="store_true", help="Train only *_f_param (freeze all weights).")
    p.add_argument(
        "--ov-freeze", "--ov_freeze",
        action="store_true", dest="ov_freeze",
        help="Freeze all attention v_proj/o_proj weights (keep f trainable).",
    )
    p.add_argument(
        "--freeze-last-mlp", "--freeze_last_mlp",
        action="store_true", dest="freeze_last_mlp",
        help="Freeze last N layers' MLP projection weights.",
    )
    p.add_argument(
        "--freeze-last-mlp-layers", "--freeze_last_mlp_layers",
        type=int, default=1, dest="freeze_last_mlp_layers",
        help="How many last layers' MLPs to freeze when --freeze-last-mlp is enabled.",
    )

    return p.parse_args()


# -----------------------------
# Main
# -----------------------------

def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Qwen3 requires transformers>=4.51.0 for tokenizer/model config registration
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

    # Load on CPU then cast
    model = _from_pretrained_fp32(args.model_name_or_path)
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # Replace linear layers with QATLinear
    qc = QATQuantConfig(n_bits=int(args.quant_bits))
    print(f"[qat] weight_bits={qc.n_bits}")
    exclude = r"(^lm_head$)" if args.skip_lm_head else None
    replace_linear_with_qat(model, qc=qc, exclude_regex=exclude, verbose=False)

    # Initialize f parameters (will be overwritten if resuming)
    init_all_f(
        model,
        qc=qc,
        method=args.init_method,
        newton_iters=args.init_newton_iters,
        newton_samples=args.init_newton_samples,
        percentile=args.init_percentile,
        verbose=False,
    )

    if args.grad_scale:
        apply_layerwise_grad_scaling(model, verbose=False)

    # Cast parameters after QATLinear creation
    model = model.to(dtype=param_dtype)

    if args.init_model_state:
        if args.resume_from_checkpoint:
            raise ValueError("--init_model_state cannot be combined with --resume_from_checkpoint.")
        ckpt_path = Path(args.init_model_state)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"init_model_state path does not exist: {ckpt_path}")
        print(f"[init] loading model state from {ckpt_path}")
        obj = torch.load(ckpt_path, map_location="cpu")
        state_dict = None
        if isinstance(obj, dict):
            if all(torch.is_tensor(v) for v in obj.values()):
                state_dict = obj
            elif "model" in obj and isinstance(obj["model"], dict):
                state_dict = obj["model"]
            elif "state_dict" in obj and isinstance(obj["state_dict"], dict):
                state_dict = obj["state_dict"]
        if state_dict is None:
            raise RuntimeError(f"Could not interpret checkpoint format at {ckpt_path}; expected a model.state_dict.")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[init] missing keys: {missing[:10]}{'...' if len(missing) > 10 else ''}")
        if unexpected:
            print(f"[init] unexpected keys: {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}")
        print("[init] model weights loaded.")

    # Optional KD cache mode
    if args.kd_cache_dir:
        ds_cache = TopKCacheIterableDataset(
            cache_dir=args.kd_cache_dir,
            shuffle_files=args.kd_cache_shuffle_files,
            seed=0,
        )

        # Cache meta check (best-effort)
        cache_max_len = ds_cache.meta.get("max_length", None)
        cache_topk = ds_cache.meta.get("topk", None)
        if cache_max_len is not None and int(cache_max_len) != int(args.max_length):
            print(
                f"[kd-cache] Note: cache max_length={cache_max_len} (you passed --max_length={args.max_length}). "
                f"Training uses cache tensors; --max_length is ignored in cache mode."
            )
        if cache_topk is not None:
            print(f"[kd-cache] cache topk={cache_topk}")

        print(
            f"[kd-cache] dir={args.kd_cache_dir} | weight={args.distill_weight} | T={args.distill_temperature} "
            f"| hard_top1={args.hard_top1_weight} | hard_full_top1={args.hard_full_top1_weight}"
        )

        _install_kd_cache_forward(
            model,
            distill_temperature=args.distill_temperature,
            distill_weight=args.distill_weight,
            hard_top1_weight=args.hard_top1_weight,
            hard_full_top1_weight=args.hard_full_top1_weight,
        )

        dl = DataLoader(
            ds_cache,
            batch_size=args.per_device_train_batch_size,
            shuffle=False,  # IterableDataset; shuffling handled by dataset if enabled
            collate_fn=_collate_cache_batch,
            drop_last=True,
        )
    else:
        # Standard SFT-QAT over dataset
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
        dl = DataLoader(
            ds,
            batch_size=args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=collator,
            drop_last=True,
        )

    # Trainable-set / freezing controls
    frozen_elems = 0
    trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if args.train_f_only:
        trainable = _apply_train_f_only(model)
        print(f"[freeze] train_f_only enabled. trainable_elements={trainable}")
    else:
        if args.ov_freeze:
            frozen_elems += _apply_ov_freeze(model)
        if args.freeze_last_mlp:
            frozen_elems += _apply_freeze_last_mlp(model, args.freeze_last_mlp_layers)

        trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if args.ov_freeze or args.freeze_last_mlp:
            # Count trainable params (by tensor count) for log readability
            total_params = sum(1 for _ in model.parameters())
            trainable_params = sum(1 for p in model.parameters() if p.requires_grad)
            print(f"[freeze] frozen_elements={frozen_elems} | trainable_params={trainable_params}/{total_params}")

    # Build loop config
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

    # Force model-only resume if needed
    force_model_only = bool(args.train_f_only or args.ov_freeze or args.freeze_last_mlp)
    resume_arg = _maybe_force_model_only_resume(
        output_dir=out,
        resume_from_checkpoint=args.resume_from_checkpoint,
        force_model_only=force_model_only,
    )

    extra_state = {"stage": "qat", "args": vars(args)}
    train_sft_single_device(
        model,
        dl,
        device,
        loop_cfg,
        tokenizer=tokenizer,
        extra_state=extra_state,
        resume_from_checkpoint=resume_arg,
    )

    # Convenience name used by the rest of the repo.
    torch.save(model.state_dict(), out / "qat_state_dict.pt")
    with open(out / "training_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Done. QAT checkpoint saved to: {out/'qat_state_dict.pt'}")


if __name__ == "__main__":
    main()
