"""
Stage B: LoRA recovery adapters for a QAT model.

Flow:
1) Load base model
2) Replace Linear -> QATLinear (same as Stage A)
3) Load QAT checkpoint (weights + learned f)
4) Freeze base weights
5) Enable LoRA on QATLinear layers (train only LoRA A/B)
6) Train with either:
   - standard SFT (CE loss), or
   - cached KD (KL loss) if --kd_cache_dir is provided
7) Save:
   - lora_only_state_dict.pt  (LoRA A/B weights only)
   - full_state_dict.pt       (entire model state_dict, including frozen base + LoRA)

Cached KD-LoRA (MPS-friendly)
-----------------------------
If --kd_cache_dir is provided, we train LoRA to match a teacher distribution without
running the teacher during training and without computing full-vocab KL.

We use a cache produced by scripts/precompute_teacher_topk*.py containing:
  topk_idx / topk_logits (and optionally rand_idx/rand_logits)

Loss in cache mode:
  KD KL on candidate set + (optional) hard top-1 terms.

Important: candidate-set KD alone can still collapse under *full-vocab* generation,
because tokens outside the candidate set are never normalized/penalized.

Fix implemented here:
  --hard_full_top1_weight
    adds a cheap full-vocab CE on teacher top-1 at the last non-pad prediction position.

Recommended:
  --hard_full_top1_weight 0.02–0.05
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterator, List, Optional

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import __version__ as transformers_version

# Ensure local package imports work without installation.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from qat_lora.quantizer import QATQuantConfig
from qat_lora.model_utils import replace_linear_with_qat, freeze_base_enable_lora, extract_lora_state_dict
from qat_lora.data import build_alpaca_messages, tokenize_chat_sft, DataCollatorForSFT
from qat_lora.mixed_precision import pick_device, resolve_amp_dtype, resolve_param_dtype
from qat_lora.train_loop import LoopConfig, train_sft_single_device


# -----------------------------
# Utilities
# -----------------------------

def _from_pretrained_fp32(model_name_or_path: str):
    try:
        return AutoModelForCausalLM.from_pretrained(model_name_or_path, dtype=torch.float32)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float32)


def _get_base_transformer(causal_lm: torch.nn.Module) -> torch.nn.Module:
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


def _load_qat_state_dict(qat_checkpoint: str) -> Dict[str, torch.Tensor]:
    """
    Accept either:
      - model-only state_dict (param_name -> tensor), OR
      - a full training checkpoint dict containing a "model" key
    """
    obj = torch.load(qat_checkpoint, map_location="cpu")
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        # full train checkpoint
        return obj["model"]
    if isinstance(obj, dict) and all(isinstance(k, str) for k in obj.keys()):
        return obj  # model-only
    raise RuntimeError(f"Unrecognized QAT checkpoint format: {qat_checkpoint}")


# -----------------------------
# KD cache dataset (same as train_qat.py)
# -----------------------------

class TopKCacheIterableDataset(IterableDataset):
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
    keys = examples[0].keys()
    batch: Dict[str, torch.Tensor] = {}
    for k in keys:
        batch[k] = torch.stack([ex[k] for ex in examples], dim=0)
    return batch


# -----------------------------
# KD cache forward (LoRA)
# -----------------------------

def _install_kd_cache_forward(
    model: torch.nn.Module,
    *,
    distill_temperature: float,
    distill_weight: float,
    hard_top1_weight: float = 0.0,
    hard_full_top1_weight: float = 0.0,
) -> None:
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

        out = base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        hidden = out.last_hidden_state[:, :-1, :]  # [B,S,H]
        B, S, H = hidden.shape

        cand_idx = topk_idx.to(torch.long)
        cand_t_logits = topk_logits.float()
        if rand_idx is not None and rand_logits is not None:
            cand_idx = torch.cat([cand_idx, rand_idx.to(torch.long)], dim=-1)
            cand_t_logits = torch.cat([cand_t_logits, rand_logits.float()], dim=-1)

        B, S, C = cand_idx.shape
        N = B * S
        h = hidden.reshape(N, H)
        i = cand_idx.reshape(N, C)

        W = lm_head.weight  # [V,H]
        Wc = W[i]           # [N,C,H]
        logits_c = torch.bmm(Wc, h.unsqueeze(-1)).squeeze(-1).reshape(B, S, C)

        # Candidate-set KL
        t = cand_t_logits / T
        s = logits_c.float() / T

        p_t = torch.softmax(t, dim=-1)
        log_p_t = torch.log_softmax(t, dim=-1)
        log_p_s_T = torch.log_softmax(s, dim=-1)
        kl = (p_t * (log_p_t - log_p_s_T)).sum(dim=-1)  # [B,S]

        if attention_mask is not None:
            m = attention_mask[:, 1:].float()
            denom = m.sum().clamp(min=1.0)
            loss = (kl * m).sum() / denom
        else:
            m = None
            denom = None
            loss = kl.mean()

        loss = loss * (T * T) * kd_w

        # Candidate hard top1 (temp=1)
        if hard_top1_weight and hard_top1_weight > 0.0:
            log_p_s_1 = torch.log_softmax(logits_c.float(), dim=-1)
            hard = -log_p_s_1[..., 0]
            if m is not None:
                hard = (hard * m).sum() / denom
            else:
                hard = hard.mean()
            loss = loss + float(hard_top1_weight) * hard

        # Full-vocab hard top1 (last non-pad pred position)
        if hard_full_top1_weight and hard_full_top1_weight > 0.0:
            top1_ids = topk_idx[..., 0].to(torch.long)  # [B,S]

            if attention_mask is not None:
                lengths = attention_mask.to(torch.long).sum(dim=1)  # [B]
                valid = lengths >= 2
                if valid.any():
                    pos = (lengths - 2).clamp(min=0)
                    b_idx = torch.arange(B, device=hidden.device)[valid]
                    pos_v = pos[valid]
                    h_sel = hidden[b_idx, pos_v, :]
                    y_sel = top1_ids[b_idx, pos_v]
                else:
                    h_sel = None
                    y_sel = None
            else:
                h_sel = hidden[:, -1, :]
                y_sel = top1_ids[:, -1]

            if h_sel is not None and y_sel is not None and h_sel.numel() > 0:
                h_in = h_sel.to(dtype=W.dtype)
                logits_full = F.linear(h_in, W)  # [Bv,V]
                hard_full = F.cross_entropy(logits_full.float(), y_sel)
                loss = loss + float(hard_full_top1_weight) * hard_full

        return SimpleNamespace(loss=loss)

    model.forward = kd_forward  # type: ignore[attr-defined]


# -----------------------------
# Args
# -----------------------------

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-0.6B")
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
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    # Device & mixed precision
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--amp_dtype", type=str, default="auto", choices=["auto", "no", "bf16", "fp16"])
    p.add_argument("--param_dtype", type=str, default="auto", choices=["auto", "fp32", "bf16", "fp16"])

    p.add_argument(
        "-q",
        "--quant_bits",
        type=int,
        default=2,
        choices=[2, 4],
        help="Weight quantization bits for QATLinear (2=default, 4=less aggressive).",
    )

    p.add_argument("--skip_lm_head", action="store_true")
    p.add_argument("--enable_thinking", action="store_true")

    # LoRA config
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=float, default=32.0)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    # KD cache mode (optional)
    p.add_argument("--kd_cache_dir", type=str, default=None, help="Path to teacher top-k cache dir.")
    p.add_argument("--kd_cache_shuffle_files", action="store_true", help="Shuffle cache shard file order each epoch.")
    p.add_argument("--distill_temperature", type=float, default=2.0)
    p.add_argument("--distill_weight", type=float, default=1.0)

    p.add_argument(
        "--hard-top1-weight", "--hard_top1_weight",
        type=float, default=0.0, dest="hard_top1_weight",
        help="(kd_cache_dir) Candidate-softmax hard top-1 weight. Try 0.05.",
    )
    p.add_argument(
        "--hard-full-top1-weight", "--hard_full_top1_weight",
        type=float, default=0.0, dest="hard_full_top1_weight",
        help="(kd_cache_dir) FULL-vocab hard top-1 CE weight. Try 0.02–0.05.",
    )

    p.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to a training checkpoint .pt file. Use auto/latest/last to pick from output_dir.",
    )

    return p.parse_args()


# -----------------------------
# Main
# -----------------------------

def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

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

    # Base model
    model = _from_pretrained_fp32(args.model_name_or_path)
    model.config.use_cache = False

    qc = QATQuantConfig(n_bits=int(args.quant_bits))
    print(f"[qat] weight_bits={qc.n_bits}")
    exclude = r"(^lm_head$)" if args.skip_lm_head else None
    replace_linear_with_qat(model, qc=qc, exclude_regex=exclude, verbose=False)

    # Load QAT weights (+ f params)
    sd = _load_qat_state_dict(args.qat_checkpoint)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"Loaded QAT checkpoint. missing={len(missing)} unexpected={len(unexpected)}")

    # Cast params
    model = model.to(dtype=param_dtype)

    # Freeze base & enable LoRA
    enabled = freeze_base_enable_lora(
        model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        verbose=False,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Enabled LoRA on {enabled} layers. Trainable params: {trainable:,}")

    # Dataloader
    if args.kd_cache_dir:
        ds_cache = TopKCacheIterableDataset(
            cache_dir=args.kd_cache_dir,
            shuffle_files=args.kd_cache_shuffle_files,
            seed=0,
        )
        cache_max_len = ds_cache.meta.get("max_length", None)
        cache_topk = ds_cache.meta.get("topk", None)
        if cache_max_len is not None and int(cache_max_len) != int(args.max_length):
            print(
                f"[kd_cache] Note: cache max_length={cache_max_len} (you passed --max_length={args.max_length}). "
                f"--max_length is ignored in cache mode."
            )
        if cache_topk is not None:
            print(f"[kd_cache] cache topk={cache_topk}")

        print(
            f"[kd_cache] Enabled cached KD-LoRA. T={args.distill_temperature} weight={args.distill_weight} "
            f"hard_top1={args.hard_top1_weight} hard_full_top1={args.hard_full_top1_weight}"
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
            shuffle=False,
            collate_fn=_collate_cache_batch,
            drop_last=True,
        )
    else:
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
        ema_decay=0.0,
    )

    extra_state = {"stage": "lora", "args": vars(args)}
    train_sft_single_device(
        model,
        dl,
        device,
        loop_cfg,
        tokenizer=tokenizer,
        extra_state=extra_state,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )

    # Save adapters
    lora_sd = extract_lora_state_dict(model)
    torch.save(lora_sd, out / "lora_only_state_dict.pt")
    torch.save(model.state_dict(), out / "full_state_dict.pt")
    with open(out / "training_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Done. Saved LoRA adapter to: {out/'lora_only_state_dict.pt'}")


if __name__ == "__main__":
    main()
