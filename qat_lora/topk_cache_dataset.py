"""
Utilities for reading teacher top-k distillation caches produced by
`scripts/precompute_teacher_topk.py`.

Each cache directory contains:
  - meta.json
  - shard_00000.pt, shard_00001.pt, ...

Each shard is a torch.save()'d dict with:
  - input_ids:     int32/long   [N, L]
  - attention_mask: uint8/bool  [N, L]  (optional; if missing, assumed all-ones)
  - topk_idx:      int32/long   [N, L-1, K]
  - topk_logits:   float16      [N, L-1, K]   (raw teacher logits, not divided by T)
  - rand_idx (optional):    int32/long [N, L-1, R]   (random negative token ids)
  - rand_logits (optional): float16    [N, L-1, R]   (teacher logits for rand_idx)

This module provides:
  - TopKCacheDataset: iterable dataset streaming examples from shards
  - topk_cache_collate: simple collator that stacks tensors into a batch
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import torch
from torch.utils.data import IterableDataset


@dataclass
class CacheMeta:
    max_length: Optional[int] = None
    topk: Optional[int] = None
    rand_neg: Optional[int] = None
    teacher_model: Optional[str] = None
    tokenizer: Optional[str] = None
    vocab_size: Optional[int] = None
    format: Optional[str] = None


def load_cache_meta(cache_dir: str | Path) -> CacheMeta:
    p = Path(cache_dir) / "meta.json"
    if not p.exists():
        return CacheMeta()
    with open(p, "r") as f:
        obj = json.load(f)
    return CacheMeta(
        max_length=obj.get("max_length"),
        topk=obj.get("topk"),
        rand_neg=obj.get("rand_neg"),
        teacher_model=obj.get("teacher_model"),
        tokenizer=obj.get("tokenizer"),
        vocab_size=obj.get("vocab_size"),
        format=obj.get("format"),
    )


class TopKCacheDataset(IterableDataset):
    """
    Streams cached distillation examples from shard_*.pt files.

    Each yielded item is a dict of CPU tensors:
      - input_ids: [L] long
      - attention_mask: [L] long
      - topk_idx: [L-1, K] long
      - topk_logits: [L-1, K] float16
    """

    def __init__(self, cache_dir: str | Path, shuffle_files: bool = False, seed: int = 0):
        super().__init__()
        self.cache_dir = Path(cache_dir)
        self.shuffle_files = shuffle_files
        self.seed = seed

        if not self.cache_dir.exists():
            raise FileNotFoundError(f"Cache dir does not exist: {self.cache_dir}")

        self.files = sorted(self.cache_dir.glob("shard_*.pt"))
        if not self.files:
            raise FileNotFoundError(f"No shard_*.pt files found in cache dir: {self.cache_dir}")

        self.meta = load_cache_meta(self.cache_dir)

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        files = list(self.files)
        if self.shuffle_files:
            rng = random.Random(self.seed)
            rng.shuffle(files)

        for f in files:
            shard = torch.load(f, map_location="cpu")
            input_ids = shard["input_ids"]
            topk_idx = shard["topk_idx"]
            topk_logits = shard["topk_logits"]
            rand_idx = shard.get("rand_idx", None)
            rand_logits = shard.get("rand_logits", None)

            # attention_mask may be missing for older caches; default to all ones.
            attn = shard.get("attention_mask", None)
            if attn is None:
                attn = torch.ones_like(input_ids, dtype=torch.long)

            # Ensure common dtypes
            if input_ids.dtype != torch.long:
                input_ids = input_ids.long()
            if attn.dtype != torch.long:
                attn = attn.long()
            if topk_idx.dtype != torch.long:
                topk_idx = topk_idx.long()
            if topk_logits.dtype != torch.float16 and topk_logits.dtype != torch.bfloat16:
                # Store logits compactly; float16 is typical.
                topk_logits = topk_logits.to(torch.float16)

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


def topk_cache_collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Stacks a list of examples into a batch. Intended for use as DataLoader collate_fn.
    """
    if not batch:
        raise ValueError("Empty batch in topk_cache_collate")

    input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
    attention_mask = torch.stack([b["attention_mask"] for b in batch], dim=0)
    topk_idx = torch.stack([b["topk_idx"] for b in batch], dim=0)
    topk_logits = torch.stack([b["topk_logits"] for b in batch], dim=0)

    out = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "topk_idx": topk_idx,
        "topk_logits": topk_logits,
    }

    # Optional rand negatives
    if "rand_idx" in batch[0]:
        out["rand_idx"] = torch.stack([b["rand_idx"] for b in batch], dim=0)
        out["rand_logits"] = torch.stack([b["rand_logits"] for b in batch], dim=0)

    return out
