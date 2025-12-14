"""
Teacher top-k cache loader used for KD-LoRA without running the teacher at training time.

Cache is produced by `scripts/precompute_teacher_topk.py` and contains:
  - meta.json
  - shard_*.pt files with tensors:
      input_ids:      int32 [N, L]
      attention_mask: int32 [N, L]
      topk_idx:       int32 [N, L-1, K]
      topk_logits:    float16 [N, L-1, K]   (raw teacher logits)
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
    format: Optional[str] = None


def load_cache_meta(cache_dir: str | Path) -> CacheMeta:
    cache_dir = Path(cache_dir)
    meta_path = cache_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta.json in cache dir: {cache_dir}")
    meta = json.loads(meta_path.read_text())
    return CacheMeta(
        max_length=int(meta["max_length"]) if "max_length" in meta else None,
        topk=int(meta["topk"]) if "topk" in meta else None,
        format=meta.get("format"),
    )


class TopKCacheDataset(IterableDataset):
    """
    Iterable dataset over cached shard_*.pt files.

    - Intended for num_workers=0.
    - Each __iter__ reshuffles shard order if shuffle_files=True.
    """

    def __init__(self, cache_dir: str | Path, *, shuffle_files: bool = False, seed: int = 0):
        super().__init__()
        self.cache_dir = Path(cache_dir)
        self.shuffle_files = bool(shuffle_files)
        self.seed = int(seed)
        self._iter_calls = 0

        if not self.cache_dir.exists():
            raise FileNotFoundError(f"Cache dir not found: {self.cache_dir}")

        # Quick validation
        _ = load_cache_meta(self.cache_dir)

    def _list_shards(self) -> List[Path]:
        shards = sorted(self.cache_dir.glob("shard_*.pt"))
        if not shards:
            raise FileNotFoundError(f"No shard_*.pt files found in {self.cache_dir}")
        return shards

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        shards = self._list_shards()

        if self.shuffle_files:
            rng = random.Random(self.seed + self._iter_calls)
            rng.shuffle(shards)
        self._iter_calls += 1

        for shard_path in shards:
            obj = torch.load(shard_path, map_location="cpu")
            input_ids = obj["input_ids"]          # [N, L] int32
            attention_mask = obj["attention_mask"]  # [N, L] int32
            topk_idx = obj["topk_idx"]            # [N, L-1, K] int32
            topk_logits = obj["topk_logits"]      # [N, L-1, K] float16

            n = input_ids.shape[0]
            for i in range(n):
                yield {
                    "input_ids": input_ids[i].to(torch.long),
                    "attention_mask": attention_mask[i].to(torch.long),
                    "topk_idx": topk_idx[i].to(torch.int32),
                    "topk_logits": topk_logits[i].to(torch.float16),
                }


def topk_cache_collate(features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate cached KD batches.
    """
    return {
        "input_ids": torch.stack([f["input_ids"] for f in features], dim=0),
        "attention_mask": torch.stack([f["attention_mask"] for f in features], dim=0),
        "topk_idx": torch.stack([f["topk_idx"] for f in features], dim=0),
        "topk_logits": torch.stack([f["topk_logits"] for f in features], dim=0),
    }

