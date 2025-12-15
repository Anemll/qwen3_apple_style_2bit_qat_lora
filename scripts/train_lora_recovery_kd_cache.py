"""
Deprecated wrapper.

KD cache support is now integrated into `scripts/train_lora_recovery.py`.
Use that script with:
  --kd_cache_dir <cache_dir> --skip_lm_head --distill_temperature 2.0 --distill_weight 1.0
"""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("train_lora_recovery.py")), run_name="__main__")
