#!/usr/bin/env python
"""
Utility to rewrite the stored quantizer bitwidth buffers in a QAT checkpoint.

Example:
python scripts/convert_qat_bits.py \
  --input runs/qwen3_q4/qat_state_dict.pt \
  --output runs/qwen3_q4/as_q2_state_dict.pt \
  --target_bits 2

The script looks for tensors named *_qat_nbits in the state dict (or nested under
"model"/"state_dict") and overwrites them with the requested bitwidth.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import torch


def _locate_state_dict(
    obj: object,
) -> Tuple[Dict[str, torch.Tensor], str | None]:
    """
    Return (state_dict, container_key).
    container_key is None when obj itself is the state dict, or the key ("model"/"state_dict")
    to write back into before saving.
    """
    if isinstance(obj, dict):
        if obj and all(torch.is_tensor(v) for v in obj.values()):
            return obj, None
        if "model" in obj and isinstance(obj["model"], dict):
            return obj["model"], "model"
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"], "state_dict"
    raise RuntimeError("Could not locate a state_dict in the checkpoint. Expected a dict of tensors or a dict with 'model'/'state_dict' keys.")


def convert_bits(state_dict: Dict[str, torch.Tensor], target_bits: int) -> int:
    updated = 0
    for name, tensor in list(state_dict.items()):
        if not isinstance(tensor, torch.Tensor):
            continue
        if name.endswith("._qat_nbits"):
            old = int(tensor.item())
            if old != target_bits:
                state_dict[name] = torch.tensor(int(target_bits), dtype=torch.int16)
            else:
                state_dict[name] = torch.tensor(old, dtype=torch.int16)
            updated += 1
    return updated


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, required=True, help="Path to an existing QAT checkpoint (qat_state_dict.pt or full checkpoint).")
    p.add_argument("--output", type=str, required=True, help="Where to save the converted checkpoint.")
    p.add_argument("--target_bits", type=int, default=2, choices=[2, 4], help="Target weight fake-quant bitwidth.")
    return p.parse_args()


def main():
    args = parse_args()
    inp = Path(args.input)
    out = Path(args.output)
    if not inp.is_file():
        raise FileNotFoundError(f"Input checkpoint does not exist: {inp}")
    ckpt = torch.load(inp, map_location="cpu")
    state_dict, container_key = _locate_state_dict(ckpt)
    count = convert_bits(state_dict, int(args.target_bits))
    if container_key is not None:
        ckpt[container_key] = state_dict
        torch.save(ckpt, out)
    else:
        torch.save(state_dict, out)
    print(f"[convert] rewrote _qat_nbits tensors in {count} layers -> {args.target_bits}-bit. Saved to {out}")


if __name__ == "__main__":
    main()
