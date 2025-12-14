"""
Utility: "snap" all QATLinear weights to the 2-bit grid and save.

This is useful for:
- verifying that after QAT, weights lie close to the intended 4-level codebook
- exporting a float checkpoint whose weights are exactly quantized

Important:
- This does NOT pack weights into 2-bit storage.
- True 2-bit inference needs custom packing + kernels.

Usage:
  python scripts/hard_quantize_checkpoint.py \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --qat_checkpoint runs/qwen3_0p6b_qat2b/qat_state_dict.pt \
    --output_path runs/qwen3_0p6b_qat2b/hard_quant_full_state_dict.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

# Ensure local package imports work without installation.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from qat_lora.quantizer import QATQuantConfig, fake_quant_weight_2bit
from qat_lora.model_utils import replace_linear_with_qat
from qat_lora.qat_linear import QATLinear


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-0.6B")
    p.add_argument("--qat_checkpoint", type=str, required=True)
    p.add_argument("--output_path", type=str, required=True)
    p.add_argument("--skip_lm_head", action="store_true")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    qc = QATQuantConfig()

    try:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, dtype=torch.float32)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.float32)
    exclude = r"(^lm_head$)" if args.skip_lm_head else None
    replace_linear_with_qat(model, qc=qc, exclude_regex=exclude, verbose=False)

    sd = torch.load(args.qat_checkpoint, map_location="cpu")
    model.load_state_dict(sd, strict=False)

    # Snap weights
    for _, m in model.named_modules():
        if isinstance(m, QATLinear):
            w_q = fake_quant_weight_2bit(m.weight, m.f(), qc)
            m.weight.data.copy_(w_q)  # overwrite
    torch.save(model.state_dict(), args.output_path)
    print(f"Saved snapped checkpoint to: {args.output_path}")


if __name__ == "__main__":
    main()
