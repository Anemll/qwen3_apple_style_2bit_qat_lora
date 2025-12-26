#!/usr/bin/env python3
"""
End-to-end KD-QAT training for Anemll quantization.

Uses train_e2e, save_checkpoint, load_checkpoint from qat_lora.

Two modes:
1. --train-weights: Train weights only (scales frozen)
2. --train-scales: Train scales only (weights frozen)

Example:
    # Train weights (scales frozen)
    python scripts/train_anemll_qat.py \
        --model-id Qwen/Qwen3-0.6B \
        --kd-cache-dir caches/alpaca_chat_think_both_L128_K32_R256 \
        --max-steps 1000 \
        --train-weights

    # Train scales (weights frozen)
    python scripts/train_anemll_qat.py \
        --model-id Qwen/Qwen3-0.6B \
        --init-state runs/weights_trained/model_state_dict.pt \
        --kd-cache-dir caches/alpaca_chat_think_both_L128_K32_R256 \
        --max-steps 500 \
        --train-scales
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from qat_lora import (
    AnemllQuantConfig,
    replace_linear_with_anemll,
    train_e2e,
    save_checkpoint,
    load_checkpoint,
)


def main():
    parser = argparse.ArgumentParser(description="End-to-end KD-QAT for Anemll quantization")

    # Model args
    parser.add_argument("--model-id", default="Qwen/Qwen3-0.6B", help="HF model id or local path")
    parser.add_argument("--init-state", default="", help="Path to initial state dict (optional)")
    parser.add_argument("--output-dir", default="runs/anemll_qat", help="Output directory")
    parser.add_argument("--device", default="auto", help="auto|cpu|cuda|mps")
    parser.add_argument("--dtype", default="bf16", help="fp16|bf16|fp32")

    # Quantization args
    parser.add_argument("--lut-size", type=int, default=16, help="LUT entries (4=2bit, 16=4bit)")
    parser.add_argument("--group-size", type=int, default=32, help="Group size")
    parser.add_argument("--scale-rank", type=int, default=4, help="Low-rank for scales")
    parser.add_argument("--quantize-attn", action="store_true", help="Quantize attention layers")
    parser.add_argument("--attn-lut-size", type=int, default=0, help="Attention LUT size (0=use --lut-size)")
    parser.add_argument("--attn-group-size", type=int, default=0, help="Attention group size")
    parser.add_argument("--attn-scale-rank", type=int, default=-1, help="Attention scale rank")

    # Training mode
    parser.add_argument("--train-weights", action="store_true", help="Train weights (scales frozen)")
    parser.add_argument("--train-scales", action="store_true", help="Train scales (weights frozen)")

    # Training args
    parser.add_argument("--kd-cache-dir", required=True, help="KD cache directory")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max training steps")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--temperature", type=float, default=2.0, help="Distillation temperature")
    parser.add_argument("--hard-top1-weight", type=float, default=0.0, help="Weight for hard label top-1 loss (stabilizes training)")
    parser.add_argument("--hard-full-weight", type=float, default=0.0005, help="Weight for hard label full vocab loss (default 0.0005)")

    # Logging/saving
    parser.add_argument("--logging-steps", type=int, default=50, help="Log every N steps")
    parser.add_argument("--eval-steps", type=int, default=200, help="Evaluate every N steps")
    parser.add_argument("--eval-samples", type=int, default=40, help="Samples for evaluation")
    parser.add_argument("--snap-weights", action="store_true", help="Snap weights to quantized values before saving")
    parser.add_argument("--skip-training", action="store_true", help="Skip training, just load and optionally snap/export")

    args = parser.parse_args()

    # Validate training mode
    if not args.skip_training and not args.train_weights and not args.train_scales:
        print("ERROR: Must specify --train-weights or --train-scales (or both), or use --skip-training")
        sys.exit(1)

    # Device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # Dtype
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    dtype = dtype_map.get(args.dtype, torch.bfloat16)

    print(f"Device: {device}, dtype: {dtype}")

    # Load model
    print(f"\nLoading model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.to(device)

    # Replace with AnemllQATLinear
    print("\nReplacing linear layers with AnemllQATLinear...")
    mlp_config = AnemllQuantConfig(
        lut_size=args.lut_size,
        group_size=args.group_size,
        scale_rank=args.scale_rank,
    )

    attn_config = AnemllQuantConfig(
        lut_size=args.attn_lut_size or args.lut_size,
        group_size=args.attn_group_size or args.group_size,
        scale_rank=args.attn_scale_rank if args.attn_scale_rank >= 0 else args.scale_rank,
    )

    count = replace_linear_with_anemll(
        model,
        mlp_config=mlp_config,
        attn_config=attn_config,
        quantize_attn=args.quantize_attn,
        quantize_lm_head=False,
        verbose=False,
    )
    print(f"Replaced {count} layers")
    model.to(device)

    # Load initial state if provided
    if args.init_state:
        init_path = Path(args.init_state)
        # Handle directory or file
        if init_path.is_dir():
            # Look for model_state_dict.pt in directory
            state_file = init_path / "model_state_dict.pt"
            if not state_file.exists():
                state_file = init_path / "best_state_dict.pt"
            if not state_file.exists():
                print(f"ERROR: No state dict found in {init_path}")
                sys.exit(1)
            init_path = state_file
        print(f"\nLoading initial state from {init_path}...")
        load_checkpoint(model, str(init_path), device=device, verbose=True)

    # Run end-to-end training (unless skipped)
    result = None
    if not args.skip_training:
        result = train_e2e(
            model=model,
            cache_dir=args.kd_cache_dir,
            device=device,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            lr=args.lr,
            temperature=args.temperature,
            train_weights=args.train_weights,
            train_scales=args.train_scales,
            hard_top1_weight=args.hard_top1_weight,
            hard_full_weight=args.hard_full_weight,
            logging_steps=args.logging_steps,
            eval_steps=args.eval_steps,
            eval_samples=args.eval_samples,
            save_dir=args.output_dir,
            verbose=True,
        )
    else:
        print("\nSkipping training (--skip-training)")

    # Snap weights if requested
    if args.snap_weights:
        from qat_lora import snap_all_weights
        print("\nSnapping weights to quantized values (LUT[idx] * scale)...")
        snap_all_weights(model, store_lut_values=False, verbose=True)

    # Save final checkpoint
    config = {
        'model_id': args.model_id,
        'lut_size': args.lut_size,
        'group_size': args.group_size,
        'scale_rank': args.scale_rank,
        'attn_lut_size': args.attn_lut_size or args.lut_size,
        'attn_group_size': args.attn_group_size or args.group_size,
        'attn_scale_rank': args.attn_scale_rank if args.attn_scale_rank >= 0 else args.scale_rank,
        'quantize_attn': args.quantize_attn,
        'train_weights': args.train_weights,
        'train_scales': args.train_scales,
        'max_steps': args.max_steps,
        'lr': args.lr,
        'snapped': args.snap_weights,
        'result': result,
    }

    save_checkpoint(model, args.output_dir, config=config, verbose=True)

    print(f"\nDone! Output saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
