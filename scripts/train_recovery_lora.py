#!/usr/bin/env python3
"""
Train recovery LoRA adapters on a quantized model.

Training Modes (--lora-mode):
    - recover: Standard CE loss on raw text (default, no teacher needed)
    - sft: Supervised fine-tuning on instruction/response pairs
    - kd: Knowledge distillation from teacher model (requires --teacher)

Generate Targets (--generate-targets):
    When enabled, uses a reference model (base or --teacher) to generate
    training targets. Essential for think mode with text datasets:
    - Reference model generates responses with <think>...</think> content
    - Quantized model learns to match these generated responses
    - Slower but produces proper thinking tokens in training data

Usage:
    # Mode: recover (default) - CE on raw text
    python scripts/train_recovery_lora.py \
        --model Qwen/Qwen3-0.6B \
        --v2-checkpoint runs/v2_q4a4_r32/best_state_dict.pt \
        --train-data-hf NeelNanda/pile-10k \
        --recovery-r 8 --max-steps 500

    # Mode: recover with think template + generated targets
    python scripts/train_recovery_lora.py \
        --model Qwen/Qwen3-0.6B \
        --v2-checkpoint runs/v2_q4a4_r32/best_state_dict.pt \
        --train-data-hf NeelNanda/pile-10k \
        --template-mode think --generate-targets \
        --recovery-r 8 --max-steps 500

    # Mode: sft - Supervised fine-tuning
    python scripts/train_recovery_lora.py \
        --model Qwen/Qwen3-0.6B \
        --v2-checkpoint runs/v2_q4a4_r32/best_state_dict.pt \
        --train-data data/alpaca.jsonl \
        --lora-mode sft --dataset-format alpaca \
        --recovery-r 8 --max-steps 1000

    # Mode: kd - Knowledge distillation from teacher
    python scripts/train_recovery_lora.py \
        --model Qwen/Qwen3-0.6B \
        --v2-checkpoint runs/v2_q4a4_r32/best_state_dict.pt \
        --train-data-hf NeelNanda/pile-10k \
        --lora-mode kd --teacher Qwen/Qwen3-4B-Instruct \
        --kd-temperature 2.0 --kd-alpha 0.5 \
        --recovery-r 8 --max-steps 500

    # Resume from saved checkpoint
    python scripts/train_recovery_lora.py \
        --model Qwen/Qwen3-0.6B \
        --v2-checkpoint runs/v2_q4a4_r32/best_state_dict.pt \
        --train-data-hf NeelNanda/pile-10k \
        --resume-from runs/recovery_lora/recovery_step500.pt \
        --max-steps 1000

Pipeline order:
    1. Load base model
    2. Replace with V2 layers
    3. Load QAT checkpoint
    4. Enable recovery LoRA
    5. (KD mode) Load teacher model
    6. Freeze base, train LoRA
    7. Save checkpoint with LoRA

Apple-style adapter placement (default):
    - Attention: Q, V, O (skip K)
    - MLP: gate_proj, up_proj, down_proj
"""

import argparse
import os
import sys
from pathlib import Path

import torch

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# TPU SUPPORT
# See docs/TPU.md for debugging and troubleshooting guide
# =============================================================================

def get_tpu_device():
    """Get TPU device if available, else None."""
    try:
        import torch_xla
        return torch_xla.device()
    except ImportError:
        return None
    except RuntimeError as e:
        print(f"[WARN] TPU init failed: {e}")
        return None


def is_tpu_device(device) -> bool:
    """Check if device is TPU/XLA."""
    return 'xla' in str(device).lower()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train recovery LoRA adapters on quantized model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model args
    parser.add_argument("--model", type=str, required=True,
                       help="Model name or path (e.g., Qwen/Qwen3-0.6B)")
    parser.add_argument("--v2-checkpoint", type=str, required=True,
                       help="Path to V2 QAT checkpoint (.pt file)")
    parser.add_argument("--resume-from", type=str, default=None,
                       help="Resume training from saved checkpoint (loads LoRA weights)")
    parser.add_argument("--save-lora-only", action="store_true",
                       help="Save only LoRA weights in checkpoints (17MB vs 5GB, requires --v2-checkpoint to resume)")
    parser.add_argument("--output", type=str, default="runs/recovery_lora",
                       help="Output directory for checkpoints")

    # Data args (one of --train-data or --train-data-hf required)
    parser.add_argument("--train-data", type=str, default=None,
                       help="Path to training data (.jsonl or .pt)")
    parser.add_argument("--train-data-hf", type=str, default=None,
                       help="HuggingFace dataset name (e.g., NeelNanda/pile-10k)")
    parser.add_argument("--hf-subset", type=str, default=None,
                       help="HF dataset subset (e.g., wikitext-103-v1)")
    parser.add_argument("--hf-split", type=str, default="train",
                       help="HF dataset split (default: train)")
    parser.add_argument("--hf-text-field", type=str, default="text",
                       help="HF dataset text field (default: text)")
    parser.add_argument("--hf-max-samples", type=int, default=None,
                       help="Max samples from HF dataset (default: all)")

    # Template/tokenization args
    parser.add_argument("--template-mode", type=str, default="none",
                       choices=["none", "no-think", "think", "both", "all"],
                       help="Tokenization mode: none=raw text, no-think=chat template, "
                            "think=chat+thinking, both=mix no-think+think, "
                            "all=random from none/no-think/think (default: none)")
    parser.add_argument("--dataset-format", type=str, default="text",
                       choices=["text", "alpaca", "sharegpt"],
                       help="Dataset format for template parsing (default: text)")
    parser.add_argument("--generate-targets", action="store_true",
                       help="Generate training targets using reference model (base or --teacher). "
                            "Useful with think mode: model generates <think>...</think> content. "
                            "Slower but produces proper thinking tokens in training data.")
    parser.add_argument("--gen-max-tokens", type=int, default=512,
                       help="Max tokens to generate for --generate-targets (default: 512)")
    parser.add_argument("--gen-temperature", type=float, default=0.7,
                       help="Temperature for --generate-targets (default: 0.7)")
    parser.add_argument("--gen-top-p", type=float, default=0.9,
                       help="Top-p for --generate-targets (default: 0.9)")

    # LoRA args
    parser.add_argument("--recovery-r", type=int, default=8,
                       help="LoRA rank (start small: 8, increase if needed)")
    parser.add_argument("--recovery-alpha", type=float, default=None,
                       help="LoRA alpha (default: same as rank)")
    parser.add_argument("--mlp-only", action="store_true",
                       help="Only enable LoRA on MLP layers (recommended first)")
    parser.add_argument("--include-k-proj", action="store_true",
                       help="Include K projection (default: skip like Apple)")

    # Training mode
    parser.add_argument("--lora-mode", type=str, default="recover",
                       choices=["recover", "sft", "kd"],
                       help="Training mode: recover=CE on raw text (default), "
                            "sft=supervised fine-tuning on instruction/response, "
                            "kd=knowledge distillation from teacher")
    parser.add_argument("--teacher", type=str, default=None,
                       help="Teacher model for KD mode (e.g., Qwen/Qwen3-4B-Instruct)")
    parser.add_argument("--kd-temperature", type=float, default=2.0,
                       help="KD temperature for softening logits (default: 2.0)")
    parser.add_argument("--kd-cache-dir", type=str, default=None,
                       help="Path to precomputed KD cache directory (fast, no teacher needed). "
                            "Overrides --train-data-hf when specified.")
    parser.add_argument("--kd-alpha", type=float, default=0.5,
                       help="KD loss weight: total = alpha*KD + (1-alpha)*CE (default: 0.5)")
    parser.add_argument("--hard-top1", type=float, default=0.0,
                       help="Hard label top-1 weight for KD cache mode (default: 0.0)")
    parser.add_argument("--hard-top1-end", type=float, default=None,
                       help="Hard label top-1 end weight for annealing (default: same as --hard-top1)")
    parser.add_argument("--hard-full", type=float, default=0.0,
                       help="Hard label full vocab weight for KD cache mode (default: 0.0)")

    # Training args
    parser.add_argument("--lr", type=float, default=3e-4,
                       help="Learning rate (LoRA tolerates higher)")
    parser.add_argument("--max-steps", type=int, default=1000,
                       help="Maximum training steps")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Batch size")
    parser.add_argument("--seq-len", type=int, default=4096,
                       help="Sequence length (4K-8K recommended)")
    parser.add_argument("--warmup-steps", type=int, default=100,
                       help="Warmup steps")
    parser.add_argument("--constant-lr", action="store_true",
                       help="Use constant LR after warmup (no cosine decay)")
    parser.add_argument("--min-lr-ratio", type=float, default=0.1,
                       help="Minimum LR ratio for cosine decay (default: 0.1)")
    parser.add_argument("--weight-decay", type=float, default=0.0,
                       help="Weight decay (0 or small for LoRA)")
    parser.add_argument("--grad-clip", "--clip-grad-norm", type=float, default=1.0,
                       help="Max gradient norm for clipping (default: 1.0, 0=disable)")
    parser.add_argument("--dropout", type=float, default=0.0,
                       help="Dropout rate (default: 0.0, try 0.1)")
    parser.add_argument("--accumulation-steps", type=int, default=1,
                       help="Gradient accumulation steps")
    parser.add_argument("--mixed-precision", action="store_true",
                       help="Use mixed precision training (BF16 on CUDA, FP16 on MPS)")

    # Anchor KL regularizer
    parser.add_argument("--anchor-kl-weight", type=float, default=0.0,
                       help="Weight for anchor KL regularizer (0 = disabled)")
    parser.add_argument("--anchor-samples", type=int, default=32,
                       help="Number of anchor samples for KL")

    # Logging/saving
    parser.add_argument("--log-interval", type=int, default=50,
                       help="Steps between logging")
    parser.add_argument("--save-steps", type=int, default=500,
                       help="Steps between checkpoints")
    parser.add_argument("--keep-checkpoints", type=int, default=3,
                       help="Number of checkpoints to keep")

    # Quantization config (for loading checkpoint)
    parser.add_argument("--lut-size", type=int, default=16,
                       help="LUT size (4=2bit, 16=4bit)")
    parser.add_argument("--scale-rank", type=int, default=32,
                       help="Scale rank")
    parser.add_argument("--quantize-attn", action="store_true", default=True,
                       help="Quantize attention layers")
    parser.add_argument("--no-quantize-attn", dest="quantize_attn", action="store_false")

    # Wandb
    parser.add_argument("--wandb", action="store_true",
                       help="Log to wandb")
    parser.add_argument("--wandb-project", type=str, default="recovery-lora",
                       help="Wandb project name")
    parser.add_argument("--wandb-run", type=str, default=None,
                       help="Wandb run name")

    # Device
    parser.add_argument("--device", type=str, default="auto",
                       help="Device (auto, cuda, mps, cpu, tpu)")
    parser.add_argument("--tpu", action="store_true",
                       help="Use TPU (sets PJRT_DEVICE=TPU, requires torch_xla)")

    # Debug
    parser.add_argument("--debug", action="store_true",
                       help="Show detailed debug output (input sequences, tokens, etc.)")

    # Model dtype
    parser.add_argument("--dtype", type=str, default="float32",
                       choices=["float32", "float16", "bfloat16"],
                       help="Model dtype (float32, float16, bfloat16)")

    args = parser.parse_args()

    # Validate data args
    if args.kd_cache_dir:
        # KD cache mode - no other data source needed
        if not os.path.exists(args.kd_cache_dir):
            parser.error(f"KD cache directory not found: {args.kd_cache_dir}")
        print(f"Using KD cache: {args.kd_cache_dir}")
        # Auto-set lora_mode to kd if using cache
        if args.lora_mode != "kd":
            print(f"Note: Setting --lora-mode=kd for cache-based training")
            args.lora_mode = "kd"
    elif args.train_data is None and args.train_data_hf is None:
        parser.error("One of --train-data, --train-data-hf, or --kd-cache-dir is required")
    if args.train_data is not None and args.train_data_hf is not None:
        parser.error("Cannot specify both --train-data and --train-data-hf")

    # Validate lora-mode args
    if args.lora_mode == "kd" and args.teacher is None and args.kd_cache_dir is None:
        parser.error("--teacher or --kd-cache-dir is required for --lora-mode=kd")
    if args.teacher is not None and args.lora_mode != "kd":
        print(f"Warning: --teacher is ignored when --lora-mode={args.lora_mode}")
    if args.lora_mode == "sft" and args.dataset_format == "text":
        print("Warning: --lora-mode=sft works best with --dataset-format=alpaca or sharegpt")

    # Auto-enable generate_targets for think mode with text format
    # Without this, training data won't contain <think> tokens
    if args.template_mode in ["think", "both", "all"] and args.dataset_format == "text":
        if not args.generate_targets:
            print(f"Note: Auto-enabling --generate-targets for {args.template_mode} mode with text format")
            print("      (Required for <think> tokens in training data)")
            args.generate_targets = True

    # Determine device
    if args.tpu or args.device == "tpu":
        # TPU mode - use torch_xla
        device = get_tpu_device()
        if device is None:
            print("ERROR: TPU requested but torch_xla not available or TPU init failed")
            print("  Install with: pip install torch_xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html")
            sys.exit(1)
        print(f"Using device: TPU ({device})")
    elif args.device == "auto":
        # Try TPU first if PJRT_DEVICE is set
        if os.environ.get("PJRT_DEVICE") == "TPU":
            device = get_tpu_device()
            if device:
                print(f"Using device: TPU ({device})")
            else:
                print("PJRT_DEVICE=TPU set but torch_xla not available, falling back...")
                device = None
        else:
            device = None

        # Fallback chain: CUDA > MPS > CPU
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
            print(f"Using device: {device}")
    else:
        device = torch.device(args.device)
        print(f"Using device: {device}")

    # Helper to print memory usage
    def print_mem(label=""):
        import gc
        gc.collect()
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"  [MEM {label}] GPU: {alloc:.2f}GB allocated, {reserved:.2f}GB reserved")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # MPS doesn't have detailed memory API, use system memory
            import psutil
            mem = psutil.Process().memory_info().rss / 1024**3
            print(f"  [MEM {label}] Process RSS: {mem:.2f}GB")

    # Parse dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    model_dtype = dtype_map[args.dtype]

    # Load model and tokenizer
    print(f"\nLoading model: {args.model} (dtype={args.dtype})")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=model_dtype,
        trust_remote_code=True,
    )
    if args.debug:
        print_mem("after model load")

    # Replace with V2 layers
    print(f"\nReplacing with V2 layers...")
    from qat_lora.ane_qat_linear_v2 import (
        AnemllQuantConfigV2,
        replace_linear_with_anemll_v2,
        load_v2_checkpoint,
    )

    # Load config.json from checkpoint directory if exists (like test_inference.py)
    import json
    checkpoint_dir = Path(args.v2_checkpoint).parent
    config_json_path = checkpoint_dir / 'config.json'
    ckpt_config = {}
    if config_json_path.exists():
        with open(config_json_path) as f:
            ckpt_config = json.load(f)
        print(f"Loaded config from {config_json_path}")

    # Use config.json values with CLI overrides
    lut_size = args.lut_size if args.lut_size != 16 else 2 ** ckpt_config.get('lut_bits', 4)
    scale_rank = args.scale_rank if args.scale_rank != 32 else ckpt_config.get('scale_rank', 32)
    force_positive_scales = ckpt_config.get('force_positive_scales', False)
    magnitude_activation = ckpt_config.get('magnitude_activation', 'identity')

    print(f"V2 config: lut_size={lut_size}, scale_rank={scale_rank}, "
          f"force_positive_scales={force_positive_scales}, magnitude_activation={magnitude_activation}")

    mlp_config = AnemllQuantConfigV2(
        lut_size=lut_size,
        scale_rank=scale_rank,
        force_positive_scales=force_positive_scales,
        magnitude_activation=magnitude_activation,
        use_ste_fp16=True,  # Enable FP16 emulation for ANE compatibility
    )
    attn_config = AnemllQuantConfigV2(
        lut_size=lut_size,
        scale_rank=scale_rank,
        force_positive_scales=force_positive_scales,
        magnitude_activation=magnitude_activation,
        use_ste_fp16=True,  # Enable FP16 emulation for ANE compatibility
    )

    replace_linear_with_anemll_v2(
        model,
        mlp_config=mlp_config,
        attn_config=attn_config,
        quantize_attn=args.quantize_attn,
        verbose=True,
        skip_init=True,  # Skip SVD, we'll load checkpoint
    )
    if args.debug:
        print_mem("after V2 replace")

    # Load QAT checkpoint
    print(f"\nLoading V2 checkpoint: {args.v2_checkpoint}")
    load_v2_checkpoint(model, args.v2_checkpoint, device=device, verbose=True)
    if args.debug:
        print_mem("after checkpoint load")

    # Move to device
    model.to(device)
    if args.debug:
        print_mem("after model.to(device)")

    # Train recovery LoRA
    from qat_lora.layer_qat import train_recovery_lora

    # Build config dict for saving with checkpoints
    import math
    lut_bits = int(math.log2(lut_size))
    recovery_config = {
        'version': 'v2',
        'model_id': args.model,
        'lut_bits': lut_bits,
        'scale_rank': scale_rank,
        'force_positive_scales': force_positive_scales,
        'magnitude_activation': magnitude_activation,
        'recovery_lora': True,
        'recovery_r': args.recovery_r,
        'recovery_alpha': args.recovery_alpha or args.recovery_r,
        'mlp_only': args.mlp_only,
        'skip_k_proj': not args.include_k_proj,
        'lora_mode': args.lora_mode,
    }
    if args.lora_mode == 'kd':
        recovery_config['teacher'] = args.teacher
        recovery_config['kd_temperature'] = args.kd_temperature
        recovery_config['kd_alpha'] = args.kd_alpha
        recovery_config['kd_cache_dir'] = args.kd_cache_dir

    print(f"\nStarting recovery LoRA training (mode={args.lora_mode})...")
    results = train_recovery_lora(
        model=model,
        train_data=args.train_data,
        train_data_hf=args.train_data_hf,
        hf_subset=args.hf_subset,
        hf_split=args.hf_split,
        hf_text_field=args.hf_text_field,
        hf_max_samples=args.hf_max_samples,
        template_mode=args.template_mode,
        dataset_format=args.dataset_format,
        generate_targets=args.generate_targets,
        gen_max_tokens=args.gen_max_tokens,
        gen_temperature=args.gen_temperature,
        gen_top_p=args.gen_top_p,
        reference_model_id=args.teacher or args.model,  # Use teacher if specified, else base
        kd_cache_dir=args.kd_cache_dir,
        device=device,
        tokenizer=tokenizer,
        recovery_r=args.recovery_r,
        recovery_alpha=args.recovery_alpha,
        mlp_only=args.mlp_only,
        skip_k_proj=not args.include_k_proj,
        lr=args.lr,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        warmup_steps=args.warmup_steps,
        constant_lr=args.constant_lr,
        min_lr_ratio=args.min_lr_ratio,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        accumulation_steps=args.accumulation_steps,
        mixed_precision=args.mixed_precision,
        dropout=args.dropout,
        log_interval=args.log_interval,
        save_dir=args.output,
        save_steps=args.save_steps,
        keep_checkpoints=args.keep_checkpoints,
        anchor_kl_weight=args.anchor_kl_weight,
        anchor_samples=args.anchor_samples,
        resume_from=args.resume_from,
        save_lora_only=args.save_lora_only,
        config=recovery_config,
        lora_mode=args.lora_mode,
        teacher_model=args.teacher,
        kd_temperature=args.kd_temperature,
        kd_alpha=args.kd_alpha,
        hard_top1_weight=args.hard_top1,
        hard_top1_end=args.hard_top1_end,
        hard_full_weight=args.hard_full,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run,
        verbose=True,
        debug=args.debug,
    )

    # Save final checkpoint
    os.makedirs(args.output, exist_ok=True)
    final_path = os.path.join(args.output, "final_recovery_lora.pt")
    torch.save(model.state_dict(), final_path)
    print(f"\nFinal checkpoint saved: {final_path}")

    # Print summary
    print(f"\n=== Training Complete ===")
    print(f"Best loss: {results['best_loss']:.4f}")
    print(f"Steps: {results['steps']}")
    print(f"Time: {results['time_sec']:.1f}s")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
