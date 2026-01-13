#!/usr/bin/env python3
"""
Minimal V2 Training Script - Uses pre-cached KD data.

Usage:
    python scripts/train_v2_simple.py \
        --v1-checkpoint runs/tmp/backup_mlp_e2e_w_0.3824.pt \
        --cache-dir caches/alpaca_chat_think_both_L128_K128_R1024
"""

import argparse
import os
import sys
import gc
from pathlib import Path
from datetime import datetime

REPO_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_DIR))

import torch
from transformers import AutoModelForCausalLM

# =============================================================================
# TPU SUPPORT
# =============================================================================

def get_device():
    """Get best available device: TPU > CUDA > CPU."""
    # Try TPU first
    try:
        import torch_xla
        device = torch_xla.device()
        return device, 'tpu'
    except ImportError:
        pass
    except RuntimeError as e:
        print(f"[WARN] TPU init failed: {e}")

    # Fallback to CUDA
    if torch.cuda.is_available():
        return torch.device('cuda'), 'cuda'

    return torch.device('cpu'), 'cpu'


def is_tpu_device(device) -> bool:
    """Check if device is TPU/XLA."""
    return 'xla' in str(device).lower()


def _maybe_disable_kv_cache(model, enabled: bool, verbose: bool = True):
    """
    Disable HuggingFace KV cache on the model.

    This prevents TPU HBM OOM at long seq_len (L>=1024) by avoiding
    the 192-256MB KV buffer allocations per batch.

    Args:
        model: HuggingFace model
        enabled: If True, disable KV cache; if False, do nothing
        verbose: Print status message
    """
    if not enabled:
        return

    disabled = False

    # Main config
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False
        disabled = True

    # Some HF models also consult generation_config
    if hasattr(model, "generation_config") and hasattr(model.generation_config, "use_cache"):
        model.generation_config.use_cache = False

    if verbose and disabled:
        print("[KV CACHE] Disabled: model.config.use_cache=False")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='V2 STE-FP16 Training (Simple)')
    parser.add_argument('--v1-checkpoint', type=str, default=None, help='V1 checkpoint (for V1->V2 conversion)')
    parser.add_argument('--v2-checkpoint', type=str, default=None, help='V2 checkpoint (skip conversion, load directly)')
    parser.add_argument('--from-scratch', action='store_true', help='Train V2 from scratch (no V1, random init)')
    parser.add_argument('--cache-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='runs/v2_output')
    parser.add_argument('--model-id', type=str, default='Qwen/Qwen3-0.6B')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-steps', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=5e-5)  # Lower LR for stability
    parser.add_argument('--hard-top1', type=float, default=0.2, help='Hard label top-1 weight (start)')
    parser.add_argument('--hard-top1-end', type=float, default=None,
                        help='Hard label top-1 weight at end (for annealing). If set, decays from --hard-top1 to this.')
    parser.add_argument('--hard-full', type=float, default=0.00005, help='Hard label full vocab weight')
    parser.add_argument('--temperature', type=float, default=2.0, help='KD temperature (default: 2.0)')
    parser.add_argument('--warmup-steps', type=int, default=100, help='LR warmup steps (default: 100)')
    parser.add_argument('--constant-lr', action='store_true',
                        help='Use constant LR (disable cosine decay)')
    parser.add_argument('--eval-steps', type=int, default=100, help='Eval every N steps (default: 100)')
    parser.add_argument('--eval-samples', type=int, default=None, help='Eval samples (default: 40, 0=skip)')
    parser.add_argument('--g-only', action='store_true', help='Train only rank_magnitude (G), freeze A and B')
    parser.add_argument('--freeze-mags', action='store_true',
                        help='Freeze rank_magnitude (G), train only A and B. Opposite of --g-only.')
    parser.add_argument('--freeze-mags-mlp', action='store_true',
                        help='Freeze rank_magnitude for MLP layers only (attention mags still trainable)')
    parser.add_argument('--freeze-all', action='store_true',
                        help='Snap + freeze ALL V2 params (scale_A, scale_B, rank_magnitude) for FP16 export. Nothing trains.')
    parser.add_argument('--mlp-only', action='store_true', help='Train only MLP layers, freeze attention')
    parser.add_argument('--attn-only', action='store_true', help='Train only attention layers, freeze MLP (for 2-phase training)')
    parser.add_argument('--train-norms-only', action='store_true',
                        help='Train ONLY LayerNorm weights (freeze all QAT params). '
                             'Use with --no-full-logits for safe L>=1024 TPU training. '
                             'Targets: model.norm, input_layernorm, post_attention_layernorm (56 tensors).')
    parser.add_argument('--save-steps', type=int, default=0, help='Save checkpoint every N steps (0=disabled)')
    parser.add_argument('--keep-checkpoints', type=int, default=0,
                        help='Keep only the last N checkpoints (0=keep all). Useful for long runs.')
    parser.add_argument('--min-lr-ratio', type=float, default=0.1,
                        help='Minimum LR as ratio of peak LR for cosine annealing (default: 0.1)')
    parser.add_argument('--clip-grad-norm', type=float, default=1.0,
                        help='Max gradient norm for clipping (default: 1.0, 0=disable)')
    # Regularization
    parser.add_argument('--weight-decay', type=float, default=0.0, help='Weight decay for AdamW (default: 0.0, try 0.01)')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (default: 0.0, try 0.1)')
    parser.add_argument('--accumulation-steps', type=int, default=1,
                        help='Gradient accumulation steps (default: 1). Effective batch = batch_size * accumulation_steps')
    # Wandb logging
    parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='qwen3-qat', help='W&B project name')
    parser.add_argument('--wandb-run', type=str, default=None, help='W&B run name (default: auto)')
    # Google Drive upload (legacy)
    parser.add_argument('--gdrive-dir', type=str, default=None,
                        help='[DEPRECATED] Use --upload instead. Google Drive directory to upload FP32 checkpoint')
    # New upload flag using gdrive_sync API
    parser.add_argument('--upload', action='store_true',
                        help='Auto-upload run to Google Drive after successful training (uses gdrive_sync.py)')
    parser.add_argument('--upload-exclude', type=str, action='append', default=None,
                        help='Glob patterns to exclude from upload (default: *checkpoint*). Can be used multiple times.')
    parser.add_argument('--upload-all', action='store_true',
                        help='Upload everything without exclusions (overrides --upload-exclude)')
    # Memory optimization
    parser.add_argument('--gradient-checkpointing', action='store_true',
                        help='Enable gradient checkpointing (trades ~15%% speed for ~40%% memory)')
    # Training precision
    parser.add_argument('--dtype', type=str, default='fp32', choices=['fp32', 'bf16', 'fp16'],
                        help='Training dtype: fp32 (default, ANE-safe), bf16 (2x faster), fp16 (fastest but risky)')
    parser.add_argument('--mixed-precision', action='store_true', default=True,
                        help='Mixed precision: FP32 master weights + BF16 compute (default: enabled)')
    parser.add_argument('--no-mixed-precision', dest='mixed_precision', action='store_false',
                        help='Disable mixed precision (use pure FP32)')
    # TPU support
    parser.add_argument('--tpu', action='store_true',
                        help='Force TPU mode (auto-detected if available)')
    parser.add_argument('--xla-cache-dir', type=str, default=None,
                        help='XLA compilation cache directory (speeds up TPU restarts)')
    parser.add_argument('--disable-kv-cache', action='store_true',
                        help='Force HF KV cache off during training: sets model.config.use_cache=False. '
                             'Recommended for TPU and/or long seq_len (L>=1024) to avoid HBM OOM.')
    # Quantization config
    parser.add_argument('--config', type=str, default='q2a4',
                        choices=['q2a4', 'q4a4', 'q4a4_r32', 'q4_r32', 'q2a2'],
                        help='Quantization config preset (default: q2a4). q4_r32 is alias for q4a4_r32')
    parser.add_argument('--mlp-lut', type=int, default=None, help='Override MLP LUT size (2-bit=4, 4-bit=16)')
    parser.add_argument('--mlp-rank', type=int, default=None, help='Override MLP scale rank')
    parser.add_argument('--attn-lut', type=int, default=None, help='Override Attention LUT size')
    parser.add_argument('--attn-rank', type=int, default=None, help='Override Attention scale rank')
    parser.add_argument('--group-size', type=int, default=32, help='Group size for scale init (default: 32)')
    parser.add_argument('--fast-init', action='store_true',
                        help='Skip SVD-based scale initialization (faster, worse initial loss)')
    # Anchor KL regularization (prevents drift from reference checkpoint)
    parser.add_argument('--anchor-ckpt', type=str, default=None,
                        help='Checkpoint to use as anchor teacher (prevents drift from this behavior)')
    parser.add_argument('--anchor-kl-weight', type=float, default=0.01,
                        help='Weight of anchor KL term (default: 0.01, 0=disabled)')
    parser.add_argument('--anchor-samples', type=int, default=16,
                        help='Number of fixed anchor samples to cache logits for (default: 16)')
    parser.add_argument('--anchor-interval', type=int, default=1,
                        help='Compute anchor KL every N steps (default: 1=every step, 10=less overhead)')
    # Auto snap+freeze rank_magnitude (CPU audit at save checkpoints)
    parser.add_argument('--auto-snap-mags', action='store_true',
                        help='Enable auto snap+freeze of rank_magnitude when stable (CPU audit at saves)')
    parser.add_argument('--auto-snap-target', type=str, default='mlp', choices=['mlp', 'attn', 'all'],
                        help='Target layers for auto-snap: mlp (84 layers), attn (112 layers), or all (196 layers)')
    parser.add_argument('--auto-snap-threshold', type=float, default=0.05,
                        help='Max abs delta between saves to consider stable (default: 0.05)')
    parser.add_argument('--auto-snap-patience', type=int, default=2,
                        help='Consecutive stable saves required before triggering snap+freeze (default: 2)')
    parser.add_argument('--auto-snap-start-step', type=int, default=100,
                        help='Don\'t audit before this step (default: 100)')
    parser.add_argument('--auto-snap-min-saves', type=int, default=2,
                        help='Minimum save checkpoints before eligible (default: 2)')
    parser.add_argument('--auto-snap-dry-run', action='store_true',
                        help='Audit and log but don\'t actually freeze (for testing)')
    parser.add_argument('--auto-snap-log-json', action='store_true',
                        help='Write audit JSON files at each save checkpoint')
    # Memory debug (TPU/XLA-safe)
    parser.add_argument('--mem-debug', action='store_true',
                        help='Enable TPU HBM memory logging (OFF by default)')
    parser.add_argument('--mem-debug-level', type=str, default='basic',
                        choices=['basic', 'tensors', 'metrics', 'hlo'],
                        help='Memory debug level: basic=HBM snapshots, tensors=+size estimates, '
                             'metrics=+XLA compile stats, hlo=+HLO dumps (heavy)')
    parser.add_argument('--mem-debug-phase', type=str, default='warmup,save',
                        help='When to log: warmup,train,save,all (comma-separated, default: warmup,save)')
    parser.add_argument('--mem-debug-interval', type=int, default=1,
                        help='Log every N optimizer steps (default: 1)')
    parser.add_argument('--mem-debug-json', type=str, default=None,
                        help='Path to append JSONL memory records')
    parser.add_argument('--mem-debug-tag', type=str, default=None,
                        help='Optional run tag for log filtering')
    parser.add_argument('--mem-debug-no-xla-metrics', action='store_true',
                        help='Skip XLA metrics calls (if they perturb compilation)')
    parser.add_argument('--mem-debug-step-axis', type=str, default='opt',
                        choices=['micro', 'opt'],
                        help='Step axis for interval filtering: micro (gradient step) or opt (optimizer step, default)')
    # Sparse logits (L1024+ TPU memory safety)
    parser.add_argument('--no-full-logits', action='store_true',
                        help='Prevent any full-vocab [B,L,V] logits materialization. Forces hard_top1=0, hard_full=0, '
                             'and uses sparse anchor-KL. Required for L>=1024 on TPU v6e-1 (16GB HBM).')
    parser.add_argument('--sampled-ce-weight', type=float, default=0.0,
                        help='Weight for sampled CE loss on K+R candidates (sparse alternative to hard_top1). '
                             'Uses topk + random negatives from cache. Default: 0.0 (disabled).')
    parser.add_argument('--sampled-negatives', type=int, default=64,
                        help='Number of random negative tokens to sample if cache lacks rand_idx (default: 64). '
                             'Ignored if cache already has rand_idx/rand_logits.')
    args = parser.parse_args()

    # Validate inputs - need v1, v2 checkpoint, or from-scratch
    if args.anchor_ckpt:
        assert os.path.exists(args.anchor_ckpt), f"Anchor checkpoint not found: {args.anchor_ckpt}"
    if args.v2_checkpoint:
        assert os.path.exists(args.v2_checkpoint), f"V2 checkpoint not found: {args.v2_checkpoint}"
    elif args.v1_checkpoint:
        assert os.path.exists(args.v1_checkpoint), f"V1 checkpoint not found: {args.v1_checkpoint}"
    elif args.from_scratch:
        pass  # No checkpoint needed
    else:
        raise ValueError("Must specify --v1-checkpoint, --v2-checkpoint, or --from-scratch")
    assert os.path.exists(args.cache_dir), f"Cache dir not found: {args.cache_dir}"

    # Validate auto-snap config (done early before imports to fail fast)
    if args.auto_snap_mags:
        # Check conflicts manually before AutoSnapState import
        if args.freeze_mags:
            raise ValueError("--auto-snap-mags conflicts with --freeze-mags")
        # Allow --freeze-mags-mlp with --auto-snap-target attn (2-phase training)
        if args.freeze_mags_mlp and args.auto_snap_target != 'attn':
            raise ValueError("--auto-snap-mags conflicts with --freeze-mags-mlp (unless --auto-snap-target attn)")
        if args.freeze_all:
            raise ValueError("--auto-snap-mags conflicts with --freeze-all")
        if args.g_only:
            raise ValueError("--auto-snap-mags conflicts with --g-only (auto-snap targets mags)")
        if args.save_steps <= 0:
            raise ValueError("--auto-snap-mags requires --save-steps > 0 (audit happens at saves)")

    # Validate train-norms-only mode
    if args.train_norms_only:
        if args.g_only:
            raise ValueError("--train-norms-only conflicts with --g-only")
        if args.mlp_only:
            raise ValueError("--train-norms-only conflicts with --mlp-only")
        if args.attn_only:
            raise ValueError("--train-norms-only conflicts with --attn-only")
        # Recommend --no-full-logits for L>=1024 TPU safety
        if not args.no_full_logits:
            print("[WARN] Consider using --no-full-logits with --train-norms-only for L>=1024 TPU safety")

    # Device detection (TPU > CUDA > CPU)
    device, device_type = get_device()
    is_tpu = device_type == 'tpu' or args.tpu

    # If --tpu flag but device not detected as TPU, force XLA device
    if args.tpu and device_type != 'tpu':
        try:
            import torch_xla.core.xla_model as xm
            device = xm.xla_device()
            device_type = 'tpu'
            print(f"[TPU] Forced XLA device via --tpu flag: {device}")
        except ImportError:
            raise RuntimeError("--tpu specified but torch_xla not installed")

    # XLA persistent compilation cache (speeds up TPU restarts)
    if args.xla_cache_dir:
        try:
            import torch_xla.runtime as xr
            os.makedirs(args.xla_cache_dir, exist_ok=True)
            xr.initialize_cache(args.xla_cache_dir, readonly=False)
            print(f"[XLA] Cache initialized: {args.xla_cache_dir}")
        except ImportError:
            print("[XLA] Warning: torch_xla.runtime not available, cache disabled")
        except Exception as e:
            print(f"[XLA] Warning: Cache init failed: {e}")

    # TPU: BF16 is native, FP32 is slow
    if is_tpu and args.dtype == 'fp32' and not args.mixed_precision:
        print("[TPU] Forcing BF16 (FP32 not recommended on TPU)")
        args.dtype = 'bf16'

    # TPU mixed precision: FP32 master weights + BF16 compute via autocast
    if is_tpu and args.mixed_precision:
        print("[TPU] Mixed Precision: FP32 master weights + BF16 compute (autocast)")

    # Dtype mapping
    dtype_map = {
        'fp32': torch.float32,
        'bf16': torch.bfloat16,
        'fp16': torch.float16,
    }

    # Mixed precision: FP32 master weights + BF16 compute
    if args.mixed_precision:
        train_dtype = torch.float32  # Always FP32 for master weights
        print(f"Device: {device_type.upper()}" + (f" ({device})" if is_tpu else ""))
        print(f"Training: Mixed Precision (FP32 weights + BF16 compute)")
    else:
        train_dtype = dtype_map[args.dtype]
        print(f"Device: {device_type.upper()}" + (f" ({device})" if is_tpu else ""))
        print(f"Training dtype: {args.dtype}")
    if args.v2_checkpoint:
        print(f"V2 checkpoint: {args.v2_checkpoint}")
    elif args.from_scratch:
        print("Mode: FROM SCRATCH (no V1)")
    else:
        print(f"V1 checkpoint: {args.v1_checkpoint}")
    print(f"Cache dir: {args.cache_dir}")

    # Quantization config presets
    CONFIG_PRESETS = {
        'q2a4': {'mlp_lut': 4, 'mlp_rank': 32, 'attn_lut': 16, 'attn_rank': 8},   # 2-bit MLP, 4-bit Attn
        'q4a4': {'mlp_lut': 16, 'mlp_rank': 4, 'attn_lut': 16, 'attn_rank': 4},    # 4-bit both, rank=4
        'q4a4_r32': {'mlp_lut': 16, 'mlp_rank': 32, 'attn_lut': 16, 'attn_rank': 32},  # 4-bit both, rank=32
        'q4_r32': {'mlp_lut': 16, 'mlp_rank': 32, 'attn_lut': 16, 'attn_rank': 32},  # Alias for q4a4_r32
        'q2a2': {'mlp_lut': 4, 'mlp_rank': 32, 'attn_lut': 4, 'attn_rank': 32},    # 2-bit both
    }

    # Load preset and apply overrides
    preset = CONFIG_PRESETS[args.config]
    MLP_LUT_SIZE = args.mlp_lut if args.mlp_lut is not None else preset['mlp_lut']
    MLP_RANK = args.mlp_rank if args.mlp_rank is not None else preset['mlp_rank']
    ATTN_LUT_SIZE = args.attn_lut if args.attn_lut is not None else preset['attn_lut']
    ATTN_RANK = args.attn_rank if args.attn_rank is not None else preset['attn_rank']
    GROUP_SIZE = args.group_size

    # Config name for display
    config_name = args.config.upper()
    if args.mlp_lut or args.mlp_rank or args.attn_lut or args.attn_rank:
        config_name += " (custom)"

    print(f"\n{config_name} Config:")
    print(f"  MLP: lut={MLP_LUT_SIZE}, rank={MLP_RANK}, group={GROUP_SIZE}")
    print(f"  Attn: lut={ATTN_LUT_SIZE}, rank={ATTN_RANK}, group={GROUP_SIZE}")

    # Import after path setup
    from qat_lora import (
        AnemllQuantConfig,
        AnemllQuantConfigV2,
        replace_linear_with_anemll,
        replace_linear_with_anemll_v2,
        freeze_Q_all,
        train_e2e,
    )
    from qat_lora.ane_qat_linear import AnemllQATLinear
    from qat_lora.ane_qat_linear_v2 import AnemllQATLinearV2
    from qat_lora.auto_snap_mags import AutoSnapState, validate_auto_snap_config
    from qat_lora.mem_debug import MemDebugConfig

    # =========================================================================
    # Load model (V2 checkpoint OR V1->V2 conversion)
    # =========================================================================
    os.makedirs(args.output_dir, exist_ok=True)

    import time

    # Only use STE-FP16 when training in FP32 (for ANE compatibility)
    use_ste = (args.dtype == 'fp32')

    if args.v2_checkpoint:
        # Direct V2 load (skip conversion)
        print("\n[1/3] Loading V2 checkpoint directly...")

        t0 = time.time()
        print("  Loading base model...", end=" ", flush=True)
        v2_model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=train_dtype,  # Training dtype (fp32/bf16/fp16)
            trust_remote_code=True,
        )
        print(f"done ({time.time()-t0:.1f}s)")

        t0 = time.time()
        print("  Replacing with V2 layers...", end=" ", flush=True)
        v2_mlp_config = AnemllQuantConfigV2(
            lut_size=MLP_LUT_SIZE,
            scale_rank=MLP_RANK,
            group_size=GROUP_SIZE,
            force_positive_scales=False,
            magnitude_activation='identity',
            use_ste_fp16=use_ste,
        )
        v2_attn_config = AnemllQuantConfigV2(
            lut_size=ATTN_LUT_SIZE,
            scale_rank=ATTN_RANK,
            group_size=GROUP_SIZE,
            force_positive_scales=False,
            magnitude_activation='identity',
            use_ste_fp16=use_ste,
        )

        replace_linear_with_anemll_v2(
            v2_model,
            mlp_config=v2_mlp_config,
            attn_config=v2_attn_config,
            quantize_attn=True,
            quantize_lm_head=False,
            skip_init=True,  # Always skip for v2 checkpoint (will load state)
        )
        print(f"done ({time.time()-t0:.1f}s)")

        t0 = time.time()
        print(f"  Loading checkpoint ({args.v2_checkpoint})...", end=" ", flush=True)
        state_dict = torch.load(args.v2_checkpoint, map_location='cpu', weights_only=False)
        v2_model.load_state_dict(state_dict, strict=False)

        # Manually load _Q buffers (None buffers don't load automatically)
        q_loaded = 0
        for name, m in v2_model.named_modules():
            if isinstance(m, AnemllQATLinearV2):
                q_key = f"{name}._Q"
                if q_key in state_dict and m._Q is None:
                    m.register_buffer("_Q", state_dict[q_key])
                    q_loaded += 1
        print(f"done ({time.time()-t0:.1f}s)")
        if q_loaded > 0:
            print(f"  Manually loaded {q_loaded} _Q buffers")

        # Freeze Q BEFORE moving to device (avoids XLA compilations on TPU)
        # Skip if checkpoint already has _Q
        has_Q = any(hasattr(m, '_Q') and m._Q is not None
                    for m in v2_model.modules() if hasattr(m, '_Q'))
        if not has_Q:
            print("  (No _Q in checkpoint, running freeze_Q)")
            freeze_Q_all(v2_model)

        t0 = time.time()
        print(f"  Moving to {device_type.upper()} ({args.dtype})...", end=" ", flush=True)
        v2_model.to(device=device, dtype=train_dtype)
        print(f"done ({time.time()-t0:.1f}s)")

    elif args.from_scratch:
        # Train V2 from scratch (no V1)
        print("\n[1/2] Creating V2 model from scratch...")

        t0 = time.time()
        print("  Loading base model...", end=" ", flush=True)
        v2_model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=train_dtype,  # Training dtype (fp32/bf16/fp16)
            trust_remote_code=True,
        )
        print(f"done ({time.time()-t0:.1f}s)")

        t0 = time.time()
        print("  Replacing with V2 layers...", end=" ", flush=True)
        v2_mlp_config = AnemllQuantConfigV2(
            lut_size=MLP_LUT_SIZE,
            scale_rank=MLP_RANK,
            group_size=GROUP_SIZE,
            force_positive_scales=False,
            magnitude_activation='identity',
            use_ste_fp16=use_ste,
        )
        v2_attn_config = AnemllQuantConfigV2(
            lut_size=ATTN_LUT_SIZE,
            scale_rank=ATTN_RANK,
            group_size=GROUP_SIZE,
            force_positive_scales=False,
            magnitude_activation='identity',
            use_ste_fp16=use_ste,
        )

        replace_linear_with_anemll_v2(
            v2_model,
            mlp_config=v2_mlp_config,
            attn_config=v2_attn_config,
            quantize_attn=True,
            quantize_lm_head=False,
            skip_init=args.fast_init,  # Skip SVD for faster init (worse initial loss)
        )
        print(f"done ({time.time()-t0:.1f}s)")

        # Freeze Q BEFORE moving to device (avoids XLA compilations on TPU)
        freeze_Q_all(v2_model)

        t0 = time.time()
        print(f"  Moving to {device_type.upper()} ({args.dtype})...", end=" ", flush=True)
        v2_model.to(device=device, dtype=train_dtype)
        print(f"done ({time.time()-t0:.1f}s)")

        # Save initial state
        initial_path = f"{args.output_dir}/v2_scratch_initial.pt"
        torch.save(v2_model.state_dict(), initial_path)
        print(f"  Saved initial V2 to {initial_path}")

    else:
        # V1 -> V2 conversion
        print("\n[1/4] Loading V1 checkpoint...")

        v1_model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

        v1_mlp_config = AnemllQuantConfig(lut_size=MLP_LUT_SIZE, scale_rank=MLP_RANK)
        v1_attn_config = AnemllQuantConfig(lut_size=ATTN_LUT_SIZE, scale_rank=ATTN_RANK)

        replace_linear_with_anemll(
            v1_model,
            mlp_config=v1_mlp_config,
            attn_config=v1_attn_config,
            quantize_attn=True,
            quantize_lm_head=False,
        )

        state_dict = torch.load(args.v1_checkpoint, map_location='cpu', weights_only=False)
        v1_model.load_state_dict(state_dict, strict=False)
        v1_model.to(device)
        print("  V1 model loaded")

        # Create V2 model and convert
        print("\n[2/4] Creating V2 model...")

        v2_model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=train_dtype,  # Training dtype (fp32/bf16/fp16)
            trust_remote_code=True,
        )

        v2_mlp_config = AnemllQuantConfigV2(
            lut_size=MLP_LUT_SIZE,
            scale_rank=MLP_RANK,
            group_size=GROUP_SIZE,
            force_positive_scales=False,
            magnitude_activation='identity',
            use_ste_fp16=use_ste,
        )
        v2_attn_config = AnemllQuantConfigV2(
            lut_size=ATTN_LUT_SIZE,
            scale_rank=ATTN_RANK,
            group_size=GROUP_SIZE,
            force_positive_scales=False,
            magnitude_activation='identity',
            use_ste_fp16=use_ste,
        )

        replace_linear_with_anemll_v2(
            v2_model,
            mlp_config=v2_mlp_config,
            attn_config=v2_attn_config,
            quantize_attn=True,
            quantize_lm_head=False,
        )

        # Convert V1 -> V2 (simple norm-based conversion)
        print("  Converting V1 -> V2...")

        converted = 0
        for (name_v1, m_v1), (name_v2, m_v2) in zip(
            v1_model.named_modules(), v2_model.named_modules()
        ):
            if isinstance(m_v1, AnemllQATLinear) and isinstance(m_v2, AnemllQATLinearV2):
                m_v2.lut.data.copy_(m_v1.lut.data.float())
                m_v2.weight.data.copy_(m_v1.weight.data.float())

                # Preserve V1 factorization (norm-based)
                A = m_v1.scale_A.float()  # [out, rank]
                B = m_v1.scale_B[:, :m_v1.in_features].float()  # [rank, in]

                A_norms = A.norm(dim=0, keepdim=True).clamp(min=1e-6)
                B_norms = B.norm(dim=1, keepdim=True).clamp(min=1e-6)

                A_dir = A / A_norms
                B_dir = B / B_norms
                magnitude = (A_norms.squeeze() * B_norms.squeeze())

                m_v2.scale_A.data.copy_(A_dir)
                m_v2.scale_B.data.copy_(B_dir)
                m_v2.rank_magnitude.data.copy_(magnitude)
                converted += 1

        print(f"  Converted {converted} layers")

        # Free V1
        del v1_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Freeze Q BEFORE moving to device (avoids XLA compilations on TPU)
        freeze_Q_all(v2_model)

        print(f"  Moving to {device_type.upper()} ({args.dtype})...", end=" ", flush=True)
        v2_model.to(device=device, dtype=train_dtype)
        print("done")

        # Save initial V2 state (before training)
        initial_path = f"{args.output_dir}/v2_initial.pt"
        torch.save(v2_model.state_dict(), initial_path)
        print(f"  Saved initial V2 to {initial_path}")

    # =========================================================================
    # KV CACHE (disable for TPU / long seq_len to avoid HBM OOM)
    # =========================================================================
    _maybe_disable_kv_cache(v2_model, args.disable_kv_cache)

    # =========================================================================
    # GRADIENT CHECKPOINTING (optional memory optimization)
    # =========================================================================
    if args.gradient_checkpointing:
        # Skip on TPU - transformers uses torch.xla which conflicts with torch_xla
        if is_tpu:
            print("\n[*] Gradient checkpointing: SKIPPED (TPU incompatible with transformers)")
        elif hasattr(v2_model, 'gradient_checkpointing_enable'):
            # use_reentrant=False is required when inputs don't have requires_grad
            v2_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            print("\n[*] Gradient checkpointing enabled (use_reentrant=False, trades ~15% speed for ~40% memory)")
        else:
            print("\n[!] Warning: Model does not support gradient_checkpointing_enable()")

    # =========================================================================
    # STEP 3: Train (freeze_Q already done before moving to device)
    # =========================================================================
    print("\n[2/3] Training with STE-FP16..." if args.v2_checkpoint else "\n[3/4] Training with STE-FP16...")
    # Note: freeze_Q_all is called BEFORE moving to device to avoid XLA compilations
    # train_e2e handles requires_grad based on train_scales, train_g_only, train_mlp_only

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build wandb config with run info
    wandb_config = {
        'model_id': args.model_id,
        'mlp_lut_size': MLP_LUT_SIZE,
        'mlp_rank': MLP_RANK,
        'attn_lut_size': ATTN_LUT_SIZE,
        'attn_rank': ATTN_RANK,
        'group_size': GROUP_SIZE,
        'v1_checkpoint': args.v1_checkpoint,
        'v2_checkpoint': args.v2_checkpoint,
        'from_scratch': args.from_scratch,
        'cache_dir': args.cache_dir,
        'dtype': args.dtype,
        'use_ste_fp16': use_ste,
        'mixed_precision': args.mixed_precision,
        'device_type': device_type,
        'is_tpu': is_tpu,
        'weight_decay': args.weight_decay,
        'dropout': args.dropout,
        'accumulation_steps': args.accumulation_steps,
        'anchor_ckpt': args.anchor_ckpt,
        'anchor_kl_weight': args.anchor_kl_weight if args.anchor_ckpt else 0.0,
        'anchor_samples': args.anchor_samples,
        'anchor_interval': args.anchor_interval,
        # Auto-snap config
        'auto_snap_mags': args.auto_snap_mags,
        'auto_snap_target': args.auto_snap_target if args.auto_snap_mags else None,
        'auto_snap_threshold': args.auto_snap_threshold if args.auto_snap_mags else None,
        'auto_snap_patience': args.auto_snap_patience if args.auto_snap_mags else None,
    }

    # Create AutoSnapState if enabled
    auto_snap_state = None
    if args.auto_snap_mags:
        auto_snap_state = AutoSnapState(
            enabled=True,
            target=args.auto_snap_target,
            threshold=args.auto_snap_threshold,
            patience=args.auto_snap_patience,
            start_step=args.auto_snap_start_step,
            min_saves=args.auto_snap_min_saves,
            dry_run=args.auto_snap_dry_run,
            log_json=args.auto_snap_log_json,
        )
        print(f"\n[AutoSnap] Enabled:")
        target_layers = {'mlp': 84, 'attn': 112, 'all': 196}.get(args.auto_snap_target, 196)
        print(f"  Target: {args.auto_snap_target} ({target_layers} layers)")
        print(f"  Threshold: {args.auto_snap_threshold} (max abs delta)")
        print(f"  Patience: {args.auto_snap_patience} consecutive stable saves")
        print(f"  Start step: {args.auto_snap_start_step}")
        if args.auto_snap_dry_run:
            print(f"  DRY RUN MODE (audit only, no freeze)")

    # Create MemDebugConfig if enabled
    mem_debug_config = MemDebugConfig.from_args(args)
    if mem_debug_config.enabled:
        print(f"\n[MemDebug] Enabled:")
        print(f"  Level: {mem_debug_config.level}")
        print(f"  Phases: {', '.join(mem_debug_config.phases)}")
        if mem_debug_config.json_path:
            print(f"  JSON output: {mem_debug_config.json_path}")

    # TPU-specific parameters
    if is_tpu:
        # Disable full vocab CE on TPU - causes massive XLA graph, slow compilation
        tpu_hard_full = 0.0
        print(f"\n[TPU] Training with optimized parameters:")
        print(f"  hard_full_weight: 0.0 (disabled for XLA)")
        print(f"  Note: First step compiles XLA graph (~4 min)")
    else:
        tpu_hard_full = args.hard_full

    # Sparse logits mode (--no-full-logits): prevents [B,L,V] materialization
    # Required for L>=1024 on TPU v6e-1 (16GB HBM)
    effective_hard_top1 = args.hard_top1
    effective_hard_full = tpu_hard_full
    if args.no_full_logits:
        effective_hard_top1 = 0.0
        effective_hard_full = 0.0
        print(f"\n[Sparse Logits] --no-full-logits enabled:")
        print(f"  hard_top1_weight: 0.0 (forced off)")
        print(f"  hard_full_weight: 0.0 (forced off)")
        if args.sampled_ce_weight > 0:
            print(f"  sampled_ce_weight: {args.sampled_ce_weight} (K+R sparse CE)")
            print(f"  sampled_negatives: {args.sampled_negatives} (fallback if cache lacks rand_idx)")
        if args.anchor_ckpt:
            print(f"  anchor-KL: sparse top-K mode (no full logits)")

    # Eval samples: CLI > TPU default (0) > GPU/CPU default (40)
    eval_samples = args.eval_samples
    if eval_samples is None:
        if is_tpu:
            eval_samples = 0  # Skip eval on TPU for speed
            print(f"  eval_samples: 0 (skip eval for speed)")
        else:
            eval_samples = 40  # Default for GPU/CPU

    result = train_e2e(
        model=v2_model,
        cache_dir=args.cache_dir,
        device=device,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        use_cosine_schedule=not args.constant_lr,
        warmup_steps=args.warmup_steps,
        temperature=args.temperature,
        train_weights=False,
        train_scales=True,
        train_g_only=args.g_only,
        train_mlp_only=args.mlp_only,
        train_attn_only=args.attn_only,
        freeze_mags=args.freeze_mags,
        freeze_mags_mlp=args.freeze_mags_mlp,
        freeze_all=args.freeze_all,
        train_norms_only=args.train_norms_only,
        hard_top1_weight=effective_hard_top1,
        hard_top1_end=args.hard_top1_end if not args.no_full_logits else 0.0,
        hard_full_weight=effective_hard_full,
        # Sparse logits mode
        sampled_ce_weight=args.sampled_ce_weight,
        sampled_negatives=args.sampled_negatives,
        no_full_logits=args.no_full_logits,
        logging_steps=20,
        eval_steps=args.eval_steps,
        eval_samples=eval_samples,
        save_dir=args.output_dir,
        save_steps=args.save_steps,
        verbose=True,
        use_fp16=False,  # STE handles FP16
        use_mixed_precision=args.mixed_precision,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run,
        wandb_config=wandb_config,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        accumulation_steps=args.accumulation_steps,
        min_lr_ratio=args.min_lr_ratio,
        keep_checkpoints=args.keep_checkpoints,
        clip_grad_norm=args.clip_grad_norm,
        # Anchor KL regularization
        anchor_ckpt=args.anchor_ckpt,
        anchor_kl_weight=args.anchor_kl_weight if args.anchor_ckpt else 0.0,
        anchor_samples=args.anchor_samples,
        anchor_interval=args.anchor_interval,
        # Auto snap+freeze
        auto_snap_state=auto_snap_state,
        # Memory debug
        mem_debug_config=mem_debug_config,
    )

    print(f"\n  Final loss: {result.get('final_loss', 'N/A')}")

    # =========================================================================
    # STEP 4: Save
    # =========================================================================
    print("\n[4/4] Saving model...")

    # Save in training dtype first
    native_path = f"{args.output_dir}/v2_{args.config}_{args.dtype}_{timestamp}.pt"
    torch.save(v2_model.state_dict(), native_path)
    print(f"  {args.dtype.upper()}: {native_path}")

    # Always save FP16 for ANE export (convert if needed)
    if args.dtype != 'fp16':
        # Preserve embed_tokens and lm_head in original precision
        # (FP16 rounding corrupts embeddings - 0.02 max error is huge for vocab)
        embed_weight = v2_model.model.embed_tokens.weight.data.clone()
        lm_head_weight = v2_model.lm_head.weight.data.clone() if hasattr(v2_model, 'lm_head') else None

        v2_model.half()

        # Restore non-quantized modules to full precision
        v2_model.model.embed_tokens.weight.data = embed_weight
        if lm_head_weight is not None:
            v2_model.lm_head.weight.data = lm_head_weight

    fp16_path = f"{args.output_dir}/v2_{args.config}_fp16_{timestamp}.pt"
    torch.save(v2_model.state_dict(), fp16_path)
    print(f"  FP16: {fp16_path}")

    # Save config.json if it doesn't exist
    config_path = Path(args.output_dir) / 'config.json'
    if not config_path.exists():
        import json
        # Compute lut_bits from LUT_SIZE (LUT16 = 4-bit, LUT4 = 2-bit)
        mlp_lut_bits = {4: 2, 16: 4}.get(MLP_LUT_SIZE, 4)
        attn_lut_bits = {4: 2, 16: 4}.get(ATTN_LUT_SIZE, 4)
        config_data = {
            'version': 'v2',
            'model_id': args.model_id,
            'config_preset': args.config,
            # Quantization config
            'lut_bits': mlp_lut_bits,
            'mlp_lut_bits': mlp_lut_bits,
            'attn_lut_bits': attn_lut_bits,
            'scale_rank': MLP_RANK,
            'mlp_scale_rank': MLP_RANK,
            'attn_scale_rank': ATTN_RANK,
            'group_size': GROUP_SIZE,
            # LoRA config (0 = no LoRA for this run)
            'lora_r': 0,
            'lora_alpha': 0,
            'lora_mlp_only': False,
            # Training info
            'max_steps': args.max_steps,
            'final_loss': result.get('final_loss'),
        }
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        print(f"  Config: {config_path}")

    # =========================================================================
    # STEP 5: Upload to Google Drive (optional)
    # =========================================================================
    if args.gdrive_dir:
        import shutil
        print(f"\n[5/5] Uploading to Google Drive...")
        gdrive_path = Path(args.gdrive_dir)

        # Create directory if missing
        if not gdrive_path.exists():
            try:
                gdrive_path.mkdir(parents=True, exist_ok=True)
                print(f"  Created: {gdrive_path}")
            except Exception as e:
                print(f"  ERROR creating directory: {e}")
                print("  Skipping upload.")
                gdrive_path = None

        if gdrive_path and gdrive_path.exists():
            # Upload native dtype checkpoint
            dest_native = gdrive_path / Path(native_path).name
            try:
                shutil.copy2(native_path, dest_native)
                print(f"  Uploaded: {dest_native}")
            except Exception as e:
                print(f"  ERROR uploading: {e}")

    # =========================================================================
    # STEP 6: Upload to Google Drive using gdrive_sync (new API)
    # =========================================================================
    if args.upload:
        print("\n[6/6] Uploading run to Google Drive...")
        try:
            # Import sync_up from gdrive_sync
            sys.path.insert(0, str(REPO_DIR / 'scripts'))
            from gdrive_sync import sync_up

            # Default exclude pattern: *checkpoint* (intermediate checkpoints are large)
            # --upload-all overrides to upload everything
            if args.upload_all:
                exclude_patterns = []
            else:
                exclude_patterns = args.upload_exclude if args.upload_exclude else ['*checkpoint*']

            # Upload the output directory (run folder)
            success = sync_up(
                local_path=args.output_dir,
                run_name=None,  # Use output_dir name
                dry_run=False,
                is_cache=False,  # This is a run, not a cache
                exclude=exclude_patterns,
                only=None,
            )
            if success:
                print(f"  Upload complete: {args.output_dir}")
            else:
                print(f"  Upload failed or skipped (check if Google Drive is mounted)")
        except ImportError as e:
            print(f"  ERROR: Could not import gdrive_sync: {e}")
        except Exception as e:
            print(f"  ERROR during upload: {e}")

    print("\nDone!")


if __name__ == '__main__':
    main()
