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
    parser.add_argument('--hard-top1', type=float, default=0.2, help='Hard label top-1 weight')
    parser.add_argument('--hard-full', type=float, default=0.00005, help='Hard label full vocab weight')
    parser.add_argument('--g-only', action='store_true', help='Train only rank_magnitude (G), freeze A and B')
    parser.add_argument('--mlp-only', action='store_true', help='Train only MLP layers, freeze attention')
    parser.add_argument('--save-steps', type=int, default=0, help='Save checkpoint every N steps (0=disabled)')
    # Wandb logging
    parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='qwen3-qat', help='W&B project name')
    parser.add_argument('--wandb-run', type=str, default=None, help='W&B run name (default: auto)')
    # Google Drive upload
    parser.add_argument('--gdrive-dir', type=str, default=None,
                        help='Google Drive directory to upload FP32 checkpoint (creates if missing)')
    # Memory optimization
    parser.add_argument('--gradient-checkpointing', action='store_true',
                        help='Enable gradient checkpointing (trades ~15%% speed for ~40%% memory)')
    # Training precision
    parser.add_argument('--dtype', type=str, default='fp32', choices=['fp32', 'bf16', 'fp16'],
                        help='Training dtype: fp32 (default, ANE-safe), bf16 (2x faster), fp16 (fastest but risky)')
    args = parser.parse_args()

    # Validate inputs - need v1, v2 checkpoint, or from-scratch
    if args.v2_checkpoint:
        assert os.path.exists(args.v2_checkpoint), f"V2 checkpoint not found: {args.v2_checkpoint}"
    elif args.v1_checkpoint:
        assert os.path.exists(args.v1_checkpoint), f"V1 checkpoint not found: {args.v1_checkpoint}"
    elif args.from_scratch:
        pass  # No checkpoint needed
    else:
        raise ValueError("Must specify --v1-checkpoint, --v2-checkpoint, or --from-scratch")
    assert os.path.exists(args.cache_dir), f"Cache dir not found: {args.cache_dir}"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dtype mapping
    dtype_map = {
        'fp32': torch.float32,
        'bf16': torch.bfloat16,
        'fp16': torch.float16,
    }
    train_dtype = dtype_map[args.dtype]

    print(f"Device: {device}")
    print(f"Training dtype: {args.dtype}")
    if args.v2_checkpoint:
        print(f"V2 checkpoint: {args.v2_checkpoint}")
    elif args.from_scratch:
        print("Mode: FROM SCRATCH (no V1)")
    else:
        print(f"V1 checkpoint: {args.v1_checkpoint}")
    print(f"Cache dir: {args.cache_dir}")

    # Q2_A4 config (hardcoded for simplicity)
    MLP_LUT_SIZE = 4
    MLP_RANK = 32
    ATTN_LUT_SIZE = 16
    ATTN_RANK = 8
    GROUP_SIZE = 32  # For scale initialization (smaller = finer granularity)

    print(f"\nQ2_A4 Config:")
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
        )
        print(f"done ({time.time()-t0:.1f}s)")

        t0 = time.time()
        print(f"  Loading checkpoint ({args.v2_checkpoint})...", end=" ", flush=True)
        state_dict = torch.load(args.v2_checkpoint, map_location='cpu')
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

        t0 = time.time()
        print(f"  Moving to GPU ({args.dtype})...", end=" ", flush=True)
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
        )
        print(f"done ({time.time()-t0:.1f}s)")

        t0 = time.time()
        print(f"  Moving to GPU ({args.dtype})...", end=" ", flush=True)
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

        state_dict = torch.load(args.v1_checkpoint, map_location='cpu')
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
        torch.cuda.empty_cache()

        print(f"  Moving to GPU ({args.dtype})...", end=" ", flush=True)
        v2_model.to(device=device, dtype=train_dtype)
        print("done")

        # Save initial V2 state (before training)
        initial_path = f"{args.output_dir}/v2_initial.pt"
        torch.save(v2_model.state_dict(), initial_path)
        print(f"  Saved initial V2 to {initial_path}")

    # =========================================================================
    # GRADIENT CHECKPOINTING (optional memory optimization)
    # =========================================================================
    if args.gradient_checkpointing:
        if hasattr(v2_model, 'gradient_checkpointing_enable'):
            # use_reentrant=False is required when inputs don't have requires_grad
            v2_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            print("\n[*] Gradient checkpointing enabled (use_reentrant=False, trades ~15% speed for ~40% memory)")
        else:
            print("\n[!] Warning: Model does not support gradient_checkpointing_enable()")

    # =========================================================================
    # STEP 3: Freeze Q and train
    # =========================================================================
    print("\n[2/3] Training with STE-FP16..." if args.v2_checkpoint else "\n[3/4] Training with STE-FP16...")

    # Skip freeze_Q if loading V2 checkpoint (Q already in saved state)
    if args.v2_checkpoint:
        # Check if _Q is already loaded from checkpoint
        has_Q = any(hasattr(m, '_Q') and m._Q is not None
                    for m in v2_model.modules() if hasattr(m, '_Q'))
        if has_Q:
            print("  (Using Q from checkpoint, skipping freeze_Q)")
        else:
            print("  (No Q in checkpoint, running freeze_Q)")
            freeze_Q_all(v2_model)
    else:
        freeze_Q_all(v2_model)
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
    }

    result = train_e2e(
        model=v2_model,
        cache_dir=args.cache_dir,
        device=device,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        use_cosine_schedule=True,
        warmup_steps=100,
        temperature=2.0,
        train_weights=False,
        train_scales=True,
        train_g_only=args.g_only,
        train_mlp_only=args.mlp_only,
        hard_top1_weight=args.hard_top1,
        hard_full_weight=args.hard_full,
        logging_steps=20,
        eval_steps=100,
        save_dir=args.output_dir,
        save_steps=args.save_steps,
        verbose=True,
        use_fp16=False,  # STE handles FP16
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run,
        wandb_config=wandb_config,
    )

    print(f"\n  Final loss: {result.get('final_loss', 'N/A')}")

    # =========================================================================
    # STEP 4: Save
    # =========================================================================
    print("\n[4/4] Saving model...")

    # Save in training dtype first
    native_path = f"{args.output_dir}/v2_q2a4_{args.dtype}_{timestamp}.pt"
    torch.save(v2_model.state_dict(), native_path)
    print(f"  {args.dtype.upper()}: {native_path}")

    # Always save FP16 for ANE export (convert if needed)
    if args.dtype != 'fp16':
        v2_model.half()
    fp16_path = f"{args.output_dir}/v2_q2a4_fp16_{timestamp}.pt"
    torch.save(v2_model.state_dict(), fp16_path)
    print(f"  FP16: {fp16_path}")

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

    print("\nDone!")


if __name__ == '__main__':
    main()
