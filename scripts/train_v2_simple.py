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
    args = parser.parse_args()

    # Validate inputs - need either v1 or v2 checkpoint
    if args.v2_checkpoint:
        assert os.path.exists(args.v2_checkpoint), f"V2 checkpoint not found: {args.v2_checkpoint}"
    elif args.v1_checkpoint:
        assert os.path.exists(args.v1_checkpoint), f"V1 checkpoint not found: {args.v1_checkpoint}"
    else:
        raise ValueError("Must specify either --v1-checkpoint or --v2-checkpoint")
    assert os.path.exists(args.cache_dir), f"Cache dir not found: {args.cache_dir}"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if args.v2_checkpoint:
        print(f"V2 checkpoint: {args.v2_checkpoint}")
    else:
        print(f"V1 checkpoint: {args.v1_checkpoint}")
    print(f"Cache dir: {args.cache_dir}")

    # Q2_A4 config (hardcoded for simplicity)
    MLP_LUT_SIZE = 4
    MLP_RANK = 32
    ATTN_LUT_SIZE = 16
    ATTN_RANK = 8

    print(f"\nQ2_A4 Config:")
    print(f"  MLP: lut={MLP_LUT_SIZE}, rank={MLP_RANK}")
    print(f"  Attn: lut={ATTN_LUT_SIZE}, rank={ATTN_RANK}")

    # Import after path setup
    from qat_lora import (
        AnemllQuantConfig,
        AnemllQuantConfigV2,
        replace_linear_with_anemll,
        replace_linear_with_anemll_v2,
        train_e2e,
    )
    from qat_lora.ane_qat_linear import AnemllQATLinear
    from qat_lora.ane_qat_linear_v2 import AnemllQATLinearV2

    # =========================================================================
    # Load model (V2 checkpoint OR V1->V2 conversion)
    # =========================================================================
    os.makedirs(args.output_dir, exist_ok=True)

    import time

    if args.v2_checkpoint:
        # Direct V2 load (skip conversion)
        print("\n[1/3] Loading V2 checkpoint directly...")

        t0 = time.time()
        print("  Loading base model...", end=" ", flush=True)
        v2_model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=torch.float32,  # FP32 master weights
            trust_remote_code=True,
        )
        print(f"done ({time.time()-t0:.1f}s)")

        t0 = time.time()
        print("  Replacing with V2 layers...", end=" ", flush=True)
        v2_mlp_config = AnemllQuantConfigV2(
            lut_size=MLP_LUT_SIZE,
            scale_rank=MLP_RANK,
            force_positive_scales=False,
            magnitude_activation='identity',
            use_ste_fp16=True,
        )
        v2_attn_config = AnemllQuantConfigV2(
            lut_size=ATTN_LUT_SIZE,
            scale_rank=ATTN_RANK,
            force_positive_scales=False,
            magnitude_activation='identity',
            use_ste_fp16=True,
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
        print(f"done ({time.time()-t0:.1f}s)")

        t0 = time.time()
        print("  Moving to GPU...", end=" ", flush=True)
        v2_model.to(device)
        print(f"done ({time.time()-t0:.1f}s)")

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
            torch_dtype=torch.float32,  # FP32 master weights
            trust_remote_code=True,
        )

        v2_mlp_config = AnemllQuantConfigV2(
            lut_size=MLP_LUT_SIZE,
            scale_rank=MLP_RANK,
            force_positive_scales=False,
            magnitude_activation='identity',
            use_ste_fp16=True,
        )
        v2_attn_config = AnemllQuantConfigV2(
            lut_size=ATTN_LUT_SIZE,
            scale_rank=ATTN_RANK,
            force_positive_scales=False,
            magnitude_activation='identity',
            use_ste_fp16=True,
        )

        replace_linear_with_anemll_v2(
            v2_model,
            mlp_config=v2_mlp_config,
            attn_config=v2_attn_config,
            quantize_attn=True,
            quantize_lm_head=False,
        )

        # Convert V1 -> V2 (proper conversion preserving Q and scales)
        print("  Converting V1 -> V2...")

        def is_lut_snapped(w, lut, tol=1e-3):
            """Check if weight is already snapped to LUT values."""
            diff = (w.unsqueeze(-1) - lut.view(1, 1, -1)).abs().min(dim=-1).values
            return (diff.max().item() < tol) and (w.abs().max().item() <= 1.2)

        # Name-based layer mapping (safer than zip)
        v1_layers = {n: m for n, m in v1_model.named_modules() if isinstance(m, AnemllQATLinear)}
        v2_layers = {n: m for n, m in v2_model.named_modules() if isinstance(m, AnemllQATLinearV2)}

        converted = 0
        snapped_count = 0
        for name, m1 in v1_layers.items():
            m2 = v2_layers.get(name)
            if m2 is None:
                print(f"    WARNING: No V2 layer for {name}")
                continue

            # --- LUT ---
            m2.lut.data.copy_(m1.lut.data.float())

            # --- Scales: use V1 get_scales() (includes clamp) ---
            S = m1.get_scales().float()[:, :m2.in_features]  # [out, in]
            U, s, Vh = torch.linalg.svd(S, full_matrices=False)
            r = m2.scale_rank
            m2.scale_A.data.copy_(U[:, :r])
            m2.scale_B.data.copy_(Vh[:r, :])
            m2.rank_magnitude.data.copy_(s[:r])

            # --- Q: check if V1 weight is already snapped ---
            w1 = m1.weight.data.float()[:, :m2.in_features]
            lut = m2.lut.data.float()

            if is_lut_snapped(w1, lut):
                # V1 weight IS the Q (already LUT values)
                Q = w1.clone()
                # Derive indices by nearest LUT entry
                indices = (Q.unsqueeze(-1) - lut.view(1, 1, -1)).abs().argmin(dim=-1)
                snapped_count += 1
            else:
                # V1 weight is FP; compute indices using V1 scales
                normalized = (w1 / S).clamp(-1.0, 1.0)
                # Map to nearest LUT index
                indices = (normalized.unsqueeze(-1) - lut.view(1, 1, -1)).abs().argmin(dim=-1)
                Q = lut[indices]

            # Store Q and indices directly (skip freeze_Q)
            m2.register_buffer('_Q', Q)
            m2.register_buffer('_indices', indices)
            m2.weight.requires_grad = False

            converted += 1

        print(f"  Converted {converted} layers ({snapped_count} were snapped)")

        # Free V1
        del v1_model
        gc.collect()
        torch.cuda.empty_cache()

        v2_model.to(device)

        # Save initial V2 state (before training)
        initial_path = f"{args.output_dir}/v2_initial.pt"
        torch.save(v2_model.state_dict(), initial_path)
        print(f"  Saved initial V2 to {initial_path}")

    # =========================================================================
    # STEP 3: Freeze Q and train
    # =========================================================================
    print("\n[2/3] Training with STE-FP16..." if args.v2_checkpoint else "\n[3/4] Training with STE-FP16...")

    # Only call freeze_Q if _Q not already set (for V2 checkpoint loading)
    for name, module in v2_model.named_modules():
        if isinstance(module, AnemllQATLinearV2):
            if not hasattr(module, '_Q') or module._Q is None:
                module.freeze_Q()
    # train_e2e handles requires_grad based on train_scales, train_g_only, train_mlp_only

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

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
        verbose=True,
        use_fp16=False,  # STE handles FP16
    )

    print(f"\n  Final loss: {result.get('final_loss', 'N/A')}")

    # =========================================================================
    # STEP 4: Save
    # =========================================================================
    print("\n[4/4] Saving model...")

    # FP32 checkpoint
    fp32_path = f"{args.output_dir}/v2_q2a4_fp32_{timestamp}.pt"
    torch.save(v2_model.state_dict(), fp32_path)
    print(f"  FP32: {fp32_path}")

    # FP16 checkpoint
    v2_model.half()
    fp16_path = f"{args.output_dir}/v2_q2a4_fp16_{timestamp}.pt"
    torch.save(v2_model.state_dict(), fp16_path)
    print(f"  FP16: {fp16_path}")

    print("\nDone!")


if __name__ == '__main__':
    main()
