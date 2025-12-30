#!/usr/bin/env python3
"""
Terminal-based V2 Training Script for Colab

Usage:
    # Q4_A4 (4-bit, rank=4)
    python scripts/train_v2.py --config q4a4 --v1-checkpoint /path/to/checkpoint.pt

    # Q2_A4 (2-bit MLP, 4-bit Attn)
    python scripts/train_v2.py --config q2a4 --v1-checkpoint /path/to/checkpoint.pt

    # Custom config
    python scripts/train_v2.py \
        --v1-checkpoint /path/to/checkpoint.pt \
        --lut-size 4 --scale-rank 32 \
        --attn-lut-size 16 --attn-scale-rank 8 \
        --batch-size 8 --epochs 3
"""

import argparse
import os
import sys
import gc
from pathlib import Path
from datetime import datetime

# Add repo to path
REPO_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_DIR))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from qat_lora import (
    AnemllQuantConfig,
    AnemllQuantConfigV2,
    replace_linear_with_anemll,
    replace_linear_with_anemll_v2,
    freeze_Q_all,
    train_e2e,
    KDCacheDataset,
    collate_fn,
)


# =============================================================================
# PRESET CONFIGS
# =============================================================================

CONFIGS = {
    'q4a4': {
        'name': 'Q4_A4 (4-bit, rank=4)',
        'mlp_lut_size': 16,
        'mlp_scale_rank': 4,
        'attn_lut_size': 16,
        'attn_scale_rank': 4,
        'batch_size': 16,
    },
    'q2a4': {
        'name': 'Q2_A4 (2-bit MLP, 4-bit Attn)',
        'mlp_lut_size': 4,
        'mlp_scale_rank': 32,
        'attn_lut_size': 16,
        'attn_scale_rank': 8,
        'batch_size': 8,  # Lower due to higher rank
    },
    'q2a2': {
        'name': 'Q2_A2 (2-bit all)',
        'mlp_lut_size': 4,
        'mlp_scale_rank': 32,
        'attn_lut_size': 4,
        'attn_scale_rank': 32,
        'batch_size': 4,  # Very memory intensive
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='V2 STE-FP16 Training for Anemll QAT',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Required
    parser.add_argument('--v1-checkpoint', type=str, required=True,
                        help='Path to V1 checkpoint (.pt file)')

    # Preset config
    parser.add_argument('--config', type=str, choices=list(CONFIGS.keys()),
                        help='Preset configuration (q4a4, q2a4, q2a2)')

    # Custom config (override preset)
    parser.add_argument('--lut-size', type=int, help='MLP LUT size (4=2-bit, 16=4-bit)')
    parser.add_argument('--scale-rank', type=int, help='MLP scale rank')
    parser.add_argument('--attn-lut-size', type=int, help='Attention LUT size')
    parser.add_argument('--attn-scale-rank', type=int, help='Attention scale rank')

    # Training params
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num-samples', type=int, default=512, help='Calibration samples')
    parser.add_argument('--seq-len', type=int, default=512, help='Sequence length')

    # Output
    parser.add_argument('--output-dir', type=str, default='/content/runs/v2_training',
                        help='Output directory for checkpoints')
    parser.add_argument('--save-to-drive', action='store_true',
                        help='Also save to Google Drive')

    # Model
    parser.add_argument('--model-id', type=str, default='Qwen/Qwen3-0.6B',
                        help='Base model ID')

    # Debug
    parser.add_argument('--dry-run', action='store_true',
                        help='Just check config, don\'t train')

    return parser.parse_args()


def get_config(args):
    """Get config from preset or custom args."""
    if args.config:
        cfg = CONFIGS[args.config].copy()
    else:
        cfg = {
            'name': 'Custom',
            'mlp_lut_size': 16,
            'mlp_scale_rank': 4,
            'attn_lut_size': 16,
            'attn_scale_rank': 4,
            'batch_size': 16,
        }

    # Override with custom args
    if args.lut_size:
        cfg['mlp_lut_size'] = args.lut_size
    if args.scale_rank:
        cfg['mlp_scale_rank'] = args.scale_rank
    if args.attn_lut_size:
        cfg['attn_lut_size'] = args.attn_lut_size
    if args.attn_scale_rank:
        cfg['attn_scale_rank'] = args.attn_scale_rank
    if args.batch_size:
        cfg['batch_size'] = args.batch_size

    return cfg


def cleanup_memory():
    """Clean up GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def main():
    args = parse_args()
    cfg = get_config(args)

    print("=" * 60)
    print("Anemll V2 STE-FP16 Training")
    print("=" * 60)
    print(f"\nConfig: {cfg['name']}")
    print(f"  MLP:  lut_size={cfg['mlp_lut_size']}, rank={cfg['mlp_scale_rank']}")
    print(f"  Attn: lut_size={cfg['attn_lut_size']}, rank={cfg['attn_scale_rank']}")
    print(f"  Batch size: {cfg['batch_size']}")
    print(f"\nCheckpoint: {args.v1_checkpoint}")
    print(f"Output: {args.output_dir}")

    if args.dry_run:
        print("\n[DRY RUN] Config looks good. Remove --dry-run to train.")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # =========================================================================
    # STEP 1: Load Teacher Model
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 1: Loading Teacher Model")
    print("=" * 60)

    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Teacher model loaded: {args.model_id}")

    # =========================================================================
    # STEP 2: Load V1 Checkpoint
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 2: Loading V1 Checkpoint")
    print("=" * 60)

    # Create V1 model
    v1_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    # V1 configs
    v1_mlp_config = AnemllQuantConfig(
        lut_size=cfg['mlp_lut_size'],
        scale_rank=cfg['mlp_scale_rank'],
    )
    v1_attn_config = AnemllQuantConfig(
        lut_size=cfg['attn_lut_size'],
        scale_rank=cfg['attn_scale_rank'],
    )

    replace_linear_with_anemll(
        v1_model,
        mlp_config=v1_mlp_config,
        attn_config=v1_attn_config,
        quantize_attn=True,
        quantize_lm_head=False,
    )

    # Load V1 checkpoint
    state_dict = torch.load(args.v1_checkpoint, map_location='cpu')
    result = v1_model.load_state_dict(state_dict, strict=False)
    print(f"  V1 checkpoint loaded")
    print(f"  Missing keys: {len(result.missing_keys)}")
    print(f"  Unexpected keys: {len(result.unexpected_keys)}")

    v1_model.to(device)

    # =========================================================================
    # STEP 3: Create V2 Model
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 3: Creating V2 Model")
    print("=" * 60)

    v2_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float32,  # FP32 master weights for STE
        trust_remote_code=True,
    )

    v2_mlp_config = AnemllQuantConfigV2(
        lut_size=cfg['mlp_lut_size'],
        scale_rank=cfg['mlp_scale_rank'],
        force_positive_scales=False,  # V1 compatibility
        magnitude_activation='identity',
        use_ste_fp16=True,  # Enable STE-FP16
    )
    v2_attn_config = AnemllQuantConfigV2(
        lut_size=cfg['attn_lut_size'],
        scale_rank=cfg['attn_scale_rank'],
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

    print(f"  V2 model created with STE-FP16 enabled")

    # =========================================================================
    # STEP 4: Convert V1 -> V2
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 4: Converting V1 -> V2")
    print("=" * 60)

    from qat_lora.ane_qat_linear_v2 import AnemllQATLinearV2
    from qat_lora.ane_qat_linear import AnemllQATLinear

    converted = 0
    for (name_v1, m_v1), (name_v2, m_v2) in zip(
        v1_model.named_modules(), v2_model.named_modules()
    ):
        if isinstance(m_v1, AnemllQATLinear) and isinstance(m_v2, AnemllQATLinearV2):
            # Copy LUT and weight
            m_v2.lut.data.copy_(m_v1.lut.data.float())
            m_v2.weight.data.copy_(m_v1.weight.data.float())

            # Convert scales: scale_A @ scale_B -> unit-norm + magnitude
            S = m_v1.scale_A @ m_v1.scale_B  # [out, in]
            S = S.float()

            # SVD to get factorized form
            U, s, Vh = torch.linalg.svd(S, full_matrices=False)
            rank = m_v2.scale_rank

            A_new = U[:, :rank]  # [out, rank]
            B_new = Vh[:rank, :]  # [rank, in]
            g_new = s[:rank]  # [rank]

            # Store with proper signs (V1 can have negative scales)
            m_v2.scale_A.data.copy_(A_new)
            m_v2.scale_B.data.copy_(B_new)
            m_v2.rank_magnitude.data.copy_(g_new)

            converted += 1

    print(f"  Converted {converted} layers")

    # Free V1 model
    del v1_model
    cleanup_memory()

    # Move V2 to device
    v2_model.to(device)

    # Verify conversion
    print("\n  Verifying conversion...")

    # Quick loss check
    test_text = "Hello world"
    inputs = tokenizer(test_text, return_tensors='pt').to(device)

    with torch.no_grad():
        teacher_out = teacher_model(**inputs)
        v2_out = v2_model(**inputs)
        loss = torch.nn.functional.mse_loss(
            v2_out.logits.float(), teacher_out.logits.float()
        ).item()

    print(f"  V2 MSE loss after conversion: {loss:.4f}")

    # =========================================================================
    # STEP 5: Prepare Dataset
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 5: Preparing Calibration Dataset")
    print("=" * 60)

    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')

    def tokenize(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=args.seq_len,
            padding='max_length',
            return_tensors='pt',
        )

    # Filter and tokenize
    texts = [t for t in dataset['text'] if len(t) > 100][:args.num_samples]
    print(f"  Using {len(texts)} samples")

    # Create cached activations
    print("  Caching teacher activations...")
    cached_data = []
    batch_texts = []

    for i, text in enumerate(texts):
        batch_texts.append(text)
        if len(batch_texts) == 32 or i == len(texts) - 1:
            inputs = tokenizer(
                batch_texts,
                truncation=True,
                max_length=args.seq_len,
                padding='max_length',
                return_tensors='pt',
            ).to(device)

            with torch.no_grad():
                outputs = teacher_model(**inputs)

            for j in range(len(batch_texts)):
                cached_data.append({
                    'input_ids': inputs['input_ids'][j].cpu(),
                    'attention_mask': inputs['attention_mask'][j].cpu(),
                    'teacher_logits': outputs.logits[j].cpu(),
                })

            batch_texts = []

            if (i + 1) % 100 == 0:
                print(f"    Cached {i + 1}/{len(texts)} samples")

    train_dataset = KDCacheDataset(cached_data)
    print(f"  Dataset ready: {len(train_dataset)} samples")

    # Free teacher model
    del teacher_model
    cleanup_memory()

    # =========================================================================
    # STEP 6: Freeze Q and Train
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 6: Training (STE-FP16)")
    print("=" * 60)

    # Freeze Q (indices)
    print("  Freezing Q matrices...")
    freeze_Q_all(v2_model)

    print(f"\n  Training config:")
    print(f"    Epochs: {args.epochs}")
    print(f"    Batch size: {cfg['batch_size']}")
    print(f"    Learning rate: {args.lr}")
    print(f"    STE-FP16: Enabled")

    # Train
    result = train_e2e(
        model=v2_model,
        train_dataset=train_dataset,
        num_epochs=args.epochs,
        batch_size=cfg['batch_size'],
        lr=args.lr,
        device=device,
        checkpoint_dir=args.output_dir,
        checkpoint_prefix=f'v2_{timestamp}',
        use_fp16=False,  # STE handles FP16
        verbose=True,
    )

    print(f"\n  Training complete!")
    print(f"  Final loss: {result.get('final_loss', 'N/A')}")

    # =========================================================================
    # STEP 7: Save Final Model
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 7: Saving Model")
    print("=" * 60)

    # Save FP32 model (training weights)
    fp32_path = f"{args.output_dir}/v2_model_fp32_{timestamp}.pt"
    torch.save(v2_model.state_dict(), fp32_path)
    print(f"  FP32 model: {fp32_path}")

    # Convert to FP16 and save
    v2_model.half()
    fp16_path = f"{args.output_dir}/v2_model_fp16_{timestamp}.pt"
    torch.save(v2_model.state_dict(), fp16_path)
    print(f"  FP16 model: {fp16_path}")

    # Copy to Drive if requested
    if args.save_to_drive:
        drive_dir = '/content/drive/MyDrive/anemll_qat/checkpoints'
        os.makedirs(drive_dir, exist_ok=True)

        import shutil
        shutil.copy(fp16_path, f"{drive_dir}/v2_model_fp16_{timestamp}.pt")
        print(f"  Copied to Drive: {drive_dir}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()
