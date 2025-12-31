#!/usr/bin/env python3
"""
Convert Q4_A4 V2 checkpoint to Q2_A4 V2 checkpoint.

- Attention: Keep lut=16 (4-bit), expand rank 4→8
- MLP: Reduce lut 16→4 (4-bit→2-bit), expand rank 4→32

Usage:
    python scripts/convert_q4_to_q2.py \
        --q4-checkpoint /path/to/q4_a4_v2.pt \
        --output /path/to/q2_a4_v2_init.pt
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

REPO_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_DIR))

from transformers import AutoModelForCausalLM
from qat_lora import AnemllQuantConfigV2, replace_linear_with_anemll_v2


def kmeans_reduce_lut(lut: torch.Tensor, new_size: int = 4, n_iter: int = 20) -> torch.Tensor:
    """
    Reduce LUT from 16 entries to 4 using simple k-means (pure PyTorch).
    Returns only the new LUT (weight indices are recomputed during freeze_Q).
    """
    lut = lut.cpu().float()
    n = lut.shape[0]

    # Initialize centers using quantiles
    quantiles = torch.linspace(0, 1, new_size + 2)[1:-1]
    sorted_lut, _ = torch.sort(lut)
    indices = (quantiles * (n - 1)).long()
    centers = sorted_lut[indices].clone()

    # K-means iterations
    for _ in range(n_iter):
        dists = (lut.unsqueeze(1) - centers.unsqueeze(0)).abs()
        labels = dists.argmin(dim=1)
        new_centers = torch.zeros(new_size, dtype=lut.dtype)
        for k in range(new_size):
            mask = labels == k
            if mask.sum() > 0:
                new_centers[k] = lut[mask].mean()
            else:
                new_centers[k] = centers[k]
        centers = new_centers

    # Sort centers
    sorted_centers, _ = torch.sort(centers)
    return sorted_centers.to(lut.dtype)


def expand_tensor(tensor: torch.Tensor, old_size: int, new_size: int,
                  dim: int, init_scale: float = 0.01) -> torch.Tensor:
    """Expand tensor along specified dimension."""
    if old_size >= new_size:
        return tensor.narrow(dim, 0, new_size).clone()

    new_shape = list(tensor.shape)
    new_shape[dim] = new_size
    new_tensor = torch.zeros(new_shape, dtype=tensor.dtype, device=tensor.device)

    # Copy existing data
    slices = [slice(None)] * len(tensor.shape)
    slices[dim] = slice(0, old_size)
    new_tensor[tuple(slices)] = tensor

    # Initialize new entries with small random values
    slices[dim] = slice(old_size, new_size)
    new_tensor[tuple(slices)] = init_scale * torch.randn_like(new_tensor[tuple(slices)])

    return new_tensor


def main():
    parser = argparse.ArgumentParser(description='Convert Q4_A4 V2 to Q2_A4 V2')
    parser.add_argument('--q4-checkpoint', type=str, required=True,
                        help='Path to Q4_A4 V2 checkpoint')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for Q2_A4 V2 checkpoint')
    parser.add_argument('--model-id', type=str, default='Qwen/Qwen3-0.6B',
                        help='Base model ID')

    # Q4 source config
    parser.add_argument('--q4-rank', type=int, default=4,
                        help='Q4 scale rank (default: 4)')

    # Q2 target config
    parser.add_argument('--q2-mlp-rank', type=int, default=32,
                        help='Q2 MLP scale rank (default: 32)')
    parser.add_argument('--q2-attn-rank', type=int, default=8,
                        help='Q2 attention scale rank (default: 8)')
    parser.add_argument('--q2-mlp-lut-size', type=int, default=4,
                        help='Q2 MLP LUT size (default: 4 for 2-bit)')
    parser.add_argument('--q2-attn-lut-size', type=int, default=16,
                        help='Q2 attention LUT size (default: 16 for 4-bit)')

    # Evaluation
    parser.add_argument('--eval', action='store_true',
                        help='Evaluate KD loss after conversion')
    parser.add_argument('--cache-dir', type=str, default=None,
                        help='KD cache dir for evaluation')
    parser.add_argument('--device', type=str, default=None,
                        help='Device for evaluation')

    args = parser.parse_args()

    print(f"=== Q4_A4 → Q2_A4 Conversion ===")
    print(f"Source: {args.q4_checkpoint}")
    print(f"Output: {args.output}")

    # Load Q4 config
    q4_config = None
    q4_dir = Path(args.q4_checkpoint).parent
    q4_config_path = q4_dir / 'config.json'
    if q4_config_path.exists():
        with open(q4_config_path) as f:
            q4_config = json.load(f)
        args.q4_rank = q4_config.get('scale_rank', args.q4_rank)
        print(f"Q4 config: rank={args.q4_rank}, loss={q4_config.get('final_loss', 'N/A')}")

    print(f"\nConversion:")
    print(f"  MLP:  lut 16→{args.q2_mlp_lut_size}, rank {args.q4_rank}→{args.q2_mlp_rank}")
    print(f"  Attn: lut 16→{args.q2_attn_lut_size}, rank {args.q4_rank}→{args.q2_attn_rank}")

    # Load Q4 checkpoint
    print("\nLoading Q4 checkpoint...")
    q4_state = torch.load(args.q4_checkpoint, map_location='cpu')
    print(f"  {len(q4_state)} keys")

    # Create Q2 model
    print("\nCreating Q2 model...")
    q2_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )

    q2_mlp_config = AnemllQuantConfigV2(
        lut_size=args.q2_mlp_lut_size,
        scale_rank=args.q2_mlp_rank,
        force_positive_scales=False,
        magnitude_activation='identity',
    )
    q2_attn_config = AnemllQuantConfigV2(
        lut_size=args.q2_attn_lut_size,
        scale_rank=args.q2_attn_rank,
        force_positive_scales=False,
        magnitude_activation='identity',
    )

    replace_linear_with_anemll_v2(
        q2_model,
        mlp_config=q2_mlp_config,
        attn_config=q2_attn_config,
        quantize_attn=True,
        quantize_lm_head=False,
    )

    # Build new state dict
    print("\nConverting...")
    new_state = {}

    attn_proj_names = ('q_proj', 'k_proj', 'v_proj', 'o_proj')
    mlp_proj_names = ('gate_proj', 'up_proj', 'down_proj')

    mlp_converted = 0
    attn_converted = 0

    # Iterate through Q2 model's expected keys
    q2_state = q2_model.state_dict()

    for key, q2_tensor in q2_state.items():
        is_mlp = any(proj in key for proj in mlp_proj_names)
        is_attn = any(proj in key for proj in attn_proj_names)
        target_dtype = q2_tensor.dtype  # Match Q2 model's dtype (FP32)

        if key in q4_state:
            q4_tensor = q4_state[key].to(target_dtype)  # Convert to target dtype

            # Check if shapes match
            if q4_tensor.shape == q2_tensor.shape:
                # Direct copy
                new_state[key] = q4_tensor.clone()
            else:
                # Need conversion
                if '.lut' in key:
                    # LUT reduction for MLP only
                    if is_mlp and q4_tensor.shape[0] > args.q2_mlp_lut_size:
                        new_state[key] = kmeans_reduce_lut(q4_tensor, args.q2_mlp_lut_size).to(target_dtype)
                    else:
                        new_state[key] = q4_tensor.clone()

                elif '.scale_A' in key:
                    # Expand rank on dim=1: [out, old_rank] -> [out, new_rank]
                    new_rank = args.q2_mlp_rank if is_mlp else args.q2_attn_rank
                    new_state[key] = expand_tensor(q4_tensor, args.q4_rank, new_rank, dim=1).to(target_dtype)

                elif '.scale_B' in key:
                    # Expand rank on dim=0: [old_rank, in] -> [new_rank, in]
                    new_rank = args.q2_mlp_rank if is_mlp else args.q2_attn_rank
                    new_state[key] = expand_tensor(q4_tensor, args.q4_rank, new_rank, dim=0).to(target_dtype)

                elif '.rank_magnitude' in key:
                    # Expand: [old_rank] -> [new_rank]
                    new_rank = args.q2_mlp_rank if is_mlp else args.q2_attn_rank
                    new_mag = torch.zeros(new_rank, dtype=target_dtype)
                    new_mag[:args.q4_rank] = q4_tensor
                    new_mag[args.q4_rank:] = 0.01  # Small init for new ranks
                    new_state[key] = new_mag

                    if is_mlp:
                        mlp_converted += 1
                    else:
                        attn_converted += 1
                else:
                    # Unknown mismatch - use Q2 default
                    print(f"  WARNING: Shape mismatch for {key}: {q4_tensor.shape} vs {q2_tensor.shape}")
                    new_state[key] = q2_tensor.clone()
        else:
            # Key not in Q4, use Q2 default
            new_state[key] = q2_tensor.clone()

    print(f"  MLP projections converted: {mlp_converted}")
    print(f"  Attn projections converted: {attn_converted}")

    # Validate
    print("\nValidating...")
    mismatches = 0
    for key in q2_state.keys():
        if key in new_state:
            if new_state[key].shape != q2_state[key].shape:
                print(f"  MISMATCH: {key}: {new_state[key].shape} vs {q2_state[key].shape}")
                mismatches += 1

    if mismatches == 0:
        print("  All shapes match!")
    else:
        print(f"  {mismatches} mismatches found!")
        return

    # Save
    print(f"\nSaving to {args.output}...")
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    torch.save(new_state, args.output)

    # Save config
    config_path = Path(args.output).parent / 'config.json'
    config = {
        'version': 'v2',
        'model_id': args.model_id,
        'converted_from': 'q4_a4',
        'source_checkpoint': str(args.q4_checkpoint),
        'lut_bits': int(np.log2(args.q2_mlp_lut_size)),
        'attn_lut_bits': int(np.log2(args.q2_attn_lut_size)),
        'scale_rank': args.q2_mlp_rank,
        'attn_scale_rank': args.q2_attn_rank,
        'force_positive_scales': False,
        'magnitude_activation': 'identity',
    }
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to {config_path}")

    # Evaluate if requested
    if args.eval:
        if not args.cache_dir:
            print("\n[eval] ERROR: --cache-dir required")
        else:
            from qat_lora import freeze_Q_all
            from qat_lora.layer_qat import evaluate_kd_loss

            print("\n=== Evaluating KD Loss ===")

            # Check cache exists
            cache_path = Path(args.cache_dir)
            if not cache_path.exists():
                print(f"[eval] ERROR: Cache directory not found: {args.cache_dir}")
            else:
                cache_files = list(cache_path.glob('*.pt'))
                if not cache_files:
                    print(f"[eval] ERROR: No .pt files in cache directory: {args.cache_dir}")
                else:
                    print(f"Found {len(cache_files)} cache files")

                    if args.device:
                        device = torch.device(args.device)
                    elif torch.cuda.is_available():
                        device = torch.device('cuda')
                    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        device = torch.device('mps')
                    else:
                        device = torch.device('cpu')
                    print(f"Device: {device}")

                    # Load into model
                    q2_model.load_state_dict(new_state, strict=True)
                    q2_model.to(device)
                    q2_model.eval()

                    # Freeze Q
                    print("Freezing Q...")
                    freeze_Q_all(q2_model, verbose=False)

                    # Evaluate
                    print(f"Evaluating on {args.cache_dir}...")
                    kd_loss = evaluate_kd_loss(
                        q2_model, args.cache_dir, device,
                        num_samples=40, temperature=2.0
                    )

                    if kd_loss == 0.0:
                        print(f"\n[WARNING] KD loss is 0.0 - check if cache format is correct")
                    else:
                        print(f"\n[Converted Q2_A4 KD Loss]: {kd_loss:.4f}")

                    if q4_config and 'final_loss' in q4_config:
                        q4_loss = q4_config['final_loss']
                        print(f"[Original Q4_A4 KD Loss]:  {q4_loss:.4f}")
                        print(f"[Difference]:              {kd_loss - q4_loss:+.4f}")

                    # Quick inference test
                    print("\n=== Quick Inference Test ===")
                    try:
                        from transformers import AutoTokenizer
                        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
                        prompt = "What is 2+2?"
                        messages = [{"role": "user", "content": prompt}]
                        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        inputs = tokenizer(text, return_tensors="pt").to(device)

                        with torch.no_grad():
                            outputs = q2_model.generate(
                                **inputs,
                                max_new_tokens=50,
                                do_sample=False,
                                pad_token_id=tokenizer.eos_token_id,
                            )

                        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                        print(f"Prompt: {prompt}")
                        print(f"Response: {response[:200]}")
                    except Exception as e:
                        print(f"Inference test failed: {e}")

    print("\n=== Done ===")
    print(f"Use with:")
    print(f"  python scripts/train_v2_simple.py \\")
    print(f"      --v2-checkpoint {args.output} \\")
    print(f"      --cache-dir caches/alpaca_chat_think_both_L128_K128_R1024 \\")
    print(f"      --output-dir runs/q2_from_q4")


if __name__ == '__main__':
    main()
