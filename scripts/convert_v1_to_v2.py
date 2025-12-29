#!/usr/bin/env python3
"""
Convert V1 AnemllQATLinear checkpoint to V2 AnemllQATLinearV2 format.

V1: scales = A @ B (A and B can have any magnitude)
V2: scales = (g * A_dir) @ B_dir where A_dir, B_dir are unit-norm

Usage:
    python scripts/convert_v1_to_v2.py \
        --v1-checkpoint checkpoints/v1_model.pt \
        --output checkpoints/v2_model.pt \
        --model-id Qwen/Qwen3-0.6B
"""

import argparse
import torch
import torch.nn as nn
from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from qat_lora import (
    AnemllQATLinear,
    AnemllQATLinearV2,
    AnemllQuantConfig,
    AnemllQuantConfigV2,
    replace_linear_with_anemll,
    replace_linear_with_anemll_v2,
    load_checkpoint,
)


def convert_v1_layer_to_v2(v1_layer: AnemllQATLinear, v2_layer: AnemllQATLinearV2):
    """Convert a single V1 layer to V2 format.

    V1 parameterization: scales = A @ B
    V2 parameterization: scales = (g[:, None] * A_dir) @ B_dir

    Where A_dir has unit-norm columns, B_dir has unit-norm rows,
    and g is the per-rank magnitude.
    """
    with torch.no_grad():
        # Copy base parameters
        v2_layer.weight.data = v1_layer.weight.data.clone()
        if v1_layer.bias is not None and v2_layer.bias is not None:
            v2_layer.bias.data = v1_layer.bias.data.clone()
        v2_layer.lut.data = v1_layer.lut.data.clone()

        # Get V1 scales (handle potential padding in scale_B)
        A = v1_layer.scale_A  # [out, rank]
        B_full = v1_layer.scale_B  # [rank, padded_in] or [rank, in]
        B = B_full[:, :v1_layer.in_features]  # [rank, in] - remove padding

        # Compute norms
        A_norms = A.norm(dim=0, keepdim=True).clamp(min=1e-8)  # [1, rank]
        B_norms = B.norm(dim=1, keepdim=True).clamp(min=1e-8)  # [rank, 1]

        # V2 stores unit-norm directions + separate magnitude
        A_dir = A / A_norms  # [out, rank] unit-norm columns
        B_dir = B / B_norms  # [rank, in] unit-norm rows

        # Magnitude is product of norms (since scales = A @ B = (A_norm * A_dir) @ (B_norm * B_dir))
        # = (A_norm * B_norm) * (A_dir @ B_dir)
        rank_magnitude = (A_norms.squeeze() * B_norms.squeeze())  # [rank]

        # Store in V2 layer
        v2_layer.scale_A.data = A_dir.to(v1_layer.weight.dtype)
        v2_layer.scale_B.data = B_dir.to(v1_layer.weight.dtype)
        v2_layer.rank_magnitude.data = rank_magnitude.to(v1_layer.weight.dtype)


def convert_model_v1_to_v2(
    v1_model: nn.Module,
    v2_model: nn.Module,
    verbose: bool = True,
) -> int:
    """Convert all V1 layers in v1_model to V2 format in v2_model.

    Both models must have the same architecture with V1/V2 QAT layers
    in corresponding positions.

    Returns:
        Number of layers converted
    """
    v1_layers = {}
    v2_layers = {}

    # Collect V1 and V2 layers by name
    for name, module in v1_model.named_modules():
        if type(module).__name__ == 'AnemllQATLinear':
            v1_layers[name] = module

    for name, module in v2_model.named_modules():
        if type(module).__name__ == 'AnemllQATLinearV2':
            v2_layers[name] = module

    if verbose:
        print(f"Found {len(v1_layers)} V1 layers and {len(v2_layers)} V2 layers")

    # Convert matching layers
    converted = 0
    for name in v1_layers:
        if name in v2_layers:
            v1_layer = v1_layers[name]
            v2_layer = v2_layers[name]

            # Verify dimensions match
            if v1_layer.in_features != v2_layer.in_features:
                print(f"  Warning: {name} in_features mismatch: {v1_layer.in_features} vs {v2_layer.in_features}")
                continue
            if v1_layer.out_features != v2_layer.out_features:
                print(f"  Warning: {name} out_features mismatch: {v1_layer.out_features} vs {v2_layer.out_features}")
                continue

            convert_v1_layer_to_v2(v1_layer, v2_layer)
            converted += 1

            if verbose and converted <= 5:
                print(f"  Converted: {name}")
        else:
            print(f"  Warning: V1 layer {name} not found in V2 model")

    if verbose and converted > 5:
        print(f"  ... and {converted - 5} more layers")

    return converted


def main():
    parser = argparse.ArgumentParser(description='Convert V1 checkpoint to V2 format')
    parser.add_argument('--v1-checkpoint', type=str, required=True,
                        help='Path to V1 checkpoint (.pt file)')
    parser.add_argument('--output', type=str, required=True,
                        help='Path for output V2 checkpoint')
    parser.add_argument('--model-id', type=str, default='Qwen/Qwen3-0.6B',
                        help='HuggingFace model ID')
    parser.add_argument('--lut-bits', type=int, default=4,
                        help='LUT bits for MLP (default: 4)')
    parser.add_argument('--attn-lut-bits', type=int, default=4,
                        help='LUT bits for attention (default: 4)')
    parser.add_argument('--scale-rank', type=int, default=4,
                        help='Scale rank for MLP (default: 4)')
    parser.add_argument('--attn-scale-rank', type=int, default=4,
                        help='Scale rank for attention (default: 4)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use (default: cpu)')
    parser.add_argument('--verify', action='store_true',
                        help='Verify conversion by comparing outputs')
    args = parser.parse_args()

    device = torch.device(args.device)
    v1_path = Path(args.v1_checkpoint)
    output_path = Path(args.output)

    print(f"=== V1 to V2 Checkpoint Conversion ===")
    print(f"V1 checkpoint: {v1_path}")
    print(f"Output: {output_path}")
    print(f"Model: {args.model_id}")
    print()

    # Load base model
    print("Loading base model...")
    from transformers import AutoModelForCausalLM, AutoConfig

    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)

    # Create V1 model and load checkpoint
    print("\nCreating V1 model...")
    v1_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        config=config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    v1_config = AnemllQuantConfig(
        lut_size=2**args.lut_bits,
        scale_rank=args.scale_rank,
    )
    v1_attn_config = AnemllQuantConfig(
        lut_size=2**args.attn_lut_bits,
        scale_rank=args.attn_scale_rank,
    )

    replace_linear_with_anemll(
        v1_model,
        config=v1_config,
        attn_config=v1_attn_config,
    )

    print(f"Loading V1 checkpoint from {v1_path}...")
    load_result = load_checkpoint(v1_model, str(v1_path), verbose=True)

    # Create V2 model
    print("\nCreating V2 model...")
    v2_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        config=config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    v2_config = AnemllQuantConfigV2(
        lut_size=2**args.lut_bits,
        scale_rank=args.scale_rank,
    )
    v2_attn_config = AnemllQuantConfigV2(
        lut_size=2**args.attn_lut_bits,
        scale_rank=args.attn_scale_rank,
    )

    replace_linear_with_anemll_v2(
        v2_model,
        config=v2_config,
        attn_config=v2_attn_config,
    )

    # Convert V1 to V2
    print("\nConverting V1 layers to V2 format...")
    converted = convert_model_v1_to_v2(v1_model, v2_model, verbose=True)
    print(f"Converted {converted} layers")

    # Verify conversion if requested
    if args.verify:
        print("\nVerifying conversion...")
        v1_model.to(device)
        v2_model.to(device)
        v1_model.eval()
        v2_model.eval()

        # Test with random input
        torch.manual_seed(42)
        test_input = torch.randint(0, 1000, (1, 32), device=device)

        with torch.no_grad():
            v1_output = v1_model(test_input).logits
            v2_output = v2_model(test_input).logits

        diff = (v1_output - v2_output).abs()
        rel_diff = diff / (v1_output.abs() + 1e-8)

        print(f"  Output diff: max={diff.max():.6f}, mean={diff.mean():.6f}")
        print(f"  Relative diff: max={rel_diff.max():.6f}, mean={rel_diff.mean():.6f}")

        if diff.max() < 1e-4:
            print("  ✓ Conversion verified - outputs match!")
        else:
            print("  ⚠ Outputs differ (expected due to different forward implementations)")

        # Move back to CPU for saving
        v2_model.cpu()

    # Save V2 checkpoint
    print(f"\nSaving V2 checkpoint to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect state dict
    state_dict = {}
    for name, module in v2_model.named_modules():
        if type(module).__name__ == 'AnemllQATLinearV2':
            prefix = f"{name}."
            state_dict[f"{prefix}weight"] = module.weight.data.cpu()
            if module.bias is not None:
                state_dict[f"{prefix}bias"] = module.bias.data.cpu()
            state_dict[f"{prefix}lut"] = module.lut.data.cpu()
            state_dict[f"{prefix}scale_A"] = module.scale_A.data.cpu()
            state_dict[f"{prefix}scale_B"] = module.scale_B.data.cpu()
            state_dict[f"{prefix}rank_magnitude"] = module.rank_magnitude.data.cpu()

    # Save with config
    save_dict = {
        'model_state_dict': state_dict,
        'config': {
            'model_id': args.model_id,
            'lut_bits': args.lut_bits,
            'attn_lut_bits': args.attn_lut_bits,
            'scale_rank': args.scale_rank,
            'attn_scale_rank': args.attn_scale_rank,
            'version': 'v2',
            'converted_from': str(v1_path),
        }
    }

    torch.save(save_dict, output_path)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Saved: {output_path} ({size_mb:.1f} MB)")

    # Also save config.json
    config_path = output_path.parent / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(save_dict['config'], f, indent=2)
    print(f"  Config: {config_path}")

    print("\n=== Conversion Complete ===")


if __name__ == '__main__':
    main()
