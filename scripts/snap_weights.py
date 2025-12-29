#!/usr/bin/env python3
"""
Snap QAT weights to their quantized values.

This converts trained QAT weights to their discrete LUT representations.

Usage:
    python scripts/snap_weights.py checkpoint.pt --output snapped.pt
    python scripts/snap_weights.py checkpoint.pt --output snapped.pt --bake-scales
"""

import argparse
import torch
from transformers import AutoModelForCausalLM
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from qat_lora.ane_qat_linear import (
    replace_linear_with_anemll,
    AnemllQuantConfig,
    snap_all_weights,
    export_quantized_model,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Snap QAT weights to quantized values')
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint .pt file')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output path for snapped checkpoint')
    parser.add_argument('--model-id', type=str, default='Qwen/Qwen3-0.6B',
                        help='Base model ID (default: Qwen/Qwen3-0.6B)')

    # Quantization config
    parser.add_argument('--lut-bits', type=int, default=2,
                        help='LUT bits for MLP (default: 2)')
    parser.add_argument('--attn-lut-bits', type=int, default=4,
                        help='LUT bits for attention (default: 4)')
    parser.add_argument('--group-size', type=int, default=16,
                        help='Group size (default: 16)')
    parser.add_argument('--scale-rank', type=int, default=32,
                        help='Scale rank for MLP (default: 32)')
    parser.add_argument('--attn-scale-rank', type=int, default=8,
                        help='Scale rank for attention (default: 8)')

    # Snap mode
    parser.add_argument('--bake-scales', action='store_true',
                        help='Bake scales into weights (LUT[idx]*scale instead of LUT[idx])')
    parser.add_argument('--export', action='store_true',
                        help='Also export full quantized representation')
    parser.add_argument('--export-path', type=str, default=None,
                        help='Path for exported representation (default: <output>_export.pt)')

    return parser.parse_args()


def main():
    args = parse_args()

    # Check checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    print(f"=== Snap Weights ===")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {args.output}")
    print(f"Config: q{args.lut_bits}_a{args.attn_lut_bits}, group={args.group_size}")
    print(f"Scale ranks: MLP={args.scale_rank}, Attn={args.attn_scale_rank}")
    print(f"Mode: {'LUT[idx]*scale (baked)' if args.bake_scales else 'LUT[idx] (normalized)'}")
    print()

    # Load base model
    print(f"Loading base model: {args.model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Replace with QAT layers
    print(f"Replacing linears...")
    mlp_config = AnemllQuantConfig(
        lut_size=2**args.lut_bits,
        group_size=args.group_size,
        scale_rank=args.scale_rank,
    )
    attn_config = AnemllQuantConfig(
        lut_size=2**args.attn_lut_bits,
        group_size=args.group_size,
        scale_rank=args.attn_scale_rank,
    )

    replace_linear_with_anemll(
        model,
        mlp_config=mlp_config,
        attn_config=attn_config,
        quantize_attn=True,
        verbose=False,
    )

    # Load checkpoint
    print(f"Loading checkpoint...")
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys:
        print(f"  Missing keys: {len(result.missing_keys)}")
    if result.unexpected_keys:
        print(f"  Unexpected keys: {len(result.unexpected_keys)}")

    # Snap weights
    print(f"\nSnapping weights...")
    store_lut_values = not args.bake_scales
    indices_dict = snap_all_weights(model, store_lut_values=store_lut_values, verbose=True)

    # Save snapped model
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving snapped model to {args.output}...")
    torch.save(model.state_dict(), args.output)

    # Compute file size
    file_size = output_path.stat().st_size / (1024 * 1024)
    print(f"  Size: {file_size:.1f} MB")

    # Save config.json with snapped_mode
    import json
    snapped_mode = 'lut' if store_lut_values else 'baked'
    config = {
        'model_id': args.model_id,
        'snapped_mode': snapped_mode,
        'lut_bits': args.lut_bits,
        'attn_lut_bits': args.attn_lut_bits,
        'scale_rank': args.scale_rank,
        'attn_scale_rank': args.attn_scale_rank,
    }
    # Save as config.json in the same directory (load_checkpoint looks for this)
    config_path = output_path.parent / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"  Config saved to {config_path}")

    # Save indices separately
    indices_path = output_path.with_suffix('.indices.pt')
    torch.save(indices_dict, indices_path)
    indices_size = indices_path.stat().st_size / (1024 * 1024)
    print(f"  Indices saved to {indices_path} ({indices_size:.1f} MB)")

    # Export if requested
    if args.export:
        export_path = args.export_path or str(output_path.with_suffix('.export.pt'))
        print(f"\nExporting quantized representation...")
        export_dict = export_quantized_model(model, verbose=True)
        torch.save(export_dict, export_path)
        export_size = Path(export_path).stat().st_size / (1024 * 1024)
        print(f"  Exported to {export_path} ({export_size:.1f} MB)")

    print("\nDone!")


if __name__ == '__main__':
    main()
