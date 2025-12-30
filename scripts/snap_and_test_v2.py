#!/usr/bin/env python3
"""
Snap V2 checkpoint for export and test inference.

Usage:
    python scripts/snap_and_test_v2.py --checkpoint /path/to/model_state_dict.pt
"""

import argparse
import json
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer
from qat_lora import (
    AnemllQuantConfigV2,
    replace_linear_with_anemll_v2,
    freeze_Q_all,
    freeze_model_for_inference_v2,
    get_inference_mode_v2,
    snap_model_for_ane_v2,
)


def main():
    parser = argparse.ArgumentParser(description='Snap V2 checkpoint and test inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to V2 checkpoint (model_state_dict.pt)')
    parser.add_argument('--model-id', type=str, default='Qwen/Qwen3-0.6B',
                        help='HuggingFace model ID')
    parser.add_argument('--lut-bits', type=int, default=2,
                        help='LUT bits for MLP (default: 2)')
    parser.add_argument('--attn-lut-bits', type=int, default=4,
                        help='LUT bits for attention (default: 4)')
    parser.add_argument('--scale-rank', type=int, default=32,
                        help='Scale rank for MLP (default: 32)')
    parser.add_argument('--attn-scale-rank', type=int, default=8,
                        help='Scale rank for attention (default: 8)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (default: auto-detect)')
    parser.add_argument('--output', type=str, default=None,
                        help='Save snapped checkpoint to this path')
    parser.add_argument('--debug', action='store_true',
                        help='Print debug information')
    parser.add_argument('--fp16', '--ane', action='store_true',
                        help='Snap for ANE export (FP16 precision, recompute indices)')
    args = parser.parse_args()

    # Set attn defaults to MLP values if not specified
    if args.attn_lut_bits is None:
        args.attn_lut_bits = args.lut_bits
    if args.attn_scale_rank is None:
        args.attn_scale_rank = args.scale_rank

    # Auto-detect device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"=== V2 Snap and Test ===")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Model: {args.model_id}")
    print(f"Device: {device}")
    print()

    # Load base model
    print("Loading base model...")
    dtype = torch.bfloat16 if device.type != 'cpu' else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

    # Replace with V2 layers (matching training config)
    print("Replacing with V2 layers...")
    print(f"  MLP: lut_bits={args.lut_bits}, rank={args.scale_rank}")
    print(f"  Attn: lut_bits={args.attn_lut_bits}, rank={args.attn_scale_rank}")

    mlp_config = AnemllQuantConfigV2(
        lut_size=2**args.lut_bits,
        scale_rank=args.scale_rank,
        force_positive_scales=False,  # Match training config
        magnitude_activation='identity',
    )
    attn_config = AnemllQuantConfigV2(
        lut_size=2**args.attn_lut_bits,
        scale_rank=args.attn_scale_rank,
        force_positive_scales=False,  # Match training config
        magnitude_activation='identity',
    )
    count = replace_linear_with_anemll_v2(
        model,
        mlp_config=mlp_config,
        attn_config=attn_config,
        quantize_attn=True,
        quantize_lm_head=False,
        verbose=False,
    )
    print(f"  Replaced {count} layers")

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')

    # Handle both raw state dict and wrapped dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print("  Found nested 'model_state_dict' key")
    else:
        state_dict = checkpoint
        print("  Using raw state dict")

    # Debug: analyze keys
    if args.debug:
        ckpt_keys = set(state_dict.keys())
        model_keys = set(model.state_dict().keys())

        common = ckpt_keys & model_keys
        only_ckpt = ckpt_keys - model_keys
        only_model = model_keys - ckpt_keys

        print(f"\n=== Key Analysis ===")
        print(f"  Checkpoint keys: {len(ckpt_keys)}")
        print(f"  Model keys: {len(model_keys)}")
        print(f"  Common: {len(common)}")
        print(f"  Only in checkpoint: {len(only_ckpt)}")
        print(f"  Only in model: {len(only_model)}")

        if only_ckpt:
            print(f"\n  Sample unexpected (checkpoint only):")
            for k in sorted(only_ckpt)[:5]:
                print(f"    - {k}")
        if only_model:
            print(f"\n  Sample missing (model only):")
            for k in sorted(only_model)[:5]:
                print(f"    - {k}")
        print()

    # Check if checkpoint has _Q buffers
    has_Q_buffers = any('_Q' in k for k in state_dict.keys())
    has_indices_buffers = any('_indices' in k for k in state_dict.keys())
    print(f"  Checkpoint has _Q buffers: {has_Q_buffers}")
    print(f"  Checkpoint has _indices buffers: {has_indices_buffers}")

    # Load state dict
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"  Missing keys: {len(missing)}")
    print(f"  Unexpected keys: {len(unexpected)}")

    if missing and args.debug:
        print("  First 10 missing:")
        for k in missing[:10]:
            print(f"    - {k}")
    if unexpected and args.debug:
        print("  First 10 unexpected:")
        for k in unexpected[:10]:
            print(f"    - {k}")

    # Manually load _Q buffers if they weren't loaded (None buffers issue)
    q_manual_loaded = 0
    for name, m in model.named_modules():
        if type(m).__name__ == 'AnemllQATLinearV2':
            q_key = f"{name}._Q"
            if q_key in state_dict and m._Q is None:
                m.register_buffer("_Q", state_dict[q_key])
                q_manual_loaded += 1
    if q_manual_loaded > 0:
        print(f"  Manually loaded {q_manual_loaded} _Q buffers")

    model.to(device)

    # Check _Q status after loading
    q_loaded = 0
    q_none = 0
    for name, module in model.named_modules():
        if type(module).__name__ == 'AnemllQATLinearV2':
            if module._Q is not None:
                q_loaded += 1
            else:
                q_none += 1
    print(f"  V2 layers with _Q loaded: {q_loaded}")
    print(f"  V2 layers with _Q=None: {q_none}")

    # Decision: freeze_Q or use loaded _Q
    if q_none > 0 and not has_Q_buffers:
        # Checkpoint doesn't have _Q - need to recompute
        print("\nRecomputing Q (checkpoint doesn't have _Q buffers)...")
        freeze_Q_all(model, verbose=False)
    elif q_none > 0:
        # _Q should have been in checkpoint but didn't load
        print(f"\nWARNING: Checkpoint has _Q buffers but {q_none} layers have _Q=None")
        print("  This might indicate a key mismatch. Computing Q for those layers...")
        for name, module in model.named_modules():
            if type(module).__name__ == 'AnemllQATLinearV2' and module._Q is None:
                module.freeze_Q()
                print(f"    Computed _Q for: {name}")
    else:
        print("\nUsing _Q from checkpoint (not recomputing)")

    # Snap for inference
    if args.fp16:
        # FP16 snap for ANE export - recompute indices in FP16 precision
        print("\nSnapping for ANE (FP16 precision)...")
        snapped = snap_model_for_ane_v2(model, recompute_indices=True, verbose=True)
        print(f"  Snapped {snapped} layers to FP16")
    else:
        # DON'T call snap_for_export() - it modifies scale params and might recompute Q
        # Instead, directly call freeze_for_inference() which caches W_eff using loaded _Q
        print("Freezing for inference (caching W_eff)...")
        frozen = 0
        for name, module in model.named_modules():
            if type(module).__name__ == 'AnemllQATLinearV2':
                # Manually cache W_eff without calling snap_for_export
                if module._Q is None:
                    print(f"  WARNING: {name} still has _Q=None, skipping")
                    continue

                with torch.no_grad():
                    # Compute scales using current params (don't modify them)
                    scales = module._compute_full_scales()
                    W_eff = module._Q * scales
                    module._cached_weight_q = W_eff.to(module.weight.dtype)
                frozen += 1
        print(f"  Froze {frozen} layers for inference")

    # Print inference mode diagnostics
    mode_info = get_inference_mode_v2(model)
    print(f"\n[V2 Inference Mode]")
    if args.fp16:
        print(f"  Precision: FP16 (ANE-ready)")
    else:
        print(f"  Precision: Model default (BF16/FP32)")
    print(f"  Mode: FACTORED (rank-by-rank) - default for V2")
    print(f"  Forward: y = Σₖ gₖ · (aₖ ⊙ (Q @ (bₖ ⊙ x)))")
    print(f"  Layers: {mode_info['total']} total, {mode_info['has_frozen_Q']} with frozen Q")

    # Optionally save checkpoint with cached weights
    if args.output:
        print(f"Saving checkpoint to {args.output}...")
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.output)
        print(f"  Saved!")

        # Save config.json for ANE tests
        config_path = output_path.parent / 'config.json'
        config_data = {
            'version': 'v2',
            'model_id': args.model_id,
            'lut_bits': args.lut_bits,
            'attn_lut_bits': args.attn_lut_bits,
            'scale_rank': args.scale_rank,
            'attn_scale_rank': args.attn_scale_rank,
            'force_positive_scales': False,
            'magnitude_activation': 'identity',
            'checkpoint': output_path.name,
            'fp16': args.fp16,
        }
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        print(f"  Config saved to {config_path}")

    # Test inference
    print("\n=== Testing Inference ===\n")
    model.eval()

    prompts = [
        "What is the capital of France?",
        "Explain quantum mechanics in one sentence.",
        "What is 2+2?",
        "What is the speed of light?",
    ]

    for prompt in prompts:
        print(f"Prompt: {prompt}")

        messages = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': prompt}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors='pt').to(device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
            )

        response = tokenizer.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        print(f"Response: {response}")
        print("-" * 50)

    print("\n=== Done ===")


if __name__ == '__main__':
    main()
