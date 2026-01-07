#!/usr/bin/env python3
"""Debug checkpoint loading - check if V2 layers have correct values."""

import argparse
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--lut-bits", type=int, default=4,
                       help="LUT bits (4=16 entries, 2=4 entries)")
    parser.add_argument("--scale-rank", type=int, default=32)
    parser.add_argument("--attn-lut-bits", type=int, default=None,
                       help="Attention LUT bits (default: same as --lut-bits)")
    parser.add_argument("--attn-scale-rank", type=int, default=None,
                       help="Attention scale rank (default: same as --scale-rank)")
    args = parser.parse_args()

    # Compute derived values
    lut_size = 2 ** args.lut_bits
    attn_lut_bits = args.attn_lut_bits or args.lut_bits
    attn_lut_size = 2 ** attn_lut_bits
    attn_scale_rank = args.attn_scale_rank or args.scale_rank

    print(f"=== Checkpoint Debug ===")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config: lut_bits={args.lut_bits} (size={lut_size}), scale_rank={args.scale_rank}")
    print(f"        attn_lut_bits={attn_lut_bits} (size={attn_lut_size}), attn_scale_rank={attn_scale_rank}")

    # 1. Load raw checkpoint and inspect keys
    print(f"\n[1] Loading raw checkpoint...")
    state_dict = torch.load(args.checkpoint, map_location='cpu')
    print(f"Total keys: {len(state_dict)}")

    # Find V2-specific keys
    v2_keys = [k for k in state_dict if any(x in k for x in ['lut', 'scale_A', 'scale_B', '_Q', '_indices', 'rank_magnitude'])]
    print(f"V2-related keys: {len(v2_keys)}")

    # Sample a layer to inspect
    sample_layer = None
    for k in state_dict:
        if 'model.layers.0.mlp.gate_proj' in k and 'scale_A' in k:
            sample_layer = k.replace('.scale_A', '')
            break

    if sample_layer is None:
        # Try another pattern
        for k in state_dict:
            if 'layers.0' in k and 'scale_A' in k:
                sample_layer = k.replace('.scale_A', '')
                break

    if sample_layer:
        print(f"\n[2] Sample layer: {sample_layer}")

        # Check scale_A
        key = f"{sample_layer}.scale_A"
        if key in state_dict:
            t = state_dict[key]
            print(f"  scale_A: {t.shape}, dtype={t.dtype}")
            print(f"    range: [{t.min():.4f}, {t.max():.4f}], mean={t.mean():.4f}")

        # Check scale_B
        key = f"{sample_layer}.scale_B"
        if key in state_dict:
            t = state_dict[key]
            print(f"  scale_B: {t.shape}, dtype={t.dtype}")
            print(f"    range: [{t.min():.4f}, {t.max():.4f}], mean={t.mean():.4f}")

        # Check rank_magnitude
        key = f"{sample_layer}.rank_magnitude"
        if key in state_dict:
            t = state_dict[key]
            print(f"  rank_magnitude: {t.shape}, dtype={t.dtype}")
            print(f"    values: {t[:8].tolist()}")  # First 8

        # Check _Q
        key = f"{sample_layer}._Q"
        if key in state_dict:
            t = state_dict[key]
            print(f"  _Q: {t.shape}, dtype={t.dtype}")
            print(f"    range: [{t.min():.4f}, {t.max():.4f}]")
            # Check if it looks reasonable (LUT values should be small)
            unique = t.unique()
            print(f"    unique values: {len(unique)} (expected: {lut_size})")
            if len(unique) <= 16:
                print(f"    LUT values: {sorted(unique.tolist())}")
        else:
            print(f"  _Q: NOT FOUND (will be computed from lut+indices)")

        # Check _indices
        key = f"{sample_layer}._indices"
        if key in state_dict:
            t = state_dict[key]
            print(f"  _indices: {t.shape}, dtype={t.dtype}")
            print(f"    range: [{t.min()}, {t.max()}] (expected: 0 to {lut_size-1})")

        # Check lut
        key = f"{sample_layer}.lut"
        if key in state_dict:
            t = state_dict[key]
            print(f"  lut: {t.shape}, dtype={t.dtype}")
            print(f"    values: {t.tolist()}")
    else:
        print("\n[2] Could not find sample layer - listing first 20 keys:")
        for k in list(state_dict.keys())[:20]:
            print(f"  {k}")

    # 3. Load model and test forward pass
    print(f"\n[3] Loading model and testing forward pass...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )

    # Test BEFORE V2 conversion (baseline)
    model.eval()
    test_input = tokenizer("The capital of France is", return_tensors="pt")["input_ids"]
    with torch.no_grad():
        baseline_logits = model(test_input).logits
        baseline_next = baseline_logits[0, -1, :].argmax()
        baseline_prob = torch.softmax(baseline_logits[0, -1, :], dim=-1).max()
    print(f"  Baseline model prediction: '{tokenizer.decode([baseline_next])}' (prob={baseline_prob:.3f})")

    # Replace with V2 layers
    from qat_lora.ane_qat_linear_v2 import (
        AnemllQuantConfigV2,
        replace_linear_with_anemll_v2,
        load_v2_checkpoint,
    )

    mlp_config = AnemllQuantConfigV2(
        lut_size=lut_size,
        scale_rank=args.scale_rank,
    )
    attn_config = AnemllQuantConfigV2(
        lut_size=attn_lut_size,
        scale_rank=attn_scale_rank,
    )

    replace_linear_with_anemll_v2(
        model,
        mlp_config=mlp_config,
        attn_config=attn_config,
        quantize_attn=True,
        verbose=False,
        skip_init=True,
    )

    # Test AFTER V2 conversion but BEFORE checkpoint load
    model.eval()
    with torch.no_grad():
        pre_ckpt_logits = model(test_input).logits
        pre_ckpt_next = pre_ckpt_logits[0, -1, :].argmax()
        pre_ckpt_loss = torch.nn.functional.cross_entropy(
            pre_ckpt_logits[0, :-1, :], test_input[0, 1:]
        )
    print(f"  V2 before checkpoint: next='{tokenizer.decode([pre_ckpt_next])}', CE loss={pre_ckpt_loss:.4f}")

    # Load checkpoint
    stats = load_v2_checkpoint(model, args.checkpoint, verbose=True)

    # Test AFTER checkpoint load
    model.eval()
    with torch.no_grad():
        post_ckpt_logits = model(test_input).logits
        post_ckpt_next = post_ckpt_logits[0, -1, :].argmax()
        post_ckpt_prob = torch.softmax(post_ckpt_logits[0, -1, :], dim=-1).max()
        post_ckpt_loss = torch.nn.functional.cross_entropy(
            post_ckpt_logits[0, :-1, :], test_input[0, 1:]
        )
    print(f"\n[4] After checkpoint load:")
    print(f"  Prediction: '{tokenizer.decode([post_ckpt_next])}' (prob={post_ckpt_prob:.3f})")
    print(f"  CE loss: {post_ckpt_loss:.4f}")

    if post_ckpt_loss > 10:
        print(f"\n  ⚠️  CE loss is very high - checkpoint may be corrupted or mismatched!")
        print(f"  Expected: ~2-5, Got: {post_ckpt_loss:.4f}")

        # Check one V2 layer's actual _Q values
        from qat_lora.ane_qat_linear_v2 import AnemllQATLinearV2
        for name, module in model.named_modules():
            if isinstance(module, AnemllQATLinearV2):
                print(f"\n  Checking layer: {name}")
                if module._Q is not None:
                    print(f"    _Q loaded: {module._Q.shape}")
                    unique = module._Q.unique()
                    print(f"    _Q unique values: {len(unique)}")
                else:
                    print(f"    _Q is None! Will compute on-the-fly")
                print(f"    lut: {module.lut.tolist()}")
                break
    else:
        print(f"\n  ✓ Checkpoint loaded successfully!")


if __name__ == "__main__":
    main()
