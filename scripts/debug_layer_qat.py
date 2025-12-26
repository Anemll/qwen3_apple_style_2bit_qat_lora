#!/usr/bin/env python3
"""
Debug script for layer-by-layer QAT.

Usage:
    python scripts/debug_layer_qat.py --model Qwen/Qwen3-0.6B
    python scripts/debug_layer_qat.py --model Qwen/Qwen3-0.6B --layer 0 --test-grad
    python scripts/debug_layer_qat.py --model Qwen/Qwen3-0.6B --layer 0 --test-train --cache caches/my_cache
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(description="Debug layer-by-layer QAT")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B", help="Model ID")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cuda/mps/cpu)")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])

    # Quantization config
    parser.add_argument("--lut-size", type=int, default=16, help="LUT size (4-bit=16, 2-bit=4)")
    parser.add_argument("--group-size", type=int, default=32, help="Group size")
    parser.add_argument("--scale-rank", type=int, default=4, help="Scale rank for A @ B")

    # Debug options
    parser.add_argument("--list-layers", action="store_true", help="List all layers and exit")
    parser.add_argument("--layer", type=int, default=None, help="Test specific layer (default: all)")
    parser.add_argument("--test-grad", action="store_true", help="Test gradient flow")
    parser.add_argument("--test-train", action="store_true", help="Test training step")
    parser.add_argument("--cache", type=str, default=None, help="KD cache directory for --test-train")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    return parser.parse_args()


def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_str)


def get_dtype(dtype_str: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_str]


def main():
    args = parse_args()
    device = get_device(args.device)
    dtype = get_dtype(args.dtype)

    print("=" * 60)
    print("LAYER-BY-LAYER QAT DEBUG")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Device: {device}, dtype: {dtype}")
    print(f"Config: lut_size={args.lut_size}, group_size={args.group_size}, scale_rank={args.scale_rank}")
    print()

    # ==========================================================================
    # STEP 1: Load model
    # ==========================================================================
    print("--- Step 1: Loading model ---")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()

    num_layers = len(model.model.layers)
    print(f"Loaded: {sum(p.numel() for p in model.parameters()):,} parameters, {num_layers} layers")
    print()

    # ==========================================================================
    # STEP 2: List Linear layers
    # ==========================================================================
    print("--- Step 2: Checking Linear layers ---")
    linear_modules = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            linear_modules.append((name, m))

    print(f"Found {len(linear_modules)} Linear modules:")
    for name, m in linear_modules[:10]:
        print(f"  {name}: [{m.out_features}, {m.in_features}]")
    if len(linear_modules) > 10:
        print(f"  ... and {len(linear_modules) - 10} more")
    print()

    if args.list_layers:
        print("\nAll Linear layers:")
        for name, m in linear_modules:
            print(f"  {name}: [{m.out_features}, {m.in_features}]")
        return

    # ==========================================================================
    # STEP 3: Replace with AnemllQATLinear
    # ==========================================================================
    print("--- Step 3: Replacing with AnemllQATLinear ---")
    from qat_lora import AnemllQuantConfig, replace_linear_with_anemll, AnemllQATLinear

    config = AnemllQuantConfig(
        lut_size=args.lut_size,
        group_size=args.group_size,
        scale_rank=args.scale_rank,
        learnable_lut=False,
    )

    count = replace_linear_with_anemll(
        model,
        mlp_config=config,
        attn_config=config,
        quantize_attn=True,
        quantize_lm_head=False,
        verbose=args.verbose,
    )

    # Verify
    qat_modules = [(name, m) for name, m in model.named_modules() if isinstance(m, AnemllQATLinear)]
    print(f"\nReplaced {count} layers, verified {len(qat_modules)} AnemllQATLinear modules")

    if len(qat_modules) == 0:
        print("\nERROR: No AnemllQATLinear modules found!")
        print("Checking why replacement failed...")
        import re
        mlp_pattern = re.compile(r'\.mlp\.(gate_proj|up_proj|down_proj)$')
        attn_pattern = re.compile(r'\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$')
        for name, m in linear_modules[:20]:
            mlp_match = mlp_pattern.search(name)
            attn_match = attn_pattern.search(name)
            print(f"  {name}: mlp={bool(mlp_match)}, attn={bool(attn_match)}")
        return
    print()

    # ==========================================================================
    # STEP 4: Test specific layer or all layers
    # ==========================================================================
    layers_to_test = [args.layer] if args.layer is not None else range(num_layers)

    for layer_idx in layers_to_test:
        print(f"--- Step 4: Testing Layer {layer_idx} ---")

        # Get modules in this layer
        layer = model.model.layers[layer_idx]
        layer_qat_modules = [(name, m) for name, m in layer.named_modules() if isinstance(m, AnemllQATLinear)]
        print(f"  Found {len(layer_qat_modules)} AnemllQATLinear modules in layer {layer_idx}:")
        for name, m in layer_qat_modules:
            print(f"    {name}: in={m.in_features}, out={m.out_features}, groups={m.num_groups}")

        if len(layer_qat_modules) == 0:
            print(f"  WARNING: No AnemllQATLinear in layer {layer_idx}!")
            continue

        # ==========================================================================
        # STEP 5: Test gradient flow
        # ==========================================================================
        if args.test_grad:
            print(f"\n--- Step 5: Testing gradient flow for layer {layer_idx} ---")

            # Freeze all, unfreeze this layer
            for p in model.parameters():
                p.requires_grad = False

            trainable_count = 0
            for name, m in layer_qat_modules:
                m.weight.requires_grad = True
                trainable_count += m.weight.numel()

            print(f"  Trainable parameters: {trainable_count:,}")

            # Test forward/backward
            test_module = layer_qat_modules[0][1]
            print(f"  Testing module: {layer_qat_modules[0][0]}")
            print(f"    weight.requires_grad = {test_module.weight.requires_grad}")

            # Simple forward
            x = torch.randn(1, 10, test_module.in_features, device=device, dtype=dtype)
            print(f"    Input shape: {x.shape}")

            y = test_module(x)
            print(f"    Output shape: {y.shape}")
            print(f"    Output requires_grad: {y.requires_grad}")
            print(f"    Output grad_fn: {y.grad_fn}")

            # Backward
            loss = y.sum()
            print(f"    Loss: {loss.item():.4f}")
            print(f"    Loss requires_grad: {loss.requires_grad}")

            try:
                loss.backward()
                if test_module.weight.grad is not None:
                    grad_norm = test_module.weight.grad.norm().item()
                    print(f"    SUCCESS: weight.grad norm = {grad_norm:.6f}")
                else:
                    print(f"    ERROR: weight.grad is None!")
            except Exception as e:
                print(f"    ERROR during backward: {e}")

            # Clear gradients
            for name, m in layer_qat_modules:
                if m.weight.grad is not None:
                    m.weight.grad = None

        # ==========================================================================
        # STEP 6: Test training step with KD cache
        # ==========================================================================
        if args.test_train and args.cache:
            print(f"\n--- Step 6: Testing training step for layer {layer_idx} ---")

            from qat_lora import train_layer, evaluate_kd_loss

            cache_path = Path(args.cache)
            if not cache_path.exists():
                print(f"  ERROR: Cache not found at {args.cache}")
                continue

            cache_files = list(cache_path.glob("*.pt"))
            print(f"  Cache: {len(cache_files)} files in {args.cache}")

            # Eval before
            print("  Computing KD loss before training...")
            loss_before = evaluate_kd_loss(model, args.cache, device, num_samples=10)
            print(f"  KD Loss before: {loss_before:.4f}")

            # Train one step
            print("  Training layer (1 epoch, small batch)...")
            try:
                loss_after = train_layer(
                    model, layer_idx, args.cache, device,
                    batch_size=2,
                    lr=1e-4,
                    epochs=1,
                    grad_accum=1,
                    temperature=2.0,
                    train_scales=False,
                    verbose=True,
                )
                print(f"  KD Loss after: {loss_after:.4f}")
                print(f"  Improvement: {loss_before - loss_after:.4f}")
            except Exception as e:
                print(f"  ERROR during training: {e}")
                import traceback
                traceback.print_exc()

        print()

    print("=" * 60)
    print("DEBUG COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
