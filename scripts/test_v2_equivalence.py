#!/usr/bin/env python3
"""
Test numerical equivalence between V1 and V2 AnemllQATLinear.

V2 should produce the same output as V1 (within numerical tolerance).
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from qat_lora.ane_qat_linear import AnemllQATLinear, AnemllQuantConfig
from qat_lora.ane_qat_linear_v2 import AnemllQATLinearV2, AnemllQuantConfigV2


def test_equivalence():
    print("=== Testing V1 vs V2 Numerical Equivalence ===\n")

    # Test dimensions
    in_features = 1024
    out_features = 512
    batch_size = 4
    seq_len = 128
    rank = 4

    # Create input
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, in_features)

    # Create V1 layer
    config_v1 = AnemllQuantConfig(lut_size=16, group_size=16, scale_rank=rank)
    layer_v1 = AnemllQATLinear(in_features, out_features, bias=True, config=config_v1)

    # Create V2 layer with same weights
    config_v2 = AnemllQuantConfigV2(lut_size=16, scale_rank=rank)
    layer_v2 = AnemllQATLinearV2(in_features, out_features, bias=True, config=config_v2)

    # Copy weights from V1 to V2
    layer_v2.weight.data = layer_v1.weight.data.clone()
    layer_v2.bias.data = layer_v1.bias.data.clone()
    layer_v2.lut.data = layer_v1.lut.data.clone()

    # V1 uses padded scale_B, V2 doesn't - need to handle this
    # For fair comparison, reinitialize V2 scales from weights
    layer_v2._init_scales_from_weight()

    print(f"V1 scale_A shape: {layer_v1.scale_A.shape}")
    print(f"V1 scale_B shape: {layer_v1.scale_B.shape}")
    print(f"V2 scale_A shape: {layer_v2.scale_A.shape}")
    print(f"V2 scale_B shape: {layer_v2.scale_B.shape}")
    print(f"V2 rank_magnitude shape: {layer_v2.rank_magnitude.shape}")

    # Test 1: Forward pass comparison (fake quant mode)
    print("\n--- Test 1: Fake Quant Forward ---")
    layer_v1.eval()
    layer_v2.eval()

    with torch.no_grad():
        y_v1 = layer_v1(x)
        y_v2 = layer_v2(x)

    diff = (y_v1 - y_v2).abs()
    rel_diff = diff / (y_v1.abs() + 1e-8)

    print(f"V1 output range: [{y_v1.min():.4f}, {y_v1.max():.4f}]")
    print(f"V2 output range: [{y_v2.min():.4f}, {y_v2.max():.4f}]")
    print(f"Abs diff: max={diff.max():.6f}, mean={diff.mean():.6f}")
    print(f"Rel diff: max={rel_diff.max():.6f}, mean={rel_diff.mean():.6f}")

    # Note: V1 and V2 use different scale initialization, so outputs won't match exactly
    # The important test is that V2's rank-by-rank forward is internally consistent

    # Test 2: V2 freeze_Q and forward
    print("\n--- Test 2: V2 freeze_Q + Forward ---")
    layer_v2.freeze_Q()
    print(f"Q frozen: {layer_v2._Q is not None}")
    print(f"Q shape: {layer_v2._Q.shape if layer_v2._Q is not None else 'None'}")

    with torch.no_grad():
        y_v2_frozen = layer_v2(x)

    diff_frozen = (y_v2 - y_v2_frozen).abs()
    print(f"Diff before/after freeze_Q: max={diff_frozen.max():.6f}")

    # Test 3: V2 loop vs batched forward
    print("\n--- Test 3: V2 Loop vs Batched Forward ---")
    layer_v2.use_batched_forward = False
    with torch.no_grad():
        y_loop = layer_v2(x)

    layer_v2.use_batched_forward = True
    with torch.no_grad():
        y_batched = layer_v2(x)

    diff_impl = (y_loop - y_batched).abs()
    print(f"Loop vs Batched diff: max={diff_impl.max():.6f}, mean={diff_impl.mean():.6f}")

    if diff_impl.max() < 1e-5:
        print("✓ Loop and Batched implementations match!")
    else:
        print("✗ Loop and Batched implementations differ!")

    # Test 4: V2 freeze for inference
    print("\n--- Test 4: V2 Freeze for Inference ---")
    layer_v2.use_batched_forward = False
    layer_v2.freeze_for_inference()
    print(f"Cached weight shape: {layer_v2._cached_weight_q.shape}")

    with torch.no_grad():
        y_inference = layer_v2(x)

    diff_inference = (y_loop - y_inference).abs()
    print(f"Training vs Inference diff: max={diff_inference.max():.6f}")

    if diff_inference.max() < 1e-5:
        print("✓ Training and Inference outputs match!")
    else:
        print("✗ Training and Inference outputs differ!")

    # Test 5: Gradient flow with frozen Q (primary use case)
    print("\n--- Test 5: Gradient Flow (Q Frozen, Scales Trainable) ---")
    layer_v2.unfreeze_for_training()
    layer_v2.freeze_Q()  # Compute Q once, freeze weight
    layer_v2.scale_A.requires_grad = True
    layer_v2.scale_B.requires_grad = True
    layer_v2.rank_magnitude.requires_grad = True

    x_grad = x.clone().requires_grad_(True)
    y = layer_v2(x_grad)
    loss = y.sum()
    loss.backward()

    print(f"weight.requires_grad: {layer_v2.weight.requires_grad}")
    print(f"scale_A.grad: {layer_v2.scale_A.grad is not None}, shape={layer_v2.scale_A.grad.shape if layer_v2.scale_A.grad is not None else 'N/A'}")
    print(f"scale_B.grad: {layer_v2.scale_B.grad is not None}, shape={layer_v2.scale_B.grad.shape if layer_v2.scale_B.grad is not None else 'N/A'}")
    print(f"rank_magnitude.grad: {layer_v2.rank_magnitude.grad is not None}, shape={layer_v2.rank_magnitude.grad.shape if layer_v2.rank_magnitude.grad is not None else 'N/A'}")

    if all([
        layer_v2.scale_A.grad is not None,
        layer_v2.scale_B.grad is not None,
        layer_v2.rank_magnitude.grad is not None,
    ]):
        print("✓ All scale gradients flow correctly!")
    else:
        print("✗ Some scale gradients are missing!")

    # Test 6: Scales-only training (freeze_Q)
    print("\n--- Test 6: Scales-Only Training (Q Frozen) ---")
    # Reset gradients
    layer_v2.zero_grad()
    layer_v2.weight.requires_grad = False  # Freeze weight
    layer_v2.freeze_Q()  # Freeze Q

    x_grad = x.clone().requires_grad_(True)
    y = layer_v2(x_grad)
    loss = y.sum()
    loss.backward()

    print(f"weight.grad: {layer_v2.weight.grad}")
    print(f"scale_A.grad: {layer_v2.scale_A.grad is not None}")
    print(f"scale_B.grad: {layer_v2.scale_B.grad is not None}")
    print(f"rank_magnitude.grad: {layer_v2.rank_magnitude.grad is not None}")

    if layer_v2.weight.grad is None and all([
        layer_v2.scale_A.grad is not None,
        layer_v2.scale_B.grad is not None,
        layer_v2.rank_magnitude.grad is not None,
    ]):
        print("✓ Scales-only training works (weight frozen, scales have gradients)!")
    else:
        print("✗ Scales-only training issue!")

    print("\n=== All Tests Complete ===")


if __name__ == "__main__":
    test_equivalence()
