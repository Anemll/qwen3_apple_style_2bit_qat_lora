#!/usr/bin/env python3
"""Debug tighten_q: compare W_baseline vs W_eff before/after tightening."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoModelForCausalLM

# Load original checkpoint (before tighten)
print("Loading original checkpoint...")
ckpt_orig = torch.load(
    "/Users/anemll/Library/CloudStorage/GoogleDrive-realanemll@gmail.com/My Drive/qwen3_runs/SR-011_q4_a4_r32_mlp_autosnap/v2_q4a4_r32_fp32_20260110_133950.pt",
    map_location='cpu',
    weights_only=False
)
if 'model_state_dict' in ckpt_orig:
    ckpt_orig = ckpt_orig['model_state_dict']

# Load tightened checkpoint
print("Loading tightened checkpoint...")
ckpt_tight = torch.load(
    "/tmp/SR-011_q4_a4_r32_mlp_autosnap/tightQ_mlp.pt",
    map_location='cpu',
    weights_only=False
)

# Load baseline weights
print("Loading baseline model...")
baseline = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    torch_dtype=torch.float32,
    trust_remote_code=True,
)

# Pick a layer to analyze
layer_name = "model.layers.0.mlp.gate_proj"

# Get data from original checkpoint
Q_orig = ckpt_orig[f"{layer_name}._Q"]
scale_A = ckpt_orig[f"{layer_name}.scale_A"]
scale_B = ckpt_orig[f"{layer_name}.scale_B"]
rank_mag = ckpt_orig[f"{layer_name}.rank_magnitude"]
weight_ckpt = ckpt_orig[f"{layer_name}.weight"]
lut = ckpt_orig[f"{layer_name}.lut"]

# Get Q from tightened
Q_tight = ckpt_tight[f"{layer_name}._Q"]

# Get baseline weight
W_baseline = None
for name, module in baseline.named_modules():
    if name == layer_name:
        W_baseline = module.weight.data
        break

print(f"\n{'='*60}")
print(f"Layer: {layer_name}")
print(f"{'='*60}")

print(f"\nShapes:")
print(f"  Q_orig: {Q_orig.shape}")
print(f"  Q_tight: {Q_tight.shape}")
print(f"  scale_A: {scale_A.shape}")
print(f"  scale_B: {scale_B.shape}")
print(f"  rank_mag: {rank_mag.shape}")
print(f"  weight_ckpt: {weight_ckpt.shape}")
print(f"  W_baseline: {W_baseline.shape}")
print(f"  lut: {lut.shape}")

print(f"\nLUT values: {lut.tolist()}")

print(f"\nQ statistics:")
print(f"  Q_orig:  min={Q_orig.min():.4f}, max={Q_orig.max():.4f}, unique={Q_orig.unique().numel()}")
print(f"  Q_tight: min={Q_tight.min():.4f}, max={Q_tight.max():.4f}, unique={Q_tight.unique().numel()}")

Q_diff = (Q_orig != Q_tight).sum().item()
print(f"  Q values changed: {Q_diff} / {Q_orig.numel()} ({100*Q_diff/Q_orig.numel():.1f}%)")

# Compute scales
# V2: S = (A * g) @ B where g broadcasts across A's columns
# scale_A: [out, rank], rank_mag: [rank], scale_B: [rank, in]
S = (scale_A * rank_mag[None, :]) @ scale_B  # [out, in]
print(f"\nScales S:")
print(f"  shape: {S.shape}")
print(f"  min={S.min():.6f}, max={S.max():.6f}, mean={S.mean():.6f}")

# Compute effective weights
W_eff_orig = Q_orig.view(S.shape) * S
W_eff_tight = Q_tight.view(S.shape) * S

print(f"\nEffective weights:")
print(f"  W_eff_orig:  min={W_eff_orig.min():.4f}, max={W_eff_orig.max():.4f}")
print(f"  W_eff_tight: min={W_eff_tight.min():.4f}, max={W_eff_tight.max():.4f}")
print(f"  W_baseline:  min={W_baseline.min():.4f}, max={W_baseline.max():.4f}")
print(f"  weight_ckpt: min={weight_ckpt.min():.4f}, max={weight_ckpt.max():.4f}")

# Check if checkpoint weight == baseline weight
weight_diff = (W_baseline - weight_ckpt).abs()
print(f"\nWeight comparison (baseline vs checkpoint):")
print(f"  max diff: {weight_diff.max():.6f}")
print(f"  mean diff: {weight_diff.mean():.6f}")
if weight_diff.max() < 1e-6:
    print(f"  => Checkpoint weight IS baseline weight (identical)")
else:
    print(f"  => Checkpoint weight DIFFERS from baseline!")

# Reconstruction errors
mae_baseline_orig = (W_baseline - W_eff_orig).abs().mean().item()
mae_baseline_tight = (W_baseline - W_eff_tight).abs().mean().item()
mae_ckpt_orig = (weight_ckpt - W_eff_orig).abs().mean().item()
mae_ckpt_tight = (weight_ckpt - W_eff_tight).abs().mean().item()

print(f"\nReconstruction MAE:")
print(f"  vs W_baseline:")
print(f"    Q_orig:  {mae_baseline_orig:.6f}")
print(f"    Q_tight: {mae_baseline_tight:.6f}  (delta: {mae_baseline_tight - mae_baseline_orig:+.6f})")
print(f"  vs weight_ckpt:")
print(f"    Q_orig:  {mae_ckpt_orig:.6f}")
print(f"    Q_tight: {mae_ckpt_tight:.6f}  (delta: {mae_ckpt_tight - mae_ckpt_orig:+.6f})")

print(f"\n{'='*60}")
print("DIAGNOSIS")
print(f"{'='*60}")
if weight_diff.max() < 1e-6:
    print("✓ Baseline and checkpoint weights are identical")
    if mae_baseline_tight < mae_baseline_orig:
        print("✓ Tightening improved reconstruction vs baseline")
        print("  BUT perplexity got worse!")
        print("  => The original Q was NOT optimized for baseline reconstruction!")
        print("  => Training evolved W_eff = Q*S away from baseline to minimize loss")
        print("  => Tightening undid this evolution!")
else:
    print("✗ Baseline and checkpoint weights DIFFER!")
    print("  => Using W_baseline as target is WRONG")
    print("  => Should use weight_ckpt as target")
