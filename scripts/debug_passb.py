#!/usr/bin/env python3
"""
Debug Pass-B (Y-matrix row smoothing) isolation tests.

Three isolation levels:
  1. FP equivalence: Base A vs Base AB logits (should be near-zero diff)
  2. Pair-level: up→down and v→o pairs individually
  3. FP4-only: Simple LUT quant without V2/SVD

Usage:
  python scripts/debug_passb.py runs/awq_scaled_a0.4 runs/awq_scaled_a0.4_y0.2
  python scripts/debug_passb.py runs/awq_scaled_a0.4 runs/awq_scaled_a0.4_y0.2 --level 2
  python scripts/debug_passb.py runs/awq_scaled_a0.4 runs/awq_scaled_a0.4_y0.2 --level 3 --lut fp4_dense
"""

import argparse
import torch
import torch.nn as nn
from pathlib import Path


def load_model(path, dtype=torch.float32):
    """Load model for testing."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import os

    # Check if path is local directory (resolve relative paths)
    abs_path = os.path.abspath(path)
    is_local = os.path.isdir(abs_path)

    if is_local:
        # Use absolute path for local dirs to avoid HF trying to download
        path = abs_path

    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=dtype,
        trust_remote_code=True,
        local_files_only=is_local,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        path,
        trust_remote_code=True,
        local_files_only=is_local,
    )
    return model, tokenizer


def test_level1_logit_diff(base_a: str, base_ab: str, prompt: str = "Who invented the iPad?"):
    """
    Level 1A: FP equivalence check.
    Base A vs Base AB should produce nearly identical logits.

    Expected: max|Δ| ~ 1e-3 or smaller, mean|Δ| much smaller.
    If NOT tiny → Pass-B is not function-preserving (bug in scaling/compensation).
    """
    print("\n" + "="*60)
    print("LEVEL 1: FP Equivalence Check (Logit Diff)")
    print("="*60)
    print(f"Base A:  {base_a}")
    print(f"Base AB: {base_ab}")
    print(f"Prompt:  '{prompt}'")
    print()

    # Load models
    print("Loading Base A...")
    mA, tok = load_model(base_a, torch.float32)
    print("Loading Base AB...")
    mAB, _ = load_model(base_ab, torch.float32)

    # Tokenize
    x = tok(prompt, return_tensors="pt")
    print(f"Input tokens: {x['input_ids'].shape[1]}")

    # Forward pass
    with torch.no_grad():
        LA = mA(**x, use_cache=False).logits
        LAB = mAB(**x, use_cache=False).logits

    # Compare
    d = (LA - LAB).abs()
    max_diff = d.max().item()
    mean_diff = d.mean().item()
    rel_diff = (d / (LA.abs() + 1e-8)).mean().item()

    print(f"Results:")
    print(f"  max|Δ|:      {max_diff:.6e}")
    print(f"  mean|Δ|:     {mean_diff:.6e}")
    print(f"  rel mean|Δ|: {rel_diff:.6e}")
    print()

    # Diagnosis
    if max_diff < 1e-3:
        print("✓ PASS: Logits match well (Pass-B is function-preserving)")
        print("  If PPL still regresses, issue is in quant/SVD interaction, not equivalence.")
        status = "PASS"
    elif max_diff < 1e-1:
        print("⚠ WARNING: Small but non-negligible diff")
        print("  May indicate numerical precision issues or minor bugs.")
        status = "WARNING"
    else:
        print("✗ FAIL: Large logit diff - Pass-B breaks FP equivalence!")
        print("  Likely bug: GQA expansion, wrong axis, double-application, or base mismatch.")
        status = "FAIL"

    del mA, mAB
    return {'max_diff': max_diff, 'mean_diff': mean_diff, 'status': status}


def test_level2_pair_up_down(base_a: str, base_ab: str, layer: int = 0):
    """
    Level 2A: up_proj → down_proj pair test.
    Test MLP pair equivalence with random input.
    """
    print("\n" + "="*60)
    print(f"LEVEL 2A: up_proj → down_proj Pair Test (Layer {layer})")
    print("="*60)

    print("Loading Base A...")
    mA, _ = load_model(base_a, torch.float32)
    print("Loading Base AB...")
    mAB, _ = load_model(base_ab, torch.float32)

    # Get MLP modules
    upA = mA.model.layers[layer].mlp.up_proj
    downA = mA.model.layers[layer].mlp.down_proj
    upB = mAB.model.layers[layer].mlp.up_proj
    downB = mAB.model.layers[layer].mlp.down_proj

    # Also test gate_proj if it exists (Qwen uses gate)
    gateA = getattr(mA.model.layers[layer].mlp, 'gate_proj', None)
    gateB = getattr(mAB.model.layers[layer].mlp, 'gate_proj', None)

    # Random input
    batch = 2
    x = torch.randn(batch, upA.in_features)

    with torch.no_grad():
        # Simple up → down path
        yA = downA(upA(x))
        yB = downB(upB(x))

        d = (yA - yB).abs()
        max_diff = d.max().item()
        mean_diff = d.mean().item()

        print(f"\nup_proj → down_proj:")
        print(f"  max|Δ|:  {max_diff:.6e}")
        print(f"  mean|Δ|: {mean_diff:.6e}")

        # Full MLP path (with gate if exists)
        if gateA is not None and gateB is not None:
            # Qwen MLP: down(silu(gate(x)) * up(x))
            act = nn.SiLU()
            yA_full = downA(act(gateA(x)) * upA(x))
            yB_full = downB(act(gateB(x)) * upB(x))

            d_full = (yA_full - yB_full).abs()
            max_diff_full = d_full.max().item()
            mean_diff_full = d_full.mean().item()

            print(f"\nFull MLP (gate * up → down):")
            print(f"  max|Δ|:  {max_diff_full:.6e}")
            print(f"  mean|Δ|: {mean_diff_full:.6e}")

    # Diagnosis
    print()
    if max_diff < 1e-4:
        print("✓ PASS: up→down pair is equivalent")
        status = "PASS"
    else:
        print("✗ FAIL: up→down pair differs - Pass-B scale_fc_fc() bug!")
        print("  Check: row vs col direction, scale inversion")
        status = "FAIL"

    del mA, mAB
    return {'max_diff': max_diff, 'mean_diff': mean_diff, 'status': status}


def test_level2_pair_v_o(base_a: str, base_ab: str, layer: int = 0):
    """
    Level 2B: v_proj → o_proj pair test (GQA sensitive).
    This is where most subtle bugs live due to GQA head expansion.
    """
    print("\n" + "="*60)
    print(f"LEVEL 2B: v_proj → o_proj Pair Test (Layer {layer})")
    print("="*60)

    print("Loading Base A...")
    mA, _ = load_model(base_a, torch.float32)
    print("Loading Base AB...")
    mAB, _ = load_model(base_ab, torch.float32)

    # Get attention modules
    attnA = mA.model.layers[layer].self_attn
    attnB = mAB.model.layers[layer].self_attn

    vA = attnA.v_proj
    oA = attnA.o_proj
    vB = attnB.v_proj
    oB = attnB.o_proj

    # GQA info
    num_heads = attnA.num_heads
    num_kv_heads = attnA.num_key_value_heads
    head_dim = attnA.head_dim
    groups = num_heads // num_kv_heads

    print(f"\nGQA config:")
    print(f"  num_heads:    {num_heads}")
    print(f"  num_kv_heads: {num_kv_heads}")
    print(f"  head_dim:     {head_dim}")
    print(f"  groups:       {groups}")

    # Random input
    batch = 2
    x = torch.randn(batch, vA.in_features)

    with torch.no_grad():
        # v_proj output (KV heads * head_dim)
        vA_out = vA(x)  # [batch, kv_heads * head_dim]
        vB_out = vB(x)

        # Expand to full heads (repeat_kv style)
        # [batch, kv_heads, head_dim] -> [batch, num_heads, head_dim]
        vA_kv = vA_out.view(batch, num_kv_heads, head_dim)
        vB_kv = vB_out.view(batch, num_kv_heads, head_dim)

        vA_full = vA_kv.repeat_interleave(groups, dim=1)  # [batch, num_heads, head_dim]
        vB_full = vB_kv.repeat_interleave(groups, dim=1)

        # Flatten for o_proj
        vA_flat = vA_full.view(batch, -1)  # [batch, num_heads * head_dim]
        vB_flat = vB_full.view(batch, -1)

        # o_proj
        oA_out = oA(vA_flat)
        oB_out = oB(vB_flat)

        d = (oA_out - oB_out).abs()
        max_diff = d.max().item()
        mean_diff = d.mean().item()

        print(f"\nv_proj → (GQA expand) → o_proj:")
        print(f"  max|Δ|:  {max_diff:.6e}")
        print(f"  mean|Δ|: {mean_diff:.6e}")

        # Check scale consistency across GQA groups
        # The scale on o_proj inputs must be constant across repeated groups
        print(f"\nGQA scale consistency check:")
        print(f"  v_proj output shape: {vA_out.shape}")
        print(f"  After GQA expand:    {vA_flat.shape}")

        # Check if v_proj weights differ between A and AB
        v_weight_diff = (vA.weight - vB.weight).abs()
        o_weight_diff = (oA.weight - oB.weight).abs()
        print(f"\n  v_proj weight diff: max={v_weight_diff.max():.6e}, mean={v_weight_diff.mean():.6e}")
        print(f"  o_proj weight diff: max={o_weight_diff.max():.6e}, mean={o_weight_diff.mean():.6e}")

    # Diagnosis
    print()
    if max_diff < 1e-4:
        print("✓ PASS: v→o pair is equivalent (GQA expansion correct)")
        status = "PASS"
    else:
        print("✗ FAIL: v→o pair differs - likely GQA expansion bug!")
        print("  Check: scale must be constant across repeated groups")
        status = "FAIL"

    del mA, mAB
    return {'max_diff': max_diff, 'mean_diff': mean_diff, 'status': status}


def test_level3_fp4_quant(base_a: str, base_ab: str, lut_name: str = "fp4_dense", layer: int = 0):
    """
    Level 3: FP4-only quant test (no V2/SVD).
    Compare quant error between Base A and Base AB.

    If Pass-B is "good for quant", Base AB should have lower quant error.
    """
    print("\n" + "="*60)
    print(f"LEVEL 3: FP4-Only Quant Test (Layer {layer})")
    print("="*60)
    print(f"LUT: {lut_name}")

    # Get LUT
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
    from anemll_quant.lut_presets import get_lut_preset

    lut = get_lut_preset(lut_name, 16)  # 16 entries for FP4
    print(f"LUT shape: {lut.shape}")
    print(f"LUT range: [{lut.min():.4f}, {lut.max():.4f}]")

    print("\nLoading Base A...")
    mA, _ = load_model(base_a, torch.float32)
    print("Loading Base AB...")
    mAB, _ = load_model(base_ab, torch.float32)

    # Get weights to quantize
    WA = mA.model.layers[layer].mlp.up_proj.weight.data.float()
    WAB = mAB.model.layers[layer].mlp.up_proj.weight.data.float()

    print(f"\nWeight shapes: {WA.shape}")
    print(f"Base A  range: [{WA.min():.4f}, {WA.max():.4f}]")
    print(f"Base AB range: [{WAB.min():.4f}, {WAB.max():.4f}]")

    def quantize_to_lut(W, lut):
        """Quantize weight matrix to nearest LUT entry."""
        # Per-tensor scale to map to LUT range
        scale = W.abs().max() / lut.abs().max()
        W_scaled = W / scale

        # Quantize to nearest LUT entry
        # [out, in] -> find nearest in lut for each element
        W_flat = W_scaled.flatten()
        dists = (W_flat.unsqueeze(-1) - lut.unsqueeze(0)).abs()
        indices = dists.argmin(dim=-1)
        W_quant = lut[indices].view(W.shape)

        # Scale back
        W_quant = W_quant * scale
        return W_quant, scale

    # Quantize both
    WA_q, sA = quantize_to_lut(WA, lut)
    WAB_q, sAB = quantize_to_lut(WAB, lut)

    # Compute quant error
    errA = (WA - WA_q).abs()
    errAB = (WAB - WAB_q).abs()

    print(f"\nQuantization Error (weight space):")
    print(f"  Base A:  max={errA.max():.6f}, mean={errA.mean():.6f}")
    print(f"  Base AB: max={errAB.max():.6f}, mean={errAB.mean():.6f}")

    # Test output error with random input
    x = torch.randn(2, WA.shape[1])  # [batch, in_features]

    with torch.no_grad():
        # FP output
        yA_fp = x @ WA.T
        yAB_fp = x @ WAB.T

        # Quantized output
        yA_q = x @ WA_q.T
        yAB_q = x @ WAB_q.T

        # Output error
        out_errA = (yA_fp - yA_q).abs()
        out_errAB = (yAB_fp - yAB_q).abs()

        print(f"\nQuantization Error (output space):")
        print(f"  Base A:  max={out_errA.max():.6f}, mean={out_errA.mean():.6f}")
        print(f"  Base AB: max={out_errAB.max():.6f}, mean={out_errAB.mean():.6f}")

    # Compare
    print()
    if errAB.mean() < errA.mean():
        pct_better = (1 - errAB.mean() / errA.mean()) * 100
        print(f"✓ Pass-B HELPS quant: {pct_better:.1f}% lower mean weight error")
        status = "BETTER"
    else:
        pct_worse = (errAB.mean() / errA.mean() - 1) * 100
        print(f"✗ Pass-B HURTS quant: {pct_worse:.1f}% higher mean weight error")
        status = "WORSE"

    del mA, mAB
    return {
        'errA_mean': errA.mean().item(),
        'errAB_mean': errAB.mean().item(),
        'status': status
    }


def main():
    parser = argparse.ArgumentParser(description="Debug Pass-B isolation tests")
    parser.add_argument("base_a", help="Path to Base A (Pass-A only)")
    parser.add_argument("base_ab", help="Path to Base AB (Pass-A + Pass-B)")
    parser.add_argument("--level", type=int, choices=[1, 2, 3], default=0,
                        help="Run specific level only (0=all)")
    parser.add_argument("--layer", type=int, default=0, help="Layer to test for Level 2/3")
    parser.add_argument("--lut", default="fp4_dense", help="LUT preset for Level 3")
    parser.add_argument("--prompt", default="Who invented the iPad?", help="Prompt for Level 1")
    args = parser.parse_args()

    results = {}

    if args.level == 0 or args.level == 1:
        results['level1'] = test_level1_logit_diff(args.base_a, args.base_ab, args.prompt)

    if args.level == 0 or args.level == 2:
        results['level2a'] = test_level2_pair_up_down(args.base_a, args.base_ab, args.layer)
        results['level2b'] = test_level2_pair_v_o(args.base_a, args.base_ab, args.layer)

    if args.level == 0 or args.level == 3:
        results['level3'] = test_level3_fp4_quant(args.base_a, args.base_ab, args.lut, args.layer)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for name, res in results.items():
        status = res.get('status', 'N/A')
        if 'max_diff' in res:
            print(f"{name}: {status} (max|Δ|={res['max_diff']:.2e})")
        elif 'errA_mean' in res:
            print(f"{name}: {status} (A={res['errA_mean']:.4f}, AB={res['errAB_mean']:.4f})")
        else:
            print(f"{name}: {status}")

    print()
    print("Decision tree:")
    if 'level1' in results and results['level1']['status'] == 'FAIL':
        print("  → Pass-B implementation bug. Fix GQA expansion / scaling direction.")
    elif 'level3' in results and results['level3']['status'] == 'WORSE':
        print("  → Pass-B is equivalent but quantization-hostile.")
        print("    Try: restrict to only v→o OR only up→down, tighten clamps.")
    elif 'level1' in results and results['level1']['status'] == 'PASS':
        print("  → Pass-B is function-preserving. Issue may be in V2/SVD interaction.")
        print("    Try: increase rank, or apply Pass-B only to v→o.")


if __name__ == '__main__':
    main()
