#!/usr/bin/env python3
"""Debug FP16 training - check why KD loss might be 0."""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer
from qat_lora import (
    AnemllQuantConfigV2,
    replace_linear_with_anemll_v2,
    freeze_Q_all,
    convert_model_to_fp16_v2,
)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model in FP16
    print("\n1. Loading model in FP16...")
    model = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen3-0.6B',
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B')

    # Get reference output BEFORE quantization
    print("\n2. Getting reference output (before QAT)...")
    text = "Hello, world!"
    inputs = tokenizer(text, return_tensors='pt').to(device)

    with torch.no_grad():
        ref_output = model(**inputs)
        ref_logits = ref_output.logits.clone()

    print(f"   Reference logits shape: {ref_logits.shape}")
    print(f"   Reference logits dtype: {ref_logits.dtype}")
    print(f"   Reference logits range: [{ref_logits.min():.4f}, {ref_logits.max():.4f}]")
    print(f"   Reference logits mean: {ref_logits.mean():.4f}")

    # Replace with V2 layers
    print("\n3. Replacing with V2 layers...")
    config = AnemllQuantConfigV2(lut_size=16, scale_rank=4)
    count = replace_linear_with_anemll_v2(
        model, mlp_config=config, attn_config=config,
        quantize_attn=True, verbose=False
    )
    print(f"   Replaced {count} layers")

    # Check V2 layer state BEFORE convert_to_fp16
    print("\n4. Checking V2 layer state BEFORE FP16 conversion...")
    for name, m in model.named_modules():
        if type(m).__name__ == 'AnemllQATLinearV2':
            print(f"   {name}:")
            print(f"      weight dtype: {m.weight.dtype}")
            print(f"      lut dtype: {m.lut.dtype}")
            print(f"      _Q: {m._Q}")
            print(f"      _indices: {m._indices}")
            break

    # Convert to FP16
    print("\n5. Converting to FP16...")
    convert_model_to_fp16_v2(model, verbose=True)

    # Check AFTER convert
    print("\n6. Checking V2 layer state AFTER FP16 conversion...")
    for name, m in model.named_modules():
        if type(m).__name__ == 'AnemllQATLinearV2':
            print(f"   {name}:")
            print(f"      weight dtype: {m.weight.dtype}")
            print(f"      lut dtype: {m.lut.dtype}")
            print(f"      lut range: [{m.lut.min():.4f}, {m.lut.max():.4f}]")
            print(f"      _Q: {m._Q}")
            print(f"      _indices: {m._indices}")
            break

    # Test forward BEFORE freeze_Q
    print("\n7. Testing forward BEFORE freeze_Q...")
    with torch.no_grad():
        try:
            out_before = model(**inputs)
            print(f"   Output logits shape: {out_before.logits.shape}")
            print(f"   Output logits range: [{out_before.logits.min():.4f}, {out_before.logits.max():.4f}]")

            # Check if output is NaN/Inf
            if torch.isnan(out_before.logits).any():
                print("   WARNING: NaN in output!")
            if torch.isinf(out_before.logits).any():
                print("   WARNING: Inf in output!")
        except Exception as e:
            print(f"   ERROR: {e}")

    # Freeze Q
    print("\n8. Freezing Q (computing indices)...")
    freeze_Q_all(model, verbose=True)

    # Check AFTER freeze_Q
    print("\n9. Checking V2 layer state AFTER freeze_Q...")
    for name, m in model.named_modules():
        if type(m).__name__ == 'AnemllQATLinearV2':
            print(f"   {name}:")
            print(f"      _Q dtype: {m._Q.dtype if m._Q is not None else None}")
            print(f"      _Q shape: {m._Q.shape if m._Q is not None else None}")
            print(f"      _Q range: [{m._Q.min():.4f}, {m._Q.max():.4f}]" if m._Q is not None else "      _Q: None")
            print(f"      _indices dtype: {m._indices.dtype if m._indices is not None else None}")
            break

    # Test forward AFTER freeze_Q
    print("\n10. Testing forward AFTER freeze_Q...")
    with torch.no_grad():
        out_after = model(**inputs)
        print(f"    Output logits shape: {out_after.logits.shape}")
        print(f"    Output logits range: [{out_after.logits.min():.4f}, {out_after.logits.max():.4f}]")
        print(f"    Output logits mean: {out_after.logits.mean():.4f}")

    # Compare with reference
    print("\n11. Comparing with reference (KD loss components)...")
    with torch.no_grad():
        # Compute KL divergence manually
        T = 2.0
        student_log_probs = torch.nn.functional.log_softmax(out_after.logits / T, dim=-1)
        teacher_probs = torch.nn.functional.softmax(ref_logits / T, dim=-1)

        kl_div = torch.nn.functional.kl_div(
            student_log_probs, teacher_probs,
            reduction='batchmean'
        ) * (T * T)

        print(f"    KL Divergence: {kl_div.item():.6f}")

        # Check if outputs are identical
        diff = (out_after.logits - ref_logits).abs()
        print(f"    Max difference: {diff.max():.6f}")
        print(f"    Mean difference: {diff.mean():.6f}")

        if diff.max() < 1e-5:
            print("\n    WARNING: Outputs nearly identical!")
            print("    This might mean quantization isn't being applied.")

    print("\n=== Done ===")


if __name__ == '__main__':
    main()
