#!/usr/bin/env python3
"""
Quick test: load checkpoint and run inference (before snapping).
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from qat_lora.ane_qat_linear import replace_linear_with_anemll, AnemllQuantConfig
from qat_lora import load_checkpoint

def run_inference(model, tokenizer, prompt, max_new_tokens=256, device='mps'):
    """Run inference and return response."""
    messages = [{'role': 'user', 'content': prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors='pt').to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response


def main():
    # Config - matching the checkpoint
    CHECKPOINT = '/Users/anemll/Downloads/anemll_q4_a4_e2e_v2_scales_only/model_state_dict.pt'
    MODEL_ID = 'Qwen/Qwen3-0.6B'

    # 4-bit config (from checkpoint config)
    lut_bits = 4
    attn_lut_bits = 4
    scale_rank = 4
    attn_scale_rank = 4
    group_size = 16

    # Device
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32 if device == 'mps' else torch.bfloat16
    print(f"Device: {device}, dtype: {dtype}")

    # Load model
    print(f"Loading base model: {MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # Replace with QAT layers
    print(f"Replacing linears (q{lut_bits}_a{attn_lut_bits})...")
    mlp_config = AnemllQuantConfig(lut_size=2**lut_bits, group_size=group_size, scale_rank=scale_rank)
    attn_config = AnemllQuantConfig(lut_size=2**attn_lut_bits, group_size=group_size, scale_rank=attn_scale_rank)

    replace_linear_with_anemll(model, mlp_config=mlp_config, attn_config=attn_config, quantize_attn=True, verbose=False)

    # Load checkpoint
    print(f"Loading checkpoint: {CHECKPOINT}")
    load_checkpoint(model, CHECKPOINT, device='cpu', verbose=True)

    # Move to device
    model.to(device)
    model.eval()

    # Test 1: Before freezing (uses fake_quant_weight)
    print("\n=== Test 1: Before freeze (fake_quant) ===")
    response = run_inference(model, tokenizer, "What is the capital of France?", max_new_tokens=100, device=device)
    print(f"Response: {response}")

    # Test 2: After freezing
    print("\n=== Test 2: After freeze (cached weights) ===")
    for m in model.modules():
        if type(m).__name__ == 'AnemllQATLinear':
            m.freeze_for_inference()

    response = run_inference(model, tokenizer, "What is the capital of France?", max_new_tokens=100, device=device)
    print(f"Response: {response}")

    # Test 3: Snap weights and test
    print("\n=== Test 3: After snap (LUT[idx] mode) ===")
    from qat_lora import snap_all_weights, unfreeze_model_for_training

    # Unfreeze first
    unfreeze_model_for_training(model)

    # Snap to LUT[idx]
    snap_all_weights(model, store_lut_values=True, verbose=False)

    # Check snapped_mode
    for name, m in model.named_modules():
        if type(m).__name__ == 'AnemllQATLinear':
            print(f"  {name}: snapped_mode={getattr(m, 'snapped_mode', None)}")
            break

    # Freeze again
    for m in model.modules():
        if type(m).__name__ == 'AnemllQATLinear':
            m.freeze_for_inference()

    response = run_inference(model, tokenizer, "What is the capital of France?", max_new_tokens=100, device=device)
    print(f"Response: {response}")


if __name__ == '__main__':
    main()
