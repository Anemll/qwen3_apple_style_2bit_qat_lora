#!/usr/bin/env python3
"""Debug Q2_A4 V1 checkpoint - check inference."""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer
from qat_lora import (
    AnemllQuantConfig,
    replace_linear_with_anemll,
    freeze_model_for_inference,
    compute_all_indices,
)


def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")

    # Q2_A4 config from checkpoint
    model_id = 'Qwen/Qwen3-0.6B'

    # MLP: 2-bit (lut_size=4), rank=32
    MLP_LUT_SIZE = 4
    MLP_SCALE_RANK = 32

    # Attention: 4-bit (lut_size=16), rank=8
    ATTN_LUT_SIZE = 16
    ATTN_SCALE_RANK = 8

    # Checkpoint paths
    checkpoint_dir = '/Users/anemll/Downloads/q2_pt_good1'
    best_checkpoint = f'{checkpoint_dir}/backup_mlp_e2e_w_0.3824.pt'

    print(f"\n1. Loading checkpoint info...")
    state_dict = torch.load(best_checkpoint, map_location='cpu')

    # Check a sample layer
    sample_keys = [k for k in state_dict.keys() if 'layers.0.mlp.gate_proj' in k]
    print(f"   Sample keys for layers.0.mlp.gate_proj:")
    for k in sorted(sample_keys):
        t = state_dict[k]
        if hasattr(t, 'shape'):
            print(f"     {k.split('gate_proj.')[-1]}: {t.dtype}, {t.shape}")

    # Check attention layer
    attn_keys = [k for k in state_dict.keys() if 'layers.0.self_attn.q_proj' in k]
    print(f"\n   Sample keys for layers.0.self_attn.q_proj:")
    for k in sorted(attn_keys):
        t = state_dict[k]
        if hasattr(t, 'shape'):
            print(f"     {k.split('q_proj.')[-1]}: {t.dtype}, {t.shape}")

    # Load model
    print(f"\n2. Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Create V1 configs
    mlp_config = AnemllQuantConfig(
        lut_size=MLP_LUT_SIZE,
        scale_rank=MLP_SCALE_RANK,
    )
    attn_config = AnemllQuantConfig(
        lut_size=ATTN_LUT_SIZE,
        scale_rank=ATTN_SCALE_RANK,
    )

    # Replace with V1 layers
    print(f"\n3. Replacing with V1 layers...")
    print(f"   MLP: {MLP_LUT_SIZE} LUT (2-bit), rank={MLP_SCALE_RANK}")
    print(f"   Attn: {ATTN_LUT_SIZE} LUT (4-bit), rank={ATTN_SCALE_RANK}")

    replace_linear_with_anemll(
        model,
        mlp_config=mlp_config,
        attn_config=attn_config,
        quantize_attn=True,
        quantize_lm_head=False,
    )

    # Load checkpoint
    print(f"\n4. Loading checkpoint...")
    result = model.load_state_dict(state_dict, strict=False)
    print(f"   Missing keys: {len(result.missing_keys)}")
    print(f"   Unexpected keys: {len(result.unexpected_keys)}")

    model.to(device)

    # Check layer state
    print(f"\n5. Checking V1 layer state...")
    for name, m in model.named_modules():
        if type(m).__name__ == 'AnemllQATLinear' and '.mlp.' in name:
            print(f"   {name} (MLP):")
            print(f"      weight: {m.weight.dtype}, {m.weight.shape}")
            print(f"      lut: {m.lut.dtype}, {m.lut.shape}, range=[{m.lut.min():.4f}, {m.lut.max():.4f}]")
            print(f"      scale_A: {m.scale_A.shape}")
            print(f"      scale_B: {m.scale_B.shape}")
            break

    for name, m in model.named_modules():
        if type(m).__name__ == 'AnemllQATLinear' and '.self_attn.' in name:
            print(f"   {name} (Attn):")
            print(f"      weight: {m.weight.dtype}, {m.weight.shape}")
            print(f"      lut: {m.lut.dtype}, {m.lut.shape}, range=[{m.lut.min():.4f}, {m.lut.max():.4f}]")
            print(f"      scale_A: {m.scale_A.shape}")
            print(f"      scale_B: {m.scale_B.shape}")
            break

    # Compute indices and freeze
    print(f"\n6. Computing indices and freezing...")
    compute_all_indices(model)
    freeze_model_for_inference(model)

    # Test inference
    print(f"\n7. Testing inference...")
    model.eval()

    def run_inference(prompt, max_new_tokens=100):
        messages = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': prompt}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors='pt').to(device)

        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

        return tokenizer.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    prompts = [
        'What is 2+2?',
        'What is the capital of France?',
        'What is Apple Neural Engine',
        'What is History of Alibaba Group',
    ]

    for prompt in prompts:
        print(f"\n   Prompt: {prompt}")
        try:
            response = run_inference(prompt)
            if len(response) > 300:
                response = response[:300] + "..."
            print(f"   Response: {response}")
        except Exception as e:
            print(f"   ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Check raw forward
    print(f"\n8. Testing raw forward pass...")
    text = "Hello"
    inputs = tokenizer(text, return_tensors='pt').to(device)

    with torch.no_grad():
        output = model(**inputs)
        logits = output.logits
        print(f"   Logits shape: {logits.shape}")
        print(f"   Logits range: [{logits.min():.4f}, {logits.max():.4f}]")
        print(f"   Has NaN: {torch.isnan(logits).any()}")
        print(f"   Has Inf: {torch.isinf(logits).any()}")

    print("\n=== Done ===")


if __name__ == '__main__':
    main()
