#!/usr/bin/env python3
"""Debug V2 checkpoint - check _Q, _indices, and inference."""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Force reimport
import importlib
import qat_lora
importlib.reload(qat_lora)

from transformers import AutoModelForCausalLM, AutoTokenizer
from qat_lora import (
    AnemllQuantConfigV2,
    replace_linear_with_anemll_v2,
    freeze_Q_all,
    freeze_model_for_inference_v2,
    load_v2_checkpoint,
)


def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")

    checkpoint_path = '/Users/anemll/Downloads/anemll_v2_q4_a4_ste_fp16_from_v1/model_state_dict.pt'
    model_id = 'Qwen/Qwen3-0.6B'

    # Load base model
    print("\n1. Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Replace with V2 layers
    print("\n2. Replacing with V2 layers...")
    config = AnemllQuantConfigV2(
        lut_size=16,
        scale_rank=4,
        force_positive_scales=False,  # V1 compatibility
        magnitude_activation='identity',
        use_ste_fp16=False,  # Inference mode
    )
    count = replace_linear_with_anemll_v2(
        model, mlp_config=config, attn_config=config,
        quantize_attn=True, verbose=False
    )
    print(f"   Replaced {count} layers")

    # Use new load_v2_checkpoint function
    print("\n3. Loading checkpoint with load_v2_checkpoint...")
    stats = load_v2_checkpoint(model, checkpoint_path, device=device, verbose=True)

    # Check V2 layer state AFTER loading
    print("\n4. V2 layer state AFTER load_v2_checkpoint...")
    for name, m in model.named_modules():
        if type(m).__name__ == 'AnemllQATLinearV2':
            print(f"   {name}:")
            print(f"      weight dtype: {m.weight.dtype}")
            print(f"      lut dtype: {m.lut.dtype}, range: [{m.lut.min():.4f}, {m.lut.max():.4f}]")
            print(f"      _Q: {m._Q.dtype if m._Q is not None else 'None'}, shape: {m._Q.shape if m._Q is not None else 'N/A'}")
            if m._Q is not None:
                print(f"      _Q range: [{m._Q.min():.4f}, {m._Q.max():.4f}]")
            print(f"      _indices: {m._indices.dtype if m._indices is not None else 'None'}")
            break

    # Freeze for inference (should be no-op if _Q already loaded)
    print("\n5. Freezing for inference...")
    freeze_model_for_inference_v2(model, verbose=False)

    # Test inference
    print("\n6. Testing inference...")
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
            # Truncate long responses
            if len(response) > 300:
                response = response[:300] + "..."
            print(f"   Response: {response}")
        except Exception as e:
            print(f"   ERROR: {e}")

    # Check raw forward
    print("\n7. Testing raw forward pass...")
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
