#!/usr/bin/env python3
"""
Run inference on an Anemll QAT checkpoint.

Supports both snapped and non-snapped checkpoints.

Example:
    # Run inference on snapped checkpoint
    python scripts/run_anemll_inference.py \
        --checkpoint runs/anemll_weights_v1_snapped \
        --prompt "What is the capital of France?"

    # Interactive mode
    python scripts/run_anemll_inference.py \
        --checkpoint runs/anemll_weights_v1 \
        --interactive
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from qat_lora import (
    AnemllQuantConfig,
    replace_linear_with_anemll,
    load_checkpoint,
    freeze_model_for_inference,
)


def run_inference(model, tokenizer, prompt, max_new_tokens=256, device="cpu"):
    """Run inference on a single prompt."""
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors='pt').to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Run inference on Anemll QAT model")

    parser.add_argument("--checkpoint", required=True, help="Checkpoint directory or state dict path")
    parser.add_argument("--prompt", default="What is the capital of France?", help="Prompt to run")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max new tokens to generate")
    parser.add_argument("--device", default="auto", help="auto|cpu|cuda|mps")
    parser.add_argument("--dtype", default="bf16", help="fp16|bf16|fp32")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")

    args = parser.parse_args()

    # Device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # Dtype
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    dtype = dtype_map.get(args.dtype, torch.bfloat16)

    print(f"Device: {device}, dtype: {dtype}")

    # Load config from checkpoint
    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.is_dir():
        config_path = checkpoint_path / "config.json"
        state_path = checkpoint_path / "model_state_dict.pt"
    else:
        config_path = checkpoint_path.parent / "config.json"
        state_path = checkpoint_path

    if not config_path.exists():
        print(f"ERROR: config.json not found at {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        config = json.load(f)

    model_id = config.get('model_id', 'Qwen/Qwen3-0.6B')
    is_snapped = config.get('snapped', False)

    print(f"\nLoading model: {model_id}")
    print(f"Snapped: {is_snapped}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.to(device)

    # Replace with AnemllQATLinear
    print("\nReplacing linear layers...")
    mlp_config = AnemllQuantConfig(
        lut_size=config.get('lut_size', 16),
        group_size=config.get('group_size', 32),
        scale_rank=config.get('scale_rank', 4),
    )

    attn_config = AnemllQuantConfig(
        lut_size=config.get('attn_lut_size', config.get('lut_size', 16)),
        group_size=config.get('attn_group_size', config.get('group_size', 32)),
        scale_rank=config.get('attn_scale_rank', config.get('scale_rank', 4)),
    )

    count = replace_linear_with_anemll(
        model,
        mlp_config=mlp_config,
        attn_config=attn_config,
        quantize_attn=config.get('quantize_attn', True),
        quantize_lm_head=False,
        verbose=False,
    )
    print(f"Replaced {count} layers")
    model.to(device)

    # Load checkpoint
    print(f"\nLoading checkpoint: {state_path}")
    load_checkpoint(model, str(state_path), device=device, verbose=True)

    # Setup for inference
    if is_snapped:
        # Snapped checkpoint: disable fake_quant, use weights directly
        print("\nUsing snapped weights (fake_quant disabled)")
        for name, module in model.named_modules():
            if type(module).__name__ == 'AnemllQATLinear':
                module.enable_fake_quant = False
    else:
        # Non-snapped: use freeze_model_for_inference to cache quantized weights
        print("\nFreezing model for inference (caching quantized weights)")
        freeze_model_for_inference(model, verbose=False)

    model.eval()
    print("\nModel ready for inference!")

    # Run inference
    if args.interactive:
        print("\n=== Interactive Mode ===")
        print("Type 'quit' to exit\n")
        while True:
            try:
                prompt = input("You: ").strip()
                if prompt.lower() in ['quit', 'exit', 'q']:
                    break
                if not prompt:
                    continue
                response = run_inference(model, tokenizer, prompt, args.max_tokens, device)
                print(f"\nAssistant: {response}\n")
            except KeyboardInterrupt:
                print("\nExiting...")
                break
    else:
        print(f"\nPrompt: {args.prompt}")
        response = run_inference(model, tokenizer, args.prompt, args.max_tokens, device)
        print(f"\nResponse: {response}")


if __name__ == "__main__":
    main()
