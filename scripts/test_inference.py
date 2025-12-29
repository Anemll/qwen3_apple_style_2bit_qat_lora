#!/usr/bin/env python3
"""
Test inference for QAT-trained Qwen3 model.

Usage:
    python scripts/test_inference.py checkpoint.pt
    python scripts/test_inference.py checkpoint.pt --prompt "What is 2+2?"
    python scripts/test_inference.py checkpoint.pt --interactive
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from qat_lora.ane_qat_linear import replace_linear_with_anemll, AnemllQuantConfig
from qat_lora import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='Test QAT model inference')
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint .pt file')
    parser.add_argument('--model-id', type=str, default='Qwen/Qwen3-0.6B',
                        help='Base model ID (default: Qwen/Qwen3-0.6B)')
    parser.add_argument('--prompt', type=str, default=None,
                        help='Single prompt to test')
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='Interactive mode')
    parser.add_argument('--max-tokens', type=int, default=512,
                        help='Max new tokens (default: 512)')
    parser.add_argument('--temperature', type=float, default=0.6,
                        help='Sampling temperature (default: 0.6)')
    parser.add_argument('--repetition-penalty', type=float, default=1.1,
                        help='Repetition penalty (default: 1.1)')
    parser.add_argument('--no-thinking', action='store_true',
                        help='Disable thinking mode')

    # Quantization config
    parser.add_argument('--lut-bits', type=int, default=2,
                        help='LUT bits for MLP (default: 2)')
    parser.add_argument('--attn-lut-bits', type=int, default=4,
                        help='LUT bits for attention (default: 4)')
    parser.add_argument('--group-size', type=int, default=16,
                        help='Group size (default: 16)')
    parser.add_argument('--scale-rank', type=int, default=32,
                        help='Scale rank for MLP (default: 32)')
    parser.add_argument('--attn-scale-rank', type=int, default=8,
                        help='Scale rank for attention (default: 8)')

    return parser.parse_args()


def load_model(args):
    """Load model with QAT layers and checkpoint."""
    # Detect device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        dtype = torch.float32  # MPS works better with float32
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        dtype = torch.bfloat16
    else:
        device = torch.device('cpu')
        dtype = torch.float32

    print(f"Device: {device}, dtype: {dtype}")
    print(f"Loading base model: {args.model_id}")

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    # Replace with QAT layers
    print(f"Replacing linears (q{args.lut_bits}_a{args.attn_lut_bits})...")

    # Create configs
    mlp_config = AnemllQuantConfig(
        lut_size=2**args.lut_bits,
        group_size=args.group_size,
        scale_rank=args.scale_rank,
    )
    attn_config = AnemllQuantConfig(
        lut_size=2**args.attn_lut_bits,
        group_size=args.group_size,
        scale_rank=args.attn_scale_rank,
    )

    replace_linear_with_anemll(
        model,
        mlp_config=mlp_config,
        attn_config=attn_config,
        quantize_attn=True,
        verbose=False,
    )

    # Load checkpoint (also restores snapped_mode from config.json if present)
    print(f"Loading checkpoint: {args.checkpoint}")
    load_checkpoint(model, args.checkpoint, device='cpu', verbose=True)

    # Move to device
    model.to(device)
    model.eval()

    # Freeze for inference (cache quantized weights)
    print("Freezing for inference...")
    for m in model.modules():
        if type(m).__name__ == 'AnemllQATLinear':
            m.freeze_for_inference()

    print("Model ready!\n")
    return model, tokenizer, device


def generate(model, tokenizer, device, prompt, args):
    """Generate response for a prompt."""
    messages = [{'role': 'user', 'content': prompt}]

    # Apply chat template
    template_kwargs = {
        'tokenize': False,
        'add_generation_prompt': True,
    }
    if not args.no_thinking:
        template_kwargs['enable_thinking'] = True

    text = tokenizer.apply_chat_template(messages, **template_kwargs)
    inputs = tokenizer(text, return_tensors='pt').to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=0.9,
            repetition_penalty=args.repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        output[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=False
    )

    # Clean up common artifacts
    response = response.replace('<|im_end|>', '').strip()
    response = response.replace('<think>\n<think>', '<think>')  # Fix double think

    return response


def run_default_prompts(model, tokenizer, device, args):
    """Run default test prompts."""
    prompts = [
        'What is the capital of France?',
        'What is Apple Neural Engine?',
        'Explain quantum mechanics briefly.',
        'What is the speed of light?',
        'Write a haiku about coding.',
    ]

    for prompt in prompts:
        print(f"Prompt: {prompt}")
        response = generate(model, tokenizer, device, prompt, args)
        print(f"Response: {response}")
        print('-' * 60)


def run_interactive(model, tokenizer, device, args):
    """Interactive prompt loop."""
    print("Interactive mode. Type 'q' or 'quit' to exit.\n")

    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not prompt:
            continue
        if prompt.lower() in ('q', 'quit', 'exit'):
            print("Bye!")
            break

        response = generate(model, tokenizer, device, prompt, args)
        print(f"\nAssistant: {response}\n")


def main():
    args = parse_args()

    # Check checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    # Load model
    model, tokenizer, device = load_model(args)

    # Run inference
    if args.prompt:
        # Single prompt
        print(f"Prompt: {args.prompt}")
        response = generate(model, tokenizer, device, args.prompt, args)
        print(f"Response: {response}")
    elif args.interactive:
        # Interactive mode
        run_interactive(model, tokenizer, device, args)
    else:
        # Default prompts
        run_default_prompts(model, tokenizer, device, args)


if __name__ == '__main__':
    main()
