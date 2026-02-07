#!/usr/bin/env python3
"""
Test inference for QAT-trained Qwen3 model (supports V1 and V2).

Usage:
    python scripts/test_inference.py checkpoint_dir/
    python scripts/test_inference.py checkpoint.pt
    python scripts/test_inference.py checkpoint.pt --prompt "What is 2+2?"
    python scripts/test_inference.py checkpoint.pt --interactive

    # Use config preset (q4_r32, q2a4, q4a4, q2a2)
    python scripts/test_inference.py checkpoint.pt --config q4_r32
    python scripts/test_inference.py checkpoint.pt --config q2a4

    # With LoRA adapter (recovery LoRA on top of quantized model)
    # Option A: Separate LoRA file
    python scripts/test_inference.py checkpoint.pt --lora lora_adapter.pt
    python scripts/test_inference.py checkpoint.pt --lora lora_adapter.pt --lora-r 8

    # Option B: Full checkpoint with embedded LoRA (auto-detected)
    python scripts/test_inference.py full_checkpoint_with_lora.pt
"""

import argparse
import json
import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Quantization config presets (same as measure_perplexity.py, train_v2_simple.py)
CONFIG_PRESETS = {
    'q2a4': {'mlp_lut': 4, 'mlp_rank': 32, 'attn_lut': 16, 'attn_rank': 8},   # 2-bit MLP, 4-bit Attn
    'q4a4': {'mlp_lut': 16, 'mlp_rank': 4, 'attn_lut': 16, 'attn_rank': 4},    # 4-bit both, rank=4
    'q4a4_r32': {'mlp_lut': 16, 'mlp_rank': 32, 'attn_lut': 16, 'attn_rank': 32},  # 4-bit both, rank=32
    'q4_r32': {'mlp_lut': 16, 'mlp_rank': 32, 'attn_lut': 16, 'attn_rank': 32},  # Alias for q4a4_r32
    'q2a2': {'mlp_lut': 4, 'mlp_rank': 32, 'attn_lut': 4, 'attn_rank': 32},    # 2-bit both
}


def _load_tokenizer_safe(model_id: str, trust_remote_code: bool = True):
    """Load tokenizer with mistral-regex fix when supported."""
    try:
        return AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            use_fast=False,
            fix_mistral_regex=True,
        )
    except TypeError:
        # Older transformers may not support fix_mistral_regex.
        return AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            use_fast=False,
        )


def _load_causal_lm_safe(model_id: str, dtype: torch.dtype, trust_remote_code: bool = True):
    """Load model preferring modern dtype argument, with backward fallback."""
    try:
        return AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
        )


def parse_args():
    parser = argparse.ArgumentParser(description='Test QAT model inference')
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint .pt file or directory')
    parser.add_argument('-b', '--base-dir', metavar='FOLDER',
                        help='Base folder for checkpoint path')
    parser.add_argument('--model-id', type=str, default='Qwen/Qwen3-0.6B',
                        help='Base model ID (default: Qwen/Qwen3-0.6B)')
    parser.add_argument('--prompt', type=str, default=None,
                        help='Single prompt to test')
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='Interactive mode')
    parser.add_argument('--max-tokens', type=int, default=512,
                        help='Max new tokens (default: 512)')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'mps', 'cuda'],
                        help='Inference device override (default: auto)')
    parser.add_argument('--dtype', type=str, default='auto',
                        choices=['auto', 'float32', 'float16', 'bfloat16'],
                        help='Inference dtype override (default: auto)')
    parser.add_argument('--temperature', type=float, default=0.6,
                        help='Sampling temperature (default: 0.6)')
    parser.add_argument('--repetition-penalty', type=float, default=1.1,
                        help='Repetition penalty (default: 1.1)')
    parser.add_argument('--sample', action='store_true',
                        help='Use sampling instead of greedy (non-deterministic, uses temperature/top_p)')
    parser.add_argument('--no-thinking', action='store_true',
                        help='Disable thinking mode (use chat template without <think>)')
    parser.add_argument('--no-template', action='store_true',
                        help='Disable chat template entirely (raw text)')
    parser.add_argument('--test-all-modes', action='store_true',
                        help='Test prompt with all 3 modes: no-template, no-think, think')
    parser.add_argument('--debug', action='store_true',
                        help='Print full prompt template for debugging')

    # Quantization config (overrides config.json if specified)
    parser.add_argument('--config', type=str, default=None,
                        help='Quantization config preset (q4_r32, q2a4, q4a4, q2a2) or path to config.json')
    parser.add_argument('--lut-bits', type=int, default=None,
                        help='LUT bits for MLP (auto-detect from config.json)')
    parser.add_argument('--attn-lut-bits', type=int, default=None,
                        help='LUT bits for attention (auto-detect from config.json)')
    parser.add_argument('--scale-rank', type=int, default=None,
                        help='Scale rank for MLP (auto-detect from config.json)')
    parser.add_argument('--attn-scale-rank', type=int, default=None,
                        help='Scale rank for attention (auto-detect from config.json)')
    parser.add_argument('--version', type=str, default=None, choices=['v1', 'v2'],
                        help='Force V1 or V2 (auto-detect from config.json)')

    # LoRA adapter
    parser.add_argument('--lora', type=str, default=None,
                        help='Path to LoRA adapter checkpoint (loads on top of base V2)')
    parser.add_argument('--lora-r', type=int, default=8,
                        help='LoRA rank (must match saved adapter)')

    return parser.parse_args()


def load_config(checkpoint_path):
    """Load config.json or v2_config.json from checkpoint directory."""
    path = Path(checkpoint_path)

    # Handle both file and directory paths
    if path.is_file():
        config_dir = path.parent
    else:
        config_dir = path

    # Try config.json first, then v2_config.json
    for config_name in ['config.json', 'v2_config.json']:
        config_path = config_dir / config_name
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            print(f"Loaded config from {config_path}")
            return config

    return {}


def load_model(args):
    """Load model with QAT layers and checkpoint."""
    # Resolve device
    if args.device == 'auto':
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    elif args.device == 'mps':
        if not torch.backends.mps.is_available():
            raise RuntimeError("Requested --device mps but MPS is not available")
        device = torch.device('mps')
    elif args.device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda but CUDA is not available")
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Resolve dtype
    if args.dtype == 'auto':
        if device.type == 'mps':
            dtype = torch.float32  # MPS path is most stable in fp32 here
        elif device.type == 'cuda':
            dtype = torch.bfloat16
        else:
            dtype = torch.float32
    elif args.dtype == 'float32':
        dtype = torch.float32
    elif args.dtype == 'float16':
        dtype = torch.float16
    else:
        dtype = torch.bfloat16

    print(f"Device: {device}, dtype: {dtype}")

    # Load config.json to get params
    config = load_config(args.checkpoint)

    # Apply config preset if specified (overrides config.json)
    if args.config:
        if args.config in CONFIG_PRESETS:
            preset = CONFIG_PRESETS[args.config]
            print(f"Using config preset: {args.config}")
            # Preset stores LUT sizes (4 or 16), convert to bits (2 or 4)
            config['mlp_lut_bits'] = int(math.log2(preset['mlp_lut']))
            config['attn_lut_bits'] = int(math.log2(preset['attn_lut']))
            config['mlp_scale_rank'] = preset['mlp_rank']
            config['attn_scale_rank'] = preset['attn_rank']
            config['version'] = 'v2'  # Presets are V2
        elif Path(args.config).exists():
            # Load from file path
            with open(args.config) as f:
                file_config = json.load(f)
            config.update(file_config)
            print(f"Loaded config from: {args.config}")
        else:
            print(f"Warning: Unknown config '{args.config}'. Available presets: {list(CONFIG_PRESETS.keys())}")

    # Determine version (V1 or V2)
    version = args.version or config.get('version', 'v1')
    print(f"Model version: {version.upper()}")

    # Get quantization params from config or args
    # Support both naming conventions: lut_bits/mlp_lut_bits, scale_rank/mlp_scale_rank
    lut_bits = args.lut_bits or config.get('lut_bits') or config.get('mlp_lut_bits', 4)
    attn_lut_bits = args.attn_lut_bits or config.get('attn_lut_bits', lut_bits)
    scale_rank = args.scale_rank or config.get('scale_rank') or config.get('mlp_scale_rank', 4)
    attn_scale_rank = args.attn_scale_rank or config.get('attn_scale_rank', scale_rank)
    model_id = config.get('model_id', args.model_id)
    group_size = config.get('group_size', 16)  # Default matches V2 init

    print(f"Config: lut_bits={lut_bits}, attn_lut_bits={attn_lut_bits}, "
          f"scale_rank={scale_rank}, attn_scale_rank={attn_scale_rank}, group_size={group_size}")

    print(f"Loading base model: {model_id}")

    # Load base model
    model = _load_causal_lm_safe(model_id, dtype=dtype, trust_remote_code=True)
    tokenizer = _load_tokenizer_safe(model_id, trust_remote_code=True)

    # Replace with QAT layers based on version
    print(f"Replacing linears (q{lut_bits}_a{attn_lut_bits}, {version})...")

    if version == 'v2':
        # V2: ANE-friendly rank-by-rank
        from qat_lora import (
            AnemllQuantConfigV2,
            replace_linear_with_anemll_v2,
            freeze_Q_all,
            get_inference_mode_v2,
        )

        mlp_config = AnemllQuantConfigV2(
            lut_size=2**lut_bits,
            scale_rank=scale_rank,
            group_size=group_size,
            force_positive_scales=False,  # Match training config (train_v2_simple.py)
            magnitude_activation='identity',
        )
        attn_config = AnemllQuantConfigV2(
            lut_size=2**attn_lut_bits,
            scale_rank=attn_scale_rank,
            group_size=group_size,
            force_positive_scales=False,  # Match training config (train_v2_simple.py)
            magnitude_activation='identity',
        )

        replace_linear_with_anemll_v2(
            model,
            mlp_config=mlp_config,
            attn_config=attn_config,
            quantize_attn=True,
            verbose=False,
            skip_init=True,  # Skip SVD since we load checkpoint immediately after
        )
    else:
        # V1: Original implementation
        from qat_lora import (
            AnemllQuantConfig,
            replace_linear_with_anemll,
        )

        mlp_config = AnemllQuantConfig(
            lut_size=2**lut_bits,
            scale_rank=scale_rank,
        )
        attn_config = AnemllQuantConfig(
            lut_size=2**attn_lut_bits,
            scale_rank=attn_scale_rank,
        )

        replace_linear_with_anemll(
            model,
            mlp_config=mlp_config,
            attn_config=attn_config,
            quantize_attn=True,
            verbose=False,
        )

    # Resolve checkpoint path
    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.is_dir():
        checkpoint_file = checkpoint_path / 'model_state_dict.pt'
    else:
        checkpoint_file = checkpoint_path

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_file}")
    state_dict = torch.load(checkpoint_file, map_location='cpu')

    # Handle wrapped state dict
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    # Categorize missing/unexpected keys for clearer diagnostics
    # Expected missing: _scales_baked_flag (new buffer, defaults to 0=False)
    # Expected unexpected: _Q (loaded manually), lora_* (loaded after LoRA enabled)
    missing_baked_flag = [k for k in missing if '_scales_baked_flag' in k]
    missing_other = [k for k in missing if '_scales_baked_flag' not in k]

    unexpected_Q = [k for k in unexpected if '._Q' in k]
    unexpected_indices = [k for k in unexpected if '._indices' in k]
    unexpected_lora = [k for k in unexpected if 'lora_' in k]
    unexpected_other = [k for k in unexpected if '._Q' not in k and '._indices' not in k and 'lora_' not in k]

    # Print summary with explanations
    print(f"  Load results:")
    if missing_baked_flag:
        print(f"    Missing: {len(missing_baked_flag)} _scales_baked_flag (OK: new buffer, defaults to False)")
    if missing_other:
        print(f"    Missing: {len(missing_other)} other keys (WARNING: may cause issues)")
        for k in missing_other[:5]:
            print(f"      - {k}")
        if len(missing_other) > 5:
            print(f"      ... and {len(missing_other) - 5} more")
    if unexpected_Q:
        print(f"    Unexpected: {len(unexpected_Q)} _Q buffers (OK: will load manually)")
    if unexpected_indices:
        print(f"    Unexpected: {len(unexpected_indices)} _indices buffers (OK: will load manually)")
    if unexpected_lora:
        print(f"    Unexpected: {len(unexpected_lora)} LoRA keys (OK: will load after LoRA enabled)")
    if unexpected_other:
        print(f"    Unexpected: {len(unexpected_other)} other keys (WARNING: not loaded)")
        for k in unexpected_other[:5]:
            print(f"      - {k}")
        if len(unexpected_other) > 5:
            print(f"      ... and {len(unexpected_other) - 5} more")

    if not missing_other and not unexpected_other:
        print(f"    All keys accounted for!")

    # V2: Manually load _Q buffers if they weren't loaded (None buffers issue)
    if version == 'v2':
        q_loaded = 0
        baked_detected = 0
        for name, m in model.named_modules():
            if type(m).__name__ == 'AnemllQATLinearV2':
                q_key = f"{name}._Q"
                if q_key in state_dict and m._Q is None:
                    # Re-register as buffer so it moves with model.to(device)
                    m.register_buffer("_Q", state_dict[q_key])
                    q_loaded += 1

                # Detect baked scales: all rank_magnitude == 1
                g_key = f"{name}.rank_magnitude"
                if g_key in state_dict:
                    g = state_dict[g_key]
                    if torch.allclose(g, torch.ones_like(g)):
                        m._scales_baked = True
                        baked_detected += 1

        if q_loaded > 0:
            print(f"  Loaded {q_loaded} _Q buffers (manual registration)")
        if baked_detected > 0:
            print(f"  Detected {baked_detected} layers with baked scales (FP16 snap)")

        # CRITICAL: Also load _indices buffers and enable indices-path for forward()
        # Without this, forward() uses stale _Q instead of lut[_indices]
        indices_loaded = 0
        for name, m in model.named_modules():
            if type(m).__name__ == 'AnemllQATLinearV2':
                idx_key = f"{name}._indices"
                if idx_key in state_dict:
                    idx = state_dict[idx_key]
                    if m._indices is None:
                        m.register_buffer("_indices", idx)
                    else:
                        m._indices = idx
                    # CRITICAL: Set _use_indices so forward() uses lut[_indices] not stale _Q
                    m._use_indices = True
                    indices_loaded += 1

        if indices_loaded > 0:
            print(f"  Loaded {indices_loaded} _indices buffers (manual registration)")
            print(f"  _use_indices=True for {indices_loaded} layers (forward uses lut[_indices])")

    # Load LoRA adapter
    # Priority: 1) --lora flag, 2) embedded in checkpoint, 3) none
    has_lora = False
    lora_r = args.lora_r  # Default from CLI
    if version == 'v2':
        from qat_lora.ane_qat_linear_v2 import enable_recovery_lora_all

        if args.lora:
            # Explicit --lora flag: load from separate file
            print(f"\nLoading LoRA adapter: {args.lora}")
            lora_state = torch.load(args.lora, map_location='cpu')
            lora_keys = [k for k in lora_state if 'lora_' in k]
        else:
            # Check if checkpoint contains LoRA weights
            lora_keys = [k for k in state_dict if 'lora_' in k]
            if lora_keys:
                print(f"\nDetected LoRA weights in checkpoint")
                lora_state = state_dict

        if args.lora or lora_keys:
            # Get LoRA rank from config if available (CLI arg takes priority)
            if config.get('recovery_r') and args.lora_r == 8:  # 8 is default
                lora_r = config.get('recovery_r')
            mlp_only = config.get('mlp_only', False)
            skip_k_proj = config.get('skip_k_proj', True)

            # Enable LoRA on V2 layers
            enable_recovery_lora_all(
                model,
                r=lora_r,
                mlp_only=mlp_only,
                skip_k_proj=skip_k_proj,
                verbose=False,
            )

            # Load LoRA weights
            lora_only = {k: lora_state[k] for k in lora_keys}
            missing, unexpected = model.load_state_dict(lora_only, strict=False)
            source = "separate file" if args.lora else "checkpoint"
            print(f"  Loaded {len(lora_keys)} LoRA tensors (from {source})")
            has_lora = True

    # Move to device
    model.to(device)
    model.eval()

    # Freeze for inference
    print("Freezing for inference...")
    if version == 'v2':
        # V2: Freeze Q first if needed
        for name, m in model.named_modules():
            if type(m).__name__ == 'AnemllQATLinearV2':
                if m._Q is None:
                    m.freeze_Q()
                # Only cache W_eff if no LoRA (LoRA adds its own contribution)
                if not has_lora:
                    with torch.no_grad():
                        scales = m._compute_full_scales()
                        m._cached_weight_q = (m._Q * scales).to(m.weight.dtype)

        # Print inference mode
        mode_info = get_inference_mode_v2(model)
        if has_lora:
            print(f"  V2 Inference: {mode_info['total']} layers with LoRA adapters")
        else:
            print(f"  V2 Inference: {mode_info['total']} layers, "
                  f"{mode_info['has_frozen_Q']} with frozen Q")
    else:
        # V1: Standard freeze
        for m in model.modules():
            if type(m).__name__ == 'AnemllQATLinear':
                m.freeze_for_inference()

    # Print diagnostic summary
    print("\n" + "=" * 50)
    print("Model Configuration Summary")
    print("=" * 50)
    print(f"  Version:        {version.upper()}")
    print(f"  Model ID:       {model_id}")
    print(f"  Device:         {device}")
    print(f"  Dtype:          {dtype}")
    print(f"  LUT bits:       {lut_bits} (MLP), {attn_lut_bits} (Attn)")
    print(f"  Scale rank:     {scale_rank} (MLP), {attn_scale_rank} (Attn)")
    print(f"  LoRA:           {'Enabled (r=' + str(lora_r) + ')' if has_lora else 'Disabled'}")
    if version == 'v2':
        print(f"  V2 layers:      {mode_info['total']}")
        if not has_lora:
            print(f"  Frozen Q:       {mode_info['has_frozen_Q']}")
    print("=" * 50)
    print("Model ready!\n")
    return model, tokenizer, device


def generate(model, tokenizer, device, prompt, args, template_mode=None):
    """Generate response for a prompt.

    Args:
        template_mode: Override template mode ('none', 'no-think', 'think')
                      If None, uses args.no_template / args.no_thinking
    """
    # Determine template mode
    if template_mode is not None:
        use_template = template_mode != 'none'
        use_thinking = template_mode == 'think'
        mode_str = template_mode
    else:
        use_template = not args.no_template
        use_thinking = not args.no_thinking and use_template
        mode_str = 'none' if not use_template else ('think' if use_thinking else 'no-think')

    # Build prompt text
    if not use_template:
        # Raw text - no chat template
        text = prompt
    else:
        # Apply chat template
        messages = [{'role': 'user', 'content': prompt}]
        template_kwargs = {
            'tokenize': False,
            'add_generation_prompt': True,
        }
        if use_thinking:
            template_kwargs['enable_thinking'] = True
        else:
            template_kwargs['enable_thinking'] = False

        try:
            text = tokenizer.apply_chat_template(messages, **template_kwargs)
        except TypeError:
            # Tokenizer doesn't support enable_thinking
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Debug: print full template
    if args.debug:
        print(f"\n[DEBUG] Template mode: {mode_str}")
        print(f"[DEBUG] Full prompt ({len(text)} chars):")
        print("-" * 40)
        print(text)
        print("-" * 40)

    inputs = tokenizer(text, return_tensors='pt').to(device)

    if args.debug:
        print(f"[DEBUG] Input tokens: {inputs['input_ids'].shape[1]}")

    with torch.no_grad():
        # Greedy decoding by default (deterministic), sampling with --sample
        if args.sample:
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=0.9,
                repetition_penalty=args.repetition_penalty,
                pad_token_id=tokenizer.eos_token_id,
            )
        else:
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                do_sample=False,
                repetition_penalty=args.repetition_penalty,
                pad_token_id=tokenizer.eos_token_id,
            )

    response = tokenizer.decode(
        output[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=False
    )

    if args.debug:
        print(f"[DEBUG] Raw response ({len(response)} chars):")
        print("-" * 40)
        print(repr(response[:500]))  # First 500 chars, repr to show special chars
        print("-" * 40)
        print(f"[DEBUG] Output tokens: {output[0].shape[0] - inputs['input_ids'].shape[1]}")

    # Clean up common artifacts
    response = response.replace('<|im_end|>', '').strip()
    response = response.replace('<think>\n<think>', '<think>')  # Fix double think

    return response


def run_all_modes(model, tokenizer, device, prompt, args):
    """Test a prompt with all 3 template modes."""
    modes = [
        ('none', 'No Template (raw text)'),
        ('no-think', 'Chat Template (no thinking)'),
        ('think', 'Chat Template (with thinking)'),
    ]

    print(f"\nPrompt: {prompt}")
    print("=" * 60)

    for mode, mode_desc in modes:
        print(f"\n[{mode_desc}]")
        response = generate(model, tokenizer, device, prompt, args, template_mode=mode)
        # Truncate long responses for comparison
        if len(response) > 500:
            response = response[:500] + "..."
        print(f"Response: {response}")
        print("-" * 60)


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

    # Apply base directory if specified
    if args.base_dir:
        args.checkpoint = str(Path(args.base_dir) / args.checkpoint)

    # Check checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    # Load model
    model, tokenizer, device = load_model(args)

    # Run inference
    if args.test_all_modes:
        # Test all 3 template modes
        prompt = args.prompt or "What is the capital of France?"
        run_all_modes(model, tokenizer, device, prompt, args)
    elif args.prompt:
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
