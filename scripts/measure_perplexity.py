#!/usr/bin/env python3
"""
Measure perplexity of a V2 QAT checkpoint or baseline model.

Usage:
    # Baseline model (original Qwen, no QAT)
    python scripts/measure_perplexity.py --baseline
    python scripts/measure_perplexity.py --baseline --model Qwen/Qwen3-0.6B

    # QAT checkpoint (WikiText-2 default)
    python scripts/measure_perplexity.py checkpoint.pt

    # Use existing KD cache
    python scripts/measure_perplexity.py checkpoint.pt --cache-dir caches/alpaca_L128

    # Use custom text file
    python scripts/measure_perplexity.py checkpoint.pt --text-file eval.txt

    # With LoRA
    python scripts/measure_perplexity.py checkpoint.pt --lora-r 8

Perplexity = exp(cross-entropy loss) on next-token prediction.
Lower is better. WikiText-2 baselines: GPT-2 ~22, good LLMs ~5-10.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


def get_device(device_arg: str = 'auto'):
    """Get device and dtype based on availability."""
    if device_arg == 'tpu':
        try:
            import torch_xla.core.xla_model as xm
            return xm.xla_device(), torch.bfloat16
        except ImportError:
            print("Warning: torch_xla not installed, falling back to CPU")
            return torch.device('cpu'), torch.float32

    if device_arg == 'auto':
        # TPU > MPS > CUDA > CPU
        try:
            import torch_xla.core.xla_model as xm
            return xm.xla_device(), torch.bfloat16
        except ImportError:
            pass
        if torch.backends.mps.is_available():
            return torch.device('mps'), torch.float32
        elif torch.cuda.is_available():
            return torch.device('cuda'), torch.bfloat16
        else:
            return torch.device('cpu'), torch.float32
    elif device_arg == 'mps':
        return torch.device('mps'), torch.float32
    elif device_arg == 'cuda':
        return torch.device('cuda'), torch.bfloat16
    else:
        return torch.device('cpu'), torch.float32


def load_wikitext2(tokenizer, split='test'):
    """Download and tokenize WikiText-2."""
    try:
        from datasets import load_dataset
    except ImportError:
        return None

    print(f"Loading WikiText-2 ({split} split)...")
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)
    text = '\n\n'.join(dataset['text'])
    encodings = tokenizer(text, return_tensors='pt')
    return encodings.input_ids


def load_from_cache(cache_dir: str, num_samples: int = 100):
    """Load sequences from existing KD cache."""
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        print(f"Cache directory not found: {cache_dir}")
        return None

    # Load metadata
    meta_path = cache_path / 'meta.json'
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"Cache: {meta.get('dataset_name', 'unknown')} "
              f"(L={meta.get('max_length', '?')}, K={meta.get('topk', '?')})")

    # Find shard files
    shards = sorted(cache_path.glob('shard_*.pt'))
    if not shards:
        # Try loading individual .pt files
        shards = sorted(cache_path.glob('*.pt'))
        shards = [s for s in shards if s.name != 'meta.json']

    if not shards:
        print(f"No shard files found in {cache_dir}")
        return None

    print(f"Loading {num_samples} samples from {len(shards)} shards...")

    all_input_ids = []
    samples_loaded = 0

    for shard_path in shards:
        if samples_loaded >= num_samples:
            break

        try:
            data = torch.load(shard_path, map_location='cpu', weights_only=True)
        except Exception:
            data = torch.load(shard_path, map_location='cpu')

        # Handle different shard formats
        if isinstance(data, dict) and 'input_ids' in data:
            input_ids = data['input_ids']
        elif isinstance(data, torch.Tensor):
            input_ids = data
        else:
            continue

        # Handle 1D, 2D tensors
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        # Take needed samples
        take = min(input_ids.size(0), num_samples - samples_loaded)
        all_input_ids.append(input_ids[:take])
        samples_loaded += take

    if not all_input_ids:
        return None

    # Concatenate all sequences
    combined = torch.cat(all_input_ids, dim=0)
    # Flatten to single long sequence for perplexity computation
    return combined.reshape(1, -1)


def load_text_file(path: str, tokenizer):
    """Load and tokenize custom text file."""
    if not os.path.exists(path):
        print(f"Text file not found: {path}")
        return None

    print(f"Loading text file: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()

    encodings = tokenizer(text, return_tensors='pt')
    return encodings.input_ids


def format_time(seconds: float) -> str:
    """Format seconds as human-readable time."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m{secs:02d}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h{mins:02d}m"


def compute_perplexity(
    model,
    input_ids: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    stride: int = 512,
    max_length: int = 1024,
    verbose: bool = False,
):
    """
    Compute perplexity using sliding window approach.

    Args:
        model: Loaded model
        input_ids: [1, seq_len] tokenized text
        device: Target device
        dtype: Target dtype
        stride: Sliding window stride
        max_length: Context window size
        verbose: Print per-chunk stats

    Returns:
        dict with perplexity, cross_entropy, tokens, time
    """
    model.eval()
    seq_len = input_ids.size(1)

    # Check if TPU
    is_tpu = 'xla' in str(device)
    if is_tpu:
        import torch_xla.core.xla_model as xm

    # Calculate number of chunks
    num_chunks = max(1, (seq_len - 1) // stride)

    nlls = []
    total_tokens = 0
    start_time = time.time()

    prev_end = 0
    chunk_num = 0
    for i in range(0, seq_len - 1, stride):
        chunk_num += 1
        begin = max(i + stride - max_length, 0)
        end = min(i + stride, seq_len)

        # Target length: how many new tokens we're predicting
        target_len = end - prev_end
        prev_end = end

        # Get input chunk
        input_chunk = input_ids[:, begin:end].to(device)

        with torch.no_grad():
            # Forward pass
            if dtype != torch.float32:
                with torch.autocast(device_type=str(device).split(':')[0], dtype=dtype):
                    outputs = model(input_chunk)
            else:
                outputs = model(input_chunk)

            logits = outputs.logits.float()

        # Compute loss on target tokens only (avoid counting overlapping tokens twice)
        # shift_logits: predict positions [begin+1:end]
        # We only want the last target_len predictions
        trg_len = min(target_len, logits.size(1) - 1)
        shift_logits = logits[:, -trg_len-1:-1, :].contiguous()
        shift_labels = input_ids[:, end-trg_len:end].to(device).contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='sum'
        )

        nlls.append(loss.item())
        total_tokens += trg_len

        # TPU: mark_step to avoid graph explosion
        if is_tpu:
            xm.mark_step()

        if verbose:
            chunk_ppl = math.exp(loss.item() / trg_len) if trg_len > 0 else float('inf')
            elapsed_so_far = time.time() - start_time
            if chunk_num > 1:
                avg_time_per_chunk = elapsed_so_far / chunk_num
                remaining_chunks = num_chunks - chunk_num
                eta_seconds = avg_time_per_chunk * remaining_chunks
                eta_str = f" ETA: {format_time(eta_seconds)}"
            else:
                eta_str = ""
            print(f"  Chunk {chunk_num}/{num_chunks}: ppl={chunk_ppl:.2f} (tokens={trg_len}){eta_str}")

    elapsed = time.time() - start_time

    # Compute final perplexity
    total_nll = sum(nlls)
    avg_nll = total_nll / total_tokens if total_tokens > 0 else float('inf')
    ppl = math.exp(avg_nll)

    return {
        'perplexity': ppl,
        'cross_entropy': avg_nll,
        'tokens': total_tokens,
        'time': elapsed,
        'tokens_per_sec': total_tokens / elapsed if elapsed > 0 else 0,
    }


def load_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    dtype: torch.dtype,
    lora_r: int = 0,
    model_name: str = "Qwen/Qwen3-0.6B",
):
    """Load V2 QAT checkpoint."""
    checkpoint_path = Path(checkpoint_path)

    # Find checkpoint file
    if checkpoint_path.is_dir():
        # Look for model_state_dict.pt or any .pt file
        ckpt_file = checkpoint_path / 'model_state_dict.pt'
        if not ckpt_file.exists():
            pt_files = list(checkpoint_path.glob('*.pt'))
            if pt_files:
                ckpt_file = pt_files[0]
            else:
                raise FileNotFoundError(f"No .pt files in {checkpoint_path}")
        config_path = checkpoint_path / 'config.json'
    else:
        ckpt_file = checkpoint_path
        config_path = checkpoint_path.parent / 'config.json'

    # Default quantization params (Q4 for both MLP and attention)
    lut_bits = 4
    attn_lut_bits = 4
    scale_rank = 32
    attn_scale_rank = 8  # Default for attention

    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        lut_bits = config.get('lut_bits', config.get('mlp_lut_bits', 4))
        attn_lut_bits = config.get('attn_lut_bits', lut_bits)
        scale_rank = config.get('scale_rank', 32)
        attn_scale_rank = config.get('attn_scale_rank', scale_rank)
        print(f"Config: Q{lut_bits}/Q{attn_lut_bits}, scale_rank={scale_rank}/{attn_scale_rank}")

    # Load base model
    print(f"Loading base model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype if dtype != torch.float32 else torch.float32,
        trust_remote_code=True,
    )

    # Import QAT modules
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from qat_lora import (
        AnemllQuantConfigV2,
        replace_linear_with_anemll_v2,
    )
    from qat_lora.ane_qat_linear_v2 import AnemllQATLinearV2

    # Convert lut_bits to lut_size
    lut_size = 2 ** lut_bits
    attn_lut_size = 2 ** attn_lut_bits

    # Create separate configs for MLP and attention (different quantization)
    print(f"Converting to V2 QAT model (LUT={lut_size}/{attn_lut_size})...")
    mlp_config = AnemllQuantConfigV2(
        lut_size=lut_size,
        scale_rank=scale_rank,
        force_positive_scales=False,  # Match training config
        magnitude_activation='identity',
    )
    attn_config = AnemllQuantConfigV2(
        lut_size=attn_lut_size,
        scale_rank=attn_scale_rank,
        force_positive_scales=False,  # Match training config
        magnitude_activation='identity',
    )

    replace_linear_with_anemll_v2(
        model,
        mlp_config=mlp_config,
        attn_config=attn_config,
        quantize_attn=True,
        verbose=False,
    )

    # Load checkpoint
    print(f"Loading checkpoint: {ckpt_file}")
    state_dict = torch.load(ckpt_file, map_location='cpu', weights_only=False)

    # Unwrap if needed
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']

    # Load state dict
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    # Expected missing: _scales_baked_flag (196 layers)
    expected_missing = [k for k in missing if '_scales_baked_flag' in k]
    real_missing = [k for k in missing if '_scales_baked_flag' not in k]

    # Handle _Q buffers manually if in state_dict but not loaded
    q_loaded = 0
    for name, module in model.named_modules():
        if isinstance(module, AnemllQATLinearV2):
            q_key = f"{name}._Q"
            if q_key in state_dict and module._Q is None:
                module.register_buffer('_Q', state_dict[q_key])
                q_loaded += 1

    if q_loaded > 0:
        print(f"  Loaded {q_loaded} _Q buffers (manual)")

    # Handle LoRA if present in checkpoint and --lora-r specified
    lora_keys = [k for k in state_dict if 'lora_' in k]
    if lora_keys and lora_r > 0:
        from qat_lora.ane_qat_linear_v2 import enable_recovery_lora_all

        print(f"  Enabling LoRA (r={lora_r})...")
        enable_recovery_lora_all(
            model,
            r=lora_r,
            mlp_only=False,
            skip_k_proj=True,
            verbose=False,
        )

        # Reload LoRA weights
        lora_only = {k: state_dict[k] for k in lora_keys}
        model.load_state_dict(lora_only, strict=False)
        print(f"  Loaded {len(lora_keys)} LoRA tensors")
    elif lora_keys and lora_r == 0:
        print(f"  Warning: checkpoint has {len(lora_keys)} LoRA keys but --lora-r not set")

    if real_missing:
        # Filter out LoRA keys if LoRA not enabled
        if lora_r == 0:
            real_missing = [k for k in real_missing if 'lora_' not in k]
        if real_missing:
            print(f"  Missing keys: {len(real_missing)}")
            for k in real_missing[:3]:
                print(f"    {k}")
            if len(real_missing) > 3:
                print(f"    ... and {len(real_missing) - 3} more")

    # Move to device
    model = model.to(device)
    model.eval()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {total_params / 1e6:.1f}M parameters")

    return model


def load_baseline_model(
    model_name: str,
    device: torch.device,
    dtype: torch.dtype,
):
    """Load original HuggingFace model (no QAT)."""
    print(f"Loading baseline model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype if dtype != torch.float32 else torch.float32,
        trust_remote_code=True,
    )
    model = model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {total_params / 1e6:.1f}M parameters (baseline)")

    return model


def main():
    parser = argparse.ArgumentParser(
        description='Measure perplexity of V2 QAT checkpoint or baseline model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('checkpoint', nargs='?', default=None,
                        help='V2 checkpoint path (.pt file or directory). Optional if --baseline used.')
    parser.add_argument('--dataset', choices=['wikitext2', 'wikitext103'], default='wikitext2',
                        help='Standard dataset to use (default: wikitext2)')
    parser.add_argument('--cache-dir', help='Use KD cache as evaluation data instead of dataset')
    parser.add_argument('--text-file', help='Use custom text file instead of dataset')
    parser.add_argument('--max-length', type=int, default=1024,
                        help='Context window size (default: 1024)')
    parser.add_argument('--stride', type=int, default=512,
                        help='Sliding window stride (default: 512)')
    parser.add_argument('--num-samples', type=int, default=100,
                        help='Number of samples from cache (default: 100)')
    parser.add_argument('--lora-r', type=int, default=0,
                        help='LoRA rank if checkpoint has LoRA (default: 0)')
    parser.add_argument('--model', type=str, default='Qwen/Qwen3-0.6B',
                        help='Base model name (default: Qwen/Qwen3-0.6B)')
    parser.add_argument('--device', choices=['auto', 'mps', 'cuda', 'cpu', 'tpu'], default='auto',
                        help='Device to use (default: auto). TPU requires torch_xla.')
    parser.add_argument('--verbose', action='store_true',
                        help='Show per-chunk perplexity')
    parser.add_argument('--baseline', action='store_true',
                        help='Measure baseline HuggingFace model (no QAT checkpoint needed)')

    args = parser.parse_args()

    # Validate args
    if not args.baseline and args.checkpoint is None:
        parser.error("checkpoint is required unless --baseline is used")

    # Get device
    device, dtype = get_device(args.device)
    print("=" * 60)
    print("PERPLEXITY MEASUREMENT")
    print("=" * 60)
    if args.baseline:
        print(f"Model:      {args.model} (baseline)")
    else:
        print(f"Checkpoint: {args.checkpoint}")
    print(f"Device:     {device} ({dtype})")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Load evaluation data
    input_ids = None
    data_source = None

    if args.text_file:
        input_ids = load_text_file(args.text_file, tokenizer)
        data_source = f"text file: {args.text_file}"
    elif args.cache_dir:
        input_ids = load_from_cache(args.cache_dir, args.num_samples)
        data_source = f"cache: {args.cache_dir} ({args.num_samples} samples)"
    else:
        # Default: WikiText-2
        input_ids = load_wikitext2(tokenizer, split='test')
        data_source = f"{args.dataset} (test split)"

    if input_ids is None:
        print("\nERROR: Could not load evaluation data.")
        if not args.cache_dir and not args.text_file:
            print("WikiText-2 requires 'datasets' library: pip install datasets")
            print("Alternatively, use --cache-dir or --text-file")
        return 1

    print(f"Dataset:    {data_source}")
    print(f"Tokens:     {input_ids.size(1):,}")
    print(f"Context:    {args.max_length} tokens")
    print(f"Stride:     {args.stride} tokens")

    # Load model
    print()
    if args.baseline:
        model = load_baseline_model(
            model_name=args.model,
            device=device,
            dtype=dtype,
        )
    else:
        model = load_checkpoint(
            args.checkpoint,
            device=device,
            dtype=dtype,
            lora_r=args.lora_r,
            model_name=args.model,
        )

    # Compute perplexity
    print("\nComputing perplexity...")
    result = compute_perplexity(
        model=model,
        input_ids=input_ids,
        device=device,
        dtype=dtype,
        stride=args.stride,
        max_length=args.max_length,
        verbose=args.verbose,
    )

    # Print results
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Perplexity:     {result['perplexity']:.2f}")
    print(f"Cross-entropy:  {result['cross_entropy']:.4f} nats")
    print(f"Tokens:         {result['tokens']:,}")
    print(f"Time:           {result['time']:.1f}s ({result['tokens_per_sec']:.0f} tok/s)")

    return 0


if __name__ == '__main__':
    sys.exit(main())
