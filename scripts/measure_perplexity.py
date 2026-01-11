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

    # Force specific dtype (fp16, bf16, or fp32)
    python scripts/measure_perplexity.py checkpoint.pt --dtype fp16
    python scripts/measure_perplexity.py --baseline --dtype bf16

    # Batch processing (faster on GPU/TPU)
    python scripts/measure_perplexity.py checkpoint.pt --batch-size 8 --seq-len 512
    python scripts/measure_perplexity.py --baseline --batch-size 4

    # Benchmark different batch sizes
    python scripts/measure_perplexity.py --baseline --benchmark
    python scripts/measure_perplexity.py checkpoint.pt --benchmark --seq-len 1024

Modes:
    - Sliding window (default): Overlapping chunks for accurate PPL
    - Batched (--batch-size N): Parallel processing, faster on GPU/TPU

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


def parse_dtype(dtype_arg: str) -> torch.dtype:
    """Parse dtype string to torch.dtype."""
    dtype_map = {
        'fp16': torch.float16,
        'bf16': torch.bfloat16,
        'fp32': torch.float32,
    }
    return dtype_map.get(dtype_arg)


def get_device(device_arg: str = 'auto', dtype_arg: str = 'auto'):
    """Get device and dtype based on availability and user preference."""
    # First determine device
    device = None
    default_dtype = None

    if device_arg == 'tpu':
        try:
            import torch_xla.core.xla_model as xm
            device = xm.xla_device()
            default_dtype = torch.bfloat16
        except ImportError:
            print("Warning: torch_xla not installed, falling back to CPU")
            device = torch.device('cpu')
            default_dtype = torch.float32
    elif device_arg == 'auto':
        # TPU > MPS > CUDA > CPU
        try:
            import torch_xla.core.xla_model as xm
            device = xm.xla_device()
            default_dtype = torch.bfloat16
        except ImportError:
            pass
        if device is None:
            if torch.backends.mps.is_available():
                device = torch.device('mps')
                default_dtype = torch.float32
            elif torch.cuda.is_available():
                device = torch.device('cuda')
                default_dtype = torch.bfloat16
            else:
                device = torch.device('cpu')
                default_dtype = torch.float32
    elif device_arg == 'mps':
        device = torch.device('mps')
        default_dtype = torch.float32
    elif device_arg == 'cuda':
        device = torch.device('cuda')
        default_dtype = torch.bfloat16
    else:
        device = torch.device('cpu')
        default_dtype = torch.float32

    # Apply dtype override if specified
    if dtype_arg != 'auto':
        dtype = parse_dtype(dtype_arg)
        if dtype is None:
            print(f"Warning: Unknown dtype '{dtype_arg}', using default")
            dtype = default_dtype
    else:
        dtype = default_dtype

    return device, dtype


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


def compute_perplexity_batched(
    model,
    input_ids: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int = 4,
    seq_len: int = 512,
    verbose: bool = False,
):
    """
    Compute perplexity using batched processing for GPU/TPU parallelism.

    Unlike sliding window, this splits data into independent non-overlapping
    chunks and processes them in batches. Faster on GPU/TPU but slightly
    less accurate than sliding window due to no context overlap.

    Args:
        model: Loaded model
        input_ids: [1, total_tokens] tokenized text
        device: Target device
        dtype: Target dtype
        batch_size: Number of sequences per batch
        seq_len: Fixed sequence length for each chunk
        verbose: Print per-batch stats

    Returns:
        dict with perplexity, cross_entropy, tokens, time, throughput
    """
    model.eval()
    total_tokens = input_ids.size(1)

    # Check if TPU
    is_tpu = 'xla' in str(device)
    if is_tpu:
        import torch_xla.core.xla_model as xm

    # Split into fixed-length chunks (discard remainder)
    # Each chunk is [seq_len] tokens
    num_chunks = total_tokens // seq_len
    if num_chunks == 0:
        print(f"Warning: input ({total_tokens} tokens) shorter than seq_len ({seq_len})")
        seq_len = total_tokens
        num_chunks = 1

    # Reshape to [num_chunks, seq_len]
    usable_tokens = num_chunks * seq_len
    chunks = input_ids[0, :usable_tokens].reshape(num_chunks, seq_len)

    if verbose:
        print(f"  Total tokens: {total_tokens:,}")
        print(f"  Usable tokens: {usable_tokens:,} ({num_chunks} chunks × {seq_len})")
        print(f"  Batch size: {batch_size}")
        print(f"  Batches: {(num_chunks + batch_size - 1) // batch_size}")

    nlls = []
    tokens_processed = 0
    start_time = time.time()

    num_batches = (num_chunks + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx + 1) * batch_size, num_chunks)
        batch_chunks = chunks[batch_start:batch_end].to(device)  # [B, L]

        with torch.no_grad():
            # Forward pass
            if dtype != torch.float32:
                device_type = 'xla' if is_tpu else str(device).split(':')[0]
                if device_type == 'xla':
                    # TPU: no autocast needed, model already in bf16
                    outputs = model(batch_chunks)
                else:
                    with torch.autocast(device_type=device_type, dtype=dtype):
                        outputs = model(batch_chunks)
            else:
                outputs = model(batch_chunks)

            logits = outputs.logits.float()  # [B, L, V]

        # Compute cross-entropy loss
        # Predict token[i+1] from position[i], so shift by 1
        # logits[:, :-1] predicts labels[:, 1:]
        shift_logits = logits[:, :-1, :].contiguous()  # [B, L-1, V]
        shift_labels = batch_chunks[:, 1:].contiguous()  # [B, L-1]

        # Flatten for cross_entropy
        B, L_minus_1, V = shift_logits.shape
        loss = F.cross_entropy(
            shift_logits.view(B * L_minus_1, V),
            shift_labels.view(B * L_minus_1),
            reduction='sum'
        )

        batch_tokens = B * L_minus_1
        nlls.append(loss.item())
        tokens_processed += batch_tokens

        # TPU: mark_step to execute graph
        if is_tpu:
            xm.mark_step()

        if verbose:
            batch_ppl = math.exp(loss.item() / batch_tokens) if batch_tokens > 0 else float('inf')
            elapsed_so_far = time.time() - start_time
            throughput = tokens_processed / elapsed_so_far if elapsed_so_far > 0 else 0
            if batch_idx > 0:
                avg_time = elapsed_so_far / (batch_idx + 1)
                eta = avg_time * (num_batches - batch_idx - 1)
                eta_str = f" ETA: {format_time(eta)}"
            else:
                eta_str = ""
            print(f"  Batch {batch_idx + 1}/{num_batches}: ppl={batch_ppl:.2f} "
                  f"({batch_tokens} tokens, {throughput:.0f} tok/s){eta_str}")

    elapsed = time.time() - start_time

    # Compute final perplexity
    total_nll = sum(nlls)
    avg_nll = total_nll / tokens_processed if tokens_processed > 0 else float('inf')
    ppl = math.exp(avg_nll)

    return {
        'perplexity': ppl,
        'cross_entropy': avg_nll,
        'tokens': tokens_processed,
        'time': elapsed,
        'tokens_per_sec': tokens_processed / elapsed if elapsed > 0 else 0,
        'batch_size': batch_size,
        'seq_len': seq_len,
        'num_batches': num_batches,
    }


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

    # Check for config.json or v2_config.json
    config_dir = checkpoint_path.parent if checkpoint_path.is_file() else checkpoint_path
    config = {}
    config_found = None
    for config_name in ['config.json', 'v2_config.json']:
        config_path = config_dir / config_name
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            config_found = config_path
            break

    if config_found:
        print(f"Config:     {config_found}")
        # Support both naming conventions: lut_bits/mlp_lut_bits, scale_rank/mlp_scale_rank
        lut_bits = config.get('lut_bits') or config.get('mlp_lut_bits') or 4
        attn_lut_bits = config.get('attn_lut_bits') or lut_bits
        scale_rank = config.get('scale_rank') or config.get('mlp_scale_rank') or 32
        attn_scale_rank = config.get('attn_scale_rank') or scale_rank
        print(f"  MLP:      Q{lut_bits} (LUT{2**lut_bits}), rank={scale_rank}")
        print(f"  Attn:     Q{attn_lut_bits} (LUT{2**attn_lut_bits}), rank={attn_scale_rank}")
        if 'lora_r' in config and config['lora_r'] > 0:
            print(f"  LoRA:     r={config['lora_r']}, alpha={config.get('lora_alpha', config['lora_r'])}")
    else:
        print(f"Config:     (not found, using defaults)")
        print(f"  MLP:      Q{lut_bits} (LUT{2**lut_bits}), rank={scale_rank}")
        print(f"  Attn:     Q{attn_lut_bits} (LUT{2**attn_lut_bits}), rank={attn_scale_rank}")

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
    parser.add_argument('--dtype', choices=['auto', 'fp16', 'bf16', 'fp32'], default='auto',
                        help='Model dtype (default: auto, uses device default)')
    parser.add_argument('--verbose', action='store_true',
                        help='Show per-chunk perplexity')
    parser.add_argument('--baseline', action='store_true',
                        help='Measure baseline HuggingFace model (no QAT checkpoint needed)')
    # Batch processing options
    parser.add_argument('--batch-size', type=int, default=0,
                        help='Batch size for parallel processing (0=use sliding window, default: 0). '
                             'Higher values = faster on GPU/TPU but use more memory.')
    parser.add_argument('--seq-len', type=int, default=512,
                        help='Sequence length for batched mode (default: 512)')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run benchmark comparing different batch sizes (1,2,4,8,16)')

    args = parser.parse_args()

    # Validate args
    if not args.baseline and args.checkpoint is None:
        parser.error("checkpoint is required unless --baseline is used")

    # Get device and dtype
    device, dtype = get_device(args.device, args.dtype)
    dtype_name = {torch.float16: 'fp16', torch.bfloat16: 'bf16', torch.float32: 'fp32'}.get(dtype, str(dtype))
    print("=" * 60)
    print("PERPLEXITY MEASUREMENT")
    print("=" * 60)
    if args.baseline:
        print(f"Model:      {args.model} (baseline)")
    else:
        print(f"Checkpoint: {args.checkpoint}")
    print(f"Device:     {device}")
    print(f"Dtype:      {dtype_name}" + (" (override)" if args.dtype != 'auto' else " (auto)"))

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

    # Show mode info
    if args.batch_size > 0 or args.benchmark:
        print(f"Mode:       Batched (seq_len={args.seq_len})")
        if args.benchmark:
            print(f"Benchmark:  Testing batch sizes 1,2,4,8,16")
        else:
            print(f"Batch size: {args.batch_size}")
    else:
        print(f"Mode:       Sliding window")
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

    # Benchmark mode: test multiple batch sizes
    if args.benchmark:
        print("\n" + "=" * 60)
        print("BATCH SIZE BENCHMARK")
        print("=" * 60)

        batch_sizes = [1, 2, 4, 8, 16]
        results = []

        for bs in batch_sizes:
            print(f"\n--- Batch size: {bs} ---")
            try:
                result = compute_perplexity_batched(
                    model=model,
                    input_ids=input_ids,
                    device=device,
                    dtype=dtype,
                    batch_size=bs,
                    seq_len=args.seq_len,
                    verbose=False,
                )
                results.append((bs, result))
                print(f"  PPL: {result['perplexity']:.2f} | "
                      f"{result['tokens_per_sec']:.0f} tok/s | "
                      f"{result['time']:.1f}s")
            except RuntimeError as e:
                if 'out of memory' in str(e).lower() or 'OOM' in str(e):
                    print(f"  OOM at batch_size={bs}")
                    break
                raise

        # Summary table
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)
        print(f"{'Batch':>6} | {'PPL':>8} | {'Tok/s':>10} | {'Time':>8} | {'Speedup':>8}")
        print("-" * 60)
        baseline_time = results[0][1]['time'] if results else 1

        for bs, res in results:
            speedup = baseline_time / res['time'] if res['time'] > 0 else 0
            print(f"{bs:>6} | {res['perplexity']:>8.2f} | {res['tokens_per_sec']:>10.0f} | "
                  f"{res['time']:>7.1f}s | {speedup:>7.2f}x")

        return 0

    # Compute perplexity (batched or sliding window)
    if args.batch_size > 0:
        print(f"\nComputing perplexity (batched, B={args.batch_size}, L={args.seq_len})...")
        result = compute_perplexity_batched(
            model=model,
            input_ids=input_ids,
            device=device,
            dtype=dtype,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            verbose=args.verbose,
        )
    else:
        print("\nComputing perplexity (sliding window)...")
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
    if args.batch_size > 0:
        print(f"Batches:        {result['num_batches']} (B={result['batch_size']}, L={result['seq_len']})")

    return 0


if __name__ == '__main__':
    sys.exit(main())
