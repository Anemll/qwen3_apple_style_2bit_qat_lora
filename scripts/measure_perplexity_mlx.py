#!/usr/bin/env python3
"""
Measure perplexity of MLX-LM models (base or quantized).

Usage:
    # Base model (HF or local)
    python scripts/measure_perplexity_mlx.py mlx-community/Qwen2.5-0.5B-Instruct-4bit
    python scripts/measure_perplexity_mlx.py Qwen/Qwen3-0.6B

    # Local MLX model
    python scripts/measure_perplexity_mlx.py ./my_mlx_model

    # Custom dataset
    python scripts/measure_perplexity_mlx.py mlx-community/Qwen2.5-0.5B-4bit --dataset wikitext --split test

    # Different sequence length
    python scripts/measure_perplexity_mlx.py mlx-community/Qwen2.5-0.5B-4bit --seq-len 1024

    # Tag for comparison (appears in results)
    python scripts/measure_perplexity_mlx.py mlx-community/Qwen2.5-0.5B-4bit --tag "mlx-4bit"

Compares MLX-LM quantization with our QAT approach.
Results saved to results/perplexity.json (same format as measure_perplexity.py).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

# MLX imports - deferred to avoid NameError on type hints
MLX_AVAILABLE = False
mx = None  # Will be set if mlx is available

if TYPE_CHECKING:
    import mlx.core as mx

try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load, generate
    MLX_AVAILABLE = True
except ImportError:
    pass  # Warning printed in main()


def format_time(seconds: float) -> str:
    """Format seconds as human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m{secs:02d}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h{mins:02d}m"


def load_wikitext2(tokenizer, max_tokens: int = None):
    """Load WikiText-2 test set and tokenize."""
    from datasets import load_dataset

    print("Loading WikiText-2 test set...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    # Concatenate all text (same as PyTorch script - don't filter empties)
    text = "\n\n".join(dataset["text"])

    # Tokenize - returns list of ints
    # Note: mlx_lm tokenizers may handle special tokens differently
    tokens = tokenizer.encode(text)

    # Debug: show first few tokens to verify tokenization
    print(f"  First 10 tokens: {tokens[:10]}")

    if max_tokens and len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]

    print(f"  Total tokens: {len(tokens):,}")
    return tokens


def load_custom_text(tokenizer, text_file: str, max_tokens: int = None):
    """Load and tokenize custom text file."""
    print(f"Loading text from: {text_file}")
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # Tokenize - returns list of ints
    tokens = tokenizer.encode(text)

    if max_tokens and len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]

    print(f"  Total tokens: {len(tokens):,}")
    return tokens


def compute_perplexity_mlx_sliding(
    model,
    tokens,
    stride: int = 512,
    max_length: int = 1024,
    verbose: bool = True,
):
    """
    Compute perplexity using sliding window (more accurate, slower).

    This matches the exact logic of the PyTorch measure_perplexity.py script.

    Args:
        model: MLX model
        tokens: list or mx.array of token IDs
        stride: Sliding window stride
        max_length: Context window size
        verbose: Print progress

    Returns:
        dict with perplexity, cross_entropy, tokens, time, throughput
    """
    # Convert to mx.array if needed
    if isinstance(tokens, list):
        tokens = mx.array(tokens)

    seq_len = len(tokens)
    num_chunks = max(1, (seq_len - 1) // stride)

    if verbose:
        print(f"\n  Total tokens: {seq_len:,}")
        print(f"  Stride: {stride}, Max length: {max_length}")
        print(f"  Chunks: ~{num_chunks}")

    nlls = []
    total_tokens = 0
    start_time = time.time()

    prev_end = 0
    chunk_num = 0

    # Match PyTorch sliding window logic exactly
    for i in range(0, seq_len - 1, stride):
        chunk_num += 1
        begin = max(i + stride - max_length, 0)
        end = min(i + stride, seq_len)

        # Target length: how many new tokens we're predicting
        target_len = end - prev_end
        prev_end = end

        # Get input chunk
        input_chunk = tokens[begin:end]
        chunk_input = mx.expand_dims(input_chunk, axis=0)  # [1, L]

        # Forward pass
        logits = model(chunk_input)
        if hasattr(logits, 'logits'):
            logits = logits.logits
        elif isinstance(logits, tuple):
            logits = logits[0]

        # Compute loss on target tokens only (avoid counting overlapping tokens twice)
        # shift_logits: predict positions [begin+1:end]
        # We only want the last target_len predictions
        trg_len = min(target_len, logits.shape[1] - 1)
        if trg_len <= 0:
            continue

        # shift_logits: logits at positions [-trg_len-1:-1]
        shift_logits = logits[0, -trg_len-1:-1, :]  # [trg_len, V]
        # shift_labels: tokens at positions [end-trg_len:end]
        shift_labels = tokens[end-trg_len:end]  # [trg_len]

        # Compute log_softmax
        log_probs = shift_logits - mx.logsumexp(shift_logits, axis=-1, keepdims=True)

        # Gather log probs at label positions
        labels_expanded = mx.expand_dims(shift_labels, axis=-1)  # [trg_len, 1]
        selected_log_probs = mx.take_along_axis(log_probs, labels_expanded, axis=-1)  # [trg_len, 1]
        selected_log_probs = mx.squeeze(selected_log_probs, axis=-1)  # [trg_len]

        # NLL = -sum(log_probs)
        nll = -mx.sum(selected_log_probs)
        mx.eval(nll)

        nlls.append(float(nll))
        total_tokens += trg_len

        # Progress every 50 chunks
        is_first = chunk_num == 1
        is_last = chunk_num == num_chunks
        is_milestone = chunk_num % 50 == 0
        if verbose and (is_first or is_last or is_milestone):
            chunk_ppl = math.exp(float(nll) / trg_len) if trg_len > 0 else float('inf')
            elapsed_so_far = time.time() - start_time
            if chunk_num > 1:
                avg_time = elapsed_so_far / chunk_num
                eta = avg_time * (num_chunks - chunk_num)
                eta_str = f" ETA: {format_time(eta)}"
            else:
                eta_str = ""
            print(f"  Chunk {chunk_num}/{num_chunks}: ppl={chunk_ppl:.2f} (tokens={trg_len}){eta_str}")

    elapsed = time.time() - start_time
    total_nll = sum(nlls)
    avg_nll = total_nll / total_tokens if total_tokens > 0 else float('inf')
    ppl = math.exp(avg_nll)

    return {
        'perplexity': ppl,
        'cross_entropy': avg_nll,
        'tokens': total_tokens,
        'time': elapsed,
        'tokens_per_sec': total_tokens / elapsed if elapsed > 0 else 0,
        'mode': 'sliding_window',
        'stride': stride,
        'max_length': max_length,
    }


def compute_perplexity_mlx(
    model,
    tokens,
    batch_size: int = 1,
    seq_len: int = 512,
    verbose: bool = True,
):
    """
    Compute perplexity using batched non-overlapping chunks (faster, less accurate).

    Args:
        model: MLX model
        tokens: list or mx.array of token IDs
        batch_size: Batch size for parallel processing
        seq_len: Sequence length per chunk
        verbose: Print progress

    Returns:
        dict with perplexity, cross_entropy, tokens, time, throughput
    """
    # Convert to mx.array if needed
    if isinstance(tokens, list):
        tokens = mx.array(tokens)

    total_tokens = len(tokens)

    # Split into fixed-length chunks
    num_chunks = total_tokens // seq_len
    if num_chunks == 0:
        print(f"Warning: input ({total_tokens} tokens) shorter than seq_len ({seq_len})")
        seq_len = total_tokens
        num_chunks = 1

    usable_tokens = num_chunks * seq_len
    chunks = tokens[:usable_tokens].reshape(num_chunks, seq_len)

    if verbose:
        print(f"\n  Total tokens: {total_tokens:,}")
        print(f"  Usable tokens: {usable_tokens:,} ({num_chunks} chunks x {seq_len})")
        print(f"  Batch size: {batch_size}")

    nlls = []
    tokens_processed = 0
    start_time = time.time()

    num_batches = (num_chunks + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx + 1) * batch_size, num_chunks)
        batch_chunks = chunks[batch_start:batch_end]  # [B, L]

        B = batch_chunks.shape[0]

        # Forward pass - MLX models expect [B, L] input
        # Most MLX-LM models return logits directly or as part of output
        try:
            # Try standard MLX-LM model interface
            logits = model(batch_chunks)

            # Handle different output formats
            if hasattr(logits, 'logits'):
                logits = logits.logits
            elif isinstance(logits, tuple):
                logits = logits[0]
        except Exception as e:
            print(f"Error in forward pass: {e}")
            raise

        # logits shape: [B, L, V]
        # Predict token[i+1] from position[i]
        shift_logits = logits[:, :-1, :]  # [B, L-1, V]
        shift_labels = batch_chunks[:, 1:]  # [B, L-1]

        # Compute cross-entropy loss using numerically stable log_softmax
        # log_softmax = x - logsumexp(x)
        log_probs = shift_logits - mx.logsumexp(shift_logits, axis=-1, keepdims=True)

        # Gather log probs at label positions using take_along_axis
        # shift_labels: [B, L-1], expand to [B, L-1, 1] for gathering
        B, L_minus_1, V = shift_logits.shape
        labels_expanded = mx.expand_dims(shift_labels, axis=-1)  # [B, L-1, 1]
        selected_log_probs = mx.take_along_axis(log_probs, labels_expanded, axis=-1)  # [B, L-1, 1]
        selected_log_probs = mx.squeeze(selected_log_probs, axis=-1)  # [B, L-1]

        # NLL = -log_prob, sum over all tokens
        nll = -mx.sum(selected_log_probs)

        # Evaluate to get value
        mx.eval(nll)
        nll_value = float(nll)

        batch_tokens = B * L_minus_1
        nlls.append(nll_value)
        tokens_processed += batch_tokens

        # Print progress
        is_first = batch_idx == 0
        is_last = batch_idx == num_batches - 1
        is_milestone = (batch_idx + 1) % 20 == 0
        if verbose and (is_first or is_last or is_milestone):
            batch_ppl = math.exp(nll_value / batch_tokens) if batch_tokens > 0 else float('inf')
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
        'mode': 'batched',
        'batch_size': batch_size,
        'seq_len': seq_len,
        'num_batches': num_batches,
    }


def save_results_to_json(
    model_path: str,
    result: dict,
    tag: str = None,
    dataset: str = "wikitext2",
):
    """Save results to results/perplexity.json."""
    from datetime import datetime

    repo_root = Path(__file__).resolve().parent.parent
    results_dir = repo_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / "perplexity.json"

    # Load existing results
    existing_results = {}
    if results_file.exists():
        try:
            with open(results_file, 'r') as f:
                existing_results = json.load(f)
        except (json.JSONDecodeError, IOError):
            existing_results = {}

    # Create key
    model_name = Path(model_path).name if '/' not in model_path else model_path.split('/')[-1]
    if tag:
        key = f"mlx:{tag}/{model_name}"
    else:
        key = f"mlx:{model_name}"

    # Create entry
    entry = {
        "perplexity": round(result['perplexity'], 2),
        "cross_entropy": round(result['cross_entropy'], 4),
        "tokens": result['tokens'],
        "time_seconds": round(result['time'], 1),
        "tokens_per_sec": round(result['tokens_per_sec'], 0),
        "model": model_path,
        "dtype": "mlx",
        "dataset": dataset,
        "timestamp": datetime.now().isoformat(),
    }

    if 'batch_size' in result:
        entry['batch_size'] = result['batch_size']
        entry['seq_len'] = result['seq_len']
    if 'mode' in result:
        entry['mode'] = result['mode']
        if result['mode'] == 'sliding_window':
            entry['stride'] = result['stride']
            entry['max_length'] = result['max_length']

    existing_results[key] = entry

    with open(results_file, 'w') as f:
        json.dump(existing_results, f, indent=2)

    print(f"\nSaved to: {results_file}")
    print(f"Key: {key}")

    return results_file, key


def print_all_results():
    """Print all saved results from results/perplexity.json."""
    repo_root = Path(__file__).resolve().parent.parent
    results_file = repo_root / "results" / "perplexity.json"

    if not results_file.exists():
        print("No results found. Run perplexity measurements first.")
        return 1

    with open(results_file, 'r') as f:
        results = json.load(f)

    if not results:
        print("No results found in perplexity.json")
        return 1

    # Sort by perplexity
    sorted_results = sorted(results.items(), key=lambda x: x[1].get('perplexity', float('inf')))

    # ANSI formatting
    BOLD = "\033[1m"
    GREEN = "\033[92m"
    CYAN = "\033[96m"
    RESET = "\033[0m"

    print()
    print("=" * 70)
    print(f"{BOLD}PERPLEXITY RESULTS{RESET} (sorted by best PPL)")
    print("=" * 70)

    for rank, (key, entry) in enumerate(sorted_results, 1):
        ppl = entry.get('perplexity', 0)
        ce = entry.get('cross_entropy', 0)
        dtype = entry.get('dtype', '?')
        dataset = entry.get('dataset', '?')

        # Truncate dataset if too long
        if len(dataset) > 15:
            dataset = dataset[:13] + ".."

        # Highlight rank
        print(f"{CYAN}#{rank}{RESET} {BOLD}{key}{RESET}")

        if rank == 1:
            print(f"    {GREEN}PPL: {ppl:.2f}{RESET}  CE: {ce:.4f}  [{dtype}]  {dataset}")
        else:
            print(f"    PPL: {ppl:.2f}  CE: {ce:.4f}  [{dtype}]  {dataset}")
        print()

    print("-" * 70)
    print(f"Total: {len(results)} results")
    print("=" * 70)

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Measure perplexity of MLX-LM models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument("model", nargs="?", default=None,
                        help="MLX model path (HF hub or local). E.g., mlx-community/Qwen2.5-0.5B-4bit")

    # Dataset options
    parser.add_argument("--dataset", type=str, default="wikitext",
                        choices=["wikitext", "custom"],
                        help="Dataset to use (default: wikitext)")
    parser.add_argument("--text-file", type=str, default=None,
                        help="Custom text file for evaluation")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Maximum tokens to evaluate (default: all)")

    # Processing options
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for batched mode (default: 1)")
    parser.add_argument("--seq-len", type=int, default=512,
                        help="Sequence length per chunk (default: 512)")
    parser.add_argument("--stride", type=int, default=512,
                        help="Stride for sliding window mode (default: 512)")
    parser.add_argument("--max-length", type=int, default=1024,
                        help="Max context length for sliding window (default: 1024)")
    parser.add_argument("--batched", action="store_true",
                        help="Use faster batched mode instead of sliding window")

    # Output options
    parser.add_argument("--tag", type=str, default=None,
                        help="Tag for this run (appears in results key)")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save results to JSON")
    parser.add_argument("--list", action="store_true",
                        help="List all saved perplexity results")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")
    parser.add_argument("-b", "--base-dir", metavar="FOLDER",
                        help="Base folder for model path")

    args = parser.parse_args()

    # Handle --list
    if args.list:
        return print_all_results()

    # Apply base directory if specified
    if args.base_dir and args.model:
        args.model = str(Path(args.base_dir) / args.model)

    # Require model for measurement
    if args.model is None:
        parser.print_help()
        print("\nError: model path required (or use --list)")
        return 1

    if not MLX_AVAILABLE:
        print("Error: mlx-lm not installed. Install with: pip install mlx-lm")
        return 1

    # Load model
    print("=" * 60)
    print(f"MLX PERPLEXITY MEASUREMENT")
    print("=" * 60)
    print(f"Model: {args.model}")

    print("\nLoading model...")
    start_load = time.time()

    try:
        model, tokenizer = load(args.model)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nTip: For HuggingFace models, try the mlx-community versions:")
        print("  mlx-community/Qwen2.5-0.5B-Instruct-4bit")
        print("  mlx-community/Llama-3.2-1B-Instruct-4bit")
        return 1

    load_time = time.time() - start_load
    print(f"  Loaded in {load_time:.1f}s")

    # Load dataset
    if args.text_file:
        tokens = load_custom_text(tokenizer, args.text_file, args.max_tokens)
        dataset_name = Path(args.text_file).name
    else:
        tokens = load_wikitext2(tokenizer, args.max_tokens)
        dataset_name = "wikitext2"

    # Compute perplexity
    if args.batched:
        print(f"\nComputing perplexity (batched mode, B={args.batch_size}, L={args.seq_len})...")
        result = compute_perplexity_mlx(
            model=model,
            tokens=tokens,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            verbose=True,
        )
    else:
        print(f"\nComputing perplexity (sliding window, stride={args.stride}, max_len={args.max_length})...")
        result = compute_perplexity_mlx_sliding(
            model=model,
            tokens=tokens,
            stride=args.stride,
            max_length=args.max_length,
            verbose=True,
        )

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Perplexity:     {result['perplexity']:.2f}")
    print(f"Cross-entropy:  {result['cross_entropy']:.4f}")
    print(f"Tokens:         {result['tokens']:,}")
    print(f"Time:           {format_time(result['time'])}")
    print(f"Throughput:     {result['tokens_per_sec']:.0f} tok/s")
    print("=" * 60)

    # Save results
    if not args.no_save:
        save_results_to_json(
            model_path=args.model,
            result=result,
            tag=args.tag,
            dataset=dataset_name,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
