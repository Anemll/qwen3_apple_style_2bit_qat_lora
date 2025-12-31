#!/usr/bin/env python3
"""
SpeedRun Benchmark - Compare training performance with different optimizations.

Usage:
    python scripts/speedrun.py --cache-dir caches/alpaca_chat_think_both_L64_K64_R128

This script runs short training benchmarks to measure:
- Step time (seconds per step)
- Peak GPU memory usage
- Throughput (tokens/sec)

Benchmarks:
1. Baseline (batch=8)
2. Gradient checkpointing (batch=8)
3. Gradient checkpointing (batch=16) - to show memory savings
"""

import argparse
import os
import sys
import gc
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

REPO_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_DIR))

import torch


@dataclass
class BenchmarkResult:
    name: str
    batch_size: int
    seq_len: int
    steps: int
    total_time: float
    peak_memory_mb: float
    step_time: float
    tokens_per_sec: float
    final_loss: float
    success: bool
    error: Optional[str] = None


def get_gpu_memory_mb() -> float:
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # MPS doesn't have easy memory tracking
        return 0.0
    return 0.0


def reset_gpu_memory():
    """Reset GPU memory tracking."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def get_seq_len_from_cache(cache_dir: str) -> int:
    """Extract sequence length from cache directory name or first shard."""
    # Try to extract from name (e.g., alpaca_chat_think_both_L64_K64_R128)
    import re
    match = re.search(r'_L(\d+)_', cache_dir)
    if match:
        return int(match.group(1))

    # Fallback: load first shard and check
    import glob
    shards = sorted(glob.glob(f"{cache_dir}/*.pt"))
    if shards:
        data = torch.load(shards[0], map_location='cpu')
        if 'input_ids' in data:
            return data['input_ids'].shape[-1]

    return 64  # Default


def run_benchmark(
    cache_dir: str,
    batch_size: int,
    steps: int,
    gradient_checkpointing: bool,
    device: torch.device,
    model_id: str = "Qwen/Qwen3-0.6B",
) -> BenchmarkResult:
    """Run a single benchmark configuration."""
    from transformers import AutoModelForCausalLM
    from qat_lora import (
        AnemllQuantConfigV2,
        replace_linear_with_anemll_v2,
        freeze_Q_all,
        train_e2e,
    )

    # Get sequence length for tokens/sec calculation
    seq_len = get_seq_len_from_cache(cache_dir)

    name = f"batch={batch_size}"
    if gradient_checkpointing:
        name += "+checkpointing"

    print(f"\n{'='*60}")
    print(f"Benchmark: {name}")
    print(f"{'='*60}")

    reset_gpu_memory()

    try:
        # Load model
        print("  Loading model...", end=" ", flush=True)
        t0 = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        print(f"done ({time.time()-t0:.1f}s)")

        # Replace with V2 layers
        print("  Replacing layers...", end=" ", flush=True)
        t0 = time.time()
        mlp_config = AnemllQuantConfigV2(
            lut_size=4,
            scale_rank=32,
            group_size=32,
            force_positive_scales=False,
            magnitude_activation='identity',
            use_ste_fp16=True,
        )
        attn_config = AnemllQuantConfigV2(
            lut_size=16,
            scale_rank=8,
            group_size=32,
            force_positive_scales=False,
            magnitude_activation='identity',
            use_ste_fp16=True,
        )
        replace_linear_with_anemll_v2(
            model,
            mlp_config=mlp_config,
            attn_config=attn_config,
            quantize_attn=True,
            quantize_lm_head=False,
            verbose=False,
        )
        print(f"done ({time.time()-t0:.1f}s)")

        # Move to device
        print("  Moving to GPU...", end=" ", flush=True)
        t0 = time.time()
        model.to(device)
        print(f"done ({time.time()-t0:.1f}s)")

        # Enable gradient checkpointing if requested
        if gradient_checkpointing:
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
                print("  Gradient checkpointing: ENABLED")
            else:
                print("  Warning: Model doesn't support gradient checkpointing")

        # Freeze Q
        print("  Freezing Q...", end=" ", flush=True)
        t0 = time.time()
        freeze_Q_all(model, verbose=False)
        print(f"done ({time.time()-t0:.1f}s)")

        # Run training benchmark
        print(f"  Running {steps} steps with batch={batch_size}...")
        reset_gpu_memory()

        start_time = time.time()
        result = train_e2e(
            model=model,
            cache_dir=cache_dir,
            device=device,
            max_steps=steps,
            batch_size=batch_size,
            lr=5e-5,
            use_cosine_schedule=True,
            warmup_steps=10,
            temperature=2.0,
            train_weights=False,
            train_scales=True,
            logging_steps=steps + 1,  # Don't log during benchmark
            eval_steps=steps + 1,  # Don't eval during benchmark
            verbose=False,
            use_fp16=False,
        )
        total_time = time.time() - start_time

        peak_memory = get_gpu_memory_mb()
        step_time = total_time / steps
        tokens_per_sec = (batch_size * seq_len * steps) / total_time
        final_loss = result.get('final_loss', 0.0)

        # Cleanup
        del model
        reset_gpu_memory()

        return BenchmarkResult(
            name=name,
            batch_size=batch_size,
            seq_len=seq_len,
            steps=steps,
            total_time=total_time,
            peak_memory_mb=peak_memory,
            step_time=step_time,
            tokens_per_sec=tokens_per_sec,
            final_loss=final_loss,
            success=True,
        )

    except Exception as e:
        # Cleanup on error
        try:
            del model
        except:
            pass
        reset_gpu_memory()

        return BenchmarkResult(
            name=name,
            batch_size=batch_size,
            seq_len=seq_len,
            steps=steps,
            total_time=0,
            peak_memory_mb=get_gpu_memory_mb(),
            step_time=0,
            tokens_per_sec=0,
            final_loss=0,
            success=False,
            error=str(e),
        )


def print_results(results: list[BenchmarkResult]):
    """Print benchmark results in a nice table."""
    print("\n")
    print("=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)

    # Show seq_len from first result
    if results:
        print(f"Sequence length: {results[0].seq_len}")
    print()

    # Header
    print(f"{'Config':<30} {'Batch':>6} {'Step(s)':>8} {'Memory':>10} {'t/s':>10} {'Status':>8}")
    print("-" * 80)

    for r in results:
        if r.success:
            print(f"{r.name:<30} {r.batch_size:>6} {r.step_time:>8.3f} {r.peak_memory_mb:>9.0f}M {r.tokens_per_sec:>10.0f} {'OK':>8}")
        else:
            error_short = r.error[:20] if r.error else "Unknown"
            print(f"{r.name:<30} {r.batch_size:>6} {'---':>8} {r.peak_memory_mb:>9.0f}M {'---':>10} {'FAIL':>8}")
            print(f"  Error: {error_short}...")

    print("-" * 80)

    # Summary
    successful = [r for r in results if r.success]
    if len(successful) >= 2:
        baseline = successful[0]
        for r in successful[1:]:
            speedup = r.tokens_per_sec / baseline.tokens_per_sec if baseline.tokens_per_sec > 0 else 0
            memory_change = (r.peak_memory_mb - baseline.peak_memory_mb) / baseline.peak_memory_mb * 100 if baseline.peak_memory_mb > 0 else 0
            print(f"\n{r.name} vs {baseline.name}:")
            print(f"  Throughput: {speedup:.2f}x")
            print(f"  Memory: {memory_change:+.1f}%")


def main():
    parser = argparse.ArgumentParser(description='SpeedRun Benchmark')
    parser.add_argument('--cache-dir', type=str, required=True, help='KD cache directory')
    parser.add_argument('--steps', type=int, default=20, help='Steps per benchmark')
    parser.add_argument('--model-id', type=str, default='Qwen/Qwen3-0.6B', help='Model ID')
    parser.add_argument('--skip-baseline', action='store_true', help='Skip baseline benchmark')
    parser.add_argument('--skip-checkpointing', action='store_true', help='Skip checkpointing benchmarks')
    args = parser.parse_args()

    assert os.path.exists(args.cache_dir), f"Cache dir not found: {args.cache_dir}"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    results = []

    # Benchmark 1: Baseline (batch=8, no checkpointing)
    if not args.skip_baseline:
        results.append(run_benchmark(
            cache_dir=args.cache_dir,
            batch_size=8,
            steps=args.steps,
            gradient_checkpointing=False,
            device=device,
            model_id=args.model_id,
        ))

    # Benchmark 2: With checkpointing (batch=8)
    if not args.skip_checkpointing:
        results.append(run_benchmark(
            cache_dir=args.cache_dir,
            batch_size=8,
            steps=args.steps,
            gradient_checkpointing=True,
            device=device,
            model_id=args.model_id,
        ))

        # Benchmark 3: With checkpointing (batch=16) - to show memory savings
        results.append(run_benchmark(
            cache_dir=args.cache_dir,
            batch_size=16,
            steps=args.steps,
            gradient_checkpointing=True,
            device=device,
            model_id=args.model_id,
        ))

    # Print results
    print_results(results)

    print("\nDone!")


if __name__ == '__main__':
    main()
