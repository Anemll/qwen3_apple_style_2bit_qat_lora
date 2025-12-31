#!/usr/bin/env python3
"""
SpeedRun Benchmark - Compare training performance with different optimizations.

Usage:
    python speedrun/benchmark.py --cache-dir caches/alpaca_chat_think_both_L64_K64_R128
    python speedrun/benchmark.py --cache-dir ... --find-max-batch  # Find max batch size

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


def get_gpu_total_memory_gb() -> float:
    """Get total GPU memory in GB."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / 1024**3
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


def print_result_header(seq_len: int):
    """Print the results table header."""
    print(f"\n{'='*88}")
    print(f"Sequence length: {seq_len}")
    print(f"{'Config':<30} {'Batch':>6} {'Step(s)':>8} {'Memory':>10} {'t/s':>10} {'Loss':>8} {'Status':>8}")
    print("-" * 88)


def print_result_row(r: BenchmarkResult):
    """Print a single result row."""
    if r.success:
        print(f"{r.name:<30} {r.batch_size:>6} {r.step_time:>8.3f} {r.peak_memory_mb:>9.0f}M {r.tokens_per_sec:>10.0f} {r.final_loss:>8.4f} {'OK':>8}")
    else:
        error_short = r.error[:30] if r.error else "Unknown"
        print(f"{r.name:<30} {r.batch_size:>6} {'---':>8} {r.peak_memory_mb:>9.0f}M {'---':>10} {'---':>8} {'FAIL':>8}")
        print(f"  Error: {error_short}...")


def create_v2_model(model_id: str = "Qwen/Qwen3-0.6B", verbose: bool = True):
    """Create a V2 quantized model (slow operation, do once)."""
    from transformers import AutoModelForCausalLM
    from qat_lora import (
        AnemllQuantConfigV2,
        replace_linear_with_anemll_v2,
    )

    if verbose:
        print("  Loading base model...", end=" ", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    if verbose:
        print(f"done ({time.time()-t0:.1f}s)")

    if verbose:
        print("  Replacing layers with V2...", end=" ", flush=True)
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
    if verbose:
        print(f"done ({time.time()-t0:.1f}s)")

    return model


def run_benchmark(
    cache_dir: str,
    batch_size: int,
    steps: int,
    gradient_checkpointing: bool,
    device: torch.device,
    model_id: str = "Qwen/Qwen3-0.6B",
    verbose: bool = True,
    v2_model_path: str = None,  # Path to saved V2 model (fast reload)
) -> BenchmarkResult:
    """Run a single benchmark configuration."""
    from qat_lora import (
        freeze_Q_all,
        train_e2e,
    )

    # Get sequence length for tokens/sec calculation
    seq_len = get_seq_len_from_cache(cache_dir)

    name = f"batch={batch_size}"
    if gradient_checkpointing:
        name += "+checkpointing"

    if verbose:
        print(f"\n{'='*60}")
        print(f"Benchmark: {name}")
        print(f"{'='*60}")

    reset_gpu_memory()

    try:
        if v2_model_path is not None:
            # Fast path: load entire model from disk
            if verbose:
                print("  Loading V2 model from cache...", end=" ", flush=True)
            t0 = time.time()
            # weights_only=False required for loading full model objects
            model = torch.load(v2_model_path, map_location='cpu', weights_only=False)
            if verbose:
                print(f"done ({time.time()-t0:.1f}s)")
        else:
            # Slow path: create V2 model from scratch
            model = create_v2_model(model_id, verbose=verbose)

        # Move to device
        if verbose:
            print("  Moving to GPU...", end=" ", flush=True)
        t0 = time.time()
        model.to(device)
        if verbose:
            print(f"done ({time.time()-t0:.1f}s)")

        # Enable gradient checkpointing if requested
        if gradient_checkpointing:
            if hasattr(model, 'gradient_checkpointing_enable'):
                # use_reentrant=False is required when inputs don't have requires_grad
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
                if verbose:
                    print("  Gradient checkpointing: ENABLED (use_reentrant=False)")
            else:
                if verbose:
                    print("  Warning: Model doesn't support gradient checkpointing")

        # Freeze Q
        if verbose:
            print("  Freezing Q...", end=" ", flush=True)
        t0 = time.time()
        freeze_Q_all(model, verbose=False)
        if verbose:
            print(f"done ({time.time()-t0:.1f}s)")

        # Run training benchmark
        if verbose:
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


def print_summary(results: list[BenchmarkResult]):
    """Print comparison summary."""
    print("-" * 88)

    successful = [r for r in results if r.success]
    if len(successful) >= 2:
        baseline = successful[0]
        for r in successful[1:]:
            speedup = r.tokens_per_sec / baseline.tokens_per_sec if baseline.tokens_per_sec > 0 else 0
            memory_change = (r.peak_memory_mb - baseline.peak_memory_mb) / baseline.peak_memory_mb * 100 if baseline.peak_memory_mb > 0 else 0
            print(f"\n{r.name} vs {baseline.name}:")
            print(f"  Throughput: {speedup:.2f}x")
            print(f"  Memory: {memory_change:+.1f}%")


def find_max_batch_size(
    cache_dir: str,
    device: torch.device,
    model_id: str,
    gradient_checkpointing: bool = True,
    steps: int = 3,
) -> int:
    """Find maximum batch size that fits in GPU memory."""
    import tempfile

    seq_len = get_seq_len_from_cache(cache_dir)
    gpu_mem_gb = get_gpu_total_memory_gb()

    print(f"\n{'='*60}")
    print(f"Finding max batch size for {gpu_mem_gb:.1f} GB GPU")
    print(f"Sequence length: {seq_len}")
    print(f"Gradient checkpointing: {'ON' if gradient_checkpointing else 'OFF'}")
    print(f"{'='*60}")

    # Create V2 model once (slow) and save to disk
    print("\n[1/2] Creating V2 model (one-time)...")
    v2_model = create_v2_model(model_id, verbose=True)

    # Save to temp file for fast reloading
    v2_model_path = tempfile.mktemp(suffix='.pt')
    print(f"  Saving to {v2_model_path}...", end=" ", flush=True)
    t0 = time.time()
    torch.save(v2_model, v2_model_path)
    print(f"done ({time.time()-t0:.1f}s)")

    del v2_model
    reset_gpu_memory()

    print("\n[2/2] Finding max batch size...")

    # Print header
    print_result_header(seq_len)

    # Start with batch=8, double until OOM
    batch_sizes = [8, 16, 32, 64, 128, 256, 512]
    max_working_batch = 0
    results = []

    try:
        for batch in batch_sizes:
            print(f"\n  Trying batch={batch}...", end=" ", flush=True)
            result = run_benchmark(
                cache_dir=cache_dir,
                batch_size=batch,
                steps=steps,
                gradient_checkpointing=gradient_checkpointing,
                device=device,
                model_id=model_id,
                verbose=False,
                v2_model_path=v2_model_path,
            )
            results.append(result)

            if result.success:
                max_working_batch = batch
                print(f"OK (mem={result.peak_memory_mb:.0f}M, t/s={result.tokens_per_sec:.0f})")
                print_result_row(result)
            else:
                print(f"OOM")
                print_result_row(result)
                break
    finally:
        # Cleanup temp file
        if os.path.exists(v2_model_path):
            os.remove(v2_model_path)

    print(f"\n{'='*60}")
    print(f"MAX BATCH SIZE: {max_working_batch}")
    if gradient_checkpointing:
        print(f"  (with gradient checkpointing)")
    else:
        print(f"  (without gradient checkpointing)")
    print(f"{'='*60}")

    return max_working_batch


def main():
    parser = argparse.ArgumentParser(description='SpeedRun Benchmark')
    parser.add_argument('--cache-dir', type=str, required=True, help='KD cache directory')
    parser.add_argument('--steps', type=int, default=20, help='Steps per benchmark')
    parser.add_argument('--model-id', type=str, default='Qwen/Qwen3-0.6B', help='Model ID')
    parser.add_argument('--skip-baseline', action='store_true', help='Skip baseline benchmark')
    parser.add_argument('--skip-checkpointing', action='store_true', help='Skip checkpointing benchmarks')
    parser.add_argument('--find-max-batch', action='store_true', help='Find maximum batch size')
    parser.add_argument('--batch-sizes', type=str, default=None,
                        help='Custom batch sizes to test (comma-separated, e.g., "8,16,32")')
    args = parser.parse_args()

    assert os.path.exists(args.cache_dir), f"Cache dir not found: {args.cache_dir}"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Find max batch mode
    if args.find_max_batch:
        find_max_batch_size(
            cache_dir=args.cache_dir,
            device=device,
            model_id=args.model_id,
            gradient_checkpointing=True,
            steps=3,  # Quick test with 3 steps
        )
        print("\nDone!")
        return

    import tempfile

    results = []
    seq_len = get_seq_len_from_cache(args.cache_dir)

    # Create V2 model once (slow) - reuse for all benchmarks
    print("\n[1/2] Creating V2 model (one-time)...")
    v2_model = create_v2_model(args.model_id, verbose=True)

    # Save to temp file for fast reloading
    v2_model_path = tempfile.mktemp(suffix='.pt')
    print(f"  Saving to {v2_model_path}...", end=" ", flush=True)
    t0 = time.time()
    torch.save(v2_model, v2_model_path)
    print(f"done ({time.time()-t0:.1f}s)")

    del v2_model
    reset_gpu_memory()

    print("\n[2/2] Running benchmarks...")

    try:
        # Print header once at the start
        print_result_header(seq_len)

        # Benchmark 1: Baseline (batch=8, no checkpointing)
        if not args.skip_baseline:
            result = run_benchmark(
                cache_dir=args.cache_dir,
                batch_size=8,
                steps=args.steps,
                gradient_checkpointing=False,
                device=device,
                model_id=args.model_id,
                v2_model_path=v2_model_path,
            )
            results.append(result)
            print_result_row(result)

        # Benchmark 2: With checkpointing (batch=8)
        if not args.skip_checkpointing:
            result = run_benchmark(
                cache_dir=args.cache_dir,
                batch_size=8,
                steps=args.steps,
                gradient_checkpointing=True,
                device=device,
                model_id=args.model_id,
                v2_model_path=v2_model_path,
            )
            results.append(result)
            print_result_row(result)

            # Benchmark 3: With checkpointing (batch=16) - to show memory savings
            result = run_benchmark(
                cache_dir=args.cache_dir,
                batch_size=16,
                steps=args.steps,
                gradient_checkpointing=True,
                device=device,
                model_id=args.model_id,
                v2_model_path=v2_model_path,
            )
            results.append(result)
            print_result_row(result)
    finally:
        # Cleanup temp file
        if os.path.exists(v2_model_path):
            os.remove(v2_model_path)

    # Print summary
    print_summary(results)

    print("\nDone!")


if __name__ == '__main__':
    main()
