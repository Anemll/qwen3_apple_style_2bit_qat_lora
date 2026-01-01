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


def get_device():
    """Get best available device: TPU > CUDA > CPU."""
    # Try TPU first
    try:
        import torch_xla
        return torch_xla.device()
    except ImportError:
        pass
    except RuntimeError as e:
        # TPU busy or init failed - fall back to CUDA/CPU
        print(f"[WARN] TPU init failed: {e}")
    # Fall back to CUDA
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def is_tpu_device(device) -> bool:
    """Check if device is a TPU."""
    return 'xla' in str(device).lower()


def get_gpu_memory_mb() -> float:
    """Get current memory usage in MB (0 for TPU - not available)."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    # TPU/MPS/CPU: memory tracking not available
    return 0.0


def get_gpu_total_memory_gb() -> float:
    """Get total memory in GB."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / 1024**3
    # TPU v6e-1 has 16GB HBM - hardcoded
    return 16.0


def reset_gpu_memory():
    """Reset memory tracking."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    # TPU: just gc.collect() (no equivalent API)


def tpu_mark_step():
    """Mark step for TPU (no-op for CUDA/CPU)."""
    try:
        import torch_xla
        torch_xla.sync()
    except (ImportError, RuntimeError):
        pass


def print_device_info(device):
    """Print device info."""
    if is_tpu_device(device):
        print(f"Device: TPU")
        print(f"Memory: 16 GB HBM (estimated)")
    elif torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print(f"Device: {device}")


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


def create_v2_model(model_id: str = "Qwen/Qwen3-0.6B", verbose: bool = True,
                    dtype: torch.dtype = torch.float32):
    """Create a V2 quantized model (slow operation, do once)."""
    from transformers import AutoModelForCausalLM
    from qat_lora import (
        AnemllQuantConfigV2,
        replace_linear_with_anemll_v2,
    )

    # STE-FP16 only used for FP32 training (ANE compatibility)
    use_ste = (dtype == torch.float32)

    if verbose:
        dtype_str = 'fp32' if dtype == torch.float32 else 'bf16'
        print(f"  Loading base model ({dtype_str})...", end=" ", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,  # Note: torch_dtype is deprecated
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
        use_ste_fp16=use_ste,
    )
    attn_config = AnemllQuantConfigV2(
        lut_size=16,
        scale_rank=8,
        group_size=32,
        force_positive_scales=False,
        magnitude_activation='identity',
        use_ste_fp16=use_ste,
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
    warmup_steps: int = 10,  # Set to 0 for fast max-batch testing
    dtype: torch.dtype = torch.float32,
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
                dtype_str = 'fp32' if dtype == torch.float32 else 'bf16'
                print(f"  Loading V2 model from cache ({dtype_str})...", end=" ", flush=True)
            t0 = time.time()
            # weights_only=False required for loading full model objects
            model = torch.load(v2_model_path, map_location='cpu', weights_only=False)
            if verbose:
                print(f"done ({time.time()-t0:.1f}s)")
        else:
            # Slow path: create V2 model from scratch
            model = create_v2_model(model_id, verbose=verbose, dtype=dtype)

        # Move to device and cast to target dtype
        if verbose:
            print("  Moving to GPU...", end=" ", flush=True)
        t0 = time.time()
        model.to(device=device, dtype=dtype)
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
            warmup_steps=warmup_steps,
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


def estimate_max_batch(gpu_mem_gb: float, seq_len: int) -> int:
    """Estimate max batch size based on GPU memory and sequence length.

    Based on A100 40GB benchmarks:
    - L64 (seq=64): max=144, uses ~30GB
    - L128 (seq=128): max=72, uses ~30GB
    - Base memory (model + overhead) ≈ 7GB
    - Per-sample memory ≈ seq_len * 2.5MB
    """
    base_memory_gb = 7.0  # Model + CUDA overhead
    per_sample_mb = seq_len * 2.5  # ~160MB for seq=64, ~320MB for seq=128

    available_gb = gpu_mem_gb * 0.85 - base_memory_gb  # 15% buffer
    available_mb = available_gb * 1024

    estimated = int(available_mb / per_sample_mb)
    # Round down to multiple of 8
    estimated = (estimated // 8) * 8
    # Clamp to reasonable range
    estimated = max(8, min(512, estimated))

    return estimated


def find_max_batch_size(
    cache_dir: str,
    device: torch.device,
    model_id: str,
    gradient_checkpointing: bool = True,
    steps: int = 2,  # Minimal: just 2 steps to test memory/throughput
    warmup_steps: int = 0,  # No warmup for speed
    rebuild_model: bool = False,
    dtype: torch.dtype = torch.float32,
) -> int:
    """Find maximum batch size that fits in GPU memory using estimation + binary search."""
    from qat_lora import freeze_Q_all, train_e2e

    seq_len = get_seq_len_from_cache(cache_dir)
    gpu_mem_gb = get_gpu_total_memory_gb()

    dtype_str = 'fp32' if dtype == torch.float32 else 'bf16'
    print(f"\n{'='*60}")
    print(f"Finding max batch size for {gpu_mem_gb:.1f} GB GPU")
    print(f"Sequence length: {seq_len}, dtype: {dtype_str}")
    print(f"Gradient checkpointing: {'ON' if gradient_checkpointing else 'OFF'}")
    print(f"{'='*60}")

    # Get or create cached V2 model path
    v2_model_path = get_or_create_v2_model(model_id, rebuild=rebuild_model)

    # Load model once and keep in memory
    model_size_mb = os.path.getsize(v2_model_path) / 1024 / 1024
    print(f"\n[*] Loading model ({model_size_mb:.0f} MB)...", end=" ", flush=True)
    t0 = time.time()
    try:
        model = torch.load(v2_model_path, map_location='cpu', weights_only=False)
    except RuntimeError as e:
        if "failed finding central directory" in str(e) or "PytorchStreamReader failed" in str(e):
            print(f"CORRUPTED!")
            print(f"  Deleting corrupted cache: {v2_model_path}")
            os.remove(v2_model_path)
            print(f"  Please re-run to reload from GDrive or recreate model.")
            raise RuntimeError(f"Corrupted cache deleted. Please re-run.")
        raise
    reset_gpu_memory()
    model.to(device=device, dtype=dtype)
    if gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    freeze_Q_all(model, verbose=False)
    print(f"done ({time.time()-t0:.1f}s)")

    # Estimate after loading - now we know actual model memory usage
    model_mem_mb = get_gpu_memory_mb()
    available_mb = (gpu_mem_gb * 1024 * 0.95) - model_mem_mb  # 5% buffer
    per_sample_mb = seq_len * 2.5  # ~160MB for seq=64
    estimated = int(available_mb / per_sample_mb)
    estimated = (estimated // 8) * 8  # Round to multiple of 8
    estimated = max(8, min(512, estimated))
    print(f"  Model memory: {model_mem_mb:.0f} MB")
    print(f"  Available for batches: {available_mb:.0f} MB")
    print(f"  Estimated max batch: {estimated}")

    print("\n[*] Testing batch sizes...")

    # Print header
    print_result_header(seq_len)

    # Helper to test a batch size (reuses model, only reloads after OOM)
    def test_batch(batch: int, current_model) -> tuple:
        """Test batch size. Returns (success, result, model).
        Model is None after OOM (needs reload)."""
        print(f"\n  batch={batch}: ", end="", flush=True)
        t0 = time.time()

        try:
            reset_gpu_memory()
            result = train_e2e(
                model=current_model,
                cache_dir=cache_dir,
                device=device,
                max_steps=steps,
                batch_size=batch,
                lr=5e-5,
                use_cosine_schedule=True,
                warmup_steps=warmup_steps,
                temperature=2.0,
                train_weights=False,
                train_scales=True,
                logging_steps=steps + 1,
                eval_steps=steps + 1,
                verbose=False,
                use_fp16=False,
            )
            elapsed = time.time() - t0
            peak_memory = get_gpu_memory_mb()
            tokens_per_sec = (batch * seq_len * steps) / elapsed
            final_loss = result.get('final_loss', 0.0)

            bench_result = BenchmarkResult(
                name=f"batch={batch}+checkpointing",
                batch_size=batch,
                seq_len=seq_len,
                steps=steps,
                total_time=elapsed,
                peak_memory_mb=peak_memory,
                step_time=elapsed / steps,
                tokens_per_sec=tokens_per_sec,
                final_loss=final_loss,
                success=True,
            )
            print(f"OK ({elapsed:.1f}s, mem={peak_memory:.0f}M, t/s={tokens_per_sec:.0f})")
            print_result_row(bench_result)
            return True, bench_result, current_model

        except Exception as e:
            elapsed = time.time() - t0
            bench_result = BenchmarkResult(
                name=f"batch={batch}+checkpointing",
                batch_size=batch,
                seq_len=seq_len,
                steps=steps,
                total_time=0,
                peak_memory_mb=get_gpu_memory_mb(),
                step_time=0,
                tokens_per_sec=0,
                final_loss=0,
                success=False,
                error=str(e)[:50],
            )
            print(f"OOM ({elapsed:.1f}s)")
            print_result_row(bench_result)
            # Model state corrupted after OOM, need to reload
            del current_model
            reset_gpu_memory()
            return False, bench_result, None

    # Helper to reload model after OOM
    def reload_model():
        print("  reloading model...", end=" ", flush=True)
        t0 = time.time()
        m = torch.load(v2_model_path, map_location='cpu', weights_only=False)
        m.to(device=device, dtype=dtype)
        if gradient_checkpointing and hasattr(m, 'gradient_checkpointing_enable'):
            m.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        freeze_Q_all(m, verbose=False)
        print(f"done ({time.time()-t0:.1f}s)")
        return m

    # Test estimated batch size
    success, _, model = test_batch(estimated, model)

    if success:
        # Estimate worked - search upward
        low = estimated
        high = min(512, int(estimated * 1.5))
        high = (high // 8) * 8
    else:
        # Estimate too high - search downward, reload model
        model = reload_model()
        low = 8
        high = estimated

    max_working = low if success else 0

    # Binary search for exact max
    print(f"\n[*] Binary search in range [{low}, {high}]...")

    while high - low > 8:
        mid = (low + high) // 2
        mid = (mid // 8) * 8  # Round to multiple of 8
        if mid == low:
            mid = low + 8
        if mid >= high:
            break

        if model is None:
            model = reload_model()

        success, _, model = test_batch(mid, model)

        if success:
            low = mid
            max_working = mid
        else:
            high = mid

    # Cleanup
    if model is not None:
        del model
    reset_gpu_memory()

    # Final result
    print(f"\n{'='*60}")
    print(f"MAX BATCH SIZE: {max_working}")
    if gradient_checkpointing:
        print(f"  (with gradient checkpointing)")
    else:
        print(f"  (without gradient checkpointing)")
    print(f"{'='*60}")

    return max_working


V2_MODEL_CACHE = "runs/speedrun/v2_benchmark_model.pt"


def get_or_create_v2_model(model_id: str, rebuild: bool = False,
                           load_from: str = None, save_to: str = None,
                           dtype: torch.dtype = torch.float32) -> str:
    """Get cached V2 model path, creating if needed.

    Args:
        model_id: HuggingFace model ID
        rebuild: Force rebuild even if cache exists
        load_from: Load pre-saved model from this path (e.g., GDrive)
        save_to: Save model to this path after creation (e.g., GDrive)
        dtype: Model dtype (fp32 or bf16)
    """
    # Priority 1: Use local cache if exists
    os.makedirs(os.path.dirname(V2_MODEL_CACHE), exist_ok=True)
    if os.path.exists(V2_MODEL_CACHE) and not rebuild:
        print(f"[*] Using cached V2 model: {V2_MODEL_CACHE}")
        return V2_MODEL_CACHE

    # Priority 2: If save_to path exists, use it as load source (avoid recreating)
    if save_to and os.path.exists(save_to) and not rebuild:
        load_from = save_to  # Use existing file instead of creating new

    # Priority 3: Load from specified path (e.g., GDrive) and copy to local
    if load_from and os.path.exists(load_from) and not rebuild:
        print(f"[*] Loading V2 model from: {load_from}")
        # Copy to local cache using rsync (verifies integrity, resumes partial)
        if load_from != V2_MODEL_CACHE:
            import subprocess
            print(f"  Copying to local cache (rsync)...", end=" ", flush=True)
            t0 = time.time()
            result = subprocess.run(
                ["rsync", "-ah", "--checksum", load_from, V2_MODEL_CACHE],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                print(f"FAILED: {result.stderr}")
                raise RuntimeError(f"rsync failed: {result.stderr}")
            print(f"done ({time.time()-t0:.1f}s)")
        return V2_MODEL_CACHE

    # Priority 4: Create new model
    print(f"\n[*] Creating V2 model (one-time)...")
    v2_model = create_v2_model(model_id, dtype=dtype, verbose=True)

    print(f"  Saving to {V2_MODEL_CACHE}...", end=" ", flush=True)
    t0 = time.time()
    torch.save(v2_model, V2_MODEL_CACHE)
    print(f"done ({time.time()-t0:.1f}s)")

    # Also save to GDrive if specified
    if save_to and save_to != V2_MODEL_CACHE:
        os.makedirs(os.path.dirname(save_to), exist_ok=True)
        print(f"  Saving to {save_to}...", end=" ", flush=True)
        t0 = time.time()
        torch.save(v2_model, save_to)
        print(f"done ({time.time()-t0:.1f}s)")

    del v2_model
    reset_gpu_memory()

    return V2_MODEL_CACHE


def main():
    parser = argparse.ArgumentParser(description='SpeedRun Benchmark')
    parser.add_argument('--cache-dir', type=str, required=True, help='KD cache directory')
    parser.add_argument('--steps', type=int, default=20, help='Steps per benchmark')
    parser.add_argument('--model-id', type=str, default='Qwen/Qwen3-0.6B', help='Model ID')
    parser.add_argument('--dtype', type=str, default='fp32', choices=['fp32', 'bf16'],
                        help='Training dtype: fp32 (default, ANE-safe), bf16 (2x faster)')
    parser.add_argument('--skip-baseline', action='store_true', help='Skip baseline benchmark')
    parser.add_argument('--skip-checkpointing', action='store_true', help='Skip checkpointing benchmarks')
    parser.add_argument('--find-max-batch', action='store_true', help='Find maximum batch size')
    parser.add_argument('--batch-sizes', type=str, default=None,
                        help='Custom batch sizes to test (comma-separated, e.g., "8,16,32")')
    parser.add_argument('--rebuild-model', action='store_true',
                        help='Rebuild V2 model cache even if it exists')
    parser.add_argument('--load-model', type=str, default=None,
                        help='Load pre-saved V2 model from path (e.g., GDrive)')
    parser.add_argument('--save-model', type=str, default=None,
                        help='Save V2 model to path after creation (e.g., GDrive)')
    args = parser.parse_args()

    # Dtype mapping
    dtype_map = {'fp32': torch.float32, 'bf16': torch.bfloat16}
    train_dtype = dtype_map[args.dtype]

    assert os.path.exists(args.cache_dir), f"Cache dir not found: {args.cache_dir}"

    device = get_device()
    print_device_info(device)

    # Get or create cached V2 model (reused across all benchmarks)
    v2_model_path = get_or_create_v2_model(
        args.model_id,
        rebuild=args.rebuild_model,
        load_from=args.load_model,
        save_to=args.save_model,
        dtype=train_dtype,
    )

    # Find max batch mode (minimal: 2 steps, no warmup)
    if args.find_max_batch:
        find_max_batch_size(
            cache_dir=args.cache_dir,
            device=device,
            model_id=args.model_id,
            gradient_checkpointing=True,
            # Uses defaults: steps=2, warmup_steps=0 for speed
            rebuild_model=False,  # Already created above
            dtype=train_dtype,
        )
        print("\nDone!")
        return

    results = []
    seq_len = get_seq_len_from_cache(args.cache_dir)

    print("\n[*] Running benchmarks...")

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
            dtype=train_dtype,
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
            dtype=train_dtype,
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
            dtype=train_dtype,
        )
        results.append(result)
        print_result_row(result)

    # Print summary
    print_summary(results)

    print("\nDone!")


if __name__ == '__main__':
    main()
