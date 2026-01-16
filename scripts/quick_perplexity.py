#!/usr/bin/env python3
"""
Quick perplexity estimator for AQ1 V2 models.

Pre-caches tokenized chunks for fast repeated evaluation during training.
Uses fixed-shape [1, L] tensors for TPU/MPS/CUDA compatibility.

Usage:
    # Quick estimate of V2 checkpoint
    python scripts/quick_perplexity.py checkpoint.pt

    # Baseline model
    python scripts/quick_perplexity.py --baseline

    # More chunks for better accuracy
    python scripts/quick_perplexity.py checkpoint.pt --num-chunks 50

    # Compare quick vs full PPL
    python scripts/quick_perplexity.py checkpoint.pt --compare

As a function (for training loops):
    from scripts.quick_perplexity import QuickPerplexityEstimator

    # Create once at start of training (caches chunks)
    estimator = QuickPerplexityEstimator(tokenizer, num_chunks=30, chunk_size=1024)

    # Call repeatedly during training (fast!)
    ppl = estimator.estimate_ppl(model)  # ~15-20s
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Optional, Dict

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


class QuickPerplexityEstimator:
    """
    Fast perplexity estimator with pre-cached fixed-shape chunks.

    Pre-computes evaluation chunks as a single [N, L] tensor.
    Processes chunks sequentially (one at a time) with fixed shapes for
    TPU/MPS/CUDA compatibility.

    Usage:
        # Create once at start of training (caches chunks)
        estimator = QuickPerplexityEstimator(tokenizer, num_chunks=30, chunk_size=1024)

        # Call repeatedly during training (fast!)
        ppl = estimator.estimate_ppl(model)  # ~15-20s on MPS
    """

    _instance: Optional['QuickPerplexityEstimator'] = None  # Singleton

    def __init__(
        self,
        tokenizer,
        num_chunks: int = 14,
        chunk_size: int = 1024,
        stride: int = 256,
        max_tokens: Optional[int] = None,
        text: Optional[str] = None,
        verbose: bool = True,
    ):
        """
        Initialize and cache fixed-shape OVERLAPPING evaluation chunks.

        Uses sliding window like full perplexity but with pre-cached chunks.
        Only counts non-overlapping tokens to avoid double-counting.

        Args:
            tokenizer: HuggingFace tokenizer
            num_chunks: Number of chunks to cache (default: 30)
            chunk_size: Fixed length for all chunks (default: 1024)
            stride: Overlap stride (default: chunk_size // 2)
            max_tokens: Limit dataset tokens (default: auto-calculate from chunks)
            text: Custom text to tokenize (default: WikiText-2 test)
            verbose: Print cache info
        """
        self.tokenizer = tokenizer
        self.num_chunks = num_chunks
        self.chunk_size = chunk_size
        self.stride = stride
        self.verbose = verbose

        # Auto-calculate max_tokens if not specified
        # Need: (num_chunks - 1) * stride + chunk_size tokens
        min_required = (num_chunks - 1) * self.stride + chunk_size
        self.max_tokens = max_tokens or min_required

        # Cache: [N, L] tensor + target lengths per chunk
        self._chunks: Optional[torch.Tensor] = None
        self._trg_lens: Optional[list] = None  # How many tokens to count per chunk
        self._prepare_cache(text)

    def _prepare_cache(self, text: Optional[str] = None):
        """Tokenize and cache fixed-shape OVERLAPPING evaluation chunks."""
        start = time.time()

        # Load WikiText-2 if no custom text
        if text is None:
            try:
                from datasets import load_dataset
                dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
                text = '\n\n'.join(dataset['text'])
            except ImportError:
                raise RuntimeError("Install datasets: pip install datasets")

        # Tokenize full text
        input_ids = self.tokenizer(text, return_tensors='pt').input_ids[0]

        # Limit tokens if specified
        if self.max_tokens is not None and len(input_ids) > self.max_tokens:
            input_ids = input_ids[:self.max_tokens]

        seq_len = len(input_ids)

        # Calculate required tokens for overlapping chunks
        # Need: (num_chunks - 1) * stride + chunk_size
        required_tokens = (self.num_chunks - 1) * self.stride + self.chunk_size
        if seq_len < required_tokens:
            # Reduce num_chunks to fit available data
            max_possible = (seq_len - self.chunk_size) // self.stride + 1
            self.num_chunks = max(1, max_possible)
            if self.verbose:
                print(f"[QuickPPL] Reduced to {self.num_chunks} chunks (need {required_tokens} tokens, have {seq_len})")

        # Create overlapping chunks [N, L] with sliding window
        chunks = []
        trg_lens = []  # How many NEW tokens to count per chunk

        for i in range(self.num_chunks):
            start_pos = i * self.stride
            end_pos = start_pos + self.chunk_size

            if end_pos > seq_len:
                # Last chunk may need padding or truncation
                break

            chunk = input_ids[start_pos:end_pos]
            chunks.append(chunk)

            # First chunk: count all predictions (chunk_size - 1)
            # Later chunks: only count NEW tokens (stride)
            if i == 0:
                trg_lens.append(self.chunk_size - 1)
            else:
                trg_lens.append(self.stride)

        self.num_chunks = len(chunks)
        self._chunks = torch.stack(chunks, dim=0)  # [num_chunks, chunk_size]
        self._trg_lens = trg_lens

        elapsed = time.time() - start
        total_counted = sum(trg_lens)
        if self.verbose:
            print(f"[QuickPPL] Cached {self.num_chunks} overlapping chunks "
                  f"(L={self.chunk_size}, stride={self.stride})")
            print(f"[QuickPPL] Will evaluate {total_counted:,} tokens in {elapsed:.1f}s")

    @torch.no_grad()
    def evaluate(
        self,
        model: torch.nn.Module,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        verbose: bool = False,
    ) -> Dict:
        """
        Evaluate perplexity using cached fixed-shape chunks.

        Processes chunks sequentially (one at a time) with fixed [1, L] shape
        for TPU/MPS/CUDA compatibility.

        Args:
            model: AQ1 V2 model to evaluate
            device: Device for eval (default: model's device)
            dtype: Dtype for autocast (default: model's dtype)
            verbose: Print per-chunk stats

        Returns:
            dict with: perplexity, cross_entropy, tokens, time, chunks_evaluated
        """
        model.eval()

        # Auto-detect device/dtype from model
        if device is None:
            device = next(model.parameters()).device
        if dtype is None:
            dtype = next(model.parameters()).dtype

        # Check hardware type
        device_str = str(device)
        is_tpu = 'xla' in device_str
        is_mps = device_str == 'mps'

        if is_tpu:
            import torch_xla.core.xla_model as xm

        # Move all chunks to device once
        chunks_on_device = self._chunks.to(device)  # [N, L]

        nlls = []
        total_tokens = 0
        start_time = time.time()

        if is_mps:
            print("  [MPS] First chunk may be slow (graph compilation)...", flush=True)

        for chunk_idx in range(self.num_chunks):
            # Get single chunk [1, L] - fixed shape for all iterations
            chunk = chunks_on_device[chunk_idx:chunk_idx+1]  # [1, L]

            # Show minimal progress even without verbose
            progress_step = max(1, round(self.num_chunks / 5))
            if chunk_idx == 0:
                print(f"  Processing chunk 1/{self.num_chunks}...", end='', flush=True)
            elif chunk_idx == self.num_chunks - 1:
                print(f" {self.num_chunks}/{self.num_chunks}", flush=True)
            elif chunk_idx % progress_step == 0:
                print(f" {chunk_idx}", end='', flush=True)

            # Forward pass
            if is_tpu:
                outputs = model(chunk)
            elif is_mps or dtype == torch.float32:
                # MPS doesn't support autocast well
                outputs = model(chunk)
            else:
                # CUDA with autocast
                with torch.autocast(device_type='cuda', dtype=dtype):
                    outputs = model(chunk)

            logits = outputs.logits.float()  # [1, L, V]

            # Get target length for this chunk (avoid double-counting overlaps)
            trg_len = self._trg_lens[chunk_idx]

            # Compute cross-entropy loss only on TARGET tokens
            # For chunk 0: all L-1 predictions
            # For later chunks: only last `stride` predictions (new tokens)
            shift_logits = logits[:, -trg_len-1:-1, :].contiguous()  # [1, trg_len, V]
            shift_labels = chunk[:, -trg_len:].contiguous()  # [1, trg_len]

            # Flatten and compute loss
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction='sum'
            )

            chunk_tokens = trg_len

            # TPU: mark_step before .item()
            if is_tpu:
                xm.mark_step()

            nlls.append(loss.item())
            total_tokens += chunk_tokens

            if verbose:
                chunk_ppl = math.exp(loss.item() / chunk_tokens)
                elapsed_so_far = time.time() - start_time
                tps = total_tokens / elapsed_so_far if elapsed_so_far > 0 else 0
                print(f"  Chunk {chunk_idx + 1}/{self.num_chunks}: "
                      f"ppl={chunk_ppl:.2f}, {tps:.0f} tok/s")

        elapsed = time.time() - start_time
        print(f"  Eval time: {elapsed:.1f}s", flush=True)

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
            'chunks_evaluated': self.num_chunks,
            'chunk_size': self.chunk_size,
            'is_estimate': True,
        }

    def estimate_ppl(self, model: torch.nn.Module, **kwargs) -> float:
        """Convenience method - just return PPL value."""
        return self.evaluate(model, **kwargs)['perplexity']

    @classmethod
    def get_instance(
        cls,
        tokenizer,
        num_chunks: int = 30,
        chunk_size: int = 1024,
        **kwargs
    ) -> 'QuickPerplexityEstimator':
        """
        Get or create singleton instance.

        Reuses existing instance if tokenizer and params match.
        """
        if (cls._instance is None or
            cls._instance.tokenizer != tokenizer or
            cls._instance.num_chunks != num_chunks or
            cls._instance.chunk_size != chunk_size):
            cls._instance = cls(tokenizer, num_chunks, chunk_size, **kwargs)
        return cls._instance


# Module-level convenience function
def quick_perplexity(
    model: torch.nn.Module,
    tokenizer,
    num_chunks: int = 30,
    chunk_size: int = 1024,
    **kwargs
) -> Dict:
    """
    Quick perplexity estimate using singleton estimator.

    First call caches chunks (~2s). Subsequent calls are fast (~15-20s).
    Works on TPU, MPS, and CUDA with fixed [1, L] tensor shape.
    """
    estimator = QuickPerplexityEstimator.get_instance(
        tokenizer, num_chunks, chunk_size, **kwargs
    )
    return estimator.evaluate(model, **kwargs)


def get_device(device_arg: str = 'auto', dtype_arg: str = 'auto'):
    """Get device and dtype based on availability and user preference."""
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
        except (ImportError, Exception):
            pass
        if device is None:
            if torch.backends.mps.is_available():
                device = torch.device('mps')
                default_dtype = torch.float16  # fp16 default for eval
            elif torch.cuda.is_available():
                device = torch.device('cuda')
                default_dtype = torch.float16  # fp16 default for eval
            else:
                device = torch.device('cpu')
                default_dtype = torch.float32
    elif device_arg == 'mps':
        device = torch.device('mps')
        default_dtype = torch.float16  # fp16 default for eval
    elif device_arg == 'cuda':
        device = torch.device('cuda')
        default_dtype = torch.float16  # fp16 default for eval
    else:
        device = torch.device('cpu')
        default_dtype = torch.float32

    # Apply dtype override
    dtype_map = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}
    if dtype_arg != 'auto' and dtype_arg in dtype_map:
        dtype = dtype_map[dtype_arg]
    else:
        dtype = default_dtype

    return device, dtype


def load_v2_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    dtype: torch.dtype,
    model_name: str = "Qwen/Qwen3-0.6B",
    config_path: Optional[str] = None,
):
    """Load V2 QAT checkpoint."""
    checkpoint_path = Path(checkpoint_path)

    # Find checkpoint file
    if checkpoint_path.is_dir():
        ckpt_file = checkpoint_path / 'model_state_dict.pt'
        if not ckpt_file.exists():
            pt_files = list(checkpoint_path.glob('*.pt'))
            if pt_files:
                ckpt_file = pt_files[0]
            else:
                raise FileNotFoundError(f"No .pt files in {checkpoint_path}")
        config_dir = checkpoint_path
    else:
        ckpt_file = checkpoint_path
        config_dir = checkpoint_path.parent

    # Load config
    config = {}
    for config_name in ['config.json', 'v2_config.json']:
        cfg_path = config_dir / config_name
        if cfg_path.exists():
            with open(cfg_path) as f:
                config = json.load(f)
            break

    if config_path:
        with open(config_path) as f:
            config = json.load(f)

    # Get quantization params
    lut_bits = config.get('lut_bits') or config.get('mlp_lut_bits') or 4
    attn_lut_bits = config.get('attn_lut_bits') or lut_bits
    scale_rank = config.get('scale_rank') or config.get('mlp_scale_rank') or 32
    attn_scale_rank = config.get('attn_scale_rank') or scale_rank

    print(f"Config: MLP Q{lut_bits}/r{scale_rank}, Attn Q{attn_lut_bits}/r{attn_scale_rank}")

    # Load base model
    print(f"Loading base model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype if dtype != torch.float32 else torch.float32,
        trust_remote_code=True,
    )

    # Import QAT modules
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from qat_lora import AnemllQuantConfigV2, replace_linear_with_anemll_v2
    from qat_lora.ane_qat_linear_v2 import AnemllQATLinearV2

    # Convert to V2 QAT
    lut_size = 2 ** lut_bits
    attn_lut_size = 2 ** attn_lut_bits

    mlp_config = AnemllQuantConfigV2(
        lut_size=lut_size,
        scale_rank=scale_rank,
        force_positive_scales=False,
        magnitude_activation='identity',
    )
    attn_config = AnemllQuantConfigV2(
        lut_size=attn_lut_size,
        scale_rank=attn_scale_rank,
        force_positive_scales=False,
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

    # Load checkpoint
    print(f"Loading checkpoint: {ckpt_file}")
    state_dict = torch.load(ckpt_file, map_location='cpu', weights_only=False)
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']

    model.load_state_dict(state_dict, strict=False)

    # Handle _Q buffers
    for name, module in model.named_modules():
        if isinstance(module, AnemllQATLinearV2):
            q_key = f"{name}._Q"
            if q_key in state_dict and module._Q is None:
                module.register_buffer('_Q', state_dict[q_key])

    # Handle LoRA if present
    lora_r = config.get('lora_r') or config.get('recovery_r') or 0
    lora_keys = [k for k in state_dict if 'lora_' in k]
    if lora_keys and lora_r > 0:
        from qat_lora.ane_qat_linear_v2 import enable_recovery_lora_all
        lora_mlp_only = config.get('lora_mlp_only') or config.get('mlp_only') or False
        print(f"Enabling LoRA (r={lora_r}, mlp_only={lora_mlp_only})...")
        enable_recovery_lora_all(model, r=lora_r, mlp_only=lora_mlp_only, skip_k_proj=True, verbose=False)
        model.load_state_dict({k: state_dict[k] for k in lora_keys}, strict=False)

    model = model.to(device)
    model.eval()

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
        description='Quick perplexity estimate for AQ1 V2 models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('checkpoint', nargs='?', default=None,
                        help='V2 checkpoint path (.pt file or directory)')
    parser.add_argument('--num-chunks', type=int, default=14,
                        help='Number of evaluation chunks (default: 14)')
    parser.add_argument('--chunk-size', type=int, default=1024,
                        help='Fixed chunk size in tokens (default: 1024)')
    parser.add_argument('--stride', type=int, default=256,
                        help='Stride between chunks (default: 256)')
    parser.add_argument('--max-tokens', type=int, default=None,
                        help='Limit dataset tokens (default: auto from chunks)')
    parser.add_argument('--baseline', action='store_true',
                        help='Evaluate baseline HuggingFace model (uses --model)')
    parser.add_argument('--hf', type=str, default=None,
                        help='Evaluate any HuggingFace model directly (e.g., --hf Qwen/Qwen3-0.6B)')
    parser.add_argument('--model', type=str, default='Qwen/Qwen3-0.6B',
                        help='Base model name for V2 checkpoint (default: Qwen/Qwen3-0.6B)')
    parser.add_argument('--device', choices=['auto', 'mps', 'cuda', 'cpu', 'tpu'], default='auto',
                        help='Device to use (default: auto)')
    parser.add_argument('--dtype', choices=['auto', 'fp16', 'bf16', 'fp32'], default='auto',
                        help='Model dtype (default: auto)')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config.json (auto-detect from checkpoint)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print per-chunk stats')
    parser.add_argument('--compare', action='store_true',
                        help='Also run full PPL from measure_perplexity.py')
    parser.add_argument('-b', '--base-dir', metavar='FOLDER',
                        help='Base folder for checkpoint path')

    args = parser.parse_args()

    # Apply base directory if specified
    if args.base_dir and args.checkpoint:
        args.checkpoint = str(Path(args.base_dir) / args.checkpoint)

    # Validate args
    if not args.baseline and not args.hf and args.checkpoint is None:
        parser.error("checkpoint is required unless --baseline or --hf is used")

    # Determine model name for tokenizer
    if args.hf:
        tokenizer_model = args.hf
    else:
        tokenizer_model = args.model

    # Get device and dtype
    device, dtype = get_device(args.device, args.dtype)
    dtype_name = {torch.float16: 'fp16', torch.bfloat16: 'bf16', torch.float32: 'fp32'}.get(dtype, str(dtype))

    # Print header
    print()
    print("=" * 60)
    print("QUICK PERPLEXITY ESTIMATE")
    print("=" * 60)
    if args.hf:
        print(f"Model:      {args.hf} (HuggingFace)")
    elif args.baseline:
        print(f"Model:      {args.model} (baseline)")
    else:
        print(f"Checkpoint: {args.checkpoint}")
    print(f"Device:     {device}")
    print(f"Dtype:      {dtype_name}")
    stride = args.stride or args.chunk_size // 2
    print(f"Chunks:     {args.num_chunks} x {args.chunk_size} (stride={stride})")
    print()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, trust_remote_code=True)

    # Create estimator (caches chunks)
    print("Caching evaluation chunks...")
    estimator = QuickPerplexityEstimator(
        tokenizer,
        num_chunks=args.num_chunks,
        chunk_size=args.chunk_size,
        stride=args.stride,
        max_tokens=args.max_tokens,
    )

    # Load model
    print()
    if args.hf:
        model = load_baseline_model(args.hf, device, dtype)
    elif args.baseline:
        model = load_baseline_model(args.model, device, dtype)
    else:
        model = load_v2_checkpoint(
            args.checkpoint,
            device=device,
            dtype=dtype,
            model_name=args.model,
            config_path=args.config,
        )

    # Evaluate
    print("\nEvaluating...")
    result = estimator.evaluate(model, verbose=args.verbose)

    # ANSI colors
    RED = "\033[91m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    # Print results
    print()
    print("=" * 60)
    print("RESULTS (ESTIMATE)")
    print("=" * 60)
    print(f"{RED}{BOLD}Quick PPL:      {result['perplexity']:.2f}{RESET}")
    print(f"Cross-entropy:  {result['cross_entropy']:.4f} nats")
    print(f"Chunks:         {result['chunks_evaluated']} x {result['chunk_size']}")
    print(f"Tokens:         {result['tokens']:,}")
    print(f"Time:           {result['time']:.1f}s ({result['tokens_per_sec']:.0f} tok/s)")
    print("-" * 60)
    print(f"Note: Quick estimate ({result['chunks_evaluated']} overlapping chunks, stride={estimator.stride})")
    print("      For full PPL: python scripts/measure_perplexity.py")
    print("=" * 60)

    # Optional: compare with full PPL
    if args.compare:
        print("\n" + "=" * 60)
        print("COMPARING WITH FULL PERPLEXITY")
        print("=" * 60)

        # Import and run full perplexity
        from measure_perplexity import compute_perplexity, load_wikitext2

        print("Loading WikiText-2...")
        input_ids = load_wikitext2(tokenizer, split='test')

        print("Computing full perplexity (sliding window)...")
        full_result = compute_perplexity(
            model=model,
            input_ids=input_ids,
            device=device,
            dtype=dtype,
            stride=512,
            max_length=1024,
            verbose=args.verbose,
        )

        print()
        print("-" * 60)
        print(f"Full PPL:   {full_result['perplexity']:.2f}")
        print(f"Quick PPL:  {result['perplexity']:.2f}")
        delta = result['perplexity'] - full_result['perplexity']
        pct = 100 * delta / full_result['perplexity']
        print(f"Delta:      {delta:+.2f} ({pct:+.1f}%)")
        print(f"Time saved: {full_result['time'] - result['time']:.1f}s "
              f"({100 * (1 - result['time'] / full_result['time']):.0f}% faster)")
        print("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
