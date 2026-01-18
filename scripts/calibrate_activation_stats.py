#!/usr/bin/env python3
"""
Calibrate activation statistics for activation-weighted LUT selection.

Collects Var(x_j) for each layer's input features by running a short calibration
pass on WikiText data. The variance captures which input features are frequently
active and thus more important for quantization accuracy.

The output can be used with select_best_lut_per_layer.py --activation-cache
to enable activation-weighted MSE scoring (Option A from the technical review).

Usage:
    # Collect activation variances (64 samples, ~30s on CPU)
    python scripts/calibrate_activation_stats.py \
        --model Qwen/Qwen3-0.6B \
        --output activation_stats.pt \
        --num-samples 64

    # Use with LUT selection
    python scripts/select_best_lut_per_layer.py checkpoint.pt \
        --activation-cache activation_stats.pt \
        --metric activation_mse

    # More samples for better estimates
    python scripts/calibrate_activation_stats.py \
        --model Qwen/Qwen3-0.6B \
        --output activation_stats.pt \
        --num-samples 256 \
        --seq-len 512
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict

REPO_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_DIR))

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


class ActivationCollector:
    """Hook-based collector for layer input activations."""

    def __init__(self):
        self.layer_inputs: Dict[str, list] = defaultdict(list)
        self.hooks = []

    def register_hooks(self, model: nn.Module, layer_names: list):
        """Register forward hooks on specified layers."""
        for name, module in model.named_modules():
            if name in layer_names:
                hook = module.register_forward_hook(
                    self._make_hook(name)
                )
                self.hooks.append(hook)

    def _make_hook(self, name: str):
        """Create a hook function for a specific layer."""
        def hook_fn(module, inputs, outputs):
            # inputs is a tuple, we want the first element (the input tensor)
            if len(inputs) > 0 and isinstance(inputs[0], torch.Tensor):
                # Store on CPU to save memory
                self.layer_inputs[name].append(inputs[0].detach().cpu())
        return hook_fn

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def compute_variances(self) -> Dict[str, torch.Tensor]:
        """Compute per-feature variance for each layer.

        For a linear layer with input shape [batch, seq, in_features],
        we compute variance across the batch and sequence dimensions,
        resulting in a variance vector of shape [in_features].
        """
        variances = {}

        for name, inputs_list in self.layer_inputs.items():
            if not inputs_list:
                continue

            # Concatenate all collected inputs
            # Shape: [total_tokens, in_features]
            all_inputs = torch.cat([x.view(-1, x.shape[-1]) for x in inputs_list], dim=0)

            # Compute variance per feature (column)
            var = all_inputs.var(dim=0)  # [in_features]

            variances[name] = var

        return variances


def get_linear_layer_names(model: nn.Module) -> list:
    """Get names of all Linear layers in the model."""
    names = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            names.append(name)
    return names


def parse_args():
    parser = argparse.ArgumentParser(
        description='Calibrate activation statistics for LUT selection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument('--model', default='Qwen/Qwen3-0.6B',
                        help='Model name or path')
    parser.add_argument('-o', '--output', default='activation_stats.pt',
                        help='Output path for activation stats')
    parser.add_argument('--num-samples', type=int, default=64,
                        help='Number of calibration samples (default: 64)')
    parser.add_argument('--seq-len', type=int, default=256,
                        help='Sequence length for calibration (default: 256)')
    parser.add_argument('--dataset', default='wikitext',
                        help='Calibration dataset (wikitext or c4)')
    parser.add_argument('--device', default='auto',
                        help='Device (auto, cpu, cuda, mps)')
    parser.add_argument('--dtype', default='fp32',
                        help='Model dtype (fp32, fp16, bf16)')

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("CALIBRATE ACTIVATION STATISTICS")
    print("=" * 60)
    print(f"Model:       {args.model}")
    print(f"Output:      {args.output}")
    print(f"Samples:     {args.num_samples}")
    print(f"Seq len:     {args.seq_len}")
    print(f"Dataset:     {args.dataset}")
    print()

    # Device selection
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Dtype
    dtype_map = {
        'fp32': torch.float32,
        'fp16': torch.float16,
        'bf16': torch.bfloat16,
    }
    dtype = dtype_map.get(args.dtype, torch.float32)

    # Load model
    print(f"\nLoading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Get linear layer names
    linear_names = get_linear_layer_names(model)
    print(f"Found {len(linear_names)} linear layers")

    # Load calibration data
    print(f"\nLoading calibration data: {args.dataset}")
    if args.dataset == 'wikitext':
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        text_column = 'text'
    elif args.dataset == 'c4':
        dataset = load_dataset('c4', 'en', split='train', streaming=True)
        text_column = 'text'
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Prepare calibration samples
    print(f"Preparing {args.num_samples} samples...")
    samples = []

    if args.dataset == 'c4':
        # Streaming dataset
        for i, example in enumerate(dataset):
            if i >= args.num_samples * 2:  # Get extra in case some are too short
                break
            text = example[text_column]
            if len(text) > 100:  # Skip very short texts
                samples.append(text)
    else:
        # Regular dataset
        for example in dataset:
            text = example[text_column]
            if len(text) > 100:
                samples.append(text)
            if len(samples) >= args.num_samples * 2:
                break

    samples = samples[:args.num_samples]
    print(f"Collected {len(samples)} samples")

    # Setup collector
    collector = ActivationCollector()
    collector.register_hooks(model, linear_names)

    # Run calibration
    print(f"\nRunning calibration pass...")
    with torch.no_grad():
        for i, text in enumerate(samples):
            if (i + 1) % 10 == 0:
                print(f"  Sample {i + 1}/{len(samples)}")

            # Tokenize
            inputs = tokenizer(
                text,
                return_tensors='pt',
                max_length=args.seq_len,
                truncation=True,
                padding=False,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Forward pass (collects activations via hooks)
            _ = model(**inputs)

    # Remove hooks
    collector.remove_hooks()

    # Compute variances
    print("\nComputing activation variances...")
    variances = collector.compute_variances()
    print(f"Computed variances for {len(variances)} layers")

    # Print summary stats
    print("\nVariance statistics:")
    for name in sorted(variances.keys())[:5]:  # First 5 layers
        var = variances[name]
        print(f"  {name}: shape={list(var.shape)}, mean={var.mean():.4e}, max={var.max():.4e}")
    if len(variances) > 5:
        print(f"  ... and {len(variances) - 5} more layers")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as dict with metadata
    save_data = {
        'variances': variances,
        'model': args.model,
        'num_samples': args.num_samples,
        'seq_len': args.seq_len,
        'dataset': args.dataset,
    }
    torch.save(save_data, output_path)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nSaved: {output_path} ({size_mb:.2f} MB)")
    print("\nDone!")

    return 0


if __name__ == '__main__':
    sys.exit(main())
