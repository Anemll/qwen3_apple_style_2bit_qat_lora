#!/usr/bin/env python3
"""
Test script for AnemllQATLinear with groupwise LUT quantization.

Tests the Anemll-style fake quantization on Qwen3 models with optional
KD cache evaluation.

Examples:
  # Basic test with 4-bit LUT
  python scripts/test_anemll_qat.py --lut-size 16 --scale-rank 4 --group-size 32

  # With attention quantization
  python scripts/test_anemll_qat.py --lut-size 16 --scale-rank 4 --quantize-attn \
      --attn-scale-rank 8 --attn-lut-size 16 --attn-group-size 32

  # With KD cache evaluation
  python scripts/test_anemll_qat.py --lut-size 16 --scale-rank 4 --group-size 32 \
      --kd-cache-dir caches/alpaca_chat_think_both_L128_K32_R256 --eval-samples 100

  # Save quantized model
  python scripts/test_anemll_qat.py --lut-size 16 --scale-rank 4 --group-size 32 \
      --save-quantized-dir /tmp/anemll_q4
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

# Add parent to path for local imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qat_lora.ane_qat_linear import AnemllQATLinear, AnemllQuantConfig


def _pick_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _pick_dtype(dtype: str) -> torch.dtype:
    dtype = dtype.lower()
    if dtype in {"fp32", "float32"}:
        return torch.float32
    if dtype in {"fp16", "float16"}:
        return torch.float16
    if dtype in {"bf16", "bfloat16"}:
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype}")


def replace_linear_with_anemll(
    model: nn.Module,
    config: AnemllQuantConfig,
    name_regex: str,
    attn_config: Optional[AnemllQuantConfig] = None,
    attn_regex: str = r"\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$",
    verbose: bool = True,
) -> int:
    """
    Replace matching nn.Linear modules with AnemllQATLinear.

    Returns number of replaced modules.
    """
    pattern = re.compile(name_regex)
    attn_pattern = re.compile(attn_regex) if attn_config else None

    replaced = 0
    replacements: List[Tuple[nn.Module, str, AnemllQATLinear]] = []

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if isinstance(module, AnemllQATLinear):
            continue

        # Check if this is an attention layer
        is_attn = attn_pattern and attn_pattern.search(name)

        # Check if matches our target pattern
        if is_attn and attn_config:
            cfg = attn_config
        elif pattern.search(name):
            cfg = config
        else:
            continue

        # Create replacement
        new_module = AnemllQATLinear.from_linear(module, config=cfg)

        # Find parent and attribute name
        parts = name.rsplit(".", 1)
        if len(parts) == 2:
            parent_name, attr = parts
            parent = dict(model.named_modules())[parent_name]
        else:
            parent = model
            attr = name

        replacements.append((parent, attr, new_module))
        replaced += 1

        if verbose:
            rank_info = f"rank={cfg.scale_rank}" if cfg.scale_rank > 0 else "full_scales"
            print(f"  [replace] {name}: lut={cfg.lut_size}, groups={new_module.num_groups}, {rank_info}")

    # Apply replacements
    for parent, attr, new_module in replacements:
        setattr(parent, attr, new_module)

    return replaced


def compute_reconstruction_error(
    original_weight: torch.Tensor,
    quantized_weight: torch.Tensor,
) -> Dict[str, float]:
    """Compute reconstruction error metrics."""
    err = (original_weight.float() - quantized_weight.float())
    mae = err.abs().mean().item()
    mse = (err * err).mean().item()
    rmse = math.sqrt(mse)
    weight_l2 = original_weight.float().norm().clamp(min=1e-8).item()
    err_l2 = err.norm().item()
    rel_l2 = err_l2 / weight_l2
    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "rel_l2": rel_l2,
        "err_l2": err_l2,
        "weight_l2": weight_l2,
    }


def evaluate_quantization_error(model: nn.Module, verbose: bool = True) -> Dict[str, float]:
    """Evaluate quantization reconstruction error for all AnemllQATLinear layers."""
    total_mae = 0.0
    total_mse = 0.0
    total_elems = 0
    total_err_l2_sq = 0.0
    total_weight_l2_sq = 0.0

    for name, module in model.named_modules():
        if not isinstance(module, AnemllQATLinear):
            continue
        if not module.enable_fake_quant:
            continue

        with torch.no_grad():
            original = module.weight.clone()
            quantized = module.fake_quant_weight()

        stats = compute_reconstruction_error(original, quantized)
        elems = original.numel()

        total_mae += stats["mae"] * elems
        total_mse += stats["mse"] * elems
        total_elems += elems
        total_err_l2_sq += stats["err_l2"] ** 2
        total_weight_l2_sq += stats["weight_l2"] ** 2

        if verbose:
            print(f"  {name}: mae={stats['mae']:.6g} rmse={stats['rmse']:.6g} rel_l2={stats['rel_l2']:.6g}")

    if total_elems == 0:
        return {"mae": 0, "rmse": 0, "rel_l2": 0}

    overall_mae = total_mae / total_elems
    overall_rmse = math.sqrt(total_mse / total_elems)
    overall_rel_l2 = math.sqrt(total_err_l2_sq) / max(1e-12, math.sqrt(total_weight_l2_sq))

    return {
        "mae": overall_mae,
        "rmse": overall_rmse,
        "rel_l2": overall_rel_l2,
        "total_params": total_elems,
    }


def load_kd_cache(cache_dir: str, limit: int = 0) -> Optional[Dict]:
    """Load KD cache for evaluation."""
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        print(f"Warning: KD cache dir not found: {cache_dir}")
        return None

    # Look for cache files
    cache_files = list(cache_path.glob("*.pt")) + list(cache_path.glob("*.safetensors"))
    if not cache_files:
        print(f"Warning: No cache files found in {cache_dir}")
        return None

    print(f"Found {len(cache_files)} cache files in {cache_dir}")

    # Load first file to check structure
    sample = torch.load(cache_files[0], map_location="cpu", weights_only=True)
    if isinstance(sample, dict):
        print(f"  Cache keys: {list(sample.keys())[:5]}...")

    return {"files": cache_files, "limit": limit}


def evaluate_kd_loss(
    model: nn.Module,
    tokenizer,
    cache_info: Dict,
    device: torch.device,
    num_samples: int = 100,
    use_cpu: bool = False,
) -> Dict[str, float]:
    """Evaluate model using KD cache logits.

    Uses memory-efficient approach: only computes logits for cached top-k indices,
    not full vocab (same as train_qat.py).

    Args:
        use_cpu: If True, run evaluation on CPU.
    """
    files = cache_info["files"]
    limit = cache_info.get("limit", 0) or num_samples

    total_kd_loss = 0.0
    total_samples = 0

    # Use CPU for evaluation if requested
    eval_device = torch.device("cpu") if use_cpu else device

    if use_cpu and device.type != "cpu":
        print("  (Using CPU for KD evaluation)")
        model_was_on = device
        model.to(eval_device)
    else:
        model_was_on = None

    model.eval()

    # Get lm_head for efficient logit computation
    if not hasattr(model, "lm_head"):
        print("  Warning: Model has no lm_head, cannot compute KD loss")
        return {"kd_loss": float("nan"), "samples": 0}

    lm_head = model.lm_head
    # Get the base model (without lm_head) for hidden states
    if hasattr(model, "model"):
        base_model = model.model
    else:
        print("  Warning: Cannot find base model for hidden states")
        return {"kd_loss": float("nan"), "samples": 0}

    for i, cache_file in enumerate(files[:limit]):
        print(f"\r  [{i+1}/{min(limit, len(files))}] Processing {cache_file.name}...", end="", flush=True)
        try:
            # Load to CPU first
            data = torch.load(cache_file, map_location="cpu", weights_only=True)

            if "input_ids" not in data:
                continue

            input_ids = data["input_ids"]
            attention_mask = data.get("attention_mask")

            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            if attention_mask is not None and attention_mask.dim() == 1:
                attention_mask = attention_mask.unsqueeze(0)

            input_ids = input_ids.to(eval_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(eval_device)

            # Get teacher logits - check both naming conventions
            teacher_logits = data.get("topk_logits", data.get("teacher_logits"))
            teacher_indices = data.get("topk_idx", data.get("teacher_indices", data.get("topk_indices")))

            if teacher_logits is None or teacher_indices is None:
                continue

            # Move teacher data to eval device
            teacher_logits = teacher_logits.to(eval_device).float()
            teacher_indices = teacher_indices.to(eval_device).long()

            # Handle shape: should be [B, L-1, K] or [L-1, K]
            if teacher_indices.dim() == 2:
                teacher_indices = teacher_indices.unsqueeze(0)
                teacher_logits = teacher_logits.unsqueeze(0)

            with torch.no_grad():
                # Get hidden states only (NOT full logits) - memory efficient!
                out = base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
                hidden = out.last_hidden_state[:, :-1, :]  # [B, S=L-1, H]
                B, S, H = hidden.shape

                # Only compute logits for top-k candidate indices
                K = teacher_indices.size(-1)
                seq_len = min(S, teacher_indices.size(1))

                # Flatten for efficient gather
                h = hidden[:, :seq_len, :].reshape(B * seq_len, H)  # [N, H]
                idx = teacher_indices[:, :seq_len, :].reshape(B * seq_len, K)  # [N, K]

                # Gather lm_head weights for candidate tokens only
                # lm_head.weight: [vocab, H] -> gather [N, K, H]
                w = lm_head.weight[idx]  # [N, K, H]

                # Compute student logits for candidates only: [N, K]
                student_topk = torch.einsum("nh,nkh->nk", h, w)
                student_topk = student_topk.view(B, seq_len, K)

                # Teacher logits for same positions
                t_logits = teacher_logits[:, :seq_len, :]

                # KL divergence
                teacher_probs = F.softmax(t_logits, dim=-1)
                student_log_probs = F.log_softmax(student_topk, dim=-1)
                kl = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")

                total_kd_loss += kl.item()
                total_samples += 1

        except Exception as e:
            print(f"\n  Warning: Failed to process {cache_file.name}: {e}")
            continue

        if total_samples >= num_samples:
            break

    print()  # Newline after progress

    # Move model back if we moved it
    if model_was_on is not None:
        model.to(model_was_on)

    if total_samples == 0:
        return {"kd_loss": float("nan"), "samples": 0}

    return {
        "kd_loss": total_kd_loss / total_samples,
        "samples": total_samples,
    }


@torch.no_grad()
def run_inference(
    model,
    tokenizer,
    device: torch.device,
    prompt: str,
    system_prompt: str = "",
    prompt_format: str = "chat",
    max_new_tokens: int = 128,
    strip_think: bool = True,
    verbose: bool = False,
    stream: bool = False,
) -> str:
    """Run inference with the model."""
    if prompt_format == "raw":
        prompt_text = prompt
    elif prompt_format in {"chat", "chat_think"}:
        messages = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        elif prompt_format == "chat_think":
            messages.append({
                "role": "system",
                "content": "You are Qwen, a helpful assistant.\nFirst think through the problem in <think>...</think>, then give a concise final answer.",
            })
        messages.append({"role": "user", "content": prompt})
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if prompt_format == "chat_think":
            prompt_text = prompt_text + "<think>\n"
    else:
        prompt_text = prompt

    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Use streamer for real-time output
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False) if stream else None

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        streamer=streamer,
    )[0]

    gen_ids = output_ids[inputs["input_ids"].shape[1]:]
    raw_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    if verbose and not stream:
        print(f"  [raw output, {len(gen_ids)} tokens]: {raw_text[:200]}...")

    text = raw_text
    if strip_think and text:
        if "</think>" in text:
            text = text.split("</think>", 1)[1].strip()
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # If stripping removed everything, return raw (truncated)
    if not text and raw_text:
        text = f"(thinking only) {raw_text[:100]}..."

    return text


def main():
    parser = argparse.ArgumentParser(description="Test AnemllQATLinear with groupwise LUT quantization")

    # Model args
    parser.add_argument("--model-id", default="Qwen/Qwen3-0.6B", help="HF model id or local path")
    parser.add_argument("--device", default="auto", help="auto|cpu|cuda|mps")
    parser.add_argument("--inference-device", default="same", help="same|auto|cpu|cuda|mps (use 'cpu' if MPS inference fails)")
    parser.add_argument("--dtype", default="bf16", help="fp16|bf16|fp32")
    parser.add_argument("--local-files-only", action="store_true")

    # MLP quantization args
    parser.add_argument("--lut-size", type=int, default=16, help="Number of LUT entries (4=2bit, 16=4bit)")
    parser.add_argument("--group-size", type=int, default=128, help="Group size along input dimension")
    parser.add_argument("--scale-rank", type=int, default=4, help="Low-rank for scales (0=full)")
    parser.add_argument("--learnable-lut", action="store_true", help="Make LUT values trainable")
    parser.add_argument("--lut-include-zero", action="store_true", help="Include 0 in LUT")
    parser.add_argument("--quantize-lm-head", action="store_true", help="Also quantize lm_head (large vocab projection, may cause MPS issues)")

    # Attention quantization args
    parser.add_argument("--quantize-attn", action="store_true", help="Also quantize attention layers")
    parser.add_argument("--attn-lut-size", type=int, default=0, help="Attention LUT size (0=use --lut-size)")
    parser.add_argument("--attn-group-size", type=int, default=0, help="Attention group size (0=use --group-size)")
    parser.add_argument("--attn-scale-rank", type=int, default=-1, help="Attention scale rank (-1=use --scale-rank)")

    # Evaluation args
    parser.add_argument("--kd-cache-dir", default="", help="KD cache directory for evaluation")
    parser.add_argument("--eval-samples", type=int, default=100, help="Number of samples for KD eval")
    parser.add_argument("--eval-cpu", action="store_true", help="Run KD evaluation on CPU (avoids MPS memory issues)")
    parser.add_argument("--skip-inference", action="store_true", help="Skip inference test")
    parser.add_argument("--run-baseline", action="store_true", help="Run baseline inference before quantization")

    # Output args
    parser.add_argument("--save-quantized-dir", default="", help="Save quantized model to this directory")
    parser.add_argument("--gmm-stats", action="store_true", help="Print GMM storage stats")
    parser.add_argument("--use-chat-template", action="store_true", help="Use chat template for inference")
    parser.add_argument("--prompt-format", default="chat", choices=["raw", "chat", "chat_think"])
    parser.add_argument("--prompt", default="What is the capital of France?")
    parser.add_argument("--system-prompt", default="", help="Optional system prompt")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--strip-think", action=argparse.BooleanOptionalAction, default=True, help="Strip <think>...</think> from output")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show raw inference output")
    parser.add_argument("--stream", action="store_true", help="Stream tokens as they're generated")

    args = parser.parse_args()

    # Handle defaults
    if args.use_chat_template:
        args.prompt_format = "chat"
    attn_lut_size = args.attn_lut_size if args.attn_lut_size > 0 else args.lut_size
    attn_group_size = args.attn_group_size if args.attn_group_size > 0 else args.group_size
    attn_scale_rank = args.attn_scale_rank if args.attn_scale_rank >= 0 else args.scale_rank

    device = _pick_device(args.device)
    inference_device = device if args.inference_device == "same" else _pick_device(args.inference_device)
    dtype = _pick_dtype(args.dtype)

    print(f"Device: {device}, inference_device: {inference_device}, dtype: {dtype}")
    print(f"MLP config: lut_size={args.lut_size}, group_size={args.group_size}, scale_rank={args.scale_rank}")
    if args.quantize_attn:
        print(f"Attn config: lut_size={attn_lut_size}, group_size={attn_group_size}, scale_rank={attn_scale_rank}")

    # Load model
    print(f"\nLoading model: {args.model_id}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        local_files_only=args.local_files_only,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        local_files_only=args.local_files_only,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()
    print(f"Loaded in {time.time() - t0:.2f}s")

    # Run baseline if requested
    if args.run_baseline and not args.skip_inference:
        print("\n--- Baseline Inference ---")
        if inference_device != device:
            model.to(inference_device)
        t_inf = time.time()
        baseline = run_inference(
            model, tokenizer, inference_device,
            prompt=args.prompt,
            system_prompt=args.system_prompt,
            prompt_format=args.prompt_format,
            max_new_tokens=args.max_new_tokens,
            strip_think=args.strip_think,
            verbose=args.verbose,
            stream=args.stream,
        )
        print(f"[baseline] {baseline} ({time.time() - t_inf:.2f}s)")
        if inference_device != device:
            model.to(device)

    # Create configs
    mlp_config = AnemllQuantConfig(
        lut_size=args.lut_size,
        group_size=args.group_size,
        scale_rank=args.scale_rank,
        learnable_lut=args.learnable_lut,
        lut_include_zero=args.lut_include_zero,
    )

    attn_config = None
    if args.quantize_attn:
        attn_config = AnemllQuantConfig(
            lut_size=attn_lut_size,
            group_size=attn_group_size,
            scale_rank=attn_scale_rank,
            learnable_lut=args.learnable_lut,
            lut_include_zero=args.lut_include_zero,
        )

    # Replace linear layers
    print("\n--- Replacing Linear layers with AnemllQATLinear ---")
    mlp_regex = r"\.mlp\.(gate_proj|up_proj|down_proj)$"
    attn_regex = r"\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$"

    combined_regex = mlp_regex
    if args.quantize_attn:
        combined_regex = f"({mlp_regex})|({attn_regex})"

    if args.quantize_lm_head:
        # Add lm_head to quantization (warning: may cause MPS issues)
        lm_head_regex = r"lm_head$"
        combined_regex = f"({combined_regex})|({lm_head_regex})"
        print("  (Including lm_head - may cause MPS memory issues)")

    count = replace_linear_with_anemll(
        model,
        config=mlp_config,
        name_regex=combined_regex,
        attn_config=attn_config,
        attn_regex=attn_regex,
        verbose=True,
    )
    print(f"Replaced {count} linear layers")

    # Move model back to device (ensures all new modules are on correct device)
    model.to(device)
    print(f"Model moved to {device}")

    # Evaluate reconstruction error
    print("\n--- Quantization Reconstruction Error ---")
    error_stats = evaluate_quantization_error(model, verbose=True)
    print(f"\nOverall: mae={error_stats['mae']:.6g} rmse={error_stats['rmse']:.6g} rel_l2={error_stats['rel_l2']:.6g}")
    print(f"Total quantized params: {error_stats.get('total_params', 0):,}")

    # GMM stats
    if args.gmm_stats:
        print("\n--- GMM Storage Stats ---")
        total_full_scales = 0
        total_ab_params = 0
        for name, module in model.named_modules():
            if isinstance(module, AnemllQATLinear):
                if module.use_low_rank:
                    ab_params = module.scale_A.numel() + module.scale_B.numel()
                    total_ab_params += ab_params
                    print(f"  {name}: A={tuple(module.scale_A.shape)}, B={tuple(module.scale_B.shape)}, total={ab_params}")
                else:
                    full_params = module.full_scales.numel()
                    total_full_scales += full_params
                    print(f"  {name}: full_scales={tuple(module.full_scales.shape)}, total={full_params}")
        print(f"Total: A@B params={total_ab_params:,}, full_scales params={total_full_scales:,}")

    # KD cache evaluation
    if args.kd_cache_dir:
        print(f"\n--- KD Cache Evaluation ---")
        cache_info = load_kd_cache(args.kd_cache_dir, limit=args.eval_samples)
        if cache_info:
            # MPS can't handle large lm_head outputs (vocab_size x seq_len), use CPU
            use_cpu_for_eval = args.eval_cpu or (device.type == "mps")
            if device.type == "mps" and not args.eval_cpu:
                print("  (Auto-using CPU: MPS can't handle lm_head output [151936 x 128])")
            kd_stats = evaluate_kd_loss(
                model, tokenizer, cache_info, device,
                num_samples=args.eval_samples,
                use_cpu=use_cpu_for_eval,
            )
            print(f"KD Loss: {kd_stats['kd_loss']:.4f} (samples={kd_stats['samples']})")

    # Inference test
    if not args.skip_inference:
        print("\n--- Quantized Inference ---")
        if inference_device != device:
            model.to(inference_device)
        t_inf = time.time()
        answer = run_inference(
            model, tokenizer, inference_device,
            prompt=args.prompt,
            system_prompt=args.system_prompt,
            prompt_format=args.prompt_format,
            max_new_tokens=args.max_new_tokens,
            strip_think=args.strip_think,
            verbose=args.verbose,
            stream=args.stream,
        )
        print(f"[quantized] {answer} ({time.time() - t_inf:.2f}s)")
        if inference_device != device:
            model.to(device)

    # Save if requested
    if args.save_quantized_dir:
        save_dir = Path(args.save_quantized_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n--- Saving to {save_dir} ---")

        # Save state dict
        state_dict = model.state_dict()
        torch.save(state_dict, save_dir / "model_state_dict.pt")

        # Save tokenizer
        tokenizer.save_pretrained(save_dir / "tokenizer")

        # Save config
        model.config.save_pretrained(save_dir / "config")

        # Save quantization config
        quant_config = {
            "mlp": {
                "lut_size": args.lut_size,
                "group_size": args.group_size,
                "scale_rank": args.scale_rank,
                "learnable_lut": args.learnable_lut,
                "lut_include_zero": args.lut_include_zero,
            },
            "attn": {
                "lut_size": attn_lut_size,
                "group_size": attn_group_size,
                "scale_rank": attn_scale_rank,
            } if args.quantize_attn else None,
        }
        import json
        with open(save_dir / "quant_config.json", "w") as f:
            json.dump(quant_config, f, indent=2)

        print(f"Saved to {save_dir}")

    print("\nDone!")


if __name__ == "__main__":
    main()
