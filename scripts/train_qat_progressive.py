"""
Progressive Layer-by-Layer QAT Training Script (v1)

This script implements progressive/layer-wise QAT + KD:
- Pass 1: Train MLP layers one at a time (local + global KD loss)
- Pass 2: Train attention layers (global KD only) [optional, v2+]
- Pass 3: MLP refinement pass [optional, v3+]
- Pass 4: E2E quantizer-only tuning (f-param only)

Key design decisions:
- "Option A" (prefix quantized): earlier layers quantized & frozen, suffix fully fp
- Local reconstruction loss using frozen fp copy of current MLP
- Token sampling BEFORE computing fp target (memory efficient)
- Mixed normalized + unnormalized loss for scale preservation
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterator, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure local package imports work without installation.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from qat_lora.model_utils import (
    apply_train_f_only,
    freeze_all_except_layer,
    get_frozen_mlp_copy,
    infer_num_layers,
    init_all_f,
    replace_linear_with_qat,
    set_quantized_prefix,
)
from qat_lora.local_kd import LocalMLPReconstructionLoss
from qat_lora.mixed_precision import pick_device, resolve_amp_dtype, resolve_param_dtype
from qat_lora.quantizer import QATQuantConfig, calibrate_model_f_init


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

# Adaptive layer training: if global loss > threshold after first pass,
# repeat training for that layer up to max_layer_repeats times
LAYER_CONVERGE_THRESHOLD = 0.5  # Global loss threshold for "converged"

# Jump detection: if new layer's starting loss > prev_layer_final * JUMP_MULTIPLIER,
# trigger backtracking to retrain previous layer
JUMP_MULTIPLIER = 10.0  # e.g., 0.02 * 10 = 0.2, so 5.5 would trigger backtrack

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _from_pretrained_fp32(model_name_or_path: str):
    """Load model in fp32."""
    try:
        return AutoModelForCausalLM.from_pretrained(model_name_or_path, dtype=torch.float32)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float32)


def _get_base_transformer(causal_lm: nn.Module) -> nn.Module:
    """Return the base transformer module (without lm_head)."""
    for attr in ("model", "transformer", "gpt_neox", "decoder"):
        if hasattr(causal_lm, attr):
            m = getattr(causal_lm, attr)
            if isinstance(m, nn.Module):
                return m
    raise AttributeError("Could not locate base transformer module.")


# -----------------------------------------------------------------------------
# KD Cache Dataset (reused from train_qat.py)
# -----------------------------------------------------------------------------

class TopKCacheIterableDataset(IterableDataset):
    """Stream samples from a KD cache directory."""

    def __init__(self, cache_dir: str, shuffle_files: bool = False, seed: int = 0):
        super().__init__()
        self.cache_dir = Path(cache_dir)
        if not self.cache_dir.exists():
            raise FileNotFoundError(f"KD cache dir not found: {self.cache_dir}")
        self.shuffle_files = bool(shuffle_files)
        self.seed = int(seed)

        self.meta = {}
        meta_path = self.cache_dir / "meta.json"
        if meta_path.exists():
            try:
                self.meta = json.loads(meta_path.read_text())
            except Exception:
                self.meta = {}

        shards = sorted(self.cache_dir.glob("shard_*.pt"))
        if not shards:
            shards = sorted([p for p in self.cache_dir.glob("*.pt")
                           if not p.name.startswith("checkpoint_")])
        if not shards:
            raise FileNotFoundError(f"No shard files found in: {self.cache_dir}")
        self.shards = shards

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        rng = random.Random(self.seed + int(torch.initial_seed()) % 10_000_000)
        files = list(self.shards)
        if self.shuffle_files:
            rng.shuffle(files)

        for path in files:
            try:
                obj = torch.load(path, map_location="cpu")
            except Exception as e:
                print(f"[WARN] Skipping corrupted cache file {path}: {e}")
                continue
            if not isinstance(obj, dict):
                continue

            input_ids = obj.get("input_ids")
            attn = obj.get("attention_mask")
            topk_idx = obj.get("topk_idx")
            topk_logits = obj.get("topk_logits")
            if input_ids is None or attn is None or topk_idx is None or topk_logits is None:
                raise ValueError(f"Cache shard missing required keys: {path}")

            rand_idx = obj.get("rand_idx", None)
            rand_logits = obj.get("rand_logits", None)

            n = input_ids.shape[0]
            for i in range(n):
                ex = {
                    "input_ids": input_ids[i],
                    "attention_mask": attn[i],
                    "topk_idx": topk_idx[i],
                    "topk_logits": topk_logits[i],
                }
                if rand_idx is not None and rand_logits is not None:
                    ex["rand_idx"] = rand_idx[i]
                    ex["rand_logits"] = rand_logits[i]
                yield ex


def _collate_cache_batch(examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate cache examples into a batch."""
    keys = examples[0].keys()
    batch: Dict[str, torch.Tensor] = {}
    for k in keys:
        batch[k] = torch.stack([ex[k] for ex in examples], dim=0)
    return batch


def infinite_iter(dataloader):
    """Create an infinite iterator over a dataloader."""
    while True:
        for batch in dataloader:
            yield batch


# -----------------------------------------------------------------------------
# Global KD Loss (Cached)
# -----------------------------------------------------------------------------

def compute_cached_kd_loss(
    model: nn.Module,
    hidden: torch.Tensor,
    batch: Dict[str, torch.Tensor],
    temperature: float = 2.0,
    hard_top1_weight: float = 0.0,
) -> torch.Tensor:
    """
    Compute KD loss using cached teacher top-k logits.

    Args:
        model: The model (need lm_head.weight)
        hidden: Hidden states [B, S, H] (S = L-1, shifted for next-token prediction)
        batch: Dict with topk_idx, topk_logits, attention_mask
        temperature: KD temperature
        hard_top1_weight: Weight for hard top-1 loss component

    Returns:
        Scalar loss tensor
    """
    lm_head = model.lm_head
    W = lm_head.weight  # [V, H]

    B, S, H = hidden.shape
    N = B * S

    # Candidate ids/logits from cache
    cand_idx = batch["topk_idx"].to(torch.long)      # [B, S, K]
    cand_t_logits = batch["topk_logits"].float()     # [B, S, K]

    # Optional random negatives
    if "rand_idx" in batch and "rand_logits" in batch:
        cand_idx = torch.cat([cand_idx, batch["rand_idx"].to(torch.long)], dim=-1)
        cand_t_logits = torch.cat([cand_t_logits, batch["rand_logits"].float()], dim=-1)

    C = cand_idx.shape[-1]

    # Flatten positions
    h = hidden.reshape(N, H)           # [N, H]
    i = cand_idx.reshape(N, C)         # [N, C]

    # Gather lm_head weight rows: [N, C, H]
    Wc = W[i]

    # Student logits on candidates: [B, S, C]
    logits_c = torch.bmm(Wc, h.unsqueeze(-1)).squeeze(-1).reshape(B, S, C)

    # KD KL on candidate set
    T = temperature
    t = cand_t_logits / T
    s = logits_c.float() / T

    p_t = torch.softmax(t, dim=-1)
    log_p_t = torch.log_softmax(t, dim=-1)
    log_p_s_T = torch.log_softmax(s, dim=-1)

    kl = (p_t * (log_p_t - log_p_s_T)).sum(dim=-1)  # [B, S]

    # Apply attention mask
    attention_mask = batch.get("attention_mask")
    if attention_mask is not None:
        m = attention_mask[:, 1:].float()  # Align with prediction positions
        denom = m.sum().clamp(min=1.0)
        loss = (kl * m).sum() / denom
    else:
        loss = kl.mean()

    loss = loss * (T * T)

    # Optional hard top-1 loss
    if hard_top1_weight > 0.0:
        log_p_s_1 = torch.log_softmax(logits_c.float(), dim=-1)
        hard = -log_p_s_1[..., 0]  # Teacher top-1 is index 0
        if attention_mask is not None:
            hard = (hard * m).sum() / denom
        else:
            hard = hard.mean()
        loss = loss + hard_top1_weight * hard

    return loss


# -----------------------------------------------------------------------------
# Progressive Training
# -----------------------------------------------------------------------------

def train_progressive_qat(args):
    """Main progressive QAT training function."""
    training_start_time = time.time()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = pick_device(args.device)
    amp_dtype = resolve_amp_dtype(args.amp_dtype, device)
    param_dtype = resolve_param_dtype(args.param_dtype, device)
    print(f"[device] {device} | amp_dtype={amp_dtype} | param_dtype={param_dtype}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[model] Loading {args.model_name_or_path}")
    model = _from_pretrained_fp32(args.model_name_or_path)
    model.config.use_cache = False

    # Optional: load initial model state
    if args.init_model_state:
        print(f"[init] Loading model state from {args.init_model_state}")
        obj = torch.load(args.init_model_state, map_location="cpu")
        if isinstance(obj, dict):
            if "model" in obj:
                state_dict = obj["model"]
            elif "state_dict" in obj:
                state_dict = obj["state_dict"]
            else:
                state_dict = obj
        else:
            raise RuntimeError(f"Could not interpret checkpoint format: {args.init_model_state}")
        model.load_state_dict(state_dict, strict=False)

    # Apply QAT
    qc = QATQuantConfig(n_bits=int(args.quant_bits))
    print(f"[qat] weight_bits={qc.n_bits}")
    exclude = r"(^lm_head$)" if args.skip_lm_head else None
    replace_linear_with_qat(model, qc=qc, exclude_regex=exclude, verbose=False)

    # Initialize f parameters
    if args.calibrate_f_init:
        print(f"[calibrate] Running f_init calibration with method='{args.calibrate_f_init}'")
        calibrate_model_f_init(model, n_bits=qc.n_bits, method=args.calibrate_f_init, verbose=True)
    else:
        init_all_f(model, qc=qc, method="newton", verbose=False)

    # Cast to param dtype
    model = model.to(dtype=param_dtype)
    model = model.to(device)

    # Load KD cache dataset
    print(f"[kd-cache] Loading from {args.kd_cache_dir}")
    kd_dataset = TopKCacheIterableDataset(args.kd_cache_dir, shuffle_files=True)
    dataloader = DataLoader(
        kd_dataset,
        batch_size=args.batch_size,
        collate_fn=_collate_cache_batch,
        drop_last=True,
    )
    data_iter = infinite_iter(dataloader)

    num_layers = infer_num_layers(model)
    print(f"[model] {num_layers} transformer layers")
    print(f"[training] batch_size={args.batch_size} steps_per_mlp={args.steps_per_layer_mlp} e2e_steps={args.e2e_steps}")

    loss_log = []  # For per-layer CSV logging
    base = _get_base_transformer(model)

    # AMP scaler (if using fp16)
    use_amp = amp_dtype in (torch.float16, torch.bfloat16)
    scaler = torch.amp.GradScaler(device.type, enabled=(amp_dtype == torch.float16))

    # =========================================================================
    # PASS 1: MLP Training (layer-by-layer)
    # =========================================================================
    if not args.skip_mlp_pass:
        print("\n" + "=" * 60)
        print("PASS 1: Progressive MLP Training")
        print("=" * 60)
        converge_threshold = args.layer_converge_threshold
        jump_multiplier = args.jump_multiplier

        # Use while loop to allow backtracking
        layer_idx = 0
        layer_final_losses = {}  # Track final loss per layer
        backtrack_count = {}  # Track how many times we've backtracked per layer
        layer_times = []  # Track time per layer for ETA
        pass_start_time = time.time()

        while layer_idx < num_layers:
            layer_start_time = time.time()
            # Adaptive repeats: train layer multiple times if not converged
            final_global_loss = float('inf')
            initial_global_loss = None  # Track starting loss for jump detection
            early_backtrack = False  # Flag for early abort due to jump detection

            for repeat_idx in range(args.max_layer_repeats):
                if early_backtrack:
                    break  # Exit repeat loop if early backtrack triggered
                repeat_str = f" (repeat {repeat_idx+1}/{args.max_layer_repeats})" if repeat_idx > 0 else ""
                print(f"\n--- Layer {layer_idx}/{num_layers-1} MLP{repeat_str} ---")

                # 1. Store frozen fp copy BEFORE training
                frozen_weights = get_frozen_mlp_copy(model, layer_idx)

                # 2. Set quantized prefix + fp suffix
                set_quantized_prefix(model, layer_idx, pass_type='mlp')

                # 3. Freeze all except current layer's MLP
                trainable = freeze_all_except_layer(
                    model, layer_idx, component='mlp', train_f_only=args.train_f_only
                )
                print(f"  Trainable params: {trainable:,}" + (" (f-only)" if args.train_f_only else ""))

                # 4. Create local loss module
                local_loss_fn = LocalMLPReconstructionLoss(
                    frozen_weights,
                    norm_weight=args.local_norm_weight,
                ).to(device)

                # 5. Create optimizer
                optimizer = torch.optim.AdamW(
                    [p for p in model.parameters() if p.requires_grad],
                    lr=args.learning_rate,
                    weight_decay=args.weight_decay,
                )

                # 6. Register hook to capture MLP input/output
                mlp_io = {'input': None, 'output': None}

                def capture_mlp_io(module, inp, out):
                    mlp_io['input'] = inp[0]
                    mlp_io['output'] = out

                hook = model.model.layers[layer_idx].mlp.register_forward_hook(capture_mlp_io)

                # 7. Train for N steps
                model.train()
                for step in range(args.steps_per_layer_mlp):
                    batch = next(data_iter)
                    batch = {k: v.to(device) for k, v in batch.items()}

                    optimizer.zero_grad(set_to_none=True)

                    with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                        # Forward pass to get hidden states
                        outputs = base(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            use_cache=False,
                            return_dict=True,
                        )
                        hidden = outputs.last_hidden_state[:, :-1, :]  # [B, S, H]

                        # Local reconstruction loss
                        # Note: slice attention_mask to match the :-1 on MLP tensors
                        attn_mask = batch.get('attention_mask')
                        if attn_mask is not None:
                            attn_mask = attn_mask[:, :-1]  # Align with prediction positions
                        local_loss = local_loss_fn(
                            mlp_io['input'][:, :-1, :],
                            mlp_io['output'][:, :-1, :],
                            attn_mask,
                            num_tokens=args.local_token_samples,
                        )

                        # Global KD loss
                        global_loss = compute_cached_kd_loss(
                            model, hidden, batch,
                            temperature=args.distill_temperature,
                            hard_top1_weight=args.hard_top1_weight,
                        )

                        loss = args.local_weight * local_loss + args.global_weight * global_loss

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()

                    # Capture initial loss for jump detection (first step of first repeat)
                    if step == 0 and repeat_idx == 0:
                        initial_global_loss = global_loss.item()

                        # EARLY jump detection: check immediately on first step
                        # If there's a huge jump from previous layer, abort early and backtrack
                        if layer_idx > 0:
                            prev_loss = layer_final_losses.get(layer_idx - 1, 0.0)
                            if prev_loss > 0:
                                expected_max = prev_loss * jump_multiplier
                                if initial_global_loss > expected_max:
                                    bt_count = backtrack_count.get(layer_idx - 1, 0)
                                    if bt_count < args.max_backtrack:
                                        print(f"\n  [EARLY BACKTRACK] Layer {layer_idx} started at {initial_global_loss:.4f} "
                                              f"(>{expected_max:.4f} = {prev_loss:.4f} * {jump_multiplier})")
                                        print(f"  [EARLY BACKTRACK] Aborting layer {layer_idx}, going back to retrain layer {layer_idx - 1} "
                                              f"(backtrack {bt_count + 1}/{args.max_backtrack})")
                                        hook.remove()
                                        del local_loss_fn, frozen_weights, optimizer
                                        backtrack_count[layer_idx - 1] = bt_count + 1
                                        # Use a flag to signal early abort
                                        early_backtrack = True
                                        break  # Exit the step loop

                    if step % args.logging_steps == 0:
                        print(f"  step {step}: local={local_loss.item():.4f} global={global_loss.item():.4f}")
                        loss_log.append({
                            'pass': 1, 'component': 'mlp', 'layer': layer_idx,
                            'step': step, 'local': local_loss.item(), 'global': global_loss.item(),
                        })

                # Skip normal cleanup if early backtrack already cleaned up
                if early_backtrack:
                    break

                # Track final global loss for this repeat
                final_global_loss = global_loss.item()

                hook.remove()
                del local_loss_fn, frozen_weights, optimizer

                # Check convergence - break if loss is low enough
                if final_global_loss <= converge_threshold:
                    if repeat_idx > 0:
                        print(f"  Layer {layer_idx} converged after {repeat_idx+1} repeats (global={final_global_loss:.4f})")
                    break
                elif repeat_idx < args.max_layer_repeats - 1:
                    print(f"  Layer {layer_idx} not converged (global={final_global_loss:.4f} > {converge_threshold}), repeating...")

            # Handle early backtrack - go back to previous layer
            if early_backtrack:
                layer_idx -= 1
                continue

            # Final status for this layer
            if final_global_loss > converge_threshold:
                print(f"  [WARN] Layer {layer_idx} did not converge after {args.max_layer_repeats} repeats (global={final_global_loss:.4f})")

            # Store final loss for this layer
            layer_final_losses[layer_idx] = final_global_loss

            # Timing info
            layer_elapsed = time.time() - layer_start_time
            layer_times.append(layer_elapsed)
            total_elapsed = time.time() - pass_start_time
            avg_time_per_layer = sum(layer_times) / len(layer_times)
            remaining_layers = num_layers - layer_idx - 1
            eta_seconds = avg_time_per_layer * remaining_layers
            print(f"  [TIME] Layer {layer_idx} took {layer_elapsed:.1f}s | "
                  f"Total: {total_elapsed/60:.1f}min | "
                  f"ETA: {eta_seconds/60:.1f}min ({remaining_layers} layers left)")

            # Optional: save per-layer checkpoint
            if args.save_layer_checkpoints:
                torch.save(model.state_dict(), out / f"checkpoint_mlp_layer{layer_idx}.pt")

            # Move to next layer
            layer_idx += 1

    # =========================================================================
    # PASS 2: Attention Training (global KD only) [v2+]
    # =========================================================================
    if not args.skip_attention_pass:
        print("\n" + "=" * 60)
        print("PASS 2: Progressive Attention Training (Global KD Only)")
        print("=" * 60)

        for layer_idx in range(num_layers):
            print(f"\n--- Layer {layer_idx}/{num_layers-1} Attention ---")

            # Set quantized prefix (all MLPs now quantized)
            set_quantized_prefix(model, layer_idx, pass_type='attn')

            # Freeze all except current layer's attention
            trainable = freeze_all_except_layer(
                model, layer_idx, component='attn', train_f_only=args.train_f_only
            )
            print(f"  Trainable params: {trainable:,}" + (" (f-only)" if args.train_f_only else ""))

            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            )

            model.train()
            for step in range(args.steps_per_layer_attn):
                batch = next(data_iter)
                batch = {k: v.to(device) for k, v in batch.items()}

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                    outputs = base(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        use_cache=False,
                        return_dict=True,
                    )
                    hidden = outputs.last_hidden_state[:, :-1, :]

                    # Global KD only for attention
                    loss = compute_cached_kd_loss(
                        model, hidden, batch,
                        temperature=args.distill_temperature,
                        hard_top1_weight=args.hard_top1_weight,
                    )

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()

                if step % args.logging_steps == 0:
                    print(f"  step {step}: loss={loss.item():.4f}")
                    loss_log.append({
                        'pass': 2, 'component': 'attn', 'layer': layer_idx,
                        'step': step, 'global': loss.item(),
                    })

            del optimizer

    # =========================================================================
    # PASS 3: MLP Refinement (shorter, addresses coupling) [v3+]
    # =========================================================================
    if not args.skip_mlp_refinement:
        print("\n" + "=" * 60)
        print("PASS 3: MLP Refinement (post-attention-quantization)")
        print("=" * 60)

        refinement_steps = max(1, args.steps_per_layer_mlp // 2)
        refinement_lr = args.learning_rate * 0.5

        for layer_idx in range(num_layers):
            print(f"\n--- Layer {layer_idx}/{num_layers-1} MLP Refinement ---")

            frozen_weights = get_frozen_mlp_copy(model, layer_idx)
            set_quantized_prefix(model, layer_idx, pass_type='mlp')
            trainable = freeze_all_except_layer(
                model, layer_idx, component='mlp', train_f_only=args.train_f_only
            )

            local_loss_fn = LocalMLPReconstructionLoss(
                frozen_weights,
                norm_weight=args.local_norm_weight,
            ).to(device)

            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=refinement_lr,
                weight_decay=args.weight_decay,
            )

            mlp_io = {'input': None, 'output': None}

            def capture_mlp_io(module, inp, out):
                mlp_io['input'] = inp[0]
                mlp_io['output'] = out

            hook = model.model.layers[layer_idx].mlp.register_forward_hook(capture_mlp_io)

            model.train()
            for step in range(refinement_steps):
                batch = next(data_iter)
                batch = {k: v.to(device) for k, v in batch.items()}

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                    outputs = base(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        use_cache=False,
                        return_dict=True,
                    )
                    hidden = outputs.last_hidden_state[:, :-1, :]

                    # Note: slice attention_mask to match the :-1 on MLP tensors
                    attn_mask = batch.get('attention_mask')
                    if attn_mask is not None:
                        attn_mask = attn_mask[:, :-1]  # Align with prediction positions
                    local_loss = local_loss_fn(
                        mlp_io['input'][:, :-1, :],
                        mlp_io['output'][:, :-1, :],
                        attn_mask,
                        num_tokens=args.local_token_samples,
                    )

                    global_loss = compute_cached_kd_loss(
                        model, hidden, batch,
                        temperature=args.distill_temperature,
                        hard_top1_weight=args.hard_top1_weight,
                    )

                    loss = args.local_weight * local_loss + args.global_weight * global_loss

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()

                if step % args.logging_steps == 0:
                    print(f"  step {step}: local={local_loss.item():.4f} global={global_loss.item():.4f}")
                    loss_log.append({
                        'pass': 3, 'component': 'mlp_refine', 'layer': layer_idx,
                        'step': step, 'local': local_loss.item(), 'global': global_loss.item(),
                    })

            hook.remove()
            del local_loss_fn, frozen_weights, optimizer

    # =========================================================================
    # PASS 4: E2E Quantizer-Only Tuning (f-param only)
    # =========================================================================
    print("\n" + "=" * 60)
    print("PASS 4: E2E Quantizer Tuning (f-only)")
    print("=" * 60)
    e2e_start_time = time.time()

    # Enable fake-quant on all layers
    set_quantized_prefix(model, num_layers - 1, pass_type='all')

    # Freeze everything except f parameters
    num_f = apply_train_f_only(model)
    print(f"  Trainable f parameters: {num_f}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.e2e_learning_rate,
        weight_decay=args.weight_decay,
    )

    model.train()
    for step in range(args.e2e_steps):
        batch = next(data_iter)
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
            outputs = base(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
                return_dict=True,
            )
            hidden = outputs.last_hidden_state[:, :-1, :]

            loss = compute_cached_kd_loss(
                model, hidden, batch,
                temperature=args.distill_temperature,
                hard_top1_weight=args.hard_top1_weight,
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        if step % args.logging_steps == 0:
            print(f"  step {step}: loss={loss.item():.4f}")
            loss_log.append({
                'pass': 4, 'component': 'e2e_f', 'layer': -1,
                'step': step, 'global': loss.item(),
            })

    e2e_elapsed = time.time() - e2e_start_time
    print(f"  [TIME] E2E pass took {e2e_elapsed/60:.1f}min")

    # =========================================================================
    # Save outputs
    # =========================================================================
    print("\n" + "=" * 60)
    print("Saving outputs")
    print("=" * 60)

    # Save final checkpoint
    torch.save(model.state_dict(), out / "qat_state_dict.pt")
    print(f"  Model saved to: {out / 'qat_state_dict.pt'}")

    # Save loss log
    if loss_log:
        import csv
        log_path = out / "loss_per_layer.csv"
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=loss_log[0].keys())
            writer.writeheader()
            writer.writerows(loss_log)
        print(f"  Loss log saved to: {log_path}")

    # Save training args
    with open(out / "training_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    total_time = time.time() - training_start_time
    print(f"\nDone! Total training time: {total_time/60:.1f}min ({total_time/3600:.2f}h)")


# -----------------------------------------------------------------------------
# Arguments
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Progressive Layer-by-Layer QAT Training")

    # Model + output
    p.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-0.6B")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--init_model_state", type=str, default=None,
                   help="Optional model state_dict to initialize from")

    # Device & precision
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--amp_dtype", type=str, default="auto", choices=["auto", "no", "bf16", "fp16"])
    p.add_argument("--param_dtype", type=str, default="auto", choices=["auto", "fp32", "bf16", "fp16"])

    # QAT config
    p.add_argument("-q", "--quant_bits", type=int, default=4, choices=[2, 4],
                   help="Weight quantization bits (default: 4 for stability)")
    p.add_argument("--skip_lm_head", action="store_true", help="Don't quantize lm_head")
    p.add_argument("--calibrate_f_init", type=str, default=None,
                   choices=["mse_grid", "newton", "percentile"],
                   help="Calibrate f_init before training using MSE grid-search (recommended for 2-bit), "
                        "Newton-like optimization, or simple percentile clipping")

    # KD cache
    p.add_argument("--kd_cache_dir", type=str, required=True,
                   help="Path to teacher top-k cache directory")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--distill_temperature", type=float, default=2.0)
    p.add_argument("--hard_top1_weight", type=float, default=0.0)

    # Progressive training steps
    p.add_argument("--steps_per_layer_mlp", type=int, default=50,
                   help="Training steps per MLP layer (Pass 1)")
    p.add_argument("--steps_per_layer_attn", type=int, default=30,
                   help="Training steps per attention layer (Pass 2)")
    p.add_argument("--e2e_steps", type=int, default=200,
                   help="E2E quantizer-only tuning steps (Pass 4)")

    # Adaptive layer repeats
    p.add_argument("--max_layer_repeats", type=int, default=1,
                   help="Max repeats per layer if global loss > threshold (default 1 = no repeats)")
    p.add_argument("--layer_converge_threshold", type=float, default=0.5,
                   help="Global loss threshold for layer convergence (default 0.5)")

    # Backtracking on big loss jumps
    p.add_argument("--jump_multiplier", type=float, default=10.0,
                   help="If new layer's starting loss > prev_final * this, trigger backtrack (default 10.0)")
    p.add_argument("--max_backtrack", type=int, default=2,
                   help="Max times to backtrack per layer (default 2)")

    # Loss weights
    p.add_argument("--local_weight", type=float, default=0.3,
                   help="Weight for local reconstruction loss")
    p.add_argument("--global_weight", type=float, default=1.0,
                   help="Weight for global KD loss")
    p.add_argument("--local_token_samples", type=int, default=128,
                   help="Number of tokens to sample for local loss")
    p.add_argument("--local_norm_weight", type=float, default=0.8,
                   help="Weight α for normalized component in mixed local loss")

    # Optimizer
    p.add_argument("--learning_rate", type=float, default=5e-6)
    p.add_argument("--e2e_learning_rate", type=float, default=1e-6)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    # Logging
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_layer_checkpoints", action="store_true",
                   help="Save checkpoint after each layer")

    # Pass control (for staged implementation)
    p.add_argument("--skip_mlp_pass", action="store_true",
                   help="Skip Pass 1 (MLP training)")
    p.add_argument("--skip_attention_pass", action="store_true",
                   help="Skip Pass 2 (attention training)")
    p.add_argument("--skip_mlp_refinement", action="store_true",
                   help="Skip Pass 3 (MLP refinement)")

    # Stability options
    p.add_argument("--train_f_only", action="store_true",
                   help="Only train _f_param (quantization scales), freeze weights. "
                        "More stable for ultra-low-bit quantization.")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_progressive_qat(args)
