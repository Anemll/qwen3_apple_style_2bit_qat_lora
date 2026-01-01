"""
Layer-by-layer QAT training utilities.

Usage:
    from qat_lora.layer_qat import (
        compute_kd_loss_batch,
        evaluate_kd_loss,
        train_layer,
        KDCacheDataset,
        collate_fn,
    )
"""

from __future__ import annotations

import csv
import os
import random
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from torch.optim import AdamW

from .ane_qat_linear import AnemllQATLinear


# ==============================================================================
# LOCAL MLP RECONSTRUCTION LOSS
# ==============================================================================


class LocalMLPLoss(nn.Module):
    """
    Computes local reconstruction loss for the MLP of a transformer layer.

    Compares the quantized MLP output to what frozen FP weights would produce.
    Uses the same approach as train_qat_progressive.py.

    The loss is:
        L = α * MSE(RMSNorm(y_q), RMSNorm(y_fp)) + (1-α) * relative_MSE(y_q, y_fp)

    Where α (norm_weight) ≈ 0.8 by default for direction preservation.
    """

    def __init__(
        self,
        frozen_weights: Dict[str, torch.Tensor],
        rms_norm_eps: float = 1e-6,
        norm_weight: float = 0.8,
        max_loss: float = 10.0,
        num_tokens: int = 128,
    ):
        """
        Args:
            frozen_weights: Dict with 'gate', 'up', 'down' weight tensors
            rms_norm_eps: Epsilon for RMSNorm computation
            norm_weight: Weight α for normalized loss component (default 0.8)
            max_loss: Maximum loss value for clamping
            num_tokens: Number of tokens to sample for loss computation
        """
        super().__init__()
        # Store frozen FP weights as buffers
        self.register_buffer('gate_w', frozen_weights['gate'].clone())
        self.register_buffer('up_w', frozen_weights['up'].clone())
        self.register_buffer('down_w', frozen_weights['down'].clone())
        self.rms_norm_eps = rms_norm_eps
        self.norm_weight = norm_weight
        self.max_loss = max_loss
        self.num_tokens = num_tokens

    def _rms_norm(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMSNorm for directional loss (without learnable scale)."""
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.rms_norm_eps)

    def _sample_tokens(
        self,
        attention_mask: Optional[torch.Tensor],
        total_tokens: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Sample token indices for loss computation."""
        if attention_mask is not None:
            valid_mask = attention_mask.reshape(-1).bool()
            valid_indices = torch.where(valid_mask)[0]
            if len(valid_indices) <= self.num_tokens:
                return valid_indices
            perm = torch.randperm(len(valid_indices), device=device)
            return valid_indices[perm[:self.num_tokens]]
        else:
            if total_tokens <= self.num_tokens:
                return torch.arange(total_tokens, device=device)
            perm = torch.randperm(total_tokens, device=device)
            return perm[:self.num_tokens]

    def forward(
        self,
        mlp_input: torch.Tensor,
        mlp_output_q: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute local MLP reconstruction loss.

        Args:
            mlp_input: Input to MLP [B, T, H]
            mlp_output_q: Quantized MLP output [B, T, H]
            attention_mask: Valid token mask [B, T]

        Returns:
            Scalar loss tensor
        """
        B, T, H = mlp_input.shape
        device = mlp_input.device

        # Sample tokens for efficiency
        sample_idx = self._sample_tokens(attention_mask, B * T, device)

        # Flatten and gather sampled tokens
        mlp_input_flat = mlp_input.reshape(B * T, H)
        mlp_output_q_flat = mlp_output_q.reshape(B * T, H)

        mlp_input_sub = mlp_input_flat[sample_idx].float()
        y_q_sub = mlp_output_q_flat[sample_idx].float()

        # Compute FP MLP output using frozen weights
        with torch.no_grad():
            # Qwen MLP: down(silu(gate(x)) * up(x))
            gate_w = self.gate_w.float()
            up_w = self.up_w.float()
            down_w = self.down_w.float()

            gate_out = F.linear(mlp_input_sub, gate_w)
            up_out = F.linear(mlp_input_sub, up_w)
            hidden = F.silu(gate_out) * up_out
            y_fp_sub = F.linear(hidden, down_w)

        # Check for NaN/inf
        if not torch.isfinite(y_q_sub).all():
            return torch.tensor(self.max_loss, device=device, requires_grad=True)

        # Mixed loss: normalized + relative MSE
        y_q_norm = self._rms_norm(y_q_sub)
        y_fp_norm = self._rms_norm(y_fp_sub)

        loss_norm = F.mse_loss(y_q_norm, y_fp_norm)

        # Relative MSE (scale-invariant)
        fp_scale = y_fp_sub.pow(2).mean().clamp(min=1e-8)
        loss_raw = F.mse_loss(y_q_sub, y_fp_sub) / fp_scale

        loss = self.norm_weight * loss_norm + (1 - self.norm_weight) * loss_raw

        if not torch.isfinite(loss):
            return torch.tensor(self.max_loss, device=device, requires_grad=True)

        return torch.clamp(loss, max=self.max_loss)


def get_mlp_frozen_weights(
    layer: nn.Module,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    """
    Extract frozen FP weights from a layer's MLP.

    Works with both nn.Linear and AnemllQATLinear modules.

    Args:
        layer: Transformer layer with MLP
        dtype: Data type for weights

    Returns:
        Dict with 'gate', 'up', 'down' weight tensors
    """
    mlp = layer.mlp
    return {
        'gate': mlp.gate_proj.weight.data.clone().to(dtype),
        'up': mlp.up_proj.weight.data.clone().to(dtype),
        'down': mlp.down_proj.weight.data.clone().to(dtype),
    }


# Keep old names for backward compatibility
LocalLayerReconstructionLoss = LocalMLPLoss


def create_frozen_fp_layer(
    layer: nn.Module,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, torch.Tensor]:
    """
    Create frozen FP weights for a layer's MLP.

    This is the simplified version that just extracts MLP weights
    instead of copying the entire layer.

    Args:
        layer: Transformer layer
        device: Device to place weights
        dtype: Data type for weights

    Returns:
        Dict with 'gate', 'up', 'down' frozen weight tensors
    """
    weights = get_mlp_frozen_weights(layer, dtype)
    return {k: v.to(device) for k, v in weights.items()}


# ==============================================================================
# KD LOSS COMPUTATION
# ==============================================================================


def compute_kd_loss_batch(
    model: nn.Module,
    batch: dict,
    device: torch.device,
    temperature: float = 2.0,
    no_grad: bool = False,
    hard_top1_weight: float = 0.0,
    hard_full_weight: float = 0.0,
) -> torch.Tensor:
    """Compute KD loss for a batch using memory-efficient approach.

    Args:
        model: The model (must have .model and .lm_head attributes)
        batch: Dict with input_ids, attention_mask, topk_idx, topk_logits
        device: Device to run on
        temperature: Distillation temperature
        no_grad: If True, wrap forward pass in no_grad (for evaluation).
                 During training, set to False to allow gradients.
        hard_top1_weight: Weight for hard label loss on top-1 (helps stabilize training)
        hard_full_weight: Weight for hard label loss on full vocab (small value helps)

    Returns:
        Combined loss scalar (KL + hard label losses)
    """
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch.get('attention_mask')
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    topk_idx = batch['topk_idx'].to(device).long()
    topk_logits = batch['topk_logits'].to(device).float()

    # Get hidden states (not full logits)
    def forward_pass():
        out = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        return out.last_hidden_state[:, :-1, :]  # [B, S, H]

    if no_grad:
        with torch.no_grad():
            hidden = forward_pass()
    else:
        hidden = forward_pass()

    B, S, H = hidden.shape

    # Only compute logits for top-k
    K = topk_idx.size(-1)
    seq_len = min(S, topk_idx.size(1))

    h = hidden[:, :seq_len, :].reshape(B * seq_len, H)
    idx = topk_idx[:, :seq_len, :].reshape(B * seq_len, K)

    # Gather lm_head weights for top-k tokens
    w = model.lm_head.weight[idx]  # [N, K, H]
    student_topk = torch.einsum('nh,nkh->nk', h, w).view(B, seq_len, K)

    # KL divergence with temperature (per-token, matching progressive training)
    t_logits = topk_logits[:, :seq_len, :]
    teacher_probs = F.softmax(t_logits / temperature, dim=-1)
    teacher_log_probs = F.log_softmax(t_logits / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_topk / temperature, dim=-1)

    # KL = sum over K dimension, then mean over B*S tokens
    kl_per_token = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)  # [B, S]
    kl_loss = kl_per_token.mean() * (temperature ** 2)

    total_loss = kl_loss

    # Hard label loss on top-1 (cross-entropy with teacher's top prediction)
    if hard_top1_weight > 0:
        # Teacher's top-1 index within top-k (index 0 is highest logit)
        # We want student to predict the top-1 token
        top1_targets = idx[:, 0]  # [B*S] - actual vocab indices of top-1

        # Compute full student logits for hard label loss
        student_full_logits = h @ model.lm_head.weight.T  # [B*S, V]
        hard_top1_loss = F.cross_entropy(student_full_logits, top1_targets)
        total_loss = total_loss + hard_top1_weight * hard_top1_loss

    # Hard label loss on full vocab (smaller weight, helps with stability)
    if hard_full_weight > 0 and hard_top1_weight == 0:
        # Only compute full logits if not already computed above
        top1_targets = idx[:, 0]
        student_full_logits = h @ model.lm_head.weight.T
        hard_full_loss = F.cross_entropy(student_full_logits, top1_targets)
        total_loss = total_loss + hard_full_weight * hard_full_loss
    elif hard_full_weight > 0:
        # Reuse the already computed loss
        total_loss = total_loss + hard_full_weight * hard_top1_loss

    return total_loss


class KDCacheDataset(IterableDataset):
    """IterableDataset that yields individual examples from KD cache shards.

    Each cache file may contain batched data [N, L, ...], so we iterate
    over individual examples to avoid shape mismatches when collating.
    """

    def __init__(self, cache_dir: str, shuffle: bool = True):
        self.cache_dir = Path(cache_dir)
        self.files = sorted(self.cache_dir.glob('*.pt'))
        self.shuffle = shuffle

    def __len__(self) -> int:
        """Approximate length (number of cache files)."""
        return len(self.files)

    def __iter__(self):
        files = list(self.files)
        if self.shuffle:
            random.shuffle(files)

        for f in files:
            data = torch.load(f, map_location='cpu', weights_only=True)

            # Get tensors and ensure proper dimensions
            ids = data['input_ids']
            if ids.dim() == 1:
                ids = ids.unsqueeze(0)

            topk_idx = data['topk_idx']
            if topk_idx.dim() == 2:
                topk_idx = topk_idx.unsqueeze(0)

            topk_logits = data['topk_logits']
            if topk_logits.dim() == 2:
                topk_logits = topk_logits.unsqueeze(0)

            attn_mask = data.get('attention_mask')
            if attn_mask is not None:
                if attn_mask.dim() == 1:
                    attn_mask = attn_mask.unsqueeze(0)
            else:
                attn_mask = torch.ones_like(ids)

            # Yield individual examples from this shard
            num_examples = ids.size(0)
            for i in range(num_examples):
                yield {
                    'input_ids': ids[i],            # [L]
                    'attention_mask': attn_mask[i], # [L]
                    'topk_idx': topk_idx[i],        # [S, K]
                    'topk_logits': topk_logits[i],  # [S, K]
                }


def collate_fn(batch: List[dict]) -> dict:
    """Collate individual examples into a batch."""
    return {
        'input_ids': torch.stack([b['input_ids'] for b in batch]),           # [B, L]
        'attention_mask': torch.stack([b['attention_mask'] for b in batch]), # [B, L]
        'topk_idx': torch.stack([b['topk_idx'] for b in batch]),             # [B, S, K]
        'topk_logits': torch.stack([b['topk_logits'] for b in batch]),       # [B, S, K]
    }


def is_qat_linear(module: nn.Module) -> bool:
    """Check if module is AnemllQATLinear or V2 (robust against module reloading)."""
    # Use class name check to handle module caching issues
    return type(module).__name__ in ('AnemllQATLinear', 'AnemllQATLinearV2')


def get_layer_modules(
    model: nn.Module,
    layer_idx: int,
) -> List[Tuple[str, nn.Module]]:
    """Get all AnemllQATLinear modules in a specific layer."""
    layer = model.model.layers[layer_idx]
    modules = []
    for name, m in layer.named_modules():
        if is_qat_linear(m):
            modules.append((f'layers.{layer_idx}.{name}', m))
    return modules


def freeze_all_except_layer(
    model: nn.Module,
    layer_idx: int,
    train_weights: bool = True,
    train_scales: bool = False,
    train_mlp_only: bool = False,
) -> int:
    """Freeze all parameters except specified layer's AnemllQATLinear params.

    Training modes:
    - train_weights=True, train_scales=False: Train weights only (default)
    - train_weights=True, train_scales=True: Train both weights and scales
    - train_weights=False, train_scales=True: Train scales only (scale optimization)
    - train_weights=False, train_scales=False: Everything frozen (no training)

    Args:
        model: The model
        layer_idx: Layer index to unfreeze
        train_weights: If True, train weight parameters
        train_scales: If True, train scale_A/scale_B parameters
        train_mlp_only: If True, skip attention projections (q/k/v/o_proj)

    Returns:
        Number of trainable parameters
    """
    # Freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # Attention projection names to skip when train_mlp_only=True
    attn_proj_names = ('q_proj', 'k_proj', 'v_proj', 'o_proj')

    # Unfreeze layer's quantized modules
    layer_modules = get_layer_modules(model, layer_idx)
    trainable = 0
    for name, m in layer_modules:
        # Skip attention projections if train_mlp_only
        if train_mlp_only and any(proj in name for proj in attn_proj_names):
            continue

        # Train weights
        if train_weights:
            m.weight.requires_grad = True
            trainable += m.weight.numel()
        else:
            m.weight.requires_grad = False

        # Train scales
        if train_scales:
            if m.scale_A is not None:
                m.scale_A.requires_grad = True
                trainable += m.scale_A.numel()
            if m.scale_B is not None:
                m.scale_B.requires_grad = True
                trainable += m.scale_B.numel()
            # V2 has rank_magnitude
            if hasattr(m, 'rank_magnitude') and m.rank_magnitude is not None:
                m.rank_magnitude.requires_grad = True
                trainable += m.rank_magnitude.numel()
        else:
            if m.scale_A is not None:
                m.scale_A.requires_grad = False
            if m.scale_B is not None:
                m.scale_B.requires_grad = False
            # V2 has rank_magnitude
            if hasattr(m, 'rank_magnitude') and m.rank_magnitude is not None:
                m.rank_magnitude.requires_grad = False

    return trainable


def evaluate_kd_loss(
    model: nn.Module,
    cache_dir: str,
    device: torch.device,
    num_samples: int = 40,
    temperature: float = 2.0,
) -> float:
    """Evaluate KD loss on cache samples.

    Args:
        model: The model to evaluate
        cache_dir: Path to KD cache directory
        device: Device to run on
        num_samples: Number of samples to evaluate
        temperature: Distillation temperature

    Returns:
        Average KD loss
    """
    cache_path = Path(cache_dir)
    files = sorted(cache_path.glob('*.pt'))

    total_loss = 0.0
    count = 0

    model.eval()
    for f in files:
        if count >= num_samples:
            break

        data = torch.load(f, map_location='cpu', weights_only=True)

        # Handle both single example and batched cache files
        ids = data['input_ids']
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)

        topk_idx = data['topk_idx']
        if topk_idx.dim() == 2:
            topk_idx = topk_idx.unsqueeze(0)

        topk_logits = data['topk_logits']
        if topk_logits.dim() == 2:
            topk_logits = topk_logits.unsqueeze(0)

        attn_mask = data.get('attention_mask')
        if attn_mask is not None:
            if attn_mask.dim() == 1:
                attn_mask = attn_mask.unsqueeze(0)
        else:
            attn_mask = torch.ones_like(ids)

        # Process each example in the shard
        for i in range(ids.size(0)):
            if count >= num_samples:
                break

            batch = {
                'input_ids': ids[i:i+1],
                'attention_mask': attn_mask[i:i+1] if attn_mask is not None else None,
                'topk_idx': topk_idx[i:i+1],
                'topk_logits': topk_logits[i:i+1],
            }

            loss = compute_kd_loss_batch(model, batch, device, temperature, no_grad=True)
            total_loss += loss.item()
            count += 1

    return total_loss / max(1, count)


def train_layer(
    model: nn.Module,
    layer_idx: int,
    cache_dir: str,
    device: torch.device,
    batch_size: int = 4,
    lr: float = 2e-5,
    epochs: int = 1,
    max_steps: int = 0,
    grad_accum: int = 4,
    temperature: float = 2.0,
    train_weights: bool = True,
    train_scales: bool = False,
    train_mlp_only: bool = False,
    verbose: bool = True,
    eval_before: bool = True,
    local_weight: float = 0.5,
    global_weight: float = 0.5,
    local_tokens: int = 128,
    hard_top1_weight: float = 0.0,
    hard_full_weight: float = 0.0,
) -> dict:
    """Train a single layer with local MLP reconstruction + global KD loss.

    Loss = local_weight * LocalMLPLoss + global_weight * GlobalKDLoss
         + hard_top1_weight * HardTop1Loss + hard_full_weight * HardFullLoss

    Training modes:
    - train_weights=True, train_scales=False: Train weights only (default QAT)
    - train_weights=True, train_scales=True: Train both weights and scales
    - train_weights=False, train_scales=True: Scale optimization only (weights frozen)

    Local loss: Compares quantized MLP output to frozen FP MLP output.
    Global loss: Compares final model output to cached teacher top-k logits.
    Hard label loss: Cross-entropy with teacher's top-1 prediction (helps convergence).

    Args:
        model: The model to train
        layer_idx: Layer index to train
        cache_dir: Path to KD cache directory
        device: Device to run on
        batch_size: Batch size
        lr: Learning rate
        epochs: Number of epochs (ignored if max_steps > 0)
        max_steps: Max optimizer steps (0 = use epochs instead)
        grad_accum: Gradient accumulation steps
        temperature: Distillation temperature
        train_weights: If True, train weight parameters
        train_scales: If True, train scale_A/scale_B parameters
        train_mlp_only: If True, skip attention projections (q/k/v/o_proj)
        verbose: Print progress
        eval_before: Evaluate global loss before training (slower but informative)
        local_weight: Weight for local MLP reconstruction loss
        global_weight: Weight for global KD loss
        local_tokens: Number of tokens to sample for local loss
        hard_top1_weight: Weight for hard label top-1 loss (helps convergence)
        hard_full_weight: Weight for hard label full vocab loss

    Returns:
        Dict with 'layer', 'before', 'after', 'improvement', 'time_sec', etc.
    """
    import time
    t_start = time.time()

    trainable = freeze_all_except_layer(model, layer_idx, train_weights=train_weights, train_scales=train_scales, train_mlp_only=train_mlp_only)

    # Describe training mode
    if train_weights and train_scales:
        mode = "weights+scales"
    elif train_weights:
        mode = "weights"
    elif train_scales:
        mode = "scales only"
    else:
        mode = "frozen (no training)"
    if verbose:
        print(f'\n=== Layer {layer_idx} === ({trainable:,} trainable params, mode={mode})')
        if hard_top1_weight > 0 or hard_full_weight > 0:
            print(f'  Hard label: top1={hard_top1_weight}, full={hard_full_weight}')

    # Get model dtype
    model_dtype = next(model.parameters()).dtype
    current_layer = model.model.layers[layer_idx]

    # --- Create local MLP loss (if enabled) ---
    use_local = local_weight > 0
    local_loss_fn = None
    mlp_io = {'input': None, 'output': None}
    hook = None

    if use_local:
        # Get frozen MLP weights
        frozen_weights = get_mlp_frozen_weights(current_layer, model_dtype)
        frozen_weights = {k: v.to(device) for k, v in frozen_weights.items()}

        local_loss_fn = LocalMLPLoss(
            frozen_weights=frozen_weights,
            num_tokens=local_tokens,
        ).to(device)

        # Hook the MLP module (not full layer)
        def capture_mlp_io(module, inp, out):
            mlp_io['input'] = inp[0]  # Keep for local loss
            mlp_io['output'] = out

        hook = current_layer.mlp.register_forward_hook(capture_mlp_io)

    # --- Evaluate BEFORE training this layer ---
    loss_before = None
    if eval_before:
        model.eval()
        loss_before = evaluate_kd_loss(model, cache_dir, device, num_samples=20, temperature=temperature)
        if verbose:
            print(f'  [Global KD Loss BEFORE]: {loss_before:.4f}')

    # --- Training ---
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(params, lr=lr)

    model.train()
    total_local_loss = 0.0
    total_global_loss = 0.0
    steps = 0
    t_train_start = time.time()
    first_local = None
    first_global = None
    last_local = None
    last_global = None

    # Use many epochs if max_steps specified (will break early)
    num_epochs = 1000 if max_steps > 0 else epochs
    done = False

    for epoch in range(num_epochs):
        if done:
            break

        # Create fresh dataloader each epoch for proper shuffling
        dataset = KDCacheDataset(cache_dir, shuffle=True)
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

        for i, batch in enumerate(dataloader):
            # Global KD loss (forward pass also captures MLP I/O via hook)
            global_loss = compute_kd_loss_batch(
                model, batch, device, temperature,
                no_grad=False,
                hard_top1_weight=hard_top1_weight,
                hard_full_weight=hard_full_weight,
            )

            # Local MLP reconstruction loss
            if use_local:
                attn_mask = batch.get('attention_mask')
                if attn_mask is not None:
                    attn_mask = attn_mask[:, :-1].to(device)

                local_loss = local_loss_fn(
                    mlp_io['input'][:, :-1, :],
                    mlp_io['output'][:, :-1, :],
                    attention_mask=attn_mask,
                )
            else:
                local_loss = torch.tensor(0.0, device=device)

            # Combined loss
            loss = local_weight * local_loss + global_weight * global_loss
            loss = loss / grad_accum
            loss.backward()

            if (i + 1) % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()
                steps += 1

                step_local = local_loss.item() if use_local else 0.0
                step_global = global_loss.item()
                total_local_loss += step_local
                total_global_loss += step_global

                # Track first and last losses
                if first_global is None:
                    first_local = step_local
                    first_global = step_global
                last_local = step_local
                last_global = step_global

                if verbose and steps % 10 == 0:
                    elapsed = time.time() - t_train_start
                    if use_local:
                        print(f'  step {steps}: local={step_local:.4f} global={step_global:.4f} ({elapsed:.1f}s)')
                    else:
                        print(f'  step {steps}: global={step_global:.4f} ({elapsed:.1f}s)')

                # Check if we've reached max_steps
                if max_steps > 0 and steps >= max_steps:
                    done = True
                    break

    # --- Cleanup ---
    if hook is not None:
        hook.remove()
    if local_loss_fn is not None:
        del local_loss_fn

    # --- Training summary ---
    avg_local = total_local_loss / steps if steps > 0 else 0
    avg_global = total_global_loss / steps if steps > 0 else 0
    local_improvement = (first_local - last_local) if first_local and last_local else 0
    global_improvement = (first_global - last_global) if first_global and last_global else 0

    # --- Evaluate AFTER training this layer ---
    model.eval()
    loss_after = evaluate_kd_loss(model, cache_dir, device, num_samples=20, temperature=temperature)

    # --- Timing ---
    t_end = time.time()
    layer_time = t_end - t_start

    # --- Report ---
    if verbose:
        print(f'  ---')
        if use_local and first_local is not None:
            print(f'  [Local Loss]:   {first_local:.4f} -> {last_local:.4f} (Δ={local_improvement:.4f})')
        if first_global is not None:
            print(f'  [Global Loss]:  {first_global:.4f} -> {last_global:.4f} (Δ={global_improvement:.4f})')
        elif steps == 0:
            print(f'  [Warning]: No training steps completed (check batch_size vs data)')
        if loss_before is not None:
            improvement = loss_before - loss_after
            pct = 100 * improvement / loss_before if loss_before > 0 else 0
            print(f'  [Eval KD]:      {loss_before:.4f} -> {loss_after:.4f} (Δ={improvement:.4f}, {pct:.1f}%)')
        print(f'  [Time]:         {layer_time:.1f}s')

    # Cleanup to free GPU memory
    del optimizer
    del params
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        'layer': layer_idx,
        'before': loss_before,
        'after': loss_after,
        'improvement': (loss_before - loss_after) if loss_before else None,
        'local_first': first_local,
        'local_last': last_local,
        'local_avg': avg_local,
        'global_first': first_global,
        'global_last': last_global,
        'global_avg': avg_global,
        'time_sec': layer_time,
    }


def train_all_layers(
    model: nn.Module,
    cache_dir: str,
    device: torch.device,
    batch_size: int = 4,
    lr: float = 2e-5,
    epochs_per_layer: int = 1,
    steps_per_layer: int = 0,
    grad_accum: int = 4,
    temperature: float = 2.0,
    train_weights: bool = True,
    train_scales: bool = False,
    train_mlp_only: bool = False,
    verbose: bool = True,
    eval_before: bool = True,
    local_weight: float = 0.5,
    global_weight: float = 0.5,
    local_tokens: int = 128,
    hard_top1_weight: float = 0.0,
    hard_full_weight: float = 0.0,
) -> List[dict]:
    """Train all layers sequentially with local + global loss.

    Training modes:
    - train_weights=True, train_scales=False: Train weights only (default QAT)
    - train_weights=True, train_scales=True: Train both weights and scales
    - train_weights=False, train_scales=True: Scale optimization only (weights frozen)

    Args:
        model: The model to train
        cache_dir: Path to KD cache directory
        device: Device to run on
        batch_size: Batch size
        lr: Learning rate
        epochs_per_layer: Epochs per layer (ignored if steps_per_layer > 0)
        steps_per_layer: Max steps per layer (0 = use epochs instead)
        grad_accum: Gradient accumulation steps
        temperature: Distillation temperature
        train_weights: If True, train weight parameters
        train_scales: If True, train scale_A/scale_B parameters
        train_mlp_only: If True, skip attention projections (q/k/v/o_proj)
        verbose: Print progress
        eval_before: Evaluate global loss before each layer
        local_weight: Weight for local reconstruction loss
        global_weight: Weight for global KD loss
        local_tokens: Number of tokens to sample for local loss
        hard_top1_weight: Weight for hard label top-1 loss (helps convergence)
        hard_full_weight: Weight for hard label full vocab loss

    Returns:
        List of dicts with layer stats including local/global loss
    """
    import time

    def format_time(seconds: float) -> str:
        """Format seconds as HH:MM:SS or MM:SS."""
        if seconds < 3600:
            return f'{int(seconds // 60)}:{int(seconds % 60):02d}'
        else:
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            s = int(seconds % 60)
            return f'{h}:{m:02d}:{s:02d}'

    num_layers = len(model.model.layers)

    # Describe training mode
    if train_weights and train_scales:
        mode = "weights+scales"
    elif train_weights:
        mode = "weights"
    elif train_scales:
        mode = "scales only"
    else:
        mode = "frozen (no training)"
    if train_mlp_only:
        mode += " (MLP only)"

    if verbose:
        print(f'Training {num_layers} layers (mode={mode})...')
        print(f'Cache: {cache_dir}')
        print(f'Batch size: {batch_size}, Grad accum: {grad_accum}')
        if steps_per_layer > 0:
            print(f'LR: {lr}, Steps per layer: {steps_per_layer}')
        else:
            print(f'LR: {lr}, Epochs per layer: {epochs_per_layer}')
        if hard_top1_weight > 0 or hard_full_weight > 0:
            print(f'Hard label: top1={hard_top1_weight}, full={hard_full_weight}')

    # Get initial global loss
    initial_loss = evaluate_kd_loss(model, cache_dir, device, num_samples=40, temperature=temperature)
    if verbose:
        print(f'\n[Initial Global KD Loss]: {initial_loss:.4f}')

    layer_results = []
    t0 = time.time()
    layer_times = []

    for layer_idx in range(num_layers):
        # Show ETA
        if verbose and layer_idx > 0 and len(layer_times) > 0:
            avg_layer_time = sum(layer_times) / len(layer_times)
            remaining_layers = num_layers - layer_idx
            eta_seconds = avg_layer_time * remaining_layers
            elapsed = time.time() - t0
            print(f'\n[Progress: {layer_idx}/{num_layers}] Elapsed: {format_time(elapsed)}, ETA: {format_time(eta_seconds)}')

        result = train_layer(
            model, layer_idx, cache_dir, device,
            batch_size=batch_size,
            lr=lr,
            epochs=epochs_per_layer,
            max_steps=steps_per_layer,
            grad_accum=grad_accum,
            temperature=temperature,
            train_weights=train_weights,
            train_scales=train_scales,
            train_mlp_only=train_mlp_only,
            verbose=verbose,
            eval_before=eval_before,
            local_weight=local_weight,
            global_weight=global_weight,
            local_tokens=local_tokens,
            hard_top1_weight=hard_top1_weight,
            hard_full_weight=hard_full_weight,
        )
        layer_results.append(result)
        layer_times.append(result['time_sec'])

    # Final global loss
    final_loss = evaluate_kd_loss(model, cache_dir, device, num_samples=40, temperature=temperature)

    if verbose:
        elapsed = time.time() - t0
        avg_time = sum(layer_times) / len(layer_times) if layer_times else 0
        total_improvement = initial_loss - final_loss
        pct_improvement = 100 * total_improvement / initial_loss if initial_loss > 0 else 0

        print(f'\n{"="*60}')
        print(f'LAYER-BY-LAYER TRAINING COMPLETE')
        print(f'{"="*60}')
        print(f'Total time:    {format_time(elapsed)} ({elapsed:.1f}s)')
        print(f'Avg per layer: {avg_time:.1f}s')
        print()
        print(f'[Initial Global KD Loss]: {initial_loss:.4f}')
        print(f'[Final Global KD Loss]:   {final_loss:.4f}')
        print(f'[Total Improvement]:      {total_improvement:.4f} ({pct_improvement:.1f}%)')
        print()
        print('Per-layer summary:')
        print(f'{"Layer":>6} {"Eval Before":>12} {"Eval After":>12} {"Eval Δ":>10} {"Local 1st":>10} {"Local Last":>10} {"Global 1st":>11} {"Global Last":>12} {"Time":>8}')
        print('-' * 105)
        for r in layer_results:
            imp = r['improvement'] if r['improvement'] is not None else 0
            before = r['before'] if r['before'] is not None else 0
            local_first = r.get('local_first', 0) or 0
            local_last = r.get('local_last', 0) or 0
            global_first = r.get('global_first', 0) or 0
            global_last = r.get('global_last', 0) or 0
            print(f"{r['layer']:>6} {before:>12.4f} {r['after']:>12.4f} {imp:>+10.4f} {local_first:>10.4f} {local_last:>10.4f} {global_first:>11.4f} {global_last:>12.4f} {r['time_sec']:>7.1f}s")

    return layer_results


def save_checkpoint(
    model: nn.Module,
    save_dir: str,
    config: dict = None,
    name: str = "model_state_dict.pt",
    save_indices: bool = True,
    verbose: bool = True,
) -> str:
    """Save model checkpoint with optional config and indices.

    Args:
        model: Model to save
        save_dir: Directory to save to
        config: Optional config dict to save as config.json
        name: Filename for state dict (default: model_state_dict.pt)
        save_indices: If True, compute and save quantization indices
        verbose: Print save info

    Returns:
        Path to saved state dict
    """
    import os
    import json
    from .ane_qat_linear import compute_all_indices

    os.makedirs(save_dir, exist_ok=True)

    # Detect snapped_mode from model (all layers should have same mode)
    snapped_mode = None
    for m in model.modules():
        if type(m).__name__ in ('AnemllQATLinear', 'AnemllQATLinearV2'):
            snapped_mode = getattr(m, 'snapped_mode', None)
            if snapped_mode is not None:
                break

    # Save state dict
    state_path = os.path.join(save_dir, name)
    torch.save(model.state_dict(), state_path)

    if verbose:
        print(f"Saved checkpoint to {save_dir}/")
        print(f"  - {name}")

    # Save indices
    if save_indices:
        indices_dict = compute_all_indices(model, verbose=False)
        if indices_dict:
            indices_path = os.path.join(save_dir, "indices.pt")
            torch.save(indices_dict, indices_path)
            # Compute size
            total_bytes = sum(idx.numel() * idx.element_size() for idx in indices_dict.values())
            if verbose:
                print(f"  - indices.pt ({len(indices_dict)} layers, {total_bytes / 1024 / 1024:.1f} MB)")

    # Save config if provided (or create one with snapped_mode)
    if config is None:
        config = {}
    if snapped_mode is not None:
        config['snapped_mode'] = snapped_mode
    if config:
        config_path = os.path.join(save_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, default=str)
        if verbose:
            print(f"  - config.json")
            if snapped_mode:
                print(f"    snapped_mode: {snapped_mode}")

    return state_path


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device = None,
    verbose: bool = True,
) -> dict:
    """Load checkpoint into model.

    Args:
        model: Model to load into (must have AnemllQATLinear layers already)
        checkpoint_path: Path to state dict .pt file
        device: Device to map to
        verbose: Print load info

    Returns:
        Dict with 'missing_keys', 'unexpected_keys', and optionally 'config'
    """
    import os
    import json

    if device is None:
        device = next(model.parameters()).device

    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    result = model.load_state_dict(state, strict=False)

    if verbose:
        print(f"Loaded checkpoint: {checkpoint_path}")
        if result.missing_keys:
            print(f"  Missing keys: {len(result.missing_keys)}")
        if result.unexpected_keys:
            print(f"  Unexpected keys: {len(result.unexpected_keys)}")
        if not result.missing_keys and not result.unexpected_keys:
            print(f"  All keys matched")

    # Try to load config.json from same directory
    config = None
    checkpoint_dir = os.path.dirname(checkpoint_path)
    config_path = os.path.join(checkpoint_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)

        # Restore snapped_mode to all layers
        # Handle both new format (snapped_mode) and old format (snapped + snap_bake_scales)
        snapped_mode = config.get('snapped_mode')
        if snapped_mode is None and config.get('snapped'):
            # Old format: infer from snapped + snap_bake_scales
            snapped_mode = 'baked' if config.get('snap_bake_scales') else 'lut'

        # Restore per-layer attributes from loaded state
        import math
        attn_proj_names = ('q_proj', 'k_proj', 'v_proj', 'o_proj')

        # Fallback lut_bits from config (used if LUT not in state)
        mlp_lut_bits_default = int(math.log2(config.get('lut_size', 16)))
        attn_lut_bits_default = int(math.log2(config.get('attn_lut_size', config.get('lut_size', 16))))

        count = 0
        lut_bits_summary = {}  # Track unique lut_bits values
        for name, m in model.named_modules():
            if type(m).__name__ in ('AnemllQATLinear', 'AnemllQATLinearV2'):
                # Set snapped_mode
                if snapped_mode:
                    m.snapped_mode = snapped_mode

                # Derive lut_bits from actual loaded LUT size (future: per-layer)
                if hasattr(m, 'lut') and m.lut is not None:
                    m.lut_bits = int(math.log2(m.lut.size(0)))
                else:
                    # Fallback to config-based assignment
                    is_attn = any(proj in name for proj in attn_proj_names)
                    m.lut_bits = attn_lut_bits_default if is_attn else mlp_lut_bits_default

                # Track for summary
                lut_bits_summary[m.lut_bits] = lut_bits_summary.get(m.lut_bits, 0) + 1
                count += 1

        if verbose:
            if snapped_mode:
                print(f"  Restored snapped_mode='{snapped_mode}' to {count} layers")
            # Show lut_bits distribution
            bits_str = ", ".join(f"{bits}-bit: {cnt}" for bits, cnt in sorted(lut_bits_summary.items()))
            print(f"  lut_bits: {bits_str}")

    return {
        'missing_keys': result.missing_keys,
        'unexpected_keys': result.unexpected_keys,
        'config': config,
    }


def train_e2e(
    model: nn.Module,
    cache_dir: str,
    device: torch.device,
    max_steps: int = 1000,
    batch_size: int = 32,
    lr: float = 1e-5,
    temperature: float = 2.0,
    train_weights: bool = True,
    train_scales: bool = False,
    train_g_only: bool = False,
    train_mlp_only: bool = False,
    hard_top1_weight: float = 0.0,
    hard_full_weight: float = 0.0005,
    logging_steps: int = 50,
    eval_steps: int = 200,
    eval_samples: int = 40,
    save_dir: str = None,
    save_steps: int = 0,
    verbose: bool = True,
    use_cosine_schedule: bool = False,
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.1,
    use_fp16: bool = False,
    use_wandb: bool = False,
    wandb_project: str = "qwen3-qat",
    wandb_run_name: str = None,
    wandb_config: dict = None,
) -> dict:
    """End-to-end KD-QAT training (all layers unfrozen).

    Args:
        model: Model with AnemllQATLinear layers
        cache_dir: Path to KD cache directory
        device: Device to run on
        max_steps: Maximum training steps
        batch_size: Batch size
        lr: Learning rate (peak LR if using schedule)
        temperature: Distillation temperature
        train_weights: If True, train weight parameters
        train_scales: If True, train scale_A/scale_B/rank_magnitude parameters
        train_g_only: If True, train only rank_magnitude (G), freeze A and B (requires train_scales=True)
        train_mlp_only: If True, freeze attention layers (q/k/v/o_proj) and only train MLP
                        (gate/up/down_proj). Useful for mixed-bit configs (e.g., 4-bit attn, 2-bit MLP)
        hard_top1_weight: Weight for hard label top-1 loss (0 to disable)
        hard_full_weight: Weight for hard label full vocab loss (default 0.0005, helps stability)
        logging_steps: Log every N steps
        eval_steps: Evaluate every N steps
        eval_samples: Samples for evaluation
        save_dir: Directory to save checkpoints (optional)
        save_steps: Save checkpoint every N steps (0 to disable periodic saves)
        verbose: Print progress
        use_cosine_schedule: Use cosine annealing LR schedule
        warmup_steps: Linear warmup steps (0 to disable)
        min_lr_ratio: Minimum LR as ratio of peak LR (default 0.1 = 10% of lr)
        use_fp16: Enable FP16 training with GradScaler (CUDA only)
        use_wandb: Enable Weights & Biases logging (requires wandb package)
        wandb_project: W&B project name (default: "qwen3-qat")
        wandb_run_name: W&B run name (default: auto-generated)
        wandb_config: Additional config dict to log to W&B

    Returns:
        Dict with 'initial_loss', 'final_loss', 'best_loss', 'steps', 'time_sec'
    """
    import time
    import os
    from torch.optim import AdamW
    from torch.utils.data import DataLoader

    t_start = time.time()

    # Setup for FP16 training
    # GradScaler is only used for mixed precision (FP32 model + FP16 compute)
    # When model is already FP16, we skip GradScaler (it can't unscale FP16 gradients)
    scaler = None
    model_dtype = next(model.parameters()).dtype
    if use_fp16:
        if device.type == 'cuda' and model_dtype != torch.float16:
            # Mixed precision: model in FP32/BF16, use GradScaler for stability
            scaler = torch.amp.GradScaler('cuda')
            if verbose:
                print(f"[FP16] Mixed precision: model={model_dtype}, using GradScaler")
        elif device.type == 'cuda':
            # Pure FP16: model already in FP16, GradScaler not needed
            if verbose:
                print(f"[FP16] Pure FP16 training: model already {model_dtype}, no GradScaler")
        else:
            if verbose:
                print(f"[FP16] Warning: FP16 training on {device.type} without GradScaler")

    # Setup wandb logging
    wandb_run = None
    if use_wandb:
        try:
            import wandb
            # Build config for wandb
            run_config = {
                'max_steps': max_steps,
                'batch_size': batch_size,
                'lr': lr,
                'temperature': temperature,
                'train_weights': train_weights,
                'train_scales': train_scales,
                'train_g_only': train_g_only,
                'train_mlp_only': train_mlp_only,
                'hard_top1_weight': hard_top1_weight,
                'hard_full_weight': hard_full_weight,
                'use_cosine_schedule': use_cosine_schedule,
                'warmup_steps': warmup_steps,
                'min_lr_ratio': min_lr_ratio,
                'use_fp16': use_fp16,
                'device': str(device),
                'model_dtype': str(model_dtype),
            }
            if wandb_config:
                run_config.update(wandb_config)

            wandb_run = wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                config=run_config,
                reinit=True,
            )
            if verbose:
                print(f"[wandb] Logging to: {wandb_run.url}")
        except ImportError:
            print("[wandb] Warning: wandb not installed. Install with: pip install wandb")
            use_wandb = False
        except Exception as e:
            print(f"[wandb] Warning: Failed to initialize wandb: {e}")
            use_wandb = False

    # Set trainable parameters
    trainable = 0
    attn_frozen = 0
    for p in model.parameters():
        p.requires_grad = False

    # Attention projection names to freeze when train_mlp_only=True
    attn_proj_names = ('q_proj', 'k_proj', 'v_proj', 'o_proj')

    for name, module in model.named_modules():
        if type(module).__name__ in ('AnemllQATLinear', 'AnemllQATLinearV2'):
            # Check if this is an attention projection
            is_attn_proj = any(proj in name for proj in attn_proj_names)

            # Skip attention layers if train_mlp_only
            if train_mlp_only and is_attn_proj:
                attn_frozen += module.weight.numel()
                if hasattr(module, 'scale_A') and module.scale_A is not None:
                    attn_frozen += module.scale_A.numel() + module.scale_B.numel()
                # V2 has rank_magnitude
                if hasattr(module, 'rank_magnitude') and module.rank_magnitude is not None:
                    attn_frozen += module.rank_magnitude.numel()
                continue  # Keep frozen

            if train_weights:
                module.weight.requires_grad = True
                trainable += module.weight.numel()
            if train_scales:
                if hasattr(module, 'scale_A') and module.scale_A is not None:
                    if not train_g_only:  # Only train A/B if not G-only mode
                        module.scale_A.requires_grad = True
                        module.scale_B.requires_grad = True
                        trainable += module.scale_A.numel() + module.scale_B.numel()
                # V2 has rank_magnitude
                if hasattr(module, 'rank_magnitude') and module.rank_magnitude is not None:
                    module.rank_magnitude.requires_grad = True
                    trainable += module.rank_magnitude.numel()

    # Describe mode
    mode_parts = []
    if train_weights:
        mode_parts.append("weights")
    if train_scales:
        if train_g_only:
            mode_parts.append("G-only")
        else:
            mode_parts.append("scales")
    mode = "+".join(mode_parts) if mode_parts else "none"
    if train_mlp_only:
        mode += " (MLP only)"

    if verbose:
        print(f"=== End-to-End KD-QAT ===")
        print(f"Mode: {mode}")
        print(f"Trainable params: {trainable:,}")
        if train_mlp_only:
            print(f"Frozen attention params: {attn_frozen:,}")
        print(f"Steps: {max_steps}, LR: {lr}, Batch: {batch_size}")
        if hard_top1_weight > 0 or hard_full_weight > 0:
            print(f"Hard label: top1={hard_top1_weight}, full={hard_full_weight}")

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("No trainable parameters! Check train_weights/train_scales flags.")
    optimizer = AdamW(params, lr=lr)

    # LR Scheduler (cosine with warmup)
    scheduler = None
    if use_cosine_schedule or warmup_steps > 0:
        import math
        min_lr = lr * min_lr_ratio

        def lr_lambda(current_step):
            # Warmup phase
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            # Cosine decay phase
            if use_cosine_schedule:
                progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
                cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
            return 1.0

        from torch.optim.lr_scheduler import LambdaLR
        scheduler = LambdaLR(optimizer, lr_lambda)

        if verbose:
            sched_info = []
            if warmup_steps > 0:
                sched_info.append(f"warmup={warmup_steps}")
            if use_cosine_schedule:
                sched_info.append(f"cosine→{min_lr:.2e}")
            print(f"LR Schedule: {', '.join(sched_info)}")

    # Initial evaluation
    model.eval()
    initial_loss = evaluate_kd_loss(model, cache_dir, device, num_samples=eval_samples, temperature=temperature)
    if verbose:
        print(f"\nInitial KD Loss: {initial_loss:.4f}")

    # Setup CSV logging
    csv_file = None
    csv_writer = None
    csv_path = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        csv_path = os.path.join(save_dir, "training_log.csv")
        csv_file = open(csv_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['step', 'train_loss', 'eval_loss', 'lr', 'elapsed_sec'])
        # Write initial eval
        csv_writer.writerow([0, '', f'{initial_loss:.6f}', f'{lr:.2e}', '0.0'])
        csv_file.flush()
        if verbose:
            print(f"CSV log: {csv_path}")

    # Training loop
    model.train()
    step = 0
    total_loss = 0.0
    best_loss = initial_loss
    best_state = None
    loss_history = []
    seq_len = None  # Will be set from first batch

    while step < max_steps:
        dataset = KDCacheDataset(cache_dir, shuffle=True)
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

        for batch in dataloader:
            if step >= max_steps:
                break

            # Get seq_len from first batch and print estimated total tokens
            if seq_len is None and 'input_ids' in batch:
                seq_len = batch['input_ids'].shape[1]
                total_tokens_est = batch_size * seq_len * max_steps
                if total_tokens_est >= 1e9:
                    tok_str = f"{total_tokens_est/1e9:.2f}B"
                elif total_tokens_est >= 1e6:
                    tok_str = f"{total_tokens_est/1e6:.1f}M"
                else:
                    tok_str = f"{total_tokens_est/1e3:.0f}K"
                if verbose:
                    print(f"Estimated total tokens: {tok_str} (batch={batch_size}, seq={seq_len}, steps={max_steps})")

            # Forward pass with optional autocast for FP16
            if use_fp16:
                with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
                    loss = compute_kd_loss_batch(
                        model, batch, device, temperature,
                        no_grad=False,
                        hard_top1_weight=hard_top1_weight,
                        hard_full_weight=hard_full_weight,
                    )
            else:
                loss = compute_kd_loss_batch(
                    model, batch, device, temperature,
                    no_grad=False,
                    hard_top1_weight=hard_top1_weight,
                    hard_full_weight=hard_full_weight,
                )

            # Backward pass with optional scaler for FP16
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                # Gradient clipping for FP16 stability
                if use_fp16:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

            # TPU: sync to compile and execute graph
            try:
                import torch_xla
                torch_xla.sync()
            except (ImportError, RuntimeError):
                pass

            total_loss += loss.item()
            step += 1

            # Logging
            if step % logging_steps == 0:
                avg_loss = total_loss / logging_steps
                elapsed = time.time() - t_start
                eta = elapsed / step * (max_steps - step) if step > 0 else 0
                # Format time as M:SS or H:MM:SS
                def fmt_time(s):
                    if s < 3600:
                        return f"{int(s)//60}:{int(s)%60:02d}"
                    return f"{int(s)//3600}:{(int(s)%3600)//60:02d}:{int(s)%60:02d}"
                # Get current LR
                current_lr = scheduler.get_last_lr()[0] if scheduler is not None else lr
                # Calculate throughput
                it_per_sec = step / elapsed if elapsed > 0 else 0
                tok_per_sec = it_per_sec * batch_size * (seq_len or 0)
                throughput_str = f"{tok_per_sec/1000:.1f}k tok/s" if seq_len else f"{it_per_sec:.2f} it/s"
                # Print if verbose
                if verbose:
                    if scheduler is not None:
                        print(f"[{step}/{max_steps}] loss={avg_loss:.4f} lr={current_lr:.2e} ({fmt_time(elapsed)}, ETA {fmt_time(eta)}, {throughput_str})")
                    else:
                        print(f"[{step}/{max_steps}] loss={avg_loss:.4f} ({fmt_time(elapsed)}, ETA {fmt_time(eta)}, {throughput_str})")
                # Write to CSV
                if csv_writer is not None:
                    csv_writer.writerow([step, f'{avg_loss:.6f}', '', f'{current_lr:.2e}', f'{elapsed:.1f}'])
                    csv_file.flush()
                # Log to wandb
                if use_wandb and wandb_run is not None:
                    import wandb
                    wandb.log({
                        'train/loss': avg_loss,
                        'train/lr': current_lr,
                        'train/step': step,
                        'train/elapsed_sec': elapsed,
                    }, step=step)
                loss_history.append(avg_loss)
                total_loss = 0.0

            # Evaluation
            if step % eval_steps == 0:
                model.eval()
                eval_loss = evaluate_kd_loss(model, cache_dir, device, num_samples=eval_samples, temperature=temperature)
                elapsed = time.time() - t_start
                current_lr = scheduler.get_last_lr()[0] if scheduler is not None else lr
                if verbose:
                    print(f"  [Eval] KD Loss: {eval_loss:.4f} (best: {best_loss:.4f})")
                # Write eval to CSV
                if csv_writer is not None:
                    csv_writer.writerow([step, '', f'{eval_loss:.6f}', f'{current_lr:.2e}', f'{elapsed:.1f}'])
                    csv_file.flush()
                # Log to wandb
                if use_wandb and wandb_run is not None:
                    import wandb
                    wandb.log({
                        'eval/loss': eval_loss,
                        'eval/best_loss': best_loss,
                    }, step=step)
                if eval_loss < best_loss:
                    best_loss = eval_loss
                    # Save best state in memory
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    if save_dir:
                        os.makedirs(save_dir, exist_ok=True)
                        torch.save(best_state, os.path.join(save_dir, "best_state_dict.pt"))
                        if verbose:
                            print(f"  [Saved best checkpoint: {best_loss:.4f}]")
                model.train()

            # Periodic checkpoint saving
            if save_steps > 0 and save_dir and step % save_steps == 0:
                os.makedirs(save_dir, exist_ok=True)
                ckpt_path = os.path.join(save_dir, f"checkpoint_step{step}.pt")
                torch.save(model.state_dict(), ckpt_path)
                if verbose:
                    print(f"  [Checkpoint saved: {ckpt_path}]")

    # Final evaluation
    model.eval()
    final_loss = evaluate_kd_loss(model, cache_dir, device, num_samples=eval_samples, temperature=temperature)
    elapsed = time.time() - t_start

    # Write final eval to CSV
    if csv_writer is not None:
        current_lr = scheduler.get_last_lr()[0] if scheduler is not None else lr
        csv_writer.writerow([step, '', f'{final_loss:.6f}', f'{current_lr:.2e}', f'{elapsed:.1f}'])
        csv_file.close()
        if verbose:
            print(f"CSV log saved to: {csv_path}")

    # Log final results to wandb
    if use_wandb and wandb_run is not None:
        import wandb
        wandb.log({
            'eval/loss': final_loss,
            'eval/best_loss': best_loss,
            'summary/initial_loss': initial_loss,
            'summary/final_loss': final_loss,
            'summary/best_loss': best_loss,
            'summary/improvement': initial_loss - final_loss,
            'summary/time_sec': elapsed,
        }, step=step)

    # Update best if final is better
    if final_loss < best_loss:
        best_loss = final_loss
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            torch.save(best_state, os.path.join(save_dir, "best_state_dict.pt"))
            if verbose:
                print(f"  [Saved best checkpoint: {best_loss:.4f}]")

    if verbose:
        print(f"\n=== Results ===")
        print(f"Initial: {initial_loss:.4f}")
        print(f"Final:   {final_loss:.4f}")
        print(f"Best:    {best_loss:.4f}")
        print(f"Improvement: {initial_loss - final_loss:.4f}")
        # Total tokens processed
        if seq_len is not None:
            total_tokens = batch_size * seq_len * step
            if total_tokens >= 1e9:
                tok_str = f"{total_tokens/1e9:.2f}B"
            elif total_tokens >= 1e6:
                tok_str = f"{total_tokens/1e6:.1f}M"
            else:
                tok_str = f"{total_tokens/1e3:.0f}K"
            avg_tok_per_sec = total_tokens / elapsed if elapsed > 0 else 0
            print(f"Tokens:  {tok_str} ({avg_tok_per_sec/1000:.1f}k tok/s avg)")
        print(f"Time: {elapsed:.1f}s")

    # Cleanup to free GPU memory
    del optimizer
    if scheduler is not None:
        del scheduler
    del params
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()

    # Finish wandb run
    wandb_url = None
    if use_wandb and wandb_run is not None:
        import wandb
        wandb_url = wandb_run.url
        wandb.finish()
        if verbose:
            print(f"[wandb] Run finished: {wandb_url}")

    return {
        'initial_loss': initial_loss,
        'final_loss': final_loss,
        'best_loss': best_loss,
        'best_state': best_state,
        'steps': step,
        'time_sec': elapsed,
        'loss_history': loss_history,
        'csv_path': csv_path,
        'wandb_url': wandb_url,
    }
