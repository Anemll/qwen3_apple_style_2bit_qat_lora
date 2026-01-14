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
from contextlib import nullcontext
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from torch.optim import AdamW

from .ane_qat_linear import AnemllQATLinear


# ==============================================================================
# TPU/XLA SUPPORT
# See docs/TPU.md for debugging and troubleshooting guide
# ==============================================================================

def is_xla_device(device) -> bool:
    """Check if device is TPU/XLA."""
    return 'xla' in str(device).lower()


def xla_mark_step():
    """Mark XLA step to trigger compilation/execution.

    On TPU, XLA buffers all operations and only compiles/executes when forced.
    Calling mark_step() breaks up the graph, preventing massive single-compilation.
    """
    try:
        import torch_xla.core.xla_model as xm
        xm.mark_step()
    except ImportError:
        pass  # Not on TPU


# ==============================================================================
# SAFE KD LOSS (TPU/XLA compatible - avoids aten::kl_div)
# ==============================================================================

def kd_soft_ce(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
               temperature: float = 1.0) -> torch.Tensor:
    """Soft cross-entropy loss for knowledge distillation.

    This is equivalent to KL divergence but uses only ops that XLA handles well
    (log_softmax, softmax, mul, sum) instead of aten::kl_div which may have
    incorrect autograd behavior on TPU.

    Args:
        student_logits: Student model logits [B, L, V] or [B, L, K]
        teacher_logits: Teacher model logits (same shape as student)
        temperature: Distillation temperature (default 1.0)

    Returns:
        Scalar loss (mean over all positions)
    """
    T = float(temperature)
    # Student log probabilities (differentiable)
    logp_student = F.log_softmax(student_logits / T, dim=-1)
    # Teacher probabilities (constant, no gradient)
    with torch.no_grad():
        p_teacher = F.softmax(teacher_logits / T, dim=-1)
    # Cross-entropy: -sum(p * log_q)
    loss = -(p_teacher * logp_student).sum(dim=-1)
    # T^2 scaling (standard distillation)
    return loss.mean() * (T * T)


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
    debug_step: int = -1,
    # Sparse sampled CE (for L1024+ TPU memory safety)
    sampled_ce_weight: float = 0.0,
    sampled_negatives: int = 64,
    vocab_size: int = 151936,  # Qwen3 vocab size default
) -> torch.Tensor:
    """Compute KD loss for a batch using memory-efficient approach.

    Args:
        model: The model (must have .model and .lm_head attributes)
        batch: Dict with input_ids, attention_mask, topk_idx, topk_logits
               Optional: rand_idx [B,L,R], rand_logits [B,L,R] for negative samples
        device: Device to run on
        temperature: Distillation temperature
        no_grad: If True, wrap forward pass in no_grad (for evaluation).
                 During training, set to False to allow gradients.
        hard_top1_weight: Weight for hard label loss on top-1 (helps stabilize training)
                          WARNING: Materializes [B*L, V] full logits tensor!
        hard_full_weight: Weight for hard label loss on full vocab (small value helps)
                          WARNING: Materializes [B*L, V] full logits tensor!
        debug_step: If >= 0, print debug info for this step (for XLA debugging)
        sampled_ce_weight: Weight for sampled CE loss on K+R candidates (sparse, no full logits).
                           Memory-safe alternative to hard_top1_weight for L>=1024.
        sampled_negatives: Number of random negatives to sample if cache lacks rand_idx.
        vocab_size: Vocabulary size for random negative sampling.

    Returns:
        Combined loss scalar (KL + hard label losses + sampled CE)
    """
    import time as _time
    _dbg = debug_step >= 0

    # Get model dtype for consistent precision (bf16 on TPU)
    if _dbg:
        print(f"    [kd_loss] step={debug_step} getting model dtype...", flush=True)
    model_dtype = next(model.parameters()).dtype
    if _dbg:
        print(f"    [kd_loss] step={debug_step} model_dtype={model_dtype}", flush=True)

    if _dbg:
        print(f"    [kd_loss] step={debug_step} moving batch to device...", flush=True)
    input_ids = batch['input_ids'].to(device)

    # Conditionally drop attention_mask if it's all-ones (no padding)
    # This avoids HF's masking_utils path which causes XLA sync noise
    # Check on CPU to avoid XLA graph breaks
    attention_mask = batch.get('attention_mask')
    if attention_mask is not None:
        # Ensure long dtype for comparison
        mask_cpu = attention_mask.cpu() if attention_mask.device.type != 'cpu' else attention_mask
        if mask_cpu.dtype not in (torch.long, torch.int64, torch.int32):
            mask_cpu = mask_cpu.long()
        # If all ones (no padding), skip mask entirely
        if torch.all(mask_cpu == 1):
            attention_mask = None
            if _dbg:
                print(f"    [kd_loss] attention_mask dropped (all ones, no padding)", flush=True)
        else:
            attention_mask = attention_mask.to(device)
            if _dbg:
                print(f"    [kd_loss] attention_mask kept (padding detected)", flush=True)

    topk_idx = batch['topk_idx'].to(device).long()
    topk_logits = batch['topk_logits'].to(device).to(model_dtype)  # Match model dtype
    if _dbg:
        print(f"    [kd_loss] step={debug_step} batch moved, input_ids.shape={input_ids.shape}", flush=True)

    # Get hidden states (not full logits)
    def forward_pass():
        out = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        return out.last_hidden_state[:, :-1, :]  # [B, S, H]

    if _dbg:
        print(f"    [kd_loss] step={debug_step} starting model.model() forward...", flush=True)
        _t0 = _time.time()
    if no_grad:
        with torch.no_grad():
            hidden = forward_pass()
    else:
        hidden = forward_pass()
    if _dbg:
        print(f"    [kd_loss] step={debug_step} model.model() done ({_time.time()-_t0:.1f}s), hidden.shape={hidden.shape}", flush=True)

    # NaN detection for debugging (only when _dbg=True to avoid XLA graph breaks)
    if _dbg and (hidden.isnan().any() or hidden.isinf().any()):
        nan_count = hidden.isnan().sum().item()
        inf_count = hidden.isinf().sum().item()
        print(f"[NaN DEBUG] hidden has {nan_count} NaN, {inf_count} inf values!")
        print(f"[NaN DEBUG] hidden shape={hidden.shape}, dtype={hidden.dtype}")
        print(f"[NaN DEBUG] hidden range: [{hidden[~hidden.isnan() & ~hidden.isinf()].min():.4f}, {hidden[~hidden.isnan() & ~hidden.isinf()].max():.4f}]" if (~hidden.isnan() & ~hidden.isinf()).any() else "[NaN DEBUG] All values are NaN/inf!")

    B, S, H = hidden.shape

    # Only compute logits for top-k
    K = topk_idx.size(-1)
    seq_len = min(S, topk_idx.size(1))

    h = hidden[:, :seq_len, :].reshape(B * seq_len, H)
    idx = topk_idx[:, :seq_len, :].reshape(B * seq_len, K)

    # Gather lm_head weights for top-k tokens
    w = model.lm_head.weight[idx]  # [N, K, H]
    student_topk = torch.einsum('nh,nkh->nk', h, w).view(B, seq_len, K)

    # NaN detection for student logits (only when _dbg=True to avoid XLA graph breaks)
    if _dbg and (student_topk.isnan().any() or student_topk.isinf().any()):
        nan_count = student_topk.isnan().sum().item()
        inf_count = student_topk.isinf().sum().item()
        print(f"[NaN DEBUG] student_topk has {nan_count} NaN, {inf_count} inf values!")
        print(f"[NaN DEBUG] student_topk shape={student_topk.shape}, dtype={student_topk.dtype}")
        # Check if h or w is the culprit
        if h.isnan().any() or h.isinf().any():
            print(f"[NaN DEBUG] h (hidden) has NaN/inf")
        if w.isnan().any() or w.isinf().any():
            print(f"[NaN DEBUG] w (lm_head weights) has NaN/inf")

    # KL divergence with temperature (per-token, matching progressive training)
    t_logits = topk_logits[:, :seq_len, :]
    teacher_probs = F.softmax(t_logits / temperature, dim=-1)
    teacher_log_probs = F.log_softmax(t_logits / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_topk / temperature, dim=-1)

    # KL = sum over K dimension, then mean over B*S tokens
    kl_per_token = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)  # [B, S]
    kl_loss = kl_per_token.mean() * (temperature ** 2)

    # NaN detection for KL loss (only when _dbg=True to avoid XLA graph breaks)
    if _dbg and (kl_loss.isnan() or kl_loss.isinf()):
        print(f"[NaN DEBUG] kl_loss is {kl_loss.item()}")
        if teacher_probs.isnan().any() or teacher_probs.isinf().any():
            print(f"[NaN DEBUG] teacher_probs has NaN/inf")
        if teacher_log_probs.isnan().any() or teacher_log_probs.isinf().any():
            print(f"[NaN DEBUG] teacher_log_probs has NaN/inf")
        if student_log_probs.isnan().any() or student_log_probs.isinf().any():
            nan_slp = student_log_probs.isnan().sum().item()
            inf_slp = student_log_probs.isinf().sum().item()
            print(f"[NaN DEBUG] student_log_probs has {nan_slp} NaN, {inf_slp} inf")
        if kl_per_token.isnan().any() or kl_per_token.isinf().any():
            nan_kl = kl_per_token.isnan().sum().item()
            inf_kl = kl_per_token.isinf().sum().item()
            print(f"[NaN DEBUG] kl_per_token has {nan_kl} NaN, {inf_kl} inf")

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

    # Sampled CE loss on K+R candidates (sparse, no full logits)
    # This is a memory-safe alternative to hard_top1_weight for L>=1024
    if sampled_ce_weight > 0:
        # Get random negatives: from cache or sample on-the-fly
        rand_idx = batch.get('rand_idx')
        if rand_idx is not None:
            # Use cached random negatives
            idx_neg = rand_idx[:, :seq_len, :].to(device).long()  # [B, S, R]
            idx_neg = idx_neg.reshape(B * seq_len, -1)  # [N, R]
            R = idx_neg.size(1)
        elif sampled_negatives > 0:
            # Sample random negatives on-the-fly (fallback)
            # Exclude top-K tokens to ensure negatives are different
            R = sampled_negatives
            idx_neg = torch.randint(0, vocab_size, (B * seq_len, R), device=device, dtype=torch.long)
        else:
            idx_neg = None
            R = 0

        if idx_neg is not None and R > 0:
            # Concatenate top-K + random negatives
            idx_cand = torch.cat([idx, idx_neg], dim=1)  # [N, K+R]

            # Gather lm_head weights for all candidates
            w_cand = model.lm_head.weight[idx_cand]  # [N, K+R, H]

            # Compute student logits for all candidates
            student_cand = torch.einsum('nh,nkh->nk', h, w_cand)  # [N, K+R]

            # Target: class 0 (the top-1 token from teacher)
            # This assumes topk_idx is sorted by teacher logit descending
            targets = torch.zeros((B * seq_len,), dtype=torch.long, device=device)

            sampled_ce = F.cross_entropy(student_cand, targets)
            total_loss = total_loss + sampled_ce_weight * sampled_ce

    return total_loss


class KDCacheDataset(IterableDataset):
    """IterableDataset that yields individual examples from KD cache shards.

    Each cache file may contain batched data [N, L, ...], so we iterate
    over individual examples to avoid shape mismatches when collating.
    """

    def __init__(self, cache_dir: str, shuffle: bool = True, preload: bool = False):
        """
        Args:
            cache_dir: Path to cache directory with .pt files
            shuffle: Shuffle files (and examples if preloaded) each epoch
            preload: Load all data into RAM on init (faster training, more RAM)
        """
        self.cache_dir = Path(cache_dir)
        self.files = sorted(self.cache_dir.glob('*.pt'))
        self.shuffle = shuffle
        self.preload = preload
        self._cached_examples = None

        if preload:
            self._preload_all()

    def _preload_all(self):
        """Load all examples into memory once."""
        self._cached_examples = []
        for f in self.files:
            data = torch.load(f, map_location='cpu', weights_only=True)
            examples = self._parse_shard(data)
            self._cached_examples.extend(examples)
        # Print once
        print(f"[KDCache] Preloaded {len(self._cached_examples)} examples into RAM")

    def _parse_shard(self, data: dict) -> list:
        """Parse a single shard file into list of examples."""
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

        # Build list of examples
        examples = []
        num_examples = ids.size(0)
        for i in range(num_examples):
            examples.append({
                'input_ids': ids[i],            # [L]
                'attention_mask': attn_mask[i], # [L]
                'topk_idx': topk_idx[i],        # [S, K]
                'topk_logits': topk_logits[i],  # [S, K]
            })
        return examples

    def __len__(self) -> int:
        """Approximate length (number of cache files or preloaded examples)."""
        if self._cached_examples is not None:
            return len(self._cached_examples)
        return len(self.files)

    def __iter__(self):
        # Fast path: preloaded data
        if self._cached_examples is not None:
            examples = self._cached_examples
            if self.shuffle:
                indices = list(range(len(examples)))
                random.shuffle(indices)
                for i in indices:
                    yield examples[i]
            else:
                for ex in examples:
                    yield ex
            return

        # Slow path: load from disk each epoch
        files = list(self.files)
        if self.shuffle:
            random.shuffle(files)

        for f in files:
            data = torch.load(f, map_location='cpu', weights_only=True)
            examples = self._parse_shard(data)
            for ex in examples:
                yield ex


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

    # Create dataset ONCE with preload for fast training
    dataset = KDCacheDataset(cache_dir, shuffle=True, preload=True)

    for epoch in range(num_epochs):
        if done:
            break

        # Just create new iterator each epoch (data already in RAM)
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


def prepare_lut_for_save(model: nn.Module, verbose: bool = True) -> int:
    """Repair trainable LUTs and update lut buffers before saving.

    For each layer with trainable LUT:
    1. Build current LUT from raw_deltas
    2. Apply symmetry-preserving repair (fixes FP16 duplicates)
    3. Update the lut buffer with repaired values

    This ensures the saved lut buffer is valid FP16 after loading.

    Args:
        model: Model with AnemllQATLinearV2 layers
        verbose: Print repair info

    Returns:
        Number of layers repaired
    """
    from .ane_qat_linear_v2 import (
        AnemllQATLinearV2,
        build_symmetric_lut16,
        repair_lut_duplicates_symmetric,
    )

    repaired = 0
    for name, module in model.named_modules():
        if not isinstance(module, AnemllQATLinearV2):
            continue
        if not module._lut_trainable:
            continue

        with torch.no_grad():
            # Build current LUT from raw_deltas
            trained_lut = module.get_lut().detach().cpu()

            # Apply symmetry-preserving repair
            repaired_lut = repair_lut_duplicates_symmetric(
                trained_lut,
                module._lut_max_abs,
            )

            # Update the lut buffer
            module.lut.data.copy_(repaired_lut.to(module.lut.device))

        repaired += 1
        if verbose:
            print(f"  [LUT repair] {name}")

    return repaired


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

    # Repair trainable LUTs before saving (ensures valid FP16 after load)
    lut_repaired = prepare_lut_for_save(model, verbose=verbose)

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
    train_attn_only: bool = False,
    freeze_mags: bool = False,
    freeze_mags_mlp: bool = False,
    freeze_all: bool = False,
    train_norms_only: bool = False,
    hard_top1_weight: float = 0.0,
    hard_top1_end: float = None,
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
    use_mixed_precision: bool = False,
    use_wandb: bool = False,
    wandb_project: str = "qwen3-qat",
    wandb_run_name: str = None,
    wandb_config: dict = None,
    weight_decay: float = 0.0,
    dropout: float = 0.0,
    accumulation_steps: int = 1,
    keep_checkpoints: int = 0,
    clip_grad_norm: float = 1.0,
    # Anchor KL regularization
    anchor_ckpt: str = None,
    anchor_kl_weight: float = 0.0,
    anchor_samples: int = 16,
    anchor_interval: int = 1,
    # Auto snap+freeze
    auto_snap_state = None,
    # Memory debug
    mem_debug_config = None,
    # Sparse logits mode (L1024+ TPU memory safety)
    sampled_ce_weight: float = 0.0,
    sampled_negatives: int = 64,
    no_full_logits: bool = False,
    # LUT training (per-tensor trainable LUTs)
    train_lut: bool = False,
    lut_only: bool = False,
    lut_scope: str = 'all',
    lut_max_abs: float = 1.0,
    lut_lr: Optional[float] = None,
    allow_bad_qc: bool = False,
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
        freeze_mags: If True, snap rank_magnitude to FP16-representable values and freeze them.
                     This is the opposite of train_g_only. Useful when mags are pre-snapped.
        freeze_mags_mlp: If True, freeze rank_magnitude for MLP layers only (attention mags trainable)
        freeze_all: If True, snap and freeze ALL V2 params (scale_A, scale_B, rank_magnitude).
                    Nothing trains. Use for FP16 snap verification.
        train_norms_only: If True, train ONLY LayerNorm weights (model.norm, input_layernorm,
                          post_attention_layernorm). All QAT params frozen. Good for stabilizing
                          long-context behavior without touching quantized weights.
        train_mlp_only: If True, freeze attention layers (q/k/v/o_proj) and only train MLP
                        (gate/up/down_proj). Useful for mixed-bit configs (e.g., 4-bit attn, 2-bit MLP)
        train_attn_only: If True, freeze MLP layers (gate/up/down_proj) and only train attention
                        (q/k/v/o_proj). Useful for 2-phase training: Phase 1 trains MLP, Phase 2 trains attn.
        hard_top1_weight: Weight for hard label top-1 loss (0 to disable), or start weight if annealing
        hard_top1_end: End weight for hard_top1 annealing (None = no annealing, use fixed weight)
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
        use_mixed_precision: Enable mixed precision (FP32 master weights + BF16 compute)
        use_wandb: Enable Weights & Biases logging (requires wandb package)
        wandb_project: W&B project name (default: "qwen3-qat")
        wandb_run_name: W&B run name (default: auto-generated)
        wandb_config: Additional config dict to log to W&B
        weight_decay: AdamW weight decay (default 0.0, try 0.01 for regularization)
        dropout: Dropout rate (default 0.0, try 0.1 for regularization)
        accumulation_steps: Gradient accumulation steps (default 1 = no accumulation)
                           Effective batch = batch_size * accumulation_steps
        keep_checkpoints: Keep only the last N checkpoints (0=keep all). Useful for long runs.
        clip_grad_norm: Max gradient norm for clipping (default 1.0, 0=disable). Improves stability.
        anchor_ckpt: Checkpoint to use as anchor teacher (prevents drift from reference behavior)
        anchor_kl_weight: Weight of anchor KL term (default 0.0 = disabled)
        anchor_samples: Number of fixed anchor samples to cache logits for (default 16)
        anchor_interval: Compute anchor KL every N steps (default 1 = every step)
        auto_snap_state: AutoSnapState instance for automatic snap+freeze of rank_magnitude.
                        When enabled, audits mags at save checkpoints (CPU-only) and triggers
                        one-time snap+freeze when stability detected. See auto_snap_mags.py.

    Returns:
        Dict with 'initial_loss', 'final_loss', 'best_loss', 'steps', 'time_sec'
    """
    import time
    import os
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from .mem_debug import mem_log, print_attn_info

    # Memory debug config shorthand (used throughout function)
    _mem_cfg = mem_debug_config

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
                resume="never",  # Always start fresh - avoids step conflicts on restart
            )
            # Define metrics to use train/step as X-axis
            wandb.define_metric("train/step")
            wandb.define_metric("train/*", step_metric="train/step")
            wandb.define_metric("eval/*", step_metric="train/step")
            wandb.define_metric("tpu/*", step_metric="train/step")
            wandb.define_metric("auto_snap/*", step_metric="train/step")
            if verbose:
                print(f"[wandb] Logging to: {wandb_run.url}")
        except ImportError:
            print("[wandb] Warning: wandb not installed. Install with: pip install wandb")
            use_wandb = False
        except Exception as e:
            print(f"[wandb] Warning: Failed to initialize wandb: {e}")
            use_wandb = False

    # Enable LUT training (must be done on CPU before device transfer)
    # NOTE: For TPU, enable_lut_training_all should be called BEFORE model.to(device)
    #       in the caller (e.g., train_v2_simple.py) to avoid XLA hangs
    lut_enabled = 0
    if train_lut:
        from .ane_qat_linear_v2 import AnemllQATLinearV2
        # Check if LUT training already enabled (called before device transfer)
        for m in model.modules():
            if isinstance(m, AnemllQATLinearV2) and m._lut_trainable:
                lut_enabled += 1

        if lut_enabled > 0:
            # Already enabled (good - was done on CPU before device transfer)
            if verbose:
                print(f"\n=== LUT Training (pre-enabled) ===")
                print(f"  {lut_enabled} layers ready")
        else:
            # Not enabled yet - try to enable (only works on CPU)
            model_device = next(model.parameters()).device
            if model_device.type != 'cpu':
                raise RuntimeError(
                    f"LUT training must be enabled on CPU before moving model to {model_device}. "
                    f"Call enable_lut_training_all() before model.to(device)."
                )
            from .ane_qat_linear_v2 import enable_lut_training_all
            if verbose:
                print(f"\n=== Enabling LUT Training ===")
                print(f"  Scope: {lut_scope}")
                print(f"  Max abs: {lut_max_abs}")
                print(f"  Allow bad QC: {allow_bad_qc}")
            lut_enabled = enable_lut_training_all(
                model,
                scope=lut_scope,
                max_abs=lut_max_abs,
                allow_bad_qc=allow_bad_qc,
                verbose=verbose,
            )
            if lut_enabled == 0:
                print("  WARNING: No layers had LUT training enabled!")

        if use_wandb and wandb_run is not None:
            wandb.define_metric("lut/*", step_metric="train/step")

    # Set trainable parameters
    # Validate mutually exclusive flags
    if train_mlp_only and train_attn_only:
        raise ValueError("Cannot use both --mlp-only and --attn-only. Choose one.")

    trainable = 0
    attn_frozen = 0
    mlp_frozen = 0
    mags_snapped = 0
    scales_snapped = 0
    norms_trained = 0
    lut_trainable = 0
    for p in model.parameters():
        p.requires_grad = False

    # Handle lut_only mode: only enable LUT params, skip normal training
    if lut_only and train_lut:
        for name, param in model.named_parameters():
            if '_lut_raw_deltas' in name:
                param.requires_grad = True
                trainable += param.numel()
                lut_trainable += 1
        if verbose:
            print(f"  LUT-only mode: {lut_trainable} LUT params trainable ({trainable:,} elements)")
        # Skip normal trainable param setup
        train_weights = False
        train_scales = False
        train_norms_only = False

    # Projection names for selective training
    attn_proj_names = ('q_proj', 'k_proj', 'v_proj', 'o_proj')
    mlp_proj_names = ('gate_proj', 'up_proj', 'down_proj')

    # LayerNorm names for train_norms_only mode
    norm_names = ('input_layernorm', 'post_attention_layernorm', 'model.norm')

    for name, module in model.named_modules():
        # Handle train_norms_only: train LayerNorm weights only
        if train_norms_only:
            # Check if this is a LayerNorm module
            is_norm = any(n in name for n in norm_names)
            if is_norm and hasattr(module, 'weight') and module.weight is not None:
                module.weight.requires_grad = True
                trainable += module.weight.numel()
                norms_trained += 1
            # Skip QAT param training entirely
            continue

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

            # Skip MLP layers if train_attn_only
            is_mlp_proj = any(proj in name for proj in mlp_proj_names)
            if train_attn_only and is_mlp_proj:
                mlp_frozen += module.weight.numel()
                if hasattr(module, 'scale_A') and module.scale_A is not None:
                    mlp_frozen += module.scale_A.numel() + module.scale_B.numel()
                # V2 has rank_magnitude
                if hasattr(module, 'rank_magnitude') and module.rank_magnitude is not None:
                    mlp_frozen += module.rank_magnitude.numel()
                continue  # Keep frozen

            if train_weights:
                module.weight.requires_grad = True
                trainable += module.weight.numel()
            if train_scales:
                if hasattr(module, 'scale_A') and module.scale_A is not None:
                    if freeze_all:
                        # Snap scales to FP16 and keep frozen
                        # Move to CPU for snapping (XLA/TPU .half() doesn't work correctly)
                        with torch.no_grad():
                            orig_device = module.scale_A.data.device
                            module.scale_A.data = module.scale_A.data.cpu().half().float().to(orig_device)
                            module.scale_B.data = module.scale_B.data.cpu().half().float().to(orig_device)
                        scales_snapped += 1
                    elif not train_g_only:  # Only train A/B if not G-only mode
                        module.scale_A.requires_grad = True
                        module.scale_B.requires_grad = True
                        trainable += module.scale_A.numel() + module.scale_B.numel()
                # V2 has rank_magnitude
                if hasattr(module, 'rank_magnitude') and module.rank_magnitude is not None:
                    is_mlp = any(p in name for p in ('gate_proj', 'up_proj', 'down_proj'))

                    # Determine if this mag should be frozen
                    should_freeze_mag = freeze_all or freeze_mags or (freeze_mags_mlp and is_mlp)

                    if should_freeze_mag:
                        # Snap to FP16-representable values (keep as FP32 for training stability)
                        # Move to CPU for snapping (XLA/TPU .half() doesn't work correctly)
                        with torch.no_grad():
                            orig_device = module.rank_magnitude.data.device
                            snapped = module.rank_magnitude.data.cpu().half().float()
                            module.rank_magnitude.data = snapped.to(orig_device)
                        mags_snapped += 1
                        # Keep frozen (requires_grad already False)
                    else:
                        module.rank_magnitude.requires_grad = True
                        trainable += module.rank_magnitude.numel()

    # Describe mode
    mode_parts = []
    if train_norms_only:
        mode = f"NORMS ONLY ({norms_trained} LayerNorm tensors)"
    elif freeze_all:
        mode = "FREEZE ALL (A+B+G snapped/frozen)"
    else:
        if train_weights:
            mode_parts.append("weights")
        if train_scales:
            if train_g_only:
                mode_parts.append("G-only")
            elif freeze_mags:
                mode_parts.append("A+B (mags frozen/snapped)")
            elif freeze_mags_mlp:
                mode_parts.append("scales (MLP mags frozen/snapped)")
            else:
                mode_parts.append("scales")
        mode = "+".join(mode_parts) if mode_parts else "none"
        if train_mlp_only:
            mode += " (MLP only)"
        elif train_attn_only:
            mode += " (Attn only)"

    if verbose:
        print(f"=== End-to-End KD-QAT ===")
        print(f"Mode: {mode}")
        if use_mixed_precision:
            print(f"Precision: Mixed (FP32 weights + BF16 compute)")
        elif use_fp16:
            print(f"Precision: FP16 with GradScaler")
        print(f"Trainable params: {trainable:,}")
        if train_mlp_only:
            print(f"Frozen attention params: {attn_frozen:,}")
        elif train_attn_only:
            print(f"Frozen MLP params: {mlp_frozen:,}")
        if scales_snapped > 0:
            print(f"Snapped & frozen scales: {scales_snapped} layers")
        if mags_snapped > 0:
            print(f"Snapped & frozen mags: {mags_snapped} layers")
        if accumulation_steps > 1:
            eff_batch = batch_size * accumulation_steps
            print(f"Steps: {max_steps}, LR: {lr}, Batch: {batch_size}x{accumulation_steps}={eff_batch}")
        else:
            print(f"Steps: {max_steps}, LR: {lr}, Batch: {batch_size}")
        if hard_top1_weight > 0 or hard_full_weight > 0:
            if hard_top1_end is not None:
                print(f"Hard label: top1={hard_top1_weight}→{hard_top1_end} (annealing), full={hard_full_weight}")
            else:
                print(f"Hard label: top1={hard_top1_weight}, full={hard_full_weight}")
        if clip_grad_norm > 0:
            print(f"Gradient clipping: max_norm={clip_grad_norm}")

    # Apply dropout to model if specified
    if dropout > 0:
        # Try to set dropout on model config (works for HuggingFace models)
        if hasattr(model, 'config'):
            if hasattr(model.config, 'attention_probs_dropout_prob'):
                model.config.attention_probs_dropout_prob = dropout
            if hasattr(model.config, 'hidden_dropout_prob'):
                model.config.hidden_dropout_prob = dropout
            if hasattr(model.config, 'attention_dropout'):
                model.config.attention_dropout = dropout
            if hasattr(model.config, 'hidden_dropout'):
                model.config.hidden_dropout = dropout
        if verbose:
            print(f"Dropout: {dropout}")

    # Optimizer (with optional separate LUT LR)
    if train_lut and lut_lr is not None:
        # Separate param groups for LUT and non-LUT params
        lut_params = []
        other_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if '_lut_raw_deltas' in name:
                lut_params.append(param)
            else:
                other_params.append(param)

        if not lut_params and not other_params:
            raise ValueError("No trainable parameters! Check train_weights/train_scales/train_lut flags.")

        param_groups = []
        if other_params:
            param_groups.append({'params': other_params, 'lr': lr})
        if lut_params:
            param_groups.append({'params': lut_params, 'lr': lut_lr})
            if verbose:
                print(f"LUT LR: {lut_lr} ({len(lut_params)} params)")

        optimizer = AdamW(param_groups, weight_decay=weight_decay)
        num_params = len(lut_params) + len(other_params)
    else:
        params = [p for p in model.parameters() if p.requires_grad]
        if not params:
            raise ValueError("No trainable parameters! Check train_weights/train_scales flags.")
        optimizer = AdamW(params, lr=lr, weight_decay=weight_decay)
        num_params = len(params)
    if verbose and weight_decay > 0:
        print(f"Weight decay: {weight_decay}")

    # Memory debug: after optimizer init
    if _mem_cfg:
        mem_log(_mem_cfg, 'after_optimizer_init', micro_step=0, opt_step=0, phase='init',
                extra={'num_params': num_params, 'lr': lr, 'weight_decay': weight_decay})

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

    # Initial evaluation (skip if eval_samples <= 0 or None, e.g., on TPU)
    if eval_samples and eval_samples > 0:
        model.eval()
        initial_loss = evaluate_kd_loss(model, cache_dir, device, num_samples=eval_samples, temperature=temperature)
        if verbose:
            print(f"\nInitial KD Loss: {initial_loss:.4f}")
    else:
        initial_loss = 0.0
        if verbose:
            print(f"\n[Eval skipped - eval_samples=0]")

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

    # Load anchor model and compute anchor logits if enabled
    anchor_model = None
    anchor_logits = None  # Full logits [samples, L, V] - only used if no_full_logits=False
    anchor_topk_idx = None  # Sparse top-K indices [samples, L, K] - used if no_full_logits=True
    anchor_topk_logits = None  # Sparse top-K logits [samples, L, K] - used if no_full_logits=True
    anchor_input_ids = None
    anchor_K = 128  # Number of top-K tokens to track for sparse anchor KL

    # Warn about potential XLA OOM with sparse anchor-KL on TPU
    # Works on multi-TPU with model parallelism, may OOM on single-chip v6e-1
    _is_tpu_early = 'xla' in str(device).lower()
    if no_full_logits and _is_tpu_early and anchor_ckpt and anchor_kl_weight > 0:
        if verbose:
            print(f"\n[Anchor KL] WARNING: sparse mode + TPU may cause XLA OOM on single-chip (v6e-1)")
            print(f"  Multi-TPU (v6e-4, v6e-8) with model parallelism should work")
            print(f"  If OOM occurs, use --anchor-kl-weight 0 to disable")

    if anchor_ckpt and anchor_kl_weight > 0 and anchor_samples > 0:
        if verbose:
            print(f"\n[Anchor KL] Loading anchor checkpoint: {anchor_ckpt}")
            print(f"  weight={anchor_kl_weight}, samples={anchor_samples}, interval={anchor_interval}")

        # Build anchor model with same architecture as training model
        from transformers import AutoModelForCausalLM
        from .ane_qat_linear_v2 import replace_linear_with_anemll_v2, AnemllQuantConfigV2

        # Get model config from training model
        model_config = model.config
        model_id = getattr(model_config, '_name_or_path', 'Qwen/Qwen3-0.6B')

        # Create a fresh model for anchor (in FP32 for precision)
        anchor_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )

        # Detect V2 config from training model
        v2_config = None
        for name, m in model.named_modules():
            if hasattr(m, 'lut_size') and hasattr(m, 'scale_rank'):
                # Found a V2 layer, get config
                from .ane_qat_linear_v2 import AnemllQuantConfigV2
                is_mlp = any(p in name for p in ['gate_proj', 'up_proj', 'down_proj'])
                if is_mlp:
                    v2_config = AnemllQuantConfigV2(
                        lut_size=m.lut_size,
                        scale_rank=m.scale_rank,
                        use_ste_fp16=getattr(m, 'use_ste_fp16', True),
                    )
                break

        if v2_config is None:
            # Fallback to default config
            v2_config = AnemllQuantConfigV2(lut_size=16, scale_rank=32, use_ste_fp16=True)

        # Replace with V2 layers
        replace_linear_with_anemll_v2(anchor_model, v2_config)

        # Load anchor checkpoint
        anchor_state = torch.load(anchor_ckpt, map_location='cpu')
        if isinstance(anchor_state, dict) and 'model_state_dict' in anchor_state:
            anchor_state = anchor_state['model_state_dict']
        anchor_model.load_state_dict(anchor_state, strict=False)

        # Freeze and move to device
        anchor_model.to(device)
        anchor_model.eval()
        for p in anchor_model.parameters():
            p.requires_grad = False

        # Get anchor samples from KD cache
        anchor_batches = []
        temp_dataset = KDCacheDataset(cache_dir, shuffle=False, preload=True)
        temp_iter = iter(DataLoader(temp_dataset, batch_size=1, shuffle=False))
        while len(anchor_batches) < anchor_samples:
            try:
                batch = next(temp_iter)
                anchor_batches.append(batch['input_ids'].squeeze(0))
            except StopIteration:
                temp_iter = iter(DataLoader(temp_dataset, batch_size=1, shuffle=False))
        anchor_input_ids = torch.stack(anchor_batches[:anchor_samples], dim=0).to(device)
        # Truncate anchor_input_ids to seq_len for consistent XLA shapes
        if seq_len > 0 and anchor_input_ids.size(1) > seq_len:
            anchor_input_ids = anchor_input_ids[:, :seq_len].contiguous()

        # Compute anchor logits (sparse top-K if no_full_logits, otherwise full)
        if verbose:
            print(f"  Computing anchor logits: {anchor_input_ids.shape}...", end=" ", flush=True)
        with torch.no_grad():
            full_logits = anchor_model(anchor_input_ids, use_cache=False).logits.detach()
            if no_full_logits:
                # Extract top-K and discard full logits (memory-safe for L1024)
                anchor_topk_logits, anchor_topk_idx = torch.topk(full_logits, k=anchor_K, dim=-1)
                anchor_topk_logits = anchor_topk_logits.to(torch.bfloat16)  # Save memory
                anchor_topk_idx = anchor_topk_idx.to(torch.int32)
                del full_logits  # Free the [samples, L, V] tensor immediately
                if verbose:
                    print(f"done (sparse): topk_idx={anchor_topk_idx.shape}, topk_logits={anchor_topk_logits.shape}")
            else:
                anchor_logits = full_logits
                if verbose:
                    print(f"done: {anchor_logits.shape}")

        # Memory debug: after anchor model init (BEFORE freeing, shows peak with anchor model)
        if _mem_cfg:
            # Use sparse shape if no_full_logits, otherwise full logits shape
            logits_shape = list(anchor_topk_logits.shape) if no_full_logits else list(anchor_logits.shape)
            mem_log(_mem_cfg, 'after_anchor_init', micro_step=0, opt_step=0, phase='init',
                    extra={'anchor_samples': anchor_samples, 'anchor_logits_shape': logits_shape,
                           'anchor_mode': 'sparse' if no_full_logits else 'full'})

        # Free anchor model memory (we only need the cached logits)
        del anchor_model
        anchor_model = None
        import gc
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # Training loop
    model.train()
    step = 0
    total_loss = 0.0
    # When eval is disabled, initial_loss=0.0 which breaks best tracking
    # Use inf so first training loss becomes the baseline
    best_loss = initial_loss if initial_loss > 0 else float('inf')
    best_state = None
    loss_history = []
    seq_len = None  # Will be set from first batch
    last_log_time = time.time()  # For tokens/sec calculation

    # TPU detection (once, not every step)
    is_tpu = 'xla' in str(device).lower()
    xm = None
    pl = None  # parallel_loader
    if is_tpu:
        try:
            import torch_xla.core.xla_model as xm
            import torch_xla.distributed.parallel_loader as pl
        except ImportError:
            is_tpu = False

    # Create dataset ONCE with preload for fast training
    # (avoids torch.load I/O every batch)
    dataset = KDCacheDataset(cache_dir, shuffle=True, preload=True)

    # TPU warmup: precompile XLA graph before timing starts
    if is_tpu and verbose:
        print("\n[TPU] Warmup: compiling XLA graph...", end=" ", flush=True)
        warmup_t0 = time.time()

        # Get one batch for warmup
        warmup_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, drop_last=True)
        warmup_batch = next(iter(warmup_loader))
        warmup_seq_len = warmup_batch['input_ids'].shape[1]

        # Truncate warmup batch if L>512 to reduce XLA compilation memory
        # (XLA backward graph for L=1024 needs ~200MB compilation workspace)
        # Training will trigger one more recompile for full L, but that's OK
        warmup_max_len = 512
        if warmup_seq_len > warmup_max_len:
            print(f"(truncating L={warmup_seq_len}→{warmup_max_len} for compile) ", end="", flush=True)
            warmup_batch = {
                k: v[:, :warmup_max_len] if v.dim() >= 2 and v.size(1) == warmup_seq_len else v
                for k, v in warmup_batch.items()
            }
            warmup_seq_len = warmup_max_len

        # Memory debug: before warmup compile
        if _mem_cfg:
            print_attn_info(model)
            mem_log(_mem_cfg, 'before_warmup_compile', micro_step=0, opt_step=0, phase='warmup',
                    batch_size=batch_size, seq_len=warmup_seq_len)

        # Run one forward+backward pass to trigger compilation
        # CRITICAL: Use EXACT same loss path as training (same args) to avoid recompilation
        model.train()
        autocast_device = 'xla'
        print("forward...", end=" ", flush=True)
        if use_mixed_precision:
            with torch.amp.autocast(device_type=autocast_device, dtype=torch.bfloat16):
                warmup_loss = compute_kd_loss_batch(
                    model, warmup_batch, device, temperature,
                    no_grad=False,
                    hard_top1_weight=hard_top1_weight,
                    hard_full_weight=hard_full_weight,
                    sampled_ce_weight=sampled_ce_weight,
                    sampled_negatives=sampled_negatives,
                )
        else:
            warmup_loss = compute_kd_loss_batch(
                model, warmup_batch, device, temperature,
                no_grad=False,
                hard_top1_weight=hard_top1_weight,
                hard_full_weight=hard_full_weight,
                sampled_ce_weight=sampled_ce_weight,
                sampled_negatives=sampled_negatives,
            )

        # Memory debug: after first forward
        if _mem_cfg:
            mem_log(_mem_cfg, 'after_first_forward', micro_step=0, opt_step=0, phase='warmup',
                    batch_size=batch_size, seq_len=warmup_seq_len)

        # Memory debug: before mark_step
        if _mem_cfg:
            mem_log(_mem_cfg, 'before_mark_step', micro_step=0, opt_step=0, phase='warmup',
                    batch_size=batch_size, seq_len=warmup_seq_len)

        if xm is not None:
            xm.mark_step()

        # Memory debug: after mark_step
        if _mem_cfg:
            mem_log(_mem_cfg, 'after_mark_step', micro_step=0, opt_step=0, phase='warmup',
                    batch_size=batch_size, seq_len=warmup_seq_len)

        print("backward...", end=" ", flush=True)
        warmup_loss.backward()

        # Memory debug: after backward
        if _mem_cfg:
            mem_log(_mem_cfg, 'after_first_backward', micro_step=0, opt_step=0, phase='warmup',
                    batch_size=batch_size, seq_len=warmup_seq_len)

        if xm is not None:
            xm.mark_step()

        # Do a FULL optimizer step to precompile optimizer graph
        # (This was missing - caused hang at first real optimizer step)
        print("optimizer...", end=" ", flush=True)
        optimizer.step()
        if xm is not None:
            xm.mark_step()

        # Clear gradients and reset optimizer state
        optimizer.zero_grad()

        # Reset optimizer state (we don't want warmup to affect training)
        # AdamW stores exp_avg and exp_avg_sq per parameter
        optimizer.state.clear()

        warmup_time = time.time() - warmup_t0
        print(f"done ({warmup_time:.1f}s)")

        # Report TPU memory after warmup
        try:
            mem = xm.get_memory_info(device)
            if "kb_total" in mem:
                total_gb = mem["kb_total"] / 1024 / 1024
                free_gb = mem["kb_free"] / 1024 / 1024
                used_gb = total_gb - free_gb
            elif "bytes_limit" in mem:
                total_gb = mem["bytes_limit"] / 1024**3
                used_gb = mem["bytes_used"] / 1024**3
                free_gb = total_gb - used_gb
            else:
                total_gb = used_gb = free_gb = 0

            if total_gb > 0:
                print(f"[TPU] Memory: {used_gb:.1f}/{total_gb:.1f} GB used ({100*used_gb/total_gb:.0f}%)")
                # Estimate max batch based on memory usage
                if used_gb > 0:
                    headroom = free_gb / used_gb
                    est_max_batch = int(batch_size * (1 + headroom * 0.8))  # 80% safety margin
                    print(f"[TPU] Estimated max batch: ~{est_max_batch} (current: {batch_size})")
        except Exception as e:
            print(f"[TPU] Memory info not available: {e}")

        print(f"[TPU] XLA compilation complete. Training t/s will be accurate.")

        # Memory debug: after warmup compile
        if _mem_cfg:
            mem_log(_mem_cfg, 'after_warmup_compile', micro_step=0, opt_step=0, phase='warmup',
                    batch_size=batch_size, seq_len=warmup_seq_len)

        # Reset timing for accurate t/s measurement
        t_start = time.time()
        last_log_time = time.time()

    # Initialize gradients before training loop
    # TPU: use set_to_none=False to avoid grad None->Tensor recompilation on step 2
    optimizer.zero_grad(set_to_none=not is_tpu)

    # With accumulation, we need max_steps * accumulation_steps micro-batches
    # to achieve max_steps optimizer steps
    total_micro_steps = max_steps * accumulation_steps
    optimizer_step = 0  # Track optimizer steps for display

    # For smoothed ETA: track (optimizer_step, time) pairs from last N logs
    # This avoids XLA compile time skewing the ETA
    from collections import deque
    eta_history = deque(maxlen=5)  # Keep last 5 log points for smoothing

    while step < total_micro_steps:
        # Just create new iterator each epoch (data already in RAM)
        # drop_last=True for TPU to avoid shape changes on last batch
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn,
                                drop_last=is_tpu)

        # NOTE: MpDeviceLoader disabled - causes hangs with our setup
        # Just use regular DataLoader with xm.optimizer_step()
        # if is_tpu and pl is not None:
        #     dataloader = pl.MpDeviceLoader(dataloader, device, batches_per_execution=4)

        for batch in dataloader:
            if step >= total_micro_steps:
                break

            # Get seq_len from first batch and print estimated total tokens
            if seq_len is None and 'input_ids' in batch:
                seq_len = batch['input_ids'].shape[1]
                total_tokens_est = batch_size * seq_len * max_steps * accumulation_steps
                if total_tokens_est >= 1e9:
                    tok_str = f"{total_tokens_est/1e9:.2f}B"
                elif total_tokens_est >= 1e6:
                    tok_str = f"{total_tokens_est/1e6:.1f}M"
                else:
                    tok_str = f"{total_tokens_est/1e3:.0f}K"
                if verbose:
                    print(f"Estimated total tokens: {tok_str} (batch={batch_size}x{accumulation_steps}, seq={seq_len}, steps={max_steps})")

                # TPU debug: ALWAYS print dtypes on first step
                if 'xla' in str(device).lower():
                    model_dtype = next(model.parameters()).dtype
                    print(f"[TPU DEBUG] model dtype: {model_dtype}, device: {device}")

                # Show that training is starting
                print(f"\nTraining:", flush=True)
                # DEBUG: Show checkpoint save settings
                print(f"  [CHECKPOINT CONFIG] save_steps={save_steps}, save_dir={save_dir!r}, accumulation_steps={accumulation_steps}", flush=True)

            # Track optimizer step timing (for debugging XLA compilation)
            # Reset timer at start of each optimizer step (not each micro-batch)
            if step % accumulation_steps == 0:
                opt_step_start_time = time.time()

            # Forward pass with optional autocast for FP16 or mixed precision
            # Note: TPU/XLA uses 'xla' device type for autocast
            autocast_device = 'xla' if is_tpu else device.type

            # DEBUG: Print before forward pass to diagnose hangs
            if step < 2 and verbose:
                print(f"  [DEBUG] step={step}, starting forward pass...", flush=True)

            # Calculate current hard_top1 (with optional annealing)
            if hard_top1_end is not None:
                # Linear decay from hard_top1_weight to hard_top1_end
                progress = min(1.0, optimizer_step / max(1, max_steps))
                current_hard_top1 = hard_top1_weight + (hard_top1_end - hard_top1_weight) * progress
            else:
                current_hard_top1 = hard_top1_weight

            if use_fp16:
                with torch.amp.autocast(device_type=autocast_device, dtype=torch.float16):
                    loss = compute_kd_loss_batch(
                        model, batch, device, temperature,
                        no_grad=False,
                        hard_top1_weight=current_hard_top1,
                        hard_full_weight=hard_full_weight,
                        sampled_ce_weight=sampled_ce_weight,
                        sampled_negatives=sampled_negatives,
                        debug_step=step if step < 1 else -1,  # Debug first step
                    )
            elif use_mixed_precision:
                # Mixed precision: FP32 master weights + BF16 compute
                with torch.amp.autocast(device_type=autocast_device, dtype=torch.bfloat16):
                    loss = compute_kd_loss_batch(
                        model, batch, device, temperature,
                        no_grad=False,
                        hard_top1_weight=current_hard_top1,
                        hard_full_weight=hard_full_weight,
                        sampled_ce_weight=sampled_ce_weight,
                        sampled_negatives=sampled_negatives,
                        debug_step=step if step < 1 else -1,  # Debug first step
                    )
            else:
                loss = compute_kd_loss_batch(
                    model, batch, device, temperature,
                    no_grad=False,
                    hard_top1_weight=current_hard_top1,
                    hard_full_weight=hard_full_weight,
                    sampled_ce_weight=sampled_ce_weight,
                    sampled_negatives=sampled_negatives,
                    debug_step=step if step < 1 else -1,  # Debug first step
                )

            # Optional anchor KL regularizer (prevents drift from reference checkpoint)
            # Supports two modes: full logits (anchor_logits) or sparse top-K (anchor_topk_idx/anchor_topk_logits)
            has_anchor = anchor_kl_weight > 0 and anchor_input_ids is not None and (
                anchor_logits is not None or (anchor_topk_idx is not None and anchor_topk_logits is not None)
            )
            if has_anchor:
                # Only compute anchor KL every anchor_interval steps
                if (optimizer_step + 1) % anchor_interval == 0 or optimizer_step == 0:
                    # Use pre-computed anchor samples for KL (cycle through them)
                    anchor_sample_idx = optimizer_step % anchor_samples
                    anchor_input = anchor_input_ids[anchor_sample_idx:anchor_sample_idx+1]  # Already on device

                    if no_full_logits and anchor_topk_idx is not None:
                        # Sparse anchor KL: compute hidden states, gather top-K logits
                        # This avoids materializing [1, L, V] tensor
                        # CHUNKED to avoid XLA OOM on TPU (L*K gather is too large)
                        if use_fp16:
                            with torch.amp.autocast(device_type=autocast_device, dtype=torch.float16):
                                hidden = model.model(anchor_input, use_cache=False).last_hidden_state
                        elif use_mixed_precision:
                            with torch.amp.autocast(device_type=autocast_device, dtype=torch.bfloat16):
                                hidden = model.model(anchor_input, use_cache=False).last_hidden_state
                        else:
                            hidden = model.model(anchor_input, use_cache=False).last_hidden_state

                        # Remove last position to align with topk tensors (which are L-1 for next-token prediction)
                        hidden = hidden[:, :-1, :]

                        # Get dimensions
                        B_anc, L_anc, H_anc = hidden.shape
                        idx = anchor_topk_idx[anchor_sample_idx].to(device).long()  # [L-1, K]
                        K_anc = idx.size(1)
                        anchor_topk = anchor_topk_logits[anchor_sample_idx].to(device).float()  # [L-1, K]

                        # Process in chunks to avoid XLA OOM (gather L*K rows is too large)
                        # For L=1024, K=128: full gather = 131K rows = 512MB
                        # With chunk=64: gather = 8K rows = 32MB per chunk
                        anchor_chunk_size = 64  # Positions per chunk
                        chunk_losses = []

                        for chunk_start in range(0, L_anc, anchor_chunk_size):
                            chunk_end = min(chunk_start + anchor_chunk_size, L_anc)
                            chunk_len = chunk_end - chunk_start

                            # Get hidden states and indices for this chunk
                            h_chunk = hidden[0, chunk_start:chunk_end]  # [chunk, H]
                            idx_chunk = idx[chunk_start:chunk_end]  # [chunk, K]

                            # Gather lm_head weights for this chunk's top-K indices
                            w_chunk = model.lm_head.weight[idx_chunk.reshape(-1)]  # [chunk*K, H]
                            w_chunk = w_chunk.reshape(chunk_len, K_anc, H_anc)  # [chunk, K, H]

                            # Compute sparse logits: einsum('ch,ckh->ck')
                            student_chunk = torch.einsum('ch,ckh->ck', h_chunk, w_chunk)  # [chunk, K]

                            # Get anchor logits for this chunk
                            teacher_chunk = anchor_topk[chunk_start:chunk_end]  # [chunk, K]

                            # Compute KL loss for this chunk
                            teacher_probs = F.softmax(teacher_chunk, dim=-1)
                            teacher_log_probs = F.log_softmax(teacher_chunk, dim=-1)
                            student_log_probs = F.log_softmax(student_chunk, dim=-1)
                            chunk_kl = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1).mean()
                            chunk_losses.append(chunk_kl)

                        # Average loss across all chunks
                        anchor_loss = sum(chunk_losses) / len(chunk_losses)

                    else:
                        # Full logits mode (original behavior)
                        if use_fp16:
                            with torch.amp.autocast(device_type=autocast_device, dtype=torch.float16):
                                current_logits = model(anchor_input, use_cache=False).logits
                        elif use_mixed_precision:
                            with torch.amp.autocast(device_type=autocast_device, dtype=torch.bfloat16):
                                current_logits = model(anchor_input, use_cache=False).logits
                        else:
                            current_logits = model(anchor_input, use_cache=False).logits

                        # Soft cross-entropy (avoids aten::kl_div for TPU/XLA compatibility)
                        anchor_loss = kd_soft_ce(current_logits, anchor_logits[anchor_sample_idx:anchor_sample_idx+1], temperature=1.0)

                    loss = loss + anchor_kl_weight * anchor_loss

            # Scale loss for gradient accumulation
            if accumulation_steps > 1:
                loss = loss / accumulation_steps

            # DEBUG: Print after forward pass (no .item() - avoid XLA sync before backward)
            if step < 2 and verbose:
                print(f"  [DEBUG] step={step}, forward done, starting backward...", flush=True)

            # TPU: mark_step before backward (matches warmup pattern)
            # This executes the forward graph before starting backward
            if is_tpu and xm is not None:
                xm.mark_step()

            # Backward pass with optional scaler for FP16
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # LUT gradient check after first backward (verify gradients flow)
            if step == 0 and train_lut and lut_enabled > 0 and verbose:
                max_lut_grad = 0.0
                lut_grad_count = 0
                for name, param in model.named_parameters():
                    if '_lut_raw_deltas' in name and param.grad is not None:
                        grad_max = param.grad.abs().max().item()
                        if grad_max > max_lut_grad:
                            max_lut_grad = grad_max
                        lut_grad_count += 1
                if lut_grad_count > 0:
                    if max_lut_grad < 1e-10:
                        print(f"  [LUT GRAD CHECK] WARNING: max|grad|={max_lut_grad:.2e} (~0!) - LUT may not be in forward path")
                    else:
                        print(f"  [LUT GRAD CHECK] OK: max|grad|={max_lut_grad:.2e} ({lut_grad_count} params)")
                else:
                    print(f"  [LUT GRAD CHECK] WARNING: No _lut_raw_deltas grads found!")

            # Optimizer step only every accumulation_steps
            if (step + 1) % accumulation_steps == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    if clip_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Gradient clipping for stability (all precisions)
                    if clip_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
                    optimizer.step()

                if scheduler is not None:
                    scheduler.step()
                # TPU: use set_to_none=False to avoid grad None->Tensor recompilation
                optimizer.zero_grad(set_to_none=not is_tpu)
                optimizer_step += 1  # Track for display

            # TPU: mark_step to execute graph (required for progress)
            if is_tpu and xm is not None:
                xm.mark_step()

                # Progress indicator with timing (helpful for debugging XLA compilation)
                # Only print after completing an optimizer step (not every micro-batch)
                just_did_opt_step = (step + 1) % accumulation_steps == 0
                if verbose and optimizer_step <= 5 and just_did_opt_step:
                    import torch_xla
                    torch_xla.sync()  # Wait for actual execution (XLA ops are async)
                    step_time = time.time() - opt_step_start_time

                    # Get compilation count from XLA metrics
                    compile_info = ""
                    try:
                        import torch_xla.debug.metrics as met
                        compile_count = met.counter_value('UncachedCompile') or 0
                        compile_info = f" [compiles: {compile_count}]"
                    except Exception:
                        pass
                    print(f"  step {optimizer_step}: {step_time:.1f}s{compile_info}", flush=True)

            # Track loss - always sync (simpler, proven to work)
            # Note: loss was scaled for accumulation, so multiply back for logging
            loss_val = loss.item() * accumulation_steps if accumulation_steps > 1 else loss.item()
            total_loss += loss_val
            step += 1

            # Logging - every logging_steps optimizer steps
            # With accumulation, log when optimizer_step is a multiple of logging_steps
            log_interval = logging_steps * accumulation_steps  # micro-batches between logs
            if step % log_interval == 0 and step > 0:
                avg_loss = total_loss / log_interval
                elapsed = time.time() - t_start
                current_time = time.time()

                # Smoothed ETA: use recent history to avoid XLA compile time skew
                eta_history.append((optimizer_step, current_time))
                if len(eta_history) >= 2:
                    # Use oldest point in history for rate calculation
                    old_step, old_time = eta_history[0]
                    steps_done = optimizer_step - old_step
                    time_taken = current_time - old_time
                    if steps_done > 0 and time_taken > 0:
                        sec_per_step = time_taken / steps_done
                        remaining_steps = max_steps - optimizer_step
                        eta = sec_per_step * remaining_steps
                    else:
                        eta = 0
                else:
                    # Fallback to simple calculation for first log
                    eta = elapsed / optimizer_step * (max_steps - optimizer_step) if optimizer_step > 0 else 0

                # Format time as M:SS or H:MM:SS
                def fmt_time(s):
                    if s < 3600:
                        return f"{int(s)//60}:{int(s)%60:02d}"
                    return f"{int(s)//3600}:{(int(s)%3600)//60:02d}:{int(s)%60:02d}"
                # Get current LR
                current_lr = scheduler.get_last_lr()[0] if scheduler is not None else lr
                # Calculate throughput using recent window (more accurate after warmup)
                if len(eta_history) >= 2:
                    old_step, old_time = eta_history[0]
                    recent_elapsed = current_time - old_time
                    recent_micro_steps = step - (old_step * accumulation_steps)
                    it_per_sec = recent_micro_steps / recent_elapsed if recent_elapsed > 0 else 0
                else:
                    it_per_sec = step / elapsed if elapsed > 0 else 0
                tok_per_sec = it_per_sec * batch_size * (seq_len or 0)
                throughput_str = f"{tok_per_sec/1000:.1f}k tok/s" if seq_len else f"{it_per_sec:.2f} it/s"
                # Print if verbose (show optimizer_step / max_steps)
                if verbose:
                    if scheduler is not None:
                        print(f"[{optimizer_step}/{max_steps}] loss={avg_loss:.4f} lr={current_lr:.2e} ({fmt_time(elapsed)}, ETA {fmt_time(eta)}, {throughput_str})")
                    else:
                        print(f"[{optimizer_step}/{max_steps}] loss={avg_loss:.4f} ({fmt_time(elapsed)}, ETA {fmt_time(eta)}, {throughput_str})")
                # Write to CSV
                if csv_writer is not None:
                    csv_writer.writerow([step, f'{avg_loss:.6f}', '', f'{current_lr:.2e}', f'{elapsed:.1f}'])
                    csv_file.flush()

                # TPU: Print metrics report at first logging step (debug recompilation)
                if is_tpu and step == log_interval and verbose:
                    try:
                        import torch_xla.debug.metrics as met
                        print("\n[TPU METRICS] (look for 'CompileTime' frequency)")
                        report = met.metrics_report()
                        # Only print key lines
                        for line in report.split('\n'):
                            if 'Compile' in line or 'Transfer' in line or 'Execute' in line:
                                print(f"  {line}")
                        print()
                    except Exception:
                        pass
                # Log to wandb
                if use_wandb and wandb_run is not None:
                    import wandb
                    # Calculate tokens/sec (tokens processed since last log)
                    log_elapsed = time.time() - last_log_time
                    tokens_per_sec = (batch_size * seq_len * log_interval) / max(log_elapsed, 0.001)
                    log_dict = {
                        'train/loss': avg_loss,
                        'train/lr': current_lr,
                        'train/step': optimizer_step,
                        'train/elapsed_sec': elapsed,
                        'train/tokens_per_sec': tokens_per_sec,
                    }
                    # Add hard_top1 if annealing
                    if hard_top1_end is not None:
                        log_dict['train/hard_top1'] = current_hard_top1
                    # Add best_loss when tracking by training loss (eval disabled)
                    if not eval_samples or eval_samples <= 0:
                        log_dict['train/best_loss'] = best_loss
                    # Add TPU memory stats if available
                    if is_tpu and xm is not None:
                        try:
                            # Try new API first (no device arg, returns object)
                            mem = xm.get_memory_info(device)
                            used_gb = 0
                            total_gb = 0
                            if hasattr(mem, 'bytes_limit'):
                                # New API: object with bytes_limit/bytes_used attributes
                                total_gb = getattr(mem, 'bytes_limit', 0) / 1e9
                                used_gb = getattr(mem, 'bytes_used', 0) / 1e9
                            elif isinstance(mem, dict):
                                # Old API: dict with kb_total or bytes_used
                                if "kb_total" in mem:
                                    used_gb = (mem["kb_total"] - mem.get("kb_free", 0)) / 1024 / 1024
                                    total_gb = mem["kb_total"] / 1024 / 1024
                                elif "bytes_used" in mem:
                                    used_gb = mem["bytes_used"] / 1e9
                                    total_gb = mem.get("bytes_limit", 0) / 1e9
                            # Log to wandb
                            log_dict['tpu/memory_used_gb'] = used_gb
                            log_dict['tpu/memory_total_gb'] = total_gb
                            if total_gb > 0:
                                log_dict['tpu/memory_pct'] = 100.0 * used_gb / total_gb
                        except Exception as e:
                            # Log first failure for debugging
                            if optimizer_step == log_interval:
                                print(f"  [TPU memory] get_memory_info failed: {e}")
                    # Add LUT metrics if LUT training enabled
                    if train_lut and lut_enabled > 0:
                        lut_metrics = compute_lut_metrics(model)
                        log_dict.update(lut_metrics)
                    wandb.log(log_dict, step=optimizer_step)
                last_log_time = time.time()
                loss_history.append(avg_loss)
                total_loss = 0.0

                # Track best by training loss when eval is disabled (e.g., TPU)
                if (not eval_samples or eval_samples <= 0) and avg_loss < best_loss:
                    best_loss = avg_loss
                    # Save best state
                    if save_dir:
                        os.makedirs(save_dir, exist_ok=True)
                        best_path = os.path.join(save_dir, "best_state_dict.pt")
                        torch.save(model.state_dict(), best_path)
                        if verbose:
                            print(f"  [Saved best (train): {best_loss:.4f}]")

            # Evaluation (skip if eval_samples <= 0 or None, e.g., on TPU)
            # Check every eval_steps optimizer steps
            eval_interval = eval_steps * accumulation_steps
            if eval_interval > 0 and step % eval_interval == 0 and step > 0 and eval_samples and eval_samples > 0:
                model.eval()
                eval_loss = evaluate_kd_loss(model, cache_dir, device, num_samples=eval_samples, temperature=temperature)
                elapsed = time.time() - t_start
                current_lr = scheduler.get_last_lr()[0] if scheduler is not None else lr
                if verbose:
                    print(f"  [Eval] KD Loss: {eval_loss:.4f} (best: {best_loss:.4f})")
                # Write eval to CSV
                if csv_writer is not None:
                    csv_writer.writerow([optimizer_step, '', f'{eval_loss:.6f}', f'{current_lr:.2e}', f'{elapsed:.1f}'])
                    csv_file.flush()
                # Log to wandb
                if use_wandb and wandb_run is not None:
                    import wandb
                    wandb.log({
                        'eval/loss': eval_loss,
                        'eval/best_loss': best_loss,
                    }, step=optimizer_step)
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

            # Periodic checkpoint saving (every save_steps optimizer steps)
            save_interval = save_steps * accumulation_steps if save_steps > 0 else 0
            # DEBUG: Check checkpoint save conditions at potential save points
            if save_interval > 0 and step % save_interval == 0 and step > 0:
                print(f"  [CHECKPOINT DEBUG] step={step}, save_interval={save_interval}, save_dir={save_dir!r}, all conditions met={bool(save_dir)}")
            if save_interval > 0 and save_dir and step % save_interval == 0 and step > 0:
                os.makedirs(save_dir, exist_ok=True)
                ckpt_path = os.path.join(save_dir, f"checkpoint_step{optimizer_step}.pt")
                # Save CPU state dict (also used for auto-snap audit)
                cpu_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                torch.save(cpu_state_dict, ckpt_path)
                if verbose:
                    print(f"  [Checkpoint saved: {ckpt_path}]")

                # Memory debug: on checkpoint save
                if _mem_cfg:
                    mem_log(_mem_cfg, 'on_checkpoint_save', micro_step=step, opt_step=optimizer_step, phase='save',
                            batch_size=batch_size, seq_len=seq_len if seq_len else 128)

                # Auto-snap audit (CPU-only, no XLA tensor reads)
                if auto_snap_state is not None and auto_snap_state.should_audit(optimizer_step):
                    from qat_lora.auto_snap_mags import (
                        audit_mags_movement,
                        apply_auto_snap_and_freeze,
                        save_audit_json,
                    )
                    audit_result = audit_mags_movement(
                        auto_snap_state,
                        cpu_state_dict,
                        optimizer_step,
                        verbose=verbose,
                    )
                    # Log to W&B if enabled
                    if use_wandb and wandb_run is not None:
                        import wandb
                        snap_log = {'auto_snap/step': optimizer_step}
                        if audit_result['movement_metrics']:
                            snap_log['auto_snap/max_delta'] = audit_result['movement_metrics']['max_abs_delta']
                            snap_log['auto_snap/mean_delta'] = audit_result['movement_metrics']['mean_abs_delta']
                            snap_log['auto_snap/num_keys'] = audit_result['movement_metrics']['num_keys']
                        if audit_result['snap_metrics']:
                            snap_log['auto_snap/fp16_snap_dist'] = audit_result['snap_metrics']['max_snap_diff']
                        snap_log['auto_snap/stable_count'] = auto_snap_state.stable_count
                        snap_log['auto_snap/auto_frozen'] = int(auto_snap_state.auto_frozen)
                        # Log disabled state if aborted
                        snap_log['auto_snap/enabled'] = int(auto_snap_state.enabled)
                        if audit_result.get('disabled'):
                            snap_log['auto_snap/disabled'] = 1
                            snap_log['auto_snap/disable_reason'] = audit_result.get('disable_reason', 'unknown')
                        wandb.log(snap_log, step=optimizer_step)

                    # Save audit JSON if enabled
                    if auto_snap_state.log_json:
                        save_audit_json(auto_snap_state, save_dir, optimizer_step)

                    # Apply snap+freeze if triggered
                    if audit_result['should_freeze']:
                        # Capture current LR and scheduler state before rebuild
                        lr_before = optimizer.param_groups[0]['lr']
                        sched_state = scheduler.state_dict() if scheduler else None

                        frozen_count, optimizer = apply_auto_snap_and_freeze(
                            model,
                            optimizer,
                            target=auto_snap_state.target,
                            verbose=verbose,
                        )
                        auto_snap_state.auto_frozen = True

                        # Rebuild scheduler with new optimizer (critical for LR continuity)
                        # Use state_dict restore for robustness (handles base_lrs correctly)
                        if scheduler is not None and sched_state is not None:
                            from torch.optim.lr_scheduler import LambdaLR
                            # Recreate with same lr_lambda closure (captured in outer scope)
                            scheduler = LambdaLR(optimizer, lr_lambda)
                            # Restore full state (includes last_epoch, base_lrs)
                            scheduler.load_state_dict(sched_state)
                            # Force LR update to current step
                            scheduler._last_lr = [lr_lambda(sched_state['last_epoch']) * base_lr
                                                  for base_lr in scheduler.base_lrs]
                            for param_group, lr in zip(optimizer.param_groups, scheduler._last_lr):
                                param_group['lr'] = lr

                            # Verify LR continuity
                            lr_after = optimizer.param_groups[0]['lr']
                            if verbose:
                                print(f"[AutoSnap] Scheduler rebuilt: step={sched_state['last_epoch']}, lr={lr_after:.2e}")
                                lr_diff = abs(lr_before - lr_after)
                                if lr_diff > 1e-8:
                                    print(f"[AutoSnap] WARNING: LR changed from {lr_before:.2e} to {lr_after:.2e} (diff={lr_diff:.2e})")
                                else:
                                    print(f"[AutoSnap] LR continuity verified ✓")

                        # Log to W&B
                        if use_wandb and wandb_run is not None:
                            import wandb
                            wandb.log({
                                'auto_snap/frozen_step': optimizer_step,
                                'auto_snap/frozen_count': frozen_count,
                                'auto_snap/lr_before': lr_before,
                                'auto_snap/lr_after': optimizer.param_groups[0]['lr'],
                            }, step=optimizer_step)

                        # Memory debug: after autosnap freeze (optimizer rebuild may trigger recompile)
                        if _mem_cfg:
                            mem_log(_mem_cfg, 'after_autosnap_freeze', micro_step=step, opt_step=optimizer_step, phase='save',
                                    batch_size=batch_size, seq_len=seq_len if seq_len else 128,
                                    extra={'frozen_count': frozen_count})

                # Clean up old checkpoints if keep_checkpoints is set
                if keep_checkpoints > 0:
                    import glob
                    ckpt_pattern = os.path.join(save_dir, "checkpoint_step*.pt")
                    ckpts = sorted(glob.glob(ckpt_pattern), key=os.path.getmtime)
                    while len(ckpts) > keep_checkpoints:
                        old_ckpt = ckpts.pop(0)
                        try:
                            os.remove(old_ckpt)
                            if verbose:
                                print(f"  [Removed old checkpoint: {os.path.basename(old_ckpt)}]")
                        except OSError:
                            pass

    # Final evaluation (skip if eval_samples <= 0 or None, e.g., on TPU)
    elapsed = time.time() - t_start
    if eval_samples and eval_samples > 0:
        model.eval()
        final_loss = evaluate_kd_loss(model, cache_dir, device, num_samples=eval_samples, temperature=temperature)

        # Write final eval to CSV
        if csv_writer is not None:
            current_lr = scheduler.get_last_lr()[0] if scheduler is not None else lr
            csv_writer.writerow([step, '', f'{final_loss:.6f}', f'{current_lr:.2e}', f'{elapsed:.1f}'])
    else:
        final_loss = 0.0
        if verbose:
            print(f"\n[Final eval skipped - eval_samples=0]")

    if csv_writer is not None:
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
        }, step=optimizer_step)

    # Update best if final is better (skip if no eval)
    if eval_samples and eval_samples > 0 and final_loss < best_loss:
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
        if eval_samples and eval_samples > 0:
            print(f"Final:   {final_loss:.4f}")
            print(f"Best:    {best_loss:.4f} (eval)")
            print(f"Improvement: {initial_loss - final_loss:.4f}")
        else:
            print(f"Final:   {loss_history[-1]:.4f}" if loss_history else "Final:   N/A")
            print(f"Best:    {best_loss:.4f} (train)")
            print(f"Improvement: {initial_loss - best_loss:.4f}")
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

        # TPU: Print compilation summary
        if is_tpu:
            try:
                import torch_xla.debug.metrics as met
                # Use counter_value API (more reliable than parsing report string)
                compile_count = met.counter_value('UncachedCompile') or 0
                print(f"\n[TPU] XLA compilations: {compile_count}")
                if compile_count > 3:
                    print(f"  Warning: {compile_count} compilations detected (expected 1-3)")
                    print(f"  This may indicate dynamic shapes or recompilation issues")
            except Exception as e:
                print(f"[TPU] Metrics not available: {e}")

    # Cleanup to free GPU memory
    del optimizer
    if scheduler is not None:
        del scheduler
    # params may not exist if using separate param groups (LUT LR)
    try:
        del params
    except NameError:
        pass  # params wasn't defined (using param_groups instead)
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


# ==============================================================================
# RECOVERY LORA TRAINING
# ==============================================================================


def _truncate_kd_batch_to_seq_len(batch: dict, seq_len: int) -> dict:
    """
    Force KD batch tensors to match seq_len.
    This ensures shapes are constant for XLA compilation.

    - input_ids: [B, L] -> [B, seq_len]
    - attention_mask: [B, L] -> [B, seq_len]
    - topk_idx/topk_logits: [B, L-1, K] -> [B, seq_len-1, K]
    - rand_idx/rand_logits (if present): same as topk_*
    """
    if seq_len <= 0:
        return batch

    # input_ids
    input_ids = batch.get("input_ids", None)
    if input_ids is not None and input_ids.size(1) > seq_len:
        batch["input_ids"] = input_ids[:, :seq_len].contiguous()

    # attention_mask
    attn_mask = batch.get("attention_mask", None)
    if attn_mask is not None and attn_mask.size(1) > seq_len:
        batch["attention_mask"] = attn_mask[:, :seq_len].contiguous()

    # For top-k tensors, sequence dimension is usually L-1 (because predicting next token)
    seq_k = max(seq_len - 1, 1)

    for k in ["topk_idx", "topk_logits", "rand_idx", "rand_logits"]:
        t = batch.get(k, None)
        if t is None:
            continue
        # Expect [B, S, ...]
        if t.dim() >= 2 and t.size(1) > seq_k:
            batch[k] = t[:, :seq_k, ...].contiguous()

    return batch


def train_recovery_lora(
    model: nn.Module,
    train_data: Optional[str] = None,
    train_data_hf: Optional[str] = None,
    hf_subset: Optional[str] = None,
    hf_split: str = "train",
    hf_text_field: str = "text",
    hf_max_samples: Optional[int] = None,
    template_mode: str = "none",
    dataset_format: str = "text",
    generate_targets: bool = False,
    gen_max_tokens: int = 512,
    gen_temperature: float = 0.7,
    gen_top_p: float = 0.9,
    reference_model_id: Optional[str] = None,
    kd_cache_dir: Optional[str] = None,
    device: torch.device = None,
    tokenizer = None,
    recovery_r: int = 8,
    recovery_alpha: float = None,
    mlp_only: bool = True,
    skip_k_proj: bool = True,
    lr: float = 3e-4,
    max_steps: int = 1000,
    batch_size: int = 4,
    seq_len: int = 4096,
    warmup_steps: int = 100,
    constant_lr: bool = False,
    min_lr_ratio: float = 0.1,
    weight_decay: float = 0.0,
    grad_clip: float = 1.0,
    accumulation_steps: int = 1,
    mixed_precision: bool = False,
    dropout: float = 0.0,
    log_interval: int = 50,
    eval_interval: int = 200,
    eval_samples: int = 100,
    save_dir: Optional[str] = None,
    save_steps: int = 500,
    keep_checkpoints: int = 3,
    anchor_kl_weight: float = 0.0,
    anchor_samples: int = 32,
    no_full_logits: bool = False,
    anchor_sparse: bool = False,
    anchor_topk: Optional[int] = None,
    anchor_interval: int = 1,
    resume_from: Optional[str] = None,
    save_lora_only: bool = False,
    config: Optional[dict] = None,
    lora_mode: str = "recover",
    teacher_model: Optional[str] = None,
    kd_temperature: float = 2.0,
    kd_alpha: float = 0.5,
    hard_top1_weight: float = 0.0,
    hard_top1_end: Optional[float] = None,
    hard_full_weight: float = 0.0,
    use_wandb: bool = False,
    wandb_project: str = "recovery-lora",
    wandb_run_name: str = None,
    verbose: bool = True,
    debug: bool = False,
    freeze_mags: bool = False,
    freeze_mags_mlp: bool = False,
    freeze_all: bool = False,
    train_norms: bool = False,
    train_embeddings: bool = False,
    train_lm_head: bool = False,
) -> dict:
    """Train recovery LoRA adapters with multiple training modes.

    Modes:
        - recover: Standard CE loss on raw text (default)
        - sft: Supervised fine-tuning on instruction/response pairs
        - kd: Knowledge distillation from teacher model

    CRITICAL ORDER (before calling this function):
        1. model = load_model(...)
        2. model.load_state_dict(checkpoint)
        3. enable_recovery_lora_all(model, r=8)  # Enable LoRA
        4. freeze_for_recovery_training(model)    # Freeze base, keep LoRA trainable
        5. train_recovery_lora(model, ...)        # This function

    Args:
        model: Model with V2 layers (LoRA already enabled and frozen)
        train_data: Path to training data (JSONL with 'text' field or packed .pt tensors)
        train_data_hf: HuggingFace dataset name (e.g., 'NeelNanda/pile-10k')
        hf_subset: HF dataset subset (e.g., 'wikitext-103-v1')
        hf_split: HF dataset split (default: 'train')
        hf_text_field: HF dataset text field (default: 'text')
        hf_max_samples: Max samples from HF dataset (default: all)
        template_mode: Tokenization mode - 'none' (raw text), 'no-think' (chat template),
                       'think' (chat+thinking), 'both' (mix no-think+think),
                       'all' (sequential from none/no-think/think per sample hash).
        dataset_format: Dataset format for parsing - 'text', 'alpaca', 'sharegpt'
        generate_targets: If True, generate training targets using reference model.
                         For think mode, this produces proper <think>...</think> content.
        gen_max_tokens: Max tokens to generate when generate_targets=True
        gen_temperature: Temperature for generation
        gen_top_p: Top-p for generation
        reference_model_id: Model ID for generating targets (default: same as training model)
        device: Device to train on
        tokenizer: Tokenizer for encoding text data (required for JSONL/HF data)
        recovery_r: LoRA rank (if LoRA not already enabled)
        recovery_alpha: LoRA alpha (default: recovery_r)
        mlp_only: If True, only enable LoRA on MLP layers
        skip_k_proj: If True, skip K projection
        lr: Learning rate (LoRA tolerates higher LR, default 3e-4)
        max_steps: Maximum training steps
        batch_size: Batch size
        seq_len: Sequence length (4K-8K recommended for recovery)
        warmup_steps: Warmup steps for LR scheduler
        weight_decay: Weight decay (0 or small for LoRA)
        grad_clip: Gradient clipping value
        accumulation_steps: Gradient accumulation steps
        log_interval: Steps between logging
        eval_interval: Steps between evaluation
        eval_samples: Number of samples for evaluation
        save_dir: Directory to save checkpoints
        save_steps: Steps between checkpoints
        keep_checkpoints: Number of checkpoints to keep
        anchor_kl_weight: Weight for anchor KL regularizer (0 = disabled)
        anchor_samples: Number of anchor samples for KL regularizer
        use_wandb: Whether to log to wandb
        wandb_project: Wandb project name
        wandb_run_name: Wandb run name
        verbose: Print progress
        freeze_mags: Snap rank_magnitude to FP16 and freeze (all 196 layers)
        freeze_mags_mlp: Snap rank_magnitude to FP16 and freeze (MLP only, 84 layers)
        freeze_all: Snap scale_A, scale_B, and rank_magnitude to FP16 and freeze all

    Returns:
        Dictionary with training results
    """
    import time
    import json
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

    # Import recovery LoRA utilities
    from .ane_qat_linear_v2 import (
        enable_recovery_lora_all,
        freeze_for_recovery_training,
        get_recovery_lora_params,
        get_recovery_lora_stats,
    )

    t_start = time.time()

    # Check if LoRA is already enabled, if not enable it
    lora_stats = get_recovery_lora_stats(model)
    if lora_stats['enabled'] == 0:
        if verbose:
            print("[Recovery LoRA] Enabling LoRA adapters...")
        enable_recovery_lora_all(
            model,
            r=recovery_r,
            alpha=recovery_alpha,
            dropout=dropout,
            mlp_only=mlp_only,
            skip_k_proj=skip_k_proj,
            verbose=verbose,
        )
        freeze_for_recovery_training(
            model,
            verbose=verbose,
            train_norms=train_norms,
            train_embeddings=train_embeddings,
            train_lm_head=train_lm_head,
        )

    # Freeze mags/scales if requested (snap to FP16 + freeze)
    if freeze_all or freeze_mags or freeze_mags_mlp:
        mags_frozen = 0
        scales_frozen = 0
        mlp_patterns = ('gate_proj', 'up_proj', 'down_proj')
        for name, m in model.named_modules():
            # Snap scale_A and scale_B when freeze_all
            if freeze_all and hasattr(m, 'scale_A') and m.scale_A is not None:
                # Move to CPU for snapping (XLA/TPU .half() doesn't work correctly)
                with torch.no_grad():
                    orig_device = m.scale_A.data.device
                    m.scale_A.data = m.scale_A.data.cpu().half().float().to(orig_device)
                    m.scale_B.data = m.scale_B.data.cpu().half().float().to(orig_device)
                m.scale_A.requires_grad = False
                m.scale_B.requires_grad = False
                scales_frozen += 1
            # Snap rank_magnitude
            if hasattr(m, 'rank_magnitude') and m.rank_magnitude is not None:
                is_mlp = any(p in name for p in mlp_patterns)
                if freeze_all or freeze_mags or (freeze_mags_mlp and is_mlp):
                    # Move to CPU for snapping (XLA/TPU .half() doesn't work correctly)
                    with torch.no_grad():
                        orig_device = m.rank_magnitude.data.device
                        snapped = m.rank_magnitude.data.cpu().half().float()
                        m.rank_magnitude.data = snapped.to(orig_device)
                    m.rank_magnitude.requires_grad = False
                    mags_frozen += 1
        if verbose:
            if scales_frozen > 0:
                print(f"  Snapped & frozen {scales_frozen} scale_A/scale_B pairs")
            if mags_frozen > 0:
                print(f"  Snapped & frozen {mags_frozen} rank_magnitude tensors")

    # Resume from checkpoint if specified
    if resume_from:
        if verbose:
            print(f"\n[Recovery LoRA] Resuming from: {resume_from}")
        resume_state = torch.load(resume_from, map_location='cpu')
        # Load only LoRA weights (lora_A, lora_B)
        lora_keys = [k for k in resume_state if 'lora_' in k]
        lora_state = {k: resume_state[k] for k in lora_keys}
        missing, unexpected = model.load_state_dict(lora_state, strict=False)
        if verbose:
            print(f"  Loaded {len(lora_keys)} LoRA tensors")
            if missing:
                # Filter to show only LoRA-related missing keys
                lora_missing = [k for k in missing if 'lora_' in k]
                if lora_missing:
                    print(f"  Missing LoRA keys: {len(lora_missing)}")

    # Get trainable params
    trainable_params = get_recovery_lora_params(model)
    if verbose:
        print(f"\n[Recovery LoRA Training]")
        print(f"  Mode: {lora_mode}" + (f" (teacher: {teacher_model})" if lora_mode == "kd" else ""))
        print(f"  Trainable params: {trainable_params:,} ({trainable_params * 4 / 1024 / 1024:.1f} MB)")
        print(f"  LR: {lr}, Max steps: {max_steps}, Batch size: {batch_size}")
        print(f"  Seq len: {seq_len}, Accumulation: {accumulation_steps}")
        if lora_mode == "kd":
            print(f"  KD: temperature={kd_temperature}, alpha={kd_alpha}")
        if generate_targets:
            print(f"  Generate targets: enabled (ref={reference_model_id})")
            print(f"    gen_max_tokens={gen_max_tokens}, temp={gen_temperature}, top_p={gen_top_p}")

    # Load reference model for generate_targets mode
    ref_model = None
    if generate_targets:
        if verbose:
            print(f"\n[Generate Targets] Loading reference model: {reference_model_id}")
        from transformers import AutoModelForCausalLM
        ref_model = AutoModelForCausalLM.from_pretrained(
            reference_model_id,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            trust_remote_code=True,
        ).to(device)
        ref_model.eval()
        if verbose:
            print(f"  Reference model loaded on {device}")

    # Helper function for generating targets with reference model
    def generate_target_sequence(prompt_text, tokenizer, ref_model, template_mode, device,
                                  max_new_tokens, temperature, top_p):
        """Generate a target sequence using reference model.

        For think mode, this produces <think>...</think> content.
        For 'all' mode, cycles through none/no-think/think based on content hash.
        Returns the full sequence (prompt + generated response) as string.
        """
        # Build prompt with chat template
        messages = [{"role": "user", "content": prompt_text}]

        # Determine actual mode for "all" (sequential based on content hash)
        actual_mode = template_mode
        if template_mode == "all":
            mode_idx = hash(prompt_text) % 3
            mode_choices = ["none", "no-think", "think"]
            actual_mode = mode_choices[mode_idx]
        elif template_mode == "both":
            # For "both", alternate based on hash
            actual_mode = "think" if hash(prompt_text) % 2 == 0 else "no-think"

        # For "none" mode, just return raw text (no generation needed)
        if actual_mode == "none":
            return prompt_text

        # Determine if thinking should be enabled
        use_thinking = actual_mode == "think"

        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=use_thinking,
            )
        except TypeError:
            # Tokenizer doesn't support enable_thinking
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        # Tokenize and generate
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = ref_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode full sequence (prompt + response)
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=False)

        # Clean up end token if present
        if full_text.endswith(tokenizer.eos_token):
            full_text = full_text[:-len(tokenizer.eos_token)]

        return full_text

    # Helper for template-based tokenization
    def render_with_template(item, tokenizer, template_mode, dataset_format):
        """Render text using chat template based on mode and format."""
        # Parse item into messages based on format
        if dataset_format == "text":
            # Raw text - just use the text field
            text = item if isinstance(item, str) else item.get('text', item.get('content', ''))
            if template_mode == "none":
                return [text] if text else []
            # For text format with template, wrap as user message
            messages = [{"role": "user", "content": text}]
        elif dataset_format == "alpaca":
            # Alpaca format: instruction, input, output
            instruction = item.get('instruction', '')
            inp = item.get('input', '')
            output = item.get('output', '')
            user_content = f"{instruction}\n{inp}".strip() if inp else instruction
            messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": output}
            ]
        elif dataset_format == "sharegpt":
            # ShareGPT format: conversations list
            convs = item.get('conversations', item.get('messages', []))
            messages = []
            for c in convs:
                role = c.get('from', c.get('role', 'user'))
                role = 'user' if role in ['human', 'user'] else 'assistant'
                messages.append({"role": role, "content": c.get('value', c.get('content', ''))})
        else:
            return []

        if not messages:
            return []

        # Apply template based on mode
        rendered = []
        if template_mode == "none":
            # Raw text - no template
            text = "\n\n".join(m.get("content", "") for m in messages if m.get("content"))
            if text:
                rendered.append(text)
        elif template_mode == "no-think":
            # Chat template without thinking
            try:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
                )
                if text:
                    rendered.append(text)
            except TypeError:
                # Tokenizer doesn't support enable_thinking
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                if text:
                    rendered.append(text)
        elif template_mode == "think":
            # Chat template with thinking enabled
            try:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False, enable_thinking=True
                )
                if text:
                    rendered.append(text)
            except TypeError:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                if text:
                    rendered.append(text)
        elif template_mode == "both":
            # Both thinking and no-thinking variants
            for enable in [False, True]:
                try:
                    text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=False, enable_thinking=enable
                    )
                    if text:
                        rendered.append(text)
                except TypeError:
                    text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=False
                    )
                    if text:
                        rendered.append(text)
                    break  # Only add once if enable_thinking not supported
        elif template_mode == "all":
            # Sequential round-robin based on content hash: none -> no-think -> think
            # Each unique sample gets deterministic mode, ensures even distribution
            content_str = "".join(m.get("content", "") for m in messages)
            mode_idx = hash(content_str) % 3
            mode_choices = ["none", "no-think", "think"]
            mode_choice = mode_choices[mode_idx]
            if mode_choice == "none":
                # Raw text - no template
                text = "\n\n".join(m.get("content", "") for m in messages if m.get("content"))
                if text:
                    rendered.append(text)
            elif mode_choice == "no-think":
                try:
                    text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
                    )
                    if text:
                        rendered.append(text)
                except TypeError:
                    text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=False
                    )
                    if text:
                        rendered.append(text)
            else:  # think
                try:
                    text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=False, enable_thinking=True
                    )
                    if text:
                        rendered.append(text)
                except TypeError:
                    text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=False
                    )
                    if text:
                        rendered.append(text)

        return rendered

    # Streaming tokenizer - samples batch_size texts, tokenizes to seq_len each step
    class StreamingTokenizer:
        """Memory-efficient tokenizer: sample texts, tokenize, truncate/pad to seq_len."""
        def __init__(self, texts, tokenizer, render_fn, template_mode, dataset_format,
                     seq_len, batch_size, generate_targets=False, gen_fn=None,
                     ref_model=None, device=None, gen_max_tokens=512,
                     gen_temperature=0.7, gen_top_p=0.9):
            # Keep reference to dataset (don't copy to list - saves memory!)
            # HuggingFace datasets support random access via ds[idx]
            self.texts = texts
            self.num_texts = len(texts)
            self.tokenizer = tokenizer
            self.render_fn = render_fn
            self.template_mode = template_mode
            self.dataset_format = dataset_format
            self.seq_len = seq_len
            self.batch_size = batch_size
            self.pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
            # Generate targets mode
            self.generate_targets = generate_targets
            self.gen_fn = gen_fn
            self.ref_model = ref_model
            self.device = device
            self.gen_max_tokens = gen_max_tokens
            self.gen_temperature = gen_temperature
            self.gen_top_p = gen_top_p

        def _tokenize_item(self, item):
            """Tokenize a single item, return tokens truncated to seq_len."""
            # Get text/prompt based on format
            if self.dataset_format == "text":
                if isinstance(item, str):
                    text = item
                else:
                    text = item.get('text', item.get('content', ''))
                if not text or not text.strip():
                    return None
                prompt_text = text
            elif self.dataset_format == "alpaca":
                instruction = item.get('instruction', '')
                inp = item.get('input', '')
                prompt_text = f"{instruction}\n{inp}".strip() if inp else instruction
                if not prompt_text:
                    return None
            else:
                # For sharegpt, use first user message as prompt
                convs = item.get('conversations', item.get('messages', []))
                prompt_text = None
                for c in convs:
                    role = c.get('from', c.get('role', 'user'))
                    if role in ['human', 'user']:
                        prompt_text = c.get('value', c.get('content', ''))
                        break
                if not prompt_text:
                    return None

            # Generate target or use template rendering
            if self.generate_targets and self.gen_fn and self.ref_model:
                # Generate target sequence using reference model
                full_text = self.gen_fn(
                    prompt_text, self.tokenizer, self.ref_model, self.template_mode,
                    self.device, self.gen_max_tokens, self.gen_temperature, self.gen_top_p
                )
                if not full_text:
                    return None
                tokens = self.tokenizer.encode(full_text, add_special_tokens=False)
            else:
                # Original template rendering
                to_render = prompt_text if self.dataset_format == "text" else item
                rendered_list = self.render_fn(to_render, self.tokenizer, self.template_mode, self.dataset_format)
                if not rendered_list:
                    return None
                tokens = self.tokenizer.encode(rendered_list[0], add_special_tokens=False)

            # Truncate to seq_len
            if len(tokens) > self.seq_len:
                tokens = tokens[:self.seq_len]
            return tokens

        def get_batch(self):
            """Sample batch_size texts, tokenize each to seq_len, return [B, seq_len]."""
            batch = []
            attempts = 0
            max_attempts = self.batch_size * 100  # More attempts for sparse datasets (WikiText has many empty lines)

            while len(batch) < self.batch_size and attempts < max_attempts:
                attempts += 1
                # Sample random text (use num_texts for length)
                idx = random.randint(0, self.num_texts - 1)
                tokens = self._tokenize_item(self.texts[idx])

                if tokens is None or len(tokens) < 32:  # Skip very short texts
                    continue

                # Pad if needed
                if len(tokens) < self.seq_len:
                    tokens = tokens + [self.pad_id] * (self.seq_len - len(tokens))

                batch.append(tokens)

            if len(batch) < self.batch_size:
                # Last resort: pad with random tokens if we can't find enough valid texts
                print(f"  Warning: only found {len(batch)}/{self.batch_size} valid texts after {attempts} attempts")
                while len(batch) < self.batch_size:
                    # Create random tokens as fallback
                    random_tokens = [random.randint(100, 10000) for _ in range(self.seq_len)]
                    batch.append(random_tokens)

            return torch.tensor(batch, dtype=torch.long)

        def get_total_texts(self):
            return self.num_texts

    if verbose and template_mode != "none":
        print(f"  Template mode: {template_mode}, Format: {dataset_format}")

    # Memory debug helper
    def _debug_mem(label):
        if not debug:
            return
        import gc
        gc.collect()
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1024**3
            print(f"    [MEM {label}] GPU: {alloc:.2f}GB")
        else:
            try:
                import psutil
                mem = psutil.Process().memory_info().rss / 1024**3
                print(f"    [MEM {label}] RSS: {mem:.2f}GB")
            except ImportError:
                pass

    _debug_mem("start of train_recovery_lora")

    # Load training data
    # Suppress tokenizer length warnings - we handle chunking ourselves
    if tokenizer is not None:
        tokenizer.model_max_length = int(1e12)

    streaming_tokenizer = None  # Will be set for HF/JSONL, None for pre-tokenized .pt
    input_ids = None  # Will be set for pre-tokenized .pt

    # KD cache mode - fast, memory-efficient, no teacher needed
    kd_cache_dataset = None
    use_kd_cache = kd_cache_dir is not None

    if use_kd_cache:
        if verbose:
            print(f"  Loading KD cache: {kd_cache_dir}")
        kd_cache_dataset = KDCacheDataset(kd_cache_dir, shuffle=True, preload=True)
        if verbose:
            print(f"  KD cache ready ({len(kd_cache_dataset)} examples)")

    elif train_data_hf is not None:
        # Load from HuggingFace - use streaming tokenizer (memory efficient)
        if verbose:
            subset_str = f" ({hf_subset})" if hf_subset else ""
            print(f"  Loading HF dataset: {train_data_hf}{subset_str}")

        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets library required for --train-data-hf. Install with: pip install datasets")

        if hf_subset:
            ds = load_dataset(train_data_hf, hf_subset, split=hf_split)
        else:
            ds = load_dataset(train_data_hf, split=hf_split)

        # Limit samples if requested
        max_samples = hf_max_samples if hf_max_samples else len(ds)
        if max_samples < len(ds):
            ds = ds.select(range(max_samples))

        if verbose:
            print(f"  Dataset size: {len(ds):,} samples")

        if tokenizer is None:
            raise ValueError("tokenizer required for HF data")

        # Create streaming tokenizer - tokenizes on-demand, not all at once
        streaming_tokenizer = StreamingTokenizer(
            texts=ds,
            tokenizer=tokenizer,
            render_fn=render_with_template,
            template_mode=template_mode,
            dataset_format=dataset_format,
            seq_len=seq_len,
            batch_size=batch_size,
            generate_targets=generate_targets,
            gen_fn=generate_target_sequence if generate_targets else None,
            ref_model=ref_model,
            device=device,
            gen_max_tokens=gen_max_tokens,
            gen_temperature=gen_temperature,
            gen_top_p=gen_top_p,
        )
        if verbose:
            mode_str = " (with target generation)" if generate_targets else ""
            print(f"  Streaming tokenizer ready ({batch_size} texts × {seq_len} tokens/batch){mode_str}")

    elif train_data is not None:
        if verbose:
            print(f"  Loading data: {train_data}")

        if train_data.endswith('.pt'):
            # Pre-tokenized packed tensors - load directly (most efficient)
            data = torch.load(train_data, map_location='cpu')
            if isinstance(data, torch.Tensor):
                input_ids = data
            else:
                input_ids = data.get('input_ids', data.get('tokens'))
            if verbose:
                print(f"  Pre-tokenized: {input_ids.numel():,} tokens")
        elif train_data.endswith('.jsonl') or train_data.endswith('.json'):
            # JSONL - use streaming tokenizer (memory efficient)
            if tokenizer is None:
                raise ValueError("tokenizer required for JSONL data")

            # Load items into list (just metadata, not tokenized)
            items = []
            with open(train_data, 'r') as f:
                for line in f:
                    items.append(json.loads(line))

            if verbose:
                print(f"  Loaded {len(items):,} items")

            # Create streaming tokenizer
            streaming_tokenizer = StreamingTokenizer(
                texts=items,
                tokenizer=tokenizer,
                render_fn=render_with_template,
                template_mode=template_mode,
                dataset_format=dataset_format,
                seq_len=seq_len,
                batch_size=batch_size,
                generate_targets=generate_targets,
                gen_fn=generate_target_sequence if generate_targets else None,
                ref_model=ref_model,
                device=device,
                gen_max_tokens=gen_max_tokens,
                gen_temperature=gen_temperature,
                gen_top_p=gen_top_p,
            )
            if verbose:
                mode_str = " (with target generation)" if generate_targets else ""
                print(f"  Streaming tokenizer ready ({batch_size} texts × {seq_len} tokens/batch){mode_str}")
        else:
            raise ValueError(f"Unsupported data format: {train_data}")
    else:
        raise ValueError("One of train_data or train_data_hf must be provided")

    # Summary
    tokens_per_step = batch_size * seq_len
    tokens_per_update = tokens_per_step * accumulation_steps
    if verbose:
        if input_ids is not None:
            print(f"  Total tokens: {input_ids.numel():,}")
        print(f"  Tokens/step: {tokens_per_step:,}, Tokens/update: {tokens_per_update:,} (batch={batch_size}x{accumulation_steps})")

    # Compute anchor logits if KL regularizer enabled
    # Supports two modes:
    #   - Dense (default): anchor_logits = [samples, L, V] (full vocab)
    #   - Sparse (no_full_logits/anchor_sparse): anchor_topk_logits/anchor_topk_idx = [samples, L, K]
    anchor_logits = None  # Full logits [samples, L, V] - only if not sparse mode
    anchor_topk_logits = None  # Sparse top-K logits [samples, L, K] - for sparse mode
    anchor_topk_idx = None  # Sparse top-K indices [samples, L, K] - for sparse mode
    anchor_input_ids = None
    anchor_K = anchor_topk if anchor_topk else 128  # Default K=128 if not specified

    # Determine if we should use sparse anchor mode
    use_sparse_anchor = anchor_sparse or no_full_logits

    if anchor_kl_weight > 0 and anchor_samples > 0:
        _anchor_start = time.time()
        mode_str = "sparse" if use_sparse_anchor else "dense"
        if verbose:
            print(f"  Computing anchor logits ({anchor_samples} samples, {mode_str} mode)...", flush=True)
        model.eval()

        if use_kd_cache:
            # Use samples from KD cache for anchors
            anchor_batches = []
            anchor_topk_idx_list = []
            anchor_topk_logits_list = []
            temp_iter = iter(kd_cache_dataset)
            while len(anchor_batches) < anchor_samples:
                try:
                    batch = next(temp_iter)
                    anchor_batches.append(batch['input_ids'])
                    # For sparse mode, also collect topk_idx and topk_logits from cache
                    if use_sparse_anchor and 'topk_idx' in batch:
                        anchor_topk_idx_list.append(batch['topk_idx'])
                        anchor_topk_logits_list.append(batch['topk_logits'])
                except StopIteration:
                    temp_iter = iter(kd_cache_dataset)
            anchor_batch = torch.stack(anchor_batches[:anchor_samples], dim=0).to(device)
            # Truncate anchor_batch to seq_len for consistent XLA shapes
            if seq_len > 0 and anchor_batch.size(1) > seq_len:
                anchor_batch = anchor_batch[:, :seq_len].contiguous()

            # For sparse mode with KD cache, use cache's topk as anchor
            if use_sparse_anchor and anchor_topk_idx_list:
                # Stack and use cache's topk directly as anchor (teacher logits)
                anchor_topk_idx = torch.stack(anchor_topk_idx_list[:anchor_samples], dim=0)  # [A, L, K]
                anchor_topk_logits = torch.stack(anchor_topk_logits_list[:anchor_samples], dim=0)  # [A, L, K]
                # Truncate anchor topk to seq_len-1 for consistent XLA shapes
                seq_k = max(seq_len - 1, 1)
                if seq_len > 0 and anchor_topk_idx.size(1) > seq_k:
                    anchor_topk_idx = anchor_topk_idx[:, :seq_k, :].contiguous()
                    anchor_topk_logits = anchor_topk_logits[:, :seq_k, :].contiguous()
                # Update K from cache
                anchor_K = anchor_topk_idx.size(-1)
                if verbose:
                    print(f"  Using KD cache topk as anchor: idx={anchor_topk_idx.shape}, logits={anchor_topk_logits.shape}")
        elif streaming_tokenizer is not None:
            # Use streaming tokenizer to get anchor samples
            anchor_batch = streaming_tokenizer.get_batch().to(device)
            # Get more samples if needed
            anchor_batches = [anchor_batch]
            while sum(b.shape[0] for b in anchor_batches) < anchor_samples:
                anchor_batches.append(streaming_tokenizer.get_batch().to(device))
            anchor_batch = torch.cat(anchor_batches, dim=0)[:anchor_samples]
            # Truncate anchor_batch to seq_len for consistent XLA shapes
            if seq_len > 0 and anchor_batch.size(1) > seq_len:
                anchor_batch = anchor_batch[:, :seq_len].contiguous()
        elif input_ids is not None:
            # Use pre-tokenized input_ids
            total_tokens = input_ids.numel()
            anchor_inputs = []
            for i in range(anchor_samples):
                start = (i * seq_len) % max(1, total_tokens - seq_len)
                end = start + seq_len
                anchor_inputs.append(input_ids[start:end].unsqueeze(0))
            anchor_batch = torch.cat(anchor_inputs, dim=0).to(device)
        else:
            if verbose:
                print(f"  Warning: No data source for anchor samples, skipping anchor KL")
            anchor_kl_weight = 0.0
            anchor_batch = None

        if anchor_batch is not None:
            anchor_input_ids = anchor_batch  # Save for later use in training loop
            if verbose:
                print(f"  Anchor batch shape: {anchor_batch.shape}, running forward...", flush=True)

            # Compute anchor logits (only if not using cache's topk as anchor)
            if anchor_topk_logits is None:
                with torch.no_grad():
                    full_logits = model(anchor_batch, use_cache=False).logits.detach()

                if use_sparse_anchor:
                    # Extract top-K and discard full logits (memory-safe for L>=1024)
                    anchor_topk_logits, anchor_topk_idx = torch.topk(full_logits, k=anchor_K, dim=-1)
                    anchor_topk_logits = anchor_topk_logits.to(torch.bfloat16).cpu()  # Save memory
                    anchor_topk_idx = anchor_topk_idx.cpu()
                    del full_logits
                    if verbose:
                        print(f"  Anchor logits computed (sparse): topk_idx={anchor_topk_idx.shape}, topk_logits={anchor_topk_logits.shape}")
                else:
                    anchor_logits = full_logits
                    if verbose:
                        print(f"  Anchor logits computed (dense): {anchor_logits.shape}")

            # TPU: mark_step after anchor forward to compile/execute
            if is_xla_device(device):
                xla_mark_step()
            _anchor_time = time.time() - _anchor_start
            if verbose:
                print(f"  Anchor precompute done ({_anchor_time:.1f}s)", flush=True)
        model.train()

    # Load teacher model for KD mode
    teacher = None
    if lora_mode == "kd" and teacher_model is not None:
        if verbose:
            print(f"\n[Recovery LoRA] Loading teacher model: {teacher_model}")
        from transformers import AutoModelForCausalLM
        teacher = AutoModelForCausalLM.from_pretrained(
            teacher_model,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            trust_remote_code=True,
        )
        teacher.to(device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        if verbose:
            print(f"  Teacher loaded: {sum(p.numel() for p in teacher.parameters()):,} params")

    # Setup optimizer (only LoRA params are trainable)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(params, lr=lr, weight_decay=weight_decay)

    # Setup LR scheduler (cosine with warmup)
    total_steps = max_steps * accumulation_steps
    # Ensure warmup doesn't exceed total steps
    effective_warmup = min(warmup_steps, total_steps - 1) if total_steps > 1 else 0
    cosine_steps = max(1, total_steps - effective_warmup)  # At least 1 to avoid division by zero

    if constant_lr:
        # Constant LR after warmup
        from torch.optim.lr_scheduler import LambdaLR
        if effective_warmup > 0:
            warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=effective_warmup)
            constant_scheduler = LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
            scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, constant_scheduler], milestones=[effective_warmup])
        else:
            scheduler = LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
        if verbose:
            print(f"  LR schedule: warmup={effective_warmup} then constant")
    else:
        # Cosine decay with min_lr_ratio
        if effective_warmup > 0:
            warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=effective_warmup)
            cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=lr * min_lr_ratio)
            scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[effective_warmup])
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=lr * min_lr_ratio)
        if verbose:
            print(f"  LR schedule: warmup={effective_warmup} then cosine (min_ratio={min_lr_ratio})")

    # Wandb setup
    wandb_run = None
    if use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=wandb_project,
                name=wandb_run_name or f"recovery_r{recovery_r}",
                config={
                    'recovery_r': recovery_r,
                    'lr': lr,
                    'max_steps': max_steps,
                    'batch_size': batch_size,
                    'seq_len': seq_len,
                    'mlp_only': mlp_only,
                    'anchor_kl_weight': anchor_kl_weight,
                    # Sparse logits mode
                    'no_full_logits': no_full_logits,
                    'anchor_sparse': anchor_sparse,
                    'anchor_topk': anchor_topk,
                    'anchor_interval': anchor_interval,
                    'sparse_logits_enabled': use_sparse_anchor,
                    'hard_logits_enabled': hard_top1_weight > 0 or hard_full_weight > 0,
                },
                resume="never",  # Always start fresh - avoids step conflicts on restart
            )
            # Define metrics to use train/step as X-axis
            wandb.define_metric("train/step")
            wandb.define_metric("train/*", step_metric="train/step")
            wandb.define_metric("eval/*", step_metric="train/step")
            wandb.define_metric("tpu/*", step_metric="train/step")
        except ImportError:
            print("wandb not installed, skipping logging")
            use_wandb = False

    # Training loop
    total_loss = 0.0
    best_loss = float('inf')
    best_state = None
    loss_history = []
    optimizer_step = 0

    # For smoothed ETA: track (optimizer_step, time) pairs from last N logs
    # This avoids XLA compile time skewing the ETA
    from collections import deque
    eta_history = deque(maxlen=5)
    last_log_time = time.time()

    # Sanity check: compare eval vs train mode CE loss
    if debug and (streaming_tokenizer is not None or input_ids is not None):
        print(f"\n  [DEBUG] Sanity check - eval vs train mode:")
        # Get a test batch
        if streaming_tokenizer is not None:
            test_batch = streaming_tokenizer.get_batch().to(device)
        else:
            test_batch = input_ids[:seq_len].unsqueeze(0).to(device)
        test_labels = test_batch.clone()

        # Eval mode
        model.eval()
        with torch.no_grad():
            eval_out = model(test_batch, use_cache=False)
            eval_logits = eval_out.logits
            eval_loss = F.cross_entropy(
                eval_logits[:, :-1, :].reshape(-1, eval_logits.size(-1)),
                test_labels[:, 1:].reshape(-1),
            )
        print(f"    Eval mode CE loss: {eval_loss.item():.4f}")
        print(f"    Eval logits range: [{eval_logits.min().item():.2f}, {eval_logits.max().item():.2f}]")

        # Train mode
        model.train()
        with torch.no_grad():
            train_out = model(test_batch, use_cache=False)
            train_logits = train_out.logits
            train_loss = F.cross_entropy(
                train_logits[:, :-1, :].reshape(-1, train_logits.size(-1)),
                test_labels[:, 1:].reshape(-1),
            )
        print(f"    Train mode CE loss: {train_loss.item():.4f}")
        print(f"    Train logits range: [{train_logits.min().item():.2f}, {train_logits.max().item():.2f}]")

        # Check if they're similar
        logit_diff = (eval_logits - train_logits).abs().max().item()
        print(f"    Max logit diff (eval vs train): {logit_diff:.6f}")
        if logit_diff > 0.01:
            print(f"    WARNING: Significant difference between eval and train mode!")

        # Clean up sanity check tensors
        del eval_out, eval_logits, eval_loss, train_out, train_logits, train_loss
        del test_batch, test_labels
        import gc
        gc.collect()

    model.train()

    # Mixed precision setup: FP32 master weights + BF16/FP16 compute via autocast
    scaler = None
    autocast_dtype = None
    use_tpu = is_xla_device(device)
    xm = None
    if use_tpu:
        try:
            import torch_xla.core.xla_model as xm
        except ImportError:
            use_tpu = False
    if mixed_precision:
        if use_tpu:
            # TPU: FP32 weights + BF16 compute via autocast (device_type='xla' is supported)
            autocast_dtype = torch.bfloat16
        elif device.type == "cuda":
            autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            if autocast_dtype == torch.float16:
                scaler = torch.amp.GradScaler('cuda')
        elif device.type == "mps":
            autocast_dtype = torch.float16
        if verbose:
            print(f"  Mixed precision: {autocast_dtype}")

    # TPU detection for debug logging (use_tpu already set above)
    if verbose:
        if use_tpu:
            print(f"\n  Starting training on TPU/XLA...")
            print(f"  Note: First step may take 5-15 min for XLA compilation", flush=True)
        else:
            print(f"\n  Starting training...")

    # KD cache iterator
    kd_cache_iter = None
    if use_kd_cache:
        kd_cache_iter = iter(kd_cache_dataset)

    # Timing for first steps (XLA debug)
    _step_times = {}

    for step in range(1, total_steps + 1):
        # Detailed timing for first 3 steps on TPU to diagnose XLA compilation
        _timing = step <= 3 and (use_tpu or debug)
        if _timing:
            _step_times[f's{step}_start'] = time.time()
            print(f"\n  [Step {step}/{total_steps}] Data loading...", flush=True)

        # Get batch - KD cache, streaming tokenizer, or pre-tokenized data
        kd_batch = None
        if use_kd_cache:
            # KD cache mode - get precomputed teacher logits
            try:
                batch_examples = [next(kd_cache_iter) for _ in range(batch_size)]
            except StopIteration:
                # Reset iterator
                kd_cache_iter = iter(kd_cache_dataset)
                batch_examples = [next(kd_cache_iter) for _ in range(batch_size)]

            # Collate batch
            kd_batch = {
                'input_ids': torch.stack([ex['input_ids'] for ex in batch_examples]).to(device),
                'attention_mask': torch.stack([ex['attention_mask'] for ex in batch_examples]).to(device),
                'topk_idx': torch.stack([ex['topk_idx'] for ex in batch_examples]).to(device),
                'topk_logits': torch.stack([ex['topk_logits'] for ex in batch_examples]).to(device),
            }
            # Truncate to seq_len for consistent XLA shapes (cache may be longer)
            kd_batch = _truncate_kd_batch_to_seq_len(kd_batch, seq_len)
            input_batch = kd_batch['input_ids']
            if _timing:
                print(f"  [Step {step}/{total_steps}] Data ready: {input_batch.shape}", flush=True)
        elif streaming_tokenizer is not None:
            input_batch = streaming_tokenizer.get_batch().to(device)  # [B, seq_len]
        else:
            # Sample random batch from pre-tokenized packed data
            total_tokens = input_ids.numel()
            batch_inputs = []
            for b in range(batch_size):
                start = random.randint(0, max(1, total_tokens - seq_len - 1))
                end = start + seq_len
                batch_inputs.append(input_ids[start:end])
            input_batch = torch.stack(batch_inputs).to(device)  # [B, seq_len]

        labels = input_batch.clone()  # Next token prediction

        # Debug: show input sequence before forward pass
        if debug and step == 1:
            print(f"\n  [DEBUG] Step {step}:")
            print(f"    Input shape: {input_batch.shape}")
            print(f"    Input tokens (first 50): {input_batch[0, :50].tolist()}")
            if tokenizer is not None:
                decoded = tokenizer.decode(input_batch[0, :100].tolist(), skip_special_tokens=False)
                print(f"    Input text (first 100 tokens): {repr(decoded[:200])}")

        # Mode-specific loss computation (with optional mixed precision)
        if _timing:
            _step_times[f's{step}_fwd_start'] = time.time()
            print(f"  [Step {step}/{total_steps}] Forward pass (XLA compile on step 1)...", flush=True)

        # Use 'xla' device_type for TPU, otherwise use device.type
        autocast_device_type = 'xla' if use_tpu else device.type
        autocast_ctx = torch.amp.autocast(autocast_device_type, dtype=autocast_dtype) if autocast_dtype else nullcontext()

        with autocast_ctx:
            if use_kd_cache and kd_batch is not None:
                # Calculate current hard_top1 weight (with optional annealing)
                current_hard_top1 = hard_top1_weight
                if hard_top1_end is not None and max_steps > 1:
                    progress = optimizer_step / max_steps
                    current_hard_top1 = hard_top1_weight + (hard_top1_end - hard_top1_weight) * progress

                # KD cache mode - use precomputed teacher logits (fast!)
                # compute_kd_loss_batch does its own forward pass
                loss = compute_kd_loss_batch(
                    model, kd_batch, device, kd_temperature,
                    no_grad=False,
                    hard_top1_weight=current_hard_top1,
                    hard_full_weight=hard_full_weight,
                )
                if debug and step == 1:
                    print(f"    KD cache mode: loss={loss.item():.4f}, hard_top1={current_hard_top1:.4f}")
            else:
                # Non-KD-cache modes: need forward pass
                outputs = model(input_batch, use_cache=False)
                logits = outputs.logits  # [B, seq_len, vocab]

                # Debug: show batch info on first step
                if debug and step == 1:
                    print(f"    Logits shape: {logits.shape}")
                    print(f"    Logits range: [{logits.min().item():.2f}, {logits.max().item():.2f}]")

                # Compute CE loss (shift for next-token prediction)
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                ce_loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )

                if lora_mode == "kd" and teacher is not None:
                    # Knowledge distillation: combine CE + soft CE from teacher
                    with torch.no_grad():
                        teacher_outputs = teacher(input_batch, use_cache=False)
                        teacher_logits = teacher_outputs.logits[:, :-1, :].contiguous()

                    # Soft cross-entropy (TPU/XLA safe, avoids aten::kl_div)
                    kd_loss = kd_soft_ce(shift_logits, teacher_logits, kd_temperature)

                    # Combined loss: alpha * KD + (1 - alpha) * CE
                    loss = kd_alpha * kd_loss + (1 - kd_alpha) * ce_loss

                    if debug and step == 1:
                        print(f"    KD mode: ce={ce_loss.item():.4f}, kd={kd_loss.item():.4f}, total={loss.item():.4f}")
                else:
                    # recover/sft mode: just CE loss
                    loss = ce_loss

                # Debug: show loss and predictions
                if debug and step == 1:
                    print(f"    CE Loss: {ce_loss.item():.4f}")
                    # Show predicted vs actual for first few tokens
                    with torch.no_grad():
                        preds = shift_logits[0, :5].argmax(dim=-1)
                        actuals = shift_labels[0, :5]
                        print(f"    Predicted tokens (0-4): {preds.tolist()}")
                        print(f"    Actual tokens (0-4):    {actuals.tolist()}")
                    if tokenizer is not None:
                        pred_text = tokenizer.decode(preds.tolist())
                        actual_text = tokenizer.decode(actuals.tolist())
                        print(f"    Predicted: {repr(pred_text)}")
                        print(f"    Actual:    {repr(actual_text)}")

        if _timing:
            _step_times[f's{step}_fwd_end'] = time.time()
            _fwd_time = _step_times[f's{step}_fwd_end'] - _step_times[f's{step}_fwd_start']
            print(f"  [Step {step}/{total_steps}] Forward done ({_fwd_time:.1f}s)", flush=True)

        # Optional anchor KL regularizer (soft CE, TPU/XLA safe)
        # Supports two modes: dense (anchor_logits) or sparse (anchor_topk_idx/anchor_topk_logits)
        anchor_enabled = anchor_kl_weight > 0 and anchor_input_ids is not None and (
            anchor_logits is not None or (anchor_topk_idx is not None and anchor_topk_logits is not None)
        )
        if anchor_enabled:
            # Apply anchor KL every anchor_interval optimizer steps
            if (optimizer_step + 1) % anchor_interval == 0 or optimizer_step == 0:
                anchor_idx = (step - 1) % anchor_samples
                anchor_input = anchor_input_ids[anchor_idx:anchor_idx+1].to(device)

                if use_sparse_anchor and anchor_topk_idx is not None:
                    # =========================================================
                    # Sparse anchor KL: compute hidden states, gather top-K logits
                    # This avoids materializing [1, L, V] tensor
                    # =========================================================
                    if mixed_precision:
                        with torch.amp.autocast(autocast_device_type, dtype=autocast_dtype):
                            hidden = model.model(anchor_input, use_cache=False).last_hidden_state
                    else:
                        hidden = model.model(anchor_input, use_cache=False).last_hidden_state

                    # Remove last position to align with topk tensors (which are L-1 for next-token prediction)
                    hidden = hidden[:, :-1, :]

                    # Get dimensions
                    B_anc, L_anc, H_anc = hidden.shape
                    idx = anchor_topk_idx[anchor_idx].to(device).long()  # [L-1, K]
                    K_anc = idx.size(1)
                    anchor_topk = anchor_topk_logits[anchor_idx].to(device).float()  # [L-1, K]

                    # Process in chunks to avoid XLA OOM (gather L*K rows is too large)
                    # For L=1024, K=128: full gather = 131K rows = 512MB
                    # With chunk=64: gather = 8K rows = 32MB per chunk
                    anchor_chunk_size = 64  # Positions per chunk
                    chunk_losses = []

                    for chunk_start in range(0, L_anc, anchor_chunk_size):
                        chunk_end = min(chunk_start + anchor_chunk_size, L_anc)
                        chunk_len = chunk_end - chunk_start

                        # Get hidden states and indices for this chunk
                        h_chunk = hidden[0, chunk_start:chunk_end]  # [chunk, H]
                        idx_chunk = idx[chunk_start:chunk_end]  # [chunk, K]

                        # Gather lm_head weights for this chunk's top-K indices
                        w_chunk = model.lm_head.weight[idx_chunk.reshape(-1)]  # [chunk*K, H]
                        w_chunk = w_chunk.reshape(chunk_len, K_anc, H_anc)  # [chunk, K, H]

                        # Compute sparse logits: einsum('ch,ckh->ck')
                        student_chunk = torch.einsum('ch,ckh->ck', h_chunk, w_chunk)  # [chunk, K]

                        # Get anchor logits for this chunk
                        teacher_chunk = anchor_topk[chunk_start:chunk_end]  # [chunk, K]

                        # Compute KL loss for this chunk
                        teacher_probs = F.softmax(teacher_chunk, dim=-1)
                        teacher_log_probs = F.log_softmax(teacher_chunk, dim=-1)
                        student_log_probs = F.log_softmax(student_chunk, dim=-1)
                        chunk_kl = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1).mean()
                        chunk_losses.append(chunk_kl)

                    # Average loss across all chunks
                    anchor_loss = sum(chunk_losses) / len(chunk_losses)
                else:
                    # =========================================================
                    # Dense anchor KL: full logits comparison
                    # =========================================================
                    current_logits = model(anchor_input, use_cache=False)
                    if hasattr(current_logits, 'logits'):
                        current_logits = current_logits.logits
                    # Soft cross-entropy (avoids aten::kl_div for TPU compatibility)
                    anchor_loss = kd_soft_ce(current_logits, anchor_logits[anchor_idx:anchor_idx+1], temperature=1.0)

                loss = loss + anchor_kl_weight * anchor_loss

        # Scale loss for accumulation
        loss = loss / accumulation_steps

        # Backward pass with optional mixed precision
        if _timing:
            _step_times[f's{step}_bwd_start'] = time.time()
            print(f"  [Step {step}/{total_steps}] Backward pass (XLA compile on step 1)...", flush=True)

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # TPU: mark_step after backward to break up XLA graph
        # Without this, XLA buffers everything until .item() causing massive compilation
        if is_xla_device(device):
            xla_mark_step()

        if _timing:
            _step_times[f's{step}_bwd_end'] = time.time()
            _bwd_time = _step_times[f's{step}_bwd_end'] - _step_times[f's{step}_bwd_start']
            print(f"  [Step {step}/{total_steps}] Backward done ({_bwd_time:.1f}s)", flush=True)

        # Accumulate loss - on TPU this causes device->host sync but mark_step mitigates it
        total_loss += loss.item() * accumulation_steps

        # Optimizer step
        if step % accumulation_steps == 0:
            if _timing:
                _step_times[f's{step}_opt_start'] = time.time()
                print(f"  [Step {step}/{total_steps}] Optimizer update...", flush=True)

            if scaler is not None:
                scaler.unscale_(optimizer)
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(params, grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(params, grad_clip)
                optimizer.step()
            scheduler.step()
            # TPU: use set_to_none=False to avoid grad None->Tensor recompilation
            # When set_to_none=True, step 1 has None grads, step 2 has Tensor grads
            # This causes XLA to see different graphs and recompile
            optimizer.zero_grad(set_to_none=not use_tpu)
            optimizer_step += 1

            # TPU: mark_step after optimizer to ensure execution
            if is_xla_device(device):
                xla_mark_step()

            if _timing:
                _step_times[f's{step}_opt_end'] = time.time()
                _opt_time = _step_times[f's{step}_opt_end'] - _step_times[f's{step}_opt_start']
                _total_step = _step_times[f's{step}_opt_end'] - _step_times[f's{step}_start']
                print(f"  [Step {step}/{total_steps}] Complete ({_opt_time:.1f}s opt, {_total_step:.1f}s total)", flush=True)

            # Logging
            if optimizer_step % log_interval == 0:
                avg_loss = total_loss / log_interval
                elapsed = time.time() - t_start
                current_time = time.time()
                current_lr = scheduler.get_last_lr()[0]

                # Smoothed ETA and tok/s using rolling window (avoids XLA compile time skew)
                eta_history.append((optimizer_step, current_time))
                if len(eta_history) >= 2:
                    old_step, old_time = eta_history[0]
                    recent_elapsed = current_time - old_time
                    recent_steps = optimizer_step - old_step
                    if recent_steps > 0 and recent_elapsed > 0:
                        sec_per_step = recent_elapsed / recent_steps
                        steps_remaining = max_steps - optimizer_step
                        eta_sec = steps_remaining * sec_per_step
                        # tok/s based on recent window
                        tok_per_sec = (recent_steps * batch_size * seq_len * accumulation_steps) / recent_elapsed
                    else:
                        eta_sec = 0
                        tok_per_sec = 0
                else:
                    # Fallback for first log
                    if optimizer_step > 0:
                        sec_per_step = elapsed / optimizer_step
                        steps_remaining = max_steps - optimizer_step
                        eta_sec = steps_remaining * sec_per_step
                        tok_per_sec = (optimizer_step * batch_size * seq_len * accumulation_steps) / elapsed
                    else:
                        eta_sec = 0
                        tok_per_sec = 0

                # Format times
                eta_min = eta_sec / 60
                eta_str = f"{eta_min:.1f}m" if eta_min < 60 else f"{eta_min/60:.1f}h"
                elapsed_min = elapsed / 60
                elapsed_str = f"{elapsed_min:.1f}m" if elapsed_min < 60 else f"{elapsed_min/60:.1f}h"

                if verbose:
                    print(f"  Step {optimizer_step}/{max_steps}: loss={avg_loss:.4f}, lr={current_lr:.2e}, tok/s={tok_per_sec:.0f}, elapsed={elapsed_str}, ETA={eta_str}")

                if use_wandb and wandb_run is not None:
                    log_dict = {
                        'step': optimizer_step,
                        'train/loss': avg_loss,
                        'train/lr': current_lr,
                        'train/tokens_per_sec': tok_per_sec,
                    }
                    # Add TPU memory stats if available
                    if use_tpu and xm is not None:
                        try:
                            mem = xm.get_memory_info(device)
                            used_gb = 0
                            total_gb = 0
                            if hasattr(mem, 'bytes_limit'):
                                total_gb = getattr(mem, 'bytes_limit', 0) / 1e9
                                used_gb = getattr(mem, 'bytes_used', 0) / 1e9
                            elif isinstance(mem, dict):
                                if "kb_total" in mem:
                                    used_gb = (mem["kb_total"] - mem.get("kb_free", 0)) / 1024 / 1024
                                    total_gb = mem["kb_total"] / 1024 / 1024
                                elif "bytes_used" in mem:
                                    used_gb = mem["bytes_used"] / 1e9
                                    total_gb = mem.get("bytes_limit", 0) / 1e9
                            log_dict['tpu/memory_used_gb'] = used_gb
                            log_dict['tpu/memory_total_gb'] = total_gb
                            if total_gb > 0:
                                log_dict['tpu/memory_pct'] = 100.0 * used_gb / total_gb
                        except Exception:
                            pass
                    wandb.log(log_dict, step=optimizer_step)

                loss_history.append(avg_loss)
                total_loss = 0.0

                # Track best checkpoint
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    if save_dir:
                        os.makedirs(save_dir, exist_ok=True)
                        # Save config.json if provided and not already present
                        config_path = os.path.join(save_dir, "config.json")
                        if config and not os.path.exists(config_path):
                            import json
                            with open(config_path, 'w') as f:
                                json.dump(config, f, indent=2)
                            if verbose:
                                print(f"    [Saved config.json]")
                        best_path = os.path.join(save_dir, "best_recovery_lora.pt")
                        if save_lora_only:
                            # Save only LoRA weights (17MB, memory efficient)
                            state = {k: v.cpu().clone() for k, v in model.state_dict().items() if 'lora_' in k}
                            torch.save(state, best_path)
                            del state
                            if verbose:
                                print(f"    [Saved best LoRA-only: {best_loss:.4f}]")
                        else:
                            # Save full model (can be used directly for inference)
                            torch.save(model.state_dict(), best_path)
                            if verbose:
                                print(f"    [Saved best: {best_loss:.4f}]")

            # Periodic checkpoint saving
            if save_steps > 0 and save_dir and optimizer_step % save_steps == 0:
                os.makedirs(save_dir, exist_ok=True)
                ckpt_path = os.path.join(save_dir, f"recovery_step{optimizer_step}.pt")
                if save_lora_only:
                    state = {k: v.cpu().clone() for k, v in model.state_dict().items() if 'lora_' in k}
                    torch.save(state, ckpt_path)
                    del state
                else:
                    torch.save(model.state_dict(), ckpt_path)
                if verbose:
                    print(f"    [Checkpoint: {ckpt_path}]")

                # Clean up old checkpoints
                if keep_checkpoints > 0:
                    import glob
                    ckpts = sorted(
                        glob.glob(os.path.join(save_dir, "recovery_step*.pt")),
                        key=os.path.getmtime
                    )
                    while len(ckpts) > keep_checkpoints:
                        os.remove(ckpts.pop(0))

            # Stop if max steps reached
            if optimizer_step >= max_steps:
                break

    elapsed = time.time() - t_start

    # Final summary
    if verbose:
        print(f"\n=== Recovery LoRA Results ===")
        final_loss_str = f"{loss_history[-1]:.4f}" if loss_history else "N/A"
        print(f"  Final loss: {final_loss_str}")
        print(f"  Best loss: {best_loss:.4f}")
        print(f"  Steps: {optimizer_step}")
        print(f"  Time: {elapsed:.1f}s")

    if use_wandb and wandb_run is not None:
        wandb.finish()

    return {
        'best_loss': best_loss,
        'final_loss': loss_history[-1] if loss_history else None,
        'best_state': best_state,
        'steps': optimizer_step,
        'time_sec': elapsed,
        'loss_history': loss_history,
    }


# ==============================================================================
# LUT METRICS (for trainable LUT monitoring)
# ==============================================================================

@torch.no_grad()
def compute_lut_metrics(model: nn.Module) -> Dict[str, float]:
    """Compute LUT metrics from CPU copies. Safe during TPU training.

    Builds LUT from CPU copies of raw_deltas parameters (no XLA sync).
    All metrics are computed in FP16 space to match ANE hardware behavior.

    Args:
        model: Model containing AnemllQATLinearV2 layers with trainable LUTs

    Returns:
        Dictionary with metrics:
            - lut/num_trainable: Number of layers with trainable LUTs
            - lut/avg_max_abs: Average max absolute value across LUTs
            - lut/avg_min_abs: Average min absolute value (should be small positive, not 0)
            - lut/total_unique_fp16: Total unique FP16 values across all LUTs
            - lut/total_duplicates: Count of FP16 duplicates (should be 0)
            - lut/all_monotonic: 1.0 if all LUTs are strictly increasing, else 0.0
    """
    from .ane_qat_linear_v2 import AnemllQATLinearV2, build_symmetric_lut16

    metrics = {
        'lut/num_trainable': 0,
        'lut/avg_max_abs': 0.0,
        'lut/avg_min_abs': 0.0,
        'lut/total_unique_fp16': 0,
        'lut/total_duplicates': 0,
        'lut/all_monotonic': 1.0,
    }

    lut_layers = []
    for name, module in model.named_modules():
        if isinstance(module, AnemllQATLinearV2) and module._lut_trainable:
            lut_layers.append((name, module))

    if not lut_layers:
        return metrics

    metrics['lut/num_trainable'] = len(lut_layers)

    max_abs_sum = 0.0
    min_abs_sum = 0.0
    all_monotonic = True

    for name, module in lut_layers:
        # Build LUT from CPU copy of params (no device sync)
        raw_deltas_cpu = module._lut_raw_deltas.detach().cpu()
        lut_cpu = build_symmetric_lut16(
            raw_deltas_cpu,
            module._lut_max_abs,
            module._lut_min_delta,
        )

        # Convert to FP16 for hardware-accurate metrics
        lut_fp16 = lut_cpu.to(torch.float16).to(torch.float32)

        max_abs_sum += lut_fp16.abs().max().item()
        min_abs_sum += lut_fp16.abs().min().item()

        # Check uniqueness in FP16
        unique_count = len(torch.unique(lut_fp16.half()))
        metrics['lut/total_unique_fp16'] += unique_count
        metrics['lut/total_duplicates'] += (len(lut_fp16) - unique_count)

        # Check strict monotonicity in FP16
        lut_fp16_actual = lut_cpu.to(torch.float16)
        if not (lut_fp16_actual[1:] > lut_fp16_actual[:-1]).all():
            all_monotonic = False

    metrics['lut/avg_max_abs'] = max_abs_sum / len(lut_layers)
    metrics['lut/avg_min_abs'] = min_abs_sum / len(lut_layers)
    metrics['lut/all_monotonic'] = 1.0 if all_monotonic else 0.0

    return metrics
