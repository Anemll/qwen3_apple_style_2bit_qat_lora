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

import random
from pathlib import Path
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from torch.optim import AdamW

from .ane_qat_linear import AnemllQATLinear


def compute_kd_loss_batch(
    model: nn.Module,
    batch: dict,
    device: torch.device,
    temperature: float = 2.0,
    no_grad: bool = False,
) -> torch.Tensor:
    """Compute KD loss for a batch using memory-efficient approach.

    Args:
        model: The model (must have .model and .lm_head attributes)
        batch: Dict with input_ids, attention_mask, topk_idx, topk_logits
        device: Device to run on
        temperature: Distillation temperature
        no_grad: If True, wrap forward pass in no_grad (for evaluation).
                 During training, set to False to allow gradients.

    Returns:
        KL divergence loss scalar
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

    # KL divergence with temperature
    t_logits = topk_logits[:, :seq_len, :]
    teacher_probs = F.softmax(t_logits / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_topk / temperature, dim=-1)
    kl = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)

    return kl


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


def get_layer_modules(
    model: nn.Module,
    layer_idx: int,
) -> List[Tuple[str, AnemllQATLinear]]:
    """Get all AnemllQATLinear modules in a specific layer."""
    layer = model.model.layers[layer_idx]
    modules = []
    for name, m in layer.named_modules():
        if isinstance(m, AnemllQATLinear):
            modules.append((f'layers.{layer_idx}.{name}', m))
    return modules


def freeze_all_except_layer(
    model: nn.Module,
    layer_idx: int,
    train_scales: bool = False,
) -> int:
    """Freeze all parameters except the specified layer's AnemllQATLinear weights.

    Args:
        model: The model
        layer_idx: Layer index to unfreeze
        train_scales: If True, also train scale_A/scale_B parameters

    Returns:
        Number of trainable parameters
    """
    # Freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze layer's quantized weights
    layer_modules = get_layer_modules(model, layer_idx)
    trainable = 0
    for name, m in layer_modules:
        # Train the main weight
        m.weight.requires_grad = True
        trainable += m.weight.numel()

        # Optionally train scales
        if train_scales:
            if m.scale_A is not None:
                m.scale_A.requires_grad = True
                trainable += m.scale_A.numel()
            if m.scale_B is not None:
                m.scale_B.requires_grad = True
                trainable += m.scale_B.numel()
        else:
            if m.scale_A is not None:
                m.scale_A.requires_grad = False
            if m.scale_B is not None:
                m.scale_B.requires_grad = False

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
    grad_accum: int = 4,
    temperature: float = 2.0,
    train_scales: bool = False,
    verbose: bool = True,
) -> float:
    """Train a single layer with KD loss.

    Args:
        model: The model to train
        layer_idx: Layer index to train
        cache_dir: Path to KD cache directory
        device: Device to run on
        batch_size: Batch size
        lr: Learning rate
        epochs: Number of epochs
        grad_accum: Gradient accumulation steps
        temperature: Distillation temperature
        train_scales: If True, also train scale_A/scale_B
        verbose: Print progress

    Returns:
        Final evaluation loss for this layer
    """
    trainable = freeze_all_except_layer(model, layer_idx, train_scales=train_scales)
    if verbose:
        print(f'\n=== Layer {layer_idx} === ({trainable:,} trainable params)')

    # Get trainable params
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(params, lr=lr)

    model.train()
    total_loss = 0.0
    steps = 0

    for epoch in range(epochs):
        # Create fresh dataloader each epoch for proper shuffling
        dataset = KDCacheDataset(cache_dir, shuffle=True)
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

        for i, batch in enumerate(dataloader):
            loss = compute_kd_loss_batch(model, batch, device, temperature, no_grad=False)
            loss = loss / grad_accum
            loss.backward()

            if (i + 1) % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()
                steps += 1
                total_loss += loss.item() * grad_accum

                if verbose and steps % 10 == 0:
                    avg = total_loss / steps
                    print(f'  Step {steps}, Loss: {avg:.4f}')

    # Final eval
    model.eval()
    eval_loss = evaluate_kd_loss(model, cache_dir, device, num_samples=20, temperature=temperature)
    if verbose:
        print(f'  Layer {layer_idx} done. Eval Loss: {eval_loss:.4f}')

    return eval_loss


def train_all_layers(
    model: nn.Module,
    cache_dir: str,
    device: torch.device,
    batch_size: int = 4,
    lr: float = 2e-5,
    epochs_per_layer: int = 1,
    grad_accum: int = 4,
    temperature: float = 2.0,
    train_scales: bool = False,
    verbose: bool = True,
) -> List[float]:
    """Train all layers sequentially.

    Args:
        model: The model to train
        cache_dir: Path to KD cache directory
        device: Device to run on
        batch_size: Batch size
        lr: Learning rate
        epochs_per_layer: Epochs per layer
        grad_accum: Gradient accumulation steps
        temperature: Distillation temperature
        train_scales: If True, also train scale_A/scale_B
        verbose: Print progress

    Returns:
        List of eval losses for each layer
    """
    import time

    num_layers = len(model.model.layers)
    if verbose:
        print(f'Training {num_layers} layers...')
        print(f'Cache: {cache_dir}')
        print(f'Batch size: {batch_size}, Grad accum: {grad_accum}')
        print(f'LR: {lr}, Epochs per layer: {epochs_per_layer}')

    layer_losses = []
    t0 = time.time()

    for layer_idx in range(num_layers):
        loss = train_layer(
            model, layer_idx, cache_dir, device,
            batch_size=batch_size,
            lr=lr,
            epochs=epochs_per_layer,
            grad_accum=grad_accum,
            temperature=temperature,
            train_scales=train_scales,
            verbose=verbose,
        )
        layer_losses.append(loss)

    if verbose:
        print(f'\nLayer-by-layer training complete in {time.time() - t0:.1f}s')
        print(f'Final losses: {[f"{l:.4f}" for l in layer_losses]}')

    return layer_losses
