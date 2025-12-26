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


def is_qat_linear(module: nn.Module) -> bool:
    """Check if module is AnemllQATLinear (robust against module reloading)."""
    # Use class name check to handle module caching issues
    return type(module).__name__ == 'AnemllQATLinear'


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

    Returns:
        Number of trainable parameters
    """
    # Freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze layer's quantized modules
    layer_modules = get_layer_modules(model, layer_idx)
    trainable = 0
    for name, m in layer_modules:
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
    train_weights: bool = True,
    train_scales: bool = False,
    verbose: bool = True,
    eval_before: bool = True,
    local_weight: float = 0.5,
    global_weight: float = 0.5,
    local_tokens: int = 128,
) -> dict:
    """Train a single layer with local MLP reconstruction + global KD loss.

    Loss = local_weight * LocalMLPLoss + global_weight * GlobalKDLoss

    Training modes:
    - train_weights=True, train_scales=False: Train weights only (default QAT)
    - train_weights=True, train_scales=True: Train both weights and scales
    - train_weights=False, train_scales=True: Scale optimization only (weights frozen)

    Local loss: Compares quantized MLP output to frozen FP MLP output.
    Global loss: Compares final model output to cached teacher top-k logits.

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
        train_weights: If True, train weight parameters
        train_scales: If True, train scale_A/scale_B parameters
        verbose: Print progress
        eval_before: Evaluate global loss before training (slower but informative)
        local_weight: Weight for local MLP reconstruction loss
        global_weight: Weight for global KD loss
        local_tokens: Number of tokens to sample for local loss

    Returns:
        Dict with 'layer', 'before', 'after', 'improvement', 'time_sec', etc.
    """
    import time
    t_start = time.time()

    trainable = freeze_all_except_layer(model, layer_idx, train_weights=train_weights, train_scales=train_scales)

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

    for epoch in range(epochs):
        # Create fresh dataloader each epoch for proper shuffling
        dataset = KDCacheDataset(cache_dir, shuffle=True)
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

        for i, batch in enumerate(dataloader):
            # Global KD loss (forward pass also captures MLP I/O via hook)
            global_loss = compute_kd_loss_batch(model, batch, device, temperature, no_grad=False)

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
        if use_local:
            print(f'  [Local Loss]:   {first_local:.4f} -> {last_local:.4f} (Δ={local_improvement:.4f})')
        print(f'  [Global Loss]:  {first_global:.4f} -> {last_global:.4f} (Δ={global_improvement:.4f})')
        if loss_before is not None:
            improvement = loss_before - loss_after
            pct = 100 * improvement / loss_before if loss_before > 0 else 0
            print(f'  [Eval KD]:      {loss_before:.4f} -> {loss_after:.4f} (Δ={improvement:.4f}, {pct:.1f}%)')
        print(f'  [Time]:         {layer_time:.1f}s')

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
    grad_accum: int = 4,
    temperature: float = 2.0,
    train_weights: bool = True,
    train_scales: bool = False,
    verbose: bool = True,
    eval_before: bool = True,
    local_weight: float = 0.5,
    global_weight: float = 0.5,
    local_tokens: int = 128,
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
        epochs_per_layer: Epochs per layer
        grad_accum: Gradient accumulation steps
        temperature: Distillation temperature
        train_weights: If True, train weight parameters
        train_scales: If True, train scale_A/scale_B parameters
        verbose: Print progress
        eval_before: Evaluate global loss before each layer
        local_weight: Weight for local reconstruction loss
        global_weight: Weight for global KD loss
        local_tokens: Number of tokens to sample for local loss

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

    if verbose:
        print(f'Training {num_layers} layers (mode={mode})...')
        print(f'Cache: {cache_dir}')
        print(f'Batch size: {batch_size}, Grad accum: {grad_accum}')
        print(f'LR: {lr}, Epochs per layer: {epochs_per_layer}')

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
            grad_accum=grad_accum,
            temperature=temperature,
            train_weights=train_weights,
            train_scales=train_scales,
            verbose=verbose,
            eval_before=eval_before,
            local_weight=local_weight,
            global_weight=global_weight,
            local_tokens=local_tokens,
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
