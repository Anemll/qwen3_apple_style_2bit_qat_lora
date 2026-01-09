"""
Auto Snap+Freeze rank_magnitude during training.

This module provides utilities for automatically detecting when rank_magnitude
values have stabilized during training and applying FP16 snap + freeze.

Key Features:
- CPU-only audit (no TPU/XLA tensor reads during training)
- Audit at save checkpoints only (no per-step overhead)
- One-time snap+freeze when stability detected
- Intended for initial training, not fine-tuning

Usage:
    state = AutoSnapState(
        enabled=True,
        target='mlp',
        threshold=0.05,
        patience=2,
        start_step=200,
        min_saves=2,
    )

    # At each save checkpoint:
    if state.should_audit(optimizer_step):
        decision = audit_mags_movement(state, cpu_state_dict, optimizer_step)
        if decision['should_freeze']:
            apply_auto_snap_and_freeze(model, optimizer, state.target, verbose=True)
            state.auto_frozen = True
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import json
import os


@dataclass
class AutoSnapState:
    """State machine for auto snap+freeze of rank_magnitude."""

    # Configuration (from CLI)
    enabled: bool = False
    target: str = 'mlp'  # 'mlp' or 'all'
    threshold: float = 0.05  # Max abs delta between saves
    patience: int = 2  # Consecutive stable saves required
    start_step: int = 0  # Don't audit before this step
    min_saves: int = 2  # Minimum saves before eligible
    dry_run: bool = False  # Audit + log but don't freeze
    log_json: bool = False  # Write audit JSON file

    # Internal state (managed during training)
    stable_count: int = 0
    num_audits: int = 0
    auto_frozen: bool = False
    prev_mags_cpu: Optional[Dict[str, torch.Tensor]] = None
    last_audit_step: Optional[int] = None
    audit_history: List[Dict[str, Any]] = field(default_factory=list)
    disable_reason: Optional[str] = None  # Set when auto-snap is disabled mid-run

    def should_audit(self, optimizer_step: int) -> bool:
        """Check if we should audit at this save checkpoint."""
        if not self.enabled:
            return False
        if self.auto_frozen:
            return False
        if optimizer_step < self.start_step:
            return False
        return True

    def reset(self):
        """Reset internal state (for testing or re-runs)."""
        self.stable_count = 0
        self.num_audits = 0
        self.auto_frozen = False
        self.prev_mags_cpu = None
        self.last_audit_step = None
        self.audit_history = []


def extract_rank_magnitudes(
    state_dict: Dict[str, torch.Tensor],
    target: str = 'mlp',
) -> Dict[str, torch.Tensor]:
    """
    Extract rank_magnitude tensors from a CPU state dict.

    Args:
        state_dict: Model state dict (must be on CPU)
        target: 'mlp' for MLP layers only, 'all' for all V2 layers

    Returns:
        Dict mapping key -> FP32 CPU tensor
    """
    mags = {}
    mlp_proj_names = ('gate_proj', 'up_proj', 'down_proj')

    for key, tensor in state_dict.items():
        if not key.endswith('.rank_magnitude'):
            continue

        # Filter by target
        if target == 'mlp':
            if not any(proj in key for proj in mlp_proj_names):
                continue

        # Convert to FP32 CPU for comparison
        mags[key] = tensor.detach().float().cpu()

    return mags


def compute_movement_metrics(
    curr_mags: Dict[str, torch.Tensor],
    prev_mags: Dict[str, torch.Tensor],
) -> Dict[str, Any]:
    """
    Compute movement metrics between two checkpoint's rank_magnitudes.

    Args:
        curr_mags: Current checkpoint's mags (FP32 CPU)
        prev_mags: Previous checkpoint's mags (FP32 CPU)

    Returns:
        Dict with:
            - max_abs_delta: Maximum absolute change across all mags
            - mean_abs_delta: Mean absolute change
            - top_k_deltas: List of (key, delta) for top 10 movers
            - num_keys: Number of keys compared
    """
    deltas = []

    for key in curr_mags:
        if key not in prev_mags:
            continue

        diff = (curr_mags[key] - prev_mags[key]).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        deltas.append({
            'key': key,
            'max_diff': max_diff,
            'mean_diff': mean_diff,
        })

    if not deltas:
        return {
            'max_abs_delta': 0.0,
            'mean_abs_delta': 0.0,
            'top_k_deltas': [],
            'num_keys': 0,
        }

    # Sort by max diff
    deltas.sort(key=lambda x: x['max_diff'], reverse=True)

    max_abs_delta = deltas[0]['max_diff'] if deltas else 0.0
    mean_abs_delta = sum(d['mean_diff'] for d in deltas) / len(deltas)

    return {
        'max_abs_delta': max_abs_delta,
        'mean_abs_delta': mean_abs_delta,
        'top_k_deltas': deltas[:10],  # Top 10 movers
        'num_keys': len(deltas),
    }


def compute_fp16_snap_distance(mags: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """
    Compute how far current mags are from FP16-representable values.

    This is diagnostic only - shows what snap would change.

    Args:
        mags: Dict of rank_magnitude tensors (FP32 CPU)

    Returns:
        Dict with max_snap_diff, mean_snap_diff, etc.
    """
    snap_diffs = []

    for key, tensor in mags.items():
        snapped = tensor.cpu().half().float()
        diff = (tensor - snapped).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        snap_diffs.append({
            'key': key,
            'max_diff': max_diff,
            'mean_diff': mean_diff,
        })

    if not snap_diffs:
        return {
            'max_snap_diff': 0.0,
            'mean_snap_diff': 0.0,
            'num_unsnapped': 0,
        }

    max_snap_diff = max(d['max_diff'] for d in snap_diffs)
    mean_snap_diff = sum(d['mean_diff'] for d in snap_diffs) / len(snap_diffs)
    num_unsnapped = sum(1 for d in snap_diffs if d['max_diff'] > 0)

    return {
        'max_snap_diff': max_snap_diff,
        'mean_snap_diff': mean_snap_diff,
        'num_unsnapped': num_unsnapped,
        'total_keys': len(snap_diffs),
    }


def audit_mags_movement(
    state: AutoSnapState,
    cpu_state_dict: Dict[str, torch.Tensor],
    optimizer_step: int,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Audit rank_magnitude movement at a save checkpoint.

    This function is called ONLY at save checkpoints and uses ONLY CPU tensors.

    Args:
        state: AutoSnapState instance
        cpu_state_dict: Model state dict on CPU
        optimizer_step: Current optimizer step
        verbose: Print progress

    Returns:
        Dict with:
            - should_freeze: Whether to trigger snap+freeze
            - movement_metrics: Movement metrics dict
            - snap_metrics: FP16 snap distance metrics
    """
    # Extract current mags
    curr_mags = extract_rank_magnitudes(cpu_state_dict, target=state.target)

    if not curr_mags:
        if verbose:
            print(f"[AutoSnap] Warning: No rank_magnitude tensors found for target='{state.target}'")
        return {'should_freeze': False, 'movement_metrics': None, 'snap_metrics': None}

    # Compute snap distance (diagnostic)
    snap_metrics = compute_fp16_snap_distance(curr_mags)

    # First audit - just store prev and return
    if state.prev_mags_cpu is None:
        state.prev_mags_cpu = curr_mags
        state.num_audits = 1
        state.last_audit_step = optimizer_step

        if verbose:
            print(f"[AutoSnap] First audit at step {optimizer_step}")
            print(f"  Target: {state.target} ({len(curr_mags)} layers)")
            print(f"  FP16 snap distance: max={snap_metrics['max_snap_diff']:.6f}")

        return {
            'should_freeze': False,
            'movement_metrics': None,
            'snap_metrics': snap_metrics,
        }

    # Guardrail: Check key count consistency
    prev_count = len(state.prev_mags_cpu)
    curr_count = len(curr_mags)
    if prev_count != curr_count:
        reason = f"key_count_mismatch:{prev_count}→{curr_count}"
        if verbose:
            print(f"[AutoSnap] ABORT: Key count changed ({prev_count} → {curr_count})")
            print(f"[AutoSnap] Auto-snap disabled for this run")
        state.enabled = False  # Disable for rest of run
        state.disable_reason = reason
        return {'should_freeze': False, 'movement_metrics': None, 'snap_metrics': snap_metrics, 'disabled': True, 'disable_reason': reason}

    # Compute movement metrics
    movement = compute_movement_metrics(curr_mags, state.prev_mags_cpu)
    state.num_audits += 1
    state.last_audit_step = optimizer_step

    # Guardrail: Check for NaN/Inf in metrics
    import math
    if math.isnan(movement['max_abs_delta']) or math.isinf(movement['max_abs_delta']):
        reason = f"nan_inf_delta:{movement['max_abs_delta']}"
        if verbose:
            print(f"[AutoSnap] ABORT: NaN/Inf detected in movement metrics")
            print(f"[AutoSnap] Auto-snap disabled for this run")
        state.enabled = False  # Disable for rest of run
        state.disable_reason = reason
        return {'should_freeze': False, 'movement_metrics': movement, 'snap_metrics': snap_metrics, 'disabled': True, 'disable_reason': reason}

    # Check if stable
    is_stable = movement['max_abs_delta'] < state.threshold

    if is_stable:
        state.stable_count += 1
    else:
        state.stable_count = 0  # Reset on any large movement

    # Store for history
    audit_record = {
        'step': optimizer_step,
        'num_audits': state.num_audits,
        'max_abs_delta': movement['max_abs_delta'],
        'mean_abs_delta': movement['mean_abs_delta'],
        'is_stable': is_stable,
        'stable_count': state.stable_count,
        'snap_max_diff': snap_metrics['max_snap_diff'],
    }
    state.audit_history.append(audit_record)

    # Print audit results
    if verbose:
        status = "STABLE" if is_stable else "MOVING"
        print(f"[AutoSnap] Audit #{state.num_audits} at step {optimizer_step}: {status}")
        print(f"  Keys: {movement['num_keys']} layers evaluated")
        print(f"  Movement: max_delta={movement['max_abs_delta']:.6f} (threshold={state.threshold})")
        print(f"  Stable count: {state.stable_count}/{state.patience}")
        if movement['top_k_deltas'] and len(movement['top_k_deltas']) >= 3:
            print(f"  Top movers:")
            for i, d in enumerate(movement['top_k_deltas'][:3]):
                layer_name = d['key'].replace('.rank_magnitude', '').split('.')[-1]
                print(f"    {i+1}. {layer_name}: Δ={d['max_diff']:.6f}")
        elif movement['top_k_deltas']:
            top = movement['top_k_deltas'][0]
            print(f"  Top mover: {top['key'].split('.')[-2]} (Δ={top['max_diff']:.6f})")

    # Check if should freeze
    should_freeze = (
        state.stable_count >= state.patience and
        state.num_audits >= state.min_saves
    )

    if should_freeze and verbose:
        print(f"[AutoSnap] *** TRIGGER: {state.stable_count} consecutive stable audits ***")
        if state.dry_run:
            print(f"[AutoSnap] Dry run mode - would freeze but skipping")

    # Update prev for next audit
    state.prev_mags_cpu = curr_mags

    return {
        'should_freeze': should_freeze and not state.dry_run,
        'movement_metrics': movement,
        'snap_metrics': snap_metrics,
    }


def apply_auto_snap_and_freeze(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    target: str = 'mlp',
    verbose: bool = True,
) -> Tuple[int, torch.optim.Optimizer]:
    """
    Apply FP16 snap and freeze to rank_magnitude parameters.

    This modifies the model in-place and rebuilds the optimizer.

    IMPORTANT: FP16 snap is done on CPU to avoid XLA lazy tensor optimizations.

    Args:
        model: Model with AnemllQATLinearV2 layers
        optimizer: Current optimizer (will be rebuilt)
        target: 'mlp' for MLP layers only, 'all' for all V2 layers
        verbose: Print progress

    Returns:
        Tuple of (num_frozen, new_optimizer)
    """
    mlp_proj_names = ('gate_proj', 'up_proj', 'down_proj')

    frozen_count = 0
    frozen_params = set()

    for name, module in model.named_modules():
        if type(module).__name__ != 'AnemllQATLinearV2':
            continue

        if not hasattr(module, 'rank_magnitude') or module.rank_magnitude is None:
            continue

        # Filter by target
        if target == 'mlp':
            if not any(proj in name for proj in mlp_proj_names):
                continue

        # CPU-based FP16 snap (critical for TPU/XLA)
        with torch.no_grad():
            orig_device = module.rank_magnitude.data.device
            orig_dtype = module.rank_magnitude.data.dtype

            # Move to CPU, snap to FP16, convert back
            snapped = module.rank_magnitude.data.cpu().half().float()

            # Copy back to device
            module.rank_magnitude.data.copy_(snapped.to(orig_device))

        # Freeze
        module.rank_magnitude.requires_grad = False
        frozen_params.add(id(module.rank_magnitude))
        frozen_count += 1

    if verbose:
        print(f"[AutoSnap] Snapped and froze {frozen_count} rank_magnitude tensors (target={target})")

    # Rebuild optimizer to exclude frozen params
    # Get current optimizer settings
    lr = optimizer.param_groups[0]['lr']
    weight_decay = optimizer.param_groups[0].get('weight_decay', 0.0)
    betas = optimizer.param_groups[0].get('betas', (0.9, 0.999))
    eps = optimizer.param_groups[0].get('eps', 1e-8)

    # Build new param list excluding frozen
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if verbose:
        print(f"[AutoSnap] Rebuilding optimizer with {len(trainable_params)} trainable params")
        print(f"[AutoSnap] Note: This may trigger one TPU recompilation")

    # Create new optimizer
    new_optimizer = torch.optim.AdamW(
        trainable_params,
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps,
    )

    return frozen_count, new_optimizer


def save_audit_json(
    state: AutoSnapState,
    save_dir: str,
    optimizer_step: int,
):
    """Save audit history to JSON file."""
    if not state.log_json:
        return

    json_path = os.path.join(save_dir, f"auto_snap_audit_step{optimizer_step}.json")

    data = {
        'config': {
            'target': state.target,
            'threshold': state.threshold,
            'patience': state.patience,
            'start_step': state.start_step,
            'min_saves': state.min_saves,
            'dry_run': state.dry_run,
        },
        'state': {
            'num_audits': state.num_audits,
            'stable_count': state.stable_count,
            'auto_frozen': state.auto_frozen,
            'last_audit_step': state.last_audit_step,
        },
        'history': state.audit_history,
    }

    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)


def validate_auto_snap_config(
    auto_snap_enabled: bool,
    freeze_mags: bool,
    freeze_mags_mlp: bool,
    freeze_all: bool,
    g_only: bool,
    save_steps: int,
) -> Tuple[bool, str]:
    """
    Validate auto-snap configuration for conflicts.

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not auto_snap_enabled:
        return True, ""

    # Check conflicts
    if freeze_mags:
        return False, "--auto-snap-mags conflicts with --freeze-mags"
    if freeze_mags_mlp:
        return False, "--auto-snap-mags conflicts with --freeze-mags-mlp"
    if freeze_all:
        return False, "--auto-snap-mags conflicts with --freeze-all"
    if g_only:
        return False, "--auto-snap-mags conflicts with --g-only (auto-snap targets mags)"
    if save_steps <= 0:
        return False, "--auto-snap-mags requires --save-steps > 0 (audit happens at saves)"

    return True, ""
