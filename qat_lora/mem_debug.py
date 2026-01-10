"""
TPU/XLA-safe memory debugging utilities.

Provides HBM snapshots, tensor size estimates, and XLA metrics logging
without perturbing XLA compilation or touching device tensors.

Usage:
    from qat_lora.mem_debug import MemDebugConfig, mem_log

    config = MemDebugConfig.from_args(args)
    mem_log(config, 'before_mark_step', step=10, extra={'batch_size': 2})
"""

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Set, Any

import torch


@dataclass
class MemDebugConfig:
    """Configuration for TPU/XLA-safe memory debugging."""

    enabled: bool = False
    level: str = 'basic'  # 'basic', 'tensors', 'metrics', 'hlo'
    phases: Set[str] = field(default_factory=lambda: {'warmup', 'save'})
    interval: int = 1  # Log every N optimizer steps
    json_path: Optional[str] = None  # Path to append JSONL records
    tag: Optional[str] = None  # Optional run tag
    no_xla_metrics: bool = False  # Skip XLA metrics calls

    # Runtime state
    _json_file: Optional[Any] = field(default=None, repr=False)
    _start_time: float = field(default_factory=time.time, repr=False)

    @classmethod
    def from_args(cls, args) -> 'MemDebugConfig':
        """Create config from argparse namespace."""
        if not getattr(args, 'mem_debug', False):
            return cls(enabled=False)

        phases_str = getattr(args, 'mem_debug_phase', 'warmup,save')
        phases = set(p.strip() for p in phases_str.split(','))

        return cls(
            enabled=True,
            level=getattr(args, 'mem_debug_level', 'basic'),
            phases=phases,
            interval=getattr(args, 'mem_debug_interval', 1),
            json_path=getattr(args, 'mem_debug_json', None),
            tag=getattr(args, 'mem_debug_tag', None),
            no_xla_metrics=getattr(args, 'mem_debug_no_xla_metrics', False),
        )

    def should_log(self, phase: str, step: int = 0) -> bool:
        """Check if we should log at this phase/step."""
        if not self.enabled:
            return False
        if 'all' in self.phases or phase in self.phases:
            if phase == 'train':
                return step % self.interval == 0
            return True
        return False


def _get_xla_memory_info() -> Optional[Dict[str, Any]]:
    """Get TPU HBM memory info via xm.get_memory_info(). Returns None if not on TPU."""
    try:
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        info = xm.get_memory_info(device)
        return {
            'hbm_total_bytes': info.get('kb_total', 0) * 1024,
            'hbm_free_bytes': info.get('kb_free', 0) * 1024,
            'hbm_used_bytes': (info.get('kb_total', 0) - info.get('kb_free', 0)) * 1024,
            'peak_bytes': info.get('peak_bytes', 0),
        }
    except ImportError:
        return None
    except Exception as e:
        return {'error': str(e)}


def _get_xla_metrics() -> Optional[Dict[str, Any]]:
    """Get XLA compilation metrics. Returns None if not available."""
    try:
        import torch_xla.debug.metrics as met
        report = met.metrics_report()

        # Parse key metrics from report
        metrics = {}
        for line in report.split('\n'):
            if ':' in line:
                key, val = line.split(':', 1)
                key = key.strip()
                val = val.strip()
                if key in ('UncachedCompile', 'CachedCompile', 'CompileTime', 'ExecuteTime'):
                    # Try to parse numeric value
                    try:
                        if 'ms' in val:
                            metrics[key] = float(val.replace('ms', '').strip())
                        else:
                            metrics[key] = int(val)
                    except ValueError:
                        metrics[key] = val
        return metrics
    except ImportError:
        return None
    except Exception as e:
        return {'error': str(e)}


def _estimate_tensor_bytes(shape: tuple, dtype: torch.dtype) -> int:
    """Estimate bytes for a tensor without touching device."""
    numel = 1
    for dim in shape:
        numel *= dim

    bytes_per_elem = {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.int64: 8,
        torch.int32: 4,
        torch.int16: 2,
        torch.int8: 1,
        torch.bool: 1,
    }.get(dtype, 4)

    return numel * bytes_per_elem


def _format_bytes(b: int) -> str:
    """Format bytes as human-readable string."""
    if b >= 1024 ** 3:
        return f"{b / 1024**3:.2f} GiB"
    elif b >= 1024 ** 2:
        return f"{b / 1024**2:.1f} MiB"
    elif b >= 1024:
        return f"{b / 1024:.1f} KiB"
    return f"{b} B"


def estimate_attn_workspace(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    dtype: torch.dtype = torch.bfloat16,
) -> Dict[str, Any]:
    """
    Estimate attention O(L^2) workspace size.

    Returns dict with estimates for:
    - attn_scores: B * H * L * L (worst case)
    - attn_probs: B * H * L * L (after softmax)
    """
    # Attention scores: [B, H, L, L]
    scores_shape = (batch_size, num_heads, seq_len, seq_len)
    scores_bytes = _estimate_tensor_bytes(scores_shape, dtype)

    # Attention probs (same shape, after softmax)
    probs_bytes = scores_bytes

    return {
        'attn_scores': {
            'shape': scores_shape,
            'dtype': str(dtype),
            'bytes': scores_bytes,
            'formatted': _format_bytes(scores_bytes),
        },
        'attn_probs': {
            'shape': scores_shape,
            'dtype': str(dtype),
            'bytes': probs_bytes,
            'formatted': _format_bytes(probs_bytes),
        },
        'total_attn_workspace': {
            'bytes': scores_bytes + probs_bytes,
            'formatted': _format_bytes(scores_bytes + probs_bytes),
        },
    }


def estimate_logits_workspace(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, Any]:
    """Estimate logits tensor size."""
    shape = (batch_size, seq_len, vocab_size)
    bytes_est = _estimate_tensor_bytes(shape, dtype)

    return {
        'logits': {
            'shape': shape,
            'dtype': str(dtype),
            'bytes': bytes_est,
            'formatted': _format_bytes(bytes_est),
        }
    }


def mem_log(
    config: MemDebugConfig,
    point: str,
    step: int = 0,
    phase: str = 'warmup',
    batch_size: int = 1,
    seq_len: int = 128,
    num_heads: int = 16,
    vocab_size: int = 151936,
    dtype: torch.dtype = torch.bfloat16,
    extra: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Log memory state at a specific point.

    Args:
        config: MemDebugConfig instance
        point: Logging point name (e.g., 'before_mark_step', 'after_forward')
        step: Current optimizer step
        phase: Current phase ('warmup', 'train', 'save')
        batch_size: Current batch size (for estimates)
        seq_len: Sequence length (for estimates)
        num_heads: Number of attention heads (for estimates)
        vocab_size: Vocabulary size (for estimates)
        dtype: Compute dtype (for estimates)
        extra: Additional data to log

    Returns:
        Dict with logged data, or None if logging disabled
    """
    if not config.should_log(phase, step):
        return None

    record = {
        'timestamp': time.time(),
        'elapsed': time.time() - config._start_time,
        'point': point,
        'step': step,
        'phase': phase,
    }

    if config.tag:
        record['tag'] = config.tag

    # Level: basic - HBM snapshots
    mem_info = _get_xla_memory_info()
    if mem_info:
        record['hbm'] = mem_info

    # Level: tensors - add size estimates
    if config.level in ('tensors', 'metrics', 'hlo'):
        record['estimates'] = {
            'batch_size': batch_size,
            'seq_len': seq_len,
            'num_heads': num_heads,
            'vocab_size': vocab_size,
            'dtype': str(dtype),
        }
        record['estimates'].update(estimate_attn_workspace(batch_size, num_heads, seq_len, dtype))
        record['estimates'].update(estimate_logits_workspace(batch_size, seq_len, vocab_size, torch.float32))

    # Level: metrics - add XLA compilation stats
    if config.level in ('metrics', 'hlo') and not config.no_xla_metrics:
        xla_metrics = _get_xla_metrics()
        if xla_metrics:
            record['xla_metrics'] = xla_metrics

    # Add extra data
    if extra:
        record['extra'] = extra

    # Print summary
    _print_mem_summary(record, config.level)

    # Write to JSON file if configured
    if config.json_path:
        try:
            with open(config.json_path, 'a') as f:
                f.write(json.dumps(record) + '\n')
        except Exception as e:
            print(f"[MEM_DEBUG] Warning: Failed to write JSON: {e}")

    return record


def _print_mem_summary(record: Dict[str, Any], level: str):
    """Print a compact memory summary line."""
    point = record.get('point', '?')
    step = record.get('step', 0)
    elapsed = record.get('elapsed', 0)

    parts = [f"[MEM_DEBUG] {point} step={step} t={elapsed:.1f}s"]

    # HBM info
    hbm = record.get('hbm', {})
    if hbm and 'error' not in hbm:
        used = hbm.get('hbm_used_bytes', 0)
        total = hbm.get('hbm_total_bytes', 0)
        free = hbm.get('hbm_free_bytes', 0)
        pct = (used / total * 100) if total > 0 else 0
        parts.append(f"HBM: {_format_bytes(used)}/{_format_bytes(total)} ({pct:.1f}%) free={_format_bytes(free)}")

    # Estimates (if tensors level)
    if level in ('tensors', 'metrics', 'hlo'):
        est = record.get('estimates', {})
        attn = est.get('total_attn_workspace', {})
        logits = est.get('logits', {})
        if attn:
            parts.append(f"[EST] attn_workspace={attn.get('formatted', '?')}")
        if logits:
            parts.append(f"[EST] logits={logits.get('formatted', '?')}")

    # XLA metrics (if metrics level)
    if level in ('metrics', 'hlo'):
        xla = record.get('xla_metrics', {})
        if xla and 'error' not in xla:
            uncached = xla.get('UncachedCompile', 0)
            cached = xla.get('CachedCompile', 0)
            if uncached or cached:
                parts.append(f"XLA: uncached={uncached} cached={cached}")

    print(' | '.join(parts))


def print_attn_info(model) -> None:
    """Print attention implementation info from model config."""
    try:
        if hasattr(model, 'config'):
            attn_impl = getattr(model.config, '_attn_implementation', 'unknown')
            use_sdpa = getattr(model.config, 'use_sdpa', None)
            print(f"[MEM_DEBUG] Attention implementation: {attn_impl}")
            if use_sdpa is not None:
                print(f"[MEM_DEBUG] use_sdpa: {use_sdpa}")
    except Exception as e:
        print(f"[MEM_DEBUG] Could not get attention info: {e}")
