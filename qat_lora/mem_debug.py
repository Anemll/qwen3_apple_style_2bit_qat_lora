"""
TPU/XLA-safe memory debugging utilities.

Provides HBM snapshots, tensor size estimates, and XLA metrics logging
without perturbing XLA compilation or touching device tensors.

Usage:
    from qat_lora.mem_debug import MemDebugConfig, mem_log

    config = MemDebugConfig.from_args(args)
    mem_log(config, 'before_mark_step', micro_step=100, opt_step=10, extra={'batch_size': 2})
"""

import json
import os
import socket
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Set, Any, Tuple

import torch

# Schema version for JSONL records (increment on breaking changes)
SCHEMA_VERSION = 2


def _get_git_commit() -> Optional[str]:
    """Get current git commit hash (short)."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


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
    step_axis: str = 'opt'  # 'micro' or 'opt' - which step to use for filtering

    # Runtime state
    _json_file: Optional[Any] = field(default=None, repr=False)
    _start_time: float = field(default_factory=time.time, repr=False)
    _git_commit: Optional[str] = field(default=None, repr=False)
    _hostname: str = field(default_factory=socket.gethostname, repr=False)
    _pid: int = field(default_factory=os.getpid, repr=False)

    def __post_init__(self):
        if self.enabled:
            self._git_commit = _get_git_commit()

    @classmethod
    def from_args(cls, args) -> 'MemDebugConfig':
        """Create config from argparse namespace."""
        if not getattr(args, 'mem_debug', False):
            return cls(enabled=False)

        phases_str = getattr(args, 'mem_debug_phase', 'warmup,save')
        # Support both comma-separated and space-separated
        if ',' in phases_str:
            phases = set(p.strip() for p in phases_str.split(','))
        else:
            phases = set(p.strip() for p in phases_str.split())

        config = cls(
            enabled=True,
            level=getattr(args, 'mem_debug_level', 'basic'),
            phases=phases,
            interval=getattr(args, 'mem_debug_interval', 1),
            json_path=getattr(args, 'mem_debug_json', None),
            tag=getattr(args, 'mem_debug_tag', None),
            no_xla_metrics=getattr(args, 'mem_debug_no_xla_metrics', False),
            step_axis=getattr(args, 'mem_debug_step_axis', 'opt'),
        )

        # Warn about HLO mode
        if config.level == 'hlo':
            xla_flags = os.environ.get('XLA_FLAGS', '')
            if 'xla_dump_to' not in xla_flags and 'xla_dump_hlo' not in xla_flags:
                print("[MEM_DEBUG] WARNING: --mem-debug-level hlo requires XLA_FLAGS to be set BEFORE process start.")
                print("[MEM_DEBUG] Example: XLA_FLAGS='--xla_dump_to=/tmp/hlo --xla_dump_hlo_as_text' python ...")

        return config

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
    """
    Get TPU HBM memory info via xm.get_memory_info().

    Returns normalized dict with:
    - used_bytes, free_bytes, total_bytes
    - used_pct
    - raw: original dict from XLA (for future-proofing)

    Returns None if not on TPU.
    """
    try:
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        # CRITICAL: pass str(device) - some torch_xla versions need string, not device object
        raw_info = xm.get_memory_info(str(device))

        # Handle different key formats across torch_xla versions:
        # - Newer: kb_total, kb_free
        # - Older: bytes_limit, bytes_used
        if 'kb_total' in raw_info:
            total_bytes = raw_info.get('kb_total', 0) * 1024
            free_bytes = raw_info.get('kb_free', 0) * 1024
            used_bytes = total_bytes - free_bytes
        elif 'bytes_limit' in raw_info:
            total_bytes = raw_info.get('bytes_limit', 0)
            used_bytes = raw_info.get('bytes_used', 0)
            free_bytes = total_bytes - used_bytes
        else:
            # Unknown format - return raw with warning
            return {
                'used_bytes': 0,
                'free_bytes': 0,
                'total_bytes': 0,
                'used_pct': 0,
                'raw': raw_info,
                'warning': f'Unknown memory info format, keys: {list(raw_info.keys())}',
            }

        used_pct = (used_bytes / total_bytes * 100) if total_bytes > 0 else 0

        return {
            'used_bytes': used_bytes,
            'free_bytes': free_bytes,
            'total_bytes': total_bytes,
            'used_pct': round(used_pct, 1),
            'peak_bytes': raw_info.get('peak_bytes', 0),
            'raw': raw_info,  # Include raw for future-proofing
        }
    except ImportError:
        return None
    except Exception as e:
        return {'error': str(e), 'error_type': type(e).__name__}


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
    num_kv_heads: Optional[int] = None,
    head_dim: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Estimate attention O(L^2) workspace size.

    This is the WORST CASE when attention scores/probs are fully materialized.
    Flash attention and fused SDPA kernels may avoid this materialization.

    Args:
        batch_size: Batch size (B)
        num_heads: Number of query attention heads (H)
        seq_len: Sequence length (L)
        dtype: Compute dtype for attention scores
        num_kv_heads: Number of KV heads (for GQA). If None, assumes num_heads.
        head_dim: Head dimension. If None, not included in output.

    Returns:
        Dict with best/worst case estimates and explicit config values.
    """
    kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
    dtype_bytes = 2 if dtype in (torch.float16, torch.bfloat16) else 4

    # Attention scores: [B, H, L, L] for classic attention
    # For GQA with score materialization: [B, H, L, L] (query heads, not KV heads)
    scores_shape = (batch_size, num_heads, seq_len, seq_len)
    scores_bytes = _estimate_tensor_bytes(scores_shape, dtype)

    # Attention probs (same shape, after softmax)
    probs_bytes = scores_bytes

    # Best case: fused kernel, no materialization
    best_case_bytes = 0

    # Worst case: both scores and probs materialized
    worst_case_bytes = scores_bytes + probs_bytes

    result = {
        'config': {
            'B': batch_size,
            'H': num_heads,
            'kv_H': kv_heads,
            'L': seq_len,
            'dtype': str(dtype),
            'dtype_bytes': dtype_bytes,
        },
        'scores_shape': list(scores_shape),
        'best_case': {
            'bytes': best_case_bytes,
            'formatted': _format_bytes(best_case_bytes),
            'note': 'fused kernel (flash/sdpa), no score materialization',
        },
        'worst_case': {
            'bytes': worst_case_bytes,
            'formatted': _format_bytes(worst_case_bytes),
            'note': 'scores + probs fully materialized [B,H,L,L]',
        },
    }

    if head_dim is not None:
        result['config']['head_dim'] = head_dim

    return result


def estimate_logits_workspace(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    dtype: torch.dtype = torch.float32,
    materialize_full_vocab: bool = True,
) -> Dict[str, Any]:
    """
    Estimate logits tensor size.

    Note: Full [B,L,V] logits are only materialized if computing full
    softmax cross-entropy on full vocabulary. If using top-k teacher
    logits or other shortcuts, this may not apply.

    Args:
        materialize_full_vocab: If True, estimate full [B,L,V] tensor.
                               If False, returns 0 with explanatory note.
    """
    shape = (batch_size, seq_len, vocab_size)
    bytes_est = _estimate_tensor_bytes(shape, dtype)

    if materialize_full_vocab:
        return {
            'logits': {
                'shape': list(shape),
                'dtype': str(dtype),
                'bytes': bytes_est,
                'formatted': _format_bytes(bytes_est),
                'note': 'if full vocab logits materialized (full CE loss)',
            }
        }
    else:
        return {
            'logits': {
                'shape': list(shape),
                'dtype': str(dtype),
                'bytes': 0,
                'formatted': '0 B',
                'note': 'not materialized (top-k teacher or sampled loss)',
            }
        }


def mem_log(
    config: MemDebugConfig,
    point: str,
    micro_step: int = 0,
    opt_step: int = 0,
    phase: str = 'warmup',
    batch_size: int = 1,
    seq_len: int = 128,
    num_heads: int = 16,
    num_kv_heads: Optional[int] = None,
    head_dim: Optional[int] = None,
    vocab_size: int = 151936,
    dtype: torch.dtype = torch.bfloat16,
    materialize_full_vocab: bool = True,
    extra: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Log memory state at a specific point.

    Args:
        config: MemDebugConfig instance
        point: Logging point name (e.g., 'before_mark_step', 'after_forward')
        micro_step: Current micro/gradient step
        opt_step: Current optimizer step (after accumulation)
        phase: Current phase ('warmup', 'train', 'save', 'init')
        batch_size: Current batch size (for estimates)
        seq_len: Sequence length (for estimates)
        num_heads: Number of query attention heads (for estimates)
        num_kv_heads: Number of KV heads for GQA (optional)
        head_dim: Head dimension (optional)
        vocab_size: Vocabulary size (for estimates)
        dtype: Compute dtype (for estimates)
        materialize_full_vocab: Whether full vocab logits are materialized
        extra: Additional data to log

    Returns:
        Dict with logged data, or None if logging disabled
    """
    # Use step_axis to determine which step to check for interval
    step = opt_step if config.step_axis == 'opt' else micro_step

    if not config.should_log(phase, step):
        return None

    # Build record with schema version and metadata
    record = {
        'schema_version': SCHEMA_VERSION,
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'elapsed_sec': round(time.time() - config._start_time, 2),
        'point': point,
        'micro_step': micro_step,
        'opt_step': opt_step,
        'phase': phase,
        'pid': config._pid,
        'hostname': config._hostname,
    }

    if config._git_commit:
        record['git_commit'] = config._git_commit
    if config.tag:
        record['run_tag'] = config.tag

    # Level: basic - HBM snapshots
    mem_info = _get_xla_memory_info()
    if mem_info:
        record['hbm'] = mem_info

    # Level: tensors - add size estimates
    if config.level in ('tensors', 'metrics', 'hlo'):
        attn_est = estimate_attn_workspace(
            batch_size, num_heads, seq_len, dtype,
            num_kv_heads=num_kv_heads, head_dim=head_dim
        )
        logits_est = estimate_logits_workspace(
            batch_size, seq_len, vocab_size, torch.float32,
            materialize_full_vocab=materialize_full_vocab
        )
        record['estimates'] = {
            'attention': attn_est,
            'logits': logits_est,
        }

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
    micro = record.get('micro_step', 0)
    opt = record.get('opt_step', 0)
    elapsed = record.get('elapsed_sec', 0)

    parts = [f"[MEM_DEBUG] {point} micro={micro} opt={opt} t={elapsed:.1f}s"]

    # HBM info
    hbm = record.get('hbm', {})
    if hbm and 'error' not in hbm:
        used = hbm.get('used_bytes', 0)
        total = hbm.get('total_bytes', 0)
        free = hbm.get('free_bytes', 0)
        pct = hbm.get('used_pct', 0)
        parts.append(f"HBM: {_format_bytes(used)}/{_format_bytes(total)} ({pct}%) free={_format_bytes(free)}")

    # Estimates (if tensors level)
    if level in ('tensors', 'metrics', 'hlo'):
        est = record.get('estimates', {})
        attn = est.get('attention', {})
        logits = est.get('logits', {})
        if attn:
            worst = attn.get('worst_case', {})
            cfg = attn.get('config', {})
            parts.append(f"[EST] attn_worst={worst.get('formatted', '?')} (B={cfg.get('B')},H={cfg.get('H')},L={cfg.get('L')})")
        if logits:
            log_info = logits.get('logits', {})
            parts.append(f"[EST] logits={log_info.get('formatted', '?')}")

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
            cfg = model.config
            attn_impl = getattr(cfg, '_attn_implementation', 'unknown')
            use_sdpa = getattr(cfg, 'use_sdpa', None)
            num_heads = getattr(cfg, 'num_attention_heads', '?')
            num_kv_heads = getattr(cfg, 'num_key_value_heads', num_heads)
            head_dim = getattr(cfg, 'head_dim', None)
            if head_dim is None and hasattr(cfg, 'hidden_size') and num_heads != '?':
                head_dim = cfg.hidden_size // num_heads

            print(f"[MEM_DEBUG] Attention config:")
            print(f"  implementation: {attn_impl}")
            print(f"  num_attention_heads: {num_heads}")
            print(f"  num_key_value_heads: {num_kv_heads}")
            if head_dim:
                print(f"  head_dim: {head_dim}")
            if use_sdpa is not None:
                print(f"  use_sdpa: {use_sdpa}")
    except Exception as e:
        print(f"[MEM_DEBUG] Could not get attention info: {e}")
