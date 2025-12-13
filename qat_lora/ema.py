"""
Exponential Moving Average (EMA) helper.

Apple reports maintaining an EMA of weights during 2-bit QAT improved stability and metrics.
We implement a lightweight EMA tracker for our custom training loop.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Callable, Optional
import torch


@dataclass
class EMA:
    decay: float = 0.999

    def __post_init__(self):
        self.shadow: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def init(self, model: torch.nn.Module, include: Optional[Callable[[str, torch.nn.Parameter], bool]] = None):
        """
        Initialize shadow weights from model parameters.

        include:
          Optional predicate(name, param)->bool to filter which params are tracked.
          Default tracks all trainable params.
        """
        self.shadow = {}
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if include is not None and not include(name, p):
                continue
            self.shadow[name] = p.detach().clone()

    @torch.no_grad()
    def update(self, model: torch.nn.Module, include: Optional[Callable[[str, torch.nn.Parameter], bool]] = None):
        """
        Update shadow weights:
            shadow = decay * shadow + (1-decay) * param
        """
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if include is not None and not include(name, p):
                continue
            if name not in self.shadow:
                self.shadow[name] = p.detach().clone()
            else:
                self.shadow[name].mul_(self.decay).add_(p.detach(), alpha=(1.0 - self.decay))

    @torch.no_grad()
    def apply_to(self, model: torch.nn.Module):
        """
        Overwrite model params with EMA (in-place).
        Returns a backup state so you can restore later.
        """
        backup = {}
        for name, p in model.named_parameters():
            if name in self.shadow:
                backup[name] = p.detach().clone()
                p.data.copy_(self.shadow[name])
        return backup

    @torch.no_grad()
    def restore(self, model: torch.nn.Module, backup: Dict[str, torch.Tensor]):
        """Restore params from a backup returned by apply_to()."""
        for name, p in model.named_parameters():
            if name in backup:
                p.data.copy_(backup[name])
