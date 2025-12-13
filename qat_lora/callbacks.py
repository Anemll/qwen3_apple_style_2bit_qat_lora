"""
Trainer callbacks:
- EMACallback: updates EMA every optimization step and optionally saves EMA weights at end.
"""

from __future__ import annotations
from typing import Optional

import os
import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from .ema import EMA


class EMACallback(TrainerCallback):
    def __init__(self, ema: EMA, save_ema_path: Optional[str] = None):
        self.ema = ema
        self.save_ema_path = save_ema_path
        self._initialized = False

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        model = kwargs["model"]
        if not self._initialized:
            self.ema.init(model)
            self._initialized = True

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        model = kwargs["model"]
        # called after optimizer step in Trainer
        self.ema.update(model)

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if self.save_ema_path is None:
            return
        model = kwargs["model"]
        os.makedirs(self.save_ema_path, exist_ok=True)
        # Save EMA shadow weights as a torch checkpoint
        torch.save(self.ema.shadow, os.path.join(self.save_ema_path, "ema_shadow.pt"))
