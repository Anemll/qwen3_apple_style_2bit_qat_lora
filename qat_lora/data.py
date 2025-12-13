"""
Dataset utilities: convert instruction datasets into Qwen3 chat-template format.

We support:
- Alpaca-style (instruction, input, output)

We produce:
- input_ids
- attention_mask
- labels with prompt tokens masked to -100 (train on assistant only)

This is a standard supervised fine-tuning (SFT) pattern.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

import torch
from transformers import PreTrainedTokenizerBase


def build_alpaca_messages(example: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Convert Alpaca sample to chat messages.
    """
    instr = example.get("instruction", "")
    inp = example.get("input", "")
    out = example.get("output", "")

    user_content = instr.strip()
    if inp and inp.strip():
        user_content = user_content + "\n\n" + inp.strip()

    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": out.strip()},
    ]
    return messages


def tokenize_chat_sft(
    tokenizer: PreTrainedTokenizerBase,
    messages: List[Dict[str, str]],
    max_length: int,
    enable_thinking: bool = False,
) -> Dict[str, List[int]]:
    """
    Tokenize a (user, assistant) chat exchange such that labels are only on assistant tokens.

    We do this by building:
      prompt_text = apply_chat_template([user], add_generation_prompt=True)
      full_text   = apply_chat_template([user, assistant], add_generation_prompt=False)

    Then:
      labels = full_ids; labels[:prompt_len] = -100
    """
    # Split user and assistant
    user_msg = [messages[0]]
    full_msgs = messages

    prompt_text = tokenizer.apply_chat_template(
        user_msg,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    full_text = tokenizer.apply_chat_template(
        full_msgs,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=enable_thinking,
    )

    prompt_ids = tokenizer(
        prompt_text,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )["input_ids"]

    full = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    input_ids = full["input_ids"]
    attention_mask = full["attention_mask"]

    # Mask prompt tokens
    labels = input_ids.copy()
    prompt_len = min(len(prompt_ids), len(labels))
    labels[:prompt_len] = [-100] * prompt_len

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


@dataclass
class DataCollatorForSFT:
    """
    Pads input_ids/attention_mask/labels.
    labels are padded with -100.
    """
    tokenizer: PreTrainedTokenizerBase

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Find max length in batch
        max_len = max(len(f["input_ids"]) for f in features)

        input_ids, attention_mask, labels = [], [], []
        pad_id = self.tokenizer.pad_token_id

        for f in features:
            ids = f["input_ids"]
            am = f["attention_mask"]
            lab = f["labels"]

            pad_n = max_len - len(ids)
            input_ids.append(ids + [pad_id] * pad_n)
            attention_mask.append(am + [0] * pad_n)
            labels.append(lab + [-100] * pad_n)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
