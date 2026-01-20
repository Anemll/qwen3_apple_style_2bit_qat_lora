# Claude Code Project Notes

## Tokenizer and Chat Template Requirements

### Qwen3 Thinking Mode (CRITICAL)

**IMPORTANT**: Qwen3 models have a "thinking" mode that outputs reasoning in `<think>...</think>` tags. To disable this mode, you MUST use the official `enable_thinking=False` parameter in `apply_chat_template()`.

**CORRECT approach** (official Qwen3 method):
```python
# Use enable_thinking=False - this pre-fills <think>\n\n</think>\n\n (17 tokens)
messages = [{"role": "user", "content": prompt}]
template_kwargs = {
    "tokenize": True,
    "return_tensors": "pt",
    "add_generation_prompt": True,
}
if no_think:
    template_kwargs["enable_thinking"] = False

input_ids = tokenizer.apply_chat_template(messages, **template_kwargs)
```

**WRONG approach** (do NOT use):
```python
# ❌ WRONG: Adding /no_think prefix to prompt content
messages = [{"role": "user", "content": f"/no_think {prompt}"}]  # WRONG!
```

**Why this matters**:
- The `/no_think` prefix approach produces different token sequences (13 tokens vs 17 tokens)
- This causes template mismatch between scripts, leading to inconsistent model behavior
- When comparing PyTorch vs CoreML outputs, template differences appear as model divergence
- The official `enable_thinking=False` approach matches `chat.py` behavior

**Tokenized prompt with `enable_thinking=False`**:
```
<|im_start|>user
What is AI?<|im_end|>
<|im_start|>assistant
<think>

</think>

```

All test scripts in `tests/dev/` should use `enable_thinking=False` for consistency.

### Training Data Generation (precompute_teacher_topk.py)

**NOTE**: For training data with `add_generation_prompt=False`, the Qwen3 tokenizer has a bug where it ignores `enable_thinking` for existing assistant messages. The `precompute_teacher_topk.py` script has a workaround that manually strips `<think>...</think>` blocks when generating no-think variants.

## QWEN TEST FILES

- `export_coreml.py` - Test file for Qwen export development
- `test_coreml_kvcache_sequential.py` - Test file for Qwen inference development
