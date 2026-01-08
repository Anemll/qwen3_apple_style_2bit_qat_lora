# TPU Training Guide

This document covers TPU/XLA-specific considerations for training on Google Cloud TPUs (including Colab TPU v6e).

## Quick Start

```bash
# Set TPU device
export PJRT_DEVICE=TPU

# Enable debug output (recommended for first runs)
PT_XLA_DEBUG=1 python scripts/train_recovery_lora.py --tpu ...
```

## Expected Timing

| Phase | Time | Notes |
|-------|------|-------|
| Anchor logits (first forward) | 5-15 min | First XLA compilation |
| Step 1 forward | 0.5-2s | May trigger small recompile |
| Step 1 backward | 3-10 min | Backward graph compilation |
| Step 1 optimizer | 1-3 min | Optimizer graph compilation |
| Steps 2-3 | 1-3 min each | Possible minor recompiles |
| Steps 4+ | 1-3 sec/step | Stable, no recompiles |

**Total first-run overhead**: ~15-30 minutes for initial compilations, then fast training.

## Debug Environment Variables

| Variable | Effect |
|----------|--------|
| `PT_XLA_DEBUG=1` | Show compilation analysis (recommended) |
| `XLA_HLO_DEBUG=1` | Show HLO-level debug info |
| `XLA_IR_DEBUG=1` | Show IR-level debug info |
| `TF_CPP_MIN_LOG_LEVEL=0` | Verbose TensorFlow logging |

### Example Debug Output

```
[Step 1] Starting...
[Step 1] Batch collated: torch.Size([4, 1024])
[Step 1] Forward pass starting...

Compilation Analysis: ================================================================================
Compilation Analysis: Compilation Cause
Compilation Analysis:   user torch_xla.sync
Compilation Analysis: Graph Info:
Compilation Analysis:   Graph Hash: 34e57f4718105340ccc7ef14b17c8910
Compilation Analysis:   Number of Graph Inputs: 1091
Compilation Analysis:   Number of Graph Outputs: 289
...

[Step 1] Forward pass done (0.5s)
[Step 1] Backward pass starting...
```

## Key XLA Concepts

### 1. Lazy Evaluation
XLA buffers all operations and only executes when forced (e.g., `.item()`, `print(tensor)`, or `xm.mark_step()`). Without explicit sync points, XLA tries to compile everything into one massive graph.

### 2. mark_step()
```python
import torch_xla.core.xla_model as xm
xm.mark_step()  # Forces XLA to compile and execute buffered operations
```

Our code calls `xla_mark_step()` after:
- Anchor logits forward pass
- Each backward pass
- Each optimizer step

This breaks up the graph into manageable pieces.

### 3. Graph Recompilation
XLA compiles a new graph when tensor shapes change. To avoid recompilation:
- Use fixed batch sizes
- Use fixed sequence lengths
- Avoid dynamic shapes in loss computation

## Troubleshooting

### Training Stuck (No Output)

**Symptoms**: Process at 100% CPU, no output for 10+ minutes

**Causes**:
1. XLA compilation in progress (normal for first steps)
2. Missing `mark_step()` causing massive graph compilation
3. Dynamic shapes causing repeated recompilation

**Debug**:
```bash
PT_XLA_DEBUG=1 python scripts/train_recovery_lora.py --tpu ...
```

Look for "Compilation Analysis" messages. If you see them, compilation is happening (just wait). If no messages appear, there may be a hang before XLA.

### Repeated Recompilation

**Symptoms**: Each step takes minutes instead of seconds

**Causes**:
1. Different batch sizes between steps
2. Dynamic tensor shapes
3. Missing `mark_step()` between operations

**Debug**: Look for multiple "Graph Hash" values in the debug output. Each unique hash is a separate compilation.

### Out of Memory (OOM)

**Symptoms**: Process killed, "RESOURCE_EXHAUSTED" error

**Solutions**:
1. Reduce batch size
2. Reduce sequence length
3. Enable gradient checkpointing
4. Use smaller model

### kl_div Autograd Warning

**Symptoms**: `aten::kl_div: autograd kernel not registered for dispatch key XLA`

**Solution**: Already fixed in our code. We use `kd_soft_ce()` instead of `F.kl_div()`:
```python
def kd_soft_ce(student_logits, teacher_logits, temperature=1.0):
    """TPU/XLA-safe KD loss using soft cross-entropy."""
    T = float(temperature)
    logp_student = F.log_softmax(student_logits / T, dim=-1)
    with torch.no_grad():
        p_teacher = F.softmax(teacher_logits / T, dim=-1)
    loss = -(p_teacher * logp_student).sum(dim=-1)
    return loss.mean() * (T * T)
```

## Code References

### TPU Detection
```python
# qat_lora/layer_qat.py
def is_xla_device(device) -> bool:
    """Check if device is TPU/XLA."""
    return 'xla' in str(device).lower()

def xla_mark_step():
    """Mark XLA step to trigger compilation/execution."""
    try:
        import torch_xla.core.xla_model as xm
        xm.mark_step()
    except ImportError:
        pass
```

### Training Script TPU Flag
```bash
python scripts/train_recovery_lora.py --tpu ...
```

Or set environment variable:
```bash
PJRT_DEVICE=TPU python scripts/train_recovery_lora.py ...
```

## Notebooks

- [`notebooks/TPU_LoRA_KD_WikiText1024.ipynb`](../notebooks/TPU_LoRA_KD_WikiText1024.ipynb) - TPU v6e-1 LoRA training example

## Performance Tips

1. **Use fixed shapes**: Batch size, sequence length should be constant
2. **Preload data**: Use `KDCacheDataset(preload=True)` to avoid I/O stalls
3. **Minimize host-device transfers**: Avoid `.item()`, `.cpu()`, `print(tensor)` in hot loops
4. **Use BFloat16**: TPU v4+ has native BF16 support
5. **Batch accumulation**: Use `--accumulation-steps` for larger effective batch without more memory

## Memory Guide (TPU v6e-1)

| Config | Batch | Seq Len | Accum | Peak Memory |
|--------|-------|---------|-------|-------------|
| Qwen3-0.6B + LoRA r=8 | 4 | 1024 | 8 | ~22GB |
| Qwen3-0.6B + LoRA r=16 | 4 | 1024 | 8 | ~24GB |
| Qwen3-0.6B + LoRA r=8 | 8 | 1024 | 4 | ~28GB |

TPU v6e-1 has **32GB HBM** per chip.
