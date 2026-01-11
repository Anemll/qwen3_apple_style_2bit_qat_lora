# TPU Training Guide

This document covers TPU/XLA-specific considerations for training on Google Cloud TPUs (including Colab TPU v6e).

## Quick Start

```bash
# Set TPU device
export PJRT_DEVICE=TPU

# Enable debug output (recommended for first runs)
PT_XLA_DEBUG=1 python scripts/train_recovery_lora.py --tpu ...
```

## Explicit TPU Control (Notebooks)

For Colab notebooks, use `USE_TPU` to explicitly control TPU workflow:

```python
# Set this at the top of your notebook
USE_TPU = True  # Set to False to force CPU/GPU

# Auto-detect TPU and set PJRT_DEVICE prefix
import os
if USE_TPU:
    os.environ['PJRT_DEVICE'] = 'TPU'
    TPU_PREFIX = 'PJRT_DEVICE=TPU'
else:
    TPU_PREFIX = ''

# Use in shell commands
!{TPU_PREFIX} python scripts/train_v2_simple.py --tpu ...
```

**Auto-detection with fallback:**

```python
import os

# Check if USE_TPU is defined, otherwise auto-detect
try:
    USE_TPU
except NameError:
    # Auto-detect: check if TPU device exists
    USE_TPU = os.path.exists('/dev/accel0') or os.path.exists('/dev/vfio/0')

if USE_TPU:
    os.environ['PJRT_DEVICE'] = 'TPU'
    TPU_PREFIX = 'PJRT_DEVICE=TPU'
    TPU_FLAG = '--tpu'
else:
    TPU_PREFIX = ''
    TPU_FLAG = ''

print(f"USE_TPU={USE_TPU}, TPU_PREFIX='{TPU_PREFIX}'")

# Use in training commands
!{TPU_PREFIX} python scripts/train_v2_simple.py {TPU_FLAG} \
    --config q4a4_r32 \
    --v2-checkpoint runs/best.pt \
    --cache-dir caches/openhermes_L128_K128 \
    --max-steps 2000
```

**Recommended config cell (copy-paste ready):**

```python
#@title Config
# USE_TPU options:
#   True  = Force TPU mode
#   False = Force CPU/GPU mode (disable TPU)
#   Comment out (#USE_TPU) = Auto-detect TPU
USE_TPU = True  #@param {type:"boolean"}

# ============================================================================
# TPU/Device Setup - DO NOT MODIFY BELOW
# ============================================================================
import os

# Check if USE_TPU is defined
try:
    _use_tpu = USE_TPU
except NameError:
    # Not defined (commented out) -> auto-detect
    _use_tpu = os.path.exists('/dev/accel0') or os.path.exists('/dev/vfio/0')
    print(f"USE_TPU not defined, auto-detected: {_use_tpu}")

if _use_tpu:
    os.environ['PJRT_DEVICE'] = 'TPU'
    TPU_PREFIX = 'PJRT_DEVICE=TPU'
    TPU_FLAG = '--tpu'
    print(f"TPU mode enabled (PJRT_DEVICE=TPU)")
else:
    TPU_PREFIX = ''
    TPU_FLAG = ''
    print("CPU/GPU mode (TPU disabled)")
```

Then use in training cells:

```python
!{TPU_PREFIX} python scripts/train_v2_simple.py {TPU_FLAG} \
    --config q4a4_r32 \
    --v2-checkpoint runs/best.pt \
    --cache-dir caches/openhermes_L128_K128 \
    --max-steps 2000 \
    --save-steps 200 \
    --wandb
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

### TPU Device Busy (VFIO Lock)

**Symptoms**: `RuntimeError: TPU initialization failed: open(/dev/vfio/0): Device or resource busy`

**Cause**: A previous Python process still holds the TPU device handle. Common after Colab kernel crashes or interrupted training.

**Solution**: Find and kill the process holding the device:

```bash
# Find process using TPU device
sudo lsof /dev/vfio/0 2>/dev/null || sudo fuser -v /dev/vfio/0

# Kill by PID
sudo kill -9 <PID>
```

If the above commands don't show a process, the handle may be held by a zombie process. Restart the Colab runtime:

```python
# In a notebook cell
import os
os.kill(os.getpid(), 9)
```

Or use the Colab menu: **Runtime → Restart runtime**

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
4. `optimizer.zero_grad(set_to_none=True)` causing grad None→Tensor transition

**Debug**: Look for multiple "Graph Hash" values in the debug output. Each unique hash is a separate compilation.

### Step 2 Recompilation (grad None→Tensor)

**Symptoms**: Step 1 backward compiles, Step 2 backward compiles again with different graph hash

**Cause**: When using `optimizer.zero_grad(set_to_none=True)`:
- Step 1: grad fields are `None`
- Step 2: grad fields become `Tensor` (created during backward)

This changes the optimizer graph, causing XLA to recompile.

**Solution**: Use `set_to_none=False` on TPU:
```python
# TPU: keep grads as zero tensors to avoid recompilation
optimizer.zero_grad(set_to_none=not is_tpu)
```

Already fixed in our code - `train_recovery_lora()` uses `set_to_none=False` when `use_tpu=True`.

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

---

## Sparse Logits Mode (L>=1024)

For long sequence training (L>=1024), the full vocab logits tensor `[B, L, V]` can consume significant memory:

| Seq Len | Vocab Size | Logits Size (FP32) |
|---------|------------|-------------------|
| 128 | 151,936 | 74 MiB |
| 512 | 151,936 | 297 MiB |
| 1024 | 151,936 | **594 MiB** |

The `--no-full-logits` flag prevents any full-vocab `[B,L,V]` tensor materialization by:

1. **Disabling hard label loss**: Forces `hard_top1_weight=0`, `hard_full_weight=0`
2. **Using sparse top-K loss**: Only computes logits for K teacher tokens + R random negatives
3. **Sparse anchor-KL**: Uses `anchor_topk_idx` instead of full anchor logits

### Usage

```bash
# L=1024 training with sparse logits
python scripts/train_v2_simple.py --tpu \
    --v2-checkpoint runs/best.pt \
    --cache-dir caches/wikitext103_L1024_K128_R64 \
    --no-full-logits \
    --disable-kv-cache \
    --batch-size 1 \
    --max-steps 2000
```

### Memory Comparison

| Mode | Logits Memory (L=1024) | Notes |
|------|----------------------|-------|
| Full vocab | 594 MiB | `[1, 1024, 151936]` × 4 bytes |
| Sparse K+R | **768 KiB** | `[1, 1024, 192]` × 4 bytes (K=128, R=64) |

### Required Flags

| Flag | Purpose |
|------|---------|
| `--no-full-logits` | Enable sparse logits mode |
| `--disable-kv-cache` | Prevent KV buffer allocations at long seq_len |

### Optional Flags

| Flag | Default | Purpose |
|------|---------|---------|
| `--sampled-ce-weight` | 0.0 | Weight for sampled CE loss on K+R candidates |
| `--sampled-negatives` | 64 | Number of random negatives (R) if cache lacks `rand_idx` |

### Cache Requirements

For sparse logits, your KD cache should contain:
- `topk_idx`: Top-K token indices from teacher `[B, L, K]`
- `topk_logits`: Corresponding teacher logits `[B, L, K]`

Optionally:
- `rand_idx`: Pre-sampled random negatives `[B, L, R]` (else sampled on-the-fly)

### Anchor-KL with Sparse Logits

**Note:** Anchor-KL regularization (`--anchor-ckpt`) with sparse logits mode on TPU will show a warning about potential XLA OOM. The sparse anchor-KL gather operation can create a large XLA graph.

| Configuration | Status |
|--------------|--------|
| Single-chip (v6e-1) | May OOM (~34GB needed vs 31GB available) |
| Multi-TPU (v6e-4, v6e-8) | Should work with model parallelism |

If you encounter XLA OOM on single-chip TPU, disable anchor-KL:

```bash
# Disable anchor-KL if OOM occurs
python scripts/train_v2_simple.py --tpu \
    --no-full-logits \
    --anchor-kl-weight 0 \
    ...
```

---

## Memory Debug Tools

TPU/XLA-safe memory debugging for tracking HBM usage without perturbing XLA compilation.

### Quick Start

```bash
# Basic HBM snapshots
python scripts/train_v2_simple.py --tpu \
    --mem-debug \
    --mem-debug-level basic

# Full diagnostics with tensor estimates
python scripts/train_v2_simple.py --tpu \
    --mem-debug \
    --mem-debug-level tensors \
    --mem-debug-phase all \
    --mem-debug-json /tmp/mem_log.jsonl
```

### CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--mem-debug` | off | Enable TPU HBM memory logging |
| `--mem-debug-level` | basic | Level: `basic`, `tensors`, `metrics`, `hlo` |
| `--mem-debug-phase` | warmup,save | Phases: `warmup`, `train`, `save`, `all` |
| `--mem-debug-interval` | 1 | Log every N optimizer steps |
| `--mem-debug-json` | None | Path for JSONL output |
| `--mem-debug-tag` | None | Optional run tag for filtering |
| `--mem-debug-no-xla-metrics` | off | Skip XLA metrics (if they perturb compilation) |
| `--mem-debug-step-axis` | opt | Step axis: `micro` or `opt` |

### Debug Levels

| Level | Information |
|-------|-------------|
| `basic` | HBM used/free/total, percentage |
| `tensors` | + Attention/logits workspace estimates |
| `metrics` | + XLA compilation counts (UncachedCompile, CachedCompile) |
| `hlo` | + HLO graph dumps (heavy, use sparingly) |

### Example Output

```
[MEM_DEBUG] before_warmup_compile micro=0 opt=0 t=2.5s | HBM: 4.52 GiB/31.25 GiB (14.5%) free=26.73 GiB | [EST] attn_worst=64.0 MiB (B=1,H=16,L=1024) | [EST] logits_full=593.5 MiB sparse=768.0 KiB
[MEM_DEBUG] after_first_forward micro=0 opt=0 t=175.3s | HBM: 6.28 GiB/31.25 GiB (20.1%) free=24.96 GiB
[MEM_DEBUG] after_mark_step micro=0 opt=0 t=208.8s | HBM: 10.52 GiB/31.25 GiB (33.7%) free=20.73 GiB
[MEM_DEBUG] after_warmup_compile micro=0 opt=0 t=242.4s | HBM: 6.71 GiB/31.25 GiB (21.5%) free=24.54 GiB
```

### Logging Points

| Point | Phase | When |
|-------|-------|------|
| `after_optimizer_init` | init | After optimizer creation |
| `after_anchor_init` | init | After anchor model logits computed |
| `before_warmup_compile` | warmup | Before TPU warmup forward |
| `after_first_forward` | warmup | After warmup forward pass |
| `before_mark_step` | warmup | Before XLA mark_step |
| `after_mark_step` | warmup | After XLA mark_step (graph executed) |
| `after_first_backward` | warmup | After warmup backward pass |
| `after_warmup_compile` | warmup | After full warmup complete |
| `on_checkpoint_save` | save | At each checkpoint save |
| `after_autosnap_freeze` | save | After auto-snap freeze triggered |

### JSONL Output

With `--mem-debug-json /tmp/mem.jsonl`, each log point writes a JSON record:

```json
{
  "timestamp": "2024-01-10T12:34:56.789",
  "point": "after_mark_step",
  "micro_step": 0,
  "opt_step": 0,
  "phase": "warmup",
  "elapsed_sec": 208.8,
  "hbm": {
    "used_bytes": 11290066944,
    "free_bytes": 22254583808,
    "total_bytes": 33544650752,
    "used_pct": 33.7
  },
  "estimates": {
    "attention": {"worst_case": {"bytes": 67108864, "formatted": "64.0 MiB"}},
    "logits": {"logits_full": {"bytes": 622329856}, "logits_sparse": {"bytes": 786432}}
  }
}
```

### XLA-Safety

The memory debug tools are designed to not perturb XLA compilation:

1. **No tensor reads**: Uses `xm.get_memory_info()` which doesn't trigger graph breaks
2. **Optional XLA metrics**: `--mem-debug-no-xla-metrics` skips `torch_xla.debug.metrics` calls
3. **Estimates only**: Tensor size estimates are computed from shapes, not actual tensors
4. **Phase filtering**: `--mem-debug-phase warmup,save` avoids training loop overhead

---

## Multi-Chip TPU Training (v6e-4, v6e-8)

For larger TPU configurations with multiple chips, use data parallelism.

### Quick Start (Quad TPU)

```bash
# Recovery LoRA on TPU v6e-4 (4 chips)
python scripts/train_recovery_lora_multi.py \
    --model Qwen/Qwen3-0.6B \
    --checkpoint runs/v2_q4/best.pt \
    --kd-cache-dir caches/wikitext103_L1024_K128 \
    --recovery-r 8 \
    --batch-size 4 \
    --accumulation-steps 8 \
    --max-steps 1000

# QAT on TPU v6e-4
python scripts/train_v2_tpu_multi.py \
    --from-scratch \
    --cache-dir caches/openhermes_L128_K128_N50K \
    --config q4_r32 \
    --batch-size 8 \
    --accumulation-steps 4 \
    --max-steps 3000
```

### Effective Batch Size

```
effective_batch = batch_size × accumulation_steps × num_chips
```

| Config | Batch | Accum | Chips | Effective |
|--------|-------|-------|-------|-----------|
| v6e-1 | 4 | 8 | 1 | 32 |
| v6e-4 | 4 | 8 | 4 | 128 |
| v6e-8 | 4 | 4 | 8 | 128 |

### Key Differences from Single-Chip

| Feature | Single Chip | Multi-Chip |
|---------|-------------|------------|
| Launch | Direct call | `xmp.spawn(train_worker, ...)` |
| Dataset | Single loader | Sharded across workers |
| Gradients | `optimizer.step()` | `xm.reduce_gradients()` + `xm.optimizer_step()` |
| Sync | `torch_xla.sync()` | Same + `xm.rendezvous()` for barriers |
| Checkpoints | `torch.save()` | `xm.save()` (all workers call) |

### Dataset Sharding

Files are automatically sharded across workers:
```python
# Worker k gets files [k, k+world_size, k+2*world_size, ...]
self.files = [f for i, f in enumerate(all_files) if i % world_size == rank]
```

Ensure your KD cache has enough shard files for even distribution.

### Gradient Synchronization

Multi-chip training requires explicit gradient sync:
```python
# After backward, before optimizer step
xm.reduce_gradients(optimizer)  # All-reduce gradients across chips
xm.optimizer_step(optimizer)    # TPU-specific optimizer step
```

### Rendezvous (Barriers)

Use `xm.rendezvous()` to synchronize all workers:
```python
xm.rendezvous("model_download")   # Wait for rank 0 to download
xm.rendezvous("before_warmup")    # Sync before XLA warmup
xm.rendezvous("before_final_save") # Sync before final checkpoint
```

### Checkpointing

`xm.save()` must be called by ALL workers (not just master):
```python
# All workers call this - only master writes, but all must sync
xm.save(model.state_dict(), save_path)
```

### Memory Guide (Multi-Chip)

| TPU | Chips | Total HBM | Recommended Batch |
|-----|-------|-----------|-------------------|
| v6e-1 | 1 | 32GB | batch=4, accum=8 |
| v6e-4 | 4 | 128GB | batch=4-8, accum=4-8 |
| v6e-8 | 8 | 256GB | batch=8, accum=4 |

### Limiting Number of Chips

```bash
# Use only 2 of 4 available chips
python scripts/train_recovery_lora_multi.py --num-chips 2 ...
```

Or via environment variable:
```bash
TPU_NUM_DEVICES=2 python scripts/train_recovery_lora_multi.py ...
```

---

## Auto Snap+Freeze rank_magnitude

The `--auto-snap-mags` feature automatically detects when `rank_magnitude` values have stabilized during training and applies FP16 snap + freeze. This is TPU/XLA-safe because:

1. **CPU-only audit**: Audit uses CPU tensors from checkpoint saves, no XLA lazy tensor reads
2. **Save checkpoints only**: Audit runs only at `--save-steps` intervals, no per-step overhead
3. **One-time snap+freeze**: Once triggered, optimizer is rebuilt with frozen params (one recompilation)

### Usage

```bash
python scripts/train_v2_simple.py \
    --v2-checkpoint runs/v2_q4/best.pt \
    --cache-dir caches/openhermes_L128_K128 \
    --save-steps 200 \
    --auto-snap-mags \
    --auto-snap-target mlp \
    --auto-snap-threshold 0.05 \
    --auto-snap-patience 2 \
    --auto-snap-start-step 200
```

### CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--auto-snap-mags` | off | Enable auto snap+freeze |
| `--auto-snap-target` | mlp | Target: `mlp` (84 layers) or `all` (196 layers) |
| `--auto-snap-threshold` | 0.05 | Max abs delta between saves to consider stable |
| `--auto-snap-patience` | 2 | Consecutive stable saves required to trigger |
| `--auto-snap-start-step` | 100 | Don't audit before this step |
| `--auto-snap-min-saves` | 2 | Minimum save checkpoints before eligible |
| `--auto-snap-dry-run` | off | Audit + log but don't freeze (for testing) |
| `--auto-snap-log-json` | off | Write audit JSON files |

### How It Works

1. **At each save checkpoint** (every `--save-steps`):
   - State dict is copied to CPU for checkpoint save
   - If `auto_snap_state.should_audit()` returns True:
     - Extract `rank_magnitude` tensors (CPU-only)
     - Compare to previous save's mags
     - Compute max/mean abs delta

2. **Stability detection**:
   - If `max_abs_delta < threshold`: increment `stable_count`
   - If `max_abs_delta >= threshold`: reset `stable_count` to 0

3. **Trigger condition**:
   - `stable_count >= patience` AND `num_audits >= min_saves`

4. **Apply snap+freeze**:
   - FP16 snap: `tensor.cpu().half().float()` (CPU-based for XLA safety)
   - Freeze: `requires_grad = False`
   - Rebuild optimizer without frozen params (one TPU recompilation)

### W&B Logging

When `--wandb` is enabled, auto-snap metrics are logged:

| Metric | Description |
|--------|-------------|
| `auto_snap/max_delta` | Max abs change from previous save |
| `auto_snap/mean_delta` | Mean abs change |
| `auto_snap/fp16_snap_dist` | Max distance to FP16-representable values |
| `auto_snap/stable_count` | Consecutive stable audit count |
| `auto_snap/auto_frozen` | 1 if frozen, 0 otherwise |
| `auto_snap/frozen_step` | Step when freeze was triggered |
| `auto_snap/frozen_count` | Number of parameters frozen |

### TPU/XLA Safety

**Why CPU-based audit?**

On TPU/XLA, reading tensors from device triggers graph breaks and can cause recompilation. The auto-snap audit avoids this by:

1. Using the CPU state dict that's already created for checkpoint saving
2. Never calling `.item()` or `.cpu()` on XLA tensors during audit
3. Only modifying model when freeze is triggered (planned recompilation)

```python
# SAFE: Uses CPU state dict from checkpoint save
cpu_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
torch.save(cpu_state_dict, ckpt_path)
audit_result = audit_mags_movement(state, cpu_state_dict, step)  # CPU-only

# UNSAFE: Would trigger XLA graph break
# for name, p in model.named_parameters():
#     if 'rank_magnitude' in name:
#         val = p.cpu()  # BAD: XLA tensor read
```

### Recommended Starting Configuration

For initial "from scratch" scale training:

```bash
python scripts/train_v2_simple.py \
    --from-scratch \
    --cache-dir caches/openhermes_L128_K128 \
    --max-steps 3000 \
    --save-steps 200 \
    --auto-snap-mags \
    --auto-snap-target mlp \
    --auto-snap-start-step 200 \
    --auto-snap-threshold 0.05 \
    --auto-snap-patience 2 \
    --auto-snap-min-saves 2 \
    --wandb
```

**Key choices:**
- `--auto-snap-start-step 200`: Skip early chaos phase (first ~200 optimizer steps)
- `--auto-snap-threshold 0.05`: Max abs delta between saves (tune based on your data)
- `--auto-snap-patience 2`: Require 2 consecutive stable saves before triggering
- `--auto-snap-min-saves 2`: Need at least 2 comparisons before eligible

**Expected behavior:**
1. Steps 0-199: No auditing (warmup)
2. Step 200: First audit (store baseline)
3. Step 400: Second audit (compare to 200)
4. Step 600+: If stable for 2 consecutive saves → trigger snap+freeze

**Important notes:**
- Triggers at most once per run
- Rebuilds optimizer/scheduler after freeze (one XLA recompile expected)
- Off by default (must explicitly enable with `--auto-snap-mags`)
- Use `--auto-snap-dry-run` to test without actually freezing

### Validation After Freeze

After a run with auto-snap, verify the checkpoint:

```bash
python scripts/check_mags_fp16.py runs/output/checkpoint_step800.pt
```

Expected output after freeze:
```
MLP layers: All snapped ✓
```
