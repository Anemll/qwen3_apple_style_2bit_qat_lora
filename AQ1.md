# ANEMLL-QUANT-1: Architecture Overview

This document describes the ANEMLL-QUANT-1 quantization architecture for efficient LLM deployment on Apple Neural Engine (ANE).

---

## 1. KD-QAT: Knowledge Distillation + Quantization-Aware Training

ANEMLL-QUANT-1 uses **KD-QAT** (Knowledge Distillation QAT), not standard QAT:

| Component | Role |
|-----------|------|
| **Teacher** | Full-precision model (FP16/BF16) |
| **Student** | Quantized model being trained |
| **Loss** | KL divergence between teacher/student logits |

```
Teacher (frozen) ──→ soft labels (logits)
                           ↓
Student (training) ──→ match teacher distribution
```

**Why KD?** Standard QAT uses hard labels (ground truth tokens). KD preserves the teacher's "dark knowledge" - the full probability distribution over vocabulary, not just the correct answer.

**KD Loss:**
```python
loss = KL_div(student_logits / T, teacher_logits / T) * T²
```
Where `T` is temperature (typically 2.0) - softens distributions for better gradient signal.

---

## 2. Weight Decomposition

ANEMLL-QUANT-1 decomposes weights into two components (similar to exponent × mantissa in floating point):

```
W = Scales × Quantized_Values
```

Where:
- **Scales**: Per-weight magnitude factors (continuous values)
- **Quantized_Values**: Discrete values from a small lookup table (LUT)

This separation allows:
- **Quantized_Values**: Store as low-bit indices (2-bit or 4-bit)
- **Scales**: Capture weight magnitudes with low-rank compression

---

## 3. LUT-Based Quantization

The quantized values come from a learned **Lookup Table (LUT)**:

```
Quantized_Values = LUT[indices]
```

| LUT Size | Bits | Example Values |
|----------|------|----------------|
| 4 | 2-bit | {-1.5, -0.5, 0.5, 1.5} |
| 16 | 4-bit | 16 balanced levels in [-1, 1] |

Each weight maps to one LUT entry via its index. The LUT values are learned during training.

---

## 4. Low-Rank Scale Compression

Per-weight scales would require [out × in] parameters - too expensive!

**Solution:** Decompose scales using low-rank factorization:

```
Scales = A @ B
```

Where:
- `A`: [out_features, rank]
- `B`: [rank, in_features]

**Compression example** (out=2048, in=2048, rank=32):
- Full scales: 2048 × 2048 = **4M params**
- Low-rank: 2048×32 + 32×2048 = **131K params** (30× compression)

---

## 5. Scale Initialization (Block Quantization + SVD)

How do we initialize A and B? We use **block quantization** to derive initial scales, then **SVD** to compress them.

### Step 1: Block-wise Scale Computation

Split each row of weights into blocks (e.g., group_size=32):

```
For each block of 32 weights:
    1. Find optimal LUT assignment (min quantization error)
    2. Compute per-block scale: scale_block = mean(|W_block|) / mean(|LUT|)
```

This gives a coarse scale map: one scale per block.

### Step 2: Expand to Full Scale Matrix

Expand block scales to full [out × in] matrix:

```
Scales_full[i, j] = block_scale[i, j // group_size]
```

Each weight gets the scale of its block.

### Step 3: SVD Decomposition

Decompose the full scale matrix into low-rank A @ B:

```
U, S, Vt = SVD(Scales_full)
A = U[:, :rank] @ diag(sqrt(S[:rank]))    # [out, rank]
B = diag(sqrt(S[:rank])) @ Vt[:rank, :]   # [rank, in]
```

The SVD finds the best rank-r approximation to the scale matrix.

### Why This Works

| Step | Purpose |
|------|---------|
| Block quantization | Captures local weight magnitudes |
| Block scales | Coarse approximation of per-weight scales |
| SVD compression | Optimal low-rank approximation |

**Result:** A and B are initialized to approximate the true scale structure, giving QAT a good starting point.

---

## 6. V1: Training-Efficient Architecture

Combining LUT quantization with low-rank scales:

```
W_q = LUT[indices] × (A @ B)
```

**Forward pass (V1):**
```python
scales = A @ B                      # [out, in] - Full matrix MATERIALIZED
W_q = lut[indices] * scales         # Element-wise
y = F.linear(x, W_q, bias)
```

### Why V1 is Efficient for Training

V1 materializes `A @ B` as a full [out × in] matrix, which seems wasteful - but this is actually **ideal for training**:

| Aspect | Why V1 Excels |
|--------|---------------|
| **Gradient flow** | Direct gradients to A and B through matmul |
| **Optimization landscape** | Smooth, well-conditioned loss surface |
| **Numerical stability** | Full precision intermediate values |
| **Hardware utilization** | GPU matmul is highly optimized |

**The key insight:** During training, we have GPU memory available. The [out × in] materialization costs memory but gives us:
- Clean backprop through `A @ B → scales → W_q → loss`
- Each element of A and B gets meaningful gradients
- No approximations or accumulation errors

**Problem for ANE deployment:** The `A @ B` materializes a full [out × in] tensor - ANE has limited memory and prefers streaming operations.

---

## 7. V2: ANE-Friendly Refactoring

V2 refactors the forward pass to avoid materializing the [out × in] scale matrix.

### The Mathematical Equivalence

V1 and V2 compute the **same output** (up to numerical precision):

```
V1: y = (Q * (A @ B)) @ x
V2: y = Σₖ gₖ · (aₖ ⊙ (Q @ (bₖ ⊙ x)))
```

The difference is **order of operations**, not the result.

### Deriving V2 from V1

Start with V1's scale matrix and factor out per-rank contributions:

```
A @ B = Σₖ A[:,k] ⊗ B[k,:]    # Outer product sum
      = Σₖ aₖ ⊗ bₖ            # Each rank contributes one outer product
```

For V2, we normalize the directions and extract magnitudes:

```
aₖ = ||A[:,k]|| · âₖ          # âₖ is unit-norm direction
bₖ = ||B[k,:]|| · b̂ₖ          # b̂ₖ is unit-norm direction
gₖ = ||A[:,k]|| · ||B[k,:]||  # Combined magnitude
```

This gives: `A @ B = Σₖ gₖ · (âₖ ⊗ b̂ₖ)`

### V1 → V2 Conversion Process

```python
# From V1 parameters A [out, rank] and B [rank, in]
A_norms = A.norm(dim=0)       # [rank] - column norms
B_norms = B.norm(dim=1)       # [rank] - row norms

# V2 parameters
A_dir = A / A_norms           # [out, rank] - unit-norm columns
B_dir = B / B_norms[:, None]  # [rank, in] - unit-norm rows
g = A_norms * B_norms         # [rank] - magnitude per rank
```

**Key insight:** Instead of computing `A @ B` first, process **rank-by-rank**:

```
y = Σₖ gₖ · (aₖ ⊙ (Q @ (bₖ ⊙ x)))
```

Where:
- `Q = LUT[indices]` - frozen quantized weights
- `aₖ = A[:, k]` - column vector [out]
- `bₖ = B[k, :]` - row vector [in]
- `gₖ` - per-rank magnitude scalar

**Forward pass (V2):**
```python
Q = lut[indices]                    # Frozen once
y = 0
for k in range(rank):
    scaled_x = x * B[k, :]          # [batch, seq, in] - vector broadcast
    Qx = F.linear(scaled_x, Q)      # [batch, seq, out] - matmul
    y += g[k] * A[:, k] * Qx        # [batch, seq, out] - vector broadcast
```

**Why this works for ANE:**
- Each iteration uses only **vectors** [in] and [out]
- No [out × in] intermediate tensors
- Fits ANE's vector/matrix unit architecture
- Memory: O(rank × (out + in)) vs O(out × in)

---

## 8. V1 vs V2 Component Comparison

| Component | V1 | V2 |
|-----------|----|----|
| `scale_A` | [out, rank] | [out, rank] **unit-norm columns** |
| `scale_B` | [rank, in] | [rank, in] **unit-norm rows** |
| `rank_magnitude` | — | [rank] per-rank magnitude |
| `_Q` | computed each forward | **frozen buffer** = lut[indices] |
| Forward | materialize A@B | rank-by-rank accumulation |

---

## 9. STE-FP16: Training for ANE Matching

ANE runs in FP16. To ensure training matches deployment, we use **Straight-Through Estimator**:

```python
def ste_fp16(x):
    """Forward: FP16 values. Backward: identity gradient."""
    x16 = x.to(float16)
    return x + (x16.to(x.dtype) - x).detach()
```

| Phase | Precision | Purpose |
|-------|-----------|---------|
| Training (params) | FP32 | Stable gradients |
| Training (forward) | FP16 (STE) | Match ANE behavior |
| Inference | FP16 | ANE deployment |

---

## 9.1 Freeze Options for ANE FP16 Deployment

When targeting Apple Neural Engine (ANE), scale parameters must be FP16-representable. Use freeze options to pre-snap and freeze parameters during training.

**Problem:** Large `rank_magnitude` values (4.7-154.0) lose precision when snapped to FP16, causing inference divergence.

**Solution:** Freeze options pre-snap parameters to FP16-representable values, so training learns to compensate with remaining trainable parameters.

### Parameter Freeze Matrix

| Parameter | `--freeze-mags` | `--freeze-mags-mlp` | `--freeze-all` |
|-----------|-----------------|---------------------|----------------|
| scale_A (MLP) | Trainable | Trainable | Frozen + FP16 snap |
| scale_A (Attn) | Trainable | Trainable | Frozen + FP16 snap |
| scale_B (MLP) | Trainable | Trainable | Frozen + FP16 snap |
| scale_B (Attn) | Trainable | Trainable | Frozen + FP16 snap |
| rank_magnitude (MLP) | Frozen + FP16 snap | Frozen + FP16 snap | Frozen + FP16 snap |
| rank_magnitude (Attn) | Frozen + FP16 snap | Trainable | Frozen + FP16 snap |

### Parameter Counts (Qwen3-0.6B, 28 layers)

| Component | Layer Count | Parameters per Layer |
|-----------|-------------|---------------------|
| MLP layers | 84 | 3 per block (gate_proj, up_proj, down_proj) |
| Attention layers | 112 | 4 per block (q_proj, k_proj, v_proj, o_proj) |
| **Total V2 layers** | **196** | |

### Use Cases

| Flag | When to Use |
|------|-------------|
| `--freeze-mags` | ANE FP16 export - full FP16 compatibility |
| `--freeze-mags-mlp` | AQ1 (2-bit MLP, 4-bit Attn) - MLP has higher compression, more sensitive |
| `--freeze-all` | Debugging - isolate training vs precision issues |

### Tensor-Level Snap/Freeze Details

| Tensor Pattern | `--freeze-mags` | `--freeze-mags-mlp` | `--freeze-all` |
|----------------|-----------------|---------------------|----------------|
| `*.gate_proj.scale_A` | Trainable | Trainable | Snap + Freeze |
| `*.gate_proj.scale_B` | Trainable | Trainable | Snap + Freeze |
| `*.gate_proj.rank_magnitude` | Snap + Freeze | Snap + Freeze | Snap + Freeze |
| `*.up_proj.scale_A` | Trainable | Trainable | Snap + Freeze |
| `*.up_proj.scale_B` | Trainable | Trainable | Snap + Freeze |
| `*.up_proj.rank_magnitude` | Snap + Freeze | Snap + Freeze | Snap + Freeze |
| `*.down_proj.scale_A` | Trainable | Trainable | Snap + Freeze |
| `*.down_proj.scale_B` | Trainable | Trainable | Snap + Freeze |
| `*.down_proj.rank_magnitude` | Snap + Freeze | Snap + Freeze | Snap + Freeze |
| `*.q_proj.scale_A` | Trainable | Trainable | Snap + Freeze |
| `*.q_proj.scale_B` | Trainable | Trainable | Snap + Freeze |
| `*.q_proj.rank_magnitude` | Snap + Freeze | Trainable | Snap + Freeze |
| `*.k_proj.scale_A` | Trainable | Trainable | Snap + Freeze |
| `*.k_proj.scale_B` | Trainable | Trainable | Snap + Freeze |
| `*.k_proj.rank_magnitude` | Snap + Freeze | Trainable | Snap + Freeze |
| `*.v_proj.scale_A` | Trainable | Trainable | Snap + Freeze |
| `*.v_proj.scale_B` | Trainable | Trainable | Snap + Freeze |
| `*.v_proj.rank_magnitude` | Snap + Freeze | Trainable | Snap + Freeze |
| `*.o_proj.scale_A` | Trainable | Trainable | Snap + Freeze |
| `*.o_proj.scale_B` | Trainable | Trainable | Snap + Freeze |
| `*.o_proj.rank_magnitude` | Snap + Freeze | Trainable | Snap + Freeze |

**Legend:**
- **Snap**: `tensor.cpu().half().float()` (round to FP16-representable values)
- **Freeze**: `requires_grad = False`
- **Trainable**: `requires_grad = True`

### FP16 Snap Operation

```python
# CPU-based snap for consistency across devices (MPS, CUDA, TPU)
def snap_to_fp16(x):
    return x.cpu().half().float().to(x.device)
```

**Important - TPU/XLA:** FP16 snap MUST be done on CPU (`.cpu().half().float()`) to avoid XLA lazy tensor optimizations. XLA may fuse or skip the FP16 round-trip if done on device, resulting in non-FP16-representable values in the checkpoint. Always move tensor to CPU before snapping.

---

## 10. Quantization Configurations

### Q2_A4: 2-bit MLP, 4-bit Attention

| Component | LUT Size | Bits | Scale Rank |
|-----------|----------|------|------------|
| MLP | 4 | 2 | 32 |
| Attention | 16 | 4 | 8 |

### Q4_A4: 4-bit Uniform

| Component | LUT Size | Bits | Scale Rank |
|-----------|----------|------|------------|
| MLP | 16 | 4 | 4 |
| Attention | 16 | 4 | 4 |

**Design choice:** Attention uses higher precision (4-bit) because it's more sensitive to quantization than MLP layers.

---

## 11. Training Workflows

### Workflow A: V1 → V2 (Best Quality)

```
Train V1 → Convert to V2 → Fine-tune V2 → Export FP16
```
- V1 training allows full A@B during optimization
- V2 conversion extracts directional components
- Fine-tuning adapts to rank-by-rank forward

### Workflow B: V2 from Scratch

```
Initialize V2 → Train directly with STE-FP16 → Export FP16
```
- Simpler pipeline
- Slower convergence (~3000 steps vs ~1000)

### Workflow C: Progressive Quantization (Best for 2-bit)

```
Train Q4_A4 → Convert to Q2_A4 → Fine-tune → Export FP16
```
- Train at higher precision first
- K-means reduces LUT: 16 → 4 entries
- Rank expansion: 4 → 32 (MLP), 4 → 8 (Attn)
- **~6x faster** than Q2 from scratch

---

## 12. Progressive Quantization: Q4 → Q2

### The Problem with 2-bit from Scratch

Training 2-bit (Q2) directly is challenging:
- Only 4 LUT values to represent entire weight distribution
- Gradient signal is noisy (many weights map to same LUT entry)
- Optimization gets stuck in poor local minima
- Convergence is slow (~3000 steps to reach acceptable loss)

### Why Progressive Works

Progressive quantization leverages a key insight: **it's easier to refine than to learn from scratch**.

| Method | Initial Loss | Time to 0.7 | Why |
|--------|-------------|-------------|-----|
| Q2 from scratch | ~10.0 | ~3 hours | Random init, noisy gradients |
| Q4→Q2 progressive | ~5.8 | ~30 min | Inherits Q4's learned structure |

The Q4 model has already learned:
- Which weights are important (via scale magnitudes)
- Relative weight relationships (via LUT assignments)
- Low-rank structure that captures weight patterns

### Conversion Process

#### Step 1: LUT Reduction (MLP only, 16 → 4 entries)

Use K-means clustering to find the best 4 representatives from 16:

```python
from sklearn.cluster import KMeans

# Q4 LUT has 16 entries, reduce to 4
kmeans = KMeans(n_clusters=4)
kmeans.fit(q4_lut.reshape(-1, 1))
q2_lut = torch.tensor(kmeans.cluster_centers_.flatten())

# Remap indices: each old index → nearest new LUT entry
new_indices = kmeans.predict(q4_lut.reshape(-1, 1))
```

**Why K-means?** It finds 4 values that minimize quantization error over the existing weight distribution.

#### Step 2: Rank Expansion (compensate for LUT reduction)

With fewer LUT values, we need more scale expressiveness:

| Component | Q4 Rank | Q2 Rank | Expansion |
|-----------|---------|---------|-----------|
| MLP | 4 | 32 | 8× |
| Attention | 4 | 8 | 2× |

```python
# Expand scale_A: [out, 4] → [out, 32]
new_A = torch.zeros(out_features, 32)
new_A[:, :4] = old_A  # Copy existing ranks

# Expand scale_B: [4, in] → [32, in]
new_B = torch.zeros(32, in_features)
new_B[:4, :] = old_B  # Copy existing ranks

# Expand rank_magnitude: [4] → [32]
new_g = torch.zeros(32)
new_g[:4] = old_g
new_g[4:] = 0.01  # Small init for new ranks
```

**Why expand rank?** The low-rank scale matrix `A @ B` must now compensate for the coarser LUT. Higher rank = more expressiveness.

#### Step 3: Fine-tune (MLP-only recommended)

After conversion, fine-tune to recover quality:
- Existing ranks already encode useful structure
- New ranks learn to fill the gaps
- ~500 steps typically sufficient

**Optimization:** Use `--mlp-only` flag for Q4→Q2 fine-tuning:

```bash
python scripts/train_v2_simple.py \
    --v2-checkpoint runs/q2_from_q4/q2_init.pt \
    --mlp-only \
    --max-steps 500
```

**Why MLP-only?**

| Component | Q4→Q2 Change | Train? |
|-----------|--------------|--------|
| MLP | LUT 16→4, rank 4→32 | ✓ Yes (major change) |
| Attention | LUT 16→16, rank 4→8 | ✗ No (minor change) |

**Speedup:** ~2× faster (train 3 MLP vs 7 total projections per layer)

### The Mathematics

**Q4 representation:**
```
W_q4 = LUT16[idx] × (A4 @ B4)    # 16 LUT values, rank 4
```

**Q2 representation:**
```
W_q2 = LUT4[idx'] × (A32 @ B32)  # 4 LUT values, rank 32
```

The goal: `W_q2 ≈ W_q4` despite having only 4 LUT values.

**Key trade-off:** Fewer LUT values × Higher rank = Similar expressiveness

---

## 13. Training Results

### V1 → V2 Conversion

| Stage | KD Loss |
|-------|---------|
| V1 baseline | ~0.38 |
| V2 after conversion | ~0.79 |
| V2 after 1000 steps | ~0.53 |

### Progressive Q4 → Q2

| Stage | KD Loss | Steps |
|-------|---------|-------|
| Q4 trained | ~0.13 | - |
| Q2 after conversion | ~5.8 | 0 |
| Q2 step 100 | ~1.6 | 100 |
| Q2 final | ~0.7 | 500 |

---

## 14. Memory & Performance

### Scale Memory Comparison

For a layer with out=2048, in=2048, rank=32:

| Approach | Memory |
|----------|--------|
| Full scales | 2048 × 2048 = 4M params |
| Low-rank V1 | 2048×32 + 32×2048 = 131K params |
| V2 rank-by-rank | Same storage, no materialization |

### ANE Efficiency

V2's rank-by-rank forward:
- Each iteration: vector ops only
- No intermediate [out × in] tensors
- Fits ANE's vector/matrix unit design

---

## 15. V3: Recovery LoRA Adapters

V3 extends V2 with **recovery LoRA adapters** to recover accuracy lost during quantization, similar to Apple's Foundation model approach.

### The Problem

After QAT, quantized models may still have accuracy gaps compared to the original FP model. Additional fine-tuning with KD is expensive and requires the teacher model.

### The Solution

Add lightweight LoRA adapters that are trained **without distillation** using standard next-token prediction:

```
W_eff = W_q + LoRA_B @ LoRA_A × scaling
```

Where:
- `W_q`: Frozen quantized weights from V2
- `LoRA_A`: [rank, in_features] - FP32 adapter
- `LoRA_B`: [out_features, rank] - FP32 adapter
- `scaling = alpha / rank`

### Apple-Style Adapter Placement

Following Apple's Foundation model, we use a selective mask:

| Component | LoRA? | Rationale |
|-----------|-------|-----------|
| Query (Q) | Yes | High impact on attention |
| Key (K) | **No** | Diminishing returns |
| Value (V) | Yes | High impact on output |
| Attn Output (O) | Yes | Controls projection |
| gate_proj | Yes | MLP expansion |
| up_proj | Yes | MLP expansion |
| down_proj | Yes | MLP compression |

### Training Pipeline

**Phase 1: QAT (existing)** - Train V2 model with KD loss

**Phase 2: Recovery LoRA (new)** - Train LoRA adapters with CE loss

```python
# CRITICAL ORDER (avoid DDP/optimizer bugs):
model = load_model(...)
model.load_state_dict(qat_checkpoint)

# 1. Enable LoRA BEFORE optimizer/DDP
enable_recovery_lora_all(model, r=8, mlp_only=True)

# 2. Freeze base, keep only LoRA trainable
freeze_for_recovery_training(model)

# 3. Create optimizer (only sees LoRA params)
optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4)

# 4. Train with CE loss (no KD needed!)
for batch in dataloader:
    logits = model(batch['input_ids'])
    loss = cross_entropy(logits, batch['labels'])
    loss.backward()
    optimizer.step()
```

### Key Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| LoRA dtype | FP32 | Optimizer stability (cast to input dtype in forward) |
| Start rank | 8 | ~4M params, increase to 16/32 if needed |
| Sequence length | 4K-8K | Longer than QAT (only LoRA trainable) |
| Loss | CE | No KD needed - smoothing quantization errors |
| K projection | Skip | Standard LoRA practice, diminishing returns |

### Hyperparameters

| Param | Start Value | Notes |
|-------|-------------|-------|
| r | 8 | Start small, increase if needed |
| lr | 3e-4 | LoRA tolerates higher LR |
| weight_decay | 0 | Very small for LoRA |
| grad_clip | 1.0 | |
| warmup_steps | 100 | |

### Optional: Anchor KL Regularizer

To prevent drift from the base model:

```python
# Save base logits on anchor samples (before training)
anchor_logits = model(anchor_batch).detach()

# During training, add small KL penalty
loss = loss_ce + 0.01 * kl_div(current_logits, anchor_logits)
```

### Export Modes

**Mode 1: Adapter-on** (for iteration)
- Export base + LoRA separately
- Runtime: `y = quant_forward(x) + lora_forward(x)`
- Simple but adds 2 matmuls per layer

**Mode 2: Merged/Resnap** (for ANE deployment)
- Merge LoRA into effective weights
- Re-quantize to new indices
- Export pure V2 model (no LoRA ops)

```python
# Merge LoRA and re-quantize
resnap_with_lora(model)
# Now model is LoRA-free, ready for ANE
```

### API Reference

```python
from qat_lora.ane_qat_linear_v2 import (
    enable_recovery_lora_all,    # Enable LoRA on all V2 layers
    freeze_for_recovery_training, # Freeze base, keep LoRA trainable
    get_recovery_lora_params,     # Count trainable params
    get_recovery_lora_stats,      # Detailed statistics
    disable_recovery_lora_all,    # Disable LoRA
    resnap_with_lora,             # Merge LoRA and re-quantize
)

from qat_lora.layer_qat import (
    train_recovery_lora,          # Training function
)
```

### Training Modes

Recovery LoRA supports three training modes via `--lora-mode`:

| Mode | Loss | Description |
|------|------|-------------|
| `recover` | CE (raw text) | Default. Model generates its own response, train on full sequence |
| `sft` | CE (supervised) | Standard supervised fine-tuning on instruction/response pairs |
| `kd` | CE + KL | Knowledge distillation from a teacher model |

**Mode: recover (default)**
- Train on raw text with cross-entropy loss
- No special formatting required
- Good for general language modeling recovery

**Mode: sft**
- Standard supervised fine-tuning
- Uses instruction/response data
- Same as `recover` for CE loss, but implies instruction-tuning intent

**Mode: kd (Knowledge Distillation)**
- Requires `--teacher` model (e.g., `Qwen/Qwen3-4B-Instruct`)
- Combined loss: `alpha * KD_loss + (1-alpha) * CE_loss`
- KD loss: KL divergence with temperature scaling
- Best for recovering teacher-level accuracy

### CLI Script

```bash
# Mode 1: recover (default) - CE on raw text
python scripts/train_recovery_lora.py \
    --model Qwen/Qwen3-0.6B \
    --checkpoint runs/v2_q4a4_r32/best_state_dict.pt \
    --train-data data/train.jsonl \
    --recovery-r 8 \
    --mlp-only \
    --lr 3e-4 \
    --max-steps 1000 \
    --seq-len 4096 \
    --output runs/recovery_r8

# Mode 2: sft - Supervised fine-tuning
python scripts/train_recovery_lora.py \
    --model Qwen/Qwen3-0.6B \
    --checkpoint runs/v2_q4a4_r32/best_state_dict.pt \
    --train-data-hf tatsu-lab/alpaca --dataset-format alpaca \
    --lora-mode sft \
    --recovery-r 8 \
    --max-steps 1000 \
    --output runs/recovery_sft

# Mode 3: kd - Knowledge distillation from teacher
python scripts/train_recovery_lora.py \
    --model Qwen/Qwen3-0.6B \
    --checkpoint runs/v2_q4a4_r32/best_state_dict.pt \
    --train-data data/train.jsonl \
    --lora-mode kd \
    --teacher Qwen/Qwen3-4B-Instruct \
    --kd-temperature 2.0 \
    --kd-alpha 0.5 \
    --recovery-r 8 \
    --max-steps 1000 \
    --output runs/recovery_kd
```

### Recovery LoRA Training Data Format

**Note:** This is for Recovery LoRA training (Phase 2) only. Phase 1 QAT uses pre-computed KD cache (see Section 1).

**Option A: HuggingFace Dataset** (recommended for quick iterations)
```bash
# Fast testing with Pile-10k (raw text)
--train-data-hf NeelNanda/pile-10k

# WikiText-103 (larger)
--train-data-hf Salesforce/wikitext --hf-subset wikitext-103-v1

# Alpaca with chat template (match KD training)
--train-data-hf tatsu-lab/alpaca --dataset-format alpaca --template-mode no-think

# Alpaca with thinking tokens (Qwen3)
--train-data-hf tatsu-lab/alpaca --dataset-format alpaca --template-mode think

# Both variants (more diverse)
--train-data-hf tatsu-lab/alpaca --dataset-format alpaca --template-mode both

# Limit samples for quick tests
--train-data-hf NeelNanda/pile-10k --hf-max-samples 1000
```

**Template modes** (should match KD training):
| Mode | Description |
|------|-------------|
| `none` | Raw text (no template) - default |
| `no-think` | Chat template without thinking tokens |
| `think` | Chat template with thinking enabled (Qwen3) |
| `both` | Mix of no-think and think variants |

**Dataset formats**:
| Format | Fields |
|--------|--------|
| `text` | `text` or `content` field |
| `alpaca` | `instruction`, `input`, `output` |
| `sharegpt` | `conversations` with `from`/`value` |

**Option B: Local JSONL file** (`.jsonl`)
```json
{"text": "This is a training sample..."}
{"text": "Another training sample..."}
```

Or with `content` field:
```json
{"content": "This is a training sample..."}
```

**Option C: Pre-tokenized tensor** (`.pt`)
```python
# Create pre-tokenized data
tokens = tokenizer(texts, return_tensors='pt')['input_ids']
torch.save(tokens, 'data/train.pt')
```

Pre-tokenized format is fastest (skips tokenization) and recommended for large-scale training.

### Results (Expected)

| Stage | KD Loss |
|-------|---------|
| V2 QAT baseline | ~0.53 |
| + Recovery LoRA (MLP-only, r=8) | ~0.45 |
| + Recovery LoRA (full, r=16) | ~0.40 |

---

## 16. Checkpoint Utilities

### 16.1 Bake LUT (`bake_lut.py`)

After LUT training, learned values are stored in `_lut_raw_deltas` but the `.lut` buffer remains unchanged. This script bakes the deltas into the LUT and syncs `_Q` for proper inference.

```bash
# Bake to new file
python scripts/bake_lut.py source.pt dest.pt

# Bake in place (overwrites original)
python scripts/bake_lut.py source.pt --inplace

# Verbose output (shows per-layer diffs)
python scripts/bake_lut.py source.pt dest.pt --verbose
```

| Argument | Description |
|----------|-------------|
| `source` | Input checkpoint with `_lut_raw_deltas` |
| `dest` | Output checkpoint path (optional with `--inplace`) |
| `--inplace` | Overwrite input checkpoint |
| `--verbose`, `-v` | Show per-layer LUT changes |

**What it does:**
1. Builds learned LUT from `_lut_raw_deltas`
2. Applies FP16 snap + repair (`repair_lut_duplicates_symmetric`)
3. Validates LUT is FP16-safe (`verify_lut_fp16`)
4. Saves repaired LUT to `.lut` buffer
5. Computes `_Q = repaired_lut[_indices]` for consistency
6. Removes `_lut_raw_deltas` (no longer needed)

### 16.2 Sync _Q (`sync_q_checkpoint.py`)

Recomputes `_Q = lut[_indices]` for all layers in a checkpoint. Use when `_Q` has become stale (e.g., after manual LUT modifications).

```bash
# Dry run (shows what would change, no save)
python scripts/sync_q_checkpoint.py checkpoint.pt

# Fix in place
python scripts/sync_q_checkpoint.py checkpoint.pt --inplace

# Save to new file
python scripts/sync_q_checkpoint.py checkpoint.pt -o fixed.pt

# Verbose (per-layer diffs)
python scripts/sync_q_checkpoint.py checkpoint.pt --verbose
```

| Argument | Description |
|----------|-------------|
| `checkpoint` | Input checkpoint file |
| `-o`, `--output` | Output checkpoint path |
| `--inplace` | Overwrite input checkpoint |
| `-v`, `--verbose` | Show per-layer details |

**Output example:**
```
Found: 196 _indices, 196 _Q, 196 .lut

Layers synced: 196
Max diff:      0.000000
All layers already consistent (diff <= 1e-3)
```

### 16.3 Measure Perplexity (`measure_perplexity.py`)

Evaluate perplexity of V2 checkpoints or baseline models on WikiText-2/103.

```bash
# Baseline (original FP model)
python scripts/measure_perplexity.py --baseline
python scripts/measure_perplexity.py --baseline --model Qwen/Qwen3-0.6B

# QAT checkpoint
python scripts/measure_perplexity.py checkpoint.pt
python scripts/measure_perplexity.py checkpoint.pt --config q2a4

# Use KD cache as eval data
python scripts/measure_perplexity.py checkpoint.pt --cache-dir caches/alpaca_L128

# Fast screening (first N chunks only)
python scripts/measure_perplexity.py checkpoint.pt --max-chunks 20

# Batched mode (faster on GPU/TPU)
python scripts/measure_perplexity.py checkpoint.pt --batch-size 8 --seq-len 512

# With LoRA adapters
python scripts/measure_perplexity.py checkpoint.pt --lora-r 8 --lora-mlp-only

# Force device/dtype
python scripts/measure_perplexity.py checkpoint.pt --device cpu --dtype fp32
python scripts/measure_perplexity.py checkpoint.pt --device tpu --dtype bf16
```

| Argument | Default | Description |
|----------|---------|-------------|
| `checkpoint` | — | V2 checkpoint path (optional with `--baseline`) |
| `--baseline` | — | Evaluate original FP model (no checkpoint) |
| `--model` | `Qwen/Qwen3-0.6B` | Base model name |
| `--config` | — | Config preset: `q2a4`, `q4a4`, `q4_r32`, `q2a2` |
| `--dataset` | `wikitext2` | Dataset: `wikitext2` or `wikitext103` |
| `--cache-dir` | — | Use KD cache as evaluation data |
| `--text-file` | — | Use custom text file |
| `--max-length` | 1024 | Sequence length for sliding window |
| `--stride` | 512 | Stride for sliding window |
| `--max-chunks` | — | Limit to first N chunks (fast screening) |
| `--batch-size` | 0 | Batch size (0 = sliding window mode) |
| `--seq-len` | 512 | Sequence length for batched mode |
| `--lora-r` | — | Enable LoRA with specified rank |
| `--lora-mlp-only` | — | Apply LoRA to MLP layers only |
| `--device` | `auto` | Device: `auto`, `mps`, `cuda`, `cpu`, `tpu` |
| `--dtype` | `auto` | Dtype: `auto`, `fp16`, `bf16`, `fp32` |
| `--verbose` | — | Show detailed output |
| `--benchmark` | — | Benchmark different batch sizes |

**Processing modes:**
- **Sliding window** (default): Overlapping chunks with stride, accurate PPL
- **Batched** (`--batch-size N`): Parallel processing, faster on GPU/TPU

**Fast screening:** Use `--max-chunks 20` for quick ~30s evaluation vs ~3min full run.

---

## References

- See [CMD_TRAIN.md](CMD_TRAIN.md) for training commands
- See [ANEMLL-QUANT-CLI.md](ANEMLL-QUANT-CLI.md) for CLI reference
