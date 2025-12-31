# ANEMLL-QUANT-1: Architecture Overview

This document describes the ANEMLL-QUANT-1 quantization architecture for efficient LLM deployment on Apple Neural Engine (ANE).

---

## 1. Core Concept: Weight Decomposition

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

## 2. LUT-Based Quantization

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

## 3. Low-Rank Scale Compression

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

## 4. Scale Initialization (Block Quantization + SVD)

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

## 5. V1: Training-Efficient Architecture

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

## 6. V2: ANE-Friendly Refactoring

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

## 7. V1 vs V2 Component Comparison

| Component | V1 | V2 |
|-----------|----|----|
| `scale_A` | [out, rank] | [out, rank] **unit-norm columns** |
| `scale_B` | [rank, in] | [rank, in] **unit-norm rows** |
| `rank_magnitude` | — | [rank] per-rank magnitude |
| `_Q` | computed each forward | **frozen buffer** = lut[indices] |
| Forward | materialize A@B | rank-by-rank accumulation |

---

## 8. STE-FP16: Training for ANE Matching

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

## 9. Quantization Configurations

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

## 10. Training Workflows

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

## 11. Progressive Quantization: Q4 → Q2

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

#### Step 3: Fine-tune

After conversion, fine-tune to recover quality:
- Existing ranks already encode useful structure
- New ranks learn to fill the gaps
- ~500 steps typically sufficient

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

## 12. Training Results

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

## 13. Memory & Performance

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

## References

- See [CMD_TRAIN.md](CMD_TRAIN.md) for training commands
- See [ANEMLL-QUANT-CLI.md](ANEMLL-QUANT-CLI.md) for CLI reference
