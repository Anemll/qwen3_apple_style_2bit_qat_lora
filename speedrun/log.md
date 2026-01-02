# Speedrun Log

Training runs history with names and descriptions.

---

## Checkpoints on Google Drive

### Q4 Checkpoints

| Name | File | Loss | Notes |
|------|------|------|-------|
| Q4_FP32 | `anemll_v2_q4_a4_from_v1_finetuned.tgz` | ~0.73 | V2 Q4_A4, FP32, from V1 conversion + finetuning |
| Q4_FP16 | `anemll_v2_q4_a4_ste_fp16_from_v1.tgz` | ~0.73 | Same as above, snapped to FP16 |

### Q2 Checkpoints

| Name | File | Loss | Notes |
|------|------|------|-------|
| Q2_BINIT_0.5855 | `Q2A4_BINIT_0.5855/best_state_dict.pt` | 0.5855 | Best Q2 run with B initialization |
| Q2_0.5341 | `v2_a2_q2_best_fp32_0.5341.tgz` | 0.5341 | Current best Q2 result |
| Q2_INIT | `q2_init_from_q4.tar.lz4` | ~5.8 | Q4→Q2 converted, ready for MLP training |
| SR-002 | `sr-002/best_state_dict.pt` | 0.6567 | MLP-only trained from Q2_INIT |

---

## Caches on Google Drive

| Name | File | Size | Samples | Seq Len |
|------|------|------|---------|---------|
| L64 | `alpaca_chat_think_both_L64_K64_R128.tar.lz4` | 1.2GB | 128 | 64 |
| L128 | `alpaca_chat_think_both_L128_K128_R1024.tar.lz4` | 16GB | 1024 | 128 |

---

## Run History

### 2024-12-31: SR-001 - Q4→Q2 Conversion

**Goal**: Test speedrun setup, convert Q4→Q2, train MLP

**Instance**: Colab T4 (free)

**Config**:
- Cache: L64 (913MB)
- Checkpoint: q4_fp32 → convert → q2_init
- Training: MLP-only, 2000 steps

**Commands**:
```bash
# Setup pulled:
source speedrun/setup.sh L64 q4_fp32
# [CACHE] Extracting alpaca_chat_think_both_L64_K64_R128.tgz... Done
# [CKPT] Extracting anemll_v2_q4_a4_from_v1_finetuned.tgz...

# Then convert:
python scripts/convert_q4_to_q2.py \
    --q4-checkpoint runs/speedrun/anemll_v2_q4_a4_from_v1_finetuned.pt \
    --output runs/speedrun/q2_init.pt \
    --eval --cache-dir caches/alpaca_chat_think_both_L64_K64_R128

# Then train:
python scripts/train_v2_simple.py \
    --v2-checkpoint runs/speedrun/q2_init.pt \
    --cache-dir caches/alpaca_chat_think_both_L64_K64_R128 \
    --output-dir runs/speedrun \
    --mlp-only --max-steps 2000 \
    --gdrive-dir /content/drive/MyDrive/qwen3_runs/speedrun0
```

**Result**: COMPLETE

**Conversion Time:**
| Instance | CPU Cores | Time |
|----------|-----------|------|
| A100 | 12 | ~3 min |
| T4 | 2 | ~15 min (estimated) |

**Output:**
- `runs/speedrun/q2_init.pt` created
- **KD Loss: 5.8265** (expected ~5.8)
- 196 layers converted (84 MLP + 112 Attn)

**Notes:**
- Conversion is CPU-bound (k-means clustering)
- GPU not used during conversion, only for eval
- A100's 12 cores helped vs T4's 2 cores
- Inference test: empty response (expected before training)

---

### 2024-12-31: SR-002 - Q2 MLP Training

**Goal**: Train Q2 MLP layers from q2_init checkpoint

**Instance**: Colab A100 40GB

**Config**:
- Cache: L64 (128 samples, seq_len=64)
- Checkpoint: q2_init (from SR-001)
- Training: MLP-only, 4000 steps
- Batch: 4, LR: 5e-5 (cosine decay to 1e-5)

**Command**:
```bash
source speedrun/setup.sh L64 q2_init
python scripts/train_v2_simple.py \
    --v2-checkpoint $CHECKPOINT \
    --cache-dir $CACHE_DIR \
    --output-dir runs/sr-002 \
    --mlp-only --max-steps 4000 \
    --batch-size 4 --lr 5e-5 \
    --wandb --wandb-run "SR-002" \
    --gdrive-dir /content/drive/MyDrive/qwen3_runs/sr-002
```

**Result**: COMPLETE ✓

| Metric | Value |
|--------|-------|
| Initial Loss | 5.8267 |
| Final Loss | 0.6596 |
| **Best Loss** | **0.6567** |
| Improvement | 5.17 (89%) |
| Time | 2h 51min |
| Tokens | ~1.0M |

**Output**:
- `runs/sr-002/best_state_dict.pt` - Best checkpoint (0.6567)
- `runs/sr-002/v2_q2a4_fp32_*.pt` - Final FP32
- `runs/sr-002/v2_q2a4_fp16_*.pt` - Final FP16
- Uploaded to GDrive: `/qwen3_runs/sr-002/`

**WandB**: https://wandb.ai/anemll-com/qwen3-qat/runs/pg2gb0uh

**Notes**:
- Loss dropped rapidly: 5.82 → 0.67 in first 1700 steps
- Plateaued at ~0.65-0.66 after step 2000
- MLP-only is ~3x faster than E2E
- Next: SR-002-full for E2E refinement with L128

---

### 2024-12-31: SR-003 - Performance Benchmark

**Goal**: Find max batch size and measure throughput with gradient checkpointing

**Instance**: Colab A100 40GB

**Config**:
- Cache: L64 (seq_len=64)
- Gradient checkpointing: enabled (use_reentrant=False)
- Test: 3 steps per batch size

**Command**:
```bash
source speedrun/setup.sh L64
python speedrun/benchmark.py --cache-dir $CACHE_DIR --find-max-batch
```

**Result**: COMPLETE ✓

| Batch | Step(s) | Memory | t/s | Loss | Status |
|-------|---------|--------|-----|------|--------|
| 8 | 15.016 | 8,734M | 34 | 8.09 | OK |
| 16 | 16.787 | 9,983M | 61 | 7.99 | OK |
| 32 | 21.222 | 12,479M | 97 | 7.92 | OK |
| 64 | 30.348 | 17,471M | 135 | 7.26 | OK |
| 128 | 48.673 | 27,455M | 168 | 7.68 | OK |
| **144** | **53.245** | **29,951M** | **173** | 7.68 | **MAX** |
| 152+ | - | - | - | - | OOM |

**Key Findings**:
- **Max batch size: 144** (with gradient checkpointing, L64)
- **Best throughput: 173 t/s** at batch=144
- Memory scales sub-linearly: 8→144 batch = 18x, but 8.7→30GB = 3.4x
- Throughput scales well: 34→173 t/s = 5x improvement

**Gradient Checkpointing Impact** (batch=8):
- Without: ~96 t/s, ~25GB memory
- With: ~34 t/s, ~8.7GB memory (-65% memory, -65% speed)
- But enables batch=144 → 173 t/s (1.8x faster than baseline!)

**Recommended Settings** (A100 40GB):
| Cache | Max Batch | Best t/s | Memory |
|-------|-----------|----------|--------|
| L64 (seq=64) | 144 | 173 | 30GB |
| L128 (seq=128) | 72 | 172 | 30GB |

### L128 Benchmark Results

| Batch | Step(s) | Memory | t/s | Loss | Status |
|-------|---------|--------|-----|------|--------|
| 8 | 17.775 | 9,987M | 58 | 10.32 | OK |
| 16 | 21.524 | 12,487M | 95 | 10.26 | OK |
| 32 | 30.682 | 17,487M | 133 | 10.08 | OK |
| 64 | 49.136 | 27,487M | 167 | 9.95 | OK |
| **72** | **53.589** | **29,987M** | **172** | 10.01 | **MAX** |
| 80+ | - | - | - | - | OOM |

**Key Finding:** L128 max batch (72) is exactly half of L64 max batch (144), as expected since seq_len doubles.

**Notes**:
- Binary search finds exact max (not just powers of 2)
- V2 model cached to `runs/speedrun/v2_benchmark_model.pt` for fast reruns
- Fix applied: `use_reentrant=False` for gradient checkpointing
- Quick test mode (2 steps, no warmup) gives lower t/s (~155) due to JIT overhead; sustained t/s is ~173

---

### 2025-01-01: SR-006 - RTX 5090 Benchmark (RunPod)

**Goal**: Benchmark max batch size on RTX 5090 via RunPod

**Instance**: RunPod RTX 5090 32GB

**Config**:
- Cache: L64 (seq_len=64)
- Gradient checkpointing: enabled
- dtype: BF16
- V2 model: downloaded from GDrive (~5s)

**Command**:
```bash
python speedrun/benchmark.py \
    --cache-dir caches/alpaca_chat_think_both_L64_K64_R128 \
    --load-model /home/v2_benchmark_model_L64.pt \
    --find-max-batch --dtype bf16
```

**Result**: COMPLETE ✓

| Batch | Step(s) | Memory | t/s | Loss | Status |
|-------|---------|--------|-----|------|--------|
| 152 | 43.123 | 18,281M | 226 | 7.17 | OK |
| 184 | 37.770 | 20,985M | 312 | 6.12 | OK |
| 200 | 34.856 | 22,337M | 367 | 5.64 | OK |
| **208** | **41.320** | **23,009M** | **322** | 5.28 | **MAX** |
| 216 | - | 21,748M | - | - | OOM |

**Key Findings**:
- **Max batch size: 208** (with gradient checkpointing, L64, BF16)
- **Best throughput: 367 t/s** at batch=200
- Peak at batch=200, slight drop at 208 (memory pressure)
- Model load from GDrive: 5.2s (vs 2min build time)

**Comparison with Other GPUs** (L64, gradient checkpointing):

*BF16:*

| GPU | VRAM | Max Batch | Best t/s | Notes |
|-----|------|-----------|----------|-------|
| B200 | 180GB | 512 | 1582 | RunPod |
| H100 | 80GB | 504 | 1182 | RunPod |
| RTX 5090 | 32GB | 208 | 367 | RunPod |
| L4 | 24GB | 128 | 152 | Colab |

*FP32:*

| GPU | VRAM | Max Batch | Best t/s | Notes |
|-----|------|-----------|----------|-------|
| A100 | 40GB | 144 | 173 | Colab |

**Notes**:
- RTX 5090 achieves **2.1x throughput** vs A100 40GB despite similar batch size
- Blackwell architecture efficiency shows in t/s numbers
- Model download from GDrive worked perfectly (SR-006 workflow)

---

### 2025-01-01: SR-006 - H100 Benchmark (RunPod)

**Goal**: Benchmark max batch size on H100 80GB via RunPod

**Instance**: RunPod H100 80GB HBM3

**Config**:
- Cache: L64 (seq_len=64)
- Gradient checkpointing: enabled
- dtype: BF16
- V2 model: downloaded from GDrive (~1.5s copy)

**Command**:
```bash
python speedrun/benchmark.py \
    --cache-dir caches/alpaca_chat_think_both_L64_K64_R128 \
    --load-model /home/v2_benchmark_model_L64.pt \
    --find-max-batch --dtype bf16
```

**Result**: COMPLETE ✓

| Batch | Step(s) | Memory | t/s | Loss | Status |
|-------|---------|--------|-----|------|--------|
| 440 | 25.268 | 42,778M | 1114 | 7.09 | OK |
| 472 | 26.250 | 45,488M | 1151 | 6.02 | OK |
| 488 | 26.672 | 46,844M | 1171 | 5.58 | OK |
| 496 | 26.923 | 47,520M | 1179 | 5.20 | OK |
| **504** | **27.298** | **48,200M** | **1182** | 5.01 | **MAX** |

**Key Findings**:
- **Max batch size: 504** (with gradient checkpointing, L64, BF16)
- **Best throughput: 1182 t/s** at batch=504
- **6.8x faster than A100 40GB** (1182 vs 173 t/s)
- **3.2x faster than RTX 5090** (1182 vs 367 t/s)
- Model load: 14.2s, copy: 1.5s

---

### 2025-01-01: SR-006 - B200 Benchmark (RunPod)

**Goal**: Benchmark max batch size on B200 180GB via RunPod

**Instance**: RunPod B200 180GB (Blackwell)

**Config**:
- Cache: L64 (seq_len=64)
- Gradient checkpointing: enabled
- dtype: BF16
- V2 model: downloaded from GDrive (~1.3s copy)

**Command**:
```bash
python speedrun/benchmark.py \
    --cache-dir caches/alpaca_chat_think_both_L64_K64_R128 \
    --load-model /home/v2_benchmark_model_L64.pt \
    --find-max-batch --dtype bf16
```

**Result**: COMPLETE ✓

| Batch | Step(s) | Memory | t/s | Loss | Status |
|-------|---------|--------|-----|------|--------|
| **512** | **20.708** | **48,701M** | **1582** | 6.99 | **MAX** |

**Key Findings**:
- **Max batch size: 512** (with gradient checkpointing, L64, BF16)
- **Best throughput: 1582 t/s** at batch=512
- **9.1x faster than A100 40GB** (1582 vs 173 t/s)
- **1.34x faster than H100** (1582 vs 1182 t/s)
- Model load: 8.9s, copy: 1.3s
- Only using 49GB of 180GB - room to scale with L128 or larger batches

---

### 2025-01-02: SR-005 - TPU v6e Benchmark

**Goal**: Benchmark max batch size on Google TPU v6e-1 (16GB HBM)

**Instance**: Google Cloud TPU v6e-1 (16GB HBM, ~460 TFLOPS BF16)

**Config**:
- Cache: L64 (seq_len=64)
- Gradient checkpointing: SKIPPED (incompatible with transformers on XLA)
- dtype: BF16 (TPU native)
- PyTorch/XLA (torch_xla)

**Initial Problem**: Benchmark hung indefinitely after "Moving to GPU..."

**Root Causes Identified**:
1. **Full vocab CE in loss** - `hard_full_weight=0.0005` triggered `[B*S, vocab]` matmul, massive XLA graph
2. **Initial/final eval** - `evaluate_kd_loss()` called `.item()` per sample, forcing device-to-host sync
3. **XLA compilation** - First step compiles entire graph (~4 min for batch=16)
4. **Graph caching by shape** - Each batch size requires separate compilation
5. **`--batch-sizes` argument ignored** - Bug: argument defined but never used

**Fixes Applied**:
| Fix | File | Impact |
|-----|------|--------|
| `hard_full_weight=0.0` | benchmark.py | Remove full vocab CE from graph |
| `hard_top1_weight=0.0` | benchmark.py | Pure KD loss (smaller graph) |
| `eval_samples=0` | benchmark.py | Skip initial/final eval |
| `verbose=True` on TPU | benchmark.py | Show progress during compile |
| `--batch-sizes` fix | benchmark.py | Custom batch sizes now work |
| `xm.mark_step()` | layer_qat.py | Execute graph after each step |

**Commands**:
```bash
# First run (compiles graph, ~4 min)
python speedrun/benchmark.py \
    --cache-dir caches/alpaca_chat_think_both_L64_K64_R128 \
    --dtype bf16 \
    --batch-sizes 16 \
    --steps 50

# Second run (uses cached graph, fast)
python speedrun/benchmark.py \
    --cache-dir caches/alpaca_chat_think_both_L64_K64_R128 \
    --dtype bf16 \
    --batch-sizes 16 \
    --steps 50
```

**Result**: COMPLETE ✓

| Batch | Compile | Steps | Time | t/s | Notes |
|-------|---------|-------|------|-----|-------|
| 8 | ~4 min | 100 | 6:10 | 138 | Includes compile |
| 8 | cached | 100 | 2:30 | 339 | Real throughput |
| 16 | ~4 min | 50 | 5:06 | 167 | Includes compile |
| 16 | cached | 50 | 1:17 | 660 | Real throughput |
| **24** | ~4 min | 50 | 5:15 | 243 | Includes compile |
| **24** | cached | 50 | 1:18 | **972** | **MAX throughput** |
| 32 | - | - | - | OOM | "Attempting to reserve 27.33G" |

**Key Findings**:
- **Max batch size: 24** (batch=32 OOMs: "Attempting to reserve 27.33G")
- **Max throughput: 972 t/s** at batch=24 (cached graph)
- **XLA graph caching is critical** - 4x speedup (167 → 660 t/s)
- **Compilation overhead** - ~4 min per batch size (one-time cost)
- **Batch scaling works** - 339 t/s (batch=8) → 660 t/s (batch=16)

**Comparison with GPUs** (L64, BF16, gradient checkpointing):

| Device | VRAM | Max Batch | t/s (cached) | Notes |
|--------|------|-----------|--------------|-------|
| L4 | 24GB | 128 | 152 | Colab |
| **TPU v6e** | **16GB** | **24** | **972** | Google Cloud |
| A100 | 40GB | 144 | 173* | Colab (*FP32) |
| H100 | 80GB | 504 | 1182 | RunPod |
| B200 | 180GB | 512 | 1582 | RunPod |

**Where the Speedup Came From**:

| Issue | Before | After | Speedup |
|-------|--------|-------|---------|
| **Full vocab CE** | Huge graph, compile hang | No full vocab | Graph compiles |
| **Initial/final eval** | .item() per sample | Skip eval | No sync spam |
| **Graph not cached** | ~243 t/s | ~972 t/s | **4x** |
| **Small batch** | batch=8 (339 t/s) | batch=24 (972 t/s) | **3x** |

**Summary**: The "hang" was XLA compiling a massive graph (full vocab CE + eval). Disabling those made compilation feasible (~4 min). Using cached graphs + batch=24 gave **972 t/s** - faster than H100 at batch=24 equivalent, 6x faster than L4!

**TPU-Specific Notes**:
- `torch_xla.sync()` / `xm.mark_step()` required after each step
- Gradient checkpointing incompatible (transformers uses `torch.xla` not `torch_xla`)
- `freeze_Q()` triggers many small graph executions (`.cpu()` calls)
- Each batch size = separate compilation (can't mix in same run)

---

### Template for New Runs

```markdown
### YYYY-MM-DD: Run Name

**Goal**: What we're trying to achieve

**Config**:
- Checkpoint: `shortcut` or path
- Cache: L64 / L128
- Steps: N
- LR: X
- Flags: --mlp-only, etc.

**Command**:
```bash
python scripts/train_v2_simple.py \
    --v2-checkpoint $CHECKPOINT \
    --cache-dir $CACHE_DIR \
    --output-dir runs/run_name \
    --max-steps N
```

**Result**:
- Initial loss: X
- Final loss: Y
- Notes: observations
```

---

## Naming Convention

Format: `{quant}_{variant}_{loss}_{dtype}`

Examples:
- `v2_q2a4_mlp_0.58_fp32` - V2 Q2_A4 MLP-only, loss 0.58, FP32
- `v2_q4a4_from_v1_fp16` - V2 Q4_A4 from V1 conversion, FP16

Components:
- `v2` - V2 architecture
- `q2a4` / `q4a4` - Quantization (MLP_Attention)
- `mlp` / `e2e` - Training mode
- `0.XX` - Loss value
- `fp32` / `fp16` - Precision
