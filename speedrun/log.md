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
