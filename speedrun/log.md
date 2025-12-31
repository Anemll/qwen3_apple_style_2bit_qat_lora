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
