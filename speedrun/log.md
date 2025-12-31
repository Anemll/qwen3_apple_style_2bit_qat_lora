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
| Q2_INIT | `q2_init_from_q4.tgz` | ~5.8 | Q4→Q2 converted, ready for MLP training |

---

## Caches on Google Drive

| Name | File | Samples | Seq Len |
|------|------|---------|---------|
| L64 | `alpaca_chat_think_both_L64_K64_R128.tgz` | 128 | 64 |
| L128 | `alpaca_chat_think_both_L128_K128_R1024.tgz` | 1024 | 128 |

---

## Run History

### 2024-XX-XX: Q4→Q2 Progressive

**Goal**: Train Q2 from Q4 base for faster convergence

**Steps**:
1. Start with `q4_fp32` (loss ~0.73)
2. Convert Q4→Q2 with `convert_q4_to_q2.py`
3. Train MLP only for 2000-4000 steps
4. Full E2E training

**Result**: TBD

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
