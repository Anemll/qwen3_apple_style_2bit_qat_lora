# Notebooks Directory

## Google Drive Directory Structure

When running notebooks on Colab, use these **consistent** paths:

```
/content/drive/MyDrive/
├── qwen3_runs/          <- CHECKPOINTS (model weights)
│   ├── qwen3_kdqat_cache_q4.tgz
│   ├── qwen3_kdqat_cache_q2_4.tgz
│   ├── progressive_qat_q2_v1_fresh.tgz
│   └── ...
│
└── qwen3_caches/        <- KD CACHES (teacher logits)
    ├── alpaca_chat_think_both_L128_K32_R256/
    ├── alpaca_chat_think_both_L128_K64_R512/
    ├── alpaca_chat_think_both_L128_K128_R512/
    └── ...
```

## Standard Config Block

Copy this to your notebooks:

```python
# ============================================================
# GOOGLE DRIVE PATHS (STANDARD)
# ============================================================

# Checkpoints/runs go here
GD_RUNS = '/content/drive/MyDrive/qwen3_runs'

# KD caches go here
GD_CACHES = '/content/drive/MyDrive/qwen3_caches'

# Local directories (on Colab VM)
LOCAL_RUNS = 'runs'
LOCAL_CACHES = 'caches'
```

## Loading Checkpoints

```python
# Load checkpoint from Google Drive
CHECKPOINT_TAR = 'qwen3_kdqat_cache_q4.tgz'
!mkdir -p {LOCAL_RUNS}
!tar -xzf {GD_RUNS}/{CHECKPOINT_TAR} -C {LOCAL_RUNS}/
```

## Loading KD Caches

```python
# Load KD cache from Google Drive
CACHE_NAME = 'alpaca_chat_think_both_L128_K64_R512'
!mkdir -p {LOCAL_CACHES}
!rsync -ah {GD_CACHES}/{CACHE_NAME}/ {LOCAL_CACHES}/{CACHE_NAME}/
```

## Saving Results

```python
# Save checkpoint to Google Drive
RUN_NAME = 'my_new_run'
!tar -czvf {RUN_NAME}.tgz -C {LOCAL_RUNS} {RUN_NAME}
!cp {RUN_NAME}.tgz {GD_RUNS}/
```

## Notebook Naming Convention

| Notebook | Purpose |
|----------|---------|
| `Generate_KD_Cache_*.ipynb` | Create KD caches from teacher model |
| `Qwen3_*bit_*.ipynb` | QAT training pipelines |
| `*_LoRA_*.ipynb` | LoRA recovery training |

## Common Mistakes

1. **Mixing runs and caches paths** - checkpoints go to `qwen3_runs`, caches go to `qwen3_caches`
2. **Extracting to wrong directory** - use `-C runs/` for checkpoints, `-C caches/` for caches
3. **Forgetting to mkdir** - always `mkdir -p` before extracting/copying
