# Speedrun Directory

Quick setup scripts for training on Colab/remote instances.

## Scripts

| Script | Purpose |
|--------|---------|
| `setup.sh` | Environment setup, pull caches/checkpoints |
| `create_q2_init.sh` | Q4→Q2 conversion on low-end instance |

## Usage

```bash
# Basic setup (env only)
source speedrun/setup.sh

# Setup + L64 cache
source speedrun/setup.sh L64

# Setup + L64 cache + checkpoint
source speedrun/setup.sh L64 q2_init
```

## Rules

### Google Drive Access
- **DO NOT read actual files without asking** - only list directories
- Prefer `.tgz` archives over directories (faster single-file transfer)
- GDrive paths auto-detected: Colab vs macOS

### Checkpoint Shortcuts

| Shortcut | File | Size | Use Case |
|----------|------|------|----------|
| `q2_init` | `q2_init_from_q4.tgz` | ~3GB | Q4→Q2 converted, start MLP training |
| `q2_best` | `Q2A4_BINIT_0.5855/best_state_dict.pt` | 4.8GB | Best Q2 result |
| `q2_0.53` | `v2_a2_q2_best_fp32_0.5341.tgz` | ~3GB | Q2 loss=0.53 |
| `q4_fp32` | `anemll_v2_q4_a4_from_v1_finetuned.tgz` | 1.6GB | Best Q4 FP32 (for Q4→Q2) |
| `q4_fp16` | `anemll_v2_q4_a4_ste_fp16_from_v1.tgz` | ~800MB | Q4 FP16 |

### Cache Shortcuts

| Shortcut | Full Name | Size |
|----------|-----------|------|
| `L64` | `alpaca_chat_think_both_L64_K64_R128` | 913MB |
| `L128` | `alpaca_chat_think_both_L128_K128_R1024` | 12GB |

### Best Practices

1. **Q4→Q2 Conversion**: Use `q4_fp32` (FP32 preserves precision for k-means)
2. **Quick Eval**: Use `L64` cache (smallest)
3. **Full Training**: Use `L128` cache (better quality)
4. **Low-end Instance**: Run `create_q2_init.sh` on T4/T3 (CPU conversion)

## Directory Structure

```
speedrun/
├── README.md       # This file
├── log.md          # Run history
├── setup.sh        # Environment setup
└── create_q2_init.sh  # Q4→Q2 conversion
```

## Google Drive Paths

| Platform | Path |
|----------|------|
| Colab | `/content/drive/MyDrive/` |
| macOS | `~/Library/CloudStorage/GoogleDrive-realanemll@gmail.com/My Drive/` |

### GDrive Directories

- `qwen3_caches/` - KD caches (.tgz preferred)
- `qwen3_runs/` - Checkpoints and training outputs
