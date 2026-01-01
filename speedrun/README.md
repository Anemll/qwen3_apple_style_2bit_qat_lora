# Speedrun Directory

Quick setup scripts for training on Colab/remote instances.

## Scripts

| Script | Purpose |
|--------|---------|
| `setup.sh` | Environment setup, pull caches/checkpoints |
| `create_q2_init.sh` | Q4→Q2 conversion on low-end instance |
| `benchmark.py` | Training performance benchmarks |

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
- Prefer `.tar.lz4` (fastest) > `.tgz` > directory
- GDrive paths auto-detected: Colab vs macOS

### Checkpoint Shortcuts

| Shortcut | File | Size | Use Case |
|----------|------|------|----------|
| `q2_init` | `q2_init_from_q4.tar.lz4` | ~2GB | Q4→Q2 converted, start MLP training |
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

## Notebooks / Tasks

| Task | Purpose |
|------|---------|
| `SR-001` | Q4→Q2 conversion + upload |
| `SR-002` | Q2 MLP training |
| `SR-003` | Performance benchmark (legacy) |
| `SR-004` | Performance benchmark (current) |

**Note:** Notebooks include a `Rev:` line at the top for version tracking. Update when making changes.

### SR-001: Q4→Q2 Conversion

Cells are labeled with `# [CELL N]` comments for easy navigation.

| Cell | Purpose | When to Run |
|------|---------|-------------|
| CELL 2 | Mount Google Drive | Always first |
| CELL 3 | Clone/update repo | Always |
| CELL 5 | Install dependencies | Always |
| CELL 8 | Pull q4_fp32 checkpoint | For conversion |
| CELL 10 | Find files, set paths | After setup |
| CELL 12 | Convert Q4→Q2 (shows loss) | Main conversion |
| CELL 13 | Upload q2_init.tar.lz4 to GDrive | After conversion |

**Flow:** 2 → 3 → 5 → 8 → 10 → 12 → 13

### SR-002: MLP Training

| Cell | Purpose |
|------|---------|
| CELL 1-4 | Setup (Drive, repo, deps) |
| CELL 5 | Print ALL terminal commands |
| CELL 6 | Print training command only |
| CELL 7 | Run training in notebook (alt) |
| CELL 8+ | Check results, upload |

**Recommended:** Run training in terminal (CELL 5 prints commands)

### SR-003: Performance Benchmark

Benchmark training optimizations (in-place LoRA, gradient checkpointing).

```bash
# Setup + run benchmarks
source speedrun/setup.sh L64
python speedrun/benchmark.py --cache-dir $CACHE_DIR --steps 20
```

**What it measures:**

| Metric | Description |
|--------|-------------|
| Step time | Seconds per training step |
| Peak memory | GPU memory usage (MB) |
| t/s | Throughput (tokens per second) |

**Benchmarks run:**
1. Baseline (batch=8) - no optimizations
2. Gradient checkpointing (batch=8) - same batch, less memory
3. Gradient checkpointing (batch=16) - larger batch enabled by memory savings

**Expected results (T4 16GB, L64 cache):**

| Config | Step(s) | Memory | t/s |
|--------|---------|--------|-----|
| batch=8 | ~0.45 | ~8.5 GB | ~1150 |
| batch=8+checkpointing | ~0.52 | ~5.2 GB | ~980 |
| batch=16+checkpointing | ~0.85 | ~9.8 GB | ~1200 |

*t/s = tokens/sec = batch × seq_len × steps / time*

**Key insight:** Checkpointing reduces memory ~40%, enabling larger batches for same throughput with better gradient estimates.

## Directory Structure

```
speedrun/
├── README.md         # This file
├── log.md            # Run history
├── setup.sh          # Environment setup
├── benchmark.py      # Training performance benchmarks
└── create_q2_init.sh # Q4→Q2 conversion
```

## Google Drive Paths

| Platform | Path |
|----------|------|
| Colab | `/content/drive/MyDrive/` |
| macOS | `~/Library/CloudStorage/GoogleDrive-realanemll@gmail.com/My Drive/` |

### GDrive Directories

- `qwen3_caches/` - KD caches (.tgz preferred)
- `qwen3_runs/` - Checkpoints and training outputs
