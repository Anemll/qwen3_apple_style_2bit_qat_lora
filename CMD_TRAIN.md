# ANEMLL-QUANT-1: V2 KD-QAT Training Commands

> **Architecture Overview:** See [AQ1.md](AQ1.md) for V1 vs V2 architecture details, STE-FP16, and progressive quantization concepts.

---

## Training Workflows

### Workflow A: V1 First, Then V2 (Recommended for Quality)

Best quality through staged training:

```
┌─────────────────────────────────────────────────────────┐
│  1. Train V1 (KD-QAT with teacher distillation)         │
│     - Full A@B materialization OK during training       │
│     - Focus on convergence quality                      │
│                                                         │
│  2. Convert V1 → V2                                     │
│     - Extract A_dir, B_dir, rank_magnitude from A, B    │
│     - Freeze Q = lut[indices]                           │
│                                                         │
│  3. Fine-tune V2 scales                                 │
│     - Train with STE-FP16 for ANE matching              │
│     - Short training (~500-1000 steps)                  │
│                                                         │
│  4. Snap to FP16 for ANE export                         │
└─────────────────────────────────────────────────────────┘
```

**Commands:**
```bash
# Step 1: Train V1 (use notebooks or train_anemll_qat.py)

# Step 2-3: Convert and fine-tune
python scripts/train_v2_simple.py \
    --v1-checkpoint runs/v1/best.pt \
    --cache-dir caches/alpaca_chat_think_both_L128_K128_R1024 \
    --output-dir runs/v2_from_v1 \
    --max-steps 1000

# Step 4: Snap for ANE
python scripts/snap_and_test_v2.py \
    --checkpoint runs/v2_from_v1/best_state_dict.pt \
    --fp16 --output runs/export/model_fp16.pt
```

---

### Workflow B: V2 from Scratch (Simpler)

Train V2 directly (slower convergence but simpler):

```
┌─────────────────────────────────────────────────────────┐
│  1. Initialize V2 model with random scales              │
│                                                         │
│  2. Train V2 directly                                   │
│     - STE-FP16 from the start                           │
│     - Longer training (~3000+ steps)                    │
│                                                         │
│  3. Snap to FP16 for ANE export                         │
└─────────────────────────────────────────────────────────┘
```

**Commands:**
```bash
python scripts/train_v2_simple.py \
    --from-scratch \
    --cache-dir caches/alpaca_chat_think_both_L128_K128_R1024 \
    --output-dir runs/v2_scratch \
    --max-steps 3000 \
    --lr 1e-4
```

---

### Workflow C: Progressive Quantization Q4 → Q2 (Best for 2-bit)

Best for 2-bit quality - train at higher precision first:

```
┌─────────────────────────────────────────────────────────┐
│  1. Train Q4_A4 (4-bit MLP, 4-bit Attn)                 │
│     - Easier to converge at higher precision            │
│                                                         │
│  2. Convert Q4 → Q2                                     │
│     - K-means LUT reduction: 16 → 4 entries (MLP only)  │
│     - Rank expansion: 4 → 32 (MLP), 4 → 8 (Attn)        │
│                                                         │
│  3. Fine-tune Q2_A4 (MLP-only, ~2x faster)              │
│     - Attention unchanged (still 4-bit from Q4)         │
│     - Focus on MLP which has major changes              │
│     - ~500 steps with --mlp-only                        │
│                                                         │
│  4. Snap to FP16 for ANE export                         │
└─────────────────────────────────────────────────────────┘
```

**Commands:**
```bash
# Step 2: Convert Q4 → Q2
python scripts/convert_q4_to_q2.py \
    --q4-checkpoint runs/q4/checkpoint.pt \
    --output runs/q2_from_q4/q2_init.pt

# Step 3: Fine-tune MLP only (attention already good from Q4)
python scripts/train_v2_simple.py \
    --v2-checkpoint runs/q2_from_q4/q2_init.pt \
    --cache-dir caches/alpaca_chat_think_both_L128_K128_R1024 \
    --output-dir runs/q2_from_q4 \
    --mlp-only \
    --max-steps 500

# Step 4: Snap for ANE
python scripts/snap_and_test_v2.py \
    --checkpoint runs/q2_from_q4/best_state_dict.pt \
    --fp16 --output runs/export/model_fp16.pt
```

See [AQ1.md](AQ1.md) for detailed progressive quantization documentation.

---

## Training Results Summary

### V1 → V2 Conversion Results

| Stage | KD Loss | Notes |
|-------|---------|-------|
| V1 baseline | ~0.38 | After V1 training |
| V2 initial | ~0.79 | After conversion (expected increase) |
| V2 trained | ~0.53 | After 1000 steps |

### V2 From Scratch Results

| Stage | KD Loss | Steps | Time |
|-------|---------|-------|------|
| Initial | ~10.0 | 0 | - |
| Mid training | ~1.3 | 1000 | ~1 hr |
| Final | ~0.7 | 3000 | ~3 hrs |

### Progressive Q4→Q2 Results

| Stage | KD Loss | Steps | Time |
|-------|---------|-------|------|
| Q4 trained | ~0.13 | - | - |
| Q2 after conversion | ~5.8 | 0 | - |
| Q2 step 100 | ~1.6 | 100 | ~5 min |
| Q2 final | ~0.7 | 500 | ~30 min |

**Key insight:** Progressive quantization is **~6x faster** than Q2 from scratch.

---

## Quick Reference - Scripts & Functions

| Script | Description |
|--------|-------------|
| `scripts/train_v2_simple.py` | Main V2 training script (V1→V2 conversion, resume, from scratch) |
| `scripts/train_recovery_lora.py` | Recovery LoRA training (recover, sft, kd modes) |
| `scripts/run_v2_training.sh` | Full training pipeline (extract, train, save to Drive) |
| `scripts/pull_cache.sh` | Extract KD cache from Google Drive |
| `scripts/test_inference.py` | Test model inference (single prompt or interactive, with LoRA support) |
| `scripts/snap_and_test_v2.py` | Snap model to FP16 for ANE export |
| `scripts/convert_q4_to_q2.py` | Convert Q4 checkpoint to Q2 (progressive quantization) |
| `scripts/convert_v1_to_v2.py` | Convert V1 checkpoint to V2 format |
| `scripts/precompute_teacher_topk.py` | Generate KD cache from teacher model |
| `scripts/plot_loss.py` | Plot training loss curves from CSV logs |
| `scripts/debug_layer_qat.py` | Debug script for layer-by-layer QAT analysis |

### Python Functions (qat_lora)

| Function | Description |
|----------|-------------|
| `replace_linear_with_anemll_v2()` | Replace nn.Linear with V2 QAT layers |
| `freeze_Q_all()` | Freeze quantization indices (before training scales) |
| `train_e2e()` | End-to-end KD training |
| `train_recovery_lora()` | Recovery LoRA training (recover, sft, kd modes) |
| `evaluate_kd_loss()` | Evaluate KD loss on cache |
| `load_v2_checkpoint()` | Load V2 checkpoint with proper _Q buffer handling |
| `snap_model_for_ane_v2()` | Snap to FP16 for ANE export |
| `convert_model_to_fp16_v2()` | Convert model to FP16 |
| `ste_fp16()` | Straight-through FP16 rounding (for STE-FP16 training) |
| `set_factored_inference_v2()` | Enable/disable rank-by-rank forward |
| `set_batched_forward_v2()` | Enable/disable batched forward mode |
| `enable_recovery_lora_all()` | Enable LoRA on all V2 layers |
| `freeze_for_recovery_training()` | Freeze base, keep LoRA trainable |

---

## Setup (Colab)

```bash
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Clone repo
cd /content
git clone https://github.com/Anemll/qwen3_apple_style_2bit_qat_lora.git
cd qwen3_apple_style_2bit_qat_lora

# Install dependencies
pip install -q transformers accelerate datasets sentencepiece protobuf

# Update repo
git pull
```

## Training Commands

### Option 1: V1 to V2 Conversion + Training

Start from a trained V1 checkpoint and convert to V2:

```bash
python scripts/train_v2_simple.py \
    --v1-checkpoint runs/tmp/backup_mlp_e2e_w_0.3824.pt \
    --cache-dir caches/alpaca_chat_think_both_L128_K128_R1024 \
    --output-dir runs/v2_output \
    --max-steps 1000 \
    --lr 5e-5 \
    --batch-size 8
```

### Option 2: Continue Training from V2 Checkpoint

Resume training from a saved V2 checkpoint:

```bash
python scripts/train_v2_simple.py \
    --v2-checkpoint runs/v2_output/v2_q2a4_fp32_TIMESTAMP.pt \
    --cache-dir caches/alpaca_chat_think_both_L128_K128_R1024 \
    --output-dir runs/v2_output \
    --max-steps 500 \
    --lr 5e-5 \
    --batch-size 8
```

### Option 3: Train from Scratch

Train V2 without any V1 checkpoint:

```bash
python scripts/train_v2_simple.py \
    --from-scratch \
    --cache-dir caches/alpaca_chat_think_both_L128_K128_R1024 \
    --output-dir runs/v2_scratch \
    --max-steps 1000 \
    --lr 1e-4 \
    --batch-size 8
```

### Long Training with Periodic Checkpoints

Save checkpoints every 500 steps:

```bash
python scripts/train_v2_simple.py \
    --v2-checkpoint runs/v2_scratch/v2_q2a4_fp32_TIMESTAMP.pt \
    --cache-dir caches/alpaca_chat_think_both_L128_K128_R1024 \
    --output-dir runs/v2_scratch \
    --max-steps 4000 \
    --lr 5e-5 \
    --batch-size 8 \
    --save-steps 500
```

## Training Options

| Flag | Description | Default |
|------|-------------|---------|
| `--v1-checkpoint` | V1 checkpoint for conversion | None |
| `--v2-checkpoint` | V2 checkpoint to resume | None |
| `--from-scratch` | Train without checkpoint | False |
| `--cache-dir` | KD cache directory | Required |
| `--output-dir` | Output directory | runs/v2_output |
| `--max-steps` | Training steps | 1000 |
| `--lr` | Learning rate | 5e-5 |
| `--batch-size` | Batch size | 8 |
| `--save-steps` | Save checkpoint every N steps | 0 (disabled) |
| `--g-only` | Train only G (rank_magnitude) | False |
| `--mlp-only` | Train only MLP layers | False |
| `--hard-top1` | Hard label top-1 weight | 0.2 |
| `--hard-full` | Hard label full vocab weight | 0.00005 |
| `--model-id` | HuggingFace model ID | Qwen/Qwen3-0.6B |
| `--wandb` | Enable Weights & Biases logging | False |
| `--wandb-project` | W&B project name | qwen3-qat |
| `--wandb-run` | W&B run name | auto |
| `--gdrive-dir` | Auto-upload FP32 checkpoint to Google Drive | None |

## Logging

Training automatically saves:
- `training_log.csv` - CSV with step, train_loss, eval_loss, lr, elapsed_sec
- Periodic checkpoints (if `--save-steps` > 0)
- Best checkpoint (`best_state_dict.pt`)

### Weights & Biases

Enable W&B logging for experiment tracking:

```bash
pip install wandb
wandb login

python scripts/train_v2_simple.py \
    --v2-checkpoint runs/v2_output/checkpoint.pt \
    --cache-dir caches/alpaca_chat_think_both_L128_K128_R1024 \
    --output-dir runs/v2_output \
    --wandb \
    --wandb-project my-qat-project \
    --wandb-run experiment-1
```

## Recovery LoRA Training

Train lightweight LoRA adapters on top of quantized V2 models to recover accuracy.

### Training Modes

| Mode | Loss | Use Case |
|------|------|----------|
| `recover` | CE (raw text) | Default. General recovery on any text data |
| `sft` | CE (supervised) | Supervised fine-tuning on instruction data |
| `kd` | CE + KL | Knowledge distillation from teacher model |

### Mode 1: recover (default)

Train on raw text with cross-entropy loss:

```bash
python scripts/train_recovery_lora.py \
    --model Qwen/Qwen3-0.6B \
    --checkpoint runs/v2_q4a4_r32/best_state_dict.pt \
    --train-data data/train.jsonl \
    --lora-mode recover \
    --recovery-r 8 \
    --lr 3e-4 \
    --max-steps 1000 \
    --output runs/recovery_r8
```

### Mode 2: sft (Supervised Fine-Tuning)

Train on instruction/response pairs:

```bash
python scripts/train_recovery_lora.py \
    --model Qwen/Qwen3-0.6B \
    --checkpoint runs/v2_q4a4_r32/best_state_dict.pt \
    --train-data-hf tatsu-lab/alpaca --dataset-format alpaca \
    --lora-mode sft \
    --recovery-r 8 \
    --lr 3e-4 \
    --max-steps 1000 \
    --output runs/recovery_sft
```

### Mode 3: kd (Knowledge Distillation)

Distill from a larger teacher model:

```bash
python scripts/train_recovery_lora.py \
    --model Qwen/Qwen3-0.6B \
    --checkpoint runs/v2_q4a4_r32/best_state_dict.pt \
    --train-data data/train.jsonl \
    --lora-mode kd \
    --teacher Qwen/Qwen3-4B-Instruct \
    --kd-temperature 2.0 \
    --kd-alpha 0.5 \
    --recovery-r 8 \
    --lr 3e-4 \
    --max-steps 1000 \
    --output runs/recovery_kd
```

### Recovery LoRA Options

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | Required | V2 QAT checkpoint |
| `--recovery-r` | 8 | LoRA rank |
| `--lora-mode` | recover | Training mode: recover, sft, kd |
| `--teacher` | None | Teacher model for KD mode |
| `--kd-temperature` | 2.0 | KD softmax temperature |
| `--kd-alpha` | 0.5 | KD loss weight (alpha*KD + (1-alpha)*CE) |
| `--mlp-only` | False | Only add LoRA to MLP layers |
| `--skip-k-proj` | True | Skip K projection (standard LoRA practice) |
| `--lr` | 3e-4 | Learning rate |
| `--max-steps` | 1000 | Training steps |
| `--seq-len` | 2048 | Sequence length |
| `--batch-size` | 4 | Batch size |
| `--save-lora-only` | False | Save only LoRA weights (smaller file) |

---

## Inference Testing

### Single Prompt

```bash
python scripts/test_inference.py checkpoint.pt \
    --version v2 \
    --lut-bits 2 \
    --attn-lut-bits 4 \
    --scale-rank 32 \
    --attn-scale-rank 8 \
    --prompt "What is the capital of France?"
```

### Interactive Mode

```bash
python scripts/test_inference.py checkpoint.pt \
    --version v2 \
    --lut-bits 2 \
    --attn-lut-bits 4 \
    --scale-rank 32 \
    --attn-scale-rank 8 \
    --interactive
```

### With LoRA Adapter

```bash
# Option A: Separate LoRA file
python scripts/test_inference.py checkpoint.pt \
    --lora lora_adapter.pt \
    --lora-r 8 \
    --interactive

# Option B: Checkpoint with embedded LoRA (auto-detected)
python scripts/test_inference.py full_checkpoint_with_lora.pt --interactive
```

### Inference Options

| Flag | Default | Description |
|------|---------|-------------|
| `--version` | auto | Force v1 or v2 (auto-detect from config.json) |
| `--lut-bits` | auto | LUT bits for MLP |
| `--attn-lut-bits` | auto | LUT bits for attention |
| `--scale-rank` | auto | Scale rank for MLP |
| `--attn-scale-rank` | auto | Scale rank for attention |
| `--lora` | None | Path to LoRA adapter checkpoint |
| `--lora-r` | 8 | LoRA rank (must match saved adapter) |
| `--max-tokens` | 512 | Max new tokens to generate |
| `--temperature` | 0.6 | Sampling temperature |
| `--repetition-penalty` | 1.1 | Repetition penalty |
| `--no-thinking` | False | Disable Qwen3 thinking mode |
| `--interactive` | False | Interactive prompt loop |

## Convert Q4_A4 to Q2_A4

Start Q2 training from a trained Q4 checkpoint (progressive quantization):

```bash
# Convert Q4_A4 (rank=4) → Q2_A4 (MLP rank=32, Attn rank=8)
python scripts/convert_q4_to_q2.py \
    --q4-checkpoint /path/to/q4_a4_v2.pt \
    --output runs/q2_from_q4/q2_init.pt \
    --eval \
    --cache-dir caches/alpaca_chat_think_both_L128_K128_R1024

# Then train
python scripts/train_v2_simple.py \
    --v2-checkpoint runs/q2_from_q4/q2_init.pt \
    --cache-dir caches/alpaca_chat_think_both_L128_K128_R1024 \
    --output-dir runs/q2_from_q4 \
    --max-steps 1000
```

Conversion does:
- **Attention**: Keep lut=16 (4-bit), expand rank 4→8
- **MLP**: Reduce lut 16→4 (k-means), expand rank 4→32

**Expected behavior after conversion:**
- KD loss increases significantly (e.g., 0.13 → 4.9) due to LUT reduction
- Model may produce poor output until trained
- Training should reduce loss back to ~0.5-1.0 within 1000-2000 steps

## Snap for ANE Export

Convert checkpoint to FP16 for ANE:

```bash
python scripts/snap_and_test_v2.py \
    --checkpoint path/to/v2_checkpoint.pt \
    --fp16 \
    --output path/to/output/snapped_fp16.pt
```

### Snap Options

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | Required | Path to V2 checkpoint (.pt file) |
| `--output` | None | Save snapped checkpoint to this path |
| `--fp16` / `--ane` | False | Snap for ANE export (FP16 precision) |
| `--lut-bits` | 2 | LUT bits for MLP |
| `--attn-lut-bits` | 4 | LUT bits for attention |
| `--scale-rank` | 32 | Scale rank for MLP |
| `--attn-scale-rank` | 8 | Scale rank for attention |
| `--device` | auto | Device (cuda/mps/cpu) |
| `--debug` | False | Print debug information |

This creates:
- `snapped_fp16.pt` - FP16 model weights
- `config.json` - Configuration for ANE

## Save to Google Drive

### Auto-upload During Training (Recommended)

Use `--gdrive-dir` to automatically upload FP32 checkpoint after training:

```bash
python scripts/train_v2_simple.py \
    --v2-checkpoint runs/v2_output/checkpoint.pt \
    --cache-dir caches/alpaca_chat_think_both_L128_K128_R1024 \
    --output-dir runs/v2_output \
    --max-steps 1000 \
    --gdrive-dir /content/drive/MyDrive/qwen3_runs/v2_output
```

This will:
- Create the directory if it doesn't exist
- Upload only the FP32 checkpoint (smaller than FP16 for same precision)

### Manual Copy

```bash
# Copy checkpoints
mkdir -p /content/drive/MyDrive/qwen3_runs/v2_output
cp runs/v2_output/*.pt /content/drive/MyDrive/qwen3_runs/v2_output/

# Or use save script (creates .tgz)
bash scripts/save_to_drive.sh runs/v2_output/v2_q2a4_fp32_TIMESTAMP.pt my_checkpoint
```

## KD Cache Options

Different caches for different batch sizes:

| Cache | L | K | R | Size | Est. Batch |
|-------|---|---|---|------|------------|
| `alpaca_chat_think_both_L128_K128_R1024` | 128 | 128 | 1024 | ~16 GB | 4-8 |
| `alpaca_chat_think_both_L64_K64_R128` | 64 | 64 | 128 | ~1.5 GB | 16-32 |

### Pull Cache from Google Drive (Recommended)

```bash
# Use the pull_cache.sh script
bash scripts/pull_cache.sh alpaca_chat_think_both_L64_K64_R128   # Small (fast)
bash scripts/pull_cache.sh alpaca_chat_think_both_L128_K128_R1024  # Full (quality)
```

### Manual Cache Extraction

```bash
# Full cache (L128_K128)
CACHE_NAME="alpaca_chat_think_both_L128_K128_R1024"
mkdir -p caches/$CACHE_NAME
tar -xzf /content/drive/MyDrive/qwen3_caches/$CACHE_NAME.tgz -C caches/

# Small cache (L64_K64) - for bigger batches
CACHE_NAME="alpaca_chat_think_both_L64_K64_R128"
mkdir -p caches/$CACHE_NAME
tar -xzf /content/drive/MyDrive/qwen3_caches/$CACHE_NAME.tgz -C caches/
```

### Generate small cache (L64_K64)

Use notebook `notebooks/Generate_KD_Cache_K64_K128.ipynb` with:
```python
TOP_K = 64
RANDOM_NEGATIVES = 128
MAX_LENGTH = 64
```

Or terminal:
```bash
python scripts/precompute_teacher_topk.py \
  --teacher_model_name_or_path Qwen/Qwen3-0.6B \
  --dataset_name tatsu-lab/alpaca \
  --dataset_split train \
  --dataset_format alpaca_chat \
  --enable_thinking both \
  --max_length 64 \
  --topk 64 \
  --rand_neg 128 \
  --num_sequences 20000 \
  --batch_size 32 \
  --device cuda \
  --output_dir caches/alpaca_chat_think_both_L64_K64_R128
```

## Q2_A4 Configuration

Default configuration for 2-bit MLP / 4-bit Attention:

| Component | LUT Size | LUT Bits | Scale Rank |
|-----------|----------|----------|------------|
| MLP | 4 | 2 | 32 |
| Attention | 16 | 4 | 8 |

## Typical Training Results

| Method | Initial Loss | Final Loss | Steps |
|--------|-------------|------------|-------|
| V1 to V2 | ~0.79 | ~0.53 | 1000 |
| From Scratch | ~10.0 | ~1.29 | 1000 |
| Continuation | ~0.54 | ~0.53 | +500 |

## Troubleshooting

### CUDA Out of Memory
- Reduce `--batch-size` (try 4 or 2)
- Ensure no other training jobs running

### Garbled Inference Output
- Ensure config matches training (`force_positive_scales=False`)
- Check `_Q` buffers loaded correctly (should see "Manually loaded N _Q buffers")

### High Initial Loss from Scratch
- Normal for random init (~10.0)
- Will decrease rapidly in first 200-300 steps
