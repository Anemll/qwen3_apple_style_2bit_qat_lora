# Terminal-Based V2 QAT Training Commands

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
| `--wandb` | Enable Weights & Biases logging | False |
| `--wandb-project` | W&B project name | qwen3-qat |
| `--wandb-run` | W&B run name | auto |

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

## Snap for ANE Export

Convert checkpoint to FP16 for ANE:

```bash
python scripts/snap_and_test_v2.py \
    --checkpoint path/to/v2_checkpoint.pt \
    --fp16 \
    --output path/to/output/snapped_fp16.pt
```

This creates:
- `snapped_fp16.pt` - FP16 model weights
- `config.json` - Configuration for ANE

## Save to Google Drive

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
| `alpaca_chat_think_both_L128_K128_R1024` | 128 | 128 | 1024 | ~16 GB | 4-6 |
| `alpaca_chat_think_both_L64_K64_R128` | 64 | 64 | 128 | ~1.5 GB | 16-32 |

### Load cache from Google Drive (Colab)

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
