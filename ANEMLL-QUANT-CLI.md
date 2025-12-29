# ANEMLL QAT Command Line Reference

This document covers command line usage for Anemll-style Quantization-Aware Training (QAT) with LUT-based quantization.

## Overview

Two training approaches:
1. **Layer-by-Layer QAT** - Train one layer at a time (lower memory, more control)
2. **End-to-End QAT** - Train all layers together (faster convergence)

Two parameter groups:
- **Weights** - The main weight matrices
- **Scales** - Per-weight scale factors (A @ B low-rank decomposition)

---

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/precompute_teacher_topk.py` | Generate KD cache from teacher model |
| `scripts/train_anemll_lbl.py` | Layer-by-layer training and testing |
| `scripts/train_anemll_qat.py` | End-to-end KD-QAT training |
| `scripts/run_anemll_inference.py` | Run inference on trained checkpoints |

---

## 1. Generate KD Cache (Required First Step)

Before training, generate a KD (Knowledge Distillation) cache from the teacher model:

```bash
python scripts/precompute_teacher_topk.py \
    --model-id Qwen/Qwen3-0.6B \
    --dataset yahma/alpaca-cleaned \
    --output-dir caches/alpaca_chat_think_both_L128_K32_R256 \
    --max-length 128 \
    --top-k 64 \
    --num-samples 1024 \
    --batch-size 8
```

**Parameters:**
- `--max-length` - Sequence length (128 recommended)
- `--top-k` - Number of top logits to store (32-128)
- `--num-samples` - Number of training samples

---

## 2. Layer-by-Layer QAT

Use `train_anemll_lbl.py` for layer-by-layer scale optimization and testing.

> **Note:** Layer-by-layer *weight* training is only available via Python API (see below). The CLI only supports layer-by-layer *scale* optimization.

### 2.1 Optimize Scales Layer-by-Layer (CLI)

```bash
python scripts/train_anemll_lbl.py \
    --model-id Qwen/Qwen3-0.6B \
    --kd-cache-dir caches/alpaca_chat_think_both_L128_K32_R256 \
    --lut-size 16 --group-size 32 --scale-rank 4 \
    --quantize-attn \
    --optimize-scales \
    --scale-lr 1e-3 \
    --scale-steps 50 \
    --scale-batch-size 32 \
    --save-quantized-dir runs/anemll_scales_optimized
```

### 2.2 Test and Evaluate (No Training)

```bash
python scripts/train_anemll_lbl.py \
    --model-id Qwen/Qwen3-0.6B \
    --kd-cache-dir caches/alpaca_chat_think_both_L128_K32_R256 \
    --lut-size 16 --group-size 32 --scale-rank 4 \
    --quantize-attn \
    --eval-samples 100 \
    --prompt "What is 2+2?"
```

### 2.3 Skip KD Evaluation

```bash
python scripts/train_anemll_lbl.py \
    --model-id Qwen/Qwen3-0.6B \
    --kd-cache-dir caches/alpaca_chat_think_both_L128_K32_R256 \
    --lut-size 16 --group-size 32 --scale-rank 4 \
    --quantize-attn \
    --optimize-scales \
    --scale-lr 1e-3 \
    --scale-steps 100 \
    --skip-kd-eval \
    --save-quantized-dir runs/anemll_scales_v1
```

### 2.4 Train Weights Layer-by-Layer (Python API)

Layer-by-layer weight training uses a mix of **local MLP reconstruction loss**, **global KD loss**, and optionally **hard label loss**:

```python
from qat_lora import train_layer, train_all_layers

# Train single layer with hard label loss for better convergence
result = train_layer(
    model=model,
    layer_idx=0,
    cache_dir="caches/alpaca_chat_think_both_L128_K32_R256",
    device=device,
    batch_size=4,
    lr=2e-5,
    max_steps=100,
    grad_accum=4,
    temperature=2.0,
    train_weights=True,   # Train weights
    train_scales=False,   # Freeze scales
    local_weight=0.5,     # Weight for local MLP loss
    global_weight=0.5,    # Weight for global KD loss
    hard_top1_weight=0.1, # Hard label top-1 (helps convergence)
    hard_full_weight=0.0, # Hard label full vocab (optional)
)

# Or train all layers sequentially
results = train_all_layers(
    model=model,
    cache_dir="caches/alpaca_chat_think_both_L128_K32_R256",
    device=device,
    lr=2e-5,
    max_steps=50,
    train_weights=True,
    train_scales=False,
    hard_top1_weight=0.1, # Recommended for weight training
)
```

**Loss function for layer-by-layer:**
- **Local MLP Loss**: MSE between quantized and FP16 MLP outputs (direction preservation)
- **Global KD Loss**: KL divergence on top-K logits with temperature scaling
- **Hard Top-1 Loss**: Cross-entropy with teacher's top-1 prediction (helps convergence)

**Recommended settings:**
- Scale optimization: `local_weight=0.5, global_weight=0.5, hard_top1_weight=0.0`
- Weight training: `local_weight=0.5, global_weight=0.5, hard_top1_weight=0.1`

See notebook `Anemll_LayerByLayer_QAT.ipynb` for complete examples.

---

## 3. End-to-End QAT Training

### 3.1 Train Scales First (Weights Frozen)

```bash
python scripts/train_anemll_qat.py \
    --model-id Qwen/Qwen3-0.6B \
    --kd-cache-dir caches/alpaca_chat_think_both_L128_K32_R256 \
    --output-dir runs/anemll_scales_v1 \
    --max-steps 500 \
    --batch-size 32 \
    --lr 1e-3 \
    --train-scales \
    --lut-size 16 --group-size 32 --scale-rank 4 \
    --quantize-attn
```

### 3.2 Train Weights (Scales Frozen)

Use `--hard-top1-weight` to prevent divergence during weight training:

```bash
python scripts/train_anemll_qat.py \
    --model-id Qwen/Qwen3-0.6B \
    --init-state runs/anemll_scales_v1 \
    --kd-cache-dir caches/alpaca_chat_think_both_L128_K32_R256 \
    --output-dir runs/anemll_weights_v1 \
    --max-steps 3000 \
    --batch-size 32 \
    --lr 5e-6 \
    --train-weights \
    --hard-top1-weight 0.1 \
    --hard-full-weight 0.0005 \
    --lut-size 16 --group-size 32 --scale-rank 4 \
    --quantize-attn
```

### 3.3 Train Both Together

```bash
python scripts/train_anemll_qat.py \
    --model-id Qwen/Qwen3-0.6B \
    --kd-cache-dir caches/alpaca_chat_think_both_L128_K32_R256 \
    --output-dir runs/anemll_both_v1 \
    --max-steps 2000 \
    --batch-size 32 \
    --lr 1e-5 \
    --train-weights --train-scales \
    --hard-top1-weight 0.1 \
    --lut-size 16 --group-size 32 --scale-rank 4 \
    --quantize-attn
```

### 3.4 Snap Weights and Save

Add `--snap-weights` to snap weights to quantized values before saving:

```bash
python scripts/train_anemll_qat.py \
    --model-id Qwen/Qwen3-0.6B \
    --init-state runs/anemll_weights_v1 \
    --kd-cache-dir caches/alpaca_chat_think_both_L128_K32_R256 \
    --output-dir runs/anemll_final_snapped \
    --max-steps 100 \
    --train-weights \
    --snap-weights \
    --lut-size 16 --group-size 32 --scale-rank 4 \
    --quantize-attn
```

### 3.5 Just Snap (No Training)

Snap weights to discrete LUT values (default: `LUT[idx]`, scales kept separate):

```bash
python scripts/train_anemll_qat.py \
    --model-id Qwen/Qwen3-0.6B \
    --init-state runs/anemll_weights_v1 \
    --kd-cache-dir caches/alpaca_chat_think_both_L128_K32_R256 \
    --output-dir runs/anemll_final_snapped \
    --skip-training \
    --snap-weights \
    --lut-size 16 --group-size 32 --scale-rank 4 \
    --quantize-attn
```

Or bake scales into weights (`LUT[idx] * scale`):

```bash
python scripts/train_anemll_qat.py \
    --model-id Qwen/Qwen3-0.6B \
    --init-state runs/anemll_weights_v1 \
    --kd-cache-dir caches/alpaca_chat_think_both_L128_K32_R256 \
    --output-dir runs/anemll_final_baked \
    --skip-training \
    --snap-weights --snap-bake-scales \
    --lut-size 16 --group-size 32 --scale-rank 4 \
    --quantize-attn
```

**Snap Modes:**

| Mode | Weight stored | Use case |
|------|--------------|----------|
| `--snap-weights` | `LUT[idx]` ∈ [-1,1] | Export with separate scales |
| `--snap-weights --snap-bake-scales` | `LUT[idx] * scale` | Simpler inference |

---

## 4. Run Inference

### 4.1 Single Prompt

```bash
python scripts/run_anemll_inference.py \
    --checkpoint runs/anemll_final_snapped \
    --prompt "What is the capital of France?" \
    --max-tokens 256
```

### 4.2 Interactive Mode

```bash
python scripts/run_anemll_inference.py \
    --checkpoint runs/anemll_final_snapped \
    --interactive
```

### 4.3 Specify Device/Dtype

```bash
python scripts/run_anemll_inference.py \
    --checkpoint runs/anemll_final_snapped \
    --prompt "Explain quantum mechanics" \
    --device mps \
    --dtype bf16 \
    --max-tokens 512
```

---

## 5. Complete Training Pipeline

### Recommended 4-bit Pipeline

```bash
# Step 1: Generate KD cache
python scripts/precompute_teacher_topk.py \
    --model-id Qwen/Qwen3-0.6B \
    --dataset yahma/alpaca-cleaned \
    --output-dir caches/alpaca_chat_think_both_L128_K32_R256 \
    --max-length 128 --top-k 64 --num-samples 2048

# Step 2: Train scales (weights frozen)
python scripts/train_anemll_qat.py \
    --model-id Qwen/Qwen3-0.6B \
    --kd-cache-dir caches/alpaca_chat_think_both_L128_K32_R256 \
    --output-dir runs/stage1_scales \
    --max-steps 500 --batch-size 32 --lr 1e-3 \
    --train-scales \
    --lut-size 16 --group-size 32 --scale-rank 4 \
    --quantize-attn

# Step 3: Train weights (scales frozen) - use hard label for stability
python scripts/train_anemll_qat.py \
    --model-id Qwen/Qwen3-0.6B \
    --init-state runs/stage1_scales \
    --kd-cache-dir caches/alpaca_chat_think_both_L128_K32_R256 \
    --output-dir runs/stage2_weights \
    --max-steps 3000 --batch-size 32 --lr 5e-6 \
    --train-weights \
    --hard-top1-weight 0.1 --hard-full-weight 0.0005 \
    --lut-size 16 --group-size 32 --scale-rank 4 \
    --quantize-attn

# Step 4: Fine-tune both together
python scripts/train_anemll_qat.py \
    --model-id Qwen/Qwen3-0.6B \
    --init-state runs/stage2_weights \
    --kd-cache-dir caches/alpaca_chat_think_both_L128_K32_R256 \
    --output-dir runs/stage3_finetune \
    --max-steps 1000 --batch-size 32 --lr 1e-6 \
    --train-weights --train-scales \
    --hard-top1-weight 0.1 \
    --lut-size 16 --group-size 32 --scale-rank 4 \
    --quantize-attn

# Step 5: Snap and export
python scripts/train_anemll_qat.py \
    --model-id Qwen/Qwen3-0.6B \
    --init-state runs/stage3_finetune \
    --kd-cache-dir caches/alpaca_chat_think_both_L128_K32_R256 \
    --output-dir runs/final_snapped \
    --skip-training --snap-weights \
    --lut-size 16 --group-size 32 --scale-rank 4 \
    --quantize-attn

# Step 6: Test inference
python scripts/run_anemll_inference.py \
    --checkpoint runs/final_snapped \
    --interactive
```

### 2-bit Pipeline (Higher compression)

```bash
# Use lut-size 4 for 2-bit quantization
python scripts/train_anemll_qat.py \
    --model-id Qwen/Qwen3-0.6B \
    --kd-cache-dir caches/alpaca_chat_think_both_L128_K32_R256 \
    --output-dir runs/2bit_scales \
    --max-steps 1000 --batch-size 32 --lr 1e-3 \
    --train-scales \
    --lut-size 4 --group-size 32 --scale-rank 8 \
    --quantize-attn --attn-lut-size 4
```

---

## 6. Parameter Reference

### train_anemll_qat.py

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model-id` | Qwen/Qwen3-0.6B | HuggingFace model ID |
| `--init-state` | - | Load initial state from checkpoint |
| `--output-dir` | runs/anemll_qat | Output directory |
| `--kd-cache-dir` | (required) | KD cache directory |
| `--device` | auto | Device (auto/cpu/cuda/mps) |
| `--dtype` | bf16 | Data type (fp16/bf16/fp32) |
| `--lut-size` | 16 | LUT entries (4=2bit, 16=4bit) |
| `--group-size` | 32 | Group size for scales |
| `--scale-rank` | 4 | Low-rank for A @ B scales |
| `--quantize-attn` | False | Quantize attention layers |
| `--attn-lut-size` | 0 | Attention LUT size (0=use --lut-size) |
| `--train-weights` | False | Train weight parameters |
| `--train-scales` | False | Train scale parameters |
| `--max-steps` | 1000 | Max training steps |
| `--batch-size` | 32 | Batch size |
| `--lr` | 1e-5 | Learning rate |
| `--temperature` | 2.0 | Distillation temperature |
| `--hard-top1-weight` | 0.0 | Hard label top-1 loss weight (stabilizes training) |
| `--hard-full-weight` | 0.0005 | Hard label full vocab loss weight |
| `--logging-steps` | 50 | Log every N steps |
| `--eval-steps` | 200 | Evaluate every N steps |
| `--snap-weights` | False | Snap weights to LUT[idx] before saving |
| `--snap-bake-scales` | False | Bake scales into snapped weights (LUT[idx]*scale) |
| `--skip-training` | False | Skip training, just load/snap |

### train_anemll_lbl.py

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model-id` | Qwen/Qwen3-0.6B | HuggingFace model ID |
| `--device` | auto | Device (auto/cpu/cuda/mps) |
| `--dtype` | bf16 | Data type (fp16/bf16/fp32) |
| `--lut-size` | 16 | LUT entries (4=2bit, 16=4bit) |
| `--group-size` | 128 | Group size for scales |
| `--scale-rank` | 4 | Low-rank for A @ B scales |
| `--quantize-attn` | False | Quantize attention layers |
| `--attn-lut-size` | 0 | Attention LUT size (0=use --lut-size) |
| `--attn-group-size` | 0 | Attention group size (0=use --group-size) |
| `--attn-scale-rank` | -1 | Attention scale rank (-1=use --scale-rank) |
| `--kd-cache-dir` | "" | KD cache directory for evaluation |
| `--eval-samples` | 100 | Number of samples for KD evaluation |
| `--skip-kd-eval` | False | Skip KD cache evaluation |
| `--skip-inference` | False | Skip inference test |
| `--optimize-scales` | False | Run layer-by-layer scale optimization |
| `--scale-lr` | 1e-3 | Learning rate for scale optimization |
| `--scale-steps` | 0 | Steps per layer (0=use epochs) |
| `--scale-epochs` | 2 | Epochs per layer (if steps=0) |
| `--scale-batch-size` | 32 | Batch size for scale optimization |
| `--save-quantized-dir` | "" | Save quantized model to directory |
| `--prompt` | "What is..." | Prompt for inference test |
| `--max-new-tokens` | 128 | Max tokens for inference |

### run_anemll_inference.py

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--checkpoint` | (required) | Checkpoint directory or .pt file |
| `--prompt` | "What is..." | Prompt to run |
| `--max-tokens` | 256 | Max new tokens |
| `--device` | auto | Device |
| `--dtype` | bf16 | Data type |
| `--interactive` | False | Interactive mode |

---

## 7. Distillation Options

### Supported Distillation Parameters

| Parameter | Script | Default | Description |
|-----------|--------|---------|-------------|
| `--temperature` | train_anemll_qat.py | 2.0 | KL divergence temperature |
| `--hard-top1-weight` | train_anemll_qat.py | 0.0 | Hard label top-1 loss weight |
| `--hard-full-weight` | train_anemll_qat.py | 0.0005 | Hard label full vocab loss weight |

### Hard Label Loss (Important for E2E Training)

The hard label loss helps stabilize E2E weight training. Without it, training can diverge.

**Recommended settings for E2E weight training:**
```bash
python scripts/train_anemll_qat.py \
    --train-weights \
    --hard-top1-weight 0.1 \
    --hard-full-weight 0.0005 \
    ...
```

**Loss function:**
- **KL Loss**: Soft label distillation on top-K logits with temperature scaling
- **Hard Top-1 Loss**: Cross-entropy with teacher's top-1 prediction
- **Hard Full Loss**: Small weight for stability

Total loss = KL_loss + hard_top1_weight * hard_top1_loss + hard_full_weight * hard_full_loss

### Not Yet Implemented

```bash
# These options from train_qat_progressive.py are NOT yet supported:
--distill_weight 1.0          # Weight for KL divergence loss (currently fixed at 1.0)
--kd_cache_shuffle_files      # Shuffle KD cache files (files are shuffled by default)
```

---

## 8. Checkpoint Structure

After training, checkpoint directory contains:

```
runs/anemll_final/
├── model_state_dict.pt    # Full model state dict
├── best_state_dict.pt     # Best checkpoint (by eval loss)
├── indices.pt             # Quantization indices for all layers
└── config.json            # Training config
```

`config.json` example:
```json
{
  "model_id": "Qwen/Qwen3-0.6B",
  "lut_size": 16,
  "group_size": 32,
  "scale_rank": 4,
  "quantize_attn": true,
  "snapped": true,
  "result": {
    "initial_loss": 1.25,
    "final_loss": 0.45,
    "best_loss": 0.42
  }
}
```

---

## 9. Tips

### Learning Rates
- **Scales**: Use higher LR (1e-3 to 1e-2) - fewer parameters
- **Weights**: Use lower LR (1e-6 to 1e-5) - many parameters
- **Both**: Use intermediate LR (1e-5)

### Training Order
1. Train scales first (quick, gets good scale initialization)
2. Train weights (main quality improvement)
3. Fine-tune both (final polish)

### Memory Optimization
- Use `--batch-size 8` with `--grad-accum 4` for effective batch 32
- Layer-by-layer uses less memory than end-to-end

### Quality vs Size
| LUT Size | Bits | Quality | Use Case |
|----------|------|---------|----------|
| 4 | 2-bit | Lower | Maximum compression |
| 16 | 4-bit | Higher | Balanced |
| 256 | 8-bit | Highest | Minimal compression |

