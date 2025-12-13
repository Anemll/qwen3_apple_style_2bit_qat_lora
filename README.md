# Apple-style 2-bit QAT + LoRA Recovery for Qwen/Qwen3-0.6B

This repo is a **practical reproduction** of the *2-bit Quantization-Aware Training (QAT) + LoRA "quality recovery adapters"* recipe described in **Apple's "Apple Intelligence Foundation Language Models: Tech Report 2025"**.

It is designed to work out-of-the-box with:

- **Model**: `Qwen/Qwen3-0.6B` (dense 0.6B Qwen3)  
- **Framework**: Hugging Face Transformers (must be **>= 4.51.0** for Qwen3 support)

> Qwen3's model card warns that older Transformers versions fail with `KeyError: 'qwen3'`.
> You should upgrade Transformers if you hit that error.

---

## What this implements (Apple-specific details)

### 1) 2-bit QAT (fake-quant during training)

During training we replace every `nn.Linear` (optionally excluding `lm_head`) with `QATLinear`,
which performs Apple-style fake-quant in the forward pass:

\[
\tilde{W} = s \cdot \left(\text{clamp}(\left\lfloor \frac{W}{s} + z \right\rceil, q_{min}, q_{max}) - z\right)
\]

and we use a **straight-through estimator (STE)** for the rounding op.

We also implement Apple's key refinements:

- **Learnable per-tensor scaling factor** `f` used to compute `s` from `max(|W|)`  
- **Newton-like clip initialization** to set `f_init = c / max(|W|)`  
- **Balanced 2-bit set** `{ -1.5, -0.5, 0.5, 1.5 }` (implemented via `qmin=-1, qmax=2, z=0.5`)
- **Weight decay = 0** for QAT (Apple's recommendation)
- Optional **EMA of weights** during QAT (Apple uses EMA to stabilize 2-bit training)

### 2) LoRA "recovery adapters"

After QAT, we:
- freeze the core model weights (including QAT scale params)
- add LoRA adapters to the quantized Linear layers
- fine-tune **only** LoRA parameters to compensate quantization error

This matches Apple's "quality recovery adapters" idea: LoRA adapts to quantization artifacts
without retraining the whole base model.

---

## Quickstart

### 0) Setup (uv)

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
# (optional, cleaner imports): uv pip install -e .
```

### 1) Stage A — QAT fine-tune (2-bit)

Example on a small SFT dataset (Alpaca):

```bash
python scripts/train_qat.py \
  --model_name_or_path Qwen/Qwen3-0.6B \
  --dataset_name tatsu-lab/alpaca \
  --dataset_format alpaca \
  --output_dir runs/qwen3_0p6b_qat2b \
  --max_length 1024 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1e-5 \
  --max_steps 2000 \
  --bf16 \
  --skip_lm_head \
  --ema_decay 0.999
```

Notes:
- If you have limited VRAM, reduce `--max_length` and/or batch size.
- If you want to exactly follow Apple's guidance, keep `--weight_decay 0.0` (default).

This script writes:
- `runs/.../qat_state_dict.pt`  (full QAT state dict incl. learned scales)
- `runs/.../training_args.json`

### 2) Stage B — LoRA recovery fine-tune

```bash
python scripts/train_lora_recovery.py \
  --model_name_or_path Qwen/Qwen3-0.6B \
  --qat_checkpoint runs/qwen3_0p6b_qat2b/qat_state_dict.pt \
  --dataset_name tatsu-lab/alpaca \
  --dataset_format alpaca \
  --output_dir runs/qwen3_0p6b_qat2b_lora \
  --max_length 1024 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --learning_rate 2e-4 \
  --max_steps 1000 \
  --bf16 \
  --lora_r 32 \
  --lora_alpha 32 \
  --lora_dropout 0.05
```

This script writes:
- `runs/.../lora_only_state_dict.pt` (only LoRA params, recommended to keep separate)
- `runs/.../full_state_dict.pt`      (QAT + LoRA together, convenience)

---

## Notes on "exactness" vs Apple's internal pipeline

Apple does not publish full pseudocode for the Newton-style rebalancing procedure.
This repo implements a **Newton-like MSE-minimizing clip initializer**, which is a reasonable, widely-used approach
for low-bit clipping/scale initialization.

Everything else is implemented literally from Apple's description:
- fake-quant formula, STE, learnable scale
- stability tricks: low LR, EMA, weight decay 0, balanced levels
- LoRA recovery with frozen base weights

---

## Repository layout

- `qat_lora/quantizer.py`  
  Core quantizer: fake-quant + STE, and clip initialization

- `qat_lora/qat_linear.py`  
  Drop-in replacement for `nn.Linear` supporting:
  - 2-bit fake-quant weight
  - optional LoRA path

- `qat_lora/model_utils.py`  
  Helpers:
  - replace `nn.Linear` -> `QATLinear`
  - initialize `f`
  - freeze base & enable LoRA

- `qat_lora/data.py`  
  Converts datasets (e.g., Alpaca) into Qwen3 chat-template training examples
  and masks labels to train only on assistant tokens.

- `qat_lora/ema.py`  
  Simple EMA helper used by the Trainer callback.

- `qat_lora/callbacks.py`  
  Trainer callback that updates EMA every optimization step and optionally saves EMA weights.

- `scripts/train_qat.py`  
  End-to-end QAT fine-tune script.

- `scripts/train_lora_recovery.py`  
  End-to-end LoRA recovery fine-tune script.

- `scripts/hard_quantize_checkpoint.py`  
  Optional utility to "snap" weights to the exact int2 grid and save a float checkpoint.
  (This does NOT pack weights into 2-bit storage; it just enforces grid values.)

---

## FAQ

### Does this produce a true 2-bit packed model?

No. This trains the model **as if** weights are int2 (fake-quant + STE), and can export weights snapped to 4 levels.
True int2 packing and kernels (like Apple Neural Engine or custom CUDA kernels) are deployment engineering.

### Why skip `lm_head`?

Quantizing output projections can disproportionately hurt quality for small models.
Apple also used mixed precision (some parts higher-bit). You can enable it if you want.

---

## License

Apache-2.0 (same as Qwen).
