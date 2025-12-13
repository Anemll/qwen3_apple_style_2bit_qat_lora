# Apple-style 2-bit QAT + LoRA Recovery for `Qwen/Qwen3-0.6B` (MPS-first, CUDA-supported)

This repo is a practical reproduction of the *2-bit Quantization-Aware Training (QAT) + LoRA "quality recovery adapters"* recipe described in Apple's **"Apple Intelligence Foundation Language Models: Tech Report 2025"**.

It is designed for:
- **Primary target**: Apple Silicon / **MPS** training (single-device)
- **Also works on CUDA** (single GPU) with fp16/bf16 autocast

> Qwen3 model card note: use **Transformers >= 4.51.0**, otherwise you may hit `KeyError: 'qwen3'`.  
> See the Qwen3-0.6B page for details.

---

## Why a custom training loop (instead of HF Trainer)

On macOS / MPS, mixed precision via Hugging Face Trainer/Accelerate has historically been flaky,
often raising errors such as *"fp16 mixed precision requires a GPU (not 'mps')"* in some versions.

Separately, **PyTorch GradScaler** is not yet reliable on MPS (it may attempt float64 ops internally, which MPS doesn't support).

So the scripts here use:
- `torch.amp.autocast(device_type="mps", dtype=...)` on MPS
- **no GradScaler** on MPS
- CUDA uses autocast + GradScaler for fp16 (standard AMP)

This gives you reproducible behavior and avoids Trainer-specific MPS checks.

---

## Setup (uv)

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
# (optional, cleaner imports):
uv pip install -e .
```

---

## Stage A — 2-bit QAT

### MPS (recommended default)

```bash
python scripts/train_qat.py \
  --model_name_or_path Qwen/Qwen3-0.6B \
  --dataset_name tatsu-lab/alpaca \
  --output_dir runs/qwen3_0p6b_qat2b \
  --device mps \
  --amp_dtype bf16 \
  --param_dtype bf16 \
  --max_length 512 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 32 \
  --learning_rate 1e-5 \
  --max_steps 2000 \
  --skip_lm_head \
  --ema_decay 0.999
```

Notes:
- `--amp_dtype bf16` attempts MPS bf16 autocast. If your build can't allocate bf16 tensors on MPS,
  it will fall back to fp16 automatically.
- `--param_dtype bf16` stores parameters in bf16 (saves memory). If you'd like maximum numerical stability,
  use `--param_dtype fp32` but expect higher memory.

### CUDA (bf16)

```bash
python scripts/train_qat.py \
  --model_name_or_path Qwen/Qwen3-0.6B \
  --dataset_name tatsu-lab/alpaca \
  --output_dir runs/qwen3_0p6b_qat2b \
  --device cuda \
  --amp_dtype bf16 \
  --param_dtype fp32 \
  --max_length 1024 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1e-5 \
  --max_steps 2000 \
  --skip_lm_head \
  --ema_decay 0.999
```

---

## Stage B — LoRA recovery adapters

### MPS

```bash
python scripts/train_lora_recovery.py \
  --model_name_or_path Qwen/Qwen3-0.6B \
  --qat_checkpoint runs/qwen3_0p6b_qat2b/qat_state_dict.pt \
  --dataset_name tatsu-lab/alpaca \
  --output_dir runs/qwen3_0p6b_qat2b_lora \
  --device mps \
  --amp_dtype bf16 \
  --param_dtype bf16 \
  --max_length 512 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 32 \
  --learning_rate 2e-4 \
  --max_steps 1000 \
  --lora_r 32 \
  --lora_alpha 32 \
  --lora_dropout 0.05
```

---

## What the scripts produce

Stage A output directory contains:
- `qat_state_dict.pt` (convenience)
- `final_state_dict.pt` (from the loop)
- optional `final_state_dict_ema.pt` (if EMA enabled)
- intermediate checkpoints every `--save_steps`

Stage B output directory contains:
- `lora_only_state_dict.pt` (recommended)
- `full_state_dict.pt` (QAT + LoRA merged)

---

## Optional: snap weights to the exact 2-bit grid

```bash
python scripts/hard_quantize_checkpoint.py \
  --model_name_or_path Qwen/Qwen3-0.6B \
  --qat_checkpoint runs/qwen3_0p6b_qat2b/qat_state_dict.pt \
  --output_path runs/qwen3_0p6b_qat2b/hard_quant_full_state_dict.pt \
  --skip_lm_head
```

---

## Practical MPS tips

- If you hit an op-not-supported error, you can try:
  `export PYTORCH_ENABLE_MPS_FALLBACK=1`
  This allows CPU fallback for unsupported ops, but it can be much slower.

- If you OOM, reduce:
  - `--max_length`
  - `--per_device_train_batch_size`
  - increase `--gradient_accumulation_steps`

---

## License

Apache-2.0
