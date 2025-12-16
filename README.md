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

## Stage A.2 — KD-QAT on Plain Text (recommended default for “preserve knowledge”)

If your goal is **preserving the base model's behavior** under 2-bit weights (rather than learning new instruction-following behavior),
you can run **knowledge-distillation QAT (KD-QAT)**:

- Student: your quantized QAT model
- Teacher: a frozen full-precision model (often the same base model)
- Loss: KL(teacher || student) over next-token distributions (optionally mixed with CE via `--distill_weight`)

### How distillation transfers “knowledge” into 2-bit QAT

At a high level:
- The **student** runs with Apple-style int2 fake-quant weights (4 balanced levels) in the forward pass (STE in backward),
  so the model learns under the same quantization noise it will face at deployment.
- The **teacher** is a frozen full-precision model that produces the target next-token probability distribution.
- KD-QAT minimizes KL divergence between teacher and student next-token distributions (with temperature `T`):
  this “soft target” contains more information than just the single correct token (“dark knowledge”).
- Optionally, if `--distill_weight < 1.0`, the loss mixes in standard LM cross-entropy on labels:
  `loss = w * KL + (1-w) * CE`.

This is a good fit when the goal is **preserve behavior/knowledge under 2-bit compression** (rather than teach new skills).

Concretely, KD-QAT here is **quantization-aware distillation**:

- Student forward uses `QATLinear` (fake-quant weights in forward, STE in backward).
- Teacher forward runs under `torch.no_grad()` and is never updated.
- Distillation loss is KL on next-token distributions with temperature `T` (scaled by `T^2`):

```text
student_logits = student(input_ids).logits      # fake-quant weights
with torch.no_grad():
  teacher_logits = teacher(input_ids).logits    # frozen teacher

loss = KL(softmax(teacher_logits/T) || softmax(student_logits/T)) * T^2
```

So the **teacher probabilities guide the gradient**, and because the student forward includes fake-quantized weights,
the student learns to match the teacher **under 2-bit weight quantization**.

### Streaming (recommended default)

```bash
python scripts/train_qat.py \
  --model_name_or_path Qwen/Qwen3-0.6B \
  --teacher_model_name_or_path Qwen/Qwen3-0.6B \
  --distill_weight 1.0 \
  --distill_temperature 2.0 \
  --dataset_name allenai/c4 \
  --dataset_config_name en \
  --dataset_split train \
  --dataset_format text \
  --dataset_text_field text \
  --streaming \
  --output_dir runs/qwen3_kdqat2b_c4_stream \
  --device mps \
  --amp_dtype bf16 \
  --param_dtype bf16 \
  --max_length 128 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 5e-6 \
  --warmup_steps 0 \
  --max_steps 1280 \
  --skip_lm_head \
  --ema_decay 0 \
  --logging_steps 10 \
  --save_steps 200
```

Notes:
- On macOS/MPS, `--ema_decay > 0` keeps an extra copy of model weights and can trigger OOM; use `--ema_decay 0` for KD-QAT unless you have headroom.
- If you see degenerate generation (repetition / “word salad”) under KD-QAT, try:
  - `--ov-freeze` (freeze all attention `v_proj/o_proj` weights), and optionally
  - `--freeze-last-mlp --freeze-last-mlp-layers 1` (freeze the last layer’s MLP projections).
- The `--ov-freeze` idea follows recent work on stabilizing low-bit training by freezing the attention `O/V` projections: https://arxiv.org/html/2403.18159v2
- If you resume from a full training checkpoint while using these freezing options (or `--train_f_only`), the script will automatically resume **model-only** (reinit optimizer/scheduler, but keep step via filename).

Why streaming is the default here:
- C4 `en` `train` is extremely large (1024 shards; hundreds of GB if fully downloaded).
- Streaming lets you train on “infinite” text with low disk usage, at the cost of requiring network access.

### Non-streaming (full download/cached, uses a lot of disk)

Remove `--streaming` to download and build an on-disk dataset cache (Arrow).
For `allenai/c4` `en` `train`, Hugging Face reports **1024 shards**; at ~300MB/shard compressed, the raw downloads alone are on the order of **hundreds of GB**.
The built dataset cache can add additional disk usage.

This is recommended if you want:
- offline repeatability (no network required after caching)
- faster/less jittery data loading
- more stable resuming/restarts

---

## Stage A — 2-bit QAT (Alpaca / instruction SFT)

### MPS (recommended default for instruction tuning / Alpaca-style SFT)

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
  --save_steps 200 \
  --skip_lm_head \
  --ema_decay 0.999
```

---

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
  --save_steps 200 \
  --skip_lm_head \
  --ema_decay 0.999
```

Notes:
- `--amp_dtype bf16` uses bf16 autocast. If you want automatic fallback, use `--amp_dtype auto`.
- `--param_dtype bf16` stores parameters in bf16 (saves memory). If you'd like maximum numerical stability,
  use `--param_dtype fp32` but expect higher memory.
- Use `--save_steps` to control checkpoint frequency and `--resume_from_checkpoint auto` to resume from the latest.

Quantization details:
- Weight codebook is the **balanced 2-bit set** `{ -1.5, -0.5, 0.5, 1.5 }` (no zero level), multiplied by a learned scale `s`.
- You can switch to 4-bit with `-q 4` / `--quant_bits 4` (less aggressive). The balanced 4-bit codebook becomes `{ -7.5, -6.5, ..., 6.5, 7.5 }`.

Resume example:

```bash
python scripts/train_qat.py \
  --output_dir runs/qwen3_0p6b_qat2b \
  --resume_from_checkpoint auto \
  --max_steps 4000
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
  --save_steps 200 \
  --lora_r 32 \
  --lora_alpha 32 \
  --lora_dropout 0.05
```

---

### Optional: KD-LoRA using a teacher top-k cache (MPS-friendly)

If you want LoRA to **preserve teacher behavior** (rather than learn new skills), you can train LoRA with a cached distillation loss:

1) Build a teacher cache (one-time):

```bash
python scripts/precompute_teacher_topk.py \
  --teacher_model_name_or_path Qwen/Qwen3-0.6B \
  --dataset_name allenai/c4 \
  --dataset_config_name en \
  --dataset_split train \
  --dataset_text_field text \
  --streaming \
  --shuffle_buffer 10000 \
  --max_length 64 \
  --topk 32 \
  --rand_neg 256 \
  --num_sequences 20000 \
  --batch_size 1 \
  --shard_size 512 \
  --device mps \
  --dtype bf16 \
  --output_dir caches/c4_qwen3_L64_K32_R256
```

Need a cache that reflects **Qwen3's chat template with optional thinking traces**? Set `--dataset_format alpaca_chat` (or `alpaca`) and choose a thinking mode:

- `--enable_thinking false` renders the chat template without `<think>` content.
- `--enable_thinking true` forces the template to include the model’s reasoning block.
- `--enable_thinking both` duplicates each conversation twice (thinking disabled+enabled) before packing, so distillation sees the same variants the student will see at inference time.

Example:

```bash
python scripts/precompute_teacher_topk.py \
  --teacher_model_name_or_path Qwen/Qwen3-0.6B \
  --dataset_name tatsu-lab/alpaca \
  --dataset_split train \
  --dataset_format alpaca_chat \
  --enable_thinking both \
  --max_length 128 \
  --topk 32 \
  --rand_neg 256 \
  --num_sequences 20000 \
  --batch_size 1 \
  --shard_size 512 \
  --device mps \
  --dtype bf16 \
  --output_dir caches/alpaca_chat_think_both_L128_K32_R256
```

Why store both? Some downstream configs ask the student to answer with thinking on, others with thinking off. If we cache only one template, the KD loss encourages the LoRA/QAT model to imitate logits for that template, but greedy decoding on the other template can diverge (different BOS tokens, different `<think>` span). Building a mixed cache lets us reuse the same teacher run for both inference styles and reduces the risk of “word salad” or repetition when toggling thinking at generation time.

2) Train LoRA with the cache (no teacher model loaded during training):

```bash
python scripts/train_lora_recovery.py \
  --model_name_or_path Qwen/Qwen3-0.6B \
  --qat_checkpoint runs/qwen3_kdqat2b_c4_stream/final_state_dict.pt \
  --output_dir runs/qwen3_kd_lora_cached \
  --device mps \
  --amp_dtype bf16 \
  --param_dtype bf16 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 5e-5 \
  --max_steps 2000 \
  --save_steps 50 \
  --logging_steps 10 \
  --skip_lm_head \
  --lora_r 16 --lora_alpha 16 --lora_dropout 0.0 \
  --kd_cache_dir caches/c4_qwen3_L64_K32_R256 \
  --distill_temperature 2.0 \
  --distill_weight 1.0
```

Notes:
- Cache mode requires `--skip_lm_head` and currently uses pure KD (`--distill_weight 1.0`).
- `--dataset_name` is ignored in cache mode (training reads cached sequences).
- If KD loss looks good but inference degenerates (repetition / “word salad”), increase negative coverage (`--rand_neg 256` or higher) so “runaway” tokens are more likely to be penalized even if they’re not in teacher top-k.
- If greedy decoding is still unstable, add one of these cache-mode stabilizers:
  - `--hard-top1-weight 0.05` (aka `--hard_top1_weight`): adds an extra NLL term on the teacher top-1 token **within the cached candidate softmax** (top-k + random negatives). Cheap and usually helps.
  - `--hard-full-top1-weight 0.02`–`0.05` (aka `--hard_full_top1_weight`): adds a **full-vocab** cross-entropy term on the teacher top-1 token (computed at the last non-pad prediction position only). More expensive, but directly targets “greedy argmax collapse”.

## What the scripts produce

Stage A output directory contains:
- `qat_state_dict.pt` (convenience)
- `final_state_dict.pt` (from the loop)
- optional `final_state_dict_ema.pt` (if EMA enabled)
- checkpoints: `checkpoint_step{N}.pt` and `checkpoint_last.pt` (full training state, used for resume)
  - older `checkpoint_step*.pt` may be model-only; resume still works but optimizer/scheduler will reinitialize

Stage B output directory contains:
- `lora_only_state_dict.pt` (recommended)
- `full_state_dict.pt` (QAT + LoRA merged)

---

## Plotting loss curves

The custom training loop writes `loss.csv` in each run directory (columns: `step,loss,lr`).
Plot it with:

```bash
python scripts/plot_loss.py --run_dir runs/qwen3_0p6b_qat2b
python scripts/plot_loss.py --run_dir runs/qwen3_0p6b_qat2b_lora
```

If the curve is too noisy, increase `--logging_steps` (e.g. `10`) so each point is an average over more optimizer steps.

---

## Inference sanity checks (QAT-only vs QAT+LoRA)

Use `scripts/run_inference.py` to compare generation with and without the LoRA adapter.

### QAT-only

```bash
python scripts/run_inference.py \
  --model_name_or_path Qwen/Qwen3-0.6B \
  --qat_checkpoint runs/qwen3_0p6b_qat2b/final_state_dict_ema.pt \
  --device mps \
  --dtype bf16 \
  --skip_lm_head \
  --prompt "Explain QAT and why LoRA recovery helps."
```

### QAT + LoRA recovery

```bash
python scripts/run_inference.py \
  --model_name_or_path Qwen/Qwen3-0.6B \
  --qat_checkpoint runs/qwen3_0p6b_qat2b/final_state_dict_ema.pt \
  --lora_checkpoint runs/qwen3_0p6b_qat2b_lora/lora_only_state_dict.pt \
  --device mps \
  --dtype bf16 \
  --skip_lm_head \
  --lora_r 32 --lora_alpha 32 --lora_dropout 0.0 \
  --prompt "Explain QAT and why LoRA recovery helps."
```

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

MIT
