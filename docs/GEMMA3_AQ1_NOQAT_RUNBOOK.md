# Gemma3 AQ1 No-QAT Runbook

This runbook documents `scripts/test-gemma3-aq1-noqat.sh`.

## 1) Default run

```bash
cd /Users/anemll/SourceRelease/GITHUB/ML_playground/qwen3_apple_style_2bit_qat_lora
bash scripts/test-gemma3-aq1-noqat.sh
```

Default output root:

- `runs/gemma3_aq1_noqat`

## 2) Clean run (recommended, non-destructive)

Recompute every stage without deleting run folders:

```bash
FORCE_REFP16=1 FORCE_REIMATRIX=1 FORCE_REAWQ=1 FORCE_REINIT=1 \
bash scripts/test-gemma3-aq1-noqat.sh
```

This is the safest "clean run" for repeatability.

## 3) Hard clean run (delete previous artifacts)

Delete only this pipeline's run directory, then run again:

```bash
rm -rf runs/gemma3_aq1_noqat
bash scripts/test-gemma3-aq1-noqat.sh
```

## 4) Fast smoke run

Uses fewer calibration tokens and short generation:

```bash
TOKENS=8192 MAX_NEW_TOKENS=64 ENABLE_PPL=false \
bash scripts/test-gemma3-aq1-noqat.sh
```

## 4b) Rank override (recommended for Gemma3-1B)

```bash
INIT_MLP_RANK=32 INIT_ATTN_RANK=32 FORCE_REINIT=1 \
bash scripts/test-gemma3-aq1-noqat.sh
```

## 5) Enable perplexity

- Full run:

```bash
ENABLE_PPL=true bash scripts/test-gemma3-aq1-noqat.sh
```

- Quick screen (first 20 chunks):

```bash
ENABLE_PPL=20 bash scripts/test-gemma3-aq1-noqat.sh
```

## 6) Inference-only check after checkpoint exists

```bash
python3 scripts/test_inference.py runs/gemma3_aq1_noqat/v2_q2a4_init/v2_initial.pt \
  --model-id runs/gemma3_aq1_noqat/model_awq_a0p5 \
  --prompt "Who invented the iPad?" \
  --max-tokens 160 \
  --device cpu \
  --dtype float32 \
  --no-thinking
```

For backend isolation, run the same prompt on both CPU and MPS:

```bash
python3 scripts/test_inference.py runs/gemma3_aq1_noqat/v2_q2a4_init/v2_initial.pt \
  --model-id runs/gemma3_aq1_noqat/model_awq_a0p5 \
  --prompt "Explain what quantization-aware initialization does in one short paragraph." \
  --max-tokens 160 --device cpu --dtype float32 --no-thinking

python3 scripts/test_inference.py runs/gemma3_aq1_noqat/v2_q2a4_init/v2_initial.pt \
  --model-id runs/gemma3_aq1_noqat/model_awq_a0p5 \
  --prompt "Explain what quantization-aware initialization does in one short paragraph." \
  --max-tokens 160 --device mps --dtype float32 --no-thinking
```

## 7) Optional HF export

```bash
ENABLE_EXPORT=true EXPORT_SNAP_ANE=true RECOMPUTE_INDICES=true \
bash scripts/test-gemma3-aq1-noqat.sh
```

Export dir:

- `runs/gemma3_aq1_noqat/hf_export`

## 8) Progress output for iMatrix

Default progress is enabled every 10 steps. Override with:

```bash
IMATRIX_PROGRESS_EVERY=5 bash scripts/test-gemma3-aq1-noqat.sh
```

Use `IMATRIX_PROGRESS_EVERY=0` to disable progress output.
