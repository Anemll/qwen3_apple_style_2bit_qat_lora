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

## 5b) Fast stage-by-stage PPL screening (recommended for debugging)

This runs tiny PPL checks after:

- Step 1 (FP16-scaled model)
- Step 3 (AWQ-scaled model)
- Step 4 (AQ1 init checkpoint)

Example:

```bash
FAST_PPL_CHECK=true FAST_PPL_DEVICE=cpu FAST_PPL_DTYPE=fp32 \
FAST_PPL_MAX_CHUNKS=2 FAST_PPL_BATCH_SIZE=1 FAST_PPL_SEQ_LEN=256 \
ENABLE_PPL=false FORCE_REINIT=1 \
bash scripts/test-gemma3-aq1-noqat.sh
```

Optional fail-fast threshold:

```bash
FAST_PPL_CHECK=true FAST_PPL_ABORT_ABOVE=120 FAST_PPL_FAIL_FAST=true \
ENABLE_PPL=false FORCE_REINIT=1 \
bash scripts/test-gemma3-aq1-noqat.sh
```

## 5c) Logging to file (for remote sharing)

The pipeline now writes:

- Full log: `${RUN_ROOT}/gemma3_aq1_noqat_<timestamp>.log`
- Compact summary: `${RUN_ROOT}/gemma3_aq1_noqat_latest.summary`

Explicit path example:

```bash
ENABLE_LOG_FILE=true LOG_DIR=runs/gemma3_aq1_noqat \
FAST_PPL_CHECK=true ENABLE_PPL=false FORCE_REINIT=1 \
bash scripts/test-gemma3-aq1-noqat.sh
```

Then share:

```bash
tail -n 120 runs/gemma3_aq1_noqat/gemma3_aq1_noqat_latest.summary
tail -n 400 runs/gemma3_aq1_noqat/gemma3_aq1_noqat_*.log
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
