#!/usr/bin/env bash
set -euo pipefail

# Test AWQ pipeline end-to-end, including optional Pass-B row-smoothing (Y-matrix)
#
# Key rules:
#  - If you AWQ-scale the base, every downstream step (init, select LUT, PPL, inference)
#    MUST use the same base (the AWQ-scaled model folder), otherwise you'll optimize
#    against the wrong W_ref and the model can collapse.
#  - Include Family E during LUT selection so "Original" baseline isn't inf.

# -------------------------
# Config knobs
# -------------------------
BASE=${BASE:-Qwen/Qwen3-0.6B}

# iMatrix (input stats) and Y-matrix (output stats)
IM=${IM:-runs/imatrix_qwen3_0.6b_random.pt}
YM=${YM:-runs/ymatrix_up_only.pt}
# Which module outputs to collect for Y-matrix. Default: up_proj only (v→o Pass-B was harmful).
YM_TARGETS=${YM_TARGETS:-".*\.mlp\.up_proj$"}

# Pass A (column) and Pass B (row) strengths
ALPHA_A=${ALPHA_A:-0.4}          # best from your sweep so far
ALPHA_ROW=${ALPHA_ROW:-0.2}      # start small; set 0 to disable Pass B

# Clamp for Pass B row scales (tighter is safer)
ROW_MIN=${ROW_MIN:-0.5}
ROW_MAX=${ROW_MAX:-2.0}

# Calibration for matrices
TOKENS=${TOKENS:-50000}
SEQ_LEN=${SEQ_LEN:-512}
BATCH_SIZE=${BATCH_SIZE:-1}
CALIB_MODE=${CALIB_MODE:-random_ids}   # random_ids|pseudo_text|textfile

# Eval device for init/transforms (CPU is safest)
DEVICE=${DEVICE:-cpu}   # cpu|tpu|cuda|mps
DTYPE=${DTYPE:-float32} # float32|float16|bfloat16

# PPL device: auto-detect best available accelerator
# Priority: TPU > CUDA > MPS (Mac) > CPU
if [[ -z "${PPL_DEVICE:-}" ]]; then
  if python -c "import torch_xla" 2>/dev/null; then
    PPL_DEVICE="tpu"
  elif python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    PPL_DEVICE="cuda"
  elif python -c "import torch; assert torch.backends.mps.is_available()" 2>/dev/null; then
    PPL_DEVICE="mps"
  else
    PPL_DEVICE="cpu"
  fi
fi

# PPL measurement: true=full, false=skip, N=max-chunks (e.g., 20)
EnablePPL=${EnablePPL:-false}

# V2 init config preset
V2_CONFIG=${V2_CONFIG:-q4a4_r32}

# LUT selection
#FAMILIES=${FAMILIES:-E,B,G}  # include E baseline
FAMILIES=${FAMILIES:-A,B,C,D,E,F,G,H}  # include E baseline
WORKERS=${WORKERS:-8}

# Output dirs
BASE_A="runs/awq_scaled_a${ALPHA_A}"
BASE_AB="runs/awq_scaled_a${ALPHA_A}_y${ALPHA_ROW}"
OUT_INIT="runs/v2_awq_a${ALPHA_A}_y${ALPHA_ROW}"
HYBRID="${OUT_INIT}/ihybrid.pt"

# Collect PPL results (simple vars for bash 3.x compat)
PPL_V2_INIT="N/A"
PPL_HYBRID="N/A"

echo "============================================================"
echo "AWQ + Y-matrix pipeline"
echo "============================================================"
echo "BASE:       ${BASE}"
echo "IM:         ${IM}"
echo "YM:         ${YM}"
echo "YM_TARGETS:${YM_TARGETS}"
echo "ALPHA_A:    ${ALPHA_A}"
echo "ALPHA_ROW:  ${ALPHA_ROW}"
echo "DEVICE:     ${DEVICE}"
echo "PPL_DEVICE: ${PPL_DEVICE}"
echo "DTYPE:      ${DTYPE}"
echo "V2_CONFIG:  ${V2_CONFIG}"
echo "EnablePPL:  ${EnablePPL}"
echo "OUT_INIT:   ${OUT_INIT}"
echo "============================================================"
echo ""

# -------------------------
# Step 0: iMatrix
# -------------------------
if [[ ! -f "${IM}" ]]; then
  echo ">>> Step 0: iMatrix not found; creating (${TOKENS} tokens, ${SEQ_LEN} ctx)"
  cmd="python scripts/compute_imatrix.py --model ${BASE} --calib-mode ${CALIB_MODE} --tokens ${TOKENS} --seq-len ${SEQ_LEN} --batch-size ${BATCH_SIZE} --out ${IM} --trust-remote-code --verbose"
  echo "CMD: $cmd"
  if [[ "${FORCE_REYM:-0}" == "1" || ! -f "${YM}" ]]; then
    $cmd
  else
    echo ">>> Step 1B: Using existing Y-matrix: ${YM}"
  fi
  echo ""
fi

# -------------------------
# Step 1A: Apply AWQ Pass A (column scaling + inverse)
# -------------------------
echo ">>> Step 1A: Applying AWQ-equivalent Pass A (alpha=${ALPHA_A})"
cmd="python scripts/apply_awq_equiv_scales_y2.py --model-id ${BASE} --imatrix ${IM} --alpha ${ALPHA_A} --output ${BASE_A} --dtype ${DTYPE} --device ${DEVICE}"
echo "CMD: $cmd"
$cmd
echo ""

# -------------------------
# Step 1B: Compute Y-matrix on the Pass-A base (optional)
# -------------------------
if [[ "${ALPHA_ROW}" != "0" && "${ALPHA_ROW}" != "0.0" ]]; then
  echo ">>> Step 1B: Computing Y-matrix on Pass-A base (${TOKENS} tokens)"
  cmd="python scripts/compute_ymatrix.py --model ${BASE_A} --calib-mode ${CALIB_MODE} --tokens ${TOKENS} --seq-len ${SEQ_LEN} --batch-size ${BATCH_SIZE} --targets-regex ${YM_TARGETS} --out ${YM} --trust-remote-code --verbose"
  echo "CMD: $cmd"
  if [[ "${FORCE_REYM:-0}" == "1" || ! -f "${YM}" ]]; then
    $cmd
  else
    echo ">>> Step 1B: Using existing Y-matrix: ${YM}"
  fi
  echo ""

  # -------------------------
  # Step 1C: Apply Pass B (row-smoothing) using Y-matrix
  # -------------------------
  echo ">>> Step 1C: Applying Pass B row-smoothing (alpha-row=${ALPHA_ROW})"
  cmd="python scripts/apply_awq_equiv_scales_y2.py --model-id ${BASE_A} --imatrix ${IM} --alpha 0.0 --ymatrix ${YM} --alpha-row ${ALPHA_ROW} --row-min-scale ${ROW_MIN} --row-max-scale ${ROW_MAX} --output ${BASE_AB} --dtype ${DTYPE} --device ${DEVICE}"
  echo "CMD: $cmd"
  if [[ "${FORCE_REYM:-0}" == "1" || ! -f "${YM}" ]]; then
    $cmd
  else
    echo ">>> Step 1B: Using existing Y-matrix: ${YM}"
  fi
  echo ""
else
  echo ">>> Pass B disabled (ALPHA_ROW=0). Using Pass-A base as final base."
  BASE_AB="${BASE_A}"
  echo ""
fi

# -------------------------
# Step 2: Initialize V2 from the final base
# -------------------------
echo ">>> Step 2: init_model_v2.py from base: ${BASE_AB}"
cmd="python scripts/init_model_v2.py --model-id ${BASE_AB} --output ${OUT_INIT} --config ${V2_CONFIG} --search-lut --imatrix ${IM} --svd-error"
echo "CMD: $cmd"
$cmd
echo ""

# -------------------------
# Step 3: Perplexity on v2_initial.pt
# -------------------------
if [[ "${EnablePPL}" == "false" ]]; then
  echo ">>> Step 3: Skipped (EnablePPL=false)"
  PPL_V2_INIT="N/A"
  echo ""
elif [[ "${EnablePPL}" == "true" ]]; then
  echo ">>> Step 3: Perplexity (v2_initial.pt) - full [${PPL_DEVICE}]"
  cmd="python scripts/measure_perplexity.py ${OUT_INIT}/v2_initial.pt --device ${PPL_DEVICE} --dtype fp16 --output-ppl"
  echo "CMD: $cmd"
  PPL_OUTPUT=$($cmd 2>&1)
  PPL=$(echo "$PPL_OUTPUT" | grep "^PPL=" | cut -d= -f2)
  PPL_V2_INIT=$PPL
  echo "    PPL (v2_init) = $PPL"
  echo ""
elif [[ "${EnablePPL}" =~ ^[0-9]+$ ]]; then
  echo ">>> Step 3: Perplexity (v2_initial.pt) - max-chunks=${EnablePPL} [${PPL_DEVICE}]"
  cmd="python scripts/measure_perplexity.py ${OUT_INIT}/v2_initial.pt --device ${PPL_DEVICE} --dtype fp16 --output-ppl --max-chunks ${EnablePPL}"
  echo "CMD: $cmd"
  PPL_OUTPUT=$($cmd 2>&1)
  PPL=$(echo "$PPL_OUTPUT" | grep "^PPL=" | cut -d= -f2)
  PPL_V2_INIT=$PPL
  echo "    PPL (v2_init) = $PPL"
  echo ""
else
  echo ">>> Step 3: Skipped (EnablePPL=${EnablePPL} invalid)"
  PPL_V2_INIT="N/A"
  echo ""
fi

# -------------------------
# Step 4: Select best LUT per tensor (per-layer script)
# IMPORTANT: must use the SAME base (BASE_AB) for W_ref!
# -------------------------
echo ">>> Step 4: select_best_lut_per_layer (families=${FAMILIES})"
cmd="python scripts/select_best_lut_per_layer.py ${OUT_INIT}/v2_initial.pt --model-id ${BASE_AB} --output ${HYBRID} --families ${FAMILIES} --metric iActMSE --imatrix ${IM_FINAL} --workers ${WORKERS}"
echo "CMD: $cmd"
$cmd
echo ""

# -------------------------
# Step 5: Perplexity on hybrid
# -------------------------
if [[ "${EnablePPL}" == "false" ]]; then
  echo ">>> Step 5: Skipped (EnablePPL=false)"
  PPL_HYBRID="N/A"
  echo ""
elif [[ "${EnablePPL}" == "true" ]]; then
  echo ">>> Step 5: Perplexity (hybrid) - full [${PPL_DEVICE}]"
  cmd="python scripts/measure_perplexity.py ${HYBRID} --device ${PPL_DEVICE} --dtype fp16 --output-ppl"
  echo "CMD: $cmd"
  PPL_OUTPUT=$($cmd 2>&1)
  PPL=$(echo "$PPL_OUTPUT" | grep "^PPL=" | cut -d= -f2)
  PPL_HYBRID=$PPL
  echo "    PPL (hybrid) = $PPL"
  echo ""
elif [[ "${EnablePPL}" =~ ^[0-9]+$ ]]; then
  echo ">>> Step 5: Perplexity (hybrid) - max-chunks=${EnablePPL} [${PPL_DEVICE}]"
  cmd="python scripts/measure_perplexity.py ${HYBRID} --device ${PPL_DEVICE} --dtype fp16 --output-ppl --max-chunks ${EnablePPL}"
  echo "CMD: $cmd"
  PPL_OUTPUT=$($cmd 2>&1)
  PPL=$(echo "$PPL_OUTPUT" | grep "^PPL=" | cut -d= -f2)
  PPL_HYBRID=$PPL
  echo "    PPL (hybrid) = $PPL"
  echo ""
else
  echo ">>> Step 5: Skipped (EnablePPL=${EnablePPL} invalid)"
  PPL_HYBRID="N/A"
  echo ""
fi

# -------------------------
# Step 6: Quick inference sanity
# -------------------------
echo ">>> Step 6: Inference sanity (hybrid)"
cmd="python scripts/test_inference.py ${HYBRID} --prompt 'Who invented the iPad?' --no-think --config q4_r32"
echo "CMD: $cmd"
python scripts/test_inference.py "${HYBRID}" --prompt "Who invented the iPad?" --no-think --config q4_r32
echo ""

echo "============================================================"
echo "SUMMARY: AWQ + Y-matrix Results (ALPHA_A=${ALPHA_A}, ALPHA_ROW=${ALPHA_ROW})"
echo "============================================================"
printf "%-20s | %-12s | %s\n" "Stage" "PPL" "Checkpoint"
echo "---------------------|--------------|------------------------------------"
printf "%-20s | %-12s | %s\n" "V2 Init" "${PPL_V2_INIT}" "${OUT_INIT}/v2_initial.pt"
printf "%-20s | %-12s | %s\n" "Hybrid" "${PPL_HYBRID}" "${HYBRID}"
echo "============================================================"
echo "  Base A:    ${BASE_A}"
echo "  Base AB:   ${BASE_AB}"
echo "============================================================"

# Compute improvement
if [[ -n "${PPL_V2_INIT}" && -n "${PPL_HYBRID}" && "${PPL_V2_INIT}" != "N/A" && "${PPL_HYBRID}" != "N/A" ]]; then
  DELTA=$(echo "${PPL_V2_INIT} - ${PPL_HYBRID}" | bc -l 2>/dev/null)
  if [[ -n "$DELTA" ]]; then
    echo ""
    echo "Improvement: $DELTA PPL (init -> hybrid)"
  fi
fi
echo ""
