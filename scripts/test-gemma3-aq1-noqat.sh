#!/usr/bin/env bash
set -euo pipefail

# Gemma3 AQ1 smoke pipeline (no QAT training).
#
# Sequence:
#   1) (Optional) Apply Gemma3 FP16 residual scaling
#   2) Compute iMatrix
#   3) Apply Gemma3 AWQ-equivalent scaling
#   4) Initialize AQ1 V2 checkpoint (q2a4 by default)
#   5) (Optional) Measure perplexity + run inference prompt
#   6) (Optional) Export baked HF weights
#
# Usage:
#   bash scripts/test-gemma3-aq1-noqat.sh
#
# Example overrides:
#   BASE_MODEL=google/gemma-3-270m-it ENABLE_PPL=20 bash scripts/test-gemma3-aq1-noqat.sh
#   SEARCH_LUT=true FORCE_REINIT=1 bash scripts/test-gemma3-aq1-noqat.sh
#   ENABLE_EXPORT=true EXPORT_SNAP_ANE=true bash scripts/test-gemma3-aq1-noqat.sh

is_true() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

pick_python() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    echo "${PYTHON_BIN}"
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    echo "python3"
    return 0
  fi
  if command -v python >/dev/null 2>&1; then
    echo "python"
    return 0
  fi
  echo "ERROR: neither python3 nor python is available on PATH" >&2
  exit 1
}

run_cmd() {
  echo "CMD: $*"
  "$@"
}

PY="$(pick_python)"

# -------------------------
# Config (override via env)
# -------------------------
BASE_MODEL="${BASE_MODEL:-google/gemma-3-1b-it}"
RUN_ROOT="${RUN_ROOT:-runs/gemma3_aq1_noqat}"
AQ1_CONFIG="${AQ1_CONFIG:-q2a4}"

USE_FP16_SCALING="${USE_FP16_SCALING:-true}"
FP16_ALPHA="${FP16_ALPHA:-auto}"
FP16_DTYPE="${FP16_DTYPE:-float32}"
FORCE_REFP16="${FORCE_REFP16:-0}"

TOKENS="${TOKENS:-50000}"
SEQ_LEN="${SEQ_LEN:-512}"
BATCH_SIZE="${BATCH_SIZE:-1}"
CALIB_MODE="${CALIB_MODE:-random_ids}"
IMATRIX_DEVICE="${IMATRIX_DEVICE:-auto}"
IMATRIX_DTYPE="${IMATRIX_DTYPE:-bf16}"
IMATRIX_PROGRESS_EVERY="${IMATRIX_PROGRESS_EVERY:-10}"
FORCE_REIMATRIX="${FORCE_REIMATRIX:-0}"

AWQ_ALPHA="${AWQ_ALPHA:-0.5}"
AWQ_DTYPE="${AWQ_DTYPE:-float32}"
TRANSFORM_DEVICE="${TRANSFORM_DEVICE:-cpu}"
FORCE_REAWQ="${FORCE_REAWQ:-0}"

SEARCH_LUT="${SEARCH_LUT:-false}"
FIXED_LUT="${FIXED_LUT:-fp4_dense}"
ENABLE_SVD_ERROR="${ENABLE_SVD_ERROR:-true}"
FORCE_REINIT="${FORCE_REINIT:-0}"

ENABLE_PPL="${ENABLE_PPL:-false}"     # false|true|N
PPL_DTYPE="${PPL_DTYPE:-fp16}"

ENABLE_INFERENCE="${ENABLE_INFERENCE:-true}"
PROMPT="${PROMPT:-Explain what quantization-aware initialization does in one short paragraph.}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-160}"

ENABLE_EXPORT="${ENABLE_EXPORT:-false}"
EXPORT_SNAP_ANE="${EXPORT_SNAP_ANE:-true}"
RECOMPUTE_INDICES="${RECOMPUTE_INDICES:-true}"

mkdir -p "${RUN_ROOT}"

ALPHA_TAG="$(echo "${AWQ_ALPHA}" | tr '.' 'p')"
FP16_MODEL_DIR="${RUN_ROOT}/model_fp16scaled"
IMATRIX_PATH="${RUN_ROOT}/imatrix.pt"
AWQ_MODEL_DIR="${RUN_ROOT}/model_awq_a${ALPHA_TAG}"
INIT_DIR="${RUN_ROOT}/v2_${AQ1_CONFIG}_init"
INIT_CKPT="${INIT_DIR}/v2_initial.pt"
EXPORT_DIR="${RUN_ROOT}/hf_export"

SOURCE_MODEL="${BASE_MODEL}"

# PPL device auto-detect: TPU > CUDA > MPS > CPU
if [[ -z "${PPL_DEVICE:-}" ]]; then
  if "${PY}" -c "import torch_xla" >/dev/null 2>&1; then
    PPL_DEVICE="tpu"
  elif "${PY}" -c "import torch; assert torch.cuda.is_available()" >/dev/null 2>&1; then
    PPL_DEVICE="cuda"
  elif "${PY}" -c "import torch; assert torch.backends.mps.is_available()" >/dev/null 2>&1; then
    PPL_DEVICE="mps"
  else
    PPL_DEVICE="cpu"
  fi
fi

echo "============================================================"
echo "Gemma3 AQ1 No-QAT Pipeline"
echo "============================================================"
echo "PY:              ${PY}"
echo "BASE_MODEL:      ${BASE_MODEL}"
echo "RUN_ROOT:        ${RUN_ROOT}"
echo "AQ1_CONFIG:      ${AQ1_CONFIG}"
echo "USE_FP16_SCALING:${USE_FP16_SCALING} (alpha=${FP16_ALPHA})"
echo "AWQ_ALPHA:       ${AWQ_ALPHA}"
echo "IM_PROGRESS:     every ${IMATRIX_PROGRESS_EVERY} step(s)"
echo "SEARCH_LUT:      ${SEARCH_LUT} (fixed_lut=${FIXED_LUT})"
echo "ENABLE_PPL:      ${ENABLE_PPL} (device=${PPL_DEVICE})"
echo "ENABLE_INFERENCE:${ENABLE_INFERENCE}"
echo "ENABLE_EXPORT:   ${ENABLE_EXPORT}"
echo "============================================================"
echo ""

# -------------------------
# Step 1: Optional FP16 residual scaling
# -------------------------
if is_true "${USE_FP16_SCALING}"; then
  if [[ "${FORCE_REFP16}" == "1" || ! -d "${FP16_MODEL_DIR}" ]]; then
    echo ">>> Step 1: Apply Gemma3 FP16 residual scaling"
    run_cmd "${PY}" scripts/apply_gemma3_fp16_scaling.py \
      --model-id "${BASE_MODEL}" \
      --alpha "${FP16_ALPHA}" \
      --dtype "${FP16_DTYPE}" \
      --output "${FP16_MODEL_DIR}"
  else
    echo ">>> Step 1: Reusing existing FP16-scaled model: ${FP16_MODEL_DIR}"
  fi
  SOURCE_MODEL="${FP16_MODEL_DIR}"
else
  echo ">>> Step 1: Skipped FP16 residual scaling"
fi
echo ""

# -------------------------
# Step 2: iMatrix
# -------------------------
if [[ "${FORCE_REIMATRIX}" == "1" || ! -f "${IMATRIX_PATH}" ]]; then
  echo ">>> Step 2: Compute iMatrix"
  run_cmd "${PY}" scripts/compute_imatrix.py \
    --model "${SOURCE_MODEL}" \
    --out "${IMATRIX_PATH}" \
    --tokens "${TOKENS}" \
    --seq-len "${SEQ_LEN}" \
    --batch-size "${BATCH_SIZE}" \
    --calib-mode "${CALIB_MODE}" \
    --device "${IMATRIX_DEVICE}" \
    --dtype "${IMATRIX_DTYPE}" \
    --progress-every "${IMATRIX_PROGRESS_EVERY}" \
    --trust-remote-code
else
  echo ">>> Step 2: Reusing existing iMatrix: ${IMATRIX_PATH}"
fi
echo ""

# -------------------------
# Step 3: Gemma3 AWQ-equivalent scaling
# -------------------------
if [[ "${FORCE_REAWQ}" == "1" || ! -d "${AWQ_MODEL_DIR}" ]]; then
  echo ">>> Step 3: Apply Gemma3 AWQ-equivalent scaling"
  run_cmd "${PY}" scripts/apply_awq_equiv_scales_gemma3.py \
    --model-id "${SOURCE_MODEL}" \
    --imatrix "${IMATRIX_PATH}" \
    --alpha "${AWQ_ALPHA}" \
    --dtype "${AWQ_DTYPE}" \
    --device "${TRANSFORM_DEVICE}" \
    --output "${AWQ_MODEL_DIR}"
else
  echo ">>> Step 3: Reusing existing AWQ-scaled model: ${AWQ_MODEL_DIR}"
fi
echo ""

# -------------------------
# Step 4: AQ1 init (no QAT training)
# -------------------------
if [[ "${FORCE_REINIT}" == "1" || ! -f "${INIT_CKPT}" ]]; then
  echo ">>> Step 4: Initialize AQ1 V2 checkpoint"
  init_cmd=(
    "${PY}" scripts/init_model_v2.py
    --model-id "${AWQ_MODEL_DIR}"
    --output "${INIT_DIR}"
    --config "${AQ1_CONFIG}"
    --imatrix "${IMATRIX_PATH}"
  )
  if is_true "${SEARCH_LUT}"; then
    init_cmd+=(--search-lut)
  else
    init_cmd+=(--lut "${FIXED_LUT}")
  fi
  if is_true "${ENABLE_SVD_ERROR}"; then
    init_cmd+=(--svd-error)
  fi
  run_cmd "${init_cmd[@]}"
else
  echo ">>> Step 4: Reusing existing V2 checkpoint: ${INIT_CKPT}"
fi
echo ""

# -------------------------
# Step 5: Optional perplexity
# -------------------------
PPL_VALUE="N/A"
if [[ "${ENABLE_PPL}" == "false" ]]; then
  echo ">>> Step 5: Skipped perplexity (ENABLE_PPL=false)"
else
  echo ">>> Step 5: Measure perplexity on v2_initial.pt"
  ppl_cmd=(
    "${PY}" scripts/measure_perplexity.py "${INIT_CKPT}"
    --model "${AWQ_MODEL_DIR}"
    --device "${PPL_DEVICE}"
    --dtype "${PPL_DTYPE}"
    --output-ppl
  )
  if [[ "${ENABLE_PPL}" =~ ^[0-9]+$ ]]; then
    ppl_cmd+=(--max-chunks "${ENABLE_PPL}")
  fi
  PPL_VALUE="$("${ppl_cmd[@]}" 2>&1 | awk -F= '/^PPL=/{print $2; exit}')"
  if [[ -z "${PPL_VALUE}" ]]; then
    PPL_VALUE="N/A"
    echo "  Warning: Could not parse PPL from measure_perplexity output."
  else
    echo "  PPL=${PPL_VALUE}"
  fi
fi
echo ""

# -------------------------
# Step 6: Optional inference prompt test
# -------------------------
if is_true "${ENABLE_INFERENCE}"; then
  echo ">>> Step 6: Run test_inference prompt"
  run_cmd "${PY}" scripts/test_inference.py "${INIT_CKPT}" \
    --model-id "${AWQ_MODEL_DIR}" \
    --config "${AQ1_CONFIG}" \
    --prompt "${PROMPT}" \
    --max-tokens "${MAX_NEW_TOKENS}" \
    --no-thinking
else
  echo ">>> Step 6: Skipped inference (ENABLE_INFERENCE=false)"
fi
echo ""

# -------------------------
# Step 7: Optional HF export
# -------------------------
if is_true "${ENABLE_EXPORT}"; then
  echo ">>> Step 7: Export V2 checkpoint to baked HF weights"
  export_cmd=(
    "${PY}" scripts/export_v2_to_hf.py
    --model-id "${AWQ_MODEL_DIR}"
    --checkpoint "${INIT_CKPT}"
    --config "${INIT_DIR}/config.json"
    --output "${EXPORT_DIR}"
  )
  if is_true "${EXPORT_SNAP_ANE}"; then
    export_cmd+=(--snap-ane)
    if is_true "${RECOMPUTE_INDICES}"; then
      export_cmd+=(--recompute-indices)
    fi
  fi
  run_cmd "${export_cmd[@]}"
else
  echo ">>> Step 7: Skipped export (ENABLE_EXPORT=false)"
fi
echo ""

echo "============================================================"
echo "Done"
echo "============================================================"
echo "iMatrix:      ${IMATRIX_PATH}"
echo "AWQ model:    ${AWQ_MODEL_DIR}"
echo "AQ1 checkpoint:${INIT_CKPT}"
echo "Perplexity:   ${PPL_VALUE}"
if is_true "${ENABLE_EXPORT}"; then
  echo "HF export:    ${EXPORT_DIR}"
fi
echo "============================================================"
