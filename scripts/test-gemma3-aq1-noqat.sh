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

# Tiny stage-level PPL screening to localize where quality collapses.
run_fast_ppl_check() {
  local label="$1"   # step1|step3|step4
  local mode="$2"    # baseline|checkpoint
  local target="$3"  # model path (baseline) or checkpoint path
  local model_path="$4"

  local cmd=("${PY}" scripts/measure_perplexity.py)
  if [[ "${mode}" == "baseline" ]]; then
    cmd+=(--baseline --model "${model_path}")
  else
    cmd+=("${target}" --model "${model_path}")
  fi
  cmd+=(
    --device "${FAST_PPL_DEVICE}"
    --dtype "${FAST_PPL_DTYPE}"
    --batch-size "${FAST_PPL_BATCH_SIZE}"
    --seq-len "${FAST_PPL_SEQ_LEN}"
    --max-chunks "${FAST_PPL_MAX_CHUNKS}"
    --output-ppl
  )
  if [[ -n "${FAST_PPL_TEXT_FILE}" ]]; then
    cmd+=(--text-file "${FAST_PPL_TEXT_FILE}")
  fi

  echo ">>> Fast PPL (${label})"
  echo "CMD: ${cmd[*]}"
  local ppl_output
  local ppl_value="N/A"
  if ppl_output="$("${cmd[@]}" 2>&1)"; then
    ppl_value="$(printf '%s\n' "${ppl_output}" | awk -F= '/^PPL=/{print $2; exit}')"
    if [[ -z "${ppl_value}" ]]; then
      ppl_value="N/A"
      echo "  Warning: Fast PPL parse failed."
    else
      echo "  Fast PPL (${label}) = ${ppl_value}"
      if [[ -n "${FAST_PPL_ABORT_ABOVE}" && "${FAST_PPL_FAIL_FAST}" != "false" ]]; then
        if "${PY}" - "${ppl_value}" "${FAST_PPL_ABORT_ABOVE}" <<'PY'
import sys
p=float(sys.argv[1]); t=float(sys.argv[2])
raise SystemExit(0 if p > t else 1)
PY
        then
          echo "  ERROR: Fast PPL ${ppl_value} > threshold ${FAST_PPL_ABORT_ABOVE}; stopping."
          exit 2
        fi
      fi
    fi
  else
    local rc=$?
    echo "  Warning: Fast PPL command failed (exit=${rc})."
    printf '%s\n' "${ppl_output}" | sed 's/^/    /'
  fi

  case "${label}" in
    step1) FAST_PPL_STEP1="${ppl_value}" ;;
    step3) FAST_PPL_STEP3="${ppl_value}" ;;
    step4) FAST_PPL_STEP4="${ppl_value}" ;;
  esac
}

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
INIT_MLP_RANK="${INIT_MLP_RANK:-32}"
INIT_ATTN_RANK="${INIT_ATTN_RANK:-32}"
INIT_SEARCH_GROUP="${INIT_SEARCH_GROUP:-}"

ENABLE_PPL="${ENABLE_PPL:-false}"     # false|true|N
PPL_DTYPE="${PPL_DTYPE:-fp16}"

ENABLE_INFERENCE="${ENABLE_INFERENCE:-true}"
PROMPT="${PROMPT:-Explain what quantization-aware initialization does in one short paragraph.}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-160}"
INFER_DEVICE="${INFER_DEVICE:-auto}"
INFER_DTYPE="${INFER_DTYPE:-auto}"

ENABLE_EXPORT="${ENABLE_EXPORT:-false}"
EXPORT_SNAP_ANE="${EXPORT_SNAP_ANE:-true}"
RECOMPUTE_INDICES="${RECOMPUTE_INDICES:-true}"

FAST_PPL_CHECK="${FAST_PPL_CHECK:-true}"
FAST_PPL_DEVICE="${FAST_PPL_DEVICE:-cpu}"
FAST_PPL_DTYPE="${FAST_PPL_DTYPE:-fp32}"
FAST_PPL_MAX_CHUNKS="${FAST_PPL_MAX_CHUNKS:-2}"
FAST_PPL_BATCH_SIZE="${FAST_PPL_BATCH_SIZE:-1}"
FAST_PPL_SEQ_LEN="${FAST_PPL_SEQ_LEN:-256}"
FAST_PPL_TEXT_FILE="${FAST_PPL_TEXT_FILE:-}"
FAST_PPL_ABORT_ABOVE="${FAST_PPL_ABORT_ABOVE:-}"
FAST_PPL_FAIL_FAST="${FAST_PPL_FAIL_FAST:-false}"
FAST_PPL_STEP1="N/A"
FAST_PPL_STEP3="N/A"
FAST_PPL_STEP4="N/A"

ENABLE_LOG_FILE="${ENABLE_LOG_FILE:-true}"
LOG_DIR="${LOG_DIR:-${RUN_ROOT}}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/gemma3_aq1_noqat_$(date +%Y%m%d_%H%M%S).log}"
SUMMARY_FILE="${SUMMARY_FILE:-${LOG_DIR}/gemma3_aq1_noqat_latest.summary}"

mkdir -p "${RUN_ROOT}"
mkdir -p "${LOG_DIR}"

if is_true "${ENABLE_LOG_FILE}"; then
  # Mirror all output to a file for remote debugging / cross-machine sharing.
  exec > >(tee -a "${LOG_FILE}") 2>&1
fi

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
echo "INIT_RANKS:      mlp=${INIT_MLP_RANK}, attn=${INIT_ATTN_RANK}"
echo "USE_FP16_SCALING:${USE_FP16_SCALING} (alpha=${FP16_ALPHA})"
echo "AWQ_ALPHA:       ${AWQ_ALPHA}"
echo "IM_PROGRESS:     every ${IMATRIX_PROGRESS_EVERY} step(s)"
echo "SEARCH_LUT:      ${SEARCH_LUT} (fixed_lut=${FIXED_LUT})"
echo "ENABLE_PPL:      ${ENABLE_PPL} (device=${PPL_DEVICE})"
echo "ENABLE_INFERENCE:${ENABLE_INFERENCE}"
echo "INFER_RUNTIME:   device=${INFER_DEVICE}, dtype=${INFER_DTYPE}"
echo "ENABLE_EXPORT:   ${ENABLE_EXPORT}"
echo "FAST_PPL_CHECK:  ${FAST_PPL_CHECK} (dev=${FAST_PPL_DEVICE}, dtype=${FAST_PPL_DTYPE}, chunks=${FAST_PPL_MAX_CHUNKS}, batch=${FAST_PPL_BATCH_SIZE}, seq=${FAST_PPL_SEQ_LEN})"
echo "LOG_FILE:        ${LOG_FILE}"
echo "SUMMARY_FILE:    ${SUMMARY_FILE}"
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

if is_true "${FAST_PPL_CHECK}"; then
  run_fast_ppl_check "step1" "baseline" "-" "${SOURCE_MODEL}"
  echo ""
fi

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

if is_true "${FAST_PPL_CHECK}"; then
  run_fast_ppl_check "step3" "baseline" "-" "${AWQ_MODEL_DIR}"
  echo ""
fi

# -------------------------
# Step 4: AQ1 init (no QAT training)
# -------------------------
REINIT_REASON=""
if [[ "${FORCE_REINIT}" != "1" && -f "${INIT_CKPT}" ]]; then
  INIT_CFG="${INIT_DIR}/config.json"
  if [[ ! -f "${INIT_CFG}" ]]; then
    REINIT_REASON="missing config.json for existing checkpoint"
  else
    # Verify reused checkpoint matches requested quant config/ranks/model.
    cfg_row="$("${PY}" - "${INIT_CFG}" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
model_id = cfg.get("model_id", "")
mlp_rank = cfg.get("mlp_scale_rank", cfg.get("scale_rank"))
attn_rank = cfg.get("attn_scale_rank", mlp_rank)
mlp_bits = cfg.get("mlp_lut_bits", cfg.get("lut_bits"))
attn_bits = cfg.get("attn_lut_bits", mlp_bits)
print(f"{model_id}\t{mlp_rank}\t{attn_rank}\t{mlp_bits}\t{attn_bits}")
PY
)"
    IFS=$'\t' read -r cfg_model_id cfg_mlp_rank cfg_attn_rank cfg_mlp_bits cfg_attn_bits <<< "${cfg_row}"

    exp_mlp_bits=""
    exp_attn_bits=""
    case "${AQ1_CONFIG}" in
      q2a4) exp_mlp_bits="2"; exp_attn_bits="4" ;;
      q2a2) exp_mlp_bits="2"; exp_attn_bits="2" ;;
      q4a4|q4a4_r32|q4_r32) exp_mlp_bits="4"; exp_attn_bits="4" ;;
    esac

    mismatch=0
    if [[ "${cfg_model_id}" != "${AWQ_MODEL_DIR}" ]]; then
      REINIT_REASON+=" model_id(${cfg_model_id} != ${AWQ_MODEL_DIR});"
      mismatch=1
    fi
    if [[ "${cfg_mlp_rank}" != "${INIT_MLP_RANK}" ]]; then
      REINIT_REASON+=" mlp_rank(${cfg_mlp_rank} != ${INIT_MLP_RANK});"
      mismatch=1
    fi
    if [[ "${cfg_attn_rank}" != "${INIT_ATTN_RANK}" ]]; then
      REINIT_REASON+=" attn_rank(${cfg_attn_rank} != ${INIT_ATTN_RANK});"
      mismatch=1
    fi
    if [[ -n "${exp_mlp_bits}" && "${cfg_mlp_bits}" != "${exp_mlp_bits}" ]]; then
      REINIT_REASON+=" mlp_lut_bits(${cfg_mlp_bits} != ${exp_mlp_bits});"
      mismatch=1
    fi
    if [[ -n "${exp_attn_bits}" && "${cfg_attn_bits}" != "${exp_attn_bits}" ]]; then
      REINIT_REASON+=" attn_lut_bits(${cfg_attn_bits} != ${exp_attn_bits});"
      mismatch=1
    fi

    if [[ "${mismatch}" == "0" ]]; then
      REINIT_REASON=""
    fi
  fi
fi

if [[ "${FORCE_REINIT}" == "1" || ! -f "${INIT_CKPT}" || -n "${REINIT_REASON}" ]]; then
  echo ">>> Step 4: Initialize AQ1 V2 checkpoint"
  if [[ -n "${REINIT_REASON}" ]]; then
    echo "    Reinit reason:${REINIT_REASON}"
  fi
  init_cmd=(
    "${PY}" scripts/init_model_v2.py
    --model-id "${AWQ_MODEL_DIR}"
    --output "${INIT_DIR}"
    --config "${AQ1_CONFIG}"
    --imatrix "${IMATRIX_PATH}"
    --mlp-rank "${INIT_MLP_RANK}"
    --attn-rank "${INIT_ATTN_RANK}"
  )
  if [[ -n "${INIT_SEARCH_GROUP}" ]]; then
    init_cmd+=(--search-group "${INIT_SEARCH_GROUP}")
  fi
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

if is_true "${FAST_PPL_CHECK}"; then
  run_fast_ppl_check "step4" "checkpoint" "${INIT_CKPT}" "${AWQ_MODEL_DIR}"
  echo ""
fi

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
  if ppl_output="$("${ppl_cmd[@]}" 2>&1)"; then
    PPL_VALUE="$(printf '%s\n' "${ppl_output}" | awk -F= '/^PPL=/{print $2; exit}')"
    if [[ -z "${PPL_VALUE}" ]]; then
      PPL_VALUE="N/A"
      echo "  Warning: Could not parse PPL from measure_perplexity output."
    else
      echo "  PPL=${PPL_VALUE}"
    fi
  else
    ppl_rc=$?
    PPL_VALUE="N/A"
    echo "  Warning: perplexity command failed (exit=${ppl_rc}). Continuing to inference."
    echo "  Perplexity output:"
    printf '%s\n' "${ppl_output}" | sed 's/^/    /'
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
    --prompt "${PROMPT}" \
    --max-tokens "${MAX_NEW_TOKENS}" \
    --device "${INFER_DEVICE}" \
    --dtype "${INFER_DTYPE}" \
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
if is_true "${FAST_PPL_CHECK}"; then
  echo "Fast PPL s1/s3/s4: ${FAST_PPL_STEP1} / ${FAST_PPL_STEP3} / ${FAST_PPL_STEP4}"
fi
if is_true "${ENABLE_EXPORT}"; then
  echo "HF export:    ${EXPORT_DIR}"
fi
echo "============================================================"

{
  echo "timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "run_root=${RUN_ROOT}"
  echo "base_model=${BASE_MODEL}"
  echo "awq_model=${AWQ_MODEL_DIR}"
  echo "aq1_checkpoint=${INIT_CKPT}"
  echo "ppl=${PPL_VALUE}"
  echo "fast_ppl_step1=${FAST_PPL_STEP1}"
  echo "fast_ppl_step3=${FAST_PPL_STEP3}"
  echo "fast_ppl_step4=${FAST_PPL_STEP4}"
  echo "infer_device=${INFER_DEVICE}"
  echo "infer_dtype=${INFER_DTYPE}"
  echo "enable_inference=${ENABLE_INFERENCE}"
  echo "enable_export=${ENABLE_EXPORT}"
  echo "log_file=${LOG_FILE}"
} > "${SUMMARY_FILE}"
echo "Wrote summary: ${SUMMARY_FILE}"
