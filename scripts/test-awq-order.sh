#!/usr/bin/env bash
set -euo pipefail

# test-awq-order.sh
# Compare transform order for AWQ-style Pass-A (X-matrix) and Pass-B (Y-matrix),
# while ensuring the FINAL LUTs are built via per-tensor k-means (Family G).
#
# Pipelines:
#   1) A→B: Pass-A then Pass-B (up→down only)
#   2) B→A: Pass-B then Pass-A (up→down only)
#
# For each pipeline:
#   - Build transformed base folder
#   - Compute IM_final on final base (for k-means/iActMSE)
#   - V2 init with fixed LUT (fp4_dense) for consistency
#   - Per-tensor LUT selection with families E,G (G = k-means) using IM_final
#   - Measure PPL (max-chunks or full)

BASE=${BASE:-Qwen/Qwen3-0.6B}

ALPHA_A=${ALPHA_A:-0.45}
ALPHA_ROW=${ALPHA_ROW:-0.10}

ROW_MIN=${ROW_MIN:-0.85}
ROW_MAX=${ROW_MAX:-1.18}

TOKENS=${TOKENS:-50000}
SEQ_LEN=${SEQ_LEN:-512}
BATCH_SIZE=${BATCH_SIZE:-1}
CALIB_MODE=${CALIB_MODE:-random_ids}

DEVICE=${DEVICE:-cpu}
DTYPE=${DTYPE:-float32}

V2_CONFIG=${V2_CONFIG:-q4a4_r32}
FIXED_LUT=${FIXED_LUT:-fp4_dense}

WORKERS=${WORKERS:-8}
SEL_FAMILIES=${SEL_FAMILIES:-E,G}

EnablePPL=${EnablePPL:-20}

# PPL device auto-detect
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

RUN_TAG="a${ALPHA_A}_y${ALPHA_ROW}_cl${ROW_MIN}-${ROW_MAX}"
OUTDIR="runs/order_diag_${RUN_TAG}"
mkdir -p "${OUTDIR}"

RESULTS_CSV="${OUTDIR}/results_order.csv"
echo "order,stage,checkpoint,ppl" > "${RESULTS_CSV}"

run_ppl () {
  local ckpt="$1"
  if [[ "${EnablePPL}" == "false" ]]; then
    echo "N/A"; return 0
  elif [[ "${EnablePPL}" == "true" ]]; then
    python scripts/measure_perplexity.py "${ckpt}" --device "${PPL_DEVICE}" --dtype fp16 --output-ppl 2>/dev/null \
      | awk -F= '/^PPL=/{print $2; exit}'
  elif [[ "${EnablePPL}" =~ ^[0-9]+$ ]]; then
    python scripts/measure_perplexity.py "${ckpt}" --device "${PPL_DEVICE}" --dtype fp16 --output-ppl --max-chunks "${EnablePPL}" 2>/dev/null \
      | awk -F= '/^PPL=/{print $2; exit}'
  else
    echo "N/A"; return 0
  fi
}

do_quant_eval () {
  local tag="$1"
  local base_id="$2"
  local im_final="$3"
  local out_prefix="$4"

  local out_init="${OUTDIR}/${out_prefix}_init"
  local ckpt_init="${out_init}/v2_initial.pt"
  local ckpt_hybrid="${OUTDIR}/${out_prefix}_hybrid_EG.pt"

  echo "    init_model_v2 -> ${ckpt_init}"
  python scripts/init_model_v2.py \
    --model-id "${base_id}" \
    --output "${out_init}" \
    --config "${V2_CONFIG}" \
    --lut "${FIXED_LUT}" \
    --imatrix "${im_final}" \
    --svd-error >/dev/null

  local ppl_init
  ppl_init=$(run_ppl "${ckpt_init}")
  echo "    PPL(init)   = ${ppl_init}"
  echo "${tag},init,${ckpt_init},${ppl_init}" >> "${RESULTS_CSV}"

  echo "    select_best_lut_per_layer (k-means G) -> ${ckpt_hybrid}"
  python scripts/select_best_lut_per_layer.py "${ckpt_init}" \
    --model-id "${base_id}" \
    --output "${ckpt_hybrid}" \
    --families "${SEL_FAMILIES}" \
    --metric iActMSE \
    --imatrix "${im_final}" \
    --workers "${WORKERS}" >/dev/null

  local ppl_h
  ppl_h=$(run_ppl "${ckpt_hybrid}")
  echo "    PPL(hybrid) = ${ppl_h}"
  echo "${tag},hybrid,${ckpt_hybrid},${ppl_h}" >> "${RESULTS_CSV}"
  echo ""
}

echo "============================================================"
echo "ORDER TEST: Pass-A vs Pass-B (k-means LUTs)"
echo "============================================================"
echo "BASE:        ${BASE}"
echo "ALPHA_A:     ${ALPHA_A}"
echo "ALPHA_ROW:   ${ALPHA_ROW} (up→down only)"
echo "ROW clamp:   [${ROW_MIN}, ${ROW_MAX}]"
echo "CALIB:       ${CALIB_MODE} tokens=${TOKENS} seq=${SEQ_LEN}"
echo "V2_CONFIG:   ${V2_CONFIG} init LUT=${FIXED_LUT}"
echo "Select:      families=${SEL_FAMILIES} workers=${WORKERS}"
echo "PPL:         EnablePPL=${EnablePPL}  PPL_DEVICE=${PPL_DEVICE}"
echo "OUTDIR:      ${OUTDIR}"
echo "============================================================"
echo ""

# IM on original base (for Pass-A driving stats)
IM_BASE="${OUTDIR}/imatrix_base.pt"
if [[ ! -f "${IM_BASE}" ]]; then
  echo "[0] compute_imatrix on BASE"
  python scripts/compute_imatrix.py \
    --model "${BASE}" \
    --calib-mode "${CALIB_MODE}" \
    --tokens "${TOKENS}" --seq-len "${SEQ_LEN}" --batch-size "${BATCH_SIZE}" \
    --out "${IM_BASE}" --trust-remote-code --verbose
  echo ""
fi

# -------------------------
# A→B
# -------------------------
A_BASE="${OUTDIR}/base_A"
AB_BASE="${OUTDIR}/base_A_then_B"
YM_A_UP="${OUTDIR}/ymatrix_A_uponly.pt"
IM_AB="${OUTDIR}/imatrix_A_then_B.pt"

echo "[1] A→B: Pass-A on BASE -> base_A"
python scripts/apply_awq_equiv_scales_y2.py \
  --model-id "${BASE}" \
  --imatrix "${IM_BASE}" \
  --alpha "${ALPHA_A}" \
  --output "${A_BASE}" \
  --dtype "${DTYPE}" --device "${DEVICE}" >/dev/null
echo ""

echo "[2] A→B: compute Y-matrix on base_A (up_proj only)"
python scripts/compute_ymatrix.py \
  --model "${A_BASE}" \
  --calib-mode "${CALIB_MODE}" \
  --tokens "${TOKENS}" --seq-len "${SEQ_LEN}" --batch-size "${BATCH_SIZE}" \
  --targets-regex ".*\.mlp\.up_proj$" \
  --out "${YM_A_UP}" --trust-remote-code --verbose
echo ""

echo "[3] A→B: Pass-B on base_A -> base_A_then_B"
python scripts/apply_awq_equiv_scales_y2.py \
  --model-id "${A_BASE}" \
  --imatrix "${IM_BASE}" \
  --alpha 0.0 \
  --ymatrix "${YM_A_UP}" \
  --alpha-row "${ALPHA_ROW}" \
  --row-min-scale "${ROW_MIN}" --row-max-scale "${ROW_MAX}" \
  --output "${AB_BASE}" \
  --dtype "${DTYPE}" --device "${DEVICE}" >/dev/null
echo ""

echo "[4] A→B: compute IM_final on base_A_then_B"
python scripts/compute_imatrix.py \
  --model "${AB_BASE}" \
  --calib-mode "${CALIB_MODE}" \
  --tokens "${TOKENS}" --seq-len "${SEQ_LEN}" --batch-size "${BATCH_SIZE}" \
  --out "${IM_AB}" --trust-remote-code --verbose
echo ""

echo "[5] A→B: quantize + k-means LUT selection"
do_quant_eval "AtoB" "${AB_BASE}" "${IM_AB}" "AtoB"

# -------------------------
# B→A
# -------------------------
B_BASE="${OUTDIR}/base_B"
BA_BASE="${OUTDIR}/base_B_then_A"
YM_BASE_UP="${OUTDIR}/ymatrix_base_uponly.pt"
IM_B="${OUTDIR}/imatrix_B.pt"
IM_BA="${OUTDIR}/imatrix_B_then_A.pt"

echo "[6] B→A: compute Y-matrix on BASE (up_proj only)"
python scripts/compute_ymatrix.py \
  --model "${BASE}" \
  --calib-mode "${CALIB_MODE}" \
  --tokens "${TOKENS}" --seq-len "${SEQ_LEN}" --batch-size "${BATCH_SIZE}" \
  --targets-regex ".*\.mlp\.up_proj$" \
  --out "${YM_BASE_UP}" --trust-remote-code --verbose
echo ""

echo "[7] B→A: Pass-B on BASE -> base_B"
python scripts/apply_awq_equiv_scales_y2.py \
  --model-id "${BASE}" \
  --imatrix "${IM_BASE}" \
  --alpha 0.0 \
  --ymatrix "${YM_BASE_UP}" \
  --alpha-row "${ALPHA_ROW}" \
  --row-min-scale "${ROW_MIN}" --row-max-scale "${ROW_MAX}" \
  --output "${B_BASE}" \
  --dtype "${DTYPE}" --device "${DEVICE}" >/dev/null
echo ""

echo "[8] B→A: compute IM on base_B"
python scripts/compute_imatrix.py \
  --model "${B_BASE}" \
  --calib-mode "${CALIB_MODE}" \
  --tokens "${TOKENS}" --seq-len "${SEQ_LEN}" --batch-size "${BATCH_SIZE}" \
  --out "${IM_B}" --trust-remote-code --verbose
echo ""

echo "[9] B→A: Pass-A on base_B -> base_B_then_A"
python scripts/apply_awq_equiv_scales_y2.py \
  --model-id "${B_BASE}" \
  --imatrix "${IM_B}" \
  --alpha "${ALPHA_A}" \
  --output "${BA_BASE}" \
  --dtype "${DTYPE}" --device "${DEVICE}" >/dev/null
echo ""

echo "[10] B→A: compute IM_final on base_B_then_A"
python scripts/compute_imatrix.py \
  --model "${BA_BASE}" \
  --calib-mode "${CALIB_MODE}" \
  --tokens "${TOKENS}" --seq-len "${SEQ_LEN}" --batch-size "${BATCH_SIZE}" \
  --out "${IM_BA}" --trust-remote-code --verbose
echo ""

echo "[11] B→A: quantize + k-means LUT selection"
do_quant_eval "BtoA" "${BA_BASE}" "${IM_BA}" "BtoA"

echo "============================================================"
echo "DONE. Results:"
echo "  ${RESULTS_CSV}"
echo ""
column -s, -t "${RESULTS_CSV}" | sed -e 's/  \+/ /g' || true
