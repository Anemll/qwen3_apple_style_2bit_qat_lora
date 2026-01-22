#!/usr/bin/env bash
set -euo pipefail

# test-awq-y3k.sh
# Like Y3, but optionally runs per-tensor LUT selection using Family G (k-means).
# This lets you test the hypothesis that "stale iMatrix" makes k-means LUTs suboptimal,
# since the LUT itself is trained with the importance weights.

BASE=${BASE:-Qwen/Qwen3-0.6B}

ALPHA_A=${ALPHA_A:-0.45}
ALPHA_ROW=${ALPHA_ROW:-0.30}
ROW_MIN=${ROW_MIN:-0.5}
ROW_MAX=${ROW_MAX:-2.0}

TOKENS=${TOKENS:-50000}
SEQ_LEN=${SEQ_LEN:-512}
BATCH_SIZE=${BATCH_SIZE:-1}
CALIB_MODE=${CALIB_MODE:-random_ids}

DEVICE=${DEVICE:-cpu}
DTYPE=${DTYPE:-float32}

V2_CONFIG_BASE=${V2_CONFIG_BASE:-q4a4_r}
RANKS=${RANKS:-32}

FIXED_LUT=${FIXED_LUT:-fp4_dense}

EnablePPL=${EnablePPL:-20}

DO_G=${DO_G:-1}
SEL_FAMILIES=${SEL_FAMILIES:-E,G}
WORKERS=${WORKERS:-8}

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

RUN_TAG="a${ALPHA_A}_y${ALPHA_ROW}"
BASE_A="runs/awq_A_${RUN_TAG}"
BASE_AB_BOTH="runs/awq_AB_both_${RUN_TAG}"
BASE_AB_VONLY="runs/awq_AB_vonly_${RUN_TAG}"
BASE_AB_UPONLY="runs/awq_AB_uponly_${RUN_TAG}"

IM_A="runs/imatrix_A_${RUN_TAG}.pt"
IM_AB_BOTH="runs/imatrix_AB_both_${RUN_TAG}.pt"
IM_AB_VONLY="runs/imatrix_AB_vonly_${RUN_TAG}.pt"
IM_AB_UPONLY="runs/imatrix_AB_uponly_${RUN_TAG}.pt"

YM_BOTH="runs/ymatrix_both_${RUN_TAG}.pt"
YM_VONLY="runs/ymatrix_vonly_${RUN_TAG}.pt"
YM_UPONLY="runs/ymatrix_uponly_${RUN_TAG}.pt"

OUTDIR="runs/passb_diag_${RUN_TAG}"
mkdir -p "${OUTDIR}"

RESULTS_CSV="${OUTDIR}/results_y3k.csv"
echo "variant,rank,imatrix_used,ppl_v2_init,ppl_hybrid,ckpt_init,ckpt_hybrid" > "${RESULTS_CSV}"

echo "============================================================"
echo "PASS-B DIAG (Y3K) - optional k-means LUT selection (Family G)"
echo "============================================================"
echo "BASE:       ${BASE}"
echo "ALPHA_A:    ${ALPHA_A}"
echo "ALPHA_ROW:  ${ALPHA_ROW}"
echo "ROW clamp:  [${ROW_MIN}, ${ROW_MAX}]"
echo "CALIB:      ${CALIB_MODE} tokens=${TOKENS} seq=${SEQ_LEN} bs=${BATCH_SIZE}"
echo "INIT:       fixed LUT=${FIXED_LUT}  ranks=[${RANKS}]"
echo "DEVICE:     ${DEVICE} dtype=${DTYPE}"
echo "PPL:        EnablePPL=${EnablePPL}  PPL_DEVICE=${PPL_DEVICE}"
echo "DO_G:       ${DO_G} (families=${SEL_FAMILIES})"
echo "OUTDIR:     ${OUTDIR}"
echo "============================================================"
echo ""

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

echo ">>> Step 1: Pass A (build BASE_A)"
python scripts/apply_awq_equiv_scales_y2.py \
  --model-id "${BASE}" \
  --imatrix "runs/imatrix_qwen3_0.6b_random.pt" \
  --alpha "${ALPHA_A}" \
  --output "${BASE_A}" \
  --dtype "${DTYPE}" --device "${DEVICE}"
echo ""

echo ">>> Step 2: Compute iMatrix_A on BASE_A"
python scripts/compute_imatrix.py \
  --model "${BASE_A}" \
  --calib-mode "${CALIB_MODE}" \
  --tokens "${TOKENS}" --seq-len "${SEQ_LEN}" --batch-size "${BATCH_SIZE}" \
  --out "${IM_A}" \
  --trust-remote-code --verbose
echo ""

echo ">>> Step 3: Compute Y-matrix variants on BASE_A"
python scripts/compute_ymatrix.py --model "${BASE_A}" --calib-mode "${CALIB_MODE}" --tokens "${TOKENS}" --seq-len "${SEQ_LEN}" --batch-size "${BATCH_SIZE}" --targets-regex ".*\.self_attn\.v_proj$,\s*.*\.mlp\.up_proj$" --out "${YM_BOTH}"  --trust-remote-code --verbose
python scripts/compute_ymatrix.py --model "${BASE_A}" --calib-mode "${CALIB_MODE}" --tokens "${TOKENS}" --seq-len "${SEQ_LEN}" --batch-size "${BATCH_SIZE}" --targets-regex ".*\.self_attn\.v_proj$"                    --out "${YM_VONLY}" --trust-remote-code --verbose
python scripts/compute_ymatrix.py --model "${BASE_A}" --calib-mode "${CALIB_MODE}" --tokens "${TOKENS}" --seq-len "${SEQ_LEN}" --batch-size "${BATCH_SIZE}" --targets-regex ".*\.mlp\.up_proj$"                        --out "${YM_UPONLY}" --trust-remote-code --verbose
echo ""

echo ">>> Step 4: Pass B (build AB variants)"
python scripts/apply_awq_equiv_scales_y2.py --model-id "${BASE_A}" --imatrix "${IM_A}" --alpha 0.0 --ymatrix "${YM_BOTH}"   --alpha-row "${ALPHA_ROW}" --row-min-scale "${ROW_MIN}" --row-max-scale "${ROW_MAX}" --output "${BASE_AB_BOTH}"   --dtype "${DTYPE}" --device "${DEVICE}"
python scripts/apply_awq_equiv_scales_y2.py --model-id "${BASE_A}" --imatrix "${IM_A}" --alpha 0.0 --ymatrix "${YM_VONLY}"  --alpha-row "${ALPHA_ROW}" --row-min-scale "${ROW_MIN}" --row-max-scale "${ROW_MAX}" --output "${BASE_AB_VONLY}"  --dtype "${DTYPE}" --device "${DEVICE}"
python scripts/apply_awq_equiv_scales_y2.py --model-id "${BASE_A}" --imatrix "${IM_A}" --alpha 0.0 --ymatrix "${YM_UPONLY}" --alpha-row "${ALPHA_ROW}" --row-min-scale "${ROW_MIN}" --row-max-scale "${ROW_MAX}" --output "${BASE_AB_UPONLY}" --dtype "${DTYPE}" --device "${DEVICE}"
echo ""

echo ">>> Step 5: Compute iMatrix_AB on AB variants"
python scripts/compute_imatrix.py --model "${BASE_AB_BOTH}"   --calib-mode "${CALIB_MODE}" --tokens "${TOKENS}" --seq-len "${SEQ_LEN}" --batch-size "${BATCH_SIZE}" --out "${IM_AB_BOTH}"   --trust-remote-code --verbose
python scripts/compute_imatrix.py --model "${BASE_AB_VONLY}"  --calib-mode "${CALIB_MODE}" --tokens "${TOKENS}" --seq-len "${SEQ_LEN}" --batch-size "${BATCH_SIZE}" --out "${IM_AB_VONLY}"  --trust-remote-code --verbose
python scripts/compute_imatrix.py --model "${BASE_AB_UPONLY}" --calib-mode "${CALIB_MODE}" --tokens "${TOKENS}" --seq-len "${SEQ_LEN}" --batch-size "${BATCH_SIZE}" --out "${IM_AB_UPONLY}" --trust-remote-code --verbose
echo ""

do_select_one () {
  local base_id="$1"
  local init_ckpt="$2"
  local im_id="$3"
  local out_hybrid="$4"
  python scripts/select_best_lut_per_layer.py "${init_ckpt}" \
    --model-id "${base_id}" \
    --output "${out_hybrid}" \
    --families "${SEL_FAMILIES}" \
    --metric iActMSE \
    --imatrix "${im_id}" \
    --workers "${WORKERS}" >/dev/null
}

do_init_one () {
  local label="$1"
  local base_id="$2"
  local im_id="$3"
  local rank="$4"

  local cfg="${V2_CONFIG_BASE}${rank}"
  local out="${OUTDIR}/v2_${label}_${RUN_TAG}_r${rank}_im$(basename "${im_id}" .pt)"
  local init_ckpt="${out}/v2_initial.pt"
  local hybrid_ckpt="${out}/ihybrid.pt"

  echo ">>> init_model_v2: ${label} rank=${rank} imatrix=$(basename "${im_id}")"
  if ! python scripts/init_model_v2.py \
      --model-id "${base_id}" \
      --output "${out}" \
      --config "${cfg}" \
      --lut "${FIXED_LUT}" \
      --imatrix "${im_id}" \
      --svd-error >/dev/null ; then
    echo "    [SKIP] init failed (missing preset ${cfg}?)"
    echo "${label},${rank},$(basename "${im_id}"),N/A,N/A,${init_ckpt},${hybrid_ckpt}" >> "${RESULTS_CSV}"
    return 0
  fi

  local ppl_init
  ppl_init=$(run_ppl "${init_ckpt}")
  echo "    PPL_init=${ppl_init}"

  local ppl_hybrid="N/A"
  if [[ "${DO_G}" == "1" ]]; then
    echo "    selecting LUTs via k-means (Family G) using imatrix=$(basename "${im_id}")"
    do_select_one "${base_id}" "${init_ckpt}" "${im_id}" "${hybrid_ckpt}"
    ppl_hybrid=$(run_ppl "${hybrid_ckpt}")
    echo "    PPL_hybrid=${ppl_hybrid}"
  fi

  echo "${label},${rank},$(basename "${im_id}"),${ppl_init},${ppl_hybrid},${init_ckpt},${hybrid_ckpt}" >> "${RESULTS_CSV}"
  echo ""
}

echo ">>> Step 6: V2 init + optional k-means selection"
for r in ${RANKS}; do
  do_init_one "A"                "${BASE_A}"        "${IM_A}"        "${r}"

  do_init_one "AB_both_stale"    "${BASE_AB_BOTH}"  "${IM_A}"        "${r}"
  do_init_one "AB_both_correct"  "${BASE_AB_BOTH}"  "${IM_AB_BOTH}"  "${r}"

  do_init_one "AB_vonly_stale"   "${BASE_AB_VONLY}" "${IM_A}"        "${r}"
  do_init_one "AB_vonly_correct" "${BASE_AB_VONLY}" "${IM_AB_VONLY}" "${r}"

  do_init_one "AB_uponly_stale"   "${BASE_AB_UPONLY}" "${IM_A}"          "${r}"
  do_init_one "AB_uponly_correct" "${BASE_AB_UPONLY}" "${IM_AB_UPONLY}"  "${r}"
done

echo "============================================================"
echo "DONE. Results:"
echo "  ${RESULTS_CSV}"
echo ""
column -s, -t "${RESULTS_CSV}" | sed -e 's/  \+/ /g' || true
