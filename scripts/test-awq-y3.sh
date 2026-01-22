#!/usr/bin/env bash
set -euo pipefail

# test-awq-y3.sh
# Diagnose Pass-B (Y-matrix row-smoothing) regressions by separating:
#   (1) stale iMatrix vs correct iMatrix
#   (2) V2/SVD/rank bottleneck vs LUT selection
#   (3) which Pass-B edge is harmful: v->o vs up->down vs both
#
# This script intentionally uses FIXED LUT init (e.g., fp4_dense) to minimize LUT-search noise.
# You can optionally run the full LUT-selection step afterwards.

# -------------------------
# Config knobs (override via env vars)
# -------------------------
BASE=${BASE:-Qwen/Qwen3-0.6B}

# Pass-A strength and Pass-B strength
ALPHA_A=${ALPHA_A:-0.45}
ALPHA_ROW=${ALPHA_ROW:-0.30}

# Clamp for Pass-B row scales (tighter is safer)
ROW_MIN=${ROW_MIN:-0.5}
ROW_MAX=${ROW_MAX:-2.0}

# Calibration for matrices
TOKENS=${TOKENS:-50000}
SEQ_LEN=${SEQ_LEN:-512}
BATCH_SIZE=${BATCH_SIZE:-1}
CALIB_MODE=${CALIB_MODE:-random_ids}   # random_ids|pseudo_text|textfile

# Use CPU for transforms & init to avoid backend nondeterminism
DEVICE=${DEVICE:-cpu}
DTYPE=${DTYPE:-float32}

# V2 init config preset (rank baked into preset name)
# For rank sweeps: set RANKS="16 32 64" and ensure matching presets exist (q4a4_r16 etc.)
V2_CONFIG_BASE=${V2_CONFIG_BASE:-q4a4_r}   # prefix, default q4a4_r
RANKS=${RANKS:-32}                         # space-separated list: "16 32 64"
GROUP_NOTE=${GROUP_NOTE:-""}

# Fixed LUT for init (to isolate SVD/scale effects)
FIXED_LUT=${FIXED_LUT:-fp4_dense}

# PPL measurement: true=full, false=skip, N=max-chunks (e.g., 20)
EnablePPL=${EnablePPL:-20}

# PPL device auto-detect (TPU > CUDA > MPS > CPU)
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

# Output dirs
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

RESULTS_CSV="${OUTDIR}/results.csv"
echo "variant,rank,imatrix_used,ppl_v2_init,checkpoint" > "${RESULTS_CSV}"

echo "============================================================"
echo "PASS-B DIAG (Y3)"
echo "============================================================"
echo "BASE:       ${BASE}"
echo "ALPHA_A:    ${ALPHA_A}"
echo "ALPHA_ROW:  ${ALPHA_ROW}"
echo "ROW clamp:  [${ROW_MIN}, ${ROW_MAX}]"
echo "CALIB:      ${CALIB_MODE} tokens=${TOKENS} seq=${SEQ_LEN} bs=${BATCH_SIZE}"
echo "INIT:       fixed LUT=${FIXED_LUT}  ranks=[${RANKS}] ${GROUP_NOTE}"
echo "DEVICE:     ${DEVICE} dtype=${DTYPE}"
echo "PPL:        EnablePPL=${EnablePPL}  PPL_DEVICE=${PPL_DEVICE}"
echo "OUTDIR:     ${OUTDIR}"
echo "============================================================"
echo ""

run_ppl () {
  local ckpt="$1"
  if [[ "${EnablePPL}" == "false" ]]; then
    echo "N/A"
    return 0
  elif [[ "${EnablePPL}" == "true" ]]; then
    python scripts/measure_perplexity.py "${ckpt}" --device "${PPL_DEVICE}" --dtype fp16 --output-ppl 2>/dev/null \
      | awk -F= '/^PPL=/{print $2; exit}'
  elif [[ "${EnablePPL}" =~ ^[0-9]+$ ]]; then
    python scripts/measure_perplexity.py "${ckpt}" --device "${PPL_DEVICE}" --dtype fp16 --output-ppl --max-chunks "${EnablePPL}" 2>/dev/null \
      | awk -F= '/^PPL=/{print $2; exit}'
  else
    echo "N/A"
    return 0
  fi
}

# -------------------------
# Step 1: Build Base A (Pass A)
# -------------------------
echo ">>> Step 1: Pass A (build BASE_A)"
python scripts/apply_awq_equiv_scales_y2.py \
  --model-id "${BASE}" \
  --imatrix "runs/imatrix_qwen3_0.6b_random.pt" \
  --alpha "${ALPHA_A}" \
  --output "${BASE_A}" \
  --dtype "${DTYPE}" --device "${DEVICE}"
echo ""

# -------------------------
# Step 2: Compute iMatrix_A on BASE_A (used as 'stale' reference downstream)
# -------------------------
echo ">>> Step 2: Compute iMatrix_A on BASE_A"
python scripts/compute_imatrix.py \
  --model "${BASE_A}" \
  --calib-mode "${CALIB_MODE}" \
  --tokens "${TOKENS}" --seq-len "${SEQ_LEN}" --batch-size "${BATCH_SIZE}" \
  --out "${IM_A}" \
  --trust-remote-code --verbose
echo ""

# -------------------------
# Step 3: Compute Y-matrix variants on BASE_A
# -------------------------
echo ">>> Step 3: Compute Y-matrix variants on BASE_A"
python scripts/compute_ymatrix.py \
  --model "${BASE_A}" \
  --calib-mode "${CALIB_MODE}" \
  --tokens "${TOKENS}" --seq-len "${SEQ_LEN}" --batch-size "${BATCH_SIZE}" \
  --targets-regex ".*\.self_attn\.v_proj$,\s*.*\.mlp\.up_proj$" \
  --out "${YM_BOTH}" \
  --trust-remote-code --verbose

python scripts/compute_ymatrix.py \
  --model "${BASE_A}" \
  --calib-mode "${CALIB_MODE}" \
  --tokens "${TOKENS}" --seq-len "${SEQ_LEN}" --batch-size "${BATCH_SIZE}" \
  --targets-regex ".*\.self_attn\.v_proj$" \
  --out "${YM_VONLY}" \
  --trust-remote-code --verbose

python scripts/compute_ymatrix.py \
  --model "${BASE_A}" \
  --calib-mode "${CALIB_MODE}" \
  --tokens "${TOKENS}" --seq-len "${SEQ_LEN}" --batch-size "${BATCH_SIZE}" \
  --targets-regex ".*\.mlp\.up_proj$" \
  --out "${YM_UPONLY}" \
  --trust-remote-code --verbose
echo ""

# -------------------------
# Step 4: Apply Pass B to produce AB variants
# -------------------------
echo ">>> Step 4: Pass B (build AB variants)"
python scripts/apply_awq_equiv_scales_y2.py \
  --model-id "${BASE_A}" \
  --imatrix "${IM_A}" \
  --alpha 0.0 \
  --ymatrix "${YM_BOTH}" \
  --alpha-row "${ALPHA_ROW}" \
  --row-min-scale "${ROW_MIN}" --row-max-scale "${ROW_MAX}" \
  --output "${BASE_AB_BOTH}" \
  --dtype "${DTYPE}" --device "${DEVICE}"

python scripts/apply_awq_equiv_scales_y2.py \
  --model-id "${BASE_A}" \
  --imatrix "${IM_A}" \
  --alpha 0.0 \
  --ymatrix "${YM_VONLY}" \
  --alpha-row "${ALPHA_ROW}" \
  --row-min-scale "${ROW_MIN}" --row-max-scale "${ROW_MAX}" \
  --output "${BASE_AB_VONLY}" \
  --dtype "${DTYPE}" --device "${DEVICE}"

python scripts/apply_awq_equiv_scales_y2.py \
  --model-id "${BASE_A}" \
  --imatrix "${IM_A}" \
  --alpha 0.0 \
  --ymatrix "${YM_UPONLY}" \
  --alpha-row "${ALPHA_ROW}" \
  --row-min-scale "${ROW_MIN}" --row-max-scale "${ROW_MAX}" \
  --output "${BASE_AB_UPONLY}" \
  --dtype "${DTYPE}" --device "${DEVICE}"
echo ""

# -------------------------
# Step 5: Compute iMatrix on each AB variant (IM_AB_*)
# -------------------------
echo ">>> Step 5: Compute iMatrix_AB on AB variants"
python scripts/compute_imatrix.py --model "${BASE_AB_BOTH}"   --calib-mode "${CALIB_MODE}" --tokens "${TOKENS}" --seq-len "${SEQ_LEN}" --batch-size "${BATCH_SIZE}" --out "${IM_AB_BOTH}"   --trust-remote-code --verbose
python scripts/compute_imatrix.py --model "${BASE_AB_VONLY}"  --calib-mode "${CALIB_MODE}" --tokens "${TOKENS}" --seq-len "${SEQ_LEN}" --batch-size "${BATCH_SIZE}" --out "${IM_AB_VONLY}"  --trust-remote-code --verbose
python scripts/compute_imatrix.py --model "${BASE_AB_UPONLY}" --calib-mode "${CALIB_MODE}" --tokens "${TOKENS}" --seq-len "${SEQ_LEN}" --batch-size "${BATCH_SIZE}" --out "${IM_AB_UPONLY}" --trust-remote-code --verbose
echo ""

# -------------------------
# Step 6: V2 init (fixed LUT) + PPL for each variant and rank
# We test each AB with both IM_A (stale) and IM_AB (correct) to validate the hypothesis.
# -------------------------
do_init_one () {
  local variant="$1"   # A | AB_*_*
  local base_id="$2"
  local im_id="$3"
  local rank="$4"

  local cfg="${V2_CONFIG_BASE}${rank}"
  local out="${OUTDIR}/v2_${variant}_${RUN_TAG}_r${rank}_im$(basename "${im_id}" .pt)"

  echo ">>> init_model_v2: variant=${variant} rank=${rank} imatrix=$(basename "${im_id}")"
  echo "    base=${base_id}"
  echo "    out=${out}"
  if ! python scripts/init_model_v2.py \
      --model-id "${base_id}" \
      --output "${out}" \
      --config "${cfg}" \
      --lut "${FIXED_LUT}" \
      --imatrix "${im_id}" \
      --svd-error ; then
    echo "    [SKIP] init failed (missing preset ${cfg}?)"
    echo "${variant},${rank},$(basename "${im_id}"),N/A,${out}/v2_initial.pt" >> "${RESULTS_CSV}"
    return 0
  fi

  local ppl
  ppl=$(run_ppl "${out}/v2_initial.pt")
  echo "    PPL=${ppl}"
  echo "${variant},${rank},$(basename "${im_id}"),${ppl},${out}/v2_initial.pt" >> "${RESULTS_CSV}"
  echo ""
}

echo ">>> Step 6: V2 init + PPL (fixed LUT) across variants/ranks"
for r in ${RANKS}; do
  do_init_one "A" "${BASE_A}" "${IM_A}" "${r}"

  do_init_one "AB_both_stale"   "${BASE_AB_BOTH}"   "${IM_A}"        "${r}"
  do_init_one "AB_both_correct" "${BASE_AB_BOTH}"   "${IM_AB_BOTH}"  "${r}"

  do_init_one "AB_vonly_stale"   "${BASE_AB_VONLY}"  "${IM_A}"        "${r}"
  do_init_one "AB_vonly_correct" "${BASE_AB_VONLY}"  "${IM_AB_VONLY}" "${r}"

  do_init_one "AB_uponly_stale"   "${BASE_AB_UPONLY}"  "${IM_A}"         "${r}"
  do_init_one "AB_uponly_correct" "${BASE_AB_UPONLY}"  "${IM_AB_UPONLY}" "${r}"
done

echo "============================================================"
echo "DONE. Results CSV:"
echo "  ${RESULTS_CSV}"
echo ""
echo "Quick view:"
column -s, -t "${RESULTS_CSV}" | sed -e 's/  \+/ /g' || true
