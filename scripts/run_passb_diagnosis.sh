#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Pass-B Diagnosis Pipeline
# =============================================================================
# Systematically identify where Pass-B regression enters:
# - V2 SVD scale init?
# - LUT selection / iMatrix mismatch?
# - Specific edge (v→o vs up→down)?
#
# Usage:
#   ./scripts/run_passb_diagnosis.sh
#   ALPHA_A=0.4 ALPHA_ROW=0.2 ./scripts/run_passb_diagnosis.sh
#   SKIP_PHASE1=true ./scripts/run_passb_diagnosis.sh  # skip slow FP4-only tests
#
# =============================================================================

# -------------------------
# Configuration
# -------------------------
BASE=${BASE:-Qwen/Qwen3-0.6B}
ALPHA_A=${ALPHA_A:-0.4}
ALPHA_ROW=${ALPHA_ROW:-0.2}
ROW_MIN=${ROW_MIN:-0.5}
ROW_MAX=${ROW_MAX:-2.0}

# Calibration
TOKENS=${TOKENS:-50000}
SEQ_LEN=${SEQ_LEN:-512}
CALIB_MODE=${CALIB_MODE:-random_ids}

# V2 config
V2_CONFIG=${V2_CONFIG:-q4a4_r32}
GROUP_SIZE=${GROUP_SIZE:-32}

# Rank sweep values
RANKS=${RANKS:-"16 32 64"}

# Device settings
DEVICE=${DEVICE:-cpu}
DTYPE=${DTYPE:-float32}

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

# PPL chunks (20 = fast ~30s, true = full ~3min)
PPL_CHUNKS=${PPL_CHUNKS:-20}

# Phase skips
SKIP_PHASE1=${SKIP_PHASE1:-true}   # FP4-only tests (slow, V2-independent)
SKIP_PHASE3=${SKIP_PHASE3:-false}  # Rank sweep
SKIP_PHASE4=${SKIP_PHASE4:-false}  # Edge isolation

# Output directories
DIAG_DIR="runs/passb_diagnosis"
mkdir -p "$DIAG_DIR"

# Results file
RESULTS_FILE="$DIAG_DIR/results.txt"
echo "Pass-B Diagnosis Results" > "$RESULTS_FILE"
echo "========================" >> "$RESULTS_FILE"
echo "Date: $(date)" >> "$RESULTS_FILE"
echo "BASE: $BASE" >> "$RESULTS_FILE"
echo "ALPHA_A: $ALPHA_A, ALPHA_ROW: $ALPHA_ROW" >> "$RESULTS_FILE"
echo "PPL_DEVICE: $PPL_DEVICE, PPL_CHUNKS: $PPL_CHUNKS" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

# -------------------------
# Helper functions
# -------------------------
log() {
  echo "[$(date +%H:%M:%S)] $*"
  echo "[$(date +%H:%M:%S)] $*" >> "$RESULTS_FILE"
}

measure_ppl() {
  local ckpt="$1"
  local label="$2"

  if [[ ! -f "$ckpt" ]]; then
    echo "N/A (missing)"
    return
  fi

  local cmd="python scripts/measure_perplexity.py $ckpt --device $PPL_DEVICE --dtype fp16 --output-ppl"
  if [[ "$PPL_CHUNKS" =~ ^[0-9]+$ ]]; then
    cmd="$cmd --max-chunks $PPL_CHUNKS"
  fi

  local output
  output=$($cmd 2>&1) || true
  local ppl
  ppl=$(echo "$output" | grep "^PPL=" | cut -d= -f2)

  if [[ -z "$ppl" ]]; then
    echo "ERROR"
  else
    echo "$ppl"
  fi
}

record_result() {
  local phase="$1"
  local variant="$2"
  local ppl="$3"
  local notes="${4:-}"

  printf "%-12s | %-25s | %-12s | %s\n" "$phase" "$variant" "$ppl" "$notes" >> "$RESULTS_FILE"
}

# -------------------------
# Print config
# -------------------------
echo "============================================================"
echo "PASS-B DIAGNOSIS PIPELINE"
echo "============================================================"
echo "BASE:        $BASE"
echo "ALPHA_A:     $ALPHA_A"
echo "ALPHA_ROW:   $ALPHA_ROW"
echo "V2_CONFIG:   $V2_CONFIG"
echo "PPL_DEVICE:  $PPL_DEVICE"
echo "PPL_CHUNKS:  $PPL_CHUNKS"
echo "DIAG_DIR:    $DIAG_DIR"
echo ""
echo "Phases:"
echo "  1 (FP4-only):     ${SKIP_PHASE1} (skip=$SKIP_PHASE1)"
echo "  2 (Fixed LUT):    enabled"
echo "  3 (Rank sweep):   ${SKIP_PHASE3} (skip=$SKIP_PHASE3)"
echo "  4 (Edge isolate): ${SKIP_PHASE4} (skip=$SKIP_PHASE4)"
echo "============================================================"
echo ""

# -------------------------
# Phase 0: Build base models & iMatrices
# -------------------------
log "PHASE 0: Building base models and iMatrices"

BASE_A="$DIAG_DIR/base_a"
BASE_AB="$DIAG_DIR/base_ab"
IM_A="$DIAG_DIR/imatrix_a.pt"
IM_AB="$DIAG_DIR/imatrix_ab.pt"

# Step 0a: iMatrix on original base (for A)
if [[ ! -f "$IM_A" ]]; then
  log "  Computing iMatrix on original base..."
  python scripts/compute_imatrix.py \
    --model "$BASE" \
    --calib-mode "$CALIB_MODE" \
    --tokens "$TOKENS" \
    --seq-len "$SEQ_LEN" \
    --out "$IM_A" \
    --trust-remote-code --verbose
fi

# Step 0b: Apply Pass-A only
if [[ ! -d "$BASE_A" ]]; then
  log "  Applying Pass-A (alpha=$ALPHA_A)..."
  python scripts/apply_awq_equiv_scales_y2.py \
    --model-id "$BASE" \
    --imatrix "$IM_A" \
    --alpha "$ALPHA_A" \
    --output "$BASE_A" \
    --dtype "$DTYPE" \
    --device "$DEVICE"
fi

# Step 0c: Apply Pass-B (A → AB)
if [[ ! -d "$BASE_AB" ]]; then
  log "  Computing Y-matrix on Pass-A base..."
  YM_AB="$DIAG_DIR/ymatrix_ab.pt"
  python scripts/compute_ymatrix.py \
    --model "$BASE_A" \
    --calib-mode "$CALIB_MODE" \
    --tokens "$TOKENS" \
    --seq-len "$SEQ_LEN" \
    --out "$YM_AB" \
    --trust-remote-code --verbose

  log "  Applying Pass-B (alpha-row=$ALPHA_ROW)..."
  python scripts/apply_awq_equiv_scales_y2.py \
    --model-id "$BASE_A" \
    --imatrix "$IM_A" \
    --alpha 0.0 \
    --ymatrix "$YM_AB" \
    --alpha-row "$ALPHA_ROW" \
    --row-min-scale "$ROW_MIN" \
    --row-max-scale "$ROW_MAX" \
    --output "$BASE_AB" \
    --dtype "$DTYPE" \
    --device "$DEVICE"
fi

# Step 0d: iMatrix on Pass-AB base (CRITICAL for Phase 2B)
if [[ ! -f "$IM_AB" ]]; then
  log "  Computing iMatrix on Pass-AB base (for correct scoring)..."
  python scripts/compute_imatrix.py \
    --model "$BASE_AB" \
    --calib-mode "$CALIB_MODE" \
    --tokens "$TOKENS" \
    --seq-len "$SEQ_LEN" \
    --out "$IM_AB" \
    --trust-remote-code --verbose
fi

# Log checksums for reproducibility
log "  Checksums:"
CKSUM_A=""
if [[ -f "$BASE_A/model.safetensors" ]]; then
  CKSUM_A=$(sha256sum "$BASE_A/model.safetensors" | cut -c1-16)
elif [[ -f "$BASE_A/pytorch_model.bin" ]]; then
  CKSUM_A=$(sha256sum "$BASE_A/pytorch_model.bin" | cut -c1-16)
fi
[[ -n "$CKSUM_A" ]] && log "    Base A:  ${CKSUM_A}..."

CKSUM_AB=""
if [[ -f "$BASE_AB/model.safetensors" ]]; then
  CKSUM_AB=$(sha256sum "$BASE_AB/model.safetensors" | cut -c1-16)
elif [[ -f "$BASE_AB/pytorch_model.bin" ]]; then
  CKSUM_AB=$(sha256sum "$BASE_AB/pytorch_model.bin" | cut -c1-16)
fi
[[ -n "$CKSUM_AB" ]] && log "    Base AB: ${CKSUM_AB}..."

echo "" >> "$RESULTS_FILE"
echo "Phase     | Variant                   | PPL          | Notes" >> "$RESULTS_FILE"
echo "----------|---------------------------|--------------|------" >> "$RESULTS_FILE"

# -------------------------
# Phase 1: FP4-only full-model PPL (optional, slow)
# -------------------------
if [[ "$SKIP_PHASE1" != "true" ]]; then
  log ""
  log "PHASE 1: FP4-only quantization (no V2)"
  log "  (This tests if Pass-B is generally quant-friendly)"

  # TODO: Implement simple FP4-only quantization script
  # For now, skip with a placeholder
  log "  [Not implemented - requires fp4_only_quant.py]"
  record_result "Phase1" "FP4-only (A)" "N/A" "not implemented"
  record_result "Phase1" "FP4-only (AB)" "N/A" "not implemented"
else
  log ""
  log "PHASE 1: Skipped (SKIP_PHASE1=true)"
fi

# -------------------------
# Phase 2A: Fixed LUT init (no search)
# -------------------------
log ""
log "PHASE 2A: Fixed LUT init (no search, no selection)"

V2_A_FIXED="$DIAG_DIR/v2_a_fixed"
V2_AB_FIXED="$DIAG_DIR/v2_ab_fixed"
V2_AB_FIXED_IM_AB="$DIAG_DIR/v2_ab_fixed_im_ab"

# Init A with fixed LUT
if [[ ! -f "$V2_A_FIXED/v2_initial.pt" ]]; then
  log "  init_model_v2 on Base A (fixed LUT, iMatrix A)..."
  python scripts/init_model_v2.py \
    --model-id "$BASE_A" \
    --output "$V2_A_FIXED" \
    --config "$V2_CONFIG" \
    --lut fp4_dense \
    --imatrix "$IM_A" \
    --svd-error
fi

# Init AB with fixed LUT (using iMatrix A - stale)
if [[ ! -f "$V2_AB_FIXED/v2_initial.pt" ]]; then
  log "  init_model_v2 on Base AB (fixed LUT, iMatrix A - STALE)..."
  python scripts/init_model_v2.py \
    --model-id "$BASE_AB" \
    --output "$V2_AB_FIXED" \
    --config "$V2_CONFIG" \
    --lut fp4_dense \
    --imatrix "$IM_A" \
    --svd-error
fi

# Init AB with fixed LUT (using iMatrix AB - correct)
if [[ ! -f "$V2_AB_FIXED_IM_AB/v2_initial.pt" ]]; then
  log "  init_model_v2 on Base AB (fixed LUT, iMatrix AB - CORRECT)..."
  python scripts/init_model_v2.py \
    --model-id "$BASE_AB" \
    --output "$V2_AB_FIXED_IM_AB" \
    --config "$V2_CONFIG" \
    --lut fp4_dense \
    --imatrix "$IM_AB" \
    --svd-error
fi

# Measure PPL
log "  Measuring PPL..."
PPL_A_FIXED=$(measure_ppl "$V2_A_FIXED/v2_initial.pt" "A fixed")
PPL_AB_FIXED=$(measure_ppl "$V2_AB_FIXED/v2_initial.pt" "AB fixed (stale IM)")
PPL_AB_FIXED_IM_AB=$(measure_ppl "$V2_AB_FIXED_IM_AB/v2_initial.pt" "AB fixed (correct IM)")

log "  Results:"
log "    A (fixed LUT, IM_A):           PPL=$PPL_A_FIXED"
log "    AB (fixed LUT, IM_A stale):    PPL=$PPL_AB_FIXED"
log "    AB (fixed LUT, IM_AB correct): PPL=$PPL_AB_FIXED_IM_AB"

record_result "Phase2A" "A (fixed, IM_A)" "$PPL_A_FIXED" "baseline"
record_result "Phase2A" "AB (fixed, IM_A stale)" "$PPL_AB_FIXED" "stale iMatrix"
record_result "Phase2A" "AB (fixed, IM_AB)" "$PPL_AB_FIXED_IM_AB" "correct iMatrix"

# -------------------------
# Phase 2B: Analysis
# -------------------------
log ""
log "PHASE 2B: Interpretation"

# Compare PPLs
if [[ "$PPL_A_FIXED" != "ERROR" && "$PPL_AB_FIXED_IM_AB" != "ERROR" ]]; then
  DELTA_STALE=$(echo "$PPL_AB_FIXED - $PPL_A_FIXED" | bc -l 2>/dev/null || echo "N/A")
  DELTA_CORRECT=$(echo "$PPL_AB_FIXED_IM_AB - $PPL_A_FIXED" | bc -l 2>/dev/null || echo "N/A")

  log "  Delta (AB stale vs A):   $DELTA_STALE"
  log "  Delta (AB correct vs A): $DELTA_CORRECT"

  if [[ "$DELTA_CORRECT" != "N/A" ]]; then
    # Check if Pass-B helps or hurts with correct iMatrix
    IS_NEGATIVE=$(echo "$DELTA_CORRECT < 0" | bc -l 2>/dev/null || echo "0")
    if [[ "$IS_NEGATIVE" == "1" ]]; then
      log "  → Pass-B HELPS with correct iMatrix (PPL improved by ${DELTA_CORRECT#-})"
    else
      log "  → Pass-B HURTS even with correct iMatrix (PPL degraded by $DELTA_CORRECT)"
      log "  → Problem is likely SVD scale init or rank bottleneck"
    fi
  fi
fi

# -------------------------
# Phase 3: Rank sweep (if AB still regresses)
# -------------------------
if [[ "$SKIP_PHASE3" != "true" ]]; then
  log ""
  log "PHASE 3: Rank sweep on Base AB"

  for RANK in $RANKS; do
    V2_AB_RANK="$DIAG_DIR/v2_ab_r${RANK}"

    if [[ ! -f "$V2_AB_RANK/v2_initial.pt" ]]; then
      log "  init_model_v2 on Base AB (rank=$RANK)..."
      python scripts/init_model_v2.py \
        --model-id "$BASE_AB" \
        --output "$V2_AB_RANK" \
        --config q4a4 \
        --mlp-rank "$RANK" \
        --attn-rank "$RANK" \
        --lut fp4_dense \
        --imatrix "$IM_AB" \
        --svd-error
    fi

    PPL_RANK=$(measure_ppl "$V2_AB_RANK/v2_initial.pt" "AB r$RANK")
    log "    rank=$RANK: PPL=$PPL_RANK"
    record_result "Phase3" "AB (r$RANK, IM_AB)" "$PPL_RANK" "rank sweep"
  done
else
  log ""
  log "PHASE 3: Skipped (SKIP_PHASE3=true)"
fi

# -------------------------
# Phase 4: Edge isolation (v→o only, up→down only)
# -------------------------
if [[ "$SKIP_PHASE4" != "true" ]]; then
  log ""
  log "PHASE 4: Edge isolation (partial Pass-B)"

  # This requires modifying compute_ymatrix.py to accept --targets filter
  # For now, we'll test if the script supports it

  # 4A: v→o only
  BASE_AB_VO="$DIAG_DIR/base_ab_vo_only"
  YM_VO="$DIAG_DIR/ymatrix_vo.pt"

  if [[ ! -d "$BASE_AB_VO" ]]; then
    log "  Computing Y-matrix for v_proj only..."

    # compute_ymatrix.py supports --targets-regex
    python scripts/compute_ymatrix.py \
      --model "$BASE_A" \
      --calib-mode "$CALIB_MODE" \
      --tokens "$TOKENS" \
      --seq-len "$SEQ_LEN" \
      --out "$YM_VO" \
      --targets-regex "v_proj" \
      --trust-remote-code --verbose

    log "  Applying Pass-B (v→o only)..."
    python scripts/apply_awq_equiv_scales_y2.py \
      --model-id "$BASE_A" \
      --imatrix "$IM_A" \
      --alpha 0.0 \
      --ymatrix "$YM_VO" \
      --alpha-row "$ALPHA_ROW" \
      --row-min-scale "$ROW_MIN" \
      --row-max-scale "$ROW_MAX" \
      --output "$BASE_AB_VO" \
      --dtype "$DTYPE" \
      --device "$DEVICE"
  fi

  # V2 init on v→o only base
  V2_AB_VO="$DIAG_DIR/v2_ab_vo_only"
  IM_AB_VO="$DIAG_DIR/imatrix_ab_vo.pt"

  if [[ -d "$BASE_AB_VO" ]]; then
    # Compute iMatrix for v→o base
    if [[ ! -f "$IM_AB_VO" ]]; then
      python scripts/compute_imatrix.py \
        --model "$BASE_AB_VO" \
        --calib-mode "$CALIB_MODE" \
        --tokens "$TOKENS" \
        --seq-len "$SEQ_LEN" \
        --out "$IM_AB_VO" \
        --trust-remote-code --verbose
    fi

    if [[ ! -f "$V2_AB_VO/v2_initial.pt" ]]; then
      log "  init_model_v2 on Base AB (v→o only)..."
      python scripts/init_model_v2.py \
        --model-id "$BASE_AB_VO" \
        --output "$V2_AB_VO" \
        --config "$V2_CONFIG" \
        --lut fp4_dense \
        --imatrix "$IM_AB_VO" \
        --svd-error
    fi

    PPL_VO=$(measure_ppl "$V2_AB_VO/v2_initial.pt" "AB v→o only")
    log "    v→o only: PPL=$PPL_VO"
    record_result "Phase4" "AB (v→o only)" "$PPL_VO" "edge isolation"
  fi

  # 4B: up→down only
  BASE_AB_UD="$DIAG_DIR/base_ab_ud_only"
  YM_UD="$DIAG_DIR/ymatrix_ud.pt"

  if [[ ! -d "$BASE_AB_UD" ]]; then
    log "  Computing Y-matrix for up_proj only..."
    python scripts/compute_ymatrix.py \
      --model "$BASE_A" \
      --calib-mode "$CALIB_MODE" \
      --tokens "$TOKENS" \
      --seq-len "$SEQ_LEN" \
      --out "$YM_UD" \
      --targets-regex "up_proj" \
      --trust-remote-code --verbose

    log "  Applying Pass-B (up→down only)..."
    python scripts/apply_awq_equiv_scales_y2.py \
      --model-id "$BASE_A" \
      --imatrix "$IM_A" \
      --alpha 0.0 \
      --ymatrix "$YM_UD" \
      --alpha-row "$ALPHA_ROW" \
      --row-min-scale "$ROW_MIN" \
      --row-max-scale "$ROW_MAX" \
      --output "$BASE_AB_UD" \
      --dtype "$DTYPE" \
      --device "$DEVICE"
  fi

  # V2 init on up→down only base
  V2_AB_UD="$DIAG_DIR/v2_ab_ud_only"
  IM_AB_UD="$DIAG_DIR/imatrix_ab_ud.pt"

  if [[ -d "$BASE_AB_UD" ]]; then
    # Compute iMatrix for up→down base
    if [[ ! -f "$IM_AB_UD" ]]; then
      python scripts/compute_imatrix.py \
        --model "$BASE_AB_UD" \
        --calib-mode "$CALIB_MODE" \
        --tokens "$TOKENS" \
        --seq-len "$SEQ_LEN" \
        --out "$IM_AB_UD" \
        --trust-remote-code --verbose
    fi

    if [[ ! -f "$V2_AB_UD/v2_initial.pt" ]]; then
      log "  init_model_v2 on Base AB (up→down only)..."
      python scripts/init_model_v2.py \
        --model-id "$BASE_AB_UD" \
        --output "$V2_AB_UD" \
        --config "$V2_CONFIG" \
        --lut fp4_dense \
        --imatrix "$IM_AB_UD" \
        --svd-error
    fi

    PPL_UD=$(measure_ppl "$V2_AB_UD/v2_initial.pt" "AB up→down only")
    log "    up→down only: PPL=$PPL_UD"
    record_result "Phase4" "AB (up→down only)" "$PPL_UD" "edge isolation"
  fi

else
  log ""
  log "PHASE 4: Skipped (SKIP_PHASE4=true)"
fi

# -------------------------
# Summary
# -------------------------
log ""
log "============================================================"
log "DIAGNOSIS COMPLETE"
log "============================================================"
log "Results saved to: $RESULTS_FILE"
log ""

cat "$RESULTS_FILE"

echo ""
echo "Interpretation guide:"
echo "  - If AB(IM_AB) ≈ A: Pass-B is fine, regression was stale iMatrix"
echo "  - If AB(IM_AB) > A even with correct IM: Check Phase 3 (rank)"
echo "  - If higher rank helps AB: Pass-B pushes complexity into S"
echo "  - If v→o or up→down alone regresses: That edge is the problem"
echo ""
