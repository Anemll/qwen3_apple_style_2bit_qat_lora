#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# test-awq-y2.sh
# End-to-end AWQ Pass-A + optional Pass-B (Y-matrix) + V2 init + k-means LUT selection.
#
# Key guarantees:
#   - Pass-B defaults to UP-ONLY (mlp.up_proj) because v->o Pass-B was harmful in diagnosis.
#   - IM_FINAL is ALWAYS computed on the FINAL base (BASE_AB) and used for:
#       * init_model_v2 LUT search/scoring
#       * select_best_lut_per_layer (Family-G k-means LUT training + iActMSE)
#   - Script is self-identifying (version + git commit + script sha256).
# ------------------------------------------------------------

SCRIPT_VERSION="test-awq-y2_v2026-01-22_01"

# --version / -V support
if [[ "${1:-}" == "--version" || "${1:-}" == "-V" ]]; then
  echo "test-awq-y2.sh version: $SCRIPT_VERSION"
  if command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "git commit: $(git rev-parse --short HEAD) ($(git branch --show-current 2>/dev/null || true))"
    if [[ -n "$(git status --porcelain 2>/dev/null || true)" ]]; then
      echo "git status: DIRTY"
    else
      echo "git status: clean"
    fi
  fi
  SCRIPT_PATH="${BASH_SOURCE[0]}"
  if command -v shasum >/dev/null 2>&1; then
    echo "script sha256: $(shasum -a 256 "$SCRIPT_PATH" | awk '{print $1}')"
  elif command -v sha256sum >/dev/null 2>&1; then
    echo "script sha256: $(sha256sum "$SCRIPT_PATH" | awk '{print $1}')"
  else
    python - "$SCRIPT_PATH" <<'PY'
import hashlib,sys
p=sys.argv[1]
print("script sha256:", hashlib.sha256(open(p,'rb').read()).hexdigest())
PY
  fi
  exit 0
fi

# -------------------------
# Config knobs (override via env vars)
# -------------------------
BASE=${BASE:-Qwen/Qwen3-0.6B}

# Matrices
# IM: used to DRIVE Pass-A scaling (can be on BASE)
IM=${IM:-runs/imatrix_qwen3_0.6b_random.pt}

# Y-matrix file + targets (default UP-ONLY)
YM=${YM:-runs/ymatrix_up_only.pt}
YM_TARGETS=${YM_TARGETS:-".*\.mlp\.up_proj$"}

# Pass strengths
ALPHA_A=${ALPHA_A:-0.45}
ALPHA_ROW=${ALPHA_ROW:-0.0}  # 0 disables Pass-B

# Clamp for Pass-B row scales
ROW_MIN=${ROW_MIN:-0.85}
ROW_MAX=${ROW_MAX:-1.18}

# Calibration for matrices
TOKENS=${TOKENS:-50000}
SEQ_LEN=${SEQ_LEN:-512}
BATCH_SIZE=${BATCH_SIZE:-1}
CALIB_MODE=${CALIB_MODE:-random_ids}   # random_ids|pseudo_text|textfile

# Build transforms/init on CPU by default for determinism
DEVICE=${DEVICE:-cpu}
DTYPE=${DTYPE:-float32}  # float32|float16|bfloat16

# PPL device auto-detect: TPU > CUDA > MPS > CPU
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

# LUT selection (k-means Family-G is included via select_best_lut_per_layer; keep E baseline)
FAMILIES=${FAMILIES:-E,G}
WORKERS=${WORKERS:-8}

# Outputs
BASE_A="runs/awq_scaled_a${ALPHA_A}"
BASE_AB="runs/awq_scaled_a${ALPHA_A}_y${ALPHA_ROW}"
OUT_INIT="runs/v2_awq_a${ALPHA_A}_y${ALPHA_ROW}"
HYBRID="${OUT_INIT}/ihybrid.pt"

# IM_FINAL: computed on FINAL base (BASE_AB) and used for LUT training/scoring
IM_FINAL=${IM_FINAL:-runs/imatrix_final_a${ALPHA_A}_y${ALPHA_ROW}.pt}

echo "============================================================"
echo "AWQ + Y-matrix pipeline"
echo "============================================================"
echo "SCRIPT_VERSION: $SCRIPT_VERSION"
if command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "GIT:        $(git rev-parse --short HEAD) ($(git branch --show-current 2>/dev/null || true))"
fi
echo "BASE:       $BASE"
echo "IM (PassA): $IM"
echo "YM:         $YM"
echo "YM_TARGETS: $YM_TARGETS"
echo "IM_FINAL:   $IM_FINAL"
echo "ALPHA_A:    $ALPHA_A"
echo "ALPHA_ROW:  $ALPHA_ROW"
echo "ROW clamp:  [$ROW_MIN, $ROW_MAX]"
echo "DEVICE:     $DEVICE"
echo "PPL_DEVICE: $PPL_DEVICE"
echo "DTYPE:      $DTYPE"
echo "V2_CONFIG:  $V2_CONFIG"
echo "FAMILIES:   $FAMILIES"
echo "TOKENS:     $TOKENS (seq_len=$SEQ_LEN, batch=$BATCH_SIZE, mode=$CALIB_MODE)"
echo "EnablePPL:  $EnablePPL"
echo "OUT_INIT:   $OUT_INIT"
echo "============================================================"
echo ""

run_ppl () {
  local ckpt="$1"
  if [[ "$EnablePPL" == "false" ]]; then
    echo "N/A"; return 0
  elif [[ "$EnablePPL" == "true" ]]; then
    python scripts/measure_perplexity.py "$ckpt" --device "$PPL_DEVICE" --dtype fp16 --output-ppl 2>/dev/null \
      | awk -F= '/^PPL=/{print $2; exit}'
  elif [[ "$EnablePPL" =~ ^[0-9]+$ ]]; then
    python scripts/measure_perplexity.py "$ckpt" --device "$PPL_DEVICE" --dtype fp16 --output-ppl --max-chunks "$EnablePPL" 2>/dev/null \
      | awk -F= '/^PPL=/{print $2; exit}'
  else
    echo "N/A"; return 0
  fi
}

# -------------------------
# Step 0: Create IM (for Pass-A) if missing
# -------------------------
if [[ ! -f "$IM" ]]; then
  echo ">>> Step 0: IM not found; creating on BASE"
  cmd="python scripts/compute_imatrix.py --model $BASE --calib-mode $CALIB_MODE --tokens $TOKENS --seq-len $SEQ_LEN --batch-size $BATCH_SIZE --out $IM --trust-remote-code --verbose"
  echo "CMD: $cmd"
  $cmd
  echo ""
fi

# -------------------------
# Step 1A: Pass A
# -------------------------
echo ">>> Step 1A: Applying Pass A (alpha=$ALPHA_A)"
cmd="python scripts/apply_awq_equiv_scales_y2.py --model-id $BASE --imatrix $IM --alpha $ALPHA_A --output $BASE_A --dtype $DTYPE --device $DEVICE"
echo "CMD: $cmd"
$cmd
echo ""

# -------------------------
# Step 1B/1C: Pass B (optional, UP-ONLY by default)
# -------------------------
if [[ "$ALPHA_ROW" != "0" && "$ALPHA_ROW" != "0.0" ]]; then
  echo ">>> Step 1B: Computing Y-matrix on BASE_A"
  cmd="python scripts/compute_ymatrix.py --model $BASE_A --calib-mode $CALIB_MODE --tokens $TOKENS --seq-len $SEQ_LEN --batch-size $BATCH_SIZE --targets-regex $YM_TARGETS --out $YM --trust-remote-code --verbose"
  echo "CMD: $cmd"
  if [[ "${FORCE_REYM:-0}" == "1" || ! -f "$YM" ]]; then
    $cmd
  else
    echo ">>> Step 1B: Using existing Y-matrix: $YM"
  fi
  echo ""

  echo ">>> Step 1C: Applying Pass B (alpha-row=$ALPHA_ROW)"
  cmd="python scripts/apply_awq_equiv_scales_y2.py --model-id $BASE_A --imatrix $IM --alpha 0.0 --ymatrix $YM --alpha-row $ALPHA_ROW --row-min-scale $ROW_MIN --row-max-scale $ROW_MAX --output $BASE_AB --dtype $DTYPE --device $DEVICE"
  echo "CMD: $cmd"
  $cmd
  echo ""
else
  echo ">>> Pass B disabled (ALPHA_ROW=0). Using BASE_A as final base."
  BASE_AB="$BASE_A"
  echo ""
fi

# -------------------------
# Step 1D: Compute IM_FINAL on FINAL base (BASE_AB)
# -------------------------
echo ">>> Step 1D: Computing IM_FINAL on FINAL base: $BASE_AB"
cmd="python scripts/compute_imatrix.py --model $BASE_AB --calib-mode $CALIB_MODE --tokens $TOKENS --seq-len $SEQ_LEN --batch-size $BATCH_SIZE --out $IM_FINAL --trust-remote-code --verbose"
echo "CMD: $cmd"
if [[ "${FORCE_REIMFINAL:-0}" == "1" || ! -f "$IM_FINAL" ]]; then
  $cmd
else
  echo ">>> Step 1D: Using existing IM_FINAL: $IM_FINAL"
fi
echo ""

# -------------------------
# Step 2: Initialize V2 from FINAL base (uses IM_FINAL)
# -------------------------
echo ">>> Step 2: init_model_v2.py from base: $BASE_AB"
cmd="python scripts/init_model_v2.py --model-id $BASE_AB --output $OUT_INIT --config $V2_CONFIG --search-lut --imatrix $IM_FINAL --svd-error"
echo "CMD: $cmd"
$cmd
echo ""

# -------------------------
# Step 3: PPL on v2_initial.pt
# -------------------------
echo ">>> Step 3: Perplexity (v2_initial.pt)"
PPL_V2_INIT=$(run_ppl "$OUT_INIT/v2_initial.pt")
echo "    PPL (v2_init) = $PPL_V2_INIT"
echo ""

# -------------------------
# Step 4: Per-tensor LUT selection (k-means Family G) using IM_FINAL
# -------------------------
echo ">>> Step 4: select_best_lut_per_layer (families=$FAMILIES)"
cmd="python scripts/select_best_lut_per_layer.py $OUT_INIT/v2_initial.pt --model-id $BASE_AB --output $HYBRID --families $FAMILIES --metric iActMSE --imatrix $IM_FINAL --workers $WORKERS"
echo "CMD: $cmd"
$cmd
echo ""

# -------------------------
# Step 5: PPL on hybrid
# -------------------------
echo ">>> Step 5: Perplexity (hybrid)"
PPL_HYBRID=$(run_ppl "$HYBRID")
echo "    PPL (hybrid) = $PPL_HYBRID"
echo ""

echo "============================================================"
echo "SUMMARY"
echo "============================================================"
printf "%-20s | %-12s | %s\n" "Stage" "PPL" "Checkpoint"
echo "---------------------|--------------|------------------------------------"
printf "%-20s | %-12s | %s\n" "V2 Init" "$PPL_V2_INIT" "$OUT_INIT/v2_initial.pt"
printf "%-20s | %-12s | %s\n" "Hybrid" "$PPL_HYBRID" "$HYBRID"
echo "============================================================"
echo "Base A:  $BASE_A"
echo "Base AB: $BASE_AB"
echo "IM_FINAL:$IM_FINAL"
echo "============================================================"
