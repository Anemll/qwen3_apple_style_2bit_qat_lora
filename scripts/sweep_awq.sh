#!/bin/bash
# Sweep AWQ alpha values and collect perplexity results
#
# MANUAL TEST COMMANDS (for single alpha, copy/paste):
# ====================================================
#
# 1. Create iMatrix (if needed):
#    python scripts/compute_imatrix.py --model Qwen/Qwen3-0.6B --tokens 50000 --seq-len 512 --calib-mode random_ids --out runs/imatrix_qwen3_0.6b_random.pt
#
# 2. Apply AWQ scales (alpha=0.5 example):
#    python scripts/apply_awq_equiv_scales.py --model-id Qwen/Qwen3-0.6B --imatrix runs/imatrix_qwen3_0.6b_random.pt --alpha 0.5 --output runs/awq_scaled_a0.5 --dtype float32
#
# 3. Initialize V2 (fast: --lut fp4_dense, or slow: --search-lut):
#    python scripts/init_model_v2.py --model-id runs/awq_scaled_a0.5 -o runs/v2_awq_a0.5 -c q4a4_r32 --lut fp4_dense --imatrix runs/imatrix_qwen3_0.6b_random.pt
#    python scripts/init_model_v2.py --model-id runs/awq_scaled_a0.5 -o runs/v2_awq_a0.5 -c q4a4_r32 --search-lut --imatrix runs/imatrix_qwen3_0.6b_random.pt
#
# 4. Measure PPL:
#    python scripts/measure_perplexity.py runs/v2_awq_a0.5/v2_initial.pt --device tpu --dtype fp16 --max-chunks 20
#
# Environment: DEVICE=tpu ./scripts/sweep_awq.sh
#              DEVICE=cuda ./scripts/sweep_awq.sh
#              CLEANUP=false ./scripts/sweep_awq.sh  # keep all intermediate folders
#

IM=runs/imatrix_qwen3_0.6b_random.pt
BASE=Qwen/Qwen3-0.6B
ALPHAS="0.0 0.25 0.5 0.75 1.0"
DEVICE=${DEVICE:-tpu}  # tpu, cuda, mps, cpu
CLEANUP=${CLEANUP:-true}  # true: remove intermediate folders, false: keep all

# Check if iMatrix exists, create if not
if [ ! -f "$IM" ]; then
  echo ">>> iMatrix not found: $IM"
  echo "    Creating from random data..."
  python scripts/compute_imatrix.py \
    --model $BASE \
    --tokens 50000 --seq-len 512 \
    --calib-mode random_ids \
    --out $IM
  if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create iMatrix"
    exit 1
  fi
  echo ""
fi

# Results array (alpha:ppl pairs)
declare -A RESULTS

echo "============================================================"
echo "AWQ ALPHA SWEEP"
echo "============================================================"
echo "Base model: $BASE"
echo "iMatrix:    $IM"
echo "Alphas:     $ALPHAS"
echo "============================================================"
echo ""

for a in $ALPHAS; do
  OUT_BASE=runs/awq_scaled_a${a}
  OUT_INIT=runs/v2_awq_a${a}
  LOG_FILE=runs/sweep_awq_a${a}.log

  echo ">>> Alpha = $a"
  echo "    AWQ output:  $OUT_BASE"
  echo "    Init output: $OUT_INIT"
  echo "    Log file:    $LOG_FILE"

  # Step 1: Apply AWQ scales
  echo ""
  echo "    [1/3] python scripts/apply_awq_equiv_scales.py --model-id $BASE --imatrix $IM --alpha $a --output $OUT_BASE --dtype float32"
  python scripts/apply_awq_equiv_scales.py \
    --model-id $BASE \
    --imatrix $IM \
    --alpha $a \
    --output $OUT_BASE \
    --dtype float32 2>&1 | tee "$LOG_FILE"

  # Step 2: Initialize V2 (--lut fp4_dense = fast, --search-lut = slow)
  echo ""
  echo "    [2/3] python scripts/init_model_v2.py --model-id $OUT_BASE -o $OUT_INIT -c q4a4_r32 --lut fp4_dense --imatrix $IM"
  python scripts/init_model_v2.py \
    --model-id $OUT_BASE \
    -o $OUT_INIT \
    -c q4a4_r32 \
    --lut fp4_dense \
    --imatrix $IM 2>&1 | tee -a "$LOG_FILE"

  # Step 3: Measure perplexity
  echo ""
  echo "    [3/3] python scripts/measure_perplexity.py $OUT_INIT/v2_initial.pt --device $DEVICE --dtype fp16 --max-chunks 20 --output-ppl"
  PPL_OUTPUT=$(python scripts/measure_perplexity.py \
    $OUT_INIT/v2_initial.pt \
    --device $DEVICE --dtype fp16 --max-chunks 20 --output-ppl 2>&1 | tee -a "$LOG_FILE")

  # Extract perplexity value (--output-ppl gives "PPL=XX.XXXX")
  PPL=$(echo "$PPL_OUTPUT" | grep "^PPL=" | cut -d= -f2)

  if [ -z "$PPL" ]; then
    PPL="ERROR"
  fi

  RESULTS[$a]=$PPL
  echo "    PPL = $PPL"
  echo ""
done

# Print summary table
echo ""
echo "============================================================"
echo "SUMMARY: AWQ Alpha Sweep Results"
echo "============================================================"
printf "%-10s | %-12s | %s\n" "Alpha" "Perplexity" "Checkpoint"
echo "-----------|--------------|------------------------------------"
for a in $ALPHAS; do
  printf "%-10s | %-12s | %s\n" "$a" "${RESULTS[$a]}" "runs/v2_awq_a${a}/v2_initial.pt"
done
echo "============================================================"

# Find best alpha
BEST_ALPHA=""
BEST_PPL=999999
for a in $ALPHAS; do
  ppl="${RESULTS[$a]}"
  if [[ "$ppl" != "ERROR" ]] && (( $(echo "$ppl < $BEST_PPL" | bc -l) )); then
    BEST_PPL=$ppl
    BEST_ALPHA=$a
  fi
done

if [ -n "$BEST_ALPHA" ]; then
  echo ""
  echo "Best: alpha=$BEST_ALPHA (PPL=$BEST_PPL)"
  echo "      runs/v2_awq_a${BEST_ALPHA}/v2_initial.pt"
fi

# Cleanup intermediate folders
if [ "$CLEANUP" = "true" ] && [ -n "$BEST_ALPHA" ]; then
  echo ""
  echo "Cleaning up intermediate folders (CLEANUP=$CLEANUP)..."

  for a in $ALPHAS; do
    # Remove AWQ-scaled model folders (large, not needed after init)
    AWQ_DIR="runs/awq_scaled_a${a}"
    if [ -d "$AWQ_DIR" ]; then
      echo "  Removing $AWQ_DIR"
      rm -rf "$AWQ_DIR"
    fi

    # Remove non-best v2 init folders
    if [ "$a" != "$BEST_ALPHA" ]; then
      V2_DIR="runs/v2_awq_a${a}"
      if [ -d "$V2_DIR" ]; then
        echo "  Removing $V2_DIR (not best)"
        rm -rf "$V2_DIR"
      fi
    fi
  done

  echo "  Kept: runs/v2_awq_a${BEST_ALPHA}/ (best)"
  echo ""
  echo "To disable cleanup: CLEANUP=false ./scripts/sweep_awq.sh"
fi
echo ""
