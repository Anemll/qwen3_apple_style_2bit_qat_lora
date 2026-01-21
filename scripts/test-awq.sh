#!/bin/bash
# Test AWQ pipeline with perplexity tracking
#
# MANUAL TEST COMMANDS (copy/paste):
# ==================================
#
# 1. Create iMatrix (if needed):
#    python scripts/compute_imatrix.py --model Qwen/Qwen3-0.6B --tokens 50000 --seq-len 512 --calib-mode random_ids --out runs/imatrix_qwen3_0.6b_random.pt
#
# 2. Apply AWQ scales:
#    python scripts/apply_awq_equiv_scales.py --model-id Qwen/Qwen3-0.6B --imatrix runs/imatrix_qwen3_0.6b_random.pt --alpha 0.5 --output runs/awq_scaled_model
#
# 3. Initialize V2 (fast: --lut fp4_dense, or slow: --search-lut):
#    python scripts/init_model_v2.py --model-id runs/awq_scaled_model --output runs/v2_awq_alpha05 --config q4a4_r32 --lut fp4_dense --imatrix runs/imatrix_qwen3_0.6b_random.pt --svd-error
#    python scripts/init_model_v2.py --model-id runs/awq_scaled_model --output runs/v2_awq_alpha05 --config q4a4_r32 --search-lut --imatrix runs/imatrix_qwen3_0.6b_random.pt --svd-error
#
# 4. Measure PPL (V2 init):
#    python scripts/measure_perplexity.py runs/v2_awq_alpha05/v2_initial.pt --device tpu --dtype fp16 
#
# 5. Select best LUT per layer (sequential, verbose - shows per-tensor stats):
#    python scripts/select_best_lut_per_layer.py runs/v2_awq_alpha05/v2_initial.pt -o runs/v2_init_imse/ihybrid.pt --metric iActMSE --imatrix runs/imatrix_qwen3_0.6b_random.pt --families E,G --no-tighten --verbose
#
# 6. Measure PPL (hybrid):
#    python scripts/measure_perplexity.py runs/v2_init_imse/ihybrid.pt --device tpu --dtype fp16 
#
# 7. Test inference:
#    python scripts/test_inference.py runs/v2_init_imse/ihybrid.pt --prompt "Who invented the iPad?" --no-think
#

IM=runs/imatrix_qwen3_0.6b_random.pt
BASE=Qwen/Qwen3-0.6B
ALPHA=0.25
DEVICE=${DEVICE:-tpu}  # tpu, cuda, mps, cpu

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

# Collect PPL results
declare -A PPL_RESULTS

echo "============================================================"
echo "AWQ PIPELINE TEST"
echo "============================================================"
echo "Base model: $BASE"
echo "iMatrix:    $IM"
echo "Alpha:      $ALPHA"
echo "============================================================"
echo ""

# Step 1: Apply AWQ transforms
echo ">>> Step 1: Applying AWQ-equivalent scale transforms"
echo "    CMD: python scripts/apply_awq_equiv_scales.py --model-id $BASE --imatrix $IM --alpha $ALPHA --output runs/awq_scaled_model"
python scripts/apply_awq_equiv_scales.py \
       --model-id $BASE \
       --imatrix $IM \
       --alpha $ALPHA \
       --output runs/awq_scaled_model

# Step 2: Initialize V2
echo ""
echo ">>> Step 2: Initializing V2 from AWQ-scaled model"
echo "    CMD: python3 scripts/init_model_v2.py --model-id runs/awq_scaled_model --output runs/v2_awq_alpha05 --config q4a4_r32 --search-lut --imatrix $IM --svd-error"
python3 scripts/init_model_v2.py \
    --model-id runs/awq_scaled_model \
    --output runs/v2_awq_alpha05 \
    --config q4a4_r32 \
    --search-lut \
    --imatrix $IM --svd-error

# PPL check after init
echo ""
echo ">>> Step 2b: Measuring PPL (V2 init)"
echo "    CMD: python3 scripts/measure_perplexity.py runs/v2_awq_alpha05/v2_initial.pt --device $DEVICE --dtype fp16  --output-ppl"
PPL_OUTPUT=$(python3 scripts/measure_perplexity.py runs/v2_awq_alpha05/v2_initial.pt --device $DEVICE --dtype fp16  --output-ppl 2>&1)
PPL=$(echo "$PPL_OUTPUT" | grep "^PPL=" | cut -d= -f2)
PPL_RESULTS["v2_init"]=$PPL
echo "    PPL (v2_init) = $PPL"

# Step 3: Quick inference test
echo ""
echo ">>> Step 3: Testing inference (V2 init)"
echo "    CMD: python3 scripts/test_inference.py runs/v2_awq_alpha05/v2_initial.pt --prompt \"Who invented the iPad?\" --no-think --config q4_r32"
python3 scripts/test_inference.py runs/v2_awq_alpha05/v2_initial.pt \
     --prompt "Who invented the iPad?" \
     --no-think \
     --config q4_r32

# Step 4: Select best LUT per layer (no --workers = sequential with verbose per-tensor stats)
echo ""
echo ">>> Step 4: Selecting best LUT per layer (E,G families)"
echo "    CMD: python3 scripts/select_best_lut_per_layer.py runs/v2_awq_alpha05/v2_initial.pt -o runs/v2_init_imse/ihybrid.pt --metric iActMSE --imatrix $IM --families E,G --no-tighten --verbose"
python3 scripts/select_best_lut_per_layer.py runs/v2_awq_alpha05/v2_initial.pt \
    -o runs/v2_init_imse/ihybrid.pt \
    --metric iActMSE \
    --imatrix $IM \
    --families A,B,C,D,E,F,G --no-tighten --verbose

# PPL check after hybrid
echo ""
echo ">>> Step 4b: Measuring PPL (E,G hybrid)"
echo "    CMD: python3 scripts/measure_perplexity.py runs/v2_init_imse/ihybrid.pt --device $DEVICE --dtype fp16  --output-ppl"
PPL_OUTPUT=$(python3 scripts/measure_perplexity.py runs/v2_init_imse/ihybrid.pt --device $DEVICE --dtype fp16  --output-ppl 2>&1)
PPL=$(echo "$PPL_OUTPUT" | grep "^PPL=" | cut -d= -f2)
PPL_RESULTS["hybrid"]=$PPL
echo "    PPL (hybrid) = $PPL"

# Step 5: Inference test on hybrid
echo ""
echo ">>> Step 5: Testing inference (E,G hybrid)"
echo "    CMD: python3 scripts/test_inference.py runs/v2_init_imse/ihybrid.pt --prompt \"Who invented the iPad?\" --no-think"
python3 scripts/test_inference.py runs/v2_init_imse/ihybrid.pt \
     --prompt "Who invented the iPad?" \
     --no-think

# Summary
echo ""
echo "============================================================"
echo "SUMMARY: AWQ Pipeline Results"
echo "============================================================"
printf "%-20s | %-12s | %s\n" "Stage" "PPL" "Checkpoint"
echo "---------------------|--------------|------------------------------------"
printf "%-20s | %-12s | %s\n" "V2 Init" "${PPL_RESULTS[v2_init]:-N/A}" "runs/v2_awq_alpha05/v2_initial.pt"
printf "%-20s | %-12s | %s\n" "E,G Hybrid" "${PPL_RESULTS[hybrid]:-N/A}" "runs/v2_init_imse/ihybrid.pt"
echo "============================================================"

# Compute improvement
if [[ -n "${PPL_RESULTS[v2_init]}" && -n "${PPL_RESULTS[hybrid]}" ]]; then
  DELTA=$(echo "${PPL_RESULTS[v2_init]} - ${PPL_RESULTS[hybrid]}" | bc -l 2>/dev/null)
  if [[ -n "$DELTA" ]]; then
    echo ""
    echo "Improvement: $DELTA PPL (init -> hybrid)"
  fi
fi
echo ""
