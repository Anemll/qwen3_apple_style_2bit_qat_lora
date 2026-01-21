#!/bin/bash
# Test AWQ pipeline with perplexity tracking

IM=runs/imatrix_qwen3_0.6b_random.pt
BASE=Qwen/Qwen3-0.6B
ALPHA=0.5
DEVICE=${DEVICE:-tpu}  # tpu, cuda, mps, cpu

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
python scripts/apply_awq_equiv_scales.py \
       --model-id $BASE \
       --imatrix $IM \
       --alpha $ALPHA \
       --output runs/awq_scaled_model

# Step 2: Initialize V2
echo ""
echo ">>> Step 2: Initializing V2 from AWQ-scaled model"
python3 scripts/init_model_v2.py \
    --model-id runs/awq_scaled_model \
    --output runs/v2_awq_alpha05 \
    --config q4a4_r32 \
    --search-lut \
    --imatrix $IM --svd-error

# PPL check after init
echo ""
echo ">>> Step 2b: Measuring PPL (V2 init)"
PPL_OUTPUT=$(python3 scripts/measure_perplexity.py runs/v2_awq_alpha05/v2_initial.pt --device $DEVICE --dtype fp16 --max-chunks 20 2>&1)
PPL=$(echo "$PPL_OUTPUT" | sed 's/\x1b\[[0-9;]*m//g' | grep -E "^Perplexity:" | grep -oE "[0-9]+\.[0-9]+")
PPL_RESULTS["v2_init"]=$PPL
echo "    PPL (v2_init) = $PPL"

# Step 3: Quick inference test
echo ""
echo ">>> Step 3: Testing inference (V2 init)"
python3 scripts/test_inference.py runs/v2_awq_alpha05/v2_initial.pt \
     --prompt "Who invented the iPad?" \
     --no-think \
     --config q4_r32

# Step 4: Select best LUT per layer
echo ""
echo ">>> Step 4: Selecting best LUT per layer (E,G families)"
python3 scripts/select_best_lut_per_layer.py runs/v2_awq_alpha05/v2_initial.pt \
    -o runs/v2_init_imse/ihybrid.pt \
    --workers 8 \
    --metric iActMSE \
    --imatrix $IM \
    --families E,G --no-tighten

# PPL check after hybrid
echo ""
echo ">>> Step 4b: Measuring PPL (E,G hybrid)"
PPL_OUTPUT=$(python3 scripts/measure_perplexity.py runs/v2_init_imse/ihybrid.pt --device $DEVICE --dtype fp16 --max-chunks 20 2>&1)
PPL=$(echo "$PPL_OUTPUT" | sed 's/\x1b\[[0-9;]*m//g' | grep -E "^Perplexity:" | grep -oE "[0-9]+\.[0-9]+")
PPL_RESULTS["hybrid"]=$PPL
echo "    PPL (hybrid) = $PPL"

# Step 5: Inference test on hybrid
echo ""
echo ">>> Step 5: Testing inference (E,G hybrid)"
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
