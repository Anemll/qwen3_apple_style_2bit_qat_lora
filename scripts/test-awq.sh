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

IM=${IM:-runs/imatrix_qwen3_0.6b_random.pt}
BASE=${BASE:-Qwen/Qwen3-0.6B}
ALPHA=${ALPHA:-0.25}
DEVICE=${DEVICE:-cpu}  # cpu for transforms (safest)

# PPL device: auto-detect best available accelerator
# Priority: TPU > CUDA > MPS (Mac) > CPU
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
EnablePPL=${EnablePPL:-true}

# LUT selection
WORKERS=${WORKERS:-8}

# Check if iMatrix exists, create if not
if [ ! -f "$IM" ]; then
  echo ">>> iMatrix not found: $IM"
  echo "    Creating from random data..."
  cmd="python scripts/compute_imatrix.py --model $BASE --tokens 50000 --seq-len 512 --calib-mode random_ids --out $IM"
  echo "CMD: $cmd"
  $cmd
  if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create iMatrix"
    exit 1
  fi
  echo ""
fi

# Collect PPL results (simple vars for bash 3.x compat)
PPL_V2_INIT="N/A"
PPL_HYBRID="N/A"

echo "============================================================"
echo "AWQ PIPELINE TEST"
echo "============================================================"
echo "Base model:  $BASE"
echo "iMatrix:     $IM"
echo "Alpha:       $ALPHA"
echo "PPL_DEVICE:  $PPL_DEVICE"
echo "EnablePPL:   $EnablePPL"
echo "============================================================"
echo ""

# Step 1: Apply AWQ transforms
echo ">>> Step 1: Applying AWQ-equivalent scale transforms"
cmd="python scripts/apply_awq_equiv_scales.py --model-id $BASE --imatrix $IM --alpha $ALPHA --output runs/awq_scaled_model"
echo "CMD: $cmd"
$cmd

# Step 2: Initialize V2
echo ""
echo ">>> Step 2: Initializing V2 from AWQ-scaled model"
cmd="python3 scripts/init_model_v2.py --model-id runs/awq_scaled_model --output runs/v2_awq_alpha05 --config q4a4_r32 --search-lut --imatrix $IM --svd-error"
echo "CMD: $cmd"
$cmd

# PPL check after init
echo ""
if [[ "${EnablePPL}" == "false" ]]; then
  echo ">>> Step 2b: Skipped (EnablePPL=false)"
  PPL_V2_INIT="N/A"
elif [[ "${EnablePPL}" == "true" ]]; then
  echo ">>> Step 2b: Measuring PPL (V2 init) - full [$PPL_DEVICE]"
  cmd="python3 scripts/measure_perplexity.py runs/v2_awq_alpha05/v2_initial.pt --device $PPL_DEVICE --dtype fp16 --output-ppl"
  echo "CMD: $cmd"
  PPL_OUTPUT=$($cmd 2>&1)
  PPL=$(echo "$PPL_OUTPUT" | grep "^PPL=" | cut -d= -f2)
  PPL_V2_INIT=$PPL
  echo "    PPL (v2_init) = $PPL"
elif [[ "${EnablePPL}" =~ ^[0-9]+$ ]]; then
  echo ">>> Step 2b: Measuring PPL (V2 init) - max-chunks=${EnablePPL} [$PPL_DEVICE]"
  cmd="python3 scripts/measure_perplexity.py runs/v2_awq_alpha05/v2_initial.pt --device $PPL_DEVICE --dtype fp16 --output-ppl --max-chunks ${EnablePPL}"
  echo "CMD: $cmd"
  PPL_OUTPUT=$($cmd 2>&1)
  PPL=$(echo "$PPL_OUTPUT" | grep "^PPL=" | cut -d= -f2)
  PPL_V2_INIT=$PPL
  echo "    PPL (v2_init) = $PPL"
else
  echo ">>> Step 2b: Skipped (EnablePPL=${EnablePPL} invalid)"
  PPL_V2_INIT="N/A"
fi

# Step 3: Quick inference test
echo ""
echo ">>> Step 3: Testing inference (V2 init)"
cmd="python3 scripts/test_inference.py runs/v2_awq_alpha05/v2_initial.pt --prompt 'Who invented the iPad?' --no-think --config q4_r32"
echo "CMD: $cmd"
python3 scripts/test_inference.py runs/v2_awq_alpha05/v2_initial.pt --prompt "Who invented the iPad?" --no-think --config q4_r32

# Step 4: Select best LUT per layer (no --workers = sequential with verbose per-tensor stats)
echo ""
echo ">>> Step 4: Selecting best LUT per layer (A-G families)"
cmd="python3 scripts/select_best_lut_per_layer.py runs/v2_awq_alpha05/v2_initial.pt -o runs/v2_init_imse/ihybrid.pt --metric iActMSE --imatrix $IM --families A,B,C,D,E,F,G --no-tighten --verbose --workers $WORKERS"
echo "CMD: $cmd"
$cmd

# PPL check after hybrid
echo ""
if [[ "${EnablePPL}" == "false" ]]; then
  echo ">>> Step 4b: Skipped (EnablePPL=false)"
  PPL_HYBRID="N/A"
elif [[ "${EnablePPL}" == "true" ]]; then
  echo ">>> Step 4b: Measuring PPL (hybrid) - full [$PPL_DEVICE]"
  cmd="python3 scripts/measure_perplexity.py runs/v2_init_imse/ihybrid.pt --device $PPL_DEVICE --dtype fp16 --output-ppl"
  echo "CMD: $cmd"
  PPL_OUTPUT=$($cmd 2>&1)
  PPL=$(echo "$PPL_OUTPUT" | grep "^PPL=" | cut -d= -f2)
  PPL_HYBRID=$PPL
  echo "    PPL (hybrid) = $PPL"
elif [[ "${EnablePPL}" =~ ^[0-9]+$ ]]; then
  echo ">>> Step 4b: Measuring PPL (hybrid) - max-chunks=${EnablePPL} [$PPL_DEVICE]"
  cmd="python3 scripts/measure_perplexity.py runs/v2_init_imse/ihybrid.pt --device $PPL_DEVICE --dtype fp16 --output-ppl --max-chunks ${EnablePPL}"
  echo "CMD: $cmd"
  PPL_OUTPUT=$($cmd 2>&1)
  PPL=$(echo "$PPL_OUTPUT" | grep "^PPL=" | cut -d= -f2)
  PPL_HYBRID=$PPL
  echo "    PPL (hybrid) = $PPL"
else
  echo ">>> Step 4b: Skipped (EnablePPL=${EnablePPL} invalid)"
  PPL_HYBRID="N/A"
fi

# Step 5: Inference test on hybrid
echo ""
echo ">>> Step 5: Testing inference (hybrid)"
cmd="python3 scripts/test_inference.py runs/v2_init_imse/ihybrid.pt --prompt 'Who invented the iPad?' --no-think"
echo "CMD: $cmd"
python3 scripts/test_inference.py runs/v2_init_imse/ihybrid.pt --prompt "Who invented the iPad?" --no-think

# Summary
echo ""
echo "============================================================"
echo "SUMMARY: AWQ Pipeline Results (ALPHA=$ALPHA)"
echo "============================================================"
printf "%-20s | %-12s | %s\n" "Stage" "PPL" "Checkpoint"
echo "---------------------|--------------|------------------------------------"
printf "%-20s | %-12s | %s\n" "V2 Init" "${PPL_V2_INIT}" "runs/v2_awq_alpha05/v2_initial.pt"
printf "%-20s | %-12s | %s\n" "E,G Hybrid" "${PPL_HYBRID}" "runs/v2_init_imse/ihybrid.pt"
echo "============================================================"

# Compute improvement
if [[ -n "${PPL_V2_INIT}" && -n "${PPL_HYBRID}" && "${PPL_V2_INIT}" != "N/A" && "${PPL_HYBRID}" != "N/A" ]]; then
  DELTA=$(echo "${PPL_V2_INIT} - ${PPL_HYBRID}" | bc -l 2>/dev/null)
  if [[ -n "$DELTA" ]]; then
    echo ""
    echo "Improvement: $DELTA PPL (init -> hybrid)"
  fi
fi
echo ""
