#!/bin/bash

echo "Step 1: Applying AWQ-equivalent scale transforms to Qwen3 model"

python scripts/apply_awq_equiv_scales.py \
       --model-id Qwen/Qwen3-0.6B \
       --imatrix runs/imatrix_qwen3_0.6b_random.pt \
       --alpha 0.5 \
       --output runs/awq_scaled_model

echo "Step 2: Initializing V2 from AWQ-scaled model"

python3 scripts/init_model_v2.py \
    --model-id runs/awq_scaled_model \
    --output runs/v2_awq_alpha05 \
    --config q4a4_r32 \
    --search-lut \
    --imatrix runs/imatrix_qwen3_0.6b_random.pt --svd-error

echo "Step 3: Testing inference on Qwen3 model"
python3 scripts/test_inference.py runs/v2_awq_alpha05/v2_initial.pt \
     --prompt "Who invented the iPad?" \
     --no-think \
     --config q4_r32 
echo "--------------------------------------------------------"

echo "Step 4: Selecting best LUT per layer for E,G families"
# Now self-contained: W_orig is computed from checkpoint's W_eff = Q * S
# Config is auto-loaded from checkpoint's config.json (group_size, model_id, etc.)
# E = original LUT (baseline), G = k-means (candidate)
python3 scripts/select_best_lut_per_layer.py runs/v2_awq_alpha05/v2_initial.pt \
    -o runs/v2_init_imse/ihybrid.pt \
    --workers 8 \
    --metric iActMSE \
    --imatrix runs/imatrix_qwen3_0.6b_random.pt \
    --families E,G --no-tighten 
    #--config q4_r32

echo "Step 5: Testing inference on Qwen3 model with E,G hybrid"
# Checkpoint is self-contained: LayerNorm weights are loaded from checkpoint
# Config auto-loaded from checkpoint's config.json
python3 scripts/test_inference.py runs/v2_init_imse/ihybrid.pt \
     --prompt "Who invented the iPad?" \
     --no-think

echo "--------------------------------------------------------"

# --prompt "Make a list of first 50 US Presidents in chronological order" \
