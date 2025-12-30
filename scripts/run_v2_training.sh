#!/bin/bash
# =============================================================================
# V2 STE-FP16 Training Script for Colab
# =============================================================================
# Usage: bash scripts/run_v2_training.sh [--force]
#
# Options:
#   --force    Force re-extraction of checkpoint and cache (skip existence check)
#
# This script:
# 1. Extracts checkpoint from Google Drive (skips if exists, unless --force)
# 2. Extracts L128 cache from Google Drive (skips if exists, unless --force)
# 3. Runs V2 training with STE-FP16
# 4. Saves output to Google Drive
# =============================================================================

set -e  # Exit on error

# Parse arguments
FORCE=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --force)
            FORCE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: bash scripts/run_v2_training.sh [--force]"
            exit 1
            ;;
    esac
done

# =============================================================================
# CONFIGURATION - Edit these paths as needed
# =============================================================================

# Google Drive paths
DRIVE_RUNS="/content/drive/MyDrive/qwen3_runs"
DRIVE_CACHES="/content/drive/MyDrive/qwen3_caches"

# Local paths
LOCAL_RUNS="runs"
LOCAL_CACHES="caches"

# Checkpoint archive name
CHECKPOINT_ARCHIVE="q2_pt_good1.tgz"
CHECKPOINT_FILE="tmp/backup_mlp_e2e_w_0.3824.pt"

# Cache archive name
CACHE_ARCHIVE="alpaca_chat_think_both_L128_K128_R1024.tgz"
CACHE_DIR="alpaca_chat_think_both_L128_K128_R1024"

# Training params
BATCH_SIZE=8
MAX_STEPS=1000
LR=1e-4

# =============================================================================
# STEP 0: Verify we're in the right directory
# =============================================================================

echo "============================================================"
echo "V2 STE-FP16 Training"
echo "============================================================"
[ "$FORCE" = true ] && echo "Mode: --force (re-extract all)"

if [ ! -f "qat_lora/__init__.py" ]; then
    echo "ERROR: Run this script from the repo root directory"
    echo "  cd /content/qwen3_apple_style_2bit_qat_lora"
    echo "  bash scripts/run_v2_training.sh"
    exit 1
fi

echo "Working directory: $(pwd)"
echo ""

# =============================================================================
# STEP 0.5: Install dependencies
# =============================================================================

echo "[0/4] Installing dependencies..."
pip install -q transformers accelerate datasets sentencepiece protobuf
echo "  Done."
echo ""

# Check GPU
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv 2>/dev/null || echo "  No GPU detected"
echo ""

# =============================================================================
# STEP 1: Extract checkpoint
# =============================================================================

echo "[1/4] Extracting checkpoint..."

mkdir -p "$LOCAL_RUNS"

if [ "$FORCE" = true ] || [ ! -f "$LOCAL_RUNS/$CHECKPOINT_FILE" ]; then
    if [ -f "$DRIVE_RUNS/$CHECKPOINT_ARCHIVE" ]; then
        [ "$FORCE" = true ] && echo "  --force: Re-extracting..."
        echo "  Extracting $CHECKPOINT_ARCHIVE..."
        tar -xzf "$DRIVE_RUNS/$CHECKPOINT_ARCHIVE" -C "$LOCAL_RUNS/"
        echo "  Done."
    else
        echo "ERROR: Checkpoint archive not found: $DRIVE_RUNS/$CHECKPOINT_ARCHIVE"
        exit 1
    fi
else
    echo "  Checkpoint already exists: $LOCAL_RUNS/$CHECKPOINT_FILE"
fi

# Verify checkpoint exists
if [ ! -f "$LOCAL_RUNS/$CHECKPOINT_FILE" ]; then
    echo "ERROR: Checkpoint not found after extraction!"
    echo "  Expected: $LOCAL_RUNS/$CHECKPOINT_FILE"
    echo "  Available files:"
    find "$LOCAL_RUNS" -name "*.pt" | head -10
    exit 1
fi

echo "  Checkpoint: $LOCAL_RUNS/$CHECKPOINT_FILE"
echo ""

# =============================================================================
# STEP 2: Extract cache
# =============================================================================

echo "[2/4] Extracting L128 cache..."

mkdir -p "$LOCAL_CACHES"

if [ "$FORCE" = true ] || [ ! -d "$LOCAL_CACHES/$CACHE_DIR" ]; then
    if [ -f "$DRIVE_CACHES/$CACHE_ARCHIVE" ]; then
        [ "$FORCE" = true ] && echo "  --force: Re-extracting..."
        echo "  Extracting $CACHE_ARCHIVE..."
        tar -xzf "$DRIVE_CACHES/$CACHE_ARCHIVE" -C "$LOCAL_CACHES/"
        echo "  Done."
    else
        echo "ERROR: Cache archive not found: $DRIVE_CACHES/$CACHE_ARCHIVE"
        exit 1
    fi
else
    echo "  Cache already exists: $LOCAL_CACHES/$CACHE_DIR"
fi

# Verify cache exists
if [ ! -d "$LOCAL_CACHES/$CACHE_DIR" ]; then
    echo "ERROR: Cache not found after extraction!"
    exit 1
fi

echo "  Cache: $LOCAL_CACHES/$CACHE_DIR"
CACHE_COUNT=$(ls "$LOCAL_CACHES/$CACHE_DIR"/*.pt 2>/dev/null | wc -l)
echo "  Files: $CACHE_COUNT .pt files"
echo ""

# =============================================================================
# STEP 3: Run training
# =============================================================================

echo "[3/4] Starting V2 STE-FP16 training..."
echo ""
echo "  Config: Q2_A4 (MLP=2bit/rank32, Attn=4bit/rank8)"
echo "  Batch size: $BATCH_SIZE"
echo "  Max steps: $MAX_STEPS"
echo "  Learning rate: $LR"
echo ""

python scripts/train_v2_simple.py \
    --v1-checkpoint "$LOCAL_RUNS/$CHECKPOINT_FILE" \
    --cache-dir "$LOCAL_CACHES/$CACHE_DIR" \
    --output-dir "$LOCAL_RUNS/v2_output" \
    --batch-size "$BATCH_SIZE" \
    --max-steps "$MAX_STEPS" \
    --lr "$LR"

echo ""

# =============================================================================
# STEP 4: Copy results to Drive
# =============================================================================

echo "[4/4] Copying results to Google Drive..."

if [ -d "$LOCAL_RUNS/v2_output" ]; then
    mkdir -p "$DRIVE_RUNS/v2_output"
    cp -v "$LOCAL_RUNS/v2_output"/*.pt "$DRIVE_RUNS/v2_output/" 2>/dev/null || echo "  No .pt files to copy"
    echo "  Results saved to: $DRIVE_RUNS/v2_output/"
else
    echo "  No output directory found"
fi

echo ""
echo "============================================================"
echo "Done!"
echo "============================================================"
