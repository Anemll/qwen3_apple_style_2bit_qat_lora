#!/bin/bash
# Create Q2 init checkpoint from Q4 and push to Google Drive
# Can run on CPU-only instance (T4 free tier or even T3)
#
# Usage: bash speedrun/create_q2_init.sh

set -e

echo "=== Q4→Q2 Conversion + Push ==="

# Detect Google Drive
if [ -d "/content/drive/MyDrive" ]; then
    GDRIVE_BASE="/content/drive/MyDrive"
    PLATFORM="colab"
elif [ -d "$HOME/Library/CloudStorage/GoogleDrive-realanemll@gmail.com/My Drive" ]; then
    GDRIVE_BASE="$HOME/Library/CloudStorage/GoogleDrive-realanemll@gmail.com/My Drive"
    PLATFORM="macos"
else
    echo "[ERROR] Google Drive not found"
    exit 1
fi

GDRIVE_RUNS="$GDRIVE_BASE/qwen3_runs"
GDRIVE_CACHES="$GDRIVE_BASE/qwen3_caches"

# Q4 source checkpoint (V2 Q4_A4 FP32 - best for conversion)
Q4_SOURCE="$GDRIVE_RUNS/anemll_v2_q4_a4_from_v1_finetuned.tgz"
OUTPUT_DIR="runs/q2_from_q4"
OUTPUT_NAME="q2_init_from_q4"

# Cache for eval (L64 is smallest available - 913MB)
CACHE_NAME="alpaca_chat_think_both_L64_K64_R128"
CACHE_TGZ="$GDRIVE_CACHES/${CACHE_NAME}.tgz"
CACHE_DIR="caches/$CACHE_NAME"

echo "[1/6] Setup..."
mkdir -p "$OUTPUT_DIR" caches

# Install deps if needed (Colab)
if [ "$PLATFORM" = "colab" ]; then
    pip install -q transformers torch
fi

echo "[2/6] Pulling L64 cache (913MB)..."
if [ -d "$CACHE_DIR" ] && [ "$(ls -A $CACHE_DIR 2>/dev/null)" ]; then
    echo "  Cache exists locally"
elif [ -f "$CACHE_TGZ" ]; then
    echo "  Extracting from $CACHE_TGZ..."
    tar -xzf "$CACHE_TGZ" -C caches/
    echo "  Done"
else
    echo "[ERROR] L64 cache not found: $CACHE_TGZ"
    exit 1
fi

echo "[3/6] Extracting Q4 checkpoint..."
if [ -f "$Q4_SOURCE" ]; then
    tar -xzf "$Q4_SOURCE" -C "$OUTPUT_DIR/"
    Q4_CKPT=$(ls "$OUTPUT_DIR"/*.pt 2>/dev/null | head -1)
    echo "  Found: $Q4_CKPT"
else
    echo "[ERROR] Q4 source not found: $Q4_SOURCE"
    exit 1
fi

echo "[4/6] Converting Q4→Q2 with eval (CPU-safe)..."
# Note: --device cpu for low-end instances
python scripts/convert_q4_to_q2.py \
    --q4-checkpoint "$Q4_CKPT" \
    --output "$OUTPUT_DIR/q2_init.pt" \
    --device cpu \
    --eval \
    --cache-dir "$CACHE_DIR"

echo "[5/6] Creating .tar.lz4 (fast compression)..."
# Install lz4 if needed
which lz4 >/dev/null || apt-get install -qq lz4
cd "$OUTPUT_DIR"
tar -I lz4 -cvf "${OUTPUT_NAME}.tar.lz4" q2_init.pt config.json 2>/dev/null || \
tar -I lz4 -cvf "${OUTPUT_NAME}.tar.lz4" q2_init.pt
cd -

echo "[6/6] Pushing to Google Drive..."
rsync -ah --progress "$OUTPUT_DIR/${OUTPUT_NAME}.tar.lz4" "$GDRIVE_RUNS/"
echo "  Uploaded: $GDRIVE_RUNS/${OUTPUT_NAME}.tar.lz4"

# Cleanup extracted Q4
rm -f "$OUTPUT_DIR"/*.pt.backup 2>/dev/null

echo ""
echo "=== Done ==="
echo "Q2 init checkpoint: $GDRIVE_RUNS/${OUTPUT_NAME}.tgz"
echo ""
echo "To train on GPU instance:"
echo "  source speedrun/setup.sh L64 q2_init"
echo "  python scripts/train_v2_simple.py \\"
echo "      --v2-checkpoint \$CHECKPOINT \\"
echo "      --cache-dir \$CACHE_DIR \\"
echo "      --output-dir runs/q2_from_q4_mlp \\"
echo "      --mlp-only --max-steps 4000"
