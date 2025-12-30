#!/bin/bash
# =============================================================================
# Save checkpoint to Google Drive
# =============================================================================
# Usage: bash scripts/save_to_drive.sh <checkpoint_path> [name]
#
# Examples:
#   bash scripts/save_to_drive.sh runs/v2_output/v2_q2a4_fp32_20251230_201946.pt
#   bash scripts/save_to_drive.sh runs/v2_output/v2_q2a4_fp32_20251230_201946.pt my_best_model
#
# This will create:
#   /content/drive/MyDrive/qwen3_runs/v2_q2a4_fp32_20251230_201946.tgz
# =============================================================================

set -e

# Check args
if [ -z "$1" ]; then
    echo "Usage: bash scripts/save_to_drive.sh <checkpoint_path> [name]"
    echo ""
    echo "Examples:"
    echo "  bash scripts/save_to_drive.sh runs/v2_output/v2_q2a4_fp32_20251230_201946.pt"
    echo "  bash scripts/save_to_drive.sh runs/v2_output/v2_q2a4_fp32_20251230_201946.pt best_v2"
    exit 1
fi

CHECKPOINT="$1"
CUSTOM_NAME="$2"

# Validate checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

# Google Drive destination
DRIVE_RUNS="/content/drive/MyDrive/qwen3_runs"

# Check Drive is mounted
if [ ! -d "/content/drive/MyDrive" ]; then
    echo "ERROR: Google Drive not mounted. Run:"
    echo "  from google.colab import drive"
    echo "  drive.mount('/content/drive')"
    exit 1
fi

# Create destination if needed
mkdir -p "$DRIVE_RUNS"

# Get filename
BASENAME=$(basename "$CHECKPOINT" .pt)
if [ -n "$CUSTOM_NAME" ]; then
    ARCHIVE_NAME="${CUSTOM_NAME}.tgz"
else
    ARCHIVE_NAME="${BASENAME}.tgz"
fi

# Get directory containing checkpoint
CHECKPOINT_DIR=$(dirname "$CHECKPOINT")

echo "Saving checkpoint to Google Drive..."
echo "  Source: $CHECKPOINT"
echo "  Dest:   $DRIVE_RUNS/$ARCHIVE_NAME"

# Create tarball
cd "$CHECKPOINT_DIR"
tar -czf "/tmp/$ARCHIVE_NAME" "$(basename "$CHECKPOINT")"
cd - > /dev/null

# Copy to Drive
cp "/tmp/$ARCHIVE_NAME" "$DRIVE_RUNS/"
rm "/tmp/$ARCHIVE_NAME"

# Show result
ls -lh "$DRIVE_RUNS/$ARCHIVE_NAME"
echo ""
echo "Done! Saved to: $DRIVE_RUNS/$ARCHIVE_NAME"
