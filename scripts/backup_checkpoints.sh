#!/bin/bash
# ==============================================================================
# One-shot backup of checkpoint files to Google Drive
# Use this for manual backups or before disconnecting
#
# Usage:
#   ./scripts/backup_checkpoints.sh LOCAL_DIR GDRIVE_DIR
#
# Examples:
#   ./scripts/backup_checkpoints.sh /content/runs /content/drive/MyDrive/qwen3_runs
#   ./scripts/backup_checkpoints.sh ./lut_candidates /content/drive/MyDrive/lut_candidates
# ==============================================================================

LOCAL_DIR="${1:?Usage: $0 LOCAL_DIR GDRIVE_DIR}"
GDRIVE_DIR="${2:?Usage: $0 LOCAL_DIR GDRIVE_DIR}"

echo "=============================================="
echo "BACKUP CHECKPOINTS TO GOOGLE DRIVE"
echo "=============================================="
echo "From: $LOCAL_DIR"
echo "To:   $GDRIVE_DIR"
echo "=============================================="

if [[ ! -d "$LOCAL_DIR" ]]; then
    echo "ERROR: Local directory does not exist: $LOCAL_DIR"
    exit 1
fi

mkdir -p "$GDRIVE_DIR"

# Count files
pt_count=$(find "$LOCAL_DIR" -name "*.pt" 2>/dev/null | wc -l)
json_count=$(find "$LOCAL_DIR" -name "*.json" 2>/dev/null | wc -l)

echo "Found: $pt_count .pt files, $json_count .json files"
echo ""

# Use rsync if available (faster, handles interruptions)
if command -v rsync &> /dev/null; then
    echo "Using rsync for efficient copy..."
    rsync -av --progress \
        --include="*/" \
        --include="*.pt" \
        --include="*.json" \
        --exclude="*" \
        "$LOCAL_DIR/" "$GDRIVE_DIR/"
else
    echo "Using cp (rsync not available)..."

    # Copy .pt files
    find "$LOCAL_DIR" -name "*.pt" -print0 | while IFS= read -r -d '' file; do
        rel_path="${file#$LOCAL_DIR/}"
        dest_dir="$GDRIVE_DIR/$(dirname "$rel_path")"
        mkdir -p "$dest_dir"
        cp -v "$file" "$GDRIVE_DIR/$rel_path"
    done

    # Copy .json files
    find "$LOCAL_DIR" -name "*.json" -print0 | while IFS= read -r -d '' file; do
        rel_path="${file#$LOCAL_DIR/}"
        dest_dir="$GDRIVE_DIR/$(dirname "$rel_path")"
        mkdir -p "$dest_dir"
        cp -v "$file" "$GDRIVE_DIR/$rel_path"
    done
fi

echo ""
echo "=============================================="
echo "BACKUP COMPLETE"
echo "=============================================="

# Show what was backed up
echo "Contents of $GDRIVE_DIR:"
ls -lh "$GDRIVE_DIR"/*.pt 2>/dev/null | head -10 || echo "(no .pt files in root)"
echo ""
echo "Total size: $(du -sh "$GDRIVE_DIR" 2>/dev/null | cut -f1)"
