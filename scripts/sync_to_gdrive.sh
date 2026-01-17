#!/bin/bash
# ==============================================================================
# Sync checkpoint files (.pt) to Google Drive for backup
# Run this in background during training to prevent data loss on Colab disconnect
#
# Usage:
#   ./scripts/sync_to_gdrive.sh LOCAL_DIR GDRIVE_DIR [INTERVAL]
#
# Examples:
#   # Sync every 60 seconds (default)
#   ./scripts/sync_to_gdrive.sh /content/runs /content/drive/MyDrive/qwen3_runs
#
#   # Sync every 30 seconds
#   ./scripts/sync_to_gdrive.sh /content/runs /content/drive/MyDrive/qwen3_runs 30
#
#   # Run in background
#   nohup ./scripts/sync_to_gdrive.sh /content/runs /content/drive/MyDrive/backup 60 &
#
# For Colab notebook cell:
#   !nohup bash scripts/sync_to_gdrive.sh {LOCAL_DIR} {GDRIVE_DIR} 60 > /tmp/sync.log 2>&1 &
#   !echo "Sync started, PID: $(pgrep -f sync_to_gdrive)"
# ==============================================================================

set -e

# Arguments
LOCAL_DIR="${1:?Usage: $0 LOCAL_DIR GDRIVE_DIR [INTERVAL_SECONDS]}"
GDRIVE_DIR="${2:?Usage: $0 LOCAL_DIR GDRIVE_DIR [INTERVAL_SECONDS]}"
INTERVAL="${3:-60}"  # Default: 60 seconds

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=============================================="
echo "CHECKPOINT SYNC TO GOOGLE DRIVE"
echo "=============================================="
echo "Local:    $LOCAL_DIR"
echo "GDrive:   $GDRIVE_DIR"
echo "Interval: ${INTERVAL}s"
echo "PID:      $$"
echo "=============================================="

# Ensure directories exist
mkdir -p "$LOCAL_DIR"
mkdir -p "$GDRIVE_DIR"

# Track synced files to avoid re-copying unchanged files
SYNC_LOG="/tmp/sync_to_gdrive_$$.log"
touch "$SYNC_LOG"

sync_files() {
    local count=0
    local errors=0

    # Find all .pt files modified in the last INTERVAL*2 seconds (with some buffer)
    # Or files that haven't been synced yet
    while IFS= read -r -d '' file; do
        if [[ -f "$file" ]]; then
            filename=$(basename "$file")
            rel_path="${file#$LOCAL_DIR/}"
            dest_dir="$GDRIVE_DIR/$(dirname "$rel_path")"
            dest_file="$GDRIVE_DIR/$rel_path"

            # Get file modification time
            local_mtime=$(stat -c %Y "$file" 2>/dev/null || stat -f %m "$file" 2>/dev/null)

            # Check if we need to sync (file is new or modified)
            need_sync=0
            if [[ ! -f "$dest_file" ]]; then
                need_sync=1
            else
                dest_mtime=$(stat -c %Y "$dest_file" 2>/dev/null || stat -f %m "$dest_file" 2>/dev/null)
                if [[ "$local_mtime" -gt "$dest_mtime" ]]; then
                    need_sync=1
                fi
            fi

            if [[ "$need_sync" -eq 1 ]]; then
                mkdir -p "$dest_dir"
                if cp "$file" "$dest_file" 2>/dev/null; then
                    echo -e "${GREEN}[SYNC]${NC} $rel_path ($(du -h "$file" | cut -f1))"
                    ((count++))
                else
                    echo -e "${RED}[ERROR]${NC} Failed to copy $rel_path"
                    ((errors++))
                fi
            fi
        fi
    done < <(find "$LOCAL_DIR" -name "*.pt" -print0 2>/dev/null)

    # Also sync JSON files (candidates_summary.json, eval_results.json, etc.)
    while IFS= read -r -d '' file; do
        if [[ -f "$file" ]]; then
            rel_path="${file#$LOCAL_DIR/}"
            dest_dir="$GDRIVE_DIR/$(dirname "$rel_path")"
            dest_file="$GDRIVE_DIR/$rel_path"

            local_mtime=$(stat -c %Y "$file" 2>/dev/null || stat -f %m "$file" 2>/dev/null)

            need_sync=0
            if [[ ! -f "$dest_file" ]]; then
                need_sync=1
            else
                dest_mtime=$(stat -c %Y "$dest_file" 2>/dev/null || stat -f %m "$dest_file" 2>/dev/null)
                if [[ "$local_mtime" -gt "$dest_mtime" ]]; then
                    need_sync=1
                fi
            fi

            if [[ "$need_sync" -eq 1 ]]; then
                mkdir -p "$dest_dir"
                if cp "$file" "$dest_file" 2>/dev/null; then
                    echo -e "${GREEN}[SYNC]${NC} $rel_path"
                    ((count++))
                else
                    echo -e "${RED}[ERROR]${NC} Failed to copy $rel_path"
                    ((errors++))
                fi
            fi
        fi
    done < <(find "$LOCAL_DIR" -name "*.json" -print0 2>/dev/null)

    if [[ "$count" -gt 0 ]]; then
        echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} Synced $count files"
    fi
    if [[ "$errors" -gt 0 ]]; then
        echo -e "${RED}[$(date '+%H:%M:%S')]${NC} $errors errors"
    fi
}

# Trap Ctrl+C and cleanup
cleanup() {
    echo -e "\n${YELLOW}[STOP]${NC} Sync stopped. Final sync..."
    sync_files
    echo -e "${GREEN}[DONE]${NC} Goodbye!"
    rm -f "$SYNC_LOG"
    exit 0
}
trap cleanup SIGINT SIGTERM

# Initial sync
echo "[$(date '+%H:%M:%S')] Initial sync..."
sync_files

# Main loop
echo "[$(date '+%H:%M:%S')] Watching for changes every ${INTERVAL}s..."
while true; do
    sleep "$INTERVAL"
    sync_files
done
