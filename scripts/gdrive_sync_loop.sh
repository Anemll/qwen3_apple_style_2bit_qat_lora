#!/bin/bash
#
# Background Google Drive sync loop.
# Periodically uploads a local folder to Google Drive.
#
# Usage:
#   ./scripts/gdrive_sync_loop.sh runs/my_training
#   ./scripts/gdrive_sync_loop.sh runs/my_training --interval 600
#   INTERVAL_SEC=300 ./scripts/gdrive_sync_loop.sh runs/my_training
#
# Environment variables:
#   SYNC_DIR     - Directory to sync (or pass as first argument)
#   INTERVAL_SEC - Seconds between syncs (default: 300 = 5 minutes)
#   LOG_DIR      - Log directory (default: logs)
#   PY_BIN       - Python binary (default: python3)
#
# To run in background:
#   nohup ./scripts/gdrive_sync_loop.sh runs/my_training &
#   # Or with screen/tmux
#
# To stop:
#   pkill -f "gdrive_sync_loop.*runs/my_training"
#

set -euo pipefail

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

# Parse arguments
SYNC_DIR="${1:-${SYNC_DIR:-}}"
shift 2>/dev/null || true

# Parse optional --interval flag
while [[ $# -gt 0 ]]; do
    case "$1" in
        --interval|-i)
            INTERVAL_SEC="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Defaults
INTERVAL_SEC="${INTERVAL_SEC:-300}"
LOG_DIR="${LOG_DIR:-logs}"
PY_BIN="${PY_BIN:-python3}"

# Validate
if [[ -z "$SYNC_DIR" ]]; then
    echo "Usage: $0 <sync_dir> [--interval SECONDS]"
    echo ""
    echo "Examples:"
    echo "  $0 runs/my_training"
    echo "  $0 runs/my_training --interval 600"
    echo "  SYNC_DIR=runs/foo INTERVAL_SEC=120 $0"
    exit 1
fi

if [[ ! -d "$SYNC_DIR" ]]; then
    echo "Error: Directory not found: $SYNC_DIR"
    exit 1
fi

# Setup logging
mkdir -p "$LOG_DIR"
SAFE_NAME="$(echo "$SYNC_DIR" | tr '/:' '__')"
LOG_FILE="$LOG_DIR/gdrive_sync_${SAFE_NAME}.log"

# Prevent multiple instances for same directory
LOCKDIR="/tmp/gdrive_sync_loop.${SAFE_NAME}.lock"
if ! mkdir "$LOCKDIR" 2>/dev/null; then
    echo "Another instance is already running for '$SYNC_DIR'"
    echo "Lock: $LOCKDIR"
    echo "To force restart, remove the lock: rmdir $LOCKDIR"
    exit 0
fi

cleanup() {
    rmdir "$LOCKDIR" 2>/dev/null || true
    echo "Cleanup done, exiting."
}
trap cleanup EXIT INT TERM

# Build command
CMD=( "$PY_BIN" scripts/gdrive_sync.py up "$SYNC_DIR" )

# Print startup info
echo "=========================================="
echo "Google Drive Sync Loop"
echo "=========================================="
echo "Sync dir:  $SYNC_DIR"
echo "Interval:  ${INTERVAL_SEC}s"
echo "Log file:  $LOG_FILE"
echo "Lock:      $LOCKDIR"
echo "Command:   ${CMD[*]}"
echo "=========================================="
echo ""
echo "Press Ctrl+C to stop."
echo ""

# Log startup
{
    echo ""
    echo "=========================================="
    echo "STARTED: $(date)"
    echo "Sync dir:  $SYNC_DIR"
    echo "Interval:  ${INTERVAL_SEC}s"
    echo "=========================================="
} >> "$LOG_FILE"

# Main loop
while true; do
    start_ts="$(date +%s)"

    {
        echo ""
        echo "=== $(date) | START ==="
        echo "CMD: ${CMD[*]}"
    } | tee -a "$LOG_FILE"

    # Run sync (capture exit code without killing the loop)
    set +e
    "${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
    rc=${PIPESTATUS[0]}
    set -e

    end_ts="$(date +%s)"
    dur=$(( end_ts - start_ts ))
    sleep_for=$(( INTERVAL_SEC - dur ))
    (( sleep_for < 0 )) && sleep_for=0

    {
        echo "=== $(date) | END rc=$rc | duration=${dur}s | sleep=${sleep_for}s ==="
        echo ""
    } | tee -a "$LOG_FILE"

    sleep "$sleep_for"
done
