#!/bin/bash
# Kill processes using TPU device
# Usage: ./scripts/killtpu.sh [--dry-run]

set -e

DRY_RUN=false
if [[ "$1" == "--dry-run" ]] || [[ "$1" == "-n" ]]; then
    DRY_RUN=true
fi

echo "============================================"
echo "TPU PROCESS KILLER"
echo "============================================"

# Try lsof first (more portable)
PIDS=""
if command -v lsof &> /dev/null; then
    PIDS=$(sudo lsof /dev/vfio/0 2>/dev/null | awk 'NR>1 {print $2}' | sort -u || true)
fi

# Fallback to fuser if lsof didn't find anything
if [[ -z "$PIDS" ]] && command -v fuser &> /dev/null; then
    PIDS=$(sudo fuser /dev/vfio/0 2>/dev/null || true)
fi

if [[ -z "$PIDS" ]]; then
    echo "No processes found using TPU (/dev/vfio/0)"
    echo ""
    echo "TPU is free!"
    exit 0
fi

echo "Processes using TPU:"
echo ""
for PID in $PIDS; do
    # Get process info
    if [[ -d "/proc/$PID" ]]; then
        CMD=$(cat /proc/$PID/cmdline 2>/dev/null | tr '\0' ' ' | head -c 80 || echo "unknown")
        USER=$(stat -c '%U' /proc/$PID 2>/dev/null || echo "unknown")
        echo "  PID $PID ($USER): $CMD"
    else
        echo "  PID $PID: (process info unavailable)"
    fi
done
echo ""

if $DRY_RUN; then
    echo "[DRY RUN] Would kill PIDs: $PIDS"
    echo ""
    echo "Run without --dry-run to actually kill processes"
    exit 0
fi

echo "Killing processes..."
for PID in $PIDS; do
    echo "  Killing PID $PID..."
    sudo kill -9 $PID 2>/dev/null || echo "    (already dead)"
done

echo ""
echo "Done! TPU should be free now."
echo ""
echo "Verify with: sudo lsof /dev/vfio/0"
