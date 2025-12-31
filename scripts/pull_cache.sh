#!/bin/bash
# =============================================================================
# Pull KD Cache from Google Drive
# =============================================================================
# Usage: bash scripts/pull_cache.sh <cache_name>
#
# Examples:
#   bash scripts/pull_cache.sh alpaca_chat_think_both_L64_K64_R128
#   bash scripts/pull_cache.sh alpaca_chat_think_both_L128_K128_R1024
#
# Available caches:
#   alpaca_chat_think_both_L64_K64_R128    (~1.5 GB, batch 16-32)
#   alpaca_chat_think_both_L128_K128_R1024 (~16 GB, batch 4-8)
# =============================================================================

set -e

# Google Drive path
DRIVE_CACHES="/content/drive/MyDrive/qwen3_caches"
LOCAL_CACHES="caches"

# Check argument
if [ -z "$1" ]; then
    echo "Usage: bash scripts/pull_cache.sh <cache_name>"
    echo ""
    echo "Available caches:"
    echo "  alpaca_chat_think_both_L64_K64_R128    (~1.5 GB, batch 16-32)"
    echo "  alpaca_chat_think_both_L128_K128_R1024 (~16 GB, batch 4-8)"
    echo ""
    echo "Examples:"
    echo "  bash scripts/pull_cache.sh alpaca_chat_think_both_L64_K64_R128"
    echo "  bash scripts/pull_cache.sh alpaca_chat_think_both_L128_K128_R1024"
    exit 1
fi

CACHE_NAME="$1"
CACHE_ARCHIVE="${CACHE_NAME}.tgz"

echo "============================================================"
echo "Pulling KD Cache: $CACHE_NAME"
echo "============================================================"

# Create local caches directory
mkdir -p "$LOCAL_CACHES"

# Check if already exists
if [ -d "$LOCAL_CACHES/$CACHE_NAME" ]; then
    echo "Cache already exists: $LOCAL_CACHES/$CACHE_NAME"
    FILE_COUNT=$(ls "$LOCAL_CACHES/$CACHE_NAME"/*.pt 2>/dev/null | wc -l)
    echo "  Files: $FILE_COUNT .pt files"
    echo ""
    read -p "Re-extract? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping extraction."
        exit 0
    fi
fi

# Check if archive exists on Drive
if [ ! -f "$DRIVE_CACHES/$CACHE_ARCHIVE" ]; then
    echo "ERROR: Cache archive not found: $DRIVE_CACHES/$CACHE_ARCHIVE"
    echo ""
    echo "Available archives on Drive:"
    ls -la "$DRIVE_CACHES"/*.tgz 2>/dev/null || echo "  (none found)"
    exit 1
fi

# Extract
echo "Extracting $CACHE_ARCHIVE..."
tar -xzf "$DRIVE_CACHES/$CACHE_ARCHIVE" -C "$LOCAL_CACHES/"

# Verify
if [ -d "$LOCAL_CACHES/$CACHE_NAME" ]; then
    FILE_COUNT=$(ls "$LOCAL_CACHES/$CACHE_NAME"/*.pt 2>/dev/null | wc -l)
    echo ""
    echo "Done!"
    echo "  Path: $LOCAL_CACHES/$CACHE_NAME"
    echo "  Files: $FILE_COUNT .pt files"
else
    echo "ERROR: Extraction failed!"
    exit 1
fi

echo "============================================================"
