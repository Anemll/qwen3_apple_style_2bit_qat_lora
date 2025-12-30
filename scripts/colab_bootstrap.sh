#!/bin/bash
# Colab Bootstrap Script for Anemll QAT Training
# Run this after uploading checkpoint archive to /content/
#
# Usage in Colab:
#   !wget -q https://raw.githubusercontent.com/YOUR_REPO/main/scripts/colab_bootstrap.sh
#   !bash colab_bootstrap.sh [checkpoint_archive.tar.gz]

set -e

echo "=== Anemll QAT Colab Bootstrap ==="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

cd /content

# 1. Clone or update repo
echo -e "${YELLOW}[1/4] Setting up repository...${NC}"
if [ -d "qwen3_apple_style_2bit_qat_lora" ]; then
    echo "  Repo exists, pulling latest..."
    cd qwen3_apple_style_2bit_qat_lora
    git fetch origin
    git reset --hard origin/main
    cd /content
else
    echo "  Cloning repo..."
    git clone https://github.com/anemll/qwen3_apple_style_2bit_qat_lora.git
fi
echo -e "${GREEN}  Done!${NC}"

# 2. Install dependencies
echo -e "${YELLOW}[2/4] Installing dependencies...${NC}"
pip install -q transformers accelerate datasets sentencepiece protobuf
echo -e "${GREEN}  Done!${NC}"

# 3. Create runs directory
echo -e "${YELLOW}[3/4] Setting up directories...${NC}"
mkdir -p /content/runs
echo -e "${GREEN}  Done!${NC}"

# 4. Extract checkpoint archive if provided
echo -e "${YELLOW}[4/4] Checking for checkpoint archive...${NC}"
ARCHIVE="$1"

if [ -z "$ARCHIVE" ]; then
    # Try to find any tar.gz in /content
    ARCHIVE=$(ls /content/*.tar.gz 2>/dev/null | head -1)
fi

if [ -n "$ARCHIVE" ] && [ -f "$ARCHIVE" ]; then
    echo "  Found archive: $ARCHIVE"
    echo "  Extracting to /content/runs/..."
    tar -xzf "$ARCHIVE" -C /content/runs/
    echo "  Contents:"
    ls -la /content/runs/

    # Find checkpoint files
    echo ""
    echo "  Checkpoint files found:"
    find /content/runs -name "*.pt" -type f 2>/dev/null | head -10
    echo -e "${GREEN}  Done!${NC}"
else
    echo "  No archive found. Upload checkpoint.tar.gz to /content/ and re-run,"
    echo "  or run: !bash scripts/colab_bootstrap.sh your_archive.tar.gz"
fi

echo ""
echo -e "${GREEN}=== Bootstrap Complete ===${NC}"
echo ""
echo "Next steps:"
echo "  1. Open the notebook: notebooks/Anemll_V1_to_V2_Q2A4_STE_FP16.ipynb"
echo "  2. Update LOCAL_RUNS path if needed"
echo "  3. Run the cells!"
echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
