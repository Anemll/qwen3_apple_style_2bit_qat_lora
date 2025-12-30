#!/bin/bash
# ============================================================
# COLAB ENVIRONMENT SETUP
# Run from terminal: bash scripts/setup_env.sh
# ============================================================

set -e

echo "============================================================"
echo "Anemll QAT Environment Setup"
echo "============================================================"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Paths
REPO_DIR="/content/qwen3_apple_style_2bit_qat_lora"
DRIVE_DIR="/content/drive/MyDrive/anemll_qat"
RUNS_DIR="/content/runs"
CACHE_DIR="/content/cache"

cd "$REPO_DIR"

# 1. Install dependencies
echo -e "\n${YELLOW}[1/5] Installing Python dependencies...${NC}"
pip install -q transformers accelerate datasets sentencepiece protobuf

# 2. Create directories
echo -e "\n${YELLOW}[2/5] Creating directories...${NC}"
mkdir -p "$RUNS_DIR"
mkdir -p "$CACHE_DIR"
mkdir -p "$DRIVE_DIR/checkpoints"
mkdir -p "$DRIVE_DIR/exports"

# 3. Link Drive directories for easy access
echo -e "\n${YELLOW}[3/5] Linking Drive directories...${NC}"
ln -sf "$DRIVE_DIR" /content/drive_qat
echo "  /content/drive_qat -> $DRIVE_DIR"

# 4. Check for checkpoint archives in Drive
echo -e "\n${YELLOW}[4/5] Looking for checkpoint archives...${NC}"
if ls "$DRIVE_DIR"/*.tar.gz 1>/dev/null 2>&1; then
    echo "  Found archives in Drive:"
    ls -la "$DRIVE_DIR"/*.tar.gz
    echo ""
    echo "  To extract, run:"
    echo "    tar -xzf $DRIVE_DIR/YOUR_ARCHIVE.tar.gz -C $RUNS_DIR/"
else
    echo "  No .tar.gz archives found in $DRIVE_DIR"
    echo "  Upload your checkpoint archive there, or copy .pt files directly"
fi

# 5. Download calibration dataset (optional)
echo -e "\n${YELLOW}[5/5] Checking calibration dataset...${NC}"
if [ -f "$CACHE_DIR/calibration_data.pt" ]; then
    echo "  Calibration data exists: $CACHE_DIR/calibration_data.pt"
else
    echo "  No cached calibration data found."
    echo "  Training will download from HuggingFace on first run."
fi

# Show GPU info
echo ""
echo "============================================================"
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo "============================================================"

# Create convenience scripts
echo -e "\n${GREEN}Creating convenience scripts...${NC}"

# Training script
cat > "$REPO_DIR/run_train.sh" << 'TRAIN_EOF'
#!/bin/bash
# Quick training launcher
# Usage: bash run_train.sh [config_name]

CONFIG=${1:-q4a4}
cd /content/qwen3_apple_style_2bit_qat_lora
python scripts/train_v2.py --config $CONFIG "$@"
TRAIN_EOF
chmod +x "$REPO_DIR/run_train.sh"

echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo "Directory structure:"
echo "  /content/qwen3_apple_style_2bit_qat_lora  - Repo"
echo "  /content/runs                             - Training outputs"
echo "  /content/drive_qat                        - Google Drive (persistent)"
echo "  /content/cache                            - Dataset cache"
echo ""
echo "Next steps:"
echo "  1. Extract checkpoint (if needed):"
echo "     tar -xzf /content/drive_qat/q2_pt_good1.tar.gz -C /content/runs/"
echo ""
echo "  2. Run training:"
echo "     python scripts/train_v2.py --help"
echo "     # or"
echo "     python scripts/train_v2.py --config q2a4 --checkpoint /content/runs/..."
echo ""
