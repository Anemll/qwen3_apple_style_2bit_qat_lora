#!/bin/bash
# Speedrun Setup Script
# Usage: source speedrun/setup.sh [cache] [checkpoint]
#
# Examples:
#   source speedrun/setup.sh                      # Just env setup
#   source speedrun/setup.sh L64                  # Env + L64 cache
#   source speedrun/setup.sh L64 q2_best          # Env + L64 cache + Q2 best checkpoint
#   source speedrun/setup.sh L128 q4_v1           # Env + L128 cache + Q4 from V1

set -e

# =============================================================================
# CONFIG - Auto-detect platform
# =============================================================================

# Detect Google Drive path
if [ -d "/content/drive/MyDrive" ]; then
    # Colab
    GDRIVE_BASE="/content/drive/MyDrive"
    PLATFORM="colab"
elif [ -d "$HOME/Library/CloudStorage/GoogleDrive-realanemll@gmail.com/My Drive" ]; then
    # macOS with Google Drive for Desktop
    GDRIVE_BASE="$HOME/Library/CloudStorage/GoogleDrive-realanemll@gmail.com/My Drive"
    PLATFORM="macos"
else
    echo "[ERROR] Google Drive not found"
    echo "  Colab: Mount with drive.mount('/content/drive')"
    echo "  macOS: Install Google Drive for Desktop"
    return 1
fi

GDRIVE_CACHES="$GDRIVE_BASE/qwen3_caches"
GDRIVE_RUNS="$GDRIVE_BASE/qwen3_runs"

# Cache names (actual files on Drive)
CACHE_L64="alpaca_chat_think_both_L64_K64_R128"
CACHE_L128="alpaca_chat_think_both_L128_K128_R1024"

# Checkpoint mappings (shortname -> path)
declare -A CKPT_MAP
CKPT_MAP["q2_best"]="$GDRIVE_RUNS/Q2A4_BINIT_0.5855/best_state_dict.pt"
CKPT_MAP["q2_0.53"]="$GDRIVE_RUNS/v2_a2_q2_best_fp32_0.5341.tgz"
CKPT_MAP["q4_fp32"]="$GDRIVE_RUNS/anemll_v2_q4_a4_from_v1_finetuned.tgz"
CKPT_MAP["q4_fp16"]="$GDRIVE_RUNS/anemll_v2_q4_a4_ste_fp16_from_v1.tgz"
CKPT_MAP["q2_init"]="$GDRIVE_RUNS/q2_init_from_q4.tgz"

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================
echo "=== Speedrun Setup ==="
echo "[ENV] Platform: $PLATFORM"
echo "[ENV] Drive: $GDRIVE_BASE"

# Colab-specific setup
if [ "$PLATFORM" = "colab" ]; then
    # Install dependencies if needed
    if ! python -c "import transformers" 2>/dev/null; then
        echo "[ENV] Installing dependencies..."
        pip install -q transformers accelerate datasets sentencepiece protobuf wandb
    fi

    # CD to repo
    REPO_DIR="/content/qwen3_apple_style_2bit_qat_lora"
    if [ -d "$REPO_DIR" ]; then
        cd "$REPO_DIR"
    else
        echo "[ENV] ERROR: Repo not found. Run:"
        echo "  git clone https://github.com/Anemll/qwen3_apple_style_2bit_qat_lora.git"
        return 1
    fi
fi

echo "[ENV] Working dir: $(pwd)"

# Create directories
mkdir -p caches runs/speedrun

# =============================================================================
# CACHE SETUP
# =============================================================================
CACHE_ARG="${1:-}"

pull_cache() {
    local cache_name="$1"
    local cache_dir="caches/$cache_name"

    if [ -d "$cache_dir" ] && [ "$(ls -A $cache_dir 2>/dev/null)" ]; then
        local count=$(ls "$cache_dir"/*.pt 2>/dev/null | wc -l)
        echo "[CACHE] $cache_name exists ($count files)"
        return 0
    fi

    # Prefer .tgz (faster single-file transfer)
    local tgz_path="$GDRIVE_CACHES/${cache_name}.tgz"
    if [ -f "$tgz_path" ]; then
        echo "[CACHE] Extracting $cache_name.tgz..."
        mkdir -p "$cache_dir"
        tar -xzf "$tgz_path" -C caches/
        echo "[CACHE] Done"
        return 0
    fi

    # Fall back to directory
    local dir_path="$GDRIVE_CACHES/$cache_name"
    if [ -d "$dir_path" ]; then
        echo "[CACHE] Copying $cache_name (directory - slower)..."
        cp -r "$dir_path" "caches/"
        echo "[CACHE] Done"
        return 0
    fi

    echo "[CACHE] ERROR: Not found: $cache_name"
    echo "[CACHE] Available:"
    ls "$GDRIVE_CACHES" 2>/dev/null | grep -E '\.(tgz|pt)$|^[^.]+$' | head -5
    return 1
}

if [ "$CACHE_ARG" = "L64" ] || [ "$CACHE_ARG" = "l64" ]; then
    pull_cache "$CACHE_L64"
    export CACHE_DIR="caches/$CACHE_L64"
elif [ "$CACHE_ARG" = "L128" ] || [ "$CACHE_ARG" = "l128" ]; then
    pull_cache "$CACHE_L128"
    export CACHE_DIR="caches/$CACHE_L128"
elif [ -n "$CACHE_ARG" ]; then
    pull_cache "$CACHE_ARG"
    export CACHE_DIR="caches/$CACHE_ARG"
fi

# =============================================================================
# CHECKPOINT SETUP
# =============================================================================
CKPT_ARG="${2:-}"

pull_checkpoint() {
    local ckpt_name="$1"
    local dest_dir="runs/speedrun"

    # Check mapping first
    local ckpt_path="${CKPT_MAP[$ckpt_name]}"

    if [ -z "$ckpt_path" ]; then
        # Try direct path
        ckpt_path="$GDRIVE_RUNS/$ckpt_name"
    fi

    # Handle .tgz vs .pt
    if [[ "$ckpt_path" == *.tgz ]]; then
        if [ -f "$ckpt_path" ]; then
            echo "[CKPT] Extracting $(basename $ckpt_path)..."
            tar -xzf "$ckpt_path" -C "$dest_dir/"
            # Find the .pt file (search recursively for subdirectories)
            export CHECKPOINT=$(find "$dest_dir" -name "*.pt" 2>/dev/null | head -1)
            echo "[CKPT] Ready: $CHECKPOINT"
        else
            echo "[CKPT] ERROR: Not found: $ckpt_path"
            return 1
        fi
    elif [ -f "$ckpt_path" ]; then
        echo "[CKPT] Copying $(basename $ckpt_path)..."
        cp "$ckpt_path" "$dest_dir/"
        export CHECKPOINT="$dest_dir/$(basename $ckpt_path)"
        echo "[CKPT] Ready: $CHECKPOINT"
    else
        echo "[CKPT] ERROR: Not found: $ckpt_name"
        echo "[CKPT] Available shortcuts: q2_best, q2_0.53, q4_v1"
        return 1
    fi
}

if [ -n "$CKPT_ARG" ]; then
    pull_checkpoint "$CKPT_ARG"
fi

# =============================================================================
# SUMMARY
# =============================================================================
echo ""
echo "=== Ready ==="
[ -n "$CACHE_DIR" ] && echo "  CACHE_DIR=$CACHE_DIR"
[ -n "$CHECKPOINT" ] && echo "  CHECKPOINT=$CHECKPOINT"
echo ""
echo "Checkpoint shortcuts:"
echo "  q2_init  -> q2_init_from_q4.tgz (Q4→Q2 converted, for MLP training)"
echo "  q2_best  -> Q2A4_BINIT_0.5855/best_state_dict.pt (4.8 GB)"
echo "  q2_0.53  -> v2_a2_q2_best_fp32_0.5341.tgz"
echo "  q4_fp32  -> anemll_v2_q4_a4_from_v1_finetuned.tgz (best for Q4→Q2)"
echo "  q4_fp16  -> anemll_v2_q4_a4_ste_fp16_from_v1.tgz"
echo ""
echo "Example:"
echo "  python scripts/train_v2_simple.py \\"
echo "      --v2-checkpoint \$CHECKPOINT \\"
echo "      --cache-dir \$CACHE_DIR \\"
echo "      --output-dir runs/speedrun \\"
echo "      --mlp-only --max-steps 2000"
