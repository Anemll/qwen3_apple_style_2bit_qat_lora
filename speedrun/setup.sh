#!/bin/bash
# Speedrun Setup Script
# Usage: source speedrun/setup.sh [cache] [checkpoint] [--force]
#
# Examples:
#   source speedrun/setup.sh                      # Just env setup
#   source speedrun/setup.sh L64                  # Env + L64 cache
#   source speedrun/setup.sh L64 q2_best          # Env + L64 cache + Q2 best checkpoint
#   source speedrun/setup.sh L128 q4_fp32         # Env + L128 cache + Q4 from V1
#   source speedrun/setup.sh L64 q4_fp32 --force  # Force re-extract even if exists

set -e

# =============================================================================
# CONFIG - Auto-detect platform
# =============================================================================

# Detect Google Drive path or TPU VM
if [ -d "/content/drive/MyDrive" ]; then
    # Colab
    GDRIVE_BASE="/content/drive/MyDrive"
    PLATFORM="colab"
elif [ -d "$HOME/Library/CloudStorage/GoogleDrive-realanemll@gmail.com/My Drive" ]; then
    # macOS with Google Drive for Desktop
    GDRIVE_BASE="$HOME/Library/CloudStorage/GoogleDrive-realanemll@gmail.com/My Drive"
    PLATFORM="macos"
elif [ -n "$TPU_NAME" ] || [ -f "/etc/tpu_name" ] || python -c "import torch_xla" 2>/dev/null; then
    # TPU VM - use GCS bucket
    PLATFORM="tpu"
    GCS_BUCKET="${GCS_BUCKET:-gs://anemll-tpu-data}"
    GDRIVE_BASE="$GCS_BUCKET"
    echo "[TPU] Using GCS bucket: $GCS_BUCKET"
    echo "[TPU] Set GCS_BUCKET env var to override"
else
    echo "[ERROR] Platform not detected"
    echo "  Colab: Mount with drive.mount('/content/drive')"
    echo "  macOS: Install Google Drive for Desktop"
    echo "  TPU:   Set GCS_BUCKET=gs://your-bucket"
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
CKPT_MAP["q2_init"]="$GDRIVE_RUNS/q2_init_from_q4.tar.lz4"

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

# Local temp dir for fast extraction (Colab SSD)
LOCAL_TMP="/content"

# =============================================================================
# HELPER: Fast archive extraction (rsync to local, extract, cleanup)
# =============================================================================
fast_extract() {
    local src_path="$1"
    local dest_dir="$2"
    local archive_name=$(basename "$src_path")
    local local_archive="/tmp/$archive_name"

    if [ "$PLATFORM" = "tpu" ]; then
        # TPU: download from GCS first
        echo "  Downloading from GCS..."
        gsutil -m cp "$src_path" "$local_archive"

        echo "  Extracting..."
        if [[ "$archive_name" == *.tar.lz4 ]]; then
            which lz4 >/dev/null || sudo apt-get install -qq lz4
            tar -I lz4 -xf "$local_archive" -C "$dest_dir/"
        elif [[ "$archive_name" == *.tgz ]]; then
            tar -xzf "$local_archive" -C "$dest_dir/"
        fi

        # Cleanup
        rm -f "$local_archive"
    elif [ "$PLATFORM" = "colab" ] && [ -d "$LOCAL_TMP" ]; then
        # Colab: rsync to local SSD first (2-3x faster)
        echo "  Copying to local SSD..."
        rsync -ah --progress "$src_path" "$LOCAL_TMP/"

        echo "  Extracting locally..."
        if [[ "$archive_name" == *.tar.lz4 ]]; then
            which lz4 >/dev/null || apt-get install -qq lz4
            tar -I lz4 -xf "$LOCAL_TMP/$archive_name" -C "$dest_dir/"
        elif [[ "$archive_name" == *.tgz ]]; then
            tar -xzf "$LOCAL_TMP/$archive_name" -C "$dest_dir/"
        fi

        # Cleanup local copy
        rm -f "$LOCAL_TMP/$archive_name"
    else
        # macOS or no local tmp: extract directly
        if [[ "$archive_name" == *.tar.lz4 ]]; then
            which lz4 >/dev/null 2>&1 || { echo "lz4 not found"; return 1; }
            tar -I lz4 -xf "$src_path" -C "$dest_dir/"
        elif [[ "$archive_name" == *.tgz ]]; then
            tar -xzf "$src_path" -C "$dest_dir/"
        fi
    fi
}

# =============================================================================
# CACHE SETUP
# =============================================================================
CACHE_ARG="${1:-}"

pull_cache() {
    local cache_name="$1"
    local cache_dir="caches/$cache_name"

    # Check if cache has .pt files (not just directory exists)
    local count=$(ls "$cache_dir"/*.pt 2>/dev/null | wc -l)
    if [ "$count" -gt 0 ]; then
        echo "[CACHE] $cache_name exists ($count files)"
        return 0
    fi

    # Check for nested directory from previous extraction
    local nested_dir="$cache_dir/$cache_name"
    if [ -d "$nested_dir" ]; then
        local nested_count=$(ls "$nested_dir"/*.pt 2>/dev/null | wc -l)
        if [ "$nested_count" -gt 0 ]; then
            echo "[CACHE] Fixing nested structure from previous extraction..."
            mv "$nested_dir"/* "$cache_dir/"
            rmdir "$nested_dir"
            echo "[CACHE] $cache_name ready ($nested_count files)"
            return 0
        fi
    fi

    # Remove empty/broken cache dir
    [ -d "$cache_dir" ] && rm -rf "$cache_dir"
    mkdir -p "$cache_dir"

# Helper to check if file exists (local or GCS)
    file_exists() {
        local path="$1"
        if [[ "$path" == gs://* ]]; then
            gsutil -q stat "$path" 2>/dev/null
        else
            [ -f "$path" ]
        fi
    }

    # Prefer .tar.lz4 (fastest)
    local lz4_path="$GDRIVE_CACHES/${cache_name}.tar.lz4"
    if file_exists "$lz4_path"; then
        echo "[CACHE] Extracting $cache_name.tar.lz4 (lz4 + rsync)..."
        fast_extract "$lz4_path" "caches"
        # Verify extraction worked
        local pt_count=$(ls "$cache_dir"/*.pt 2>/dev/null | wc -l)
        if [ "$pt_count" -gt 0 ]; then
            echo "[CACHE] Done ($pt_count files)"
            return 0
        fi
        # Check for nested directory (lz4 bug: name/name/*.pt)
        local nested_dir="$cache_dir/$cache_name"
        if [ -d "$nested_dir" ]; then
            local nested_count=$(ls "$nested_dir"/*.pt 2>/dev/null | wc -l)
            if [ "$nested_count" -gt 0 ]; then
                echo "[CACHE] Fixing nested structure..."
                mv "$nested_dir"/* "$cache_dir/"
                rmdir "$nested_dir"
                echo "[CACHE] Done ($nested_count files)"
                return 0
            fi
        fi
        echo "[CACHE] WARN: lz4 extraction failed, trying tgz..."
        rm -rf "$cache_dir"
        mkdir -p "$cache_dir"
    fi

    # Fall back to .tgz
    local tgz_path="$GDRIVE_CACHES/${cache_name}.tgz"
    if file_exists "$tgz_path"; then
        echo "[CACHE] Extracting $cache_name.tgz (rsync + extract)..."
        fast_extract "$tgz_path" "caches"
        local pt_count=$(ls "$cache_dir"/*.pt 2>/dev/null | wc -l)
        if [ "$pt_count" -gt 0 ]; then
            echo "[CACHE] Done ($pt_count files)"
            return 0
        fi
        # Check for nested directory (tgz bug: name/name/*.pt)
        local nested_dir="$cache_dir/$cache_name"
        if [ -d "$nested_dir" ]; then
            local nested_count=$(ls "$nested_dir"/*.pt 2>/dev/null | wc -l)
            if [ "$nested_count" -gt 0 ]; then
                echo "[CACHE] Fixing nested structure..."
                mv "$nested_dir"/* "$cache_dir/"
                rmdir "$nested_dir"
                echo "[CACHE] Done ($nested_count files)"
                return 0
            fi
        fi
        echo "[CACHE] Done ($pt_count files)"
        return 0
    fi

    # Fall back to directory
    local dir_path="$GDRIVE_CACHES/$cache_name"
    if [ -d "$dir_path" ]; then
        echo "[CACHE] Copying $cache_name (directory - slower)..."
        rsync -ah --progress "$dir_path/" "$cache_dir/"
        echo "[CACHE] Done"
        return 0
    fi

    echo "[CACHE] ERROR: Not found: $cache_name"
    echo "[CACHE] Available:"
    ls "$GDRIVE_CACHES" 2>/dev/null | grep -E '\.(tar\.lz4|tgz|pt)$|^[^.]+$' | head -5
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
FORCE_FLAG="${3:-}"

pull_checkpoint() {
    local ckpt_name="$1"
    local force="$2"
    local dest_dir="runs/speedrun"

    # Check mapping first
    local ckpt_path="${CKPT_MAP[$ckpt_name]}"

    if [ -z "$ckpt_path" ]; then
        # Try direct path
        ckpt_path="$GDRIVE_RUNS/$ckpt_name"
    fi

    # Check if already extracted (skip unless --force)
    local existing_pt=$(find "$dest_dir" -name "*.pt" 2>/dev/null | head -1)
    if [ -n "$existing_pt" ] && [ "$force" != "--force" ]; then
        echo "[CKPT] Already exists: $existing_pt"
        echo "[CKPT] Use --force to re-extract"
        export CHECKPOINT="$existing_pt"
        return 0
    fi

    # Handle .tar.lz4 vs .tgz vs .pt
    if [[ "$ckpt_path" == *.tar.lz4 ]]; then
        if [ -f "$ckpt_path" ]; then
            echo "[CKPT] Extracting $(basename $ckpt_path) (lz4 + rsync)..."
            fast_extract "$ckpt_path" "$dest_dir"
            export CHECKPOINT=$(find "$dest_dir" -name "*.pt" 2>/dev/null | head -1)
            echo "[CKPT] Ready: $CHECKPOINT"
        else
            echo "[CKPT] ERROR: Not found: $ckpt_path"
            return 1
        fi
    elif [[ "$ckpt_path" == *.tgz ]]; then
        if [ -f "$ckpt_path" ]; then
            echo "[CKPT] Extracting $(basename $ckpt_path) (rsync + extract)..."
            fast_extract "$ckpt_path" "$dest_dir"
            export CHECKPOINT=$(find "$dest_dir" -name "*.pt" 2>/dev/null | head -1)
            echo "[CKPT] Ready: $CHECKPOINT"
        else
            echo "[CKPT] ERROR: Not found: $ckpt_path"
            return 1
        fi
    elif [ -f "$ckpt_path" ]; then
        echo "[CKPT] Copying $(basename $ckpt_path)..."
        rsync -ah --progress "$ckpt_path" "$dest_dir/"
        export CHECKPOINT="$dest_dir/$(basename $ckpt_path)"
        echo "[CKPT] Ready: $CHECKPOINT"
    else
        echo "[CKPT] ERROR: Not found: $ckpt_name"
        echo "[CKPT] Available shortcuts: q2_best, q2_0.53, q4_fp32"
        return 1
    fi
}

if [ -n "$CKPT_ARG" ]; then
    pull_checkpoint "$CKPT_ARG" "$FORCE_FLAG"
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
