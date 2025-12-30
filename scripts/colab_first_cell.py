# ============================================================
# COLAB FIRST CELL - Copy/paste this into your first Colab cell
# ============================================================
# This connects Google Drive and clones the repo.
# After this, everything runs from terminal.

from google.colab import drive
import os

# 1. Mount Google Drive
print("Mounting Google Drive...")
drive.mount('/content/drive')

# 2. Clone repo (or pull if exists)
os.chdir('/content')
if os.path.exists('qwen3_apple_style_2bit_qat_lora'):
    print("Repo exists, pulling latest...")
    !cd qwen3_apple_style_2bit_qat_lora && git fetch && git reset --hard origin/main
else:
    print("Cloning repo...")
    !git clone https://github.com/anemll/qwen3_apple_style_2bit_qat_lora.git

print("\n" + "="*50)
print("Done! Now open terminal and run:")
print("  cd /content/qwen3_apple_style_2bit_qat_lora")
print("  bash scripts/setup_env.sh")
print("="*50)
