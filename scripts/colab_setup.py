#!/usr/bin/env python3
"""
Colab Setup Script - Paste this entire file into first Colab cell and run.

Usage:
    # In first Colab cell:
    !wget -q https://raw.githubusercontent.com/anemll/qwen3_apple_style_2bit_qat_lora/main/scripts/colab_setup.py
    exec(open('colab_setup.py').read())

Or just copy-paste the code below into a cell.
"""

import os
import subprocess
import sys

def run(cmd, check=True):
    """Run shell command and print output."""
    print(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr and result.returncode != 0:
        print(result.stderr)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")
    return result

def setup_colab():
    """Bootstrap Colab environment for Anemll QAT training."""

    print("=" * 50)
    print("Anemll QAT Colab Setup")
    print("=" * 50)

    os.chdir('/content')

    # 1. Clone or update repo
    print("\n[1/5] Setting up repository...")
    if os.path.exists('qwen3_apple_style_2bit_qat_lora'):
        os.chdir('qwen3_apple_style_2bit_qat_lora')
        run('git fetch origin')
        run('git reset --hard origin/main')
        os.chdir('/content')
    else:
        run('git clone https://github.com/anemll/qwen3_apple_style_2bit_qat_lora.git')

    # 2. Install dependencies
    print("\n[2/5] Installing dependencies...")
    run('pip install -q transformers accelerate datasets sentencepiece protobuf')

    # 3. Add to path
    print("\n[3/5] Adding to Python path...")
    repo_path = '/content/qwen3_apple_style_2bit_qat_lora'
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    print(f"  Added: {repo_path}")

    # 4. Create directories
    print("\n[4/5] Creating directories...")
    os.makedirs('/content/runs', exist_ok=True)

    # 5. Extract checkpoint if archive exists
    print("\n[5/5] Looking for checkpoint archive...")
    archives = [f for f in os.listdir('/content') if f.endswith('.tar.gz')]

    if archives:
        archive = archives[0]
        print(f"  Found: {archive}")
        run(f'tar -xzf /content/{archive} -C /content/runs/')

        # List checkpoint files
        print("\n  Checkpoint files:")
        for root, dirs, files in os.walk('/content/runs'):
            for f in files:
                if f.endswith('.pt'):
                    full_path = os.path.join(root, f)
                    size_mb = os.path.getsize(full_path) / 1e6
                    print(f"    {full_path} ({size_mb:.1f} MB)")
    else:
        print("  No .tar.gz archive found in /content/")
        print("  Upload your checkpoint archive and re-run this cell")

    # 6. GPU info
    print("\n" + "=" * 50)
    print("GPU Info:")
    run('nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv', check=False)

    print("\n" + "=" * 50)
    print("Setup Complete!")
    print("=" * 50)
    print("""
Next: Run the imports cell:

    import sys
    sys.path.insert(0, '/content/qwen3_apple_style_2bit_qat_lora')

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from qat_lora import (
        AnemllQuantConfig,
        AnemllQuantConfigV2,
        replace_linear_with_anemll,
        replace_linear_with_anemll_v2,
        freeze_Q_all,
        train_e2e,
        ste_fp16,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
""")

    return True


if __name__ == '__main__' or 'google.colab' in sys.modules:
    setup_colab()
