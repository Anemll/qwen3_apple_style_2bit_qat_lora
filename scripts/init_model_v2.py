#!/usr/bin/env python3
"""
Initialize V2 QAT Model from Scratch

This script creates an optimized, trainable V2 QAT model from a base HuggingFace model.
It's designed for rapid development with clear, modular steps that are easy to improve
and evaluate independently.

PIPELINE OVERVIEW:
==================

  Step 1: Configure         - Set model, quantization, device params
  Step 2: Load Base Model   - Download/load pretrained model (FP32 for accuracy)
  Step 3: Create V2 Configs - MLP & Attention quant configs with constraints
  Step 4: Replace Linears   - Convert nn.Linear -> AnemllQATLinearV2 (SVD init)
  Step 5: Freeze Q          - Lock quantized weight LUTs (trainable: scales only)
  Step 6: Validate          - Inference test + layer count verification
  Step 7: Save              - Export checkpoint + config.json
  Step 8: TightenQ + PPL    - (Optional) Optimize Q and measure perplexity

QUANTIZATION PRESETS:
====================

  q2a4    : 2-bit MLP (LUT4), 4-bit Attn (LUT16), rank=32  [smallest, lowest quality]
  q4a4    : 4-bit MLP (LUT16), 4-bit Attn (LUT16), rank=32 [balanced]
  q4a4_r32: 4-bit MLP (LUT16), 4-bit Attn (LUT16), rank=32 [same as q4a4]
  q4_r64  : 4-bit MLP (LUT16), 4-bit Attn (LUT16), rank=64 [higher quality]

USAGE:
======

  # Create Q4_A4_r32 model (default)
  python scripts/init_model_v2.py --output runs/my_run

  # Create Q2_A4_r32 model (smaller)
  python scripts/init_model_v2.py --config q2a4 --output runs/q2_run

  # Use different base model
  python scripts/init_model_v2.py --model-id Qwen/Qwen3-1.8B --output runs/1.8b_run

  # Skip validation (faster, for testing)
  python scripts/init_model_v2.py --output runs/test --no-validate

  # TPU mode (auto-detected, or force with --tpu)
  python scripts/init_model_v2.py --tpu --output runs/tpu_run

OUTPUT:
=======

  <output_dir>/
    v2_initial.pt           - Initial V2 checkpoint (FP32, trainable)
    config.json             - Quantization config for training scripts
    init_metrics.json       - Initialization metrics (for comparison)

NEXT STEPS:
===========

  After creating the initial model, train with:

    python scripts/train_v2_simple.py \\
        --v2-checkpoint runs/my_run/v2_initial.pt \\
        --cache-dir caches/alpaca_L128 \\
        --mlp-only --auto-snap-mags \\
        --output-dir runs/my_run \\
        --max-steps 1000

IMATRIX (Importance Matrix):
===========================

  The iMatrix captures input activation statistics (σ² = E[x_i²]) for each linear
  layer's input features. This enables importance-weighted LUT selection, which
  places more LUT entries where activation error matters most for PPL.

  CREATING AN IMATRIX:
  -------------------

    # Option 1: compute_imatrix.py (recommended, faster)
    python scripts/compute_imatrix.py \\
        --model Qwen/Qwen3-0.6B \\
        --calib-mode random_ids \\
        --tokens 100000 --seq-len 512 \\
        --out runs/imatrix.pt

    # Option 2: calibrate_activation_stats.py (uses WikiText data)
    python scripts/calibrate_activation_stats.py \\
        --model Qwen/Qwen3-0.6B \\
        --output runs/imatrix.pt \\
        --num-samples 256 --seq-len 512

  USING IMATRIX WITH init_model_v2.py:
  -----------------------------------

    # Use iMatrix for iMSE-weighted LUT search during initialization
    python scripts/init_model_v2.py \\
        --model-id Qwen/Qwen3-0.6B \\
        --output runs/v2_imse \\
        --config q4a4 \\
        --search-lut \\
        --imatrix runs/imatrix.pt

    The --imatrix flag enables iMSE (importance-weighted MSE) scoring when
    --search-lut is used. Without it, standard MSE is used.

  USING IMATRIX WITH select_best_lut_per_layer.py:
  -----------------------------------------------

    # BEST: Use iActMSE metric for per-layer LUT selection
    python scripts/select_best_lut_per_layer.py \\
        runs/v2_initial.pt \\
        -o runs/v2_hybrid.pt \\
        --metric iActMSE \\
        --imatrix runs/imatrix.pt \\
        --families E,G

    iActMSE = Σ σ²[i] * S[o,i]² * (Q_target - Q_quant)²

    This is the theoretically correct metric because it measures expected
    activation error, not just weight reconstruction error.

  OUTPUT FORMAT:
  -------------

    The iMatrix .pt file contains:
    {
      "sigma2": { "<layer_name>": tensor[in_features], ... },  # E[x_i²]
      "count":  { "<layer_name>": int, ... },                  # samples per feature
      "meta":   { ... }                                        # run metadata
    }

AWQ (Activation-aware Weight Quantization):
==========================================

  AWQ protects important channels by applying an EQUIVALENT TRANSFORM:
    - Scale up weight columns for important channels (s > 1)
    - Apply INVERSE scaling to preceding layer (RMSNorm or previous Linear)
    - Net effect: y = (W*s) @ (x/s) = W @ x (function preserved!)

  This concentrates quantization error in less important channels.

  FULL AWQ WORKFLOW (recommended):
  --------------------------------

    # Step 1: Create iMatrix (activation statistics)
    python scripts/compute_imatrix.py \\
        --model Qwen/Qwen3-0.6B \\
        --tokens 100000 --seq-len 512 \\
        --out runs/imatrix.pt

    # Step 2: Apply AWQ-equivalent transforms to FP model
    python scripts/apply_awq_equiv_scales.py \\
        --model-id Qwen/Qwen3-0.6B \\
        --imatrix runs/imatrix.pt \\
        --alpha 0.5 \\
        --output runs/awq_scaled_model

    # Step 3: Initialize V2 from AWQ-scaled model with LUT search
    python scripts/init_model_v2.py \\
        --model-id runs/awq_scaled_model \\
        --output runs/v2_awq \\
        --config q4a4_r32 \\
        --search-lut \\
        --imatrix runs/imatrix.pt

    # Step 4: Select best LUT per layer (E=baseline, G=k-means)
    python scripts/select_best_lut_per_layer.py runs/v2_awq/v2_initial.pt \\
        -o runs/v2_hybrid/hybrid.pt \\
        --workers 8 \\
        --metric iActMSE \\
        --imatrix runs/imatrix.pt \\
        --families E,G

    # Step 5: Measure perplexity
    python scripts/quick_perplexity.py runs/v2_hybrid/hybrid.pt --max-chunks 20

  This properly applies AWQ with the required inverse compensation:
    - input_layernorm → q/k/v_proj (RMSNorm scales)
    - v_proj → o_proj (Linear→Linear with GQA handling)
    - post_attention_layernorm → gate/up_proj
    - up_proj → down_proj

  iActMSE METRIC:
    score = Σ_{o,i} σ²[i] * S[o,i]² * (Q_target - Q_quant)²

    Where σ² is input activation variance from iMatrix. This measures
    expected squared output error, weighting by channel importance.

  NOTE: The --awq-alpha flag is DEPRECATED and does NOT work correctly
        because it lacks the inverse compensation. Use the proper workflow above.

Author: ANEMLL Team
"""

import argparse
import gc
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import torch
import torch.nn as nn

# Add repo root to path
REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))


def _load_tokenizer_safe(model_id: str, trust_remote_code: bool = True):
    """
    Load tokenizer with mistral-regex fix when supported.

    Some transformers versions expose `fix_mistral_regex`; older versions do not.
    """
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            use_fast=False,
            fix_mistral_regex=True,
        )
    except TypeError:
        # Fallback for older transformers versions.
        return AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            use_fast=False,
        )


def _load_causal_lm_safe(
    model_id: str,
    dtype: torch.dtype,
    trust_remote_code: bool = True,
):
    """
    Load causal LM preferring modern `dtype=` with backward fallback.
    """
    from transformers import AutoModelForCausalLM

    try:
        return AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
    except TypeError as e:
        # Only fallback when this transformers version does not recognize `dtype=`.
        msg = str(e)
        if "dtype" not in msg and "unexpected keyword argument" not in msg:
            raise
        return AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
        )


def _format_bytes(n: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    x = float(max(0, n))
    for unit in units:
        if x < 1024.0 or unit == units[-1]:
            return f"{x:.1f}{unit}"
        x /= 1024.0
    return f"{n}B"


def _estimate_state_dict_bytes(state_dict: Dict[str, Any]) -> int:
    total = 0
    for value in state_dict.values():
        if torch.is_tensor(value):
            total += value.numel() * value.element_size()
    return int(total)


def _ensure_free_space(path: Path, required_bytes: int, reserve_bytes: int = 512 * 1024 * 1024) -> None:
    """
    Check available space before writing a large checkpoint.
    """
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(parent)
    available = int(usage.free)
    needed = int(required_bytes + reserve_bytes)
    if available < needed:
        raise RuntimeError(
            "Insufficient disk space for checkpoint save: "
            f"available={_format_bytes(available)}, "
            f"required~{_format_bytes(required_bytes)} + reserve={_format_bytes(reserve_bytes)}. "
            f"Free space on '{parent}' and retry."
        )


def _safe_torch_save(obj: Any, path: Path, required_bytes_hint: Optional[int] = None) -> None:
    """
    Atomic save with cleanup and a legacy-serialization fallback.
    """
    tmp_path = path.with_name(path.name + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    if required_bytes_hint is not None:
        _ensure_free_space(path, required_bytes_hint)

    try:
        torch.save(obj, tmp_path)
    except RuntimeError as e:
        msg = str(e).lower()
        if tmp_path.exists():
            tmp_path.unlink()
        # PyTorch zip writer can fail late; retry with legacy serializer.
        if "pytorchstreamwriter" in msg or "inline_container" in msg or "file write failed" in msg:
            if required_bytes_hint is not None:
                _ensure_free_space(path, required_bytes_hint)
            torch.save(obj, tmp_path, _use_new_zipfile_serialization=False)
        else:
            raise

    os.replace(tmp_path, path)


def _create_checkpoint_alias(target_path: Path, alias_path: Path) -> str:
    """
    Create alias path for compatibility without duplicating multi-GB checkpoint data.
    Returns alias type: hardlink/symlink/copy.
    """
    if alias_path.exists() or alias_path.is_symlink():
        alias_path.unlink()

    try:
        os.link(target_path, alias_path)
        return "hardlink"
    except OSError:
        pass

    try:
        # Relative symlink keeps the directory relocatable.
        os.symlink(target_path.name, alias_path)
        return "symlink"
    except OSError:
        shutil.copy2(target_path, alias_path)
        return "copy"


def _release_memory(device: Optional[torch.device] = None) -> None:
    """
    Best-effort memory release for long-running init flows.
    """
    gc.collect()

    try:
        dev_type = (device.type if isinstance(device, torch.device) else str(device or "")).lower()
    except Exception:
        dev_type = ""

    # Clear CUDA allocator cache when relevant.
    if dev_type in ("", "cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Clear MPS cache if available.
    if dev_type in ("", "mps") and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


# =============================================================================
# LUT CANDIDATES (FP16-snapped)
# =============================================================================

def build_fp16_lut(target_values: List[float]) -> torch.Tensor:
    """
    Build a monotonically increasing FP16 LUT with distinct values.

    Takes target values and ensures:
    1. All values are exactly representable in FP16
    2. All values are distinct (no duplicates)
    3. Values are monotonically increasing

    If snapping creates duplicates, increments to next distinct FP16 value.

    Args:
        target_values: Desired LUT values (will be adjusted to valid FP16)

    Returns:
        Tensor of distinct, monotonic FP16 values
    """
    import struct

    def float_to_fp16_bits(f: float) -> int:
        """Convert float to FP16 bit representation."""
        return struct.unpack('H', struct.pack('e', f))[0]

    def fp16_bits_to_float(bits: int) -> float:
        """Convert FP16 bits back to float."""
        return struct.unpack('e', struct.pack('H', bits))[0]

    def next_fp16(f: float) -> float:
        """Get the next larger FP16 value."""
        bits = float_to_fp16_bits(f)
        if f >= 0:
            return fp16_bits_to_float(bits + 1)
        else:
            return fp16_bits_to_float(bits - 1)

    def snap_to_fp16(f: float) -> float:
        """Snap to nearest FP16 value."""
        return float(torch.tensor(f, dtype=torch.float32).to(torch.float16).item())

    # Sort target values
    targets = sorted(target_values)

    # Build LUT with distinct FP16 values
    result = []
    prev_fp16 = float('-inf')

    for target in targets:
        # Snap to FP16
        fp16_val = snap_to_fp16(target)

        # Ensure monotonically increasing and distinct
        while fp16_val <= prev_fp16:
            fp16_val = next_fp16(fp16_val)

        result.append(fp16_val)
        prev_fp16 = fp16_val

    return torch.tensor(result, dtype=torch.float32)


def make_lut_candidates(lut_size: int = 16) -> Dict[str, torch.Tensor]:
    """
    Generate FP16 LUT candidates for per-tensor LUT search.

    All LUTs have exactly lut_size distinct, monotonically increasing FP16 values.

    Args:
        lut_size: Number of LUT entries (default: 16 for 4-bit)

    Returns:
        Dict mapping LUT name to FP16 tensor of values
    """
    import numpy as np

    # Helper to create symmetric LUT from positive half
    def symmetric_lut(positive_values: List[float]) -> torch.Tensor:
        """Create symmetric LUT: [-pos_n, ..., -pos_1, pos_1, ..., pos_n]"""
        n = len(positive_values)
        if n * 2 != lut_size:
            raise ValueError(f"Need {lut_size // 2} positive values for {lut_size}-entry LUT")
        negative = [-v for v in reversed(positive_values)]
        return build_fp16_lut(negative + positive_values)

    half = lut_size // 2  # 8 positive values for 16-entry LUT

    # Generate positive half values for different distributions
    y = np.linspace(0, 1, half + 1)[1:]  # [0.125, 0.25, ..., 1.0] for half=8

    candidates = {
        # === Standard [-1, 1] LUTs ===

        # === FP4-ish LUTs with outlier bins ===

        # 7 uniform in (0,1] + 1 outlier at 1.5
        #'fp4_soft': symmetric_lut([0.143, 0.286, 0.429, 0.571, 0.714, 0.857, 1.0, 1.5]),

        # 7 uniform in (0,1] + 1 outlier at 2.0
        #'fp4_med': symmetric_lut([0.143, 0.286, 0.429, 0.571, 0.714, 0.857, 1.0, 2.0]),

        # 6 uniform in (0,1] + 2 outliers at 1.5, 3.0
        #'fp4_strong': symmetric_lut([0.167, 0.333, 0.5, 0.667, 0.833, 1.0, 1.5, 3.0]),


        # Uniform: linspace - good default
        'uniform': symmetric_lut(y.tolist()),


        # Dense near 0 (power-law spacing) + 1 outlier at 2.0
        'fp4_dense': symmetric_lut([0.0625, 0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 2.0]),
   

        # Power-2 (quadratic): more resolution near 0
        #'power2': symmetric_lut((y ** 2.0).tolist()),

        # Power-3 (cubic): much more resolution near 0
        #'power3': symmetric_lut((y ** 3.0).tolist()),

        # Inverse μ-law (μ=50): dense near 0
        #'inv_mu50': symmetric_lut((np.expm1(y * np.log1p(50)) / 50).tolist()),


   
    }

    return candidates


# Default LUT candidates for LUT16 (4-bit) search
LUT16_CANDIDATES = None  # Lazily initialized


def get_lut16_candidates() -> Dict[str, torch.Tensor]:
    """Get or create the default LUT16 candidates."""
    global LUT16_CANDIDATES
    if LUT16_CANDIDATES is None:
        LUT16_CANDIDATES = make_lut_candidates(lut_size=16)
    return LUT16_CANDIDATES


# =============================================================================
# STEP 1: CONFIGURATION
# =============================================================================

@dataclass
class QuantPreset:
    """Quantization preset configuration."""
    name: str
    mlp_lut_bits: int
    mlp_rank: int
    attn_lut_bits: int
    attn_rank: int
    description: str


# Available presets
PRESETS: Dict[str, QuantPreset] = {
    'q2a4': QuantPreset(
        name='q2a4',
        mlp_lut_bits=2,
        mlp_rank=32,
        attn_lut_bits=4,
        attn_rank=8,
        description='2-bit MLP (LUT4), 4-bit Attention (LUT16), rank=32/8 [smallest]'
    ),
    'q2a2': QuantPreset(
        name='q2a2',
        mlp_lut_bits=2,
        mlp_rank=32,
        attn_lut_bits=2,
        attn_rank=8,
        description='2-bit MLP+Attn (LUT4), rank=32/8 [extreme compression]'
    ),
    'q4a4': QuantPreset(
        name='q4a4',
        mlp_lut_bits=4,
        mlp_rank=32,
        attn_lut_bits=4,
        attn_rank=32,
        description='4-bit MLP+Attn (LUT16), rank=32 [balanced]'
    ),
    'q4a4_r32': QuantPreset(
        name='q4a4_r32',
        mlp_lut_bits=4,
        mlp_rank=32,
        attn_lut_bits=4,
        attn_rank=32,
        description='4-bit MLP+Attn (LUT16), rank=32 [same as q4a4]'
    ),
    'q4_r64': QuantPreset(
        name='q4_r64',
        mlp_lut_bits=4,
        mlp_rank=64,
        attn_lut_bits=4,
        attn_rank=64,
        description='4-bit MLP+Attn (LUT16), rank=64 [higher quality]'
    ),
}


def get_device(force_tpu: bool = False, force_cpu: bool = False) -> Tuple[torch.device, str]:
    """
    Detect and return the best available device.

    Priority: TPU > CUDA > MPS > CPU

    Returns:
        (device, device_type) tuple
    """
    if force_cpu:
        return torch.device('cpu'), 'cpu'

    if force_tpu:
        try:
            import torch_xla.core.xla_model as xm
            return xm.xla_device(), 'tpu'
        except ImportError:
            print("[WARN] --tpu requested but torch_xla not available")

    # Auto-detect
    try:
        import torch_xla.core.xla_model as xm
        return xm.xla_device(), 'tpu'
    except ImportError:
        pass

    if torch.cuda.is_available():
        return torch.device('cuda'), 'cuda'

    if torch.backends.mps.is_available():
        return torch.device('mps'), 'mps'

    return torch.device('cpu'), 'cpu'


# =============================================================================
# STEP 2: LOAD BASE MODEL
# =============================================================================

def load_base_model(
    model_id: str,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    verbose: bool = True,
) -> Tuple[nn.Module, Any]:
    """
    Load the base HuggingFace model.

    IMPORTANT: Use FP32 for initialization to ensure SVD accuracy.
    Training can later use BF16/FP16 for speed.

    Args:
        model_id: HuggingFace model ID (e.g., 'Qwen/Qwen3-0.6B')
        device: Target device
        dtype: Model dtype (FP32 recommended for init)
        verbose: Print status messages

    Returns:
        (model, tokenizer) tuple
    """
    if verbose:
        print(f"\n[Step 2] Loading base model: {model_id}")
        print(f"  dtype: {dtype}")

    t0 = time.time()

    tokenizer = _load_tokenizer_safe(model_id, trust_remote_code=True)
    model = _load_causal_lm_safe(model_id, dtype=dtype, trust_remote_code=True)

    # Count parameters before moving to device
    total_params = sum(p.numel() for p in model.parameters())

    # Move to device
    model.to(device)
    model.eval()

    elapsed = time.time() - t0

    if verbose:
        print(f"  Parameters: {total_params:,} ({total_params / 1e6:.1f}M)")
        print(f"  Load time: {elapsed:.1f}s")

    return model, tokenizer


# =============================================================================
# STEP 3: CREATE V2 CONFIGS
# =============================================================================

def create_v2_configs(
    preset: QuantPreset,
    group_size: int = 32,
    force_positive_scales: bool = False,
    magnitude_activation: str = "identity",
    verbose: bool = True,
) -> Tuple[Any, Any]:
    """
    Create V2 quantization configs for MLP and Attention layers.

    KEY DESIGN DECISIONS:
    - force_positive_scales=False: SVD produces signed U,Vh; abs() would break approximation
    - magnitude_activation='identity': SVD singular values are already positive, no transform needed
    - learnable_lut=False: LUT values are fixed after initialization
    - group_size=32: Group size for SVD scale initialization

    NOTE: The SVD initialization produces:
      - scale_A = U[:, :r]  (can have negative entries)
      - scale_B = Vh[:r, :] (can have negative entries)
      - rank_magnitude = S[:r] (always positive - SVD property)

    Using force_positive_scales=True with abs() would destroy the signed U,Vh entries,
    breaking the SVD approximation. Using softplus on already-positive S values
    would distort them (softplus(1.0) = 1.31, not 1.0).

    Args:
        preset: Quantization preset
        group_size: Group size for scale initialization (default: 32)
        force_positive_scales: Use positive scale constraints (default: False for SVD compat)
        magnitude_activation: How to transform rank_magnitude (default: 'identity' for SVD compat)
        verbose: Print config details

    Returns:
        (mlp_config, attn_config) tuple
    """
    from qat_lora.ane_qat_linear_v2 import AnemllQuantConfigV2

    if verbose:
        print(f"\n[Step 3] Creating V2 configs: {preset.name}")
        print(f"  {preset.description}")

    mlp_lut_size = 2 ** preset.mlp_lut_bits
    attn_lut_size = 2 ** preset.attn_lut_bits

    # MLP config
    # NOTE: force_positive_scales=False and magnitude_activation='identity'
    # are required for compatibility with SVD initialization
    mlp_config = AnemllQuantConfigV2(
        lut_size=mlp_lut_size,
        scale_rank=preset.mlp_rank,
        group_size=group_size,
        learnable_lut=False,
        # SVD-compatible settings (don't transform the decomposition)
        force_positive_scales=force_positive_scales,
        positive_scale_method="abs",
        magnitude_activation=magnitude_activation,
        magnitude_eps=0.0,  # No eps needed with identity
    )

    # Attention config
    attn_config = AnemllQuantConfigV2(
        lut_size=attn_lut_size,
        scale_rank=preset.attn_rank,
        group_size=group_size,
        learnable_lut=False,
        force_positive_scales=force_positive_scales,
        positive_scale_method="abs",
        magnitude_activation=magnitude_activation,
        magnitude_eps=0.0,
    )

    if verbose:
        print(f"  MLP:  LUT{mlp_lut_size} ({preset.mlp_lut_bits}-bit), rank={preset.mlp_rank}")
        print(f"  Attn: LUT{attn_lut_size} ({preset.attn_lut_bits}-bit), rank={preset.attn_rank}")
        print(f"  Group size: {group_size}")
        print(f"  Scale config: force_positive_scales={force_positive_scales}, magnitude_activation={magnitude_activation}")

    return mlp_config, attn_config


# =============================================================================
# STEP 4: REPLACE LINEAR LAYERS
# =============================================================================

def replace_linear_layers(
    model: nn.Module,
    mlp_config: Any,
    attn_config: Any,
    quantize_attn: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Replace nn.Linear layers with AnemllQATLinearV2.

    This step performs SVD-based initialization of scale parameters,
    which is computationally expensive but critical for quality.

    WHAT HAPPENS:
    1. Each nn.Linear is replaced with AnemllQATLinearV2
    2. Weight is copied and quantized to LUT indices
    3. Scale parameters (A, B, rank_magnitude) are initialized via SVD
    4. LUT values are computed from weight distribution

    Args:
        model: Base model with nn.Linear layers
        mlp_config: Config for MLP layers
        attn_config: Config for attention layers
        quantize_attn: Whether to quantize attention layers
        verbose: Print replacement progress

    Returns:
        Dict with replacement stats
    """
    from qat_lora.ane_qat_linear_v2 import replace_linear_with_anemll_v2

    if verbose:
        print(f"\n[Step 4] Replacing Linear layers with AnemllQATLinearV2")
        print(f"  quantize_attn: {quantize_attn}")

    t0 = time.time()

    # Count layers before
    linear_before = sum(1 for m in model.modules() if isinstance(m, nn.Linear))

    # Replace
    replaced_count = replace_linear_with_anemll_v2(
        model,
        mlp_config=mlp_config,
        attn_config=attn_config,
        quantize_attn=quantize_attn,
        quantize_lm_head=False,  # Never quantize lm_head
        verbose=verbose,
    )

    elapsed = time.time() - t0

    # Count layers after
    from qat_lora.ane_qat_linear_v2 import AnemllQATLinearV2
    v2_count = sum(1 for m in model.modules() if isinstance(m, AnemllQATLinearV2))
    linear_after = sum(1 for m in model.modules() if isinstance(m, nn.Linear) and not isinstance(m, AnemllQATLinearV2))

    stats = {
        'replaced_count': replaced_count,
        'v2_layers': v2_count,
        'remaining_linear': linear_after,
        'time_seconds': elapsed,
    }

    if verbose:
        print(f"  Replaced: {replaced_count} layers")
        print(f"  V2 layers: {v2_count}")
        print(f"  Remaining Linear: {linear_after} (embed_tokens, lm_head)")
        print(f"  SVD init time: {elapsed:.1f}s")

    return stats


# =============================================================================
# STEP 4b: MEASURE SVD APPROXIMATION ERROR
# =============================================================================

@torch.no_grad()
def measure_svd_approximation_error(
    model: nn.Module,
    model_id: str,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Calculate MAE between original weights and SVD-approximated weights.

    For each V2 layer, computes:
        W_ref = original HuggingFace weights
        W_eff = _Q * S (where S = _compute_full_scales())
        MAE = mean(|W_ref - W_eff|)

    This measures how well the SVD initialization approximates the original weights.

    Args:
        model: Model with AnemllQATLinearV2 layers (after SVD init)
        model_id: HuggingFace model ID for loading baseline weights
        verbose: Print progress

    Returns:
        Dict with per-layer and aggregate MAE statistics
    """
    from qat_lora.ane_qat_linear_v2 import AnemllQATLinearV2

    if verbose:
        print(f"\n[Step 4b] Measuring SVD approximation error")

    t0 = time.time()

    # Load baseline weights
    if verbose:
        print(f"  Loading baseline weights from {model_id}...")

    baseline = _load_causal_lm_safe(
        model_id,
        dtype=torch.float32,
        trust_remote_code=True,
    )

    # Create W_ref map
    W_ref_map = {}
    for name, module in baseline.named_modules():
        if hasattr(module, 'weight') and isinstance(module.weight, nn.Parameter):
            W_ref_map[name] = module.weight.data.clone()

    del baseline
    _release_memory()

    if verbose:
        print(f"  Loaded {len(W_ref_map)} baseline weight tensors")
        print(f"  Computing MAE for each V2 layer...")

    # Compute MAE for each V2 layer
    layer_stats = []
    mlp_maes = []
    attn_maes = []

    v2_layers = [(name, m) for name, m in model.named_modules() if isinstance(m, AnemllQATLinearV2)]

    for layer_idx, (name, module) in enumerate(v2_layers):
        # Get W_ref
        W_ref = W_ref_map.get(name)
        if W_ref is None:
            continue

        # Move W_ref to module device
        device = next(module.parameters()).device
        W_ref = W_ref.to(device)

        # Get effective weight: W_eff = Q * S
        # Note: For unfrozen layers, we need to snap Q first
        # For frozen layers (_Q is not None), we can compute Q * S directly
        if module._Q is None:
            # Freeze Q to populate _Q buffer
            module.freeze_Q()

        Q = module._Q.view(module.out_features, module.in_features)
        S = module._compute_full_scales()
        W_eff = Q * S

        # Ensure shapes match
        if W_eff.shape != W_ref.shape:
            continue

        # Calculate MAE
        mae = (W_ref - W_eff).abs().mean().item()

        # Calculate relative MAE (normalized by weight magnitude)
        w_scale = W_ref.abs().mean().item()
        rel_mae = mae / max(w_scale, 1e-8)

        # Determine layer type
        is_mlp = 'mlp' in name
        is_attn = 'self_attn' in name or 'attention' in name

        layer_stat = {
            'name': name,
            'mae': mae,
            'rel_mae': rel_mae,
            'w_scale': w_scale,
            'shape': list(W_ref.shape),
            'type': 'mlp' if is_mlp else ('attn' if is_attn else 'other'),
        }
        layer_stats.append(layer_stat)

        if is_mlp:
            mlp_maes.append(mae)
        elif is_attn:
            attn_maes.append(mae)

        # Print progress for first few and worst layers
        if verbose and layer_idx < 5:
            short_name = name.split('.')[-2] + '.' + name.split('.')[-1] if '.' in name else name
            print(f"    {short_name}: MAE={mae:.6f} (rel={rel_mae:.4f})")

        # Avoid retaining large temporary tensors between layers.
        del W_ref, Q, S, W_eff

    # Aggregate statistics
    all_maes = [s['mae'] for s in layer_stats]
    all_rel_maes = [s['rel_mae'] for s in layer_stats]

    stats = {
        'num_layers': len(layer_stats),
        'avg_mae': sum(all_maes) / len(all_maes) if all_maes else 0,
        'max_mae': max(all_maes) if all_maes else 0,
        'min_mae': min(all_maes) if all_maes else 0,
        'avg_rel_mae': sum(all_rel_maes) / len(all_rel_maes) if all_rel_maes else 0,
        'mlp_avg_mae': sum(mlp_maes) / len(mlp_maes) if mlp_maes else 0,
        'attn_avg_mae': sum(attn_maes) / len(attn_maes) if attn_maes else 0,
        'layer_stats': layer_stats,
        'time_seconds': time.time() - t0,
    }

    # Find worst tensors
    sorted_by_mae = sorted(layer_stats, key=lambda x: x['mae'], reverse=True)
    stats['worst_layers'] = sorted_by_mae[:10]

    if verbose:
        print(f"\n  SVD Approximation Error Summary:")
        print(f"    Tensors measured: {stats['num_layers']}")
        print(f"    Average MAE:      {stats['avg_mae']:.6f}")
        print(f"    Max MAE:          {stats['max_mae']:.6f}")
        print(f"    Avg Relative MAE: {stats['avg_rel_mae']:.4f} ({stats['avg_rel_mae']*100:.2f}%)")
        if mlp_maes:
            print(f"    MLP avg MAE:      {stats['mlp_avg_mae']:.6f}")
        if attn_maes:
            print(f"    Attn avg MAE:     {stats['attn_avg_mae']:.6f}")
        print(f"\n  Worst 10 tensors by MAE:")
        for i, layer in enumerate(stats['worst_layers'][:10]):
            # Show layer number: layers.N.mlp.gate_proj (last 4 parts)
            parts = layer['name'].split('.')
            short_name = '.'.join(parts[-4:]) if len(parts) >= 4 else layer['name']
            print(f"    {i+1:2d}. {short_name}: MAE={layer['mae']:.6f}")
        print(f"  Time: {stats['time_seconds']:.1f}s")

    # Free baseline map before returning potentially large stats payload.
    W_ref_map.clear()
    del W_ref_map
    _release_memory()

    return stats


# =============================================================================
# STEP 4c: SEARCH OPTIMAL GROUP SIZE PER TENSOR
# =============================================================================

@torch.no_grad()
def search_optimal_group_sizes(
    model: nn.Module,
    model_id: str,
    group_sizes: list,
    mlp_lut_bits: int = 4,
    mlp_scale_rank: int = 32,
    attn_lut_bits: int = 4,
    attn_scale_rank: int = 32,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Search for optimal group_size for each tensor by testing different sizes
    and selecting the one that minimizes reconstruction MAE.

    For each linear layer, tests each group_size by:
    1. Computing SVD-based scale initialization
    2. Quantizing to LUT
    3. Measuring reconstruction MAE = |W_ref - Q * S|
    4. Selecting group_size with minimum MAE

    Args:
        model: Base model with nn.Linear layers (BEFORE V2 replacement)
        model_id: HuggingFace model ID (for reference, model already loaded)
        group_sizes: List of group sizes to test (e.g., [128, 64, 32, 16])
        mlp_lut_bits: LUT bits for MLP layers (default: 4 = 16 values)
        mlp_scale_rank: Scale rank for MLP layers (default: 32)
        attn_lut_bits: LUT bits for Attention layers (default: 4 = 16 values)
        attn_scale_rank: Scale rank for Attention layers (default: 32)
        verbose: Print progress

    Returns:
        Dict with per-layer optimal group sizes and statistics
    """
    from qat_lora.ane_qat_linear_v2 import AnemllQuantConfigV2, AnemllQATLinearV2

    if verbose:
        print(f"\n[Step 4c] Searching optimal group sizes")
        print(f"  Testing group sizes: {group_sizes}")
        print(f"  MLP: LUT{2**mlp_lut_bits} ({mlp_lut_bits}-bit), rank={mlp_scale_rank}")
        print(f"  Attn: LUT{2**attn_lut_bits} ({attn_lut_bits}-bit), rank={attn_scale_rank}")

    t0 = time.time()

    # Collect all linear layers that would be quantized (MLP + Attn, not lm_head/embeddings)
    linear_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Skip lm_head and embeddings
            if 'lm_head' in name or 'embed' in name:
                continue
            # Only include MLP and attention layers
            if 'mlp' in name or 'self_attn' in name or 'attention' in name:
                linear_layers.append((name, module))

    if verbose:
        print(f"  Found {len(linear_layers)} linear layers to analyze")

    # Results storage
    layer_results = []
    group_size_counts = {gs: 0 for gs in group_sizes}
    mlp_group_counts = {gs: 0 for gs in group_sizes}
    attn_group_counts = {gs: 0 for gs in group_sizes}

    for layer_idx, (name, linear_module) in enumerate(linear_layers):
        # Get device from original module
        device = linear_module.weight.device
        W_ref = linear_module.weight.data.clone()
        out_features, in_features = W_ref.shape

        # Determine layer type and select appropriate config
        is_mlp = 'mlp' in name
        is_attn = 'self_attn' in name or 'attention' in name
        layer_type = 'mlp' if is_mlp else ('attn' if is_attn else 'other')

        # Use appropriate LUT size and scale rank for this layer type
        if is_mlp:
            lut_size = 2 ** mlp_lut_bits
            scale_rank = mlp_scale_rank
        else:  # Attention
            lut_size = 2 ** attn_lut_bits
            scale_rank = attn_scale_rank

        # Test each group_size
        best_group_size = group_sizes[0]
        best_mae = float('inf')
        group_maes = {}

        for gs in group_sizes:
            # Create temporary config with this group_size
            # NOTE: Use identity for SVD compatibility (same as create_v2_configs)
            temp_config = AnemllQuantConfigV2(
                lut_size=lut_size,
                scale_rank=scale_rank,
                group_size=gs,
                force_positive_scales=False,
                positive_scale_method="abs",
                magnitude_activation="identity",
                magnitude_eps=0.0,
            )

            # Create temporary V2 layer to get SVD initialization
            try:
                temp_v2 = AnemllQATLinearV2.from_linear(
                    linear_module,
                    config=temp_config,
                    skip_init=False,  # Do full SVD init
                )

                # Ensure all tensors are on the same device
                temp_v2.to(device)

                # Freeze Q to populate _Q buffer (needed to compute effective weight)
                temp_v2.freeze_Q()

                # Get effective weight: W_eff = Q * S
                Q = temp_v2._Q.view(temp_v2.out_features, temp_v2.in_features)
                S = temp_v2._compute_full_scales()
                W_eff = Q * S

                # Compute MAE
                mae = (W_ref - W_eff).abs().mean().item()
                group_maes[gs] = mae

                if mae < best_mae:
                    best_mae = mae
                    best_group_size = gs

                # Clean up
                del Q, S, W_eff
                del temp_v2

            except Exception as e:
                if verbose:
                    print(f"    [WARN] group_size={gs} failed for {name}: {e}")
                group_maes[gs] = float('inf')

        # Release per-layer temporaries before moving to the next layer.
        del W_ref
        if (layer_idx + 1) % 8 == 0:
            _release_memory(device)

        layer_result = {
            'name': name,
            'shape': [out_features, in_features],
            'type': layer_type,
            'optimal_group_size': best_group_size,
            'optimal_mae': best_mae,
            'all_maes': group_maes,
        }
        layer_results.append(layer_result)

        # Update counts
        group_size_counts[best_group_size] += 1
        if is_mlp:
            mlp_group_counts[best_group_size] += 1
        elif is_attn:
            attn_group_counts[best_group_size] += 1

        # Progress
        if verbose and (layer_idx % 20 == 0 or layer_idx == len(linear_layers) - 1):
            short_name = name.split('.')[-2] + '.' + name.split('.')[-1] if '.' in name else name
            mae_str = ', '.join([f"g{gs}={m:.6f}" for gs, m in sorted(group_maes.items())])
            print(f"    [{layer_idx+1}/{len(linear_layers)}] {short_name}: best=g{best_group_size} ({mae_str})")

    elapsed = time.time() - t0

    # Build optimal group size map
    optimal_group_map = {r['name']: r['optimal_group_size'] for r in layer_results}

    # Aggregate statistics
    stats = {
        'num_layers': len(layer_results),
        'group_sizes_tested': group_sizes,
        'group_size_counts': group_size_counts,
        'mlp_group_counts': mlp_group_counts,
        'attn_group_counts': attn_group_counts,
        'optimal_group_map': optimal_group_map,
        'layer_results': layer_results,
        'time_seconds': elapsed,
    }

    if verbose:
        print(f"\n  Group Size Search Results:")
        print(f"    Layers analyzed: {stats['num_layers']}")
        print(f"    Time: {elapsed:.1f}s")
        print(f"\n  Optimal Group Size Distribution:")
        for gs in sorted(group_sizes):
            count = group_size_counts[gs]
            pct = 100 * count / max(1, len(layer_results))
            mlp_count = mlp_group_counts[gs]
            attn_count = attn_group_counts[gs]
            print(f"    group_size={gs:3d}: {count:3d} layers ({pct:5.1f}%) - MLP: {mlp_count}, Attn: {attn_count}")

        # Show breakdown by layer type
        print(f"\n  MLP layers group distribution:")
        mlp_total = sum(mlp_group_counts.values())
        for gs in sorted(group_sizes):
            if mlp_group_counts[gs] > 0:
                pct = 100 * mlp_group_counts[gs] / max(1, mlp_total)
                print(f"    g{gs}: {mlp_group_counts[gs]} ({pct:.1f}%)")

        print(f"\n  Attention layers group distribution:")
        attn_total = sum(attn_group_counts.values())
        for gs in sorted(group_sizes):
            if attn_group_counts[gs] > 0:
                pct = 100 * attn_group_counts[gs] / max(1, attn_total)
                print(f"    g{gs}: {attn_group_counts[gs]} ({pct:.1f}%)")

    return stats


def replace_linear_layers_with_optimal_groups(
    model: nn.Module,
    optimal_group_map: Dict[str, int],
    mlp_lut_bits: int = 4,
    mlp_scale_rank: int = 32,
    attn_lut_bits: int = 4,
    attn_scale_rank: int = 32,
    quantize_attn: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Replace linear layers using per-layer optimal group sizes.

    Args:
        model: Base model with nn.Linear layers
        optimal_group_map: Dict mapping layer name -> optimal group_size
        mlp_lut_bits: LUT bits for MLP layers
        mlp_scale_rank: Scale rank for MLP layers
        attn_lut_bits: LUT bits for Attention layers
        attn_scale_rank: Scale rank for Attention layers
        quantize_attn: Whether to quantize attention layers
        verbose: Print progress

    Returns:
        Dict with replacement stats
    """
    from qat_lora.ane_qat_linear_v2 import AnemllQuantConfigV2, AnemllQATLinearV2

    if verbose:
        print(f"\n[Step 4d] Replacing layers with optimal group sizes")
        print(f"  MLP: LUT{2**mlp_lut_bits} ({mlp_lut_bits}-bit), rank={mlp_scale_rank}")
        print(f"  Attn: LUT{2**attn_lut_bits} ({attn_lut_bits}-bit), rank={attn_scale_rank}")

    t0 = time.time()

    # Collect layers to replace
    layers_to_replace = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if 'lm_head' in name or 'embed' in name:
                continue
            is_mlp = 'mlp' in name
            is_attn = 'self_attn' in name or 'attention' in name
            if is_mlp or (quantize_attn and is_attn):
                layers_to_replace.append((name, module))

    if verbose:
        print(f"  Replacing {len(layers_to_replace)} layers")

    replaced_count = 0
    group_size_used = {}

    for layer_idx, (name, linear_module) in enumerate(layers_to_replace):
        # Get optimal group size for this layer
        group_size = optimal_group_map.get(name, 32)  # Default to 32 if not found

        # Determine layer type and use appropriate LUT size and scale rank
        is_mlp = 'mlp' in name
        if is_mlp:
            lut_size = 2 ** mlp_lut_bits
            scale_rank = mlp_scale_rank
        else:  # Attention
            lut_size = 2 ** attn_lut_bits
            scale_rank = attn_scale_rank

        # Create config with optimal group_size
        # NOTE: Use identity for SVD compatibility (same as create_v2_configs)
        config = AnemllQuantConfigV2(
            lut_size=lut_size,
            scale_rank=scale_rank,
            group_size=group_size,
            force_positive_scales=False,
            positive_scale_method="abs",
            magnitude_activation="identity",
            magnitude_eps=0.0,
        )

        # Create V2 layer
        v2_layer = AnemllQATLinearV2.from_linear(linear_module, config=config, skip_init=False)

        # Replace in model
        parent_name, layer_name = name.rsplit('.', 1) if '.' in name else ('', name)
        parent = model
        for part in parent_name.split('.'):
            if part:
                parent = getattr(parent, part)
        setattr(parent, layer_name, v2_layer)

        replaced_count += 1
        group_size_used[group_size] = group_size_used.get(group_size, 0) + 1

        # Progress
        if verbose and (layer_idx % 40 == 0 or layer_idx == len(layers_to_replace) - 1):
            short_name = name.split('.')[-2] + '.' + name.split('.')[-1] if '.' in name else name
            print(f"    [{layer_idx+1}/{len(layers_to_replace)}] {short_name} (g={group_size})")

    elapsed = time.time() - t0

    stats = {
        'replaced_count': replaced_count,
        'group_sizes_used': group_size_used,
        'time_seconds': elapsed,
    }

    if verbose:
        print(f"\n  Replaced {replaced_count} layers in {elapsed:.1f}s")
        print(f"  Group sizes used: {group_size_used}")

    return stats


# =============================================================================
# STEP 4e: SEARCH OPTIMAL LUT PER TENSOR
# =============================================================================

@torch.no_grad()
def search_optimal_luts(
    model: nn.Module,
    model_id: str,
    lut_candidates: Dict[str, torch.Tensor],
    group_size: int = 32,
    mlp_lut_bits: int = 4,
    mlp_scale_rank: int = 32,
    attn_lut_bits: int = 4,
    attn_scale_rank: int = 32,
    imatrix: Optional[Dict[str, torch.Tensor]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Search for optimal LUT per tensor by testing candidates and minimizing MSE or iMSE.

    For each 4-bit linear layer, tests each LUT candidate by:
    1. Computing SVD-based scale initialization with that LUT
    2. Quantizing to LUT
    3. Measuring reconstruction error:
       - MSE = mean((W_ref - Q * S)²)  if no imatrix
       - iMSE = weighted MSE using σ² from importance matrix (if imatrix provided)
    4. Selecting LUT with minimum error

    NOTE: Only searches for 4-bit layers (LUT16). 2-bit layers keep uniform LUT.

    Args:
        model: Base model with nn.Linear layers (BEFORE V2 replacement)
        model_id: HuggingFace model ID (for reference, model already loaded)
        lut_candidates: Dict mapping LUT name to tensor (e.g., {'uniform': tensor, ...})
        group_size: Group size for SVD scale initialization
        mlp_lut_bits: LUT bits for MLP layers
        mlp_scale_rank: Scale rank for MLP layers
        attn_lut_bits: LUT bits for Attention layers
        attn_scale_rank: Scale rank for Attention layers
        imatrix: Dict mapping layer name -> σ² tensor (importance weights per input dim)
        verbose: Print progress

    Returns:
        Dict with per-layer optimal LUT names and statistics
    """
    from qat_lora.ane_qat_linear_v2 import AnemllQuantConfigV2, AnemllQATLinearV2

    use_imatrix = imatrix is not None and len(imatrix) > 0
    if verbose:
        print(f"\n[Step 4e] Searching optimal LUT per tensor")
        print(f"  Testing LUTs: {list(lut_candidates.keys())}")
        print(f"  MLP: LUT{2**mlp_lut_bits} ({mlp_lut_bits}-bit), rank={mlp_scale_rank}")
        print(f"  Attn: LUT{2**attn_lut_bits} ({attn_lut_bits}-bit), rank={attn_scale_rank}")
        if use_imatrix:
            print(f"  Scoring: iMSE (importance-weighted, {len(imatrix)} layers)")
        else:
            print(f"  Scoring: MSE (uniform weighting)")

    t0 = time.time()

    # Collect all linear layers that would be quantized (MLP + Attn, not lm_head/embeddings)
    linear_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Skip lm_head and embeddings
            if 'lm_head' in name or 'embed' in name:
                continue
            # Only include MLP and attention layers
            if 'mlp' in name or 'self_attn' in name or 'attention' in name:
                linear_layers.append((name, module))

    if verbose:
        print(f"  Found {len(linear_layers)} linear layers to analyze")

    # Results storage
    layer_results = []
    lut_names = list(lut_candidates.keys())
    lut_counts = {name: 0 for name in lut_names}
    mlp_lut_counts = {name: 0 for name in lut_names}
    attn_lut_counts = {name: 0 for name in lut_names}

    for layer_idx, (name, linear_module) in enumerate(linear_layers):
        # Get device from original module
        device = linear_module.weight.device
        W_ref = linear_module.weight.data.clone()
        out_features, in_features = W_ref.shape

        # Determine layer type and select appropriate config
        is_mlp = 'mlp' in name
        is_attn = 'self_attn' in name or 'attention' in name
        layer_type = 'mlp' if is_mlp else ('attn' if is_attn else 'other')

        # Use appropriate LUT size and scale rank for this layer type
        if is_mlp:
            lut_bits = mlp_lut_bits
            lut_size = 2 ** mlp_lut_bits
            scale_rank = mlp_scale_rank
        else:  # Attention
            lut_bits = attn_lut_bits
            lut_size = 2 ** attn_lut_bits
            scale_rank = attn_scale_rank

        # Only search LUT for 4-bit layers (LUT16)
        # 2-bit layers (LUT4) keep uniform - we'll handle that later
        if lut_bits != 4:
            # Skip search, use uniform
            layer_result = {
                'name': name,
                'shape': [out_features, in_features],
                'type': layer_type,
                'optimal_lut': 'uniform',
                'optimal_mse': 0.0,
                'all_mses': {'uniform': 0.0},
                'skipped': True,
                'skip_reason': f'{lut_bits}-bit, search only for 4-bit',
            }
            layer_results.append(layer_result)
            lut_counts['uniform'] += 1
            if is_mlp:
                mlp_lut_counts['uniform'] += 1
            elif is_attn:
                attn_lut_counts['uniform'] += 1
            continue

        # Test each LUT candidate
        best_lut_name = 'uniform'
        best_mse = float('inf')
        lut_mses = {}

        for lut_name, lut_tensor in lut_candidates.items():
            # Create temporary config
            temp_config = AnemllQuantConfigV2(
                lut_size=lut_size,
                scale_rank=scale_rank,
                group_size=group_size,
                force_positive_scales=False,
                positive_scale_method="abs",
                magnitude_activation="identity",
                magnitude_eps=0.0,
            )

            # Create temporary V2 layer with this LUT
            try:
                temp_v2 = AnemllQATLinearV2.from_linear(
                    linear_module,
                    config=temp_config,
                    skip_init=False,  # Do full SVD init
                    custom_lut=lut_tensor,  # Use this LUT candidate
                )

                # Ensure all tensors are on the same device
                temp_v2.to(device)

                # Snap rank_magnitude to FP16 before computing scales
                # (SVD init gives FP32 values; we need FP16-representable values for accurate MSE)
                with torch.no_grad():
                    temp_v2.rank_magnitude.data = temp_v2.rank_magnitude.data.to(torch.float16).to(torch.float32)

                # DEBUG: Print LUT values for first layer
                if layer_idx == 0 and verbose:
                    print(f"      DEBUG: {lut_name} LUT values: {temp_v2.lut.tolist()[:4]}...{temp_v2.lut.tolist()[-4:]}")

                # Freeze Q to populate _Q buffer
                temp_v2.freeze_Q()

                # Get effective weight: W_eff = Q * S
                Q = temp_v2._Q.view(temp_v2.out_features, temp_v2.in_features)
                S = temp_v2._compute_full_scales()
                W_eff = Q * S

                # DEBUG: Print Q stats for first layer
                if layer_idx == 0 and verbose:
                    unique_q = torch.unique(Q)
                    # Index histogram
                    indices = temp_v2._indices
                    idx_counts = torch.bincount(indices.flatten(), minlength=16)
                    idx_hist = idx_counts.tolist()
                    print(f"      DEBUG: {lut_name} Q unique={len(unique_q)}, range=[{Q.min():.4f}, {Q.max():.4f}]")
                    print(f"      DEBUG: {lut_name} idx histogram: {idx_hist}")

                # Compute error score: iMSE (importance-weighted) or plain MSE
                err2 = (W_ref - W_eff).pow(2)  # [out, in]

                if use_imatrix and name in imatrix:
                    # iMSE: importance-weighted MSE using σ² from calibration
                    sigma2 = imatrix[name].to(device=err2.device, dtype=err2.dtype)  # [in]
                    # Normalize: sum(err² * σ²) / (out * sum(σ²))
                    score = (err2 * sigma2.view(1, -1)).sum().item() / (err2.shape[0] * sigma2.sum().item())
                else:
                    # Plain MSE (uniform weighting)
                    score = err2.mean().item()

                lut_mses[lut_name] = score

                if score < best_mse:
                    best_mse = score
                    best_lut_name = lut_name

                # Clean up
                del Q, S, W_eff, err2
                del temp_v2

            except Exception as e:
                if verbose:
                    print(f"    [WARN] LUT '{lut_name}' failed for {name}: {e}")
                lut_mses[lut_name] = float('inf')

        # Release per-layer temporaries before moving to the next layer.
        del W_ref
        if (layer_idx + 1) % 8 == 0:
            _release_memory(device)

        layer_result = {
            'name': name,
            'shape': [out_features, in_features],
            'type': layer_type,
            'optimal_lut': best_lut_name,
            'optimal_score': best_mse,
            'all_scores': lut_mses,
            'score_type': 'iMSE' if use_imatrix else 'MSE',
            'skipped': False,
            # Legacy keys for backwards compatibility
            'optimal_mse': best_mse,
            'all_mses': lut_mses,
        }
        layer_results.append(layer_result)

        # Update counts
        lut_counts[best_lut_name] += 1
        if is_mlp:
            mlp_lut_counts[best_lut_name] += 1
        elif is_attn:
            attn_lut_counts[best_lut_name] += 1

        # Progress (print every layer for debugging)
        if verbose:
            short_name = name.split('.')[-2] + '.' + name.split('.')[-1] if '.' in name else name
            # Show scores for ALL LUTs, sorted by value (scientific notation)
            # Highlight lowest (best) score in green if terminal supports colors
            sorted_scores = sorted(lut_mses.items(), key=lambda x: x[1])
            use_color = sys.stdout.isatty()
            GREEN = '\033[92m' if use_color else ''
            RESET = '\033[0m' if use_color else ''
            score_parts = []
            for i, (n, m) in enumerate(sorted_scores):
                if i == 0:  # Best (lowest) score - show in green
                    score_parts.append(f"{GREEN}{n}={m:.2e}{RESET}")
                else:
                    score_parts.append(f"{n}={m:.2e}")
            score_str = ', '.join(score_parts)
            score_label = "iMSE" if use_imatrix else "MSE"
            print(f"    [{layer_idx+1}/{len(linear_layers)}] {short_name}: best={best_lut_name}")
            print(f"        {score_label}: {score_str}")

    elapsed = time.time() - t0

    # Build optimal LUT map
    optimal_lut_map = {r['name']: r['optimal_lut'] for r in layer_results}

    # Aggregate statistics
    stats = {
        'num_layers': len(layer_results),
        'lut_names_tested': lut_names,
        'lut_counts': lut_counts,
        'mlp_lut_counts': mlp_lut_counts,
        'attn_lut_counts': attn_lut_counts,
        'optimal_lut_map': optimal_lut_map,
        'layer_results': layer_results,
        'time_seconds': elapsed,
        'score_type': 'iMSE' if use_imatrix else 'MSE',
        'imatrix_used': use_imatrix,
    }

    if verbose:
        print(f"\n  LUT Search Results:")
        print(f"    Layers analyzed: {stats['num_layers']}")
        print(f"    Scoring method: {stats['score_type']}")
        print(f"    Time: {elapsed:.1f}s")
        print(f"\n  Optimal LUT Distribution:")
        for lut_name in lut_names:
            count = lut_counts[lut_name]
            pct = 100 * count / max(1, len(layer_results))
            mlp_count = mlp_lut_counts[lut_name]
            attn_count = attn_lut_counts[lut_name]
            print(f"    {lut_name:10s}: {count:3d} layers ({pct:5.1f}%) - MLP: {mlp_count}, Attn: {attn_count}")

        # Show breakdown by layer type
        print(f"\n  MLP layers LUT distribution:")
        mlp_total = sum(mlp_lut_counts.values())
        for lut_name in lut_names:
            if mlp_lut_counts[lut_name] > 0:
                pct = 100 * mlp_lut_counts[lut_name] / max(1, mlp_total)
                print(f"    {lut_name}: {mlp_lut_counts[lut_name]} ({pct:.1f}%)")

        print(f"\n  Attention layers LUT distribution:")
        attn_total = sum(attn_lut_counts.values())
        for lut_name in lut_names:
            if attn_lut_counts[lut_name] > 0:
                pct = 100 * attn_lut_counts[lut_name] / max(1, attn_total)
                print(f"    {lut_name}: {attn_lut_counts[lut_name]} ({pct:.1f}%)")

    return stats


def replace_linear_layers_with_optimal_luts(
    model: nn.Module,
    optimal_lut_map: Dict[str, str],
    lut_candidates: Dict[str, torch.Tensor],
    group_size: int = 32,
    mlp_lut_bits: int = 4,
    mlp_scale_rank: int = 32,
    attn_lut_bits: int = 4,
    attn_scale_rank: int = 32,
    quantize_attn: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Replace linear layers using per-layer optimal LUTs.

    Args:
        model: Base model with nn.Linear layers
        optimal_lut_map: Dict mapping layer name -> optimal LUT name
        lut_candidates: Dict mapping LUT name -> tensor
        group_size: Group size for SVD scale initialization
        mlp_lut_bits: LUT bits for MLP layers
        mlp_scale_rank: Scale rank for MLP layers
        attn_lut_bits: LUT bits for Attention layers
        attn_scale_rank: Scale rank for Attention layers
        quantize_attn: Whether to quantize attention layers
        verbose: Print progress

    Returns:
        Dict with replacement stats
    """
    from qat_lora.ane_qat_linear_v2 import AnemllQuantConfigV2, AnemllQATLinearV2

    if verbose:
        print(f"\n[Step 4f] Replacing layers with optimal LUTs")
        print(f"  MLP: LUT{2**mlp_lut_bits} ({mlp_lut_bits}-bit), rank={mlp_scale_rank}")
        print(f"  Attn: LUT{2**attn_lut_bits} ({attn_lut_bits}-bit), rank={attn_scale_rank}")

    t0 = time.time()

    # Collect layers to replace
    layers_to_replace = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if 'lm_head' in name or 'embed' in name:
                continue
            is_mlp = 'mlp' in name
            is_attn = 'self_attn' in name or 'attention' in name
            if is_mlp or (quantize_attn and is_attn):
                layers_to_replace.append((name, module))

    if verbose:
        print(f"  Replacing {len(layers_to_replace)} layers")

    replaced_count = 0
    lut_used = {}

    for layer_idx, (name, linear_module) in enumerate(layers_to_replace):
        # Get optimal LUT for this layer
        lut_name = optimal_lut_map.get(name, 'uniform')
        custom_lut = lut_candidates.get(lut_name)

        # Determine layer type and use appropriate config
        is_mlp = 'mlp' in name
        if is_mlp:
            lut_size = 2 ** mlp_lut_bits
            scale_rank = mlp_scale_rank
        else:  # Attention
            lut_size = 2 ** attn_lut_bits
            scale_rank = attn_scale_rank

        # Create config
        config = AnemllQuantConfigV2(
            lut_size=lut_size,
            scale_rank=scale_rank,
            group_size=group_size,
            force_positive_scales=False,
            positive_scale_method="abs",
            magnitude_activation="identity",
            magnitude_eps=0.0,
        )

        # Create V2 layer with custom LUT
        v2_layer = AnemllQATLinearV2.from_linear(
            linear_module,
            config=config,
            skip_init=False,
            custom_lut=custom_lut,
        )

        # Snap rank_magnitude to FP16 (SVD init gives FP32 values)
        with torch.no_grad():
            v2_layer.rank_magnitude.data = v2_layer.rank_magnitude.data.to(torch.float16).to(torch.float32)

        # Replace in model
        parent_name, layer_name = name.rsplit('.', 1) if '.' in name else ('', name)
        parent = model
        for part in parent_name.split('.'):
            if part:
                parent = getattr(parent, part)
        setattr(parent, layer_name, v2_layer)

        replaced_count += 1
        lut_used[lut_name] = lut_used.get(lut_name, 0) + 1

        # Progress
        if verbose and (layer_idx % 40 == 0 or layer_idx == len(layers_to_replace) - 1):
            short_name = name.split('.')[-2] + '.' + name.split('.')[-1] if '.' in name else name
            print(f"    [{layer_idx+1}/{len(layers_to_replace)}] {short_name} (lut={lut_name})")

    elapsed = time.time() - t0

    stats = {
        'replaced_count': replaced_count,
        'luts_used': lut_used,
        'time_seconds': elapsed,
    }

    if verbose:
        print(f"\n  Replaced {replaced_count} layers in {elapsed:.1f}s")
        print(f"  LUTs used: {lut_used}")

    return stats


# =============================================================================
# STEP 5: FREEZE Q (QUANTIZED WEIGHTS)
# =============================================================================

def freeze_quantized_weights(
    model: nn.Module,
    verbose: bool = True,
    detailed: bool = False,
    original_weights: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, Any]:
    """
    Freeze Q (quantized weight lookup tables) for all V2 layers.

    After this step:
    - Q values (_Q buffer) are FROZEN - cannot change during training
    - Only scale parameters are trainable: scale_A, scale_B, rank_magnitude
    - LUT values are also fixed

    This is the key to the V2 training paradigm:
    - Q captures the "what" (quantized weight structure)
    - Scales capture the "how much" (magnitudes, can be optimized)

    Args:
        model: Model with AnemllQATLinearV2 layers
        verbose: Print freeze progress
        detailed: Show LUT values and MSE per layer (requires original_weights)
        original_weights: Optional dict of original FP32 weights for MSE computation

    Returns:
        Dict with freeze stats
    """
    from qat_lora.ane_qat_linear_v2 import AnemllQATLinearV2

    if verbose:
        print(f"\n[Step 5] Freezing Q (quantized weight LUTs)")

    t0 = time.time()

    # Print LUT values once (all layers use the same default LUT per type)
    if detailed:
        first_mlp_lut_printed = False
        first_attn_lut_printed = False
        for name, module in model.named_modules():
            if isinstance(module, AnemllQATLinearV2):
                lut = module.lut.cpu()
                lut_size = lut.numel()
                is_mlp = 'mlp' in name.lower()

                if is_mlp and not first_mlp_lut_printed:
                    lut_str = ", ".join([f"{v:+.4f}" for v in lut.tolist()])
                    print(f"  MLP LUT ({lut_size} entries): [{lut_str}]")
                    first_mlp_lut_printed = True
                elif not is_mlp and not first_attn_lut_printed:
                    lut_str = ", ".join([f"{v:+.4f}" for v in lut.tolist()])
                    print(f"  Attn LUT ({lut_size} entries): [{lut_str}]")
                    first_attn_lut_printed = True

                if first_mlp_lut_printed and first_attn_lut_printed:
                    break

    # Freeze Q for all layers, computing MSE if detailed
    frozen_count = 0
    frozen_verified = 0
    unfrozen = 0
    total_mse = 0.0
    layer_stats = []

    for name, module in model.named_modules():
        if isinstance(module, AnemllQATLinearV2):
            # Freeze Q
            module.freeze_Q()
            frozen_count += 1

            if module._Q is not None:
                frozen_verified += 1

                # Compute MSE if detailed mode and original weights provided
                if detailed and original_weights is not None:
                    W_ref = original_weights.get(name)
                    if W_ref is None:
                        W_ref = original_weights.get(f"{name}.weight")

                    if W_ref is not None:
                        # Compute W_effective after freeze
                        S = module._compute_full_scales()
                        W_eff = module._Q * S

                        # Ensure W_ref is on same device
                        W_ref_dev = W_ref.to(W_eff.device)
                        mse = ((W_eff - W_ref_dev) ** 2).mean().item()
                        mae = (W_eff - W_ref_dev).abs().mean().item()
                        total_mse += mse

                        # LUT range
                        lut_min = module.lut.min().item()
                        lut_max = module.lut.max().item()

                        # Short layer name for display
                        short_name = name.replace('model.layers.', '').replace('.self_attn.', '.').replace('.mlp.', '.')
                        print(f"  {short_name:40s} LUT=[{lut_min:+.4f}, {lut_max:+.4f}] MSE={mse:.2e} MAE={mae:.2e}")

                        layer_stats.append({'name': name, 'mse': mse, 'mae': mae})
            else:
                unfrozen += 1
                if verbose:
                    print(f"  [WARN] Not frozen: {name}")

    elapsed = time.time() - t0

    stats = {
        'frozen_count': frozen_count,
        'verified_frozen': frozen_verified,
        'unfrozen': unfrozen,
        'time_seconds': elapsed,
    }

    if detailed and layer_stats:
        stats['total_mse'] = total_mse
        stats['layer_count'] = len(layer_stats)

    if verbose:
        print(f"  Frozen: {frozen_count} layers")
        print(f"  Verified: {frozen_verified}/{frozen_verified + unfrozen}")
        if detailed and layer_stats:
            print(f"  Total MSE (sum): {total_mse:.6e}")
        if unfrozen > 0:
            print(f"  [ERROR] {unfrozen} layers failed to freeze!")

    return stats


# =============================================================================
# STEP 6: VALIDATE
# =============================================================================

def validate_model(
    model: nn.Module,
    tokenizer: Any,
    device: torch.device,
    test_prompts: Optional[list] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Validate the initialized model with inference tests.

    CHECKS:
    1. Model can run forward pass without errors
    2. Output logits have reasonable range
    3. Generation produces coherent text (optional)

    Args:
        model: Initialized V2 model
        tokenizer: Tokenizer for test prompts
        device: Target device
        test_prompts: List of test prompts (default: built-in tests)
        verbose: Print validation details

    Returns:
        Dict with validation results
    """
    if verbose:
        print(f"\n[Step 6] Validating model")

    if test_prompts is None:
        test_prompts = [
            "The capital of France is",
            "2 + 2 =",
            "def fibonacci(n):",
        ]

    model.eval()
    results = {
        'passed': True,
        'tests': [],
        'errors': [],
    }

    for i, prompt in enumerate(test_prompts):
        test_result = {
            'prompt': prompt,
            'passed': False,
            'output': None,
            'logits_range': None,
            'error': None,
        }

        try:
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            # Forward pass
            with torch.no_grad():
                outputs = model(**inputs)

            logits = outputs.logits
            logits_min = logits.min().item()
            logits_max = logits.max().item()
            test_result['logits_range'] = (logits_min, logits_max)

            # Check logits are reasonable (not NaN, not extreme)
            if torch.isnan(logits).any():
                test_result['error'] = "NaN in logits"
                results['errors'].append(f"Test {i+1}: NaN in logits")
            elif logits_max > 1000 or logits_min < -1000:
                test_result['error'] = f"Extreme logits: [{logits_min:.1f}, {logits_max:.1f}]"
                results['errors'].append(f"Test {i+1}: {test_result['error']}")
            else:
                # Generate a few tokens
                gen_output = model.generate(
                    inputs['input_ids'],
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
                generated = tokenizer.decode(gen_output[0], skip_special_tokens=True)
                test_result['output'] = generated
                test_result['passed'] = True

        except Exception as e:
            test_result['error'] = str(e)
            results['errors'].append(f"Test {i+1}: {e}")

        results['tests'].append(test_result)

        if verbose:
            status = "PASS" if test_result['passed'] else "FAIL"
            print(f"  Test {i+1}: [{status}] '{prompt[:30]}...'")
            if test_result['output']:
                # Show just the generated part
                gen_part = test_result['output'][len(prompt):].strip()[:50]
                print(f"         -> '{gen_part}...'")
            if test_result['error']:
                print(f"         Error: {test_result['error']}")

    # Overall pass/fail
    results['passed'] = len(results['errors']) == 0

    if verbose:
        if results['passed']:
            print(f"  All {len(test_prompts)} tests passed!")
        else:
            print(f"  {len(results['errors'])} test(s) failed")

    return results


# =============================================================================
# STEP 8: TIGHTEN Q + MEASURE PERPLEXITY (OPTIONAL)
# =============================================================================

def tighten_and_measure_ppl(
    model: nn.Module,
    tokenizer: Any,
    model_id: str,
    device: torch.device,
    num_chunks: int = 14,
    verbose: bool = True,
    skip_ppl: bool = False,
) -> Dict[str, Any]:
    """
    Tighten Q with baseline weights and optionally measure perplexity.

    This step:
    1. Loads baseline HF weights for W_ref (fresh, not quantized)
    2. Recalculates _Q to be consistent with current scales
    3. Optionally measures perplexity using quick_perplexity

    IMPORTANT: This gives a more accurate estimate of model quality
    than using the SVD-initialized Q values directly.

    Args:
        model: V2 model with frozen Q
        tokenizer: Tokenizer for perplexity evaluation
        model_id: Base model ID for loading W_ref
        device: Target device
        num_chunks: Number of chunks for perplexity evaluation
        verbose: Print progress
        skip_ppl: If True, only tighten Q without measuring PPL

    Returns:
        Dict with tighten stats and perplexity (if not skipped)
    """
    from qat_lora.ane_qat_linear_v2 import AnemllQATLinearV2
    # Import tighten functions from tighten_q.py
    from scripts.tighten_q import tighten_q_layer, load_baseline_weights

    if verbose:
        action = "Tighten Q" if skip_ppl else "Tighten Q + Measure Perplexity"
        print(f"\n[Step 7a] {action}")

    results = {
        'tighten': {},
        'perplexity': {},
    }

    t0 = time.time()

    # --- Step 8a: Load baseline weights ---
    if verbose:
        print(f"  Loading baseline weights from {model_id}...")

    W_ref_map = load_baseline_weights(model_id)

    if verbose:
        print(f"  Loaded {len(W_ref_map)} baseline weight tensors")

    # --- Step 8b: Snap magnitudes to FP16 before tightening ---
    v2_layers = [(name, m) for name, m in model.named_modules() if isinstance(m, AnemllQATLinearV2)]

    # Snap all rank_magnitudes to FP16 FIRST (before computing scales)
    mags_snapped = {'mlp': 0, 'attn': 0, 'total': 0}
    with torch.no_grad():
        for name, module in v2_layers:
            if hasattr(module, 'rank_magnitude') and module.rank_magnitude is not None:
                module.rank_magnitude.data = module.rank_magnitude.data.to(torch.float16).to(torch.float32)
                mags_snapped['total'] += 1
                if 'mlp' in name:
                    mags_snapped['mlp'] += 1
                elif 'self_attn' in name:
                    mags_snapped['attn'] += 1

    if verbose:
        print(f"  Snapped magnitudes to FP16:")
        print(f"    MLP:  {mags_snapped['mlp']} layers")
        print(f"    Attn: {mags_snapped['attn']} layers")

    # --- Step 8c: Tighten Q for all V2 layers using tighten_q.py logic ---
    if verbose:
        print(f"  Tightening Q (recalculating to match scales)...")

    tighten_stats = {
        'layers_tightened': 0,
        'total_changed': 0,
        'total_params': 0,
        'avg_mse_delta': 0,
    }
    mse_deltas = []

    for layer_idx, (name, module) in enumerate(v2_layers):
        if verbose and (layer_idx % 20 == 0 or layer_idx == len(v2_layers) - 1):
            print(f"    Layer {layer_idx + 1}/{len(v2_layers)}: {name.split('.')[-2]}.{name.split('.')[-1]}")

        # Get W_ref for this layer
        W_ref = W_ref_map.get(name)
        if W_ref is None:
            continue

        if module._Q is None:
            continue

        # Force CPU for tighten (TPU/XLA is slow for per-layer ops)
        orig_device = module._Q.device
        is_xla = 'xla' in str(orig_device).lower() or 'tpu' in str(orig_device).lower()
        if is_xla:
            module.to('cpu')

        # W_ref stays on CPU (already loaded on CPU)

        # Use tighten_q_layer from tighten_q.py (includes clamp_q=True, updates _indices)
        qc = tighten_q_layer(
            module=module,
            W_ref=W_ref,
            eps=1e-4,
            clamp_q=True,  # Same as --clamp-q flag
            chunk_size=4096,
        )

        tighten_stats['layers_tightened'] += 1
        tighten_stats['total_changed'] += qc['num_changed']
        tighten_stats['total_params'] += qc['total']
        mse_deltas.append(qc['mse_delta'])

        if (layer_idx + 1) % 8 == 0:
            _release_memory(orig_device)

    # Model stays on CPU after tightening (fine for saving, training script moves to device)

    if mse_deltas:
        tighten_stats['avg_mse_delta'] = sum(mse_deltas) / len(mse_deltas)

    if verbose:
        pct = 100 * tighten_stats['total_changed'] / max(1, tighten_stats['total_params'])
        print(f"  Tightened {tighten_stats['layers_tightened']} layers")
        print(f"  Q values changed: {tighten_stats['total_changed']:,} / {tighten_stats['total_params']:,} ({pct:.1f}%)")
        print(f"  Avg MSE delta: {tighten_stats['avg_mse_delta']:+.2e}")

    results['tighten'] = tighten_stats

    # Baseline map is no longer needed after tighten.
    W_ref_map.clear()
    del W_ref_map
    _release_memory(device)

    # --- Step 8c: Measure perplexity (if not skipped) ---
    if not skip_ppl:
        if verbose:
            print(f"\n  Measuring perplexity ({num_chunks} chunks)...")

        # Ensure all model parameters are on the correct device after tightening
        # (some buffers like W_ref might have been on CPU)
        model.to(device)

        # Import quick perplexity
        from scripts.quick_perplexity import QuickPerplexityEstimator

        # Create estimator (caches chunks)
        estimator = QuickPerplexityEstimator(
            tokenizer,
            num_chunks=num_chunks,
            chunk_size=1024,
            stride=256,
            verbose=False,
        )

        # Evaluate
        model.eval()
        ppl_result = estimator.evaluate(model, device=device, verbose=False)

        results['perplexity'] = {
            'perplexity': ppl_result['perplexity'],
            'cross_entropy': ppl_result['cross_entropy'],
            'tokens': ppl_result['tokens'],
            'time': ppl_result['time'],
        }

        if verbose:
            print(f"\n  Quick PPL: {ppl_result['perplexity']:.2f}")
            print(f"  Cross-entropy: {ppl_result['cross_entropy']:.4f} nats")
        del estimator, ppl_result
        _release_memory(device)

    total_time = time.time() - t0
    results['time_seconds'] = total_time

    if verbose:
        print(f"  Total time: {total_time:.1f}s")

    return results


# =============================================================================
# STEP 7: SAVE
# =============================================================================

def save_checkpoint(
    model: nn.Module,
    output_dir: str,
    preset: QuantPreset,
    model_id: str,
    group_size: int,
    init_metrics: Dict[str, Any],
    optimal_lut_map: Optional[Dict[str, str]] = None,
    verbose: bool = True,
) -> Dict[str, str]:
    """
    Save the initialized model and config.

    OUTPUT FILES:
    - v2_initial.pt: Full model state dict (FP32, trainable)
    - config.json: Quantization config for training scripts
    - init_metrics.json: Initialization metrics for comparison

    Args:
        model: Initialized V2 model
        output_dir: Output directory path
        preset: Quantization preset used
        model_id: Base model ID
        group_size: Group size used for initialization
        init_metrics: Collected metrics from all steps
        optimal_lut_map: Per-layer optimal LUT names (from LUT search)
        verbose: Print save progress

    Returns:
        Dict mapping file type to path
    """
    if verbose:
        print(f"\n[Step 7] Saving checkpoint to {output_dir}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    paths = {}

    # Save model state dict
    checkpoint_path = output_path / "v2_initial.pt"
    state_dict = model.state_dict()
    estimated_bytes = _estimate_state_dict_bytes(state_dict)
    _safe_torch_save(state_dict, checkpoint_path, required_bytes_hint=estimated_bytes)
    del state_dict
    _release_memory()
    paths['checkpoint'] = str(checkpoint_path)

    # Get checkpoint size
    ckpt_size_mb = checkpoint_path.stat().st_size / 1024 / 1024

    if verbose:
        print(f"  Checkpoint: {checkpoint_path} ({ckpt_size_mb:.1f} MB)")

    # Save config.json
    config_data = {
        'version': 'v2',
        'model_id': model_id,
        'config_preset': preset.name,
        # MLP config
        'lut_bits': preset.mlp_lut_bits,
        'mlp_lut_bits': preset.mlp_lut_bits,
        'scale_rank': preset.mlp_rank,
        'mlp_scale_rank': preset.mlp_rank,
        # Attention config
        'attn_lut_bits': preset.attn_lut_bits,
        'attn_scale_rank': preset.attn_rank,
        # Meta
        'group_size': group_size,
        'description': preset.description,
        # CRITICAL: Scale config (must match between init/tighten/inference)
        # SVD initialization requires identity mode - do not change these!
        'force_positive_scales': False,
        'magnitude_activation': 'identity',
    }

    # Add LUT search results if present
    if optimal_lut_map is not None:
        config_data['lut_search_enabled'] = True
        config_data['optimal_lut_map'] = optimal_lut_map
    else:
        config_data['lut_search_enabled'] = False

    config_path = output_path / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2)
    paths['config'] = str(config_path)

    if verbose:
        print(f"  Config: {config_path}")

    # Save init metrics
    metrics_path = output_path / "init_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(init_metrics, f, indent=2, default=str)
    paths['metrics'] = str(metrics_path)

    if verbose:
        print(f"  Metrics: {metrics_path}")

    return paths


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def init_v2_model(
    model_id: str = "Qwen/Qwen3-0.6B",
    preset_name: str = "q4a4",
    output_dir: str = "runs/init_test",
    force_tpu: bool = False,
    force_cpu: bool = False,
    quantize_attn: bool = True,
    group_size: int = 32,
    validate: bool = True,
    measure_ppl: bool = False,
    ppl_chunks: int = 14,
    measure_svd_error: bool = False,
    search_group_sizes: Optional[List[int]] = None,
    search_lut: bool = False,
    default_lut: Optional[str] = None,
    imatrix_path: Optional[str] = None,
    awq_alpha: float = 0.0,
    verbose: bool = True,
    detailed: bool = False,
) -> Dict[str, Any]:
    """
    Complete pipeline to initialize a V2 QAT model from scratch.

    This is the main entry point that orchestrates all steps.

    Args:
        model_id: HuggingFace model ID
        preset_name: Quantization preset name
        output_dir: Output directory for checkpoint and config
        force_tpu: Force TPU mode
        force_cpu: Force CPU mode
        quantize_attn: Whether to quantize attention layers
        group_size: Group size for SVD scale initialization (default: 32)
        validate: Run validation tests
        measure_ppl: Tighten Q and measure perplexity
        ppl_chunks: Number of chunks for perplexity measurement
        measure_svd_error: Measure SVD approximation error (MAE vs original)
        search_group_sizes: List of group sizes to test per tensor (e.g., [128, 64, 32, 16])
        search_lut: Search for optimal LUT per tensor (only for 4-bit layers)
        default_lut: Use specific LUT for all 4-bit layers (e.g., 'fp4_dense'). Skips search.
        imatrix_path: Path to importance matrix (.pt file) for iMSE scoring
        awq_alpha: AWQ-style importance weighting for scale initialization (default: 0.0).
                   0.0 = no effect (standard SVD), 0.5 = moderate (recommended), 1.0 = strong.
                   Higher values bias scales toward important columns (from iMatrix).
                   Requires imatrix_path. This can improve PPL by reducing quantization
                   error for important weights.
        verbose: Print progress
        detailed: Show LUT values and MSE per layer during freeze step

    Returns:
        Dict with all metrics and paths
    """
    total_start = time.time()
    metrics = {
        'model_id': model_id,
        'preset': preset_name,
        'steps': {},
    }

    # Header
    if verbose:
        print("=" * 60)
        print("V2 QAT MODEL INITIALIZATION")
        print("=" * 60)
        print(f"Model:  {model_id}")
        print(f"Preset: {preset_name}")
        print(f"Output: {output_dir}")

    # Step 1: Configuration
    if verbose:
        print(f"\n[Step 1] Configuration")

    device, device_type = get_device(force_tpu, force_cpu)
    preset = PRESETS.get(preset_name)

    if preset is None:
        available = ", ".join(PRESETS.keys())
        raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")

    metrics['device'] = str(device)
    metrics['device_type'] = device_type

    if verbose:
        print(f"  Device: {device} ({device_type})")
        print(f"  Preset: {preset.description}")

    # Step 2: Load base model
    # CRITICAL: Use FP32 for SVD init accuracy
    model, tokenizer = load_base_model(
        model_id=model_id,
        device=device,
        dtype=torch.float32,  # Always FP32 for init
        verbose=verbose,
    )

    # Disk-space preflight before expensive replacement/search steps.
    # V2 checkpoints can be much larger than base FP32 weights due to extra
    # quantization tensors (_Q, _indices, scales, LUT buffers).
    base_param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    estimated_v2_ckpt_bytes = int(base_param_bytes * 3.6)
    checkpoint_target = Path(output_dir) / "v2_initial.pt"
    try:
        _ensure_free_space(checkpoint_target, required_bytes=estimated_v2_ckpt_bytes)
    except RuntimeError as e:
        raise RuntimeError(
            f"{e} (preflight estimate for V2 init checkpoint: {_format_bytes(estimated_v2_ckpt_bytes)})"
        ) from e
    if verbose:
        free_bytes = shutil.disk_usage(Path(output_dir)).free
        print(
            f"  Disk preflight: free={_format_bytes(free_bytes)}, "
            f"estimated_v2_ckpt~{_format_bytes(estimated_v2_ckpt_bytes)}"
        )

    # Capture original weights if detailed output is requested
    original_weights = {}
    if detailed:
        if verbose:
            print(f"  Capturing original FP32 weights for detailed output...")
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                original_weights[name] = module.weight.data.float().cpu().clone()
        if verbose:
            print(f"  Captured {len(original_weights)} linear layer weights")

    # Step 3: Create V2 configs
    # NOTE: Use defaults (force_positive_scales=False, magnitude_activation='identity')
    # for SVD compatibility
    mlp_config, attn_config = create_v2_configs(
        preset=preset,
        group_size=group_size,
        verbose=verbose,
    )

    # Step 4: Replace linear layers
    # Pipeline order: LUT search (if enabled) -> group_size search (if enabled) -> replace

    # Initialize optimal maps
    optimal_lut_map = None
    lut_candidates = None

    # Load importance matrix if provided
    imatrix = None
    if imatrix_path:
        if verbose:
            print(f"\n[INFO] Loading importance matrix from: {imatrix_path}")
        imatrix_data = torch.load(imatrix_path, weights_only=False)
        imatrix = imatrix_data.get('sigma2', {})
        if verbose:
            print(f"  Loaded σ² for {len(imatrix)} layers")
            if 'metadata' in imatrix_data:
                meta = imatrix_data['metadata']
                print(f"  Source: {meta.get('model_id', 'N/A')}, tokens={meta.get('num_tokens', 'N/A')}")
        del imatrix_data
        _release_memory(device)

    # Step 4a: LUT selection (search or fixed)
    # Option 1: Use fixed LUT for all layers (--lut fp4_dense)
    # Option 2: Search for optimal LUT per layer (--search-lut)

    if default_lut is not None and (preset.mlp_lut_bits == 4 or preset.attn_lut_bits == 4):
        # Fixed LUT for all 4-bit layers - no search needed
        lut_candidates = get_lut16_candidates()
        if default_lut not in lut_candidates:
            raise ValueError(f"Unknown LUT '{default_lut}'. Available: {list(lut_candidates.keys())}")

        if verbose:
            print(f"\n[Step 4a] Using fixed LUT: {default_lut} (no search)")

        # Build optimal_lut_map with fixed LUT for all MLP/Attn layers
        optimal_lut_map = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if 'lm_head' in name or 'embed' in name:
                    continue
                if 'mlp' in name or 'self_attn' in name or 'attention' in name:
                    optimal_lut_map[name] = default_lut

        if verbose:
            print(f"  Applied to {len(optimal_lut_map)} layers")

        metrics['steps']['lut_search'] = {
            'type': 'fixed',
            'lut_name': default_lut,
            'num_layers': len(optimal_lut_map),
        }

    elif search_lut and (preset.mlp_lut_bits == 4 or preset.attn_lut_bits == 4):
        if verbose:
            print(f"\n[Step 4a] Searching optimal LUT per tensor...")

        # Get LUT candidates
        lut_candidates = get_lut16_candidates()

        # Search for optimal LUT per tensor
        # Use group_size=16 for LUT search (fixed, regardless of final group_size)
        lut_search_stats = search_optimal_luts(
            model=model,
            model_id=model_id,
            lut_candidates=lut_candidates,
            group_size=16,  # Fixed for LUT search
            mlp_lut_bits=preset.mlp_lut_bits,
            mlp_scale_rank=preset.mlp_rank,
            attn_lut_bits=preset.attn_lut_bits,
            attn_scale_rank=preset.attn_rank,
            imatrix=imatrix,
            verbose=verbose,
        )
        metrics['steps']['lut_search'] = lut_search_stats
        optimal_lut_map = lut_search_stats['optimal_lut_map']
    elif search_lut or default_lut:
        if verbose:
            print(f"\n[Step 4a] LUT search SKIPPED (only for 4-bit layers, preset has {preset.mlp_lut_bits}-bit MLP, {preset.attn_lut_bits}-bit Attn)")

    # Step 4b: Group size search (if enabled, and LUT search not done)
    # Note: Currently LUT search and group_size search are mutually exclusive
    # If both are requested, LUT search takes precedence

    if optimal_lut_map is not None:
        # LUT search was done - use replace_linear_layers_with_optimal_luts
        if verbose:
            print(f"\n[Step 4c] Replacing layers with optimal LUTs...")

        replace_stats = replace_linear_layers_with_optimal_luts(
            model=model,
            optimal_lut_map=optimal_lut_map,
            lut_candidates=lut_candidates,
            group_size=group_size,
            mlp_lut_bits=preset.mlp_lut_bits,
            mlp_scale_rank=preset.mlp_rank,
            attn_lut_bits=preset.attn_lut_bits,
            attn_scale_rank=preset.attn_rank,
            quantize_attn=quantize_attn,
            verbose=verbose,
        )
        metrics['steps']['replace'] = replace_stats

    elif search_group_sizes is not None and len(search_group_sizes) > 1:
        if verbose:
            print(f"\n[Step 4b] Searching optimal group sizes per tensor...")

        # Search for optimal group sizes (on base model with nn.Linear layers)
        search_stats = search_optimal_group_sizes(
            model=model,
            model_id=model_id,
            group_sizes=search_group_sizes,
            mlp_lut_bits=preset.mlp_lut_bits,
            mlp_scale_rank=preset.mlp_rank,
            attn_lut_bits=preset.attn_lut_bits,
            attn_scale_rank=preset.attn_rank,
            verbose=verbose,
        )
        metrics['steps']['group_search'] = search_stats

        # Replace with optimal group sizes per layer
        replace_stats = replace_linear_layers_with_optimal_groups(
            model=model,
            optimal_group_map=search_stats['optimal_group_map'],
            mlp_lut_bits=preset.mlp_lut_bits,
            mlp_scale_rank=preset.mlp_rank,
            attn_lut_bits=preset.attn_lut_bits,
            attn_scale_rank=preset.attn_rank,
            quantize_attn=quantize_attn,
            verbose=verbose,
        )
        metrics['steps']['replace'] = replace_stats

    elif search_group_sizes is not None and len(search_group_sizes) == 1:
        # Single group size specified - use it directly without searching
        single_group_size = search_group_sizes[0]
        if verbose:
            print(f"\n[Step 4] Using single group size: {single_group_size} (no search needed)")

        # Update configs with the specified group size
        mlp_config, attn_config = create_v2_configs(
            preset=preset,
            group_size=single_group_size,
            verbose=False,  # Already printed above
        )

        # Standard replacement with the specified group_size
        replace_stats = replace_linear_layers(
            model=model,
            mlp_config=mlp_config,
            attn_config=attn_config,
            quantize_attn=quantize_attn,
            verbose=verbose,
        )
        metrics['steps']['replace'] = replace_stats
        # Update group_size for saving
        group_size = single_group_size

    else:
        # Standard replacement with uniform group_size
        replace_stats = replace_linear_layers(
            model=model,
            mlp_config=mlp_config,
            attn_config=attn_config,
            quantize_attn=quantize_attn,
            verbose=verbose,
        )
        metrics['steps']['replace'] = replace_stats

    # Step 4b-AWQ: AWQ-style importance weighting (NOT IMPLEMENTED)
    # AWQ requires BOTH:
    #   1. Scaling weights by importance factor
    #   2. Applying INVERSE factor to inputs (via RMSNorm.weight modification)
    # Without the inverse, the model output changes and inference breaks.
    # For Qwen3: Would need to modify post_attention_layernorm.weight for gate/up_proj,
    # input_layernorm.weight for q/k/v_proj. down_proj and o_proj are harder.
    # TODO: Implement proper AWQ with RMSNorm compensation
    if awq_alpha > 0:
        print(f"\n[Step 4b] ERROR: --awq-alpha is NOT IMPLEMENTED correctly!")
        print(f"  AWQ requires applying inverse scaling to inputs (via RMSNorm.weight),")
        print(f"  but this is not implemented. Using --awq-alpha will BREAK the model.")
        print(f"  Skipping AWQ weighting. Remove --awq-alpha from your command.")
        print()
        metrics['awq_alpha'] = 0
        metrics['awq_skipped'] = f'NOT IMPLEMENTED (requested alpha={awq_alpha})'

    # Ensure all model tensors are on the correct device after replacement
    # (SVD initialization may leave some buffers on CPU)
    model.to(device)

    # Step 4b: Measure SVD approximation error (optional)
    if measure_svd_error:
        svd_error_stats = measure_svd_approximation_error(
            model=model,
            model_id=model_id,
            verbose=verbose,
        )
        metrics['steps']['svd_error'] = svd_error_stats
    else:
        if verbose:
            print(f"\n[Step 4b] SVD error measurement SKIPPED (use --svd-error to enable)")

    # Step 5: Freeze Q
    freeze_stats = freeze_quantized_weights(
        model=model,
        verbose=verbose,
        detailed=detailed,
        original_weights=original_weights if detailed else None,
    )
    metrics['steps']['freeze'] = freeze_stats

    # Detailed original FP32 copies are no longer needed after freeze.
    if detailed and original_weights:
        original_weights.clear()
        _release_memory(device)

    # Step 6: Save (BEFORE tightening - this is the untightened initial checkpoint)
    save_paths = save_checkpoint(
        model=model,
        output_dir=output_dir,
        preset=preset,
        model_id=model_id,
        group_size=group_size,
        init_metrics=metrics,
        optimal_lut_map=optimal_lut_map,
        verbose=verbose,
    )
    metrics['paths'] = save_paths

    # Search artifacts are no longer needed after save.
    imatrix = None
    lut_candidates = None
    _release_memory(device)

    # Step 7+8: Tighten Q, Validate, and/or Measure PPL (if either is enabled)
    need_tightened_model = validate or measure_ppl
    if need_tightened_model:
        # Free the initial model before loading a fresh tightened model.
        # This avoids holding two ~1B models in memory simultaneously.
        model.to("cpu")
        del model
        _release_memory(device)

        # Load fresh model with SAME config as initialization
        # IMPORTANT: Config must match or _compute_full_scales() will give wrong values!
        if verbose:
            print(f"\n[Step 7] Loading fresh model for tightening (same config as init)...")

        from qat_lora.ane_qat_linear_v2 import AnemllQuantConfigV2, replace_linear_with_anemll_v2, AnemllQATLinearV2

        # Load fresh base model
        tightened_model = _load_causal_lm_safe(
            model_id,
            dtype=torch.float32,
            trust_remote_code=True,
        )

        # Create V2 configs with SAME settings as initialization
        # (force_positive_scales=False, magnitude_activation='identity' for SVD compatibility)
        mlp_lut_size = 2 ** preset.mlp_lut_bits
        attn_lut_size = 2 ** preset.attn_lut_bits

        tighten_mlp_config = AnemllQuantConfigV2(
            lut_size=mlp_lut_size,
            scale_rank=preset.mlp_rank,
            group_size=group_size,
            force_positive_scales=False,
            positive_scale_method="abs",
            magnitude_activation="identity",
            magnitude_eps=0.0,
        )
        tighten_attn_config = AnemllQuantConfigV2(
            lut_size=attn_lut_size,
            scale_rank=preset.attn_rank,
            group_size=group_size,
            force_positive_scales=False,
            positive_scale_method="abs",
            magnitude_activation="identity",
            magnitude_eps=0.0,
        )

        # Replace with V2 layers (skip_init=True since we'll load from checkpoint)
        replace_linear_with_anemll_v2(
            tightened_model,
            mlp_config=tighten_mlp_config,
            attn_config=tighten_attn_config,
            quantize_attn=quantize_attn,
            quantize_lm_head=False,
            skip_init=True,
        )

        # Load the saved checkpoint
        checkpoint_path = save_paths['checkpoint']
        if verbose:
            print(f"  Loading checkpoint: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        tightened_model.load_state_dict(state_dict, strict=False)

        # Manually load _Q buffers (None buffers don't load automatically)
        q_loaded = 0
        for name, m in tightened_model.named_modules():
            if isinstance(m, AnemllQATLinearV2):
                q_key = f"{name}._Q"
                if q_key in state_dict and m._Q is None:
                    m.register_buffer("_Q", state_dict[q_key])
                    q_loaded += 1
        if q_loaded > 0 and verbose:
            print(f"  Manually loaded {q_loaded} _Q buffers")

        # Full checkpoint dict no longer needed after manual _Q load.
        del state_dict
        _release_memory(device)

        # Move to device
        tightened_model.to(device)

        # Tighten Q
        tighten_results = tighten_and_measure_ppl(
            model=tightened_model,
            tokenizer=tokenizer,
            model_id=model_id,
            device=device,
            num_chunks=ppl_chunks,
            verbose=verbose,
            skip_ppl=not measure_ppl,  # Only measure PPL if requested
        )

        # Step 7b: Validate on tightened model (if enabled)
        if validate:
            if verbose:
                print(f"\n[Step 7b] Validating tightened model...")
            val_results = validate_model(
                model=tightened_model,
                tokenizer=tokenizer,
                device=device,
                verbose=verbose,
            )
            metrics['steps']['validate'] = val_results

            if not val_results['passed']:
                print("\n[ERROR] Validation failed!")
                if not verbose:
                    for err in val_results['errors']:
                        print(f"  {err}")

        # Store PPL results if measured
        if measure_ppl:
            metrics['steps']['perplexity'] = tighten_results

        # Save tightened checkpoint (both names for compatibility)
        tightened_path = Path(output_dir) / "v2_tightened.pt"
        tightq_path = Path(output_dir) / "tightQ_all.pt"
        tightened_state = tightened_model.state_dict()
        tightened_bytes = _estimate_state_dict_bytes(tightened_state)
        _safe_torch_save(tightened_state, tightened_path, required_bytes_hint=tightened_bytes)
        alias_kind = _create_checkpoint_alias(tightened_path, tightq_path)
        del tightened_state
        _release_memory(device)
        save_paths['tightened'] = str(tightened_path)
        save_paths['tightq_all'] = str(tightq_path)
        if verbose:
            ckpt_size_mb = tightened_path.stat().st_size / 1024 / 1024
            print(f"\n  Saved tightened checkpoints:")
            print(f"    {tightened_path} ({ckpt_size_mb:.1f} MB)")
            print(f"    {tightq_path} ({ckpt_size_mb:.1f} MB, {alias_kind})")

        # Clean up
        del tightened_model
        _release_memory(device)
    else:
        # No further model use after checkpoint save.
        del model
        _release_memory(device)
        if verbose:
            print(f"\n[Step 7-8] Tightening/Validation/PPL SKIPPED")
            print(f"  Use --inference-eval for validation, --ppl for perplexity")

    # Summary
    total_time = time.time() - total_start
    metrics['total_time_seconds'] = total_time

    if verbose:
        print(f"\n" + "=" * 60)
        print("INITIALIZATION COMPLETE")
        print("=" * 60)
        print(f"Total time: {total_time:.1f}s")
        print(f"Checkpoint: {save_paths['checkpoint']}")
        print(f"\nNext step: Train with")
        print(f"  python scripts/train_v2_simple.py \\")
        print(f"      --v2-checkpoint {save_paths['checkpoint']} \\")
        print(f"      --cache-dir caches/alpaca_L128 \\")
        print(f"      --mlp-only --auto-snap-mags \\")
        print(f"      --output-dir {output_dir}")

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description='Initialize V2 QAT model from scratch',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Presets:
  q2a4     : 2-bit MLP, 4-bit Attn, rank=32/8 [smallest]
  q2a2     : 2-bit MLP+Attn, rank=32/8 [extreme compression]
  q4a4     : 4-bit MLP+Attn, rank=32 [balanced, default]
  q4a4_r32 : Same as q4a4
  q4_r64   : 4-bit MLP+Attn, rank=64 [higher quality]

Examples:
  # Basic initialization
  python scripts/init_model_v2.py --output runs/my_run

  # Use 2-bit preset
  python scripts/init_model_v2.py --config q2a4 --output runs/q2_run

  # Use different base model
  python scripts/init_model_v2.py --model-id Qwen/Qwen3-1.8B --output runs/1.8b
""",
    )

    # Output (required unless --list-presets)
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output directory for checkpoint and config')

    # Model selection
    parser.add_argument('--model-id', type=str, default='Qwen/Qwen3-0.6B',
                        help='HuggingFace model ID (default: Qwen/Qwen3-0.6B)')

    # Quantization config
    parser.add_argument('--config', '-c', type=str, default='q4a4',
                        choices=list(PRESETS.keys()),
                        help='Quantization preset (default: q4a4)')

    # Custom quantization (override preset)
    parser.add_argument('--mlp-lut-bits', type=int, default=None,
                        help='Override MLP LUT bits')
    parser.add_argument('--mlp-rank', type=int, default=None,
                        help='Override MLP scale rank')
    parser.add_argument('--attn-lut-bits', type=int, default=None,
                        help='Override Attention LUT bits')
    parser.add_argument('--attn-rank', type=int, default=None,
                        help='Override Attention scale rank')
    parser.add_argument('--group-size', type=int, default=32,
                        help='Group size for SVD scale initialization (default: 32)')

    # Layer selection
    parser.add_argument('--mlp-only', action='store_true',
                        help='Only quantize MLP layers (skip attention)')

    # Device (CPU is default for model construction - TPU only needed for PPL measurement)
    parser.add_argument('--tpu', action='store_true',
                        help='Force TPU mode (not recommended for init, use for PPL only)')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU mode (default)')

    # Evaluation options
    parser.add_argument('--no-eval', action='store_true',
                        help='Skip ALL evaluation (inference and perplexity)')
    parser.add_argument('--no-validate', action='store_true',
                        help='Skip validation tests (alias for --no-eval)')
    parser.add_argument('--inference-eval', action='store_true',
                        help='Run inference validation tests (off by default)')
    parser.add_argument('--ppl', action='store_true',
                        help='Measure perplexity (tighten Q + quick_perplexity)')
    parser.add_argument('--ppl-chunks', type=int, default=14,
                        help='Number of chunks for perplexity (default: 14)')
    parser.add_argument('--svd-error', action='store_true',
                        help='Measure SVD approximation error (MAE vs original weights)')
    parser.add_argument('--search-group', type=str, default=None,
                        help='Search optimal group size per tensor. Comma-separated sizes (e.g., "128,64,32,16")')
    parser.add_argument('--search-lut', action='store_true',
                        help='Search optimal LUT per tensor (4-bit layers only). Tests: uniform, fp4_dense')
    parser.add_argument('--lut', type=str, default=None,
                        help='Use specific LUT for all 4-bit layers (skips search). Options: uniform, fp4_dense')
    parser.add_argument('--imatrix', type=str, default=None,
                        help='Path to importance matrix (.pt file) for iMSE scoring. Use compute_imatrix.py to generate.')
    parser.add_argument('--awq-alpha', type=float, default=0.0,
                        help='[NOT IMPLEMENTED] AWQ-style importance weighting. '
                             'AWQ requires applying inverse scaling to RMSNorm, which is not yet implemented. '
                             'Using this option will print an error and skip AWQ. '
                             'TODO: Implement proper AWQ with RMSNorm compensation for Qwen3.')

    # Other options
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Minimal output')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed LUT values and MSE per layer during freeze')

    # List presets
    parser.add_argument('--list-presets', action='store_true',
                        help='List available presets and exit')

    args = parser.parse_args()

    # List presets
    if args.list_presets:
        print("Available presets:")
        print("-" * 60)
        for name, preset in PRESETS.items():
            print(f"  {name:12s} : {preset.description}")
        return 0

    # Validate --output is provided
    if args.output is None:
        parser.error("--output is required")

    # Warn about --awq-alpha (not implemented)
    if args.awq_alpha > 0:
        print(f"\nWARNING: --awq-alpha is NOT IMPLEMENTED and will be IGNORED.")
        print(f"  AWQ requires inverse scaling in RMSNorm, which is not implemented.")
        print(f"  Your checkpoint will be created without AWQ weighting.\n")

    # Get preset and apply overrides
    preset = PRESETS[args.config]

    if args.mlp_lut_bits is not None:
        preset = QuantPreset(
            name=f"custom_{preset.name}",
            mlp_lut_bits=args.mlp_lut_bits,
            mlp_rank=args.mlp_rank or preset.mlp_rank,
            attn_lut_bits=args.attn_lut_bits or preset.attn_lut_bits,
            attn_rank=args.attn_rank or preset.attn_rank,
            description=f"Custom based on {preset.name}",
        )
    elif args.mlp_rank is not None or args.attn_lut_bits is not None or args.attn_rank is not None:
        preset = QuantPreset(
            name=f"custom_{preset.name}",
            mlp_lut_bits=preset.mlp_lut_bits,
            mlp_rank=args.mlp_rank or preset.mlp_rank,
            attn_lut_bits=args.attn_lut_bits or preset.attn_lut_bits,
            attn_rank=args.attn_rank or preset.attn_rank,
            description=f"Custom based on {preset.name}",
        )
        PRESETS[preset.name] = preset

    # Handle evaluation flags
    # Validation is OFF by default - only run if --inference-eval is explicitly passed
    run_validation = args.inference_eval

    # Parse --search-group argument (comma-separated string -> list of ints)
    search_group_sizes = None
    if args.search_group:
        try:
            search_group_sizes = [int(x.strip()) for x in args.search_group.split(',')]
            if not args.quiet:
                print(f"[INFO] Will search optimal group sizes: {search_group_sizes}")
        except ValueError as e:
            parser.error(f"Invalid --search-group format: {args.search_group}. Expected comma-separated integers (e.g., '128,64,32,16')")

    # Run pipeline
    try:
        metrics = init_v2_model(
            model_id=args.model_id,
            preset_name=preset.name if preset.name in PRESETS else args.config,
            output_dir=args.output,
            force_tpu=args.tpu,
            force_cpu=not args.tpu,  # Default to CPU for model construction
            quantize_attn=not args.mlp_only,
            group_size=args.group_size,
            validate=run_validation,
            measure_ppl=args.ppl,
            ppl_chunks=args.ppl_chunks,
            measure_svd_error=args.svd_error,
            search_group_sizes=search_group_sizes,
            search_lut=args.search_lut,
            default_lut=args.lut,
            imatrix_path=args.imatrix,
            awq_alpha=args.awq_alpha,
            verbose=not args.quiet,
            detailed=args.verbose,
        )

        return 0 if metrics.get('steps', {}).get('validate', {}).get('passed', True) else 1

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
