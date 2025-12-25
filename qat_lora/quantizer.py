"""
Apple-style fake quantizer for low-bit (2/4-bit) QAT.

Apple's tech report (2025) describes simulating quantization in the forward pass:

    W_tilde = s * ( clamp( round(W/s + z), qmin, qmax ) - z )

and using the straight-through estimator (STE) for backprop through rounding.

Apple also:
- uses a learnable scaling factor f per tensor (not derived purely from W)
- emphasizes careful initialization of f in the 2-bit regime via a Newton-inspired clip estimator
- finds a "balanced" 2-bit set {-1.5, -0.5, 0.5, 1.5} is smoother than {-2, -1, 0, 1}

This repo generalizes the same *balanced* (no-zero) construction to 4-bit by using
half-integer codebooks:
  - 2-bit: {-1.5, -0.5, 0.5, 1.5}
  - 4-bit: {-7.5, -6.5, ..., 6.5, 7.5}  (16 levels)

This module implements those pieces.

NOTE: Apple does not publish exact pseudocode for their Newton-like rebalancing algorithm.
We implement a Newton-like *MSE-minimizing clipping scalar* routine as a reasonable stand-in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class QATQuantConfig:
    """
    Quantization config for Apple-style low-bit weight QAT.

    We want the *dequantized* codebook to be "balanced" (no zero level) using
    half-integer centers.

    One way to realize that with Apple's (qmin,qmax,z) formulation is to use:
      z = 0.5 and choose an integer range [qmin, qmax] of size 2^b such that
      (q - z) produces half-integers.

    For b=2:
      qmin=-1, qmax=2 -> (q - 0.5) gives {-1.5, -0.5, 0.5, 1.5}

    For b=4:
      qmin=-7, qmax=8 -> (q - 0.5) gives {-7.5, -6.5, ..., 6.5, 7.5}
    """
    n_bits: int = 2
    z: float = 0.5

    def __post_init__(self):
        if self.n_bits not in (2, 4):
            raise ValueError(f"Only 2 or 4 bits supported (got {self.n_bits})")

    @property
    def qmin(self) -> int:
        # 2-bit: -1, 4-bit: -7
        return -((1 << (self.n_bits - 1)) - 1)

    @property
    def qmax(self) -> int:
        # 2-bit: 2, 4-bit: 8
        return 1 << (self.n_bits - 1)

    # Scale formula described by Apple:
    #   s = f * max(|W|) / qmax
    #
    # Here qmax is positive (2).
    # We treat max(|W|) as a constant (detach) because max is non-differentiable and
    # Apple explicitly introduces a *learnable* f instead.
    def compute_scale(self, weight: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        amax = weight.detach().abs().max().clamp(min=1e-8)
        return f * amax / float(self.qmax)


def _round_ste(x: torch.Tensor) -> torch.Tensor:
    """
    Straight-through rounding.
    Forward: round(x)
    Backward: identity (dx/dx = 1)
    """
    return x + (torch.round(x) - x).detach()


def fake_quant_weight_nbit(
    weight: torch.Tensor,
    f: torch.Tensor,
    qc: QATQuantConfig,
) -> torch.Tensor:
    """
    Fake-quantize a weight tensor using Apple-style formula with STE on rounding.

    weight: FP weight tensor
    f: learnable scalar (or per-tensor parameter) controlling quant range
    qc: quant config (qmin,qmax,z)

    Returns:
        dequantized tensor (same dtype as weight), but constrained to 2^b levels per scale.
    """
    s = qc.compute_scale(weight, f)

    # Normalize into "quant space"
    # Apple's formula: round(W/s + z)
    w_scaled = weight / s + qc.z

    # STE through rounding. We keep clamp gradients (0 outside range).
    w_rounded = _round_ste(w_scaled)

    # Clamp to representable integer codes
    w_q = torch.clamp(w_rounded, qc.qmin, qc.qmax)

    # Dequantize back
    w_deq = (w_q - qc.z) * s
    return w_deq


def fake_quant_weight_2bit(weight: torch.Tensor, f: torch.Tensor, qc: Optional[QATQuantConfig] = None) -> torch.Tensor:
    """
    Back-compat wrapper. Prefer fake_quant_weight_nbit(..., QATQuantConfig(n_bits=...)).
    """
    qc2 = qc if qc is not None else QATQuantConfig(n_bits=2)
    # If caller passed a config with n_bits != 2, honor the wrapper name and force 2-bit.
    qc2 = QATQuantConfig(n_bits=2, z=float(getattr(qc2, "z", 0.5)))
    return fake_quant_weight_nbit(weight, f, qc2)

@torch.no_grad()
def _mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.mean((a - b) ** 2)


@torch.no_grad()
def estimate_clip_scalar_newton(
    weight: torch.Tensor,
    qc: QATQuantConfig,
    iters: int = 4,
    sample_size: int = 65536,
    init_percentile: float = 99.5,
) -> float:
    """
    Newton-like clip estimator.

    Apple says:
      - initializing f is crucial for stable 2-bit QAT
      - they estimate a clipping scalar c that reflects the central distribution,
        mitigating outliers, via a Newton-Raphson-inspired iterative method
      - then initialize f_init = c / max(|W|)

    They do not publish full pseudocode. A common, principled approach is:
      pick c that minimizes MSE between W and quantize_dequantize(W; s=c/qmax).

    We implement Newton steps on that 1D objective.

    Returns:
        c (float) - recommended clipping scalar
    """
    w = weight.detach().float().flatten()

    # Subsample weights for speed (clipping is a robust statistic; sampling is OK).
    if w.numel() > sample_size:
        idx = torch.randint(0, w.numel(), (sample_size,), device=w.device)
        w = w[idx]

    abs_w = w.abs()
    amax = float(abs_w.max().clamp(min=1e-8))

    # Initialize from percentile to downweight outliers (robust start).
    # (Percentile on GPU isn't directly available; we use kthvalue approximation.)
    k = max(1, int((init_percentile / 100.0) * abs_w.numel()))
    k = min(k, abs_w.numel())
    c = float(abs_w.kthvalue(k).values)  # approx percentile

    c = max(1e-6, min(c, amax))

    def objective(c_scalar: float) -> float:
        s = torch.tensor(c_scalar / float(qc.qmax), device=w.device)
        # Quantize/dequantize with fixed s (not learnable in this inner objective)
        w_scaled = w / s + qc.z
        w_rounded = torch.round(w_scaled)  # no STE inside objective
        w_q = torch.clamp(w_rounded, qc.qmin, qc.qmax)
        w_deq = (w_q - qc.z) * s
        return float(_mse(w, w_deq))

    # Newton iterations on derivative f'(c)=0
    for _ in range(iters):
        f0 = objective(c)
        eps = max(1e-6, 0.01 * c)

        f_plus = objective(min(amax, c + eps))
        f_minus = objective(max(1e-6, c - eps))

        # central difference
        f1 = (f_plus - f_minus) / (2.0 * eps)
        # second derivative
        f2 = (f_plus - 2.0 * f0 + f_minus) / (eps * eps)

        # If curvature is tiny or negative (noisy), fall back to a small gradient step
        if abs(f2) < 1e-12:
            c = c - 0.1 * f1
        else:
            c = c - f1 / f2

        c = float(max(1e-6, min(c, amax)))

    return c


@torch.no_grad()
def init_f_from_weight(
    weight: torch.Tensor,
    qc: QATQuantConfig,
    method: str = "newton",
    newton_iters: int = 4,
    newton_samples: int = 65536,
    percentile: float = 99.5,
) -> float:
    """
    Compute f_init for a given weight tensor, following Apple's formula:

        f_init = c / max(|W|)

    where c is a robust clipping scalar.

    method:
      - "newton": Newton-like MSE-minimizing clipping (recommended)
      - "percentile": simple percentile clip (faster)
    """
    absmax = float(weight.detach().abs().max().clamp(min=1e-8))

    if method == "newton":
        c = estimate_clip_scalar_newton(
            weight=weight,
            qc=qc,
            iters=newton_iters,
            sample_size=newton_samples,
            init_percentile=percentile,
        )
    elif method == "percentile":
        w = weight.detach().float().flatten()
        abs_w = w.abs()
        k = max(1, int((percentile / 100.0) * abs_w.numel()))
        k = min(k, abs_w.numel())
        c = float(abs_w.kthvalue(k).values)
        c = float(max(1e-6, min(c, absmax)))
    else:
        raise ValueError(f"Unknown init method: {method}")

    f_init = float(c / absmax)
    # Keep f positive and not too tiny.
    return float(max(1e-4, f_init))


@torch.no_grad()
def init_f_mse_grid_search(
    weight: torch.Tensor,
    qc: QATQuantConfig,
    percentiles: tuple = (99.0, 99.3, 99.5, 99.7, 99.9, 100.0),
    sample_size: int = 65536,
) -> float:
    """
    MSE grid-search for optimal f_init.

    Tries multiple clipping percentiles and picks the one with lowest
    quantization MSE. More robust than single-percentile methods for 2-bit.

    Args:
        weight: Weight tensor to calibrate
        qc: Quantization config
        percentiles: Percentiles to try (default covers 99.0-100.0)
        sample_size: Number of weight samples for MSE calculation

    Returns:
        f_init: Optimal initialization for f parameter
    """
    w = weight.detach().float().flatten()

    # Subsample for speed
    if w.numel() > sample_size:
        idx = torch.randint(0, w.numel(), (sample_size,), device=w.device)
        w = w[idx]

    abs_w = w.abs()
    amax = float(abs_w.max().clamp(min=1e-8))

    best_mse = float('inf')
    best_f = 1.0

    for pct in percentiles:
        # Get clip value for this percentile
        if pct >= 100.0:
            c = amax
        else:
            k = max(1, int((pct / 100.0) * abs_w.numel()))
            k = min(k, abs_w.numel())
            c = float(abs_w.kthvalue(k).values)

        c = max(1e-6, min(c, amax))

        # Compute scale and quantize
        s = c / float(qc.qmax)
        w_scaled = w / s + qc.z
        w_rounded = torch.round(w_scaled)
        w_q = torch.clamp(w_rounded, qc.qmin, qc.qmax)
        w_deq = (w_q - qc.z) * s

        mse = float(torch.mean((w - w_deq) ** 2))

        if mse < best_mse:
            best_mse = mse
            best_f = c / amax

    return float(max(1e-4, best_f))


@torch.no_grad()
def calibrate_model_f_init(
    model: "torch.nn.Module",
    n_bits: int = 2,
    method: str = "mse_grid",
    verbose: bool = True,
) -> int:
    """
    Calibrate f_init for all QATLinear modules in a model.

    This should be called BEFORE training to give QAT a good starting point,
    especially important for 2-bit where bad init can be catastrophic.

    Args:
        model: Model with QATLinear modules
        n_bits: Bit width for calibration
        method: "mse_grid" (recommended), "newton", or "percentile"
        verbose: Print calibration progress

    Returns:
        Number of modules calibrated
    """
    from .qat_linear import QATLinear

    qc = QATQuantConfig(n_bits=n_bits)
    count = 0

    for name, module in model.named_modules():
        if isinstance(module, QATLinear):
            weight = module.weight

            if method == "mse_grid":
                f_init = init_f_mse_grid_search(weight, qc)
            elif method == "newton":
                f_init = init_f_from_weight(weight, qc, method="newton")
            elif method == "percentile":
                f_init = init_f_from_weight(weight, qc, method="percentile")
            else:
                raise ValueError(f"Unknown method: {method}")

            module.set_f_init(f_init)
            count += 1

            if verbose:
                print(f"  [calibrate] {name}: f_init={f_init:.4f}")

    if verbose:
        print(f"[calibrate] Calibrated {count} QATLinear modules with method='{method}'")

    return count
