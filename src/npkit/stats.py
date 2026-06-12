"""
npkit.stats — minimal, dependency-light statistical primitives (NumPy only).

Why this exists
---------------
The original pipeline used scipy for exactly two things:
    * scipy.stats.gamma.pdf  -> the hemodynamic response function (HRF) shape
    * scipy.stats.f.sf       -> p-values for the per-voxel ANOVA

Neither is hard to compute directly, and replacing them removes scipy from the
dependency tree (which matters because the sandbox cannot install it). Every
function here is unit-tested against known reference values in
`tests/test_stats.py`.

Nothing in this file is approximate beyond floating-point: the incomplete-beta
continued fraction below is the same algorithm scipy/Numerical-Recipes use.
"""

from __future__ import annotations
import math
import numpy as np


# ---------------------------------------------------------------------------
# Gamma-shaped hemodynamic response function
# ---------------------------------------------------------------------------
def gamma_pdf(t: np.ndarray, a: float, scale: float) -> np.ndarray:
    """
    Probability density of a Gamma(shape=a, scale=scale) distribution.

        f(t) = t^(a-1) * exp(-t/scale) / (scale^a * Gamma(a))

    We work in log-space and exponentiate so large factorials never overflow.
    Defined to be 0 for t <= 0 (the HRF has no response before the stimulus).

    Parameters
    ----------
    t     : array of time points (seconds)
    a     : shape parameter (a=6 gives the canonical single-gamma HRF)
    scale : scale parameter; the mode (peak) sits at (a-1)*scale

    Returns
    -------
    array, same shape as t
    """
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t)
    pos = t > 0
    tp = t[pos]
    log_pdf = (
        (a - 1.0) * np.log(tp)
        - tp / scale
        - a * math.log(scale)
        - math.lgamma(a)
    )
    out[pos] = np.exp(log_pdf)
    return out


# ---------------------------------------------------------------------------
# Regularized incomplete beta function  I_x(a, b)
# ---------------------------------------------------------------------------
def _betacf(a: float, b: float, x: float, itmax: int = 200, eps: float = 3e-16) -> float:
    """Continued fraction for the incomplete beta (Lentz's algorithm)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        # even step
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        # odd step
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def betai(a: float, b: float, x: float) -> float:
    """
    Regularized incomplete beta function I_x(a, b), scalar.

    I_x(a,b) = B(x;a,b) / B(a,b), the CDF backbone for the Beta, Student-t,
    and F distributions. Uses the standard symmetry trick to keep the
    continued fraction in its fast-converging regime.
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_beta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(ln_beta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


# ---------------------------------------------------------------------------
# F-distribution survival function  P(F > f)
# ---------------------------------------------------------------------------
def f_sf(f: np.ndarray, d1: float, d2: float) -> np.ndarray:
    """
    Upper-tail probability P(F > f) for an F(d1, d2) distribution.

    Identity used:
        P(F > f) = I_x(d2/2, d1/2),  with  x = d2 / (d2 + d1*f)

    Vectorized over an array of F statistics (one per voxel). betai is scalar,
    so we loop — for ~16k voxels this is still well under a second, and it runs
    once per CV fold, not per epoch.
    """
    f = np.asarray(f, dtype=float)
    flat = f.ravel()
    out = np.empty_like(flat)
    half_d1, half_d2 = d1 / 2.0, d2 / 2.0
    for i, fi in enumerate(flat):
        if fi <= 0:
            out[i] = 1.0
        else:
            x = d2 / (d2 + d1 * fi)
            out[i] = betai(half_d2, half_d1, x)
    return out.reshape(f.shape)
