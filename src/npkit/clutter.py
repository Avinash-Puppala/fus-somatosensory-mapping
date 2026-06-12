"""
npkit.clutter — SVD clutter / nuisance filtering for fUS (NumPy only).

Background
----------
Functional ultrasound separates blood flow from surrounding tissue by spectral
content of the slow-time (ensemble) signal. The standard tool is the SVD clutter
filter (Demene et al., IEEE TMI 2015): stack an ensemble of compounded frames
into a Casorati matrix (slow_time x voxels), take its SVD, and discard the first
few singular components — these capture the high-energy, spatially-coherent,
low-frequency *tissue* motion (clutter). Optionally discard the last components
(spatially-incoherent thermal noise). What remains is the blood signal, from
which a power-Doppler image is formed.

Two entry points, for the two shapes real data arrives in:

  * svd_clutter_filter() / power_doppler()  — for RAW ensemble IQ data, when a
    dataset ships pre-beamformed slow-time stacks. Produces clutter-free frames.

  * remove_low_rank_nuisance()  — for ALREADY power-Doppler timeseries (what most
    published decoding datasets distribute). Removes global low-rank confounds
    (slow drift, motion-correlated global hemodynamics) across the recording,
    which is the decoding-level analogue of clutter rejection.

Both are plain SVD; nothing here needs scipy.
"""

from __future__ import annotations
import numpy as np


def svd_clutter_filter(ensemble: np.ndarray, n_clutter: int = 1,
                       n_noise: int = 0) -> np.ndarray:
    """
    Remove tissue clutter (and optionally thermal noise) from one ensemble.

    Parameters
    ----------
    ensemble : (n_slowtime, n_voxels) real or complex Casorati matrix — the
               compounded frames of a single power-Doppler acquisition.
    n_clutter: number of leading singular components to remove (tissue clutter).
    n_noise  : number of trailing singular components to remove (white noise).

    Returns
    -------
    filtered : same shape as `ensemble`, clutter/noise components projected out.
    """
    if ensemble.ndim != 2:
        raise ValueError(f"ensemble must be 2-D (slow_time, voxels); got {ensemble.shape}")
    nt = ensemble.shape[0]
    U, S, Vh = np.linalg.svd(ensemble, full_matrices=False)
    keep = np.ones_like(S)
    keep[:n_clutter] = 0.0                       # drop tissue clutter
    if n_noise > 0:
        keep[len(S) - n_noise:] = 0.0            # drop thermal noise
    return (U * (S * keep)) @ Vh


def power_doppler(ensemble: np.ndarray, n_clutter: int = 1,
                  n_noise: int = 0) -> np.ndarray:
    """
    Clutter-filter an ensemble and reduce it to a power-Doppler image:
    the mean power (|signal|^2) over slow time, per voxel.

    Returns (n_voxels,). Apply once per acquired frame to build a fUS movie.
    """
    filt = svd_clutter_filter(ensemble, n_clutter, n_noise)
    return (np.abs(filt) ** 2).mean(axis=0)


def estimate_clutter_rank(ensemble: np.ndarray, energy_gap: float = 0.5) -> int:
    """
    Heuristic for how many leading components are clutter: clutter singular
    values dominate, so look for the first large drop in the singular spectrum
    (ratio of consecutive singular values below `energy_gap`). Returns >=1.
    """
    S = np.linalg.svd(ensemble, compute_uv=False)
    if len(S) < 2:
        return 1
    ratios = S[1:] / np.maximum(S[:-1], 1e-12)
    drops = np.where(ratios < energy_gap)[0]
    return int(drops[0] + 1) if len(drops) else 1


def _randomized_left_topk(M: np.ndarray, k: int, n_iter: int = 2,
                          oversample: int = 5, seed: int = 0) -> np.ndarray:
    """
    Approximate the top-k left singular vectors of M (n_rows x n_cols) via a
    randomized range finder with a couple of power iterations. Returns an
    orthonormal (n_rows, k) basis. Far cheaper than a full SVD when we only need
    a handful of dominant components (Halko, Martinsson & Tropp 2011).
    """
    rng = np.random.default_rng(seed)
    p = min(k + oversample, M.shape[1])
    Omega = rng.standard_normal((M.shape[1], p)).astype(M.dtype)
    Y = M @ Omega                                         # (rows, p)
    for _ in range(n_iter):                               # sharpen toward top space
        Y = M @ (M.T @ Y)
    Q, _ = np.linalg.qr(Y)                                # (rows, p) orthonormal
    B = Q.T @ M                                           # (p, cols) — small
    Ub, _, _ = np.linalg.svd(B, full_matrices=False)
    return Q @ Ub[:, :k]                                  # (rows, k)


def remove_low_rank_nuisance(X: np.ndarray, n_components: int = 1) -> np.ndarray:
    """
    Decoding-level nuisance removal for already power-Doppler data.

    X : (n_trials, n_voxels, n_timepoints). We build a voxels x (trials*time)
    matrix and project out its leading `n_components` spatial modes (the global
    low-rank structure common across the recording — drift, global CBV,
    motion-correlated signal), then reshape back. This protects against a decoder
    locking onto a global confound rather than the local task response.

    Uses a randomized top-k SVD so it stays fast and memory-light even at
    ~16k voxels (a full SVD there is intractable).
    """
    n, v, t = X.shape
    M = np.transpose(X, (1, 0, 2)).reshape(v, n * t).astype(np.float32)  # (voxels, trials*time)
    M = M - M.mean(axis=1, keepdims=True)
    Uk = _randomized_left_topk(M, k=n_components)         # (voxels, k)
    M_filt = M - Uk @ (Uk.T @ M)                          # project out the modes
    return np.transpose(M_filt.reshape(v, n, t), (1, 0, 2)).astype(X.dtype)
