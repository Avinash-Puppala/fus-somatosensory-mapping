"""
npkit.data — synthetic fUS trial generator (NumPy only).

This is a faithful port of src/generate_data.py. The simulator physics are
unchanged so that results remain directly comparable to the original pipeline:
  * 5 finger patches laid out horizontally across S1 (somatotopic order)
  * gamma-shaped HRF with domain-randomized peak timing (4-10 s)
  * trial-level randomization of noise, signal strength, patch radius,
    reliability (dropout), and cross-patch leakage
  * cardiac (~1 Hz) and respiratory (~0.3 Hz) physiological oscillations

The ONLY substantive change vs. the original is that the HRF comes from
npkit.stats.gamma_pdf instead of scipy.stats.gamma, removing the scipy
dependency. generate_hrf() reproduces the original's normalization (peak = 1).
"""

from __future__ import annotations
import numpy as np

from .stats import gamma_pdf

FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]


def generate_hrf(duration: float = 20, tr: float = 1.0, peak_time: float = 6.0):
    """Canonical single-gamma HRF, normalized to unit peak (as in the original)."""
    t = np.arange(0, duration, tr)
    scale = max(peak_time / 5.0, 0.5)      # mode of Gamma(6, scale) = 5*scale = peak_time
    hrf = gamma_pdf(t, a=6, scale=scale)
    hrf = hrf / hrf.max()
    return hrf, t


def generate_finger_patches(image_size: int = 128, patch_radius: int = 8):
    """Boolean S1 patch masks + center coordinates for each finger."""
    cols = np.linspace(0.20, 0.80, 5) * image_size
    row = 0.5 * image_size
    rows_grid, cols_grid = np.mgrid[0:image_size, 0:image_size]
    centers, patches = {}, {}
    for i, name in enumerate(FINGER_NAMES):
        center = (int(row), int(cols[i]))
        centers[name] = center
        dist = np.sqrt((rows_grid - center[0]) ** 2 + (cols_grid - center[1]) ** 2)
        patches[name] = dist <= patch_radius
    return patches, centers


def generate_trials(
    n_trials_per_finger: int = 100,
    image_size: int = 128,
    patch_radius: int = 8,
    signal_strength: float = 0.15,
    noise_std: float = 0.05,
    tr: float = 1.0,
    trial_duration: int = 20,
    trial_reliability: float = 0.85,
    leakage_fraction: float = 0.15,
    domain_randomize: bool = True,
    seed: int | None = None,
):
    """
    Generate synthetic fUS trials for all 5 fingers.

    Returns
    -------
    X       : (n_trials, n_voxels, n_timepoints)
    y       : (n_trials,) finger labels 0-4
    patches : dict name -> 2D boolean mask  (for visualization)
    centers : dict name -> (row, col)        (for visualization)
    """
    rng = np.random.default_rng(seed)
    n_fingers = len(FINGER_NAMES)
    n_trials = n_trials_per_finger * n_fingers

    if domain_randomize:
        trial_noise_stds       = rng.uniform(0.03, 0.10, n_trials)
        trial_signal_strengths = rng.uniform(0.08, 0.25, n_trials)
        trial_peak_times       = rng.uniform(4.0, 10.0, n_trials)
        trial_reliabilities    = rng.uniform(0.70, 0.95, n_trials)
        trial_leakages         = rng.uniform(0.05, 0.30, n_trials)
        trial_patch_radii      = rng.integers(6, 13, n_trials)
    else:
        trial_noise_stds       = np.full(n_trials, noise_std)
        trial_signal_strengths = np.full(n_trials, signal_strength)
        trial_peak_times       = np.full(n_trials, 6.0)
        trial_reliabilities    = np.full(n_trials, trial_reliability)
        trial_leakages         = np.full(n_trials, leakage_fraction)
        trial_patch_radii      = np.full(n_trials, patch_radius, dtype=int)

    patches, centers = generate_finger_patches(image_size, patch_radius)

    n_timepoints = int(trial_duration / tr)
    n_voxels = image_size * image_size
    # float32 halves memory (the 16k-voxel x 20-t arrays are the dominant cost)
    # and speeds up the linear algebra, with no measurable accuracy impact here.
    X = np.zeros((n_trials, n_voxels, n_timepoints), dtype=np.float32)
    y = np.zeros(n_trials, dtype=int)

    # Precompute per-finger distance maps once (vectorized masking later).
    rows_grid, cols_grid = np.mgrid[0:image_size, 0:image_size]
    dist_maps = {}
    for fname in FINGER_NAMES:
        cx, cy = centers[fname]
        dist_maps[fname] = np.sqrt(
            (rows_grid - cx) ** 2 + (cols_grid - cy) ** 2
        ).flatten()

    t = np.arange(n_timepoints) * tr
    trial_idx = 0
    for finger_idx, name in enumerate(FINGER_NAMES):
        adjacent = []
        if finger_idx > 0:
            adjacent.append(FINGER_NAMES[finger_idx - 1])
        if finger_idx < n_fingers - 1:
            adjacent.append(FINGER_NAMES[finger_idx + 1])

        for _ in range(n_trials_per_finger):
            ns   = trial_noise_stds[trial_idx]
            sig  = trial_signal_strengths[trial_idx]
            peak = trial_peak_times[trial_idx]
            rel  = trial_reliabilities[trial_idx]
            leak = trial_leakages[trial_idx]
            rad  = trial_patch_radii[trial_idx]

            trial = rng.normal(0.0, ns, (n_voxels, n_timepoints))

            cardiac_amp     = 0.2  * ns * rng.uniform(0.5, 1.5)
            respiratory_amp = 0.15 * ns * rng.uniform(0.5, 1.5)
            physio = (
                cardiac_amp     * np.sin(2 * np.pi * 1.0 * t + rng.uniform(0, 2 * np.pi)) +
                respiratory_amp * np.sin(2 * np.pi * 0.3 * t + rng.uniform(0, 2 * np.pi))
            )
            trial += physio[np.newaxis, :]

            if rng.random() < rel:
                hrf, _ = generate_hrf(duration=trial_duration, tr=tr, peak_time=peak)
                active = np.where(dist_maps[name] <= rad)[0]
                scales = sig * rng.uniform(0.8, 1.2, len(active))
                trial[active] += scales[:, np.newaxis] * hrf
                for adj in adjacent:
                    av = np.where(dist_maps[adj] <= rad)[0]
                    asc = sig * leak * rng.uniform(0.8, 1.2, len(av))
                    trial[av] += asc[:, np.newaxis] * hrf

            X[trial_idx] = trial
            y[trial_idx] = finger_idx
            trial_idx += 1

    shuffle = rng.permutation(n_trials)
    return X[shuffle], y[shuffle], patches, centers
