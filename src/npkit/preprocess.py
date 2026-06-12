"""
npkit.preprocess — baseline normalization, adaptive HRF window, feature selection.

Two improvements over the original src/preprocess.py:

  1. estimate_peak_window(): the original hard-codes the HRF "memory period" to
     timepoints 3-8. With domain-randomized peak timing (4-10 s) that window is
     wrong for many trials. Here we estimate it from the data: average the most
     task-responsive voxels' timecourses and take the window around the group
     peak. Estimated on TRAINING data only (caller passes train trials), so it
     never leaks test information.

  2. select_features() uses npkit.stats.f_sf (no scipy) and accepts the window
     so feature selection and decoding share one consistent temporal window.

Everything is NumPy. The ANOVA F-statistic is computed in closed form across all
voxels at once; only the p-value tail call loops (once per fold, not per epoch).
"""

from __future__ import annotations
import numpy as np

from .stats import f_sf


def baseline_normalize(X: np.ndarray, n_baseline_timepoints: int = 3) -> np.ndarray:
    """Express each voxel as change from its own pre-stimulus baseline mean."""
    baseline = X[:, :, :n_baseline_timepoints].mean(axis=2, keepdims=True)
    return X - baseline


def estimate_peak_window(X: np.ndarray, y: np.ndarray, width: int = 5,
                         top_frac: float = 0.02) -> tuple[int, int]:
    """
    Adaptively locate the HRF response window from the data.

    Strategy
    --------
    1. For every voxel, measure how strongly its post-baseline signal varies
       across the 5 finger conditions (between-condition variance of the
       per-voxel timecourse). Task voxels score high; pure-noise voxels low.
    2. Average the timecourses of the top `top_frac` voxels -> a group HRF.
    3. The window is `width` timepoints centred on that group HRF's peak,
       clipped to valid range.

    Returns (start, end) with end exclusive, suitable for X[:, :, start:end].
    """
    n_trials, n_voxels, n_timepoints = X.shape

    # Per-voxel, per-condition mean timecourse, then variance across conditions.
    cond_means = np.stack(
        [X[y == c].mean(axis=0) for c in np.unique(y)], axis=0
    )  # (n_classes, n_voxels, n_timepoints)
    between_var = cond_means.var(axis=0).mean(axis=1)  # (n_voxels,) task-responsiveness

    k = max(1, int(top_frac * n_voxels))
    top = np.argsort(between_var)[-k:]
    group_hrf = X[:, top, :].mean(axis=(0, 1))  # (n_timepoints,) average response

    peak = int(np.argmax(group_hrf))
    half = width // 2
    start = max(0, peak - half)
    end = min(n_timepoints, start + width)
    start = max(0, end - width)  # keep width if clipped at the right edge
    return start, end


def collapse_window(X: np.ndarray, window: tuple[int, int]) -> np.ndarray:
    """Mean over the response window -> one value per voxel per trial."""
    s, e = window
    return X[:, :, s:e].mean(axis=2)


def anova_select(feat: np.ndarray, y: np.ndarray, q_threshold: float = 0.05):
    """
    Benjamini-Hochberg FDR-controlled one-way ANOVA on an already-collapsed
    feature matrix `feat` of shape (n_trials, n_voxels). Returns (mask, p_values).

    This is the temporal-feature-agnostic version of select_features: whatever
    temporal extractor produced `feat` (mean window, matched filter, ...), the
    voxel selection is run on exactly the features the decoder will see.
    """
    n_trials, n_voxels = feat.shape
    classes = np.unique(y)
    n_groups = len(classes)
    masks = [y == c for c in classes]
    n_per = np.array([m.sum() for m in masks])
    group_means = np.array([feat[m].mean(axis=0) for m in masks])
    grand = feat.mean(axis=0)
    ss_between = sum(n_per[g] * (group_means[g] - grand) ** 2 for g in range(n_groups))
    ss_within  = sum(((feat[masks[g]] - group_means[g]) ** 2).sum(axis=0)
                     for g in range(n_groups))
    df_b, df_w = n_groups - 1, n_trials - n_groups
    f_stats = (ss_between / df_b) / np.maximum(ss_within / df_w, 1e-12)
    p_values = f_sf(f_stats, df_b, df_w)

    order = np.argsort(p_values)
    ranks = np.arange(1, n_voxels + 1)
    bh = (ranks / n_voxels) * q_threshold
    passing = np.where(p_values[order] <= bh)[0]
    mask = np.zeros(n_voxels, dtype=bool)
    if len(passing) > 0:
        mask[order[: passing[-1] + 1]] = True
    return mask, p_values


def select_features(X: np.ndarray, y: np.ndarray, window: tuple[int, int],
                    q_threshold: float = 0.05):
    """
    One-way ANOVA per voxel over the 5 finger conditions, with Benjamini-Hochberg
    FDR control. Returns the boolean voxel mask plus raw p-values.

    Parameters
    ----------
    X      : (n_trials, n_voxels, n_timepoints) baseline-normalized
    y      : (n_trials,) labels 0-4
    window : (start, end) response window used to collapse time before the test
    """
    n_trials, n_voxels, _ = X.shape
    feat = collapse_window(X, window)  # (n_trials, n_voxels)

    classes = np.unique(y)
    n_groups = len(classes)
    masks = [y == c for c in classes]
    n_per = np.array([m.sum() for m in masks])
    group_means = np.array([feat[m].mean(axis=0) for m in masks])  # (G, V)
    grand = feat.mean(axis=0)                                      # (V,)

    ss_between = sum(n_per[g] * (group_means[g] - grand) ** 2 for g in range(n_groups))
    ss_within  = sum(((feat[masks[g]] - group_means[g]) ** 2).sum(axis=0)
                     for g in range(n_groups))
    df_b, df_w = n_groups - 1, n_trials - n_groups
    f_stats = (ss_between / df_b) / np.maximum(ss_within / df_w, 1e-12)
    p_values = f_sf(f_stats, df_b, df_w)

    # Benjamini-Hochberg
    order = np.argsort(p_values)
    ranks = np.arange(1, n_voxels + 1)
    bh = (ranks / n_voxels) * q_threshold
    passing = np.where(p_values[order] <= bh)[0]
    mask = np.zeros(n_voxels, dtype=bool)
    if len(passing) > 0:
        mask[order[: passing[-1] + 1]] = True
    return mask, p_values
