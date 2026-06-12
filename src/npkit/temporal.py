"""
npkit.temporal — temporal feature extractors that collapse a voxel timecourse
(length T) into a single number per voxel per trial.

This is the heart of the accuracy work. The original pipeline collapses the
response by averaging a fixed 3-8 s window. That has two weaknesses:
  * a flat mean ignores the HRF *shape* (it weights noise timepoints equally
    with the peak), and
  * a fixed window cannot follow the per-trial HRF peak jitter (4-10 s here).

A matched filter fixes both: it correlates each timecourse with the expected
HRF shape, which is the optimal linear detector of a known signal in additive
noise. We provide four extractors behind one fit/transform interface so they
can be ablated head-to-head under identical CV:

    'fixed'        mean over a fixed window           (reproduces the original)
    'adaptive'     mean over a train-estimated window (follows the average peak)
    'matched'      projection onto ONE empirical HRF template estimated from
                   the training task voxels                (shape-aware)
    'matchedbank'  max projection over a bank of HRF templates spanning peak
                   times 4-10 s                  (shape-aware AND timing-robust)

All "learning" (window/template estimation) happens in fit() on training data
only, so transform() on test trials introduces no leakage.
"""

from __future__ import annotations
import numpy as np

from .data import generate_hrf
from .preprocess import estimate_peak_window


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


class TemporalFeature:
    def __init__(self, kind: str = "matchedbank", fixed_window=(3, 8),
                 window_width: int = 5, bank_peaks=None, n_bank: int = 7,
                 tr: float = 1.0):
        """
        bank_peaks : explicit HRF peak times (s) for the matched bank. If None,
                     the bank is built adaptively from the data: n_bank peaks
                     spread across the plausible HRF range of THIS recording
                     (derived from its timepoint count and TR). Pass an explicit
                     tuple only to pin the bank for controlled comparisons.
        tr         : seconds per timepoint of the dataset being decoded.
        """
        self.kind = kind
        self.fixed_window = fixed_window
        self.window_width = window_width
        self.bank_peaks = bank_peaks
        self.n_bank = n_bank
        self.tr = tr

    def _bank_peaks_for(self, n_timepoints: int):
        """Adaptive peak times (s) spanning ~15%-75% of the trial, or explicit."""
        if self.bank_peaks is not None:
            return self.bank_peaks
        duration = n_timepoints * self.tr
        lo = max(self.tr, 0.15 * duration)
        hi = max(lo + self.tr, 0.75 * duration)
        return np.linspace(lo, hi, self.n_bank)

    # -- fit: estimate any train-derived window/template -------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "TemporalFeature":
        T = X.shape[2]
        duration = T * self.tr
        b = min(3, T // 4)   # baseline points used to de-mean templates
        if self.kind == "fixed":
            self.window_ = self.fixed_window
        elif self.kind == "adaptive":
            self.window_ = estimate_peak_window(X, y, width=self.window_width)
        elif self.kind == "matched":
            # empirical template = average timecourse of the most task-responsive
            # voxels (between-condition variance), baseline-removed + unit-norm.
            cond_means = np.stack([X[y == c].mean(axis=0) for c in np.unique(y)])
            resp = cond_means.var(axis=0).mean(axis=1)
            top = np.argsort(resp)[-max(1, X.shape[1] // 50):]
            tmpl = X[:, top, :].mean(axis=(0, 1))
            self.template_ = _unit(tmpl - tmpl[:b].mean()).astype(np.float32)
        elif self.kind == "matchedbank":
            bank = []
            for pk in self._bank_peaks_for(T):
                hrf, _ = generate_hrf(duration, self.tr, peak_time=float(pk))
                bank.append(_unit(hrf - hrf[:b].mean()))
            self.bank_ = np.array(bank, dtype=np.float32)   # (K, T)
        else:
            raise ValueError(f"unknown temporal feature kind: {self.kind}")
        return self

    # -- transform: collapse (n, V, T) -> (n, V) ---------------------------
    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.kind in ("fixed", "adaptive"):
            s, e = self.window_
            return X[:, :, s:e].mean(axis=2)
        if self.kind == "matched":
            return np.tensordot(X, self.template_, axes=([2], [0]))  # (n, V)
        if self.kind == "matchedbank":
            # max projection across the template bank; loop over K to keep the
            # peak memory flat (avoid materializing (n, V, K) at once).
            out = None
            for k in range(self.bank_.shape[0]):
                proj = np.tensordot(X, self.bank_[k], axes=([2], [0]))  # (n, V)
                out = proj if out is None else np.maximum(out, proj)
            return out
        raise ValueError(self.kind)
