"""
npkit.regress — continuous-target decoding (ridge regression) for fUS.

The classification path (LDA) answers "which condition?". Some fUS studies decode
a *continuous* quantity instead — self-motion velocity, head angle, a kinematic.
This module provides the regression analogue with the same temporal-feature and
PCA front-end, so switching tasks is a one-line change rather than a new pipeline.

RidgeDecoder is closed-form ridge regression in PCA space (no sklearn). The CV
routine mirrors npkit.evaluate but reports R^2 and Pearson r, and selects voxels
by correlation with the target (the regression analogue of the ANOVA selector).
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .temporal import TemporalFeature
from .decoder import PCA


@dataclass
class RegressionConfig:
    feature: str = "matchedbank"
    n_bank: int = 7
    tr: float = 1.0
    n_components: int = 20
    alpha: float = 1.0          # ridge strength
    top_voxels: int = 500       # keep this many by |corr| with target (per fold)
    n_folds: int = 5


class RidgeDecoder:
    """Ridge regression in PCA space.  w = (ZᵀZ + αI)⁻¹ Zᵀ(y - ȳ)."""

    def __init__(self, n_components: int = 20, alpha: float = 1.0):
        self.pca = PCA(n_components)
        self.alpha = alpha

    def fit(self, X2d: np.ndarray, y: np.ndarray) -> "RidgeDecoder":
        Z = self.pca.fit_transform(X2d)
        self.y_mean_ = y.mean()
        yc = y - self.y_mean_
        d = Z.shape[1]
        self.w_ = np.linalg.solve(Z.T @ Z + self.alpha * np.eye(d), Z.T @ yc)
        return self

    def predict(self, X2d: np.ndarray) -> np.ndarray:
        return self.pca.transform(X2d) @ self.w_ + self.y_mean_


def _kfold(n: int, n_folds: int, seed: int):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    for f in range(n_folds):
        test = idx[f::n_folds]
        train = np.setdiff1d(idx, test, assume_unique=False)
        yield train, test


def _corr_select(feat: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    """Top-k voxels by absolute Pearson correlation with the continuous target."""
    fc = feat - feat.mean(0)
    yc = y - y.mean()
    denom = np.linalg.norm(fc, axis=0) * np.linalg.norm(yc) + 1e-12
    corr = np.abs(fc.T @ yc) / denom
    mask = np.zeros(feat.shape[1], dtype=bool)
    mask[np.argsort(corr)[-min(k, feat.shape[1]):]] = True
    return mask


def cross_validate_regression(X_norm, y, cfg: RegressionConfig, seed: int = 0):
    """
    Leakage-free CV for continuous decoding. Temporal feature + voxel selection
    are fit on train only. Returns R^2, Pearson r, and predictions.
    """
    y = np.asarray(y, dtype=float)
    y_pred = np.zeros(len(y))
    for train, test in _kfold(len(y), cfg.n_folds, seed):
        Xtr, Xte = X_norm[train], X_norm[test]
        ytr = y[train]
        tf = TemporalFeature(kind=cfg.feature, n_bank=cfg.n_bank, tr=cfg.tr).fit(Xtr, ytr)
        ftr, fte = tf.transform(Xtr), tf.transform(Xte)
        mask = _corr_select(ftr, ytr, cfg.top_voxels)
        dec = RidgeDecoder(cfg.n_components, cfg.alpha).fit(ftr[:, mask], ytr)
        y_pred[test] = dec.predict(fte[:, mask])

    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    r = np.corrcoef(y, y_pred)[0, 1]
    return {"r2": float(r2), "pearson_r": float(r), "y_true": y, "y_pred": y_pred}
