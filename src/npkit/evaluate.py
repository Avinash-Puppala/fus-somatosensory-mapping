"""
npkit.evaluate — leakage-free cross-validation and seed-averaged evaluation.

Design goals
------------
* Everything that "learns" from data (HRF window estimation, ANOVA feature
  selection, PCA, LDA) is fit INSIDE the fold on training trials only. This is
  the fix for the circular-feature-selection bug noted in the README, applied
  consistently to the temporal window as well.
* Results are averaged over several random CV partitions (seeds) so a single
  lucky/unlucky split cannot masquerade as a real effect. We report mean +/- std.
* A `feature` switch ('fixed' vs 'adaptive') lets us ablate the adaptive-window
  improvement against the original hard-coded 3-8 s window.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from .preprocess import baseline_normalize, anova_select
from .temporal import TemporalFeature
from .decoder import LinearDecoder


@dataclass
class DecoderConfig:
    # temporal feature: 'fixed' | 'adaptive' | 'matched' | 'matchedbank'
    feature: str = "matchedbank"
    fixed_window: tuple = (3, 8)   # used when feature == 'fixed'
    window_width: int = 5          # used when feature == 'adaptive'
    bank_peaks: Optional[tuple] = None  # explicit peaks; None -> adaptive to data
    n_bank: int = 7                # number of templates when bank is adaptive
    tr: float = 1.0                # seconds per timepoint (from the dataset spec)
    n_components: int = 20
    shrinkage: float = 0.1
    q_threshold: float = 0.05
    n_folds: int = 5


def stratified_kfold(y: np.ndarray, n_folds: int, seed: int):
    """Yield (train_idx, test_idx) with class proportions preserved per fold."""
    rng = np.random.default_rng(seed)
    fold_of = np.empty(len(y), dtype=int)
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        fold_of[idx] = np.arange(len(idx)) % n_folds
    for f in range(n_folds):
        test = np.where(fold_of == f)[0]
        train = np.where(fold_of != f)[0]
        yield train, test


def confusion_matrix(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def _make_feature(cfg: DecoderConfig) -> TemporalFeature:
    return TemporalFeature(
        kind=cfg.feature,
        fixed_window=cfg.fixed_window,
        window_width=cfg.window_width,
        bank_peaks=cfg.bank_peaks,
        n_bank=cfg.n_bank,
        tr=cfg.tr,
    )


def cross_validate(X_norm, y, cfg: DecoderConfig, seed: int):
    """One full stratified CV pass. Returns dict of fold results for this seed."""
    n_classes = len(np.unique(y))
    n_voxels = X_norm.shape[1]
    y_true = np.zeros(len(y), dtype=int)
    y_pred = np.zeros(len(y), dtype=int)
    margins = np.zeros(len(y))
    per_fold, n_sel = [], []
    imp_accum = np.zeros(n_voxels)
    imp_count = np.zeros(n_voxels)

    for train, test in stratified_kfold(y, cfg.n_folds, seed):
        Xtr, Xte = X_norm[train], X_norm[test]
        ytr, yte = y[train], y[test]

        # 1) temporal feature: fit window/template on TRAIN, apply to both
        tf = _make_feature(cfg).fit(Xtr, ytr)
        feat_tr = tf.transform(Xtr)          # (n_train, n_voxels)
        feat_te = tf.transform(Xte)          # (n_test,  n_voxels)

        # 2) voxel selection on the TRAIN features the decoder will actually see
        mask, _ = anova_select(feat_tr, ytr, cfg.q_threshold)
        if mask.sum() == 0:                  # degenerate guard
            mask[np.argsort(feat_tr.var(axis=0))[-10:]] = True

        # 3) decode (with per-trial confidence margin for selective prediction)
        dec = LinearDecoder(cfg.n_components, cfg.shrinkage).fit(feat_tr[:, mask], ytr)
        preds, margin = dec.predict_with_margin(feat_te[:, mask])

        y_true[test] = yte
        y_pred[test] = preds
        margins[test] = margin
        per_fold.append((preds == yte).mean())
        n_sel.append(int(mask.sum()))

        imp_accum[mask] += dec.voxel_importance()
        imp_count[mask] += 1

    importance = imp_accum / np.where(imp_count > 0, imp_count, 1)
    return {
        "accuracy": float(np.mean(per_fold)),
        "per_fold_accuracy": per_fold,
        "confusion_matrix": confusion_matrix(y_true, y_pred, n_classes),
        "y_true": y_true,
        "y_pred": y_pred,
        "margins": margins,
        "importance": importance,
        "mean_selected_voxels": float(np.mean(n_sel)),
    }


def risk_coverage(y_true, y_pred, margins, n_points: int = 21):
    """
    Selective-prediction curve. Sort trials by confidence (margin) and, for each
    coverage level (fraction of most-confident trials we choose to act on),
    report the accuracy on that retained subset.

    For a TFUS targeting device this answers the operational question: "if I only
    stimulate when the decoder is confident, how accurate am I, and how often do
    I act?"
    """
    correct = (y_true == y_pred).astype(float)
    order = np.argsort(-margins)              # most confident first
    correct_sorted = correct[order]
    coverages = np.linspace(1.0, 0.1, n_points)
    out = []
    n = len(correct)
    for cov in coverages:
        k = max(1, int(round(cov * n)))
        out.append((cov, float(correct_sorted[:k].mean())))
    return out


def run_repeats(X_norm, y, cfg: DecoderConfig, seeds=(0, 1, 2)):
    """Seed-averaged evaluation. Returns aggregate mean/std plus last-seed detail."""
    accs, last = [], None
    for s in seeds:
        last = cross_validate(X_norm, y, cfg, seed=s)
        accs.append(last["accuracy"])
    return {
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy": float(np.std(accs)),
        "seed_accuracies": accs,
        "detail": last,         # confusion matrix / importance from the last seed
        "config": cfg,
    }
