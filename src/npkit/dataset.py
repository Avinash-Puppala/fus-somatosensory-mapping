"""
npkit.dataset — dataset-agnostic loading, validation, and metadata.

The decoder makes no assumptions about a specific recording. A dataset is just:
    X : (n_trials, n_voxels, n_timepoints) float array
    y : (n_trials,) integer class labels
    spec : DatasetSpec  — acquisition metadata the pipeline needs
           (TR, baseline length, optional class names + 2-D shape for plotting)

Canonical on-disk format is a single .npz, loaded with allow_pickle=False so a
data file can never execute code on load (the one real hazard we flagged for
downloaded data). Real recordings usually arrive as .mat / HDF5; those require
scipy/h5py which aren't available in this sandbox, so the conversion to .npz is
done once on a machine that has them — see scripts/convert_to_npz.py. The
decoder itself only ever sees the validated arrays, regardless of origin.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class DatasetSpec:
    tr: float = 1.0                                  # seconds per timepoint
    baseline_timepoints: int = 3                     # pre-stimulus baseline length
    class_names: Optional[Tuple[str, ...]] = None    # for labels in plots
    spatial_shape: Optional[Tuple[int, int]] = None  # (rows, cols) for weight-map viz
    source: str = "unknown"

    def n_classes_hint(self) -> Optional[int]:
        return len(self.class_names) if self.class_names else None


def validate_dataset(X: np.ndarray, y: np.ndarray,
                     regression: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Check shapes/dtypes. For classification, normalize labels to a contiguous
    0..C-1 range; for regression, keep y as float. Raises ValueError with an
    actionable message on anything malformed.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    if X.ndim != 3:
        raise ValueError(f"X must be 3-D (trials, voxels, timepoints); got {X.shape}")
    if y.ndim != 1 or len(y) != X.shape[0]:
        raise ValueError(f"y must be 1-D with len == n_trials ({X.shape[0]}); got {y.shape}")
    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or Inf — clean or impute before decoding")
    if X.shape[2] < 4:
        raise ValueError(f"need >=4 timepoints to estimate an HRF window; got {X.shape[2]}")

    if regression:
        return X.astype(np.float32, copy=False), y.astype(float)

    # Classification: remap arbitrary integer/string labels to 0..C-1.
    classes, y_remapped = np.unique(y, return_inverse=True)
    if len(classes) < 2:
        raise ValueError(f"need >=2 classes to decode; found {len(classes)}")
    counts = np.bincount(y_remapped)
    if counts.min() < 2:
        raise ValueError(f"every class needs >=2 trials for CV; smallest has {counts.min()}")

    return X.astype(np.float32, copy=False), y_remapped.astype(int)


def load_dataset(path: str, regression: bool = False) -> tuple[np.ndarray, np.ndarray, DatasetSpec]:
    """
    Load a dataset from a .npz produced by save_dataset (or any .npz exposing
    'X' and 'y'). Safe: allow_pickle=False blocks code execution on load.
    """
    if not path.endswith(".npz"):
        raise ValueError(
            "Only .npz is read directly (safe, pickle-free). Convert .mat/.h5 "
            "with scripts/convert_to_npz.py on a machine with scipy/h5py."
        )
    try:
        with np.load(path, allow_pickle=False) as d:
            if "X" not in d or "y" not in d:
                raise ValueError(f"{path} must contain arrays 'X' and 'y'")
            X, y = d["X"], d["y"]
            spec = DatasetSpec(
                tr=float(d["tr"]) if "tr" in d else 1.0,
                baseline_timepoints=int(d["baseline_timepoints"])
                    if "baseline_timepoints" in d else 3,
                class_names=tuple(str(s) for s in d["class_names"])
                    if "class_names" in d else None,
                spatial_shape=tuple(int(v) for v in d["spatial_shape"])
                    if "spatial_shape" in d else None,
                source=str(d["source"]) if "source" in d else path,
            )
    except ValueError as e:
        if "allow_pickle" in str(e) or "pickled" in str(e).lower():
            raise ValueError(
                f"{path} requires pickle to load (object arrays). For safety this "
                "is refused. Re-export as plain numeric arrays via convert_to_npz.py."
            ) from e
        raise
    X, y = validate_dataset(X, y, regression=regression)
    return X, y, spec


def save_dataset(path: str, X: np.ndarray, y: np.ndarray, spec: DatasetSpec) -> None:
    """Write a pickle-free .npz the loader can read back."""
    fields = {"X": np.asarray(X, dtype=np.float32),
              "y": np.asarray(y, dtype=np.int64),
              "tr": np.float64(spec.tr),
              "baseline_timepoints": np.int64(spec.baseline_timepoints),
              "source": np.str_(spec.source)}
    if spec.class_names is not None:
        fields["class_names"] = np.array(spec.class_names, dtype=np.str_)
    if spec.spatial_shape is not None:
        fields["spatial_shape"] = np.array(spec.spatial_shape, dtype=np.int64)
    np.savez_compressed(path, **fields)
