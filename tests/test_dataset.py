"""
Validate the dataset-agnostic layer:
  * a dataset with a DIFFERENT shape/TR/class-count than the synthetic default
    flows through load -> baseline -> matched-bank CV and decodes planted signal,
  * .npz round-trips through save/load with spec metadata intact,
  * the loader refuses pickle-bearing files (the safe-load guarantee).
"""
import os, sys, tempfile
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from npkit.dataset import save_dataset, load_dataset, DatasetSpec, validate_dataset
from npkit.preprocess import baseline_normalize
from npkit.evaluate import DecoderConfig, cross_validate


def _planted_dataset(n_per=40, n_classes=3, V=40, T=10, tr=0.5, seed=0):
    """Each class puts a bump in its own voxel; everything else is noise."""
    rng = np.random.default_rng(seed)
    n = n_per * n_classes
    X = rng.normal(0, 0.1, (n, V, T)).astype(np.float32)
    y = np.repeat(np.arange(n_classes), n_per)
    bump = np.hanning(T).astype(np.float32)            # smooth response shape
    for i in range(n):
        X[i, y[i] % V] += 1.0 * bump
    idx = rng.permutation(n)
    return X[idx], y[idx]


def test_agnostic_shapes_decode():
    X, y = _planted_dataset()
    assert X.shape == (120, 40, 10)
    Xn = baseline_normalize(X, n_baseline_timepoints=2)
    # TR != 1, classes != 5, timepoints != 20 -> exercises the adaptive bank
    res = cross_validate(Xn, y, DecoderConfig(feature="matchedbank", tr=0.5), seed=0)
    assert res["accuracy"] > 0.7, res["accuracy"]          # planted signal is decodable
    assert res["confusion_matrix"].shape == (3, 3)


def test_npz_roundtrip():
    X, y = _planted_dataset()
    spec = DatasetSpec(tr=0.5, baseline_timepoints=2,
                       class_names=("a", "b", "c"), spatial_shape=(8, 5),
                       source="unit-test")
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "ds.npz")
        save_dataset(p, X, y, spec)
        X2, y2, spec2 = load_dataset(p)
        assert np.array_equal(X2, X.astype(np.float32))
        assert np.array_equal(y2, y)                  # labels already 0..C-1
        assert spec2.tr == 0.5 and spec2.baseline_timepoints == 2
        assert spec2.class_names == ("a", "b", "c")
        assert spec2.spatial_shape == (8, 5)


def test_loader_refuses_pickle():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "bad.npz")
        # object array forces pickle on load; allow_pickle=False must refuse
        np.savez(p, X=np.array([1, {"danger": 2}], dtype=object), y=np.array([0, 1]))
        try:
            load_dataset(p)
            raised = False
        except ValueError:
            raised = True
        assert raised, "loader must refuse pickle-bearing files"


def test_validate_rejects_bad_shapes():
    for bad in [
        (np.zeros((10, 5)), np.zeros(10)),            # X not 3-D
        (np.zeros((10, 5, 8)), np.zeros(9)),          # y length mismatch
        (np.zeros((10, 5, 2)), np.arange(10) % 2),    # too few timepoints
    ]:
        try:
            validate_dataset(*bad)
            ok = False
        except ValueError:
            ok = True
        assert ok


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn(); print(f"  PASS  {fn.__name__}"); passed += 1
        except AssertionError as e:
            print(f"  FAIL  {fn.__name__}: {e}")
    print(f"\n{passed}/{len(fns)} dataset tests passed")
    sys.exit(0 if passed == len(fns) else 1)
