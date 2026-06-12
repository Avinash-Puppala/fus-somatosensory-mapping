"""
Validate the three new pieces: SVD clutter filter, epocher, regression head.
Each is checked against a synthetic case with a known answer.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from npkit.clutter import svd_clutter_filter, estimate_clutter_rank, remove_low_rank_nuisance
from npkit.epoch import epoch_continuous
from npkit.preprocess import baseline_normalize
from npkit.regress import RegressionConfig, cross_validate_regression


def test_clutter_filter_removes_dominant_mode():
    rng = np.random.default_rng(0)
    T, V = 64, 256
    blood_idx = np.arange(200, 256)                       # blood lives here only
    # rank-1 tissue clutter, spatially disjoint from blood, 100x stronger
    u_t = np.cos(2 * np.pi * 0.5 * np.arange(T) / T); u_t /= np.linalg.norm(u_t)
    s_v = np.zeros(V); s_v[:200] = rng.normal(size=200)
    M = 100.0 * np.outer(u_t, s_v)
    M[:, blood_idx] += rng.normal(size=(T, len(blood_idx)))   # blood flux
    M += 0.1 * rng.normal(size=(T, V))                        # thermal noise

    assert estimate_clutter_rank(M) >= 1
    filt = svd_clutter_filter(M, n_clutter=1)

    # energy along the clutter temporal mode must collapse
    before = np.linalg.norm(u_t @ M)
    after = np.linalg.norm(u_t @ filt)
    assert after < 0.05 * before, (after, before)
    # blood voxels (clutter-free) must be essentially preserved
    retained = (np.linalg.norm(filt[:, blood_idx]) /
                np.linalg.norm(M[:, blood_idx]))
    assert retained > 0.9, retained


def test_low_rank_nuisance_removes_global_mode():
    rng = np.random.default_rng(1)
    n, V, T = 20, 50, 8
    X = rng.normal(0, 0.1, (n, V, T)).astype(np.float32)
    # plant a strong global mode: same spatial pattern scaled per trial
    sp = rng.normal(size=V).astype(np.float32)
    g = (rng.normal(size=(n, T)) * 10.0).astype(np.float32)
    X += g[:, None, :] * sp[None, :, None]
    Xf = remove_low_rank_nuisance(X, n_components=1)
    assert Xf.shape == X.shape and Xf.dtype == X.dtype
    # projection onto the global spatial pattern must shrink dramatically
    sp_u = sp / np.linalg.norm(sp)
    before = np.linalg.norm(np.tensordot(X, sp_u, axes=([1], [0])))
    after = np.linalg.norm(np.tensordot(Xf, sp_u, axes=([1], [0])))
    assert after < 0.2 * before, (after, before)


def test_epocher_alignment_and_bounds():
    V, Tt = 10, 100
    rec = np.arange(V * Tt, dtype=float).reshape(V, Tt)   # (voxels, time)
    onsets = [10, 30, 50, 95]                              # last overruns -> dropped
    X, y = epoch_continuous(rec, onsets, window=20, pre=2,
                            labels=[0, 1, 2, 3], time_axis=1)
    assert X.shape == (3, V, 20)                           # 3 valid trials
    assert list(y) == [0, 1, 2]
    assert np.array_equal(X[0], rec[:, 8:28])              # onset 10 - pre 2 = 8


def test_regression_recovers_continuous_target():
    rng = np.random.default_rng(3)
    n, V, T = 150, 60, 12
    y = rng.uniform(-1, 1, n)
    bump = np.hanning(T)
    X = rng.normal(0, 0.1, (n, V, T)).astype(np.float32)
    for i in range(n):
        X[i, 0] += (y[i] * bump).astype(np.float32)        # amplitude encodes target
    Xn = baseline_normalize(X, n_baseline_timepoints=2)
    res = cross_validate_regression(Xn, y, RegressionConfig(tr=0.5, top_voxels=20), seed=0)
    assert res["r2"] > 0.5, res["r2"]
    assert res["pearson_r"] > 0.7, res["pearson_r"]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn(); print(f"  PASS  {fn.__name__}"); passed += 1
        except AssertionError as e:
            print(f"  FAIL  {fn.__name__}: {e}")
    print(f"\n{passed}/{len(fns)} adapter tests passed")
    sys.exit(0 if passed == len(fns) else 1)
