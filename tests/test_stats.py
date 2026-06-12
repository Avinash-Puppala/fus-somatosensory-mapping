"""Validate npkit.stats against analytically-known reference values."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from npkit.stats import gamma_pdf, betai, f_sf


def approx(a, b, tol=1e-6):
    return abs(a - b) < tol


def test_betai_known_values():
    # Beta(0.5,0.5) is the arcsine distribution, symmetric about 0.5
    assert approx(betai(0.5, 0.5, 0.5), 0.5), betai(0.5, 0.5, 0.5)
    # I_0.5(2,3) worked out by hand = 0.6875
    assert approx(betai(2, 3, 0.5), 0.6875), betai(2, 3, 0.5)
    # boundary behaviour
    assert betai(2, 3, 0.0) == 0.0
    assert betai(2, 3, 1.0) == 1.0
    # complement symmetry: I_x(a,b) = 1 - I_{1-x}(b,a)
    assert approx(betai(2.5, 4.0, 0.3), 1.0 - betai(4.0, 2.5, 0.7))


def test_f_sf_symmetry_and_range():
    # F(d,d) at f=1 has survival 0.5 (median is 1 for equal dof)
    assert approx(f_sf(np.array([1.0]), 1, 1)[0], 0.5)
    assert approx(f_sf(np.array([1.0]), 10, 10)[0], 0.5)
    # survival is monotonically decreasing in f
    fs = f_sf(np.array([0.5, 1.0, 2.0, 5.0, 20.0]), 4, 30)
    assert np.all(np.diff(fs) < 0), fs
    # all probabilities in [0,1]
    assert np.all((fs >= 0) & (fs <= 1))
    # f<=0 -> survival 1
    assert approx(f_sf(np.array([-1.0, 0.0]), 4, 30)[0], 1.0)


def test_f_sf_montecarlo():
    # Independent validation against the DEFINITION of the F-distribution:
    # F(d1,d2) = (chi2_d1 / d1) / (chi2_d2 / d2), and chi2_k = sum of k squared
    # standard normals. We draw a large sample and compare the empirical tail
    # P(F > f) to f_sf. This relies on nothing in npkit, so it cannot be circular.
    rng = np.random.default_rng(0)
    d1, d2, N = 5, 12, 400_000
    chi1 = (rng.standard_normal((N, d1)) ** 2).sum(axis=1)
    chi2 = (rng.standard_normal((N, d2)) ** 2).sum(axis=1)
    F = (chi1 / d1) / (chi2 / d2)
    for f in (1.0, 2.0, 3.0):
        empirical = np.mean(F > f)
        analytic = f_sf(np.array([f]), d1, d2)[0]
        # Monte-Carlo std error ~ sqrt(p(1-p)/N) < 1e-3 here; allow 3 s.e. + margin
        assert abs(empirical - analytic) < 5e-3, (f, empirical, analytic)


def test_gamma_pdf_peak_and_mass():
    t = np.arange(0, 40, 0.01)
    # canonical HRF: shape a=6, scale=1 -> mode at (a-1)*scale = 5s
    pdf = gamma_pdf(t, a=6, scale=1.0)
    peak_t = t[np.argmax(pdf)]
    assert approx(peak_t, 5.0, tol=0.02), peak_t
    # integrates to ~1 (it is a density)
    mass = np.trapezoid(pdf, t)
    assert approx(mass, 1.0, tol=1e-3), mass
    # zero for t<=0
    assert gamma_pdf(np.array([-1.0, 0.0]), 6, 1.0).sum() == 0.0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {fn.__name__}: got {e}")
    print(f"\n{passed}/{len(fns)} stats tests passed")
    sys.exit(0 if passed == len(fns) else 1)
