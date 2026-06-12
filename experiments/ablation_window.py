"""
Ablation: original fixed 3-8 s window vs. adaptive per-fold window.

Both use the identical NumPy PCA + shrinkage-LDA decoder and identical
leakage-free CV, so any accuracy difference is attributable to the temporal
window alone. Seed-averaged. Prints a small table and timing.
"""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from npkit.data import generate_trials
from npkit.preprocess import baseline_normalize
from npkit.evaluate import DecoderConfig, run_repeats

SEEDS = (0, 1, 2)


def main(n_trials_per_finger=100):
    t0 = time.time()
    X, y, patches, centers = generate_trials(
        n_trials_per_finger=n_trials_per_finger, domain_randomize=True, seed=42
    )
    X = baseline_normalize(X)
    t_data = time.time() - t0
    print(f"data: {X.shape[0]} trials x {X.shape[1]} voxels x {X.shape[2]} t "
          f"(generated in {t_data:.1f}s)\n")

    configs = {
        "baseline (fixed 3-8 mean)":    DecoderConfig(feature="fixed", fixed_window=(3, 8)),
        "adaptive window mean":         DecoderConfig(feature="adaptive", window_width=5),
        "matched filter (1 template)":  DecoderConfig(feature="matched"),
        "matched bank (timing-robust)": DecoderConfig(feature="matchedbank"),
    }

    print(f"{'config':<30} {'acc (mean+/-std)':<20} {'sel.voxels':<12} {'time':<8}")
    print("-" * 72)
    for name, cfg in configs.items():
        t1 = time.time()
        res = run_repeats(X, y, cfg, seeds=SEEDS)
        dt = time.time() - t1
        print(f"{name:<30} {res['mean_accuracy']:.3f} +/- {res['std_accuracy']:.3f}      "
              f"{res['detail']['mean_selected_voxels']:<12.0f} {dt:.1f}s")
    print(f"\nchance = {1/len(np.unique(y)):.3f}   (5 classes)")
    print(f"total wall-clock: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    main(n)
