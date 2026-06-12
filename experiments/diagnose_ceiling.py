"""
Where does accuracy go? Decompose the error by turning the simulator's
degradations on one at a time, holding the decoder fixed (matched-bank).

If the decoder is near the Bayes ceiling, accuracy should be ~near-perfect on
clean data and fall predictably as we add (a) realistic noise, (b) trial
dropout, (c) cross-patch leakage. Trial dropout in particular imposes a hard
ceiling: a non-responsive trial is pure noise and cannot be classified above
chance, so expected ceiling ~= reliability*1.0 + (1-reliability)*0.2.
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from npkit.data import generate_trials
from npkit.preprocess import baseline_normalize
from npkit.evaluate import DecoderConfig, run_repeats

CFG = DecoderConfig(feature="matchedbank")
SEEDS = (0, 1, 2)
N = 100


def evaluate(label, ceiling=None, **gen):
    X, y, *_ = generate_trials(n_trials_per_finger=N, domain_randomize=False,
                               seed=42, **gen)
    X = baseline_normalize(X)
    res = run_repeats(X, y, CFG, seeds=SEEDS)
    cz = f"   (ceiling ~{ceiling:.2f})" if ceiling is not None else ""
    print(f"{label:<46} {res['mean_accuracy']:.3f} +/- {res['std_accuracy']:.3f}{cz}")
    return res['mean_accuracy']


def main():
    print(f"{'condition':<46} {'accuracy':<18}")
    print("-" * 70)
    evaluate("clean (no noise-floor, full reliability)",
             noise_std=0.02, trial_reliability=1.0, leakage_fraction=0.0)
    evaluate("+ realistic noise",
             noise_std=0.05, trial_reliability=1.0, leakage_fraction=0.0)
    evaluate("+ trial dropout (reliability=0.825)", ceiling=0.825 + 0.175 * 0.2,
             noise_std=0.05, trial_reliability=0.825, leakage_fraction=0.0)
    evaluate("+ cross-patch leakage (0.175)", ceiling=0.825 + 0.175 * 0.2,
             noise_std=0.05, trial_reliability=0.825, leakage_fraction=0.175)
    print("-" * 70)
    # full domain randomization (the headline setting)
    X, y, *_ = generate_trials(n_trials_per_finger=N, domain_randomize=True, seed=42)
    X = baseline_normalize(X)
    res = run_repeats(X, y, CFG, seeds=SEEDS)
    print(f"{'full domain randomization (headline)':<46} "
          f"{res['mean_accuracy']:.3f} +/- {res['std_accuracy']:.3f}")
    print(f"\nchance = 0.200")


if __name__ == "__main__":
    main()
