"""
Selective prediction for TFUS targeting.

A sensory-restoration device should only stimulate when the decode is reliable.
We rank test trials by the decoder's confidence margin and measure accuracy as a
function of coverage (the fraction of trials we choose to act on). The expectation,
given that the accuracy ceiling is set by non-responsive trials, is that
abstaining on the least-confident trials recovers near-perfect accuracy.
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from npkit.data import generate_trials
from npkit.preprocess import baseline_normalize
from npkit.evaluate import DecoderConfig, cross_validate, risk_coverage


def main():
    X, y, *_ = generate_trials(n_trials_per_finger=100, domain_randomize=True, seed=42)
    X = baseline_normalize(X)
    res = cross_validate(X, y, DecoderConfig(feature="matchedbank"), seed=0)
    rc = risk_coverage(res["y_true"], res["y_pred"], res["margins"])

    print(f"{'coverage':<12}{'accuracy':<12}")
    print("-" * 24)
    for cov, acc in rc:
        print(f"{cov:<12.2f}{acc:<12.3f}")

    covs = [c for c, _ in rc]
    accs = [a for _, a in rc]
    plt.figure(figsize=(7, 5))
    plt.plot([c * 100 for c in covs], [a * 100 for a in accs], "o-", color="#c0392b")
    plt.axhline(res["accuracy"] * 100, ls="--", color="gray",
                label=f"full-coverage acc = {res['accuracy']*100:.1f}%")
    plt.xlabel("Coverage — % of trials acted on (most confident first)")
    plt.ylabel("Accuracy on retained trials (%)")
    plt.title("Selective prediction: abstaining on low-confidence trials\n"
              "recovers near-perfect targeting accuracy")
    plt.gca().invert_xaxis()
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out = "outputs/figures/selective_prediction.png"
    plt.savefig(out, dpi=150)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
