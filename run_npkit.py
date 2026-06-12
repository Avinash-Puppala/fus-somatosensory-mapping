"""
One-command entry point for the NumPy-only (npkit) pipeline.

    python run_npkit.py

Generates synthetic fUS data, runs the matched-bank + PCA + shrinkage-LDA
decoder under leakage-free, seed-averaged cross-validation, prints the headline
metrics, and writes the standard figures to outputs/figures/.

This is the dependency-light counterpart to main.py (which uses torch/sklearn).
Everything here runs on CPU with only numpy + matplotlib installed.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, "src")
from npkit.data import generate_trials, generate_finger_patches
from npkit.preprocess import baseline_normalize
from npkit.evaluate import DecoderConfig, run_repeats, risk_coverage

FINGERS = ["Thumb", "Index", "Middle", "Ring", "Pinky"]


def plot_confusion(cm, path):
    cmn = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(5)); ax.set_xticklabels(FINGERS, rotation=45, ha="right")
    ax.set_yticks(range(5)); ax.set_yticklabels(FINGERS)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=9)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion matrix (row-normalized)")
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def plot_weight_map(importance, centers, path, image_size=128):
    wm = importance.reshape(image_size, image_size)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(wm, cmap="hot")
    for name, (r, c) in centers.items():
        ax.plot(c, r, "c+", markersize=14, markeredgewidth=2)
        ax.text(c, r - 10, name, ha="center", color="cyan", fontsize=8)
    ax.set_title("Decoder voxel-importance map\n(should align with S1 finger patches)")
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def main():
    os.makedirs("outputs/figures", exist_ok=True)
    print("=" * 60)
    print("npkit pipeline — matched-bank + PCA + shrinkage-LDA")
    print("=" * 60)

    X, y, patches, centers = generate_trials(
        n_trials_per_finger=100, domain_randomize=True, seed=42)
    X = baseline_normalize(X)
    print(f"data: {X.shape[0]} trials x {X.shape[1]} voxels x {X.shape[2]} t")

    cfg = DecoderConfig(feature="matchedbank")
    res = run_repeats(X, y, cfg, seeds=(0, 1, 2))
    det = res["detail"]

    print(f"\nmean accuracy : {res['mean_accuracy']:.3f} +/- {res['std_accuracy']:.3f}"
          f"  (seeds {res['seed_accuracies']})")
    print(f"chance        : 0.200")
    print(f"sel. voxels   : {det['mean_selected_voxels']:.0f}")

    rc = risk_coverage(det["y_true"], det["y_pred"], det["margins"])
    # accuracy at ~80% coverage (operational point for a targeting device)
    cov80 = min(rc, key=lambda t: abs(t[0] - 0.80))
    print(f"selective     : {cov80[1]*100:.1f}% accuracy at {cov80[0]*100:.0f}% coverage")

    plot_confusion(det["confusion_matrix"], "outputs/figures/npkit_confusion.png")
    plot_weight_map(det["importance"], centers, "outputs/figures/npkit_weight_map.png")
    print("\nfigures: outputs/figures/npkit_confusion.png, npkit_weight_map.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
