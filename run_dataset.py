"""
Dataset-agnostic runner for the npkit decoder.

    python run_dataset.py                              # synthetic (classification)
    python run_dataset.py data.npz                     # real categorical dataset
    python run_dataset.py data.npz --task regression   # continuous-target dataset
    python run_dataset.py data.npz --clutter 1         # SVD nuisance removal first

The same matched-bank feature front-end runs for both task types. Class count,
voxels, timepoints, TR, baseline length, labels, and 2-D map shape are inferred
from the data / DatasetSpec, so no decoder code changes when real recordings
arrive. Optional --clutter N removes N leading low-rank nuisance modes
(decoding-level analogue of SVD clutter filtering) before decoding.
"""
import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "src")
from npkit.dataset import load_dataset, DatasetSpec
from npkit.preprocess import baseline_normalize
from npkit.clutter import remove_low_rank_nuisance
from npkit.evaluate import DecoderConfig, run_repeats, risk_coverage
from npkit.regress import RegressionConfig, cross_validate_regression


def load_or_synthesize(path, regression):
    if path:
        X, y, spec = load_dataset(path, regression=regression)
        print(f"loaded {path}  ({spec.source})")
        return X, y, spec
    if regression:
        sys.exit("regression needs a dataset path (no synthetic continuous target)")
    from npkit.data import generate_trials
    X, y, patches, centers = generate_trials(
        n_trials_per_finger=100, domain_randomize=True, seed=42)
    spec = DatasetSpec(tr=1.0, baseline_timepoints=3,
                       class_names=("Thumb", "Index", "Middle", "Ring", "Pinky"),
                       spatial_shape=(128, 128), source="synthetic")
    return X, y, spec


def run_classification(X, y, spec):
    n_classes = len(np.unique(y))
    names = list(spec.class_names) if spec.class_names else [f"class{c}" for c in range(n_classes)]
    res = run_repeats(X, y, DecoderConfig(feature="matchedbank", tr=spec.tr), seeds=(0, 1, 2))
    det = res["detail"]
    print(f"\nmean accuracy : {res['mean_accuracy']:.3f} +/- {res['std_accuracy']:.3f}")
    print(f"chance        : {1/n_classes:.3f}  ({n_classes}-class)")
    rc = risk_coverage(det["y_true"], det["y_pred"], det["margins"])
    cov80 = min(rc, key=lambda t: abs(t[0] - 0.80))
    print(f"selective     : {cov80[1]*100:.1f}% accuracy at {cov80[0]*100:.0f}% coverage")

    cm = det["confusion_matrix"]
    cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(1.1 * n_classes + 1, n_classes))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n_classes)); ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticks(range(n_classes)); ax.set_yticklabels(names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(f"Confusion ({spec.source})")
    plt.colorbar(im, ax=ax, fraction=0.046); plt.tight_layout()
    plt.savefig("outputs/figures/dataset_confusion.png", dpi=150); plt.close()
    if spec.spatial_shape is not None:
        plt.figure(figsize=(6, 5))
        plt.imshow(det["importance"].reshape(spec.spatial_shape), cmap="hot")
        plt.colorbar(fraction=0.046); plt.title(f"Voxel importance ({spec.source})")
        plt.tight_layout(); plt.savefig("outputs/figures/dataset_weight_map.png", dpi=150); plt.close()
    print("figures: outputs/figures/dataset_confusion.png" +
          (", dataset_weight_map.png" if spec.spatial_shape is not None else ""))


def run_regression(X, y, spec):
    res = cross_validate_regression(X, y, RegressionConfig(feature="matchedbank", tr=spec.tr))
    print(f"\nR^2        : {res['r2']:.3f}")
    print(f"Pearson r  : {res['pearson_r']:.3f}")
    plt.figure(figsize=(5, 5))
    plt.scatter(res["y_true"], res["y_pred"], s=12, alpha=0.5, color="#c0392b")
    lim = [min(res["y_true"].min(), res["y_pred"].min()),
           max(res["y_true"].max(), res["y_pred"].max())]
    plt.plot(lim, lim, "k--", alpha=0.5)
    plt.xlabel("True target"); plt.ylabel("Predicted")
    plt.title(f"Regression decode ({spec.source})\nr={res['pearson_r']:.2f}")
    plt.tight_layout(); plt.savefig("outputs/figures/dataset_regression.png", dpi=150); plt.close()
    print("figure: outputs/figures/dataset_regression.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default=None)
    ap.add_argument("--task", choices=["classification", "regression"], default="classification")
    ap.add_argument("--clutter", type=int, default=0, help="leading low-rank modes to remove")
    args = ap.parse_args()

    os.makedirs("outputs/figures", exist_ok=True)
    regression = args.task == "regression"
    X, y, spec = load_or_synthesize(args.path, regression)
    print(f"data: {X.shape[0]} trials x {X.shape[1]} voxels x {X.shape[2]} t "
          f"| task={args.task} | TR={spec.tr}s")

    if args.clutter > 0:
        X = remove_low_rank_nuisance(X, n_components=args.clutter)
        print(f"clutter: removed {args.clutter} leading low-rank mode(s)")
    X = baseline_normalize(X, n_baseline_timepoints=spec.baseline_timepoints)

    run_regression(X, y, spec) if regression else run_classification(X, y, spec)


if __name__ == "__main__":
    main()
