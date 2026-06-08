"""Run the full fUS somatosensory mapping pipeline."""
import os
import sys

sys.path.insert(0, 'src')

from generate_data import generate_trials
from preprocess import baseline_normalize
from train import train
from visualize import plot_confusion_matrix, plot_weight_map, plot_per_fold_accuracy

FINGER_NAMES = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']


def run_pipeline(
    n_trials_per_finger=100,
    n_folds=5,
    n_epochs=30,
    batch_size=32,
    lr=1e-3,
    temporal_channels=32,
    domain_randomize=True,
    save_figures=True,
):
    """
    Run the full pipeline end to end.

    Phase 1 — Generate synthetic fUS data (with domain randomization).
    Phase 2 — Baseline normalization.
    Phase 3 — Cross-validated neural decoder (FingerprintDecoder).
               Feature selection is performed inside each fold to avoid
               the circular leakage bug present in the old CPCA+LDA pipeline.
    Phase 4 — Visualize results.

    Parameters:
        n_trials_per_finger : trials generated per finger (500 total at default)
        n_folds             : cross-validation folds
        n_epochs            : training epochs per fold
        batch_size          : mini-batch size for Adam
        lr                  : Adam learning rate
        temporal_channels   : width of the TemporalEncoder (model capacity)
        domain_randomize    : vary HRF timing, noise, patch size etc. per trial
        save_figures        : write PNGs to outputs/figures/
    """
    os.makedirs('outputs/figures', exist_ok=True)

    # --- Phase 1: Generate synthetic data ---
    print("=" * 50)
    print("Phase 1: Generating synthetic fUS data")
    print("=" * 50)
    X, y, patches, centers = generate_trials(
        n_trials_per_finger=n_trials_per_finger,
        domain_randomize=domain_randomize,
    )
    print(f"  Trials generated    : {X.shape[0]}")
    print(f"  Domain randomize    : {domain_randomize}")
    print(f"  Image size          : {X.shape[1]} voxels "
          f"({int(X.shape[1]**0.5)}x{int(X.shape[1]**0.5)})")
    print(f"  Timepoints          : {X.shape[2]}")

    # --- Phase 2: Preprocess ---
    print("\n" + "=" * 50)
    print("Phase 2: Preprocessing")
    print("=" * 50)
    X_norm = baseline_normalize(X)
    print("  Baseline normalization complete")
    # Note: select_features is now called inside each CV fold in train.py,
    # which eliminates the circular feature-selection bug from the old pipeline.

    # --- Phase 3: Train neural decoder ---
    print("\n" + "=" * 50)
    print("Phase 3: Training FingerprintDecoder (cross-validated)")
    print("=" * 50)
    results = train(
        X_norm, y,
        n_folds=n_folds,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        temporal_channels=temporal_channels,
    )
    print(f"\n  Mean accuracy  : {results['accuracy']:.3f}")
    print(f"  Chance level   : 0.200  (5-class)")
    print(f"  Per-fold       : {[f'{a:.3f}' for a in results['per_fold_accuracy']]}")

    # --- Phase 4: Visualize ---
    print("\n" + "=" * 50)
    print("Phase 4: Visualizing results")
    print("=" * 50)

    cm_path = 'outputs/figures/confusion_matrix.png' if save_figures else None
    wm_path = 'outputs/figures/weight_map.png'       if save_figures else None
    fa_path = 'outputs/figures/per_fold_accuracy.png' if save_figures else None

    # Build a feature mask covering all voxels (attention weights are already
    # in full voxel space — zeros where a voxel was never selected)
    import numpy as np
    full_mask = results['attention_weights'] > 0

    plot_confusion_matrix(results['confusion_matrix'], FINGER_NAMES,
                          save_path=cm_path)
    plot_weight_map(results['attention_weights'], full_mask, centers,
                    save_path=wm_path)
    plot_per_fold_accuracy(results['per_fold_accuracy'], save_path=fa_path)

    if save_figures:
        print("  Figures saved to outputs/figures/")

    print("\n" + "=" * 50)
    print("Pipeline complete.")
    print("=" * 50)

    return results


if __name__ == "__main__":
    run_pipeline()
