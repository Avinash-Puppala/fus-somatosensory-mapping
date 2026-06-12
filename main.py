"""Run the full fUS somatosensory mapping pipeline."""
import os
import sys

sys.path.insert(0, 'src')

from generate_data import generate_trials
from preprocess import baseline_normalize
from decoder import decode
from visualize import plot_confusion_matrix, plot_weight_map, plot_per_fold_accuracy

FINGER_NAMES = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']

def run_pipeline(
    n_trials_per_finger=100,
    n_components=20,
    n_folds=5,
    trial_reliability=0.85,
    leakage_fraction=0.15,
    save_figures=True,
):
    """
    Run the full Phase A pipeline end to end.

    Parameters:
        n_trials_per_finger : number of trials to generate per finger
        n_components        : number of PCA components for the decoder
        n_folds             : number of cross-validation folds
        trial_reliability   : probability a trial produces an HRF response
                              (0.85 = 15% of trials are noise-only)
        leakage_fraction    : fraction of signal bleeding into adjacent patches
                              (0.15 = neighbouring patches get 15% of active signal)
        save_figures        : whether to save output figures to disk
    """
    os.makedirs('outputs/figures', exist_ok=True)

    # --- Phase 1: Generate synthetic data ---
    print("=" * 50)
    print("Phase 1: Generating synthetic fUS data")
    print("=" * 50)
    X, y, patches, centers = generate_trials(
        n_trials_per_finger=n_trials_per_finger,
        trial_reliability=trial_reliability,
        leakage_fraction=leakage_fraction,
    )
    print(f"  Trials generated  : {X.shape[0]}")
    print(f"  Trial reliability : {trial_reliability} ({int((1-trial_reliability)*100)}% noise-only trials)")
    print(f"  Leakage fraction  : {leakage_fraction} ({int(leakage_fraction*100)}% signal bleed to adjacent patches)")
    print(f"  Image size        : {X.shape[1]} voxels ({int(X.shape[1]**0.5)}x{int(X.shape[1]**0.5)})")
    print(f"  Timepoints        : {X.shape[2]}")

    # --- Phase 2: Preprocess ---
    print("\n" + "=" * 50)
    print("Phase 2: Preprocessing")
    print("=" * 50)
    X_norm = baseline_normalize(X)
    print("  Baseline normalization complete")
    print("  Feature selection will run per fold inside the decoder (no leakage)")

    # --- Phase 3: Decode ---
    print("\n" + "=" * 50)
    print("Phase 3: Running CPCA + LDA decoder")
    print("=" * 50)
    results = decode(X_norm, y, n_components=n_components, n_folds=n_folds)
    print(f"\n  Mean accuracy  : {results['accuracy']:.3f}")
    print(f"  Chance level   : 0.200  (5-class)")
    print(f"  Per-fold       : {[f'{a:.3f}' for a in results['per_fold_accuracy']]}")
    print(f"  HRF window     : t={results['memory_window'][0]}–{results['memory_window'][1]} (averaged across folds)")
    print(f"  Consensus mask : {results['feature_mask'].sum()} voxels selected in ≥50% of folds")

    # --- Phase 4: Visualize ---
    print("\n" + "=" * 50)
    print("Phase 4: Visualizing results")
    print("=" * 50)

    cm_path = 'outputs/figures/confusion_matrix.png' if save_figures else None
    wm_path = 'outputs/figures/weight_map.png' if save_figures else None
    fa_path = 'outputs/figures/per_fold_accuracy.png' if save_figures else None

    plot_confusion_matrix(results['confusion_matrix'], FINGER_NAMES, save_path=cm_path)
    plot_weight_map(results['decoder_weights'], results['feature_mask'], centers, save_path=wm_path)
    plot_per_fold_accuracy(results['per_fold_accuracy'], save_path=fa_path)

    if save_figures:
        print("  Figures saved to outputs/figures/")

    print("\n" + "=" * 50)
    print("Pipeline complete.")
    print("=" * 50)

    return results


if __name__ == "__main__":
    run_pipeline()
