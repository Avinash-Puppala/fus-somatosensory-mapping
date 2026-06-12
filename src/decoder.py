"""Part 3 — CPCA + LDA decoder (Norman et al. 2021)."""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score

from preprocess import select_features


def estimate_memory_period(X_train, n_timepoints=20):
    """
    Adaptively estimate the HRF peak window from training data.

    Rather than hardcoding timepoints 3-8, we compute the mean signal
    across all training trials and voxels at each timepoint, find the
    peak, and take a window of ±2 timepoints around it.

    Parameters:
        X_train     : array of shape (n_train_trials, n_voxels, n_timepoints)
        n_timepoints: total number of timepoints (default 20)

    Returns:
        start : first timepoint of estimated peak window (inclusive)
        end   : last timepoint of estimated peak window (exclusive)
    """
    # Mean signal across all trials and voxels at each timepoint
    mean_signal = X_train.mean(axis=(0, 1))  # (n_timepoints,)

    # Find the timepoint with maximum mean signal
    peak = int(np.argmax(mean_signal))

    # Take a window of ±2 around the peak, clamped to valid range
    start = max(0, peak - 2)
    end = min(n_timepoints, peak + 3)  # +3 because slice end is exclusive

    return start, end


def extract_memory_period(X, start, end):
    """
    Collapse the HRF peak window into a single value per voxel per trial.

    Parameters:
        X     : array of shape (n_trials, n_voxels, n_timepoints)
        start : first timepoint of memory period (inclusive)
        end   : last timepoint of memory period (exclusive)

    Returns:
        X_mem : array of shape (n_trials, n_voxels)
    """
    return X[:, :, start:end].mean(axis=2)


def decode(X_norm, y, n_components=20, n_folds=5, random_state=42):
    """
    Cross-validated feature selection + PCA + LDA decoder.

    Feature selection now runs inside the cross-validation loop on training
    data only — eliminating the data leakage that occurred when select_features()
    was applied to all trials before splitting.

    The HRF peak window is also estimated adaptively from training data each
    fold rather than using the hardcoded timepoints 3-8.

    On each fold:
      1. Split into train and test
      2. Run select_features() on training data only
      3. Apply feature mask to test data
      4. Estimate HRF peak window from training data
      5. Collapse timepoints to memory period average
      6. Fit PCA on train, apply to test
      7. Fit LDA on train, predict on test
      8. Accumulate weights in full voxel space

    Parameters:
        X_norm       : array of shape (n_trials, n_voxels, n_timepoints)
                       baseline-normalized data — NOT pre-feature-selected
        y            : array of shape (n_trials,) with finger labels 0-4
        n_components : number of PCA components to retain (default=20)
        n_folds      : number of cross-validation folds (default=5)
        random_state : random seed for reproducibility

    Returns:
        results : dict with keys:
            'accuracy'          : float, mean accuracy across folds
            'per_fold_accuracy' : list of per-fold accuracy values
            'confusion_matrix'  : (n_classes, n_classes) array
            'y_true'            : ground-truth labels for all trials
            'y_pred'            : predicted labels for all trials
            'decoder_weights'   : (n_all_voxels,) importance map in full
                                  voxel space, averaged across folds
            'feature_mask'      : (n_all_voxels,) boolean — voxels selected
                                  in at least half of folds
            'memory_window'     : (start, end) averaged across folds
    """
    n_trials, n_all_voxels, n_timepoints = X_norm.shape

    # Storage arrays
    y_true_all = np.zeros(n_trials, dtype=int)
    y_pred_all = np.zeros(n_trials, dtype=int)
    per_fold_accuracy = []

    # Accumulate in full voxel space so each fold's mask doesn't need to match
    weight_accumulator = np.zeros(n_all_voxels)
    mask_accumulator = np.zeros(n_all_voxels)  # counts how often each voxel is selected
    window_starts, window_ends = [], []

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                          random_state=random_state)

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_norm, y)):
        X_train_full = X_norm[train_idx]   # (n_train, n_all_voxels, n_timepoints)
        X_test_full  = X_norm[test_idx]    # (n_test,  n_all_voxels, n_timepoints)
        y_train = y[train_idx]
        y_test  = y[test_idx]

        # --- Step 1: Feature selection on training data only ---
        X_train_sel, feature_mask, _ = select_features(X_train_full, y_train)
        X_test_sel = X_test_full[:, feature_mask, :]
        mask_accumulator += feature_mask.astype(float)

        # --- Step 2: Adaptive HRF window from training data ---
        start, end = estimate_memory_period(X_train_sel, n_timepoints)
        window_starts.append(start)
        window_ends.append(end)

        # --- Step 3: Collapse timepoints ---
        X_train_mem = extract_memory_period(X_train_sel, start, end)
        X_test_mem  = extract_memory_period(X_test_sel,  start, end)

        # --- Step 4: PCA on training data only ---
        n_comp = min(n_components, X_train_mem.shape[0], X_train_mem.shape[1])
        pca = PCA(n_components=n_comp)
        X_train_pca = pca.fit_transform(X_train_mem)
        X_test_pca  = pca.transform(X_test_mem)

        # --- Step 5: LDA ---
        lda = LinearDiscriminantAnalysis()
        lda.fit(X_train_pca, y_train)
        y_pred = lda.predict(X_test_pca)

        # --- Step 6: Store predictions ---
        y_true_all[test_idx] = y_test
        y_pred_all[test_idx] = y_pred
        fold_acc = accuracy_score(y_test, y_pred)
        per_fold_accuracy.append(fold_acc)

        # --- Step 7: Project LDA weights back to full voxel space ---
        # lda.coef_ (n_classes, n_pca) @ pca.components_ (n_pca, n_selected)
        # gives (n_classes, n_selected) — project norm into full voxel space
        lda_in_sel_space = np.dot(lda.coef_, pca.components_)  # (n_classes, n_selected)
        importance_sel = np.linalg.norm(lda_in_sel_space, axis=0)  # (n_selected,)

        importance_full = np.zeros(n_all_voxels)
        importance_full[feature_mask] = importance_sel
        weight_accumulator += importance_full

        print(f"  Fold {fold_idx + 1}/{n_folds} — accuracy: {fold_acc:.3f}  "
              f"| voxels selected: {feature_mask.sum()}  "
              f"| HRF window: t={start}–{end}")

    # Average weights and build consensus feature mask
    decoder_weights = weight_accumulator / n_folds
    # A voxel is in the consensus mask if selected in at least half of folds
    consensus_mask = mask_accumulator >= (n_folds / 2)

    results = {
        'accuracy':          np.mean(per_fold_accuracy),
        'per_fold_accuracy': per_fold_accuracy,
        'confusion_matrix':  confusion_matrix(y_true_all, y_pred_all),
        'y_true':            y_true_all,
        'y_pred':            y_pred_all,
        'decoder_weights':   decoder_weights,
        'feature_mask':      consensus_mask,
        'memory_window':     (int(np.mean(window_starts)), int(np.mean(window_ends))),
    }

    return results
