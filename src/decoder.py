"""Part 3 — CPCA + LDA decoder (Norman et al. 2021)."""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score

# Note: The 3-8 second time period associated with the HRF needs to eventually be more accurate or refined when working with real data.
def extract_memory_period(X, start=3, end=8):
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

def decode(X_selected, y, n_components=20, n_folds=5, random_state=42):
    """
    Cross-validated PCA + LDA decoder.

    On each fold:
      1. Fit PCA on training trials only (cross-validated — no test leakage)
      2. Project both train and test into PCA space
      3. Fit LDA on train projections
      4. Classify test projections
      5. Store predictions and decoder weights

    Parameters:
        X_selected   : array of shape (n_trials, n_voxels, n_timepoints)
                       output of preprocess.select_features()
        y            : array of shape (n_trials,) with finger labels 0-4
        n_components : number of PCA components to retain (default=20)
        n_folds      : number of cross-validation folds (default=5)
        random_state : random seed for reproducibility

    Returns:
        results : dict with keys:
            'accuracy'          : float, mean accuracy across folds
            'per_fold_accuracy' : list of per-fold accuracy values
            'confusion_matrix'  : (5, 5) array, summed across folds
            'y_true'            : ground-truth labels for all trials
            'y_pred'            : predicted labels for all trials
            'decoder_weights'   : (n_voxels,) array — LDA weights projected
                                  back to voxel space (averaged across folds)
    """
    # Collapse time to a single feature per voxel
    X_mem = extract_memory_period(X_selected)  # (n_trials, n_voxels)

    n_trials, n_voxels = X_mem.shape

    # Storage arrays to collect results across all folds
    y_true_all = np.zeros(n_trials, dtype=int)
    y_pred_all = np.zeros(n_trials, dtype=int)
    per_fold_accuracy = []
    weight_accumulator = np.zeros(n_voxels)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                          random_state=random_state)

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_mem, y)):
        X_train, X_test = X_mem[train_idx], X_mem[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # --- Step 1: PCA on training data only ---
        # n_components must not exceed n_samples or n_features
        n_comp = min(n_components, X_train.shape[0], X_train.shape[1])
        pca = PCA(n_components=n_comp)
        X_train_pca = pca.fit_transform(X_train)  # fit on train only
        X_test_pca = pca.transform(X_test)         # apply to test

        # --- Step 2: LDA in PCA space ---
        lda = LinearDiscriminantAnalysis()
        lda.fit(X_train_pca, y_train)
        y_pred = lda.predict(X_test_pca)

        # --- Step 3: Store predictions ---
        y_true_all[test_idx] = y_test
        y_pred_all[test_idx] = y_pred
        fold_acc = accuracy_score(y_test, y_pred)
        per_fold_accuracy.append(fold_acc)

        # --- Step 4: Project LDA weights back to voxel space ---
        # lda.coef_ shape: (n_classes, n_pca_components)
        # pca.components_ shape: (n_pca_components, n_voxels)
        # dot product gives (n_classes, n_voxels) — one weight map per class
        # L2 norm across classes collapses to a single importance map
        lda_in_voxel_space = np.dot(lda.coef_, pca.components_)  # (n_classes, n_voxels)
        importance = np.linalg.norm(lda_in_voxel_space, axis=0)   # (n_voxels,)
        weight_accumulator += importance

        print(f"  Fold {fold_idx + 1}/{n_folds} — accuracy: {fold_acc:.3f}")

    # Average weights across folds
    decoder_weights = weight_accumulator / n_folds

    results = {
        'accuracy': np.mean(per_fold_accuracy),
        'per_fold_accuracy': per_fold_accuracy,
        'confusion_matrix': confusion_matrix(y_true_all, y_pred_all),
        'y_true': y_true_all,
        'y_pred': y_pred_all,
        'decoder_weights': decoder_weights,
    }

    return results
