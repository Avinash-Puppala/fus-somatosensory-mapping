"""
Cross-validated training loop for FingerprintDecoder.

Key design decisions:
  - Feature selection (ANOVA) runs INSIDE each fold on training data only.
    This fixes the circular feature selection bug from the original decoder.py.
  - The neural model is trained and evaluated entirely within each fold.
  - Attention weights are averaged across folds to produce a voxel importance
    map equivalent to the old LDA weight map.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score

from preprocess import baseline_normalize, select_features
from model import FingerprintDecoder


def train(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    n_epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    temporal_channels: int = 32,
    random_state: int = 42,
) -> dict:
    """
    Cross-validated training and evaluation of FingerprintDecoder.

    Parameters:
        X               : (n_trials, n_voxels, n_timepoints) — preprocessed
        y               : (n_trials,) finger labels 0-4
        n_folds         : cross-validation folds
        n_epochs        : training epochs per fold
        batch_size      : mini-batch size
        lr              : Adam learning rate
        temporal_channels : output channels of TemporalEncoder (model width)
        random_state    : reproducibility seed

    Returns:
        dict with keys:
            accuracy          : mean accuracy across folds
            per_fold_accuracy : list of per-fold values
            confusion_matrix  : (5, 5) summed across folds
            y_true / y_pred   : full trial-level labels
            attention_weights : (n_voxels,) importance map averaged across folds
                                (in original voxel space, zeros for unselected voxels)
    """
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    n_trials, n_voxels, _ = X.shape

    y_true_all   = np.zeros(n_trials, dtype=int)
    y_pred_all   = np.zeros(n_trials, dtype=int)
    weight_accum = np.zeros(n_voxels)         # accumulates attention maps
    weight_count = np.zeros(n_voxels)         # tracks how many folds each voxel appeared in
    per_fold_acc = []

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                          random_state=random_state)

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # --- Feature selection on TRAINING data only ---
        # ANOVA sees only X_train here, so test data cannot influence which
        # voxels are selected. This is the fix for the circular bug.
        X_train_sel, feature_mask, _ = select_features(X_train, y_train)
        X_test_sel = X_test[:, feature_mask, :]

        n_selected_voxels = X_train_sel.shape[1]

        # --- Build model for this fold ---
        model     = FingerprintDecoder(n_classes=5,
                                       temporal_channels=temporal_channels)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train_sel)
        y_train_t = torch.LongTensor(y_train)
        dataset   = TensorDataset(X_train_t, y_train_t)
        loader    = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # --- Training ---
        model.train()
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                logits = model(xb)
                loss   = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(f"    Fold {fold_idx+1} | Epoch {epoch+1:3d}/{n_epochs}"
                      f" | loss {epoch_loss/len(loader):.4f}")

        # --- Evaluation ---
        model.eval()
        with torch.no_grad():
            X_test_t = torch.FloatTensor(X_test_sel)
            logits   = model(X_test_t)
            preds    = logits.argmax(dim=1).numpy()

            # Collect attention weights for visualization
            # get_voxel_importance returns (n_test_trials, n_selected_voxels)
            attn = model.get_voxel_importance(X_test_t).numpy()
            # Average across test trials → one importance value per selected voxel
            mean_attn = attn.mean(axis=0)

        fold_acc = accuracy_score(y_test, preds)
        per_fold_acc.append(fold_acc)
        y_true_all[test_idx] = y_test
        y_pred_all[test_idx] = preds

        # Map selected-voxel attention back to full voxel space
        weight_accum[feature_mask] += mean_attn
        weight_count[feature_mask] += 1

        print(f"  Fold {fold_idx+1}/{n_folds} — accuracy: {fold_acc:.3f}"
              f" | selected voxels: {n_selected_voxels}")

    # Average attention weights only over folds where each voxel was selected
    safe_count = np.where(weight_count > 0, weight_count, 1)
    attention_weights = weight_accum / safe_count

    return {
        'accuracy':          float(np.mean(per_fold_acc)),
        'per_fold_accuracy': per_fold_acc,
        'confusion_matrix':  confusion_matrix(y_true_all, y_pred_all),
        'y_true':            y_true_all,
        'y_pred':            y_pred_all,
        'attention_weights': attention_weights,   # replaces decoder_weights
    }
