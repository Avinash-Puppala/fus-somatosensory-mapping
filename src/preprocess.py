"""Part 2 — preprocessing."""
import numpy as np
from scipy import stats

def baseline_normalize(X, n_baseline_timepoints=3):
    """
    Normalize each voxel's signal to express it as change from baseline.

    Parameters:
        X : array of shape (n_trials, n_voxels, n_timepoints)
        n_baseline_timepoints : number of timepoints at start of trial to use 
                                as baseline reference (default=3)
    Returns:
        X_normalized :  same shape as X, each voxel expressed as change
                        from its own baseline mean

    """

    # Compute baseline mean for each voxel in each trial
    # baseline shape: (n_trials, n_voxels, 1)
    # the keepdims=True keeps the third dimension so subtraction broadcasts correctly
    baseline = X[:, :, :n_baseline_timepoints].mean(axis=2, keepdims=True)

    # Subtract baseline from each timepoint to get change from baseline
    X_normalized = X - baseline

    return X_normalized

