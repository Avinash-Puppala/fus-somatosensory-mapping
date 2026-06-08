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

def select_features(X, y, q_threshold=0.05):
    """
    Select voxels that carry task-related signal using a statistical test.

    For each voxel, run a one-way ANOVA across the 5 finger conditions. 
    Voxels where signal differs signnificantly across conditions are informative.
    Voxels that don't difffer are just noise - drop them.

    This mirrors the statistical parametric mapping approach in Norman et al.

    Parameters:
        x               : array of shape (n_trials, n_voxels, n_timepoints) 
        y               : array of shape (n_trials,) with finger labels 0-4 
        q_threshold     : FDR-corrected significance threshold (default=0.05)
    Returns:
        X_selected      : array of shape (n_trials, n_selected_voxels, n_timepoints)
        feature_mask    : boolean array of shape (n_voxels,) with finger labels 0-4
        p_values        : raw p-value for every voxel (useful for visualiztion)
    """
    n_trials, n_voxels, n_timepoints = X.shape

    # For each voxel, compute its mean signal across the memory period
    # We use timepoints 3-8 — the rising and peak phase of the HRF
    # This is analogous to Norman et al.'s memory delay period
    memory_period = X[:, :, 3:8].mean(axis=2) # shape: (n_trials, n_voxels)

    # Run one-way ANOVA for each voxel
    # Groups are the 5 finger conditions
    # For each voxel: does mean activity differ across the 5 fingers?
    p_values = np.zeros(n_voxels)

    for voxel in range(n_voxels):
        groups = [memory_period[y == finger, voxel] for finger in range(5)]
        _, p_values[voxel] = stats.f_oneway(*groups)

    # Apply FDR correction
    # Sort p-values and apply Benjamini-Hochberg procedure
    n_tests = n_voxels
    sorted_idx = np.argsort(p_values)
    sorted_p_values = p_values[sorted_idx]
    
    # Benjamini-Hochberg threshold for each rank
    ranks = np.arange(1, n_tests + 1)
    bh_threshold = (ranks / n_tests) * q_threshold

    # A voxel passes if its sorted p-value is below its BH threshold
    below_threshold = sorted_p_values <= bh_threshold

    # Find the largest rank that passes - all voxels up to that rank are selected
    passing_ranks = np.where(below_threshold)[0]

    feature_mask = np.zeros(n_voxels, dtype=bool)

    if len(passing_ranks) > 0:
        cutoff = passing_ranks[-1]
        selected_sorted = sorted_idx[:cutoff + 1]
        feature_mask[selected_sorted] = True
    
    X_selected = X[:, feature_mask, :]

    return X_selected, feature_mask, p_values

if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt
    sys.path.insert(0, 'src')
    from generate_data import generate_trials

    print("Generating trials...")
    X, y, patches, centers = generate_trials(n_trials_per_finger=100)
    print(f"Raw X shape: {X.shape}")

    # Test baseline normalization
    print("\nApplying baseline normalization...")
    X_norm = baseline_normalize(X)
    print(f"Normalized X shape: {X_norm.shape}")
    print(f"Pre-normalization mean of first 3 timepoints: "
          f"{X[:, 0, :3].mean():.4f}")
    print(f"Post-normalization mean of first 3 timepoints: "
          f"{X_norm[:, 0, :3].mean():.4f}  (should be ~0)")

    # Test feature selection
    print("\nRunning feature selection...")
    X_selected, feature_mask, p_values = select_features(X_norm, y)
    print(f"Voxels before selection: {X.shape[1]}")
    print(f"Voxels after selection:  {X_selected.shape[1]}")
    print(f"Reduction: {100*(1 - X_selected.shape[1]/X.shape[1]):.1f}% of voxels removed")

    # Visualize which voxels were selected
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1 — p-value map
    p_map = p_values.reshape(128, 128)
    im1 = axes[0].imshow(np.log10(p_map + 1e-10), cmap='hot_r')
    axes[0].set_title('Voxel p-values (log scale)\nBrighter = more significant')
    axes[0].set_xlabel('Voxels (x)')
    axes[0].set_ylabel('Voxels (y)')
    plt.colorbar(im1, ax=axes[0], label='log10(p-value)')

    # Plot 2 — selected feature mask
    mask_map = feature_mask.reshape(128, 128)
    axes[1].imshow(mask_map, cmap='hot')

    # Overlay patch centers for reference
    finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
    for name, (r, c) in centers.items():
        axes[1].plot(c, r, 'w+', markersize=15, markeredgewidth=2)
        axes[1].text(c, r - 12, name, ha='center', color='white', fontsize=9)

    axes[1].set_title(f'Selected voxels (FDR corrected)\n'
                      f'{feature_mask.sum()} of {len(feature_mask)} voxels kept')
    axes[1].set_xlabel('Voxels (x)')
    axes[1].set_ylabel('Voxels (y)')

    plt.tight_layout()
    plt.savefig('outputs/figures/feature_selection.png', dpi=150)
    plt.show()

    print("\nPreprocessing complete.")