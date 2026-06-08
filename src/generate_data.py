"""Part 1 — synthetic data generator."""
import numpy as np
from scipy.stats import gamma

def generate_hrf(duration=20, tr=1.0, peak_time=6.0):
    """
    Generate a hemodynamic response function (HRF).

    Parameters:
        duration  : total time of the response in seconds
        tr        : time resolution in seconds (1.0 = 1 image per second,
                    matching the 1Hz refresh rate in Norman et al.)
        peak_time : time in seconds at which the HRF peaks (default=6.0).
                    Domain randomization varies this so the model cannot rely
                    on a fixed memory window.

    Returns:
        hrf : normalized array of shape (duration/tr,)
        t   : time axis in seconds
    """
    t = np.arange(0, duration, tr)
    # peak ≈ (a-1)*scale for a gamma; with a=6, scale = peak_time/5
    scale = max(peak_time / 5.0, 0.5)
    hrf = gamma.pdf(t, a=6, scale=scale)
    hrf = hrf / hrf.max()
    return hrf, t

def generate_finger_patches(image_size=128, patch_radius=8):
    """
    Define the spatial location of each finger's S1 patch.
    
    Parameters:
        image_size   : width and height of the fUS image in voxels
        patch_radius : radius of each finger's active patch in voxels

    Returns:
        patches : dict mapping finger name to a 2D boolean mask
                  (True = voxels that activate for that finger)
        centers : dict mapping finger name to (row, col) center coordinate
    """
    # Finger centers arranged in a horizontal line across the middle of the image
    # This mirrors the real somatotopic organization of S1
    finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
    
    # Space them evenly across the horizontal axis
    # From 20% to 80% of image width, centered vertically
    cols = np.linspace(0.20, 0.80, 5) * image_size  
    row = 0.5 * image_size  # vertical center
    
    centers = {}
    patches = {}
    
    for i, name in enumerate(finger_names):
        center = (int(row), int(cols[i]))
        centers[name] = center
        
        # Build a circular mask around this center
        mask = np.zeros((image_size, image_size), dtype=bool)
        for r in range(image_size):
            for c in range(image_size):
                dist = np.sqrt((r - center[0])**2 + (c - center[1])**2)
                if dist <= patch_radius:
                    mask[r, c] = True
        
        patches[name] = mask
    
    return patches, centers

def generate_trials(n_trials_per_finger=100, image_size=128,
                    patch_radius=8, signal_strength=0.15,
                    noise_std=0.05, tr=1.0, trial_duration=20,
                    trial_reliability=0.85, leakage_fraction=0.15,
                    domain_randomize=True):
    """
    Generate synthetic fUS trials for all 5 fingers.

    Parameters:
        n_trials_per_finger : number of trials per finger (100 = 500 total)
        image_size          : width and height of fUS image in voxels
        patch_radius        : radius of each finger's active patch (used as
                              centre of range when domain_randomize=True)
        signal_strength     : peak blood volume change (centre of range when
                              domain_randomize=True)
        noise_std           : std of background Gaussian noise (centre of range)
        tr                  : time resolution in seconds (1Hz like Norman et al.)
        trial_duration      : length of each trial in seconds
        trial_reliability   : fraction of trials that produce an HRF response
        leakage_fraction    : fraction of signal bleeding into adjacent patches
        domain_randomize    : if True, sample noise_std, signal_strength,
                              patch_radius, HRF peak timing, trial_reliability,
                              and leakage_fraction from ranges around their
                              default values. Forces the model to learn features
                              that are robust across conditions rather than
                              specific to one set of parameters.

    Returns:
        X        : array of shape (n_trials, n_voxels, n_timepoints)
        y        : array of shape (n_trials,) with finger labels 0-4
        patches  : spatial patch masks (for visualization)
        centers  : patch center coordinates (for visualization)
    """
    finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
    n_fingers = len(finger_names)
    n_trials = n_trials_per_finger * n_fingers

    # --- Domain randomization: sample trial-level parameters ---
    # Each trial gets its own noise level, signal strength, and HRF shape.
    # This prevents the model from overfitting to one set of simulator parameters.
    if domain_randomize:
        trial_noise_stds      = np.random.uniform(0.03, 0.10, n_trials)
        trial_signal_strengths = np.random.uniform(0.08, 0.25, n_trials)
        trial_peak_times      = np.random.uniform(4.0, 10.0, n_trials)
        trial_reliabilities   = np.random.uniform(0.70, 0.95, n_trials)
        trial_leakages        = np.random.uniform(0.05, 0.30, n_trials)
        trial_patch_radii     = np.random.randint(6, 13, n_trials)
    else:
        trial_noise_stds       = np.full(n_trials, noise_std)
        trial_signal_strengths = np.full(n_trials, signal_strength)
        trial_peak_times       = np.full(n_trials, 6.0)
        trial_reliabilities    = np.full(n_trials, trial_reliability)
        trial_leakages         = np.full(n_trials, leakage_fraction)
        trial_patch_radii      = np.full(n_trials, patch_radius, dtype=int)

    # Use a fixed patch layout for visualization consistency; per-trial radius
    # only affects which voxels receive signal, not the stored patch masks.
    patches, centers = generate_finger_patches(image_size, patch_radius)

    n_timepoints = int(trial_duration / tr)
    n_voxels = image_size * image_size

    X = np.zeros((n_trials, n_voxels, n_timepoints))
    y = np.zeros(n_trials, dtype=int)

    # Precompute distance maps for every finger center once.
    # dist_maps[name] shape: (n_voxels,) — Euclidean distance from center.
    # Per-trial masking then becomes a single numpy comparison (no Python loop).
    rows_grid, cols_grid = np.mgrid[0:image_size, 0:image_size]
    dist_maps = {}
    for fname in finger_names:
        cx, cy = centers[fname]
        dist_maps[fname] = np.sqrt(
            (rows_grid - cx) ** 2 + (cols_grid - cy) ** 2
        ).flatten()

    # Time axis for physiological noise
    t = np.arange(n_timepoints) * tr

    trial_idx = 0
    for finger_idx, name in enumerate(finger_names):
        adjacent_names = []
        if finger_idx > 0:
            adjacent_names.append(finger_names[finger_idx - 1])
        if finger_idx < n_fingers - 1:
            adjacent_names.append(finger_names[finger_idx + 1])

        for _ in range(n_trials_per_finger):
            t_noise_std = trial_noise_stds[trial_idx]
            t_sig       = trial_signal_strengths[trial_idx]
            t_peak      = trial_peak_times[trial_idx]
            t_rel       = trial_reliabilities[trial_idx]
            t_leak      = trial_leakages[trial_idx]
            t_radius    = trial_patch_radii[trial_idx]

            # Vectorized mask: one comparison instead of 16,384 iterations
            active_mask = dist_maps[name] <= t_radius

            trial = np.random.normal(0.0, t_noise_std, (n_voxels, n_timepoints))

            cardiac_amp     = 0.2  * t_noise_std * np.random.uniform(0.5, 1.5)
            respiratory_amp = 0.15 * t_noise_std * np.random.uniform(0.5, 1.5)
            physio_noise = (
                cardiac_amp     * np.sin(2 * np.pi * 1.0 * t + np.random.uniform(0, 2 * np.pi)) +
                respiratory_amp * np.sin(2 * np.pi * 0.3 * t + np.random.uniform(0, 2 * np.pi))
            )
            trial += physio_noise[np.newaxis, :]

            if np.random.random() < t_rel:
                hrf, _ = generate_hrf(duration=trial_duration, tr=tr,
                                      peak_time=t_peak)

                active_voxels = np.where(active_mask)[0]
                voxel_scales  = t_sig * np.random.uniform(0.8, 1.2, len(active_voxels))
                trial[active_voxels] += voxel_scales[:, np.newaxis] * hrf

                for adj_name in adjacent_names:
                    adj_mask   = dist_maps[adj_name] <= t_radius
                    adj_voxels = np.where(adj_mask)[0]
                    adj_scales = t_sig * t_leak * np.random.uniform(0.8, 1.2, len(adj_voxels))
                    trial[adj_voxels] += adj_scales[:, np.newaxis] * hrf

            X[trial_idx] = trial
            y[trial_idx] = finger_idx
            trial_idx += 1

    shuffle_idx = np.random.permutation(n_trials)
    X = X[shuffle_idx]
    y = y[shuffle_idx]

    return X, y, patches, centers

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Test HRF
    hrf, t = generate_hrf()
    print("HRF shape:", hrf.shape)
    print("Peak at t =", t[np.argmax(hrf)], "seconds")

    # Test finger patches
    patches, centers = generate_finger_patches()

    # Visualize all 5 patches on one image
    fig, ax = plt.subplots(figsize=(8, 8))
    
    combined = np.zeros((128, 128))
    colors_map = {'thumb': 1, 'index': 2, 'middle': 3, 'ring': 4, 'pinky': 5}
    
    for name, mask in patches.items():
        combined[mask] = colors_map[name]
    
    im = ax.imshow(combined, cmap='tab10', vmin=0, vmax=10)
    
    for name, (r, c) in centers.items():
        ax.text(c, r, name, ha='center', va='center',
                fontsize=10, fontweight='bold', color='white')
    
    ax.set_title('Simulated S1 Somatotopic Map\n(5 finger patches on 128x128 grid)')
    ax.set_xlabel('Voxels (x)')
    ax.set_ylabel('Voxels (y)')
    plt.tight_layout()
    plt.savefig('outputs/figures/somatotopic_map.png', dpi=150)
    plt.show()
    
    print("\nFinger patch centers:")
    for name, center in centers.items():
        voxels_active = patches[name].sum()
        print(f"  {name}: center={center}, active voxels={voxels_active}")

    # Test trial generator
    print("\nGenerating trials...")
    X, y, patches, centers = generate_trials(n_trials_per_finger=100)
    
    print(f"X shape: {X.shape}  (trials x voxels x timepoints)")
    print(f"y shape: {y.shape}")
    print(f"Trials per finger: {[(y==i).sum() for i in range(5)]}")
    print(f"X value range: {X.min():.3f} to {X.max():.3f}")
    
    example_finger = 0
    example_trial_idx = np.where(y == example_finger)[0][0]
    
    active_mask = patches['thumb'].flatten()
    active_voxel = np.where(active_mask)[0][0]
    inactive_voxel = np.where(~active_mask)[0][0]
    
    plt.figure(figsize=(10, 4))
    plt.plot(X[example_trial_idx, active_voxel, :], 
             color='tomato', linewidth=2, label='Active voxel (thumb patch)')
    plt.plot(X[example_trial_idx, inactive_voxel, :], 
             color='steelblue', linewidth=2, label='Inactive voxel (noise only)')
    plt.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Blood volume change')
    plt.title('Single Trial — Active vs Inactive Voxel Signal')
    plt.legend()
    plt.tight_layout()
    plt.savefig('outputs/figures/example_trial.png', dpi=150)
    plt.show()