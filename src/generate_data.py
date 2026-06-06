"""Part 1 — synthetic data generator."""
import numpy as np
from scipy.stats import gamma

def generate_hrf(duration=20, tr=1.0):
    """
    Generate a hemodynamic response function (HRF).
    
    Parameters:
        duration : total time of the response in seconds
        tr       : time resolution in seconds (1.0 = 1 image per second,
                   matching the 1Hz refresh rate in Norman et al.)
    
    Returns:
        hrf : normalized array of shape (duration/tr,)
        t   : time axis in seconds
    """
    t = np.arange(0, duration, tr)
    hrf = gamma.pdf(t, a=6, scale=1)
    hrf = hrf / hrf.max()  # normalize peak to 1.0
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
                    noise_std=0.05, tr=1.0, trial_duration=20):
    """
    Generate synthetic fUS trials for all 5 fingers.
    
    Parameters:
        n_trials_per_finger : number of trials per finger (100 = 500 total)
        image_size          : width and height of fUS image in voxels
        patch_radius        : radius of each finger's active patch
        signal_strength     : peak blood volume change (0.15 = 15% above baseline,
                              matching Mace et al. observations)
        noise_std           : standard deviation of background noise
        tr                  : time resolution in seconds (1Hz like Norman et al.)
        trial_duration      : length of each trial in seconds

    Returns:
        X : array of shape (n_trials, n_voxels, n_timepoints)
            each trial is a flattened 2D image over time
        y : array of shape (n_trials,) 
            integer label 0-4 indicating which finger was touched
        patches  : the spatial patch masks (for later visualization)
        centers  : the patch center coordinates (for later visualization)
    """
    finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
    patches, centers = generate_finger_patches(image_size, patch_radius)
    hrf, _ = generate_hrf(duration=trial_duration, tr=tr)
    
    n_fingers = len(finger_names)
    n_trials = n_trials_per_finger * n_fingers
    n_timepoints = len(hrf)
    n_voxels = image_size * image_size
    
    # Pre-allocate output arrays
    X = np.zeros((n_trials, n_voxels, n_timepoints))
    y = np.zeros(n_trials, dtype=int)
    
    trial_idx = 0
    
    for finger_idx, name in enumerate(finger_names):
        mask = patches[name].flatten()  # flatten 128x128 -> 16384 voxels
        
        for _ in range(n_trials_per_finger):
            # Start with pure noise across all voxels and timepoints
            trial = np.random.normal(loc=0.0, scale=noise_std,
                                     size=(n_voxels, n_timepoints))
            
            # Add the HRF signal to the active patch voxels only
            # Each active voxel gets the HRF shape scaled by signal_strength
            # plus a small per-voxel random variation (+/- 20% of signal)
            for voxel in np.where(mask)[0]:
                voxel_scale = signal_strength * np.random.uniform(0.8, 1.2)
                trial[voxel, :] += voxel_scale * hrf
            
            X[trial_idx] = trial
            y[trial_idx] = finger_idx
            trial_idx += 1
    
    # Shuffle trials so fingers aren't in blocks
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