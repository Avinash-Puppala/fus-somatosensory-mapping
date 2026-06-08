"""Part 4 — maps and evaluation."""
import numpy as np
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm, finger_names, save_path=None):
    """
    Plot the confusion matrix from decoder results.

    Parameters:
        cm           : (5, 5) confusion matrix array from decode()
        finger_names : list of 5 finger name strings
        save_path    : optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(finger_names, rotation=45, ha='right')
    ax.set_yticklabels(finger_names)
    ax.set_xlabel('Predicted finger')
    ax.set_ylabel('True finger')
    ax.set_title(f'Confusion matrix\n(accuracy: {cm.diagonal().sum() / cm.sum():.3f})')
    plt.colorbar(im, ax=ax, label='Trial count')

    for i in range(5):
        for j in range(5):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black',
                    fontsize=12)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_weight_map(decoder_weights, feature_mask, centers, image_size=128, save_path=None):
    """
    Project decoder weights back onto the full brain image and plot.

    Parameters:
        decoder_weights : (n_selected_voxels,) importance array from decode()
        feature_mask    : (n_all_voxels,) boolean mask from select_features()
        centers         : dict mapping finger name to (row, col) patch center
        image_size      : width/height of the brain image (default 128)
        save_path       : optional path to save the figure
    """
    # Expand selected voxel weights back to full image space
    full_weights = np.zeros(feature_mask.shape)
    full_weights[feature_mask] = decoder_weights
    weight_map = full_weights.reshape(image_size, image_size)

    fig, ax = plt.subplots(figsize=(7, 7))

    im = ax.imshow(weight_map, cmap='hot')
    plt.colorbar(im, ax=ax, label='Decoder importance')

    # Overlay true patch centers so we can confirm alignment
    for name, (r, c) in centers.items():
        ax.plot(c, r, 'w+', markersize=15, markeredgewidth=2)
        ax.text(c, r - 12, name, ha='center', color='white', fontsize=9)

    ax.set_title('Decoder weight map\n(bright = voxels driving classification)')
    ax.set_xlabel('Voxels (x)')
    ax.set_ylabel('Voxels (y)')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_per_fold_accuracy(per_fold_accuracy, save_path=None):
    """
    Bar chart of accuracy across the 5 cross-validation folds.

    Parameters:
        per_fold_accuracy : list of 5 accuracy values from decode()
        save_path         : optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    folds = [f'Fold {i+1}' for i in range(len(per_fold_accuracy))]
    bars = ax.bar(folds, per_fold_accuracy, color='steelblue', edgecolor='white')
    ax.axhline(y=0.2, color='red', linestyle='--', linewidth=1.5, label='Chance (0.20)')
    ax.axhline(y=np.mean(per_fold_accuracy), color='green', linestyle='--',
               linewidth=1.5, label=f'Mean ({np.mean(per_fold_accuracy):.3f})')
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Accuracy')
    ax.set_title('Decoder accuracy per fold')
    ax.legend()

    for bar, acc in zip(bars, per_fold_accuracy):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


if __name__ == "__main__":
    import sys
    sys.path.insert(0, 'src')
    from generate_data import generate_trials
    from preprocess import baseline_normalize, select_features
    from decoder import decode

    finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']

    print("Generating trials...")
    X, y, patches, centers = generate_trials(n_trials_per_finger=100)

    print("Normalizing...")
    X_norm = baseline_normalize(X)

    print("Selecting features...")
    X_selected, feature_mask, p_values = select_features(X_norm, y)
    print(f"  {X_selected.shape[1]} voxels selected")

    print("\nRunning decoder...")
    results = decode(X_selected, y, n_components=20, n_folds=5)
    print(f"  Mean accuracy: {results['accuracy']:.3f}")

    print("\nPlotting results...")
    plot_confusion_matrix(
        results['confusion_matrix'],
        finger_names,
        save_path='outputs/figures/confusion_matrix.png'
    )

    plot_weight_map(
        results['decoder_weights'],
        feature_mask,
        centers,
        save_path='outputs/figures/weight_map.png'
    )

    plot_per_fold_accuracy(
        results['per_fold_accuracy'],
        save_path='outputs/figures/per_fold_accuracy.png'
    )

    print("\nVisualization complete.")
