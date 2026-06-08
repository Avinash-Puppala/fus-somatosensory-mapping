"""
Spatiotemporal decoder for fUS somatosensory mapping.

Architecture:
    TemporalEncoder   — 1D CNN applied per-voxel (shared weights); learns
                        HRF-like filters without a hardcoded time window.
    SpatialAttention  — learned weighted sum over voxels; geometry-agnostic
                        so the model transfers to different probe layouts.
    Classifier        — two-layer MLP → 5 finger logits.

The input interface is (batch, voxels, timepoints) throughout, matching the
existing (n_trials, n_voxels, n_timepoints) convention in the rest of the
pipeline. When real fUS data arrives, only the DataLoader changes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalEncoder(nn.Module):
    """
    Apply a shared 1D CNN to every voxel's timeseries independently.

    By reshaping (B, V, T) → (B*V, 1, T) before the convolutions, a single
    set of filters is applied to all voxels simultaneously. This means:
      - The model learns what an HRF looks like, not where it is.
      - No hardcoded memory window (replaces the fixed 3-8s window in decoder.py).
      - Works with any T ≥ kernel_size (real data may have different trial lengths).

    Output: (B, V, out_channels) — one feature vector per voxel per trial.
    """

    def __init__(self, out_channels: int = 32):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, out_channels, kernel_size=3, padding=1)
        self.pool  = nn.AdaptiveAvgPool1d(1)   # collapses T adaptively

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, V, T)
        B, V, T = x.shape
        x = x.reshape(B * V, 1, T)            # treat each voxel as a sequence
        x = F.relu(self.conv1(x))             # (B*V, 16, T)
        x = F.relu(self.conv2(x))             # (B*V, out_channels, T)
        x = self.pool(x).squeeze(-1)          # (B*V, out_channels)
        return x.reshape(B, V, -1)            # (B, V, out_channels)


class SpatialAttention(nn.Module):
    """
    Learn a scalar importance weight for each voxel, then collapse V by
    taking a weighted sum.

    This replaces the ANOVA feature selection step. Because the weights are
    learned jointly with the rest of the model (gradients flow through),
    the model selects voxels that are discriminative *for classification*,
    not just voxels that differ across conditions on average.

    get_weights() returns the raw attention map for visualization — it should
    highlight the same regions ANOVA would, which is a useful sanity check.

    Output: (B, in_channels) — one fused feature vector per trial.
    """

    def __init__(self, in_channels: int = 32):
        super().__init__()
        self.score = nn.Linear(in_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, V, C)
        weights = torch.softmax(self.score(x), dim=1)   # (B, V, 1)
        return (weights * x).sum(dim=1)                  # (B, C)

    def get_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Return attention weights for visualization. Shape: (B, V)."""
        with torch.no_grad():
            return torch.softmax(self.score(x), dim=1).squeeze(-1)


class FingerprintDecoder(nn.Module):
    """
    Full end-to-end decoder: temporal encoding → spatial attention → classifier.

    Fine-tuning path for real data:
        Freeze temporal_encoder (it already knows what an HRF looks like).
        Re-train spatial_attention + classifier on the new data.
        This works because the temporal module is data-source agnostic —
        it learns HRF shape, which is the same signal in both synthetic and
        real fUS.
    """

    def __init__(self, n_classes: int = 5, temporal_channels: int = 32):
        super().__init__()
        self.temporal_encoder  = TemporalEncoder(out_channels=temporal_channels)
        self.spatial_attention = SpatialAttention(in_channels=temporal_channels)
        self.classifier = nn.Sequential(
            nn.Linear(temporal_channels, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, V, T)
        x = self.temporal_encoder(x)    # (B, V, C)
        x = self.spatial_attention(x)   # (B, C)
        return self.classifier(x)       # (B, n_classes)

    def get_voxel_importance(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return per-voxel attention weights for visualization.
        Replaces the LDA weight map from the old pipeline.
        Shape: (B, V)
        """
        features = self.temporal_encoder(x)
        return self.spatial_attention.get_weights(features)
