"""
npkit.epoch — segment a continuous fUS recording into trials.

The decoder consumes (n_trials, n_voxels, n_timepoints). Some datasets instead
ship a continuous power-Doppler movie plus a list of event onsets. This module
slices the movie into fixed windows locked to each event, producing exactly the
array the rest of the pipeline expects.
"""

from __future__ import annotations
import numpy as np


def epoch_continuous(recording: np.ndarray, onsets, window: int,
                     pre: int = 0, labels=None, time_axis: int = 1):
    """
    Cut a continuous recording into event-locked trials.

    Parameters
    ----------
    recording : 2-D array of (voxels, time) or (time, voxels); set time_axis.
    onsets    : iterable of integer frame indices where each event begins.
    window    : number of frames per trial (the epoch length).
    pre       : frames to include BEFORE each onset (becomes the baseline). The
                trial spans [onset-pre, onset-pre+window).
    labels    : optional per-event labels; returned aligned to kept trials.
    time_axis : which axis of `recording` is time (default 1 -> (voxels, time)).

    Returns
    -------
    X : (n_kept_trials, n_voxels, window)
    y : (n_kept_trials,) labels, or None if labels not supplied

    Events whose window falls outside the recording are dropped (with their
    labels), so partial trials never reach the decoder.
    """
    rec = recording if time_axis == 1 else recording.T   # -> (voxels, time)
    n_voxels, n_time = rec.shape
    onsets = np.asarray(list(onsets), dtype=int)

    starts = onsets - pre
    valid = (starts >= 0) & (starts + window <= n_time)
    if not valid.all():
        kept = int(valid.sum())
        print(f"epoch_continuous: dropping {len(onsets) - kept} of {len(onsets)} "
              f"events whose window exceeds the recording bounds")
    starts = starts[valid]

    X = np.empty((len(starts), n_voxels, window), dtype=rec.dtype)
    for i, s in enumerate(starts):
        X[i] = rec[:, s:s + window]

    y = None
    if labels is not None:
        y = np.asarray(list(labels))[valid]
    return X, y
