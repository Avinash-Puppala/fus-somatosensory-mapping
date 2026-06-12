# npkit — NumPy-only pipeline, results & findings

This document summarizes a dependency-light rebuild of the decoding pipeline
(`src/npkit/`) and the experiments run on it. It is meant to be read top to
bottom to understand both *what* changed and *why*.

## Why a rebuild

The original pipeline depends on `scipy`, `scikit-learn`, and `torch`. The
working environment could not install any of those, so the entire pipeline was
re-implemented using **NumPy only**. This was not just a port — it was an
opportunity to tighten the methodology and to test a premise: *does this problem
actually need a neural network?*

Each numerical primitive that scipy/sklearn provided was re-implemented and
unit-tested against analytically known values (`tests/test_stats.py`):
the gamma HRF, the F-distribution tail (via the regularized incomplete beta),
PCA (via SVD), and LDA (written out explicitly with shrinkage).

## The pipeline (`src/npkit/`)

| module | role |
|---|---|
| `stats.py` | gamma HRF, incomplete-beta, F-distribution survival (no scipy) |
| `data.py` | synthetic fUS generator (faithful port; float32) |
| `preprocess.py` | baseline normalization, **adaptive HRF window**, ANOVA+FDR selection |
| `temporal.py` | four temporal feature extractors incl. **matched filter** |
| `decoder.py` | PCA + **shrinkage-LDA**, with confidence margins |
| `evaluate.py` | leakage-free, **seed-averaged** CV; selective-prediction curve |

Entry point: `python run_npkit.py` (≈6 s, ≈2 GB RAM, CPU only).

## Methodological fixes vs. the original

1. **Circular feature selection** — fixed: ANOVA, window estimation, PCA and LDA
   are all fit inside each CV fold on training trials only.
2. **Hard-coded 3–8 s HRF window** — replaced with `estimate_peak_window()`,
   estimated per fold from the training task voxels.
3. **Flat-mean time collapse** — replaced with a **matched filter** (projection
   onto the HRF shape). The `matchedbank` variant maxes over templates spanning
   peak times 4–10 s, making the feature robust to per-trial HRF timing jitter.
4. **Single-split evaluation** — replaced with seed-averaged CV reporting
   mean ± std, so noise is not mistaken for signal.

## Results

### Temporal-feature ablation (500 trials, 3 seeds, identical decoder & CV)

| temporal feature | accuracy |
|---|---|
| fixed 3–8 s mean (original method) | 0.836 ± 0.002 |
| adaptive window mean | 0.841 ± 0.004 |
| matched filter (1 template) | 0.844 ± 0.003 |
| **matched bank (timing-robust)** | **0.847 ± 0.002** |

### The accuracy ceiling is set by the data, not the decoder

Turning the simulator's degradations on one at a time (`diagnose_ceiling.py`):

| condition | accuracy |
|---|---|
| clean signal | 1.000 |
| + realistic noise | 1.000 |
| + 17.5 % trial dropout | 0.840  (ceiling ≈ 0.86) |
| + cross-patch leakage | 0.840 |
| full domain randomization | 0.847 |

The decoder is **perfect when signal is present**. The entire gap from 100 %
is explained by *trial dropout* — trials the simulator sets to pure noise, which
are unclassifiable by any model. **Conclusion: on this data a deeper model
(Phase C neural net) cannot beat a 4-second linear decoder.** The lever for
higher accuracy is better data (real recordings, or a simulator whose
"non-responsive" trials retain partial signal), not decoder capacity.

### Selective prediction — the operationally relevant result

For a TFUS targeting device, acting only on confident decodes matters more than
raw accuracy. Ranking trials by the LDA confidence margin (`selective_prediction.py`):

| coverage (act on most-confident) | accuracy |
|---|---|
| 100 % | 84.6 % |
| 82 % | 99.0 % |
| ≤ 78 % | 100.0 % |

The margin cleanly separates responsive trials (decoded ~99–100 %) from the
non-responsive ones the device should skip. This is the most useful single
addition for the application: **stimulate when confident, abstain otherwise.**

## Speed

The full seed-averaged, 5-fold, matched-bank pipeline over 500 trials runs in
~6 s on 4 CPU cores at ~2 GB RAM, with no GPU and no deep-learning framework.
The original trains five torch models (one per fold) for 30 epochs each.

## Dataset-agnostic layer (bringing real data)

The decoder no longer hard-codes anything synthetic-specific. Class count, voxel
count, timepoints, TR, baseline length, plot labels, and 2-D map geometry are
all inferred from the data or a small `DatasetSpec`. The matched-filter bank
builds its templates from the recording's own timepoint count and TR, so it
adapts to any acquisition rather than the synthetic 1 Hz / 20-timepoint setup.

Pipeline for a real recording:

1. On a machine with scipy/h5py, convert the raw `.mat`/HDF5 to the canonical
   `.npz`:
   ```
   python scripts/convert_to_npz.py recording.mat data/real.npz \
       --x-var data --y-var labels --axes t v n --tr 0.4 --baseline 4
   ```
2. Decode with the exact same pipeline:
   ```
   python run_dataset.py data/real.npz
   ```

`src/npkit/dataset.py` loads with `allow_pickle=False`, so a data file can never
execute code on load; pickle-bearing files are refused with a clear message.
`tests/test_dataset.py` verifies a 3-class / 10-timepoint / TR=0.5 set decodes
through the same path, the `.npz` round-trips, and the pickle guard fires.

Note: real fUS datasets (Norman PPC motor-intent, PNAS vestibular self-motion)
are *not* somatosensory finger tasks. Slotting them in validates that the
pipeline survives real fUS signal characteristics; the decode target (and the
expected accuracy) changes accordingly.

## Open threads

- **Real fUS data** (next): slot recordings into the same
  `(trials, voxels, timepoints)` interface; the loader is the only new code
  needed. Candidate public sources are listed in the README.
- **Simulator realism**: model partial/variable responses rather than all-or-
  nothing dropout, so the ceiling reflects graded SNR like real tissue.
- **Trial-quality model**: predict responsiveness explicitly and fold it into
  the selective-prediction threshold.
