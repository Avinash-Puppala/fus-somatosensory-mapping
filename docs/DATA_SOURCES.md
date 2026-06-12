# Real fUS data — sources, status, and how to bring one in

I cannot download these from the sandbox (no network egress). You download into
`data/`, then convert + run. Each entry lists the landing page, the realistic
availability, and what to look for in its **Data Availability** statement.

> Important: none of these is a somatosensory finger task. They validate that the
> pipeline survives *real fUS signal*, with a different decode target (and
> different expected accuracy). Treat them as engine tests, not finger-task
> validation.

## 0. OpenfUS / PyfUS example dataset — OPEN, direct download (start here)

- Record: https://zenodo.org/records/13341387  (DOI 10.5281/zenodo.13341387, CC-BY-4.0)
- Direct file: https://zenodo.org/records/13341387/files/example_dataset_gratings.zip?download=1
  (9.1 GB, md5 c815c8de6e78d33012962a5212a214ab)
- Contents: 3 mice, 25 sessions, 5 trials/session, matrix-array transducer at
  2 Hz (TR = 0.5 s). Visual drifting gratings, 0° (forward) vs 180° (backward).
- Target: grating orientation — a real **2-class** decode (chance 50%) → straight
  into the classification path.
- Notes: no registration. 9.1 GB — process per-session (sandbox RAM is ~3.8 GB).
  On-disk format is PyfUS's own; inspect one session, then map it with
  convert_to_npz.py (`--tr 0.5`). Continuous sessions → epoch first.

## 1. PNAS — single-trial self-motion decoding (recommended)

- Paper: https://www.pnas.org/doi/10.1073/pnas.2414354122
- Target: physical self-motion condition (categorical) from macaque hemodynamics.
- Availability: the Data Availability statement says data + code are deposited at
  the **Brain Science Data Center, Chinese Academy of Sciences**, publicly
  available. Open the paper's "Data Availability" section for the exact accession
  URL; a free BSDC registration may be required.
- Fit: single-trial categorical fUS decoding — drops straight into the
  **classification** path.

## 2. OfUSA / OpenfUS Analyzer (good for preprocessing + example data)

- Paper: https://www.biorxiv.org/content/10.1101/2025.09.16.676515v1
- This is primarily an **analysis framework**, not a large dataset release. Check
  the preprint's Code/Data Availability for its GitHub/Zenodo link; any bundled
  example recordings there are the easiest "real-ish" data to try first.
- Fit: use bundled rodent/primate example movies via the **classification** path
  (or the **epocher** if a recording is continuous).

## 3. Norman et al. 2021 — movement intentions (closest task, least open)

- Paper: https://www.cell.com/neuron/fulltext/S0896-6273(21)00151-3
  (open PDF: https://www.vis.caltech.edu/documents/18635/neuron_article_2021.pdf)
- Target: reach/saccade direction (categorical) from posterior parietal cortex —
  methodologically nearest to ours.
- Availability: no open download surfaced; your README already marks access as
  request-pending. Realistically an email-the-authors path.

## Bringing a dataset in (any of the above)

```bash
# 1) convert raw .mat / HDF5 -> the canonical .npz (run where scipy/h5py exist)
python scripts/convert_to_npz.py recording.mat data/real.npz \
    --x-var data --y-var labels --axes t v n --tr 0.4 --baseline 4

# 2a) categorical target (most cases)
python run_dataset.py data/real.npz

# 2b) continuous target (e.g. a kinematic)
python run_dataset.py data/real.npz --task regression

# optional: remove N leading low-rank nuisance modes first
python run_dataset.py data/real.npz --clutter 1
```

`--x-var/--y-var` are the variable names inside the file; `--axes` is the order of
its X dimensions mapped onto (trials n, voxels v, timepoints t). If a dataset is a
continuous movie + event list rather than pre-cut trials, epoch it first with
`npkit.epoch.epoch_continuous(recording, onsets, window, pre)`.
