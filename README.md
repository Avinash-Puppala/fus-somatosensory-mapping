# fUS Somatosensory Mapping Pipeline

A computational pipeline for functional ultrasound (fUS) guided somatotopic
mapping of primary somatosensory cortex (S1), designed as a targeting system
for transcranial focused ultrasound (TFUS) sensory restoration in amputees.

## Motivation

Amputees retain an intact somatotopic map in S1 even after limb loss. If we
can precisely identify which S1 voxels correspond to which phantom limb
locations using fUS neuroimaging, we can target TFUS stimulation pulses to
evoke specific sensations — restoring sensory feedback for prosthetic users.

## Current Status

**Phase A is complete.** The full synthetic data pipeline runs end to end:
data generation → preprocessing → decoding → visualization. Accuracy on
synthetic data is ~85% with realistic noise parameters (down from 100% on
clean data), confirming the pipeline is working correctly and that the added
realism is meaningfully challenging the decoder.

### What is working
- Synthetic fUS data generation with 5-finger somatotopic patches on a 128×128 grid
- Hemodynamic response function (HRF) modelling at 1Hz (matching Norman et al.)
- Baseline normalization and ANOVA-based feature selection with FDR correction
- Cross-validated CPCA + LDA decoder (5-fold, Norman et al. method)
- Confusion matrix, decoder weight map, and per-fold accuracy visualization
- Realistic noise: trial reliability (15% non-responsive trials) and cross-patch
  leakage (15% signal bleed to adjacent finger patches)

### Known issues to fix before Phase B
1. **Circular feature selection** — `select_features()` currently runs on all
   500 trials before cross-validation, meaning test data leaks into feature
   selection. Fix: move feature selection inside the cross-validation loop in
   `decoder.py` so it only sees training data per fold.
2. **Hardcoded HRF window** — memory period is fixed at timepoints 3–8. Real
   fUS data will have variable HRF peak timing across subjects and sessions.
   Fix: estimate the peak window adaptively from the data.

## Pipeline Phases

### Phase A — Synthetic validation (complete)
Validate the full decoding pipeline on synthetic data before touching real
recordings. Synthetic data simulates 5-finger somatotopic mapping in S1 with
realistic HRF shapes, noise, trial dropout, and cross-patch leakage. The goal
is to confirm the pipeline mechanics are correct and establish a performance
baseline.

### Phase B — Real fUS data (next)
Transition the pipeline to real fUS recordings. Two parallel tracks:

**Bug fixes first:**
- Fix circular feature selection (see known issues above)
- Implement adaptive HRF window estimation

**Real data integration:**
- Slot real fUS recordings into the existing pipeline — the pipeline already
  accepts arbitrary `(trials, voxels, timepoints)` arrays, so minimal
  modification is needed
- Update `n_baseline_timepoints` and memory period window to match actual trial
  structure of incoming data
- Expected accuracy drop from ~85% to somewhere in the 50–80% range depending
  on data quality — this is normal and informative

**Data sources to pursue:**
- [OfUSA / OpenfUS](https://www.biorxiv.org/content/10.1101/2025.09.16.676515v1.full) —
  open-source fUS analysis framework with rodent datasets (2025)
- Norman et al. NHP somatosensory data (request pending)
- [PNAS self-motion decoding dataset](https://www.pnas.org/doi/10.1073/pnas.2414354122) —
  deposited at Brain Science Data Center, Chinese Academy of Sciences

**Simulator enhancements (parallel track):**
- Add physiological noise: cardiac (~1 Hz) and respiratory (~0.3 Hz) artifacts,
  slow CBV drift over session
- Implement SVD clutter filter (Demené et al. 2015) in `src/clutter_filter.py`
  and benchmark SNR recovery

### Phase C — Neural network decoder (future)
Replace the linear CPCA + LDA decoder with a neural network. LDA assumes equal
covariance across finger classes and discards temporal structure by collapsing
20 timepoints to one number. A convolutional or recurrent architecture can
learn spatiotemporal patterns across the full timeseries and handle non-linear
class boundaries. Candidate architectures: 1D CNN over the timeseries, LSTM,
or a spatiotemporal transformer.

**Note on gas vesicle imaging:** The Shapiro Lab at Caltech has developed
genetically encodable gas vesicles (GVs) as direct neural activity reporters
for ultrasound ([acoustic biomolecules paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC6955150/)).
If GV calcium sensors mature to sub-second kinetics in neurons, Phase C will
need to revisit the temporal model entirely — the HRF assumption breaks down
and direct neural signals would replace hemodynamic proxies.

## Scientific Foundation

- Norman et al. 2021 (Neuron) — single-trial fUS decoding
- Macé et al. 2011 (Nature Methods) — fUS imaging foundation
- Demené et al. 2015 (IEEE TMI) — SVD clutter filtering
- Legon et al. 2014 (Nature Neuroscience) — human TFUS of S1
- Shapiro Lab / Maresca et al. 2020 (NeuroImage) — acoustic biomolecules for fUS

## Project Structure

```
fus_somatosensory_mapping/
├── src/
│   ├── generate_data.py    # Phase A: synthetic fUS data generator
│   ├── preprocess.py       # Baseline normalization + feature selection
│   ├── decoder.py          # CPCA + LDA decoder (Norman et al. method)
│   └── visualize.py        # Somatotopic map reconstruction + evaluation
├── outputs/figures/        # Output figures (not tracked)
└── main.py                 # Full pipeline runner
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy scipy matplotlib scikit-learn
python main.py
```

## Running the pipeline

```bash
python main.py
```

Parameters can be adjusted at the top of `main.py`:

| Parameter | Default | Description |
|---|---|---|
| `n_trials_per_finger` | 100 | Trials generated per finger (500 total) |
| `n_components` | 20 | PCA components retained before LDA |
| `n_folds` | 5 | Cross-validation folds |
| `trial_reliability` | 0.85 | Fraction of trials that produce an HRF response |
| `leakage_fraction` | 0.15 | Signal bleed to adjacent finger patches |
