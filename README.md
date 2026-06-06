# fUS Somatosensory Mapping Pipeline

A computational pipeline for functional ultrasound (fUS) guided somatotopic 
mapping of primary somatosensory cortex (S1), designed as a targeting system 
for transcranial focused ultrasound (TFUS) sensory restoration in amputees.

## Motivation

Amputees retain an intact somatotopic map in S1 even after limb loss. If we 
can precisely identify which S1 voxels correspond to which phantom limb 
locations using fUS neuroimaging, we can target TFUS stimulation pulses to 
evoke specific sensations — restoring sensory feedback for prosthetic users.

## Pipeline Phases

- **Phase A (current):** Synthetic fUS data simulating somatotopic finger 
  mapping in S1 — validate the full decoding pipeline before real data
- **Phase B (next):** Transition to real neuroimaging data
- **Phase C (future):** Replace linear decoder with neural network

## Scientific Foundation

Built on methods from:
- Norman et al. 2021 (Neuron) — single-trial fUS decoding
- Macé et al. 2011 (Nature Methods) — fUS imaging foundation  
- Demené et al. 2015 (IEEE TMI) — SVD clutter filtering
- Legon et al. 2014 (Nature Neuroscience) — human TFUS of S1

## Project Structure

\`\`\`
fus_somatosensory_mapping/
├── src/
│   ├── generate_data.py    # Phase A: synthetic fUS data generator
│   ├── preprocess.py       # Baseline normalization + feature selection
│   ├── decoder.py          # CPCA + LDA decoder (Norman et al. method)
│   └── visualize.py        # Somatotopic map reconstruction
├── data/synthetic/         # Generated data (not tracked)
├── outputs/figures/        # Output figures (not tracked)
└── main.py                 # Full pipeline runner
\`\`\`

## Setup

\`\`\`bash
python3 -m venv venv
source venv/bin/activate
pip install numpy scipy matplotlib scikit-learn
python src/generate_data.py
\`\`\`
