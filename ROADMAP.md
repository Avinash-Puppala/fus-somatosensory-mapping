# Project Roadmap & Direction

_Last updated: 2026-06-09. Supersedes the "Pipeline Phases" section of the README,
which is partially stale (see Housekeeping)._

## The reframe

The center of gravity of this project has moved. It is no longer "build and tune a
decoder" — the decoder works and the synthetic pipeline is validated. The bottleneck
now is **data and ingestion**, not the model. Specifically:

- A device purchase does **not** help. The decoder needs *data*, not hardware, and
  human fUS requires a surgically implanted acoustic cranial window regardless — not a
  thing we can do ourselves.
- Our current accuracy (85.4% on synthetic) is a fact about our **simulator**, not about
  the decoder's real-world ability. The accuracy number that matters — single-trial
  decoding on real fUS — **does not exist yet** because no real multiclass data has been
  run through the pipeline.
- The open mouse dataset we have hardens **ingestion, preprocessing, registration, and
  visualization** — and builds the atlas-targeting machinery we need later — but it
  **cannot** test or improve the decoder (one visual condition, no classes).

So the direction is: prove the pipeline ingests real fUS and recovers known biology,
and secure real multiclass somatosensory data. The decoder improvement work (Phase C)
comes *after* there is a real baseline to beat.

## Status

- **Phase A — synthetic validation: COMPLETE.** Full pipeline runs end to end.
  Verified mean accuracy 85.4% (5-fold, single-trial, 5-class, chance 20%) with
  realistic noise; ~100% on clean synthetic.
- **Real-data accuracy: DOES NOT EXIST YET.** This is the gap.

## Housekeeping (do first, cheap)

- [ ] Update README "Known issues": both listed bugs are **already fixed** in `decoder.py`
  — feature selection runs inside the CV loop (no leakage), and `estimate_memory_period`
  estimates the HRF window adaptively. The README still lists them as open.
- [ ] Update README "Pipeline Phases" to point at this roadmap.

## Track 1 — Real-data ingestion & infrastructure  (START NOW; uses the mouse data we have)

Goal: prove we can get real Power Doppler fUS into the pipeline and recover known biology.
Deliverable is a **validated ingestion + registration pipeline**, NOT an accuracy number.

- [ ] **`.mat` loader** — read `scanfus.Data` `(143,128,20,70)` volume, reshape into the
  `(trials, voxels, timepoints)` contract. Pure `scipy.io.loadmat`; no MATLAB.
- [ ] **Data-driven baseline window** — replace hardcoded `n_baseline_timepoints=3` with a
  window derived from actual stimulus timing (70 frames, unknown onset).
- [ ] **Generalize feature selection** — current ANOVA needs ≥2 classes and can't run on a
  single-condition dataset. Add a stimulus-ON vs baseline contrast (t-test / GLM) so it
  runs, and so it's decoupled from "must be 5 fingers."
- [ ] **Atlas registration + region labeling (net-new)** — use the shipped `Transformation.mat`
  affine and the Allen atlas (509 regions, incl. VISp and SSp subfields). This is the
  targeting machinery we need for TFUS later.
- [ ] **Volume + atlas visualization** — replace 2D `reshape(128,128)` + finger markers with
  plane-by-plane volume views and anatomical labels.
- [ ] **Validation smoke test** — confirm the pipeline recovers stimulus-evoked Power Doppler
  in visual cortex (VISp). If known biology comes out, the machinery is sound on real noise,
  real HRF/CBV shape, and real clutter residue.

**Do NOT** manufacture fake classes from this single-condition data to give the decoder
something to classify — that recreates the circular-validation trap we already fixed.

## Track 2 — Real decoding benchmark  (PARALLEL; long lead time — start chasing now)

Goal: the first *meaningful* accuracy number. Requires real **multiclass** data.

- [ ] **Norman et al. NHP somatosensory data** — the right target (real single-trial fUS
  somatosensory decoding; same method our decoder copies). Request is pending — chase it.
  Ask three questions up front: (1) what processing stage — raw RF / IQ / Power Doppler?
  (2) trial/event structure? (3) is there a non-MATLAB loader? Those answers decide how much
  code sits between their files and our pipeline, and whether a one-time MATLAB beamforming
  pass is unavoidable.
- [ ] **Fallback: PNAS self-motion dataset** (CAS Brain Science Data Center) — real fUS, real
  decoding, but vestibular not somatotopic. Use only if Norman stalls.
- [ ] Slot into pipeline; expect accuracy to drop from ~85% synthetic to ~50–80%. That drop
  is normal and informative — it's the first honest number.

**Risk:** NHP fUS data is often shared by collaboration agreement, not open download. This is
the single biggest schedule risk — request early so it's ready when Track 1 lands.

## Track 3 — Decoder improvement (Phase C)  (AFTER a real baseline exists)

Only meaningful once real data sets a baseline to beat. Do not start before then.

- [ ] Replace linear CPCA + LDA with a model that uses the full spatiotemporal timeseries.
  LDA assumes equal covariance across classes and collapses 20 timepoints to one number —
  a lot of discarded structure. Candidates: 1D CNN over the timeseries, LSTM, or a
  spatiotemporal transformer.
- [ ] Evaluate against the real-data baseline from Track 2, not against synthetic.

## Deprioritized

- **Simulator enhancements** (physiological noise, SVD clutter filter as a synthetic proving
  ground). Real data is now the better use of time. Keep the SVD clutter filter idea, but
  apply it to *real* data in Track 1 rather than building a harder simulator.

## Novelty thesis & direction

**Thesis: fUS as a *generalizable* somatotopic localizer for sensory write-in.**

Reproducing CPCA + LDA on the Griggs data is reproduction, not novelty — their decoder
already *is* classwise PCA. Competing head-on with Caltech / Forest Neurotech on motor
read-out decoding is a losing race (they have the scanner, the animals, the funding, a
multi-year head start). The defensible novelty is in the gaps a compute-only team can own:

The three novel directions are **not parallel projects — they are a dependency stack**:

- **Spatiotemporal decoder = substrate.** A learnable, HRF-aware representation. CPCA + LDA
  collapses 20 timepoints to one number and has nothing to transfer.
- **Transfer / generalization = the property trained into that substrate, and the headline.**
  The field recalibrates every session; nobody has cross-subject transfer. Runs entirely on
  open data. This is the publishable, no-device contribution.
- **Sensory-targeting framework = what the map is for.** map → TFUS target → acoustic field
  sim → predicted sensation. The mission, the moat, the part no one else works on.

### Sequenced program (Phase B, dependency-ordered)

- [ ] **B.0 — Reproduce (necessary, not novel).** Griggs data in → CPCA + LDA baseline →
  confirm the pipeline decodes real fUS. The floor every later result must beat.
- [ ] **B.1 — Spatiotemporal decoder.** Data-efficient, HRF-aware neural decoder; benchmark
  vs the B.0 baseline. Substrate for B.2.
- [ ] **B.2 — Transfer / generalization (headline).** Train the B.1 model cross-session, then
  cross-subject (monkey A→B); mouse set as out-of-distribution probe.
- [ ] **B.3 — Sensory-targeting framework (north star).** map → target → acoustic sim →
  predicted sensation. Sim-validate now; hand to a wet-lab collaborator later.

### Commercial hedge (goal = both / undecided)

The non-commercial license attaches to the Griggs **data**, not to **our code**. Keep them
separable so a later commercial pivot is uncompromised:

- Swappable data layer behind a clean loader interface; Griggs used strictly for research
  benchmarking, never inside a commercial artifact.
- Decoder, transfer method, and targeting framework are our own IP — commercially usable.
  A product pivot means retrain + re-benchmark on clean data, no model rewrite.
- Build this seam now (it costs nothing) rather than retrofitting it later.

### Open dependency to track

Griggs is motor PPC, **not S1 finger somatotopy**. B.1/B.2 prove the *method* generalizes;
the *sensory* claim in B.3 ultimately needs somatosensory data we don't have yet — the
Norman 2021 set (by request) or our own recordings. B.3's real-data validation is gated on
that. Track it; don't block B.0–B.2 on it.

### Immediate next step

License-safe data layer + `.mat` loader for the Griggs CaltechDATA set → B.0 baseline.
