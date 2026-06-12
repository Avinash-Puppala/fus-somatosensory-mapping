# Decision Log

Append-only record of project-shaping decisions. Entries are immutable once written; to
change a decision, append a new entry that supersedes the old one (note the superseded ID).
The living statement of each decision lives in ROADMAP.md; this log records *why* and *when*.

Format per entry: Context → Decision → Why → Consequences.

---

## 0001 — Novelty positioning: generalization-first, sensory targeting as north star

- **Date:** 2026-06-11
- **Stage / task:** Stage 0, S01
- **Status:** Accepted

**Context.** As of mid-2026 the fUS frontier a compute-only team might chase is already
well served by better-resourced labs: motor read-out is demonstrated (Griggs et al. 2023,
macaque PPC, closed-loop, 8 directions, with within-subject cross-session pretraining), human
finger somatotopy has been mapped volumetrically and decoded single-trial (2025 miniaturized
4D fUS device), and read+write hardware exists (dual-mode imaging/neuromodulation probes,
2025). Competing on more voxels, depth, or accuracy is a hardware-and-animals race we cannot
win. On-hand data is non-somatosensory: a macaque motor set (Griggs, CaltechDATA
`pa710-cdn95`, CC-BY-NC) and a mouse single-condition visual set with the Allen atlas
(NERF / Urban lab). Somatosensory data is not yet in hand (pending request — S03).

**Decision.** Position the project on generalizable fUS decoding as the near-term,
provable contribution, with sensory write-in targeting named as the longer-term north star.
The headline is a single falsifiable hypothesis: a fUS decoder trained on one subject can
decode a second subject above chance without per-session recalibration (macaque A→B; mouse
as out-of-distribution probe). Within-session stability and cross-session transfer are
reported as characterization metrics, not novelty claims.

**Why.** (1) It is provable now on open data we already hold, with no device and no
somatosensory data dependency. (2) It is distinct from prior art: Griggs's transfer result is
within-subject/cross-session; cross-*subject* transfer is, to our knowledge, unclaimed. (3) It
does not cede the mission — the subject-invariant representation is the substrate the sensory
targeting framework needs. (4) The claim is falsifiable (rejected if transfer drops to
chance), satisfying the S01 done-bar: it deletes work (no hardware/voxel race), it survives
the "already done" dismissal, and it states an explicit failure condition.

**Consequences.** Near-term work targets cross-subject transfer on the Griggs set; the mouse
set is repurposed as an OOD probe and for ingestion/registration hardening (Track 1). The
sensory claim (B.3) remains gated on somatosensory data still to be requested (S03). Document
discipline begins here: ROADMAP.md holds the living thesis; this log holds the dated rationale.

**References.**
- Griggs et al. (2023), *Decoding Motor Plans Using a Closed-Loop Ultrasonic Brain-Machine
  Interface.* CaltechDATA 10.22002/pa710-cdn95; Nat. Neurosci. 10.1038/s41593-023-01500-7.
- Miniaturized 4D functional ultrasound for mapping human brain activity (2025 preprint).
  medRxiv 2025.08.19.25332261.
- Wearable dual-mode probe for image-guided closed-loop ultrasound neuromodulation (2025).
  PMC12416340.
