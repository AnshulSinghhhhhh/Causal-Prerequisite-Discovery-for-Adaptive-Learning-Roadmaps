# LightGAP Results

Dated JSON/markdown snapshots of every evaluation run (Section 13 / Section 11
three-track protocol). Everything the pipeline writes — calibrated thresholds,
held-out applications, model-selection decisions, Track A/B metrics — lands
here so every operating point is auditable and reproducible.

## Conventions

- Filenames embed the stage and a UTC timestamp, e.g.
  `stage1_20250101-120000.json`, `calibration_20250101-120000.json`.
- Every file records its **calibration provenance** (`calibrated_by`) and, for
  model selection, that `training_only` was true.
- **Negative results are deliverables** (Section 1.2): when a variant does not
  beat baseline, record the number, the proposed mechanism, and move on. Do not
  silently omit a failed comparison.

## Expected artifacts

| File | Producer | Content |
|---|---|---|
| `embeddings_*.npz` | `run_full_pipeline.py --stage embed` | cached concept vectors |
| `stage1_*.json` | `--stage stage1` | Youden calibration + single held-out application |
| `model_selection_*.json` | `calibration.record_selection` | CV-selected ensemble component |
| `trackA_*.json` | Track A eval | per-stage precision/recall/F1 + PR curve over τ_edge |
| `trackB_*.json` | Track B eval | budget-satisfaction, steps-to-mastery, remediation convergence |
| `trackC_*.md` | Track C user study | completion time, post-test delta, SUS |

## The one rule that matters

Train-only CV decides thresholds and variant choices; the held-out / gold set is
applied **exactly once**. Any future evaluation code must go through
`backend/app/services/module1/calibration.py`, never re-derive thresholds
inline.