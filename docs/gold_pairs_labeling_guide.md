# Gold Pairs Labeling Guide

The gold set (`data/labeled/gold_pairs.csv`) is the **held-out target-domain
evaluation set**. It must never influence any training, threshold, or
model-variant decision (Section 1.1 non-circular calibration rule).

## CSV format

```
source,target,label
global_health,epidemiology,1
epidemiology,global_health,0
```

- `source`, `target` — concept identifiers (must match concept text/embeddings
  used downstream).
- `label` — `1` if `source` **must** precede `target` as a prerequisite, else
  `0`. Do **not** emit the reversed pair as a separate row unless you are
  explicitly testing directionality (the engine scores directed pairs, so a
  single `(u, v, 1)` already implies `(v, u, 0)` for the reverse).

## Labeling rules

1. **Label the directed prerequisite, not topical relatedness.** Two concepts
   can be topically close (both "matrix" topics) without a prerequisite edge.
   Ask: "Can a learner plausibly master `target` without `source`?"
2. **Only direct prerequisites.** If `source -> x -> target` already implies
   the relationship, `source -> target` may be labeled `1` only when it is a
   *genuinely direct* prereq; otherwise mark `0` (transitive subsumption is a
   non-issue for the DAG, see the collateral-damage audit).
3. **Be conservative.** When unsure, label `0`. A false positive in the gold
   set silently corrupts precision/recall on the held-out evaluation.
4. **Cover both directions for disputed pairs.** If directionality is the
   specific question, add both rows so the asymmetric feature is tested.

## Process

- Two independent labelers annotate the same pairs.
- Third labeler (or curator) adjudicates disagreements.
- Record an inter-annotator agreement note in `docs/results/` before freezing.

## Anti-leakage guarantees

- `data/labeled/gold_pairs.csv` is the only path the evaluation harness reads
  for the held-out set.
- `calibration.apply_once` is called exactly once; `select_model` never takes a
  file path, so the gold set cannot enter variant selection.
- Add a unit test (see `tests/backend/test_calibration.py`) if you add a new
  selection procedure.