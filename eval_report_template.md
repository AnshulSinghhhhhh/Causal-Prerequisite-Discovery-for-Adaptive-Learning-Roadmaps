# LightGAP evaluation run — {{run_name}}

- **Date:** {{YYYY-MM-DD}}
- **Git SHA:** {{git_sha}}
- **Run type:** {{calibration | model_selection | final_eval}}

## Model version

- Name: `{{model_versions.name}}`
- Kind: `{{model_versions.kind}}`
- Trained on: `{{model_versions.trained_on}}`
- Frozen hyperparameters: {{hyperparams as a short list, e.g. "LoRA rank 16, lr 2e-4, 3 epochs"}}
- Frozen threshold (if applicable): {{frozen_threshold}}
- Artifact: {{artifact_uri, or "n/a — no persisted weights for this run"}}

## Evaluation set

- Name: `{{eval_sets.name}}`
- Purpose: **{{calibration | heldout}}**
- Pairs: {{n_pairs}}
- SHA256: `{{sha256}}`

## Metrics

| Metric | Value |
|---|---|
| Accuracy | {{}} |
| Precision | {{}} |
| Recall | {{}} |
| F1 | {{}} |
| ROC-AUC | {{}} |

**Confusion matrix:**

|  | Predicted negative | Predicted positive |
|---|---|---|
| **Actual negative** | {{tn}} | {{fp}} |
| **Actual positive** | {{fn}} | {{tp}} |

**Subclass breakdown:**

| Subclass | Count | Correct | Subset accuracy |
|---|---|---|---|
| Prerequisite | {{}} | {{}} | {{}} |
| Reversal | {{}} | {{}} | {{}} |
| Unrelated | {{}} | {{}} | {{}} |

## Interpretation

{{One paragraph. If this is compared against a prior run, state whether it's better or worse and give the causal reason — not just the direction of the change.}}

## For the `heldout` set only — one-shot confirmation

{{Confirm explicitly: this is the first and only evaluation_runs row for this model_version_id against this eval_set_id. The database's unique index on (model_version_id, eval_set_id) where run_type='final_eval' enforces this, but state it here too since this is the number that gets cited.}}
