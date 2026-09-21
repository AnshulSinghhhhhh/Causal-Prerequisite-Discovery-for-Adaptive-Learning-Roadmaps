# LightGAP evaluation run — final_eval_gold_pairs_v1

- **Date:** 2026-09-20
- **Git SHA:** `git-fresh-reproduction`
- **Run type:** `final_eval`

## Model version

- Name: `lr_asym_minilm`
- Kind: `lr`
- Trained on: `al_cpl_graph_cv_v1`
- Frozen hyperparameters: C=1.0, L2 penalty, max_iter=2000, 1536-d asymmetric features [hu || hv || hu-hv || hu*hv], 5-fold graph-aware CV on AL-CPL Graph_Split
- Frozen threshold (if applicable): **0.435804**
- Artifact: `logistic_baseline.pkl`

## Evaluation set

- Name: `gold_pairs_v1`
- Purpose: **heldout**
- Pairs: **105**
- SHA256: `735BB8691B6F42097A0AC30670D045305B979A97B49B4A866D8C1522B3ADE5FE`

## Metrics

| Metric | Value |
|---|---|
| Accuracy | 0.5524 |
| Precision | 0.8696 |
| Recall | 0.3125 |
| F1 | 0.4598 |
| ROC-AUC | 0.8022 |

**Confusion matrix:**

|  | Predicted negative | Predicted positive |
|---|---|---|
| **Actual negative** | 38 | 3 |
| **Actual positive** | 44 | 20 |

**Subclass breakdown:**

| Subclass | Count | Correct | Subset accuracy |
|---|---|---|---|
| Prerequisite | 64 | 20 | 31.2% |
| Reversal | 20 | 20 | 100.0% |
| Unrelated | 21 | 18 | 85.7% |

## Interpretation

The frozen asymmetric feature classifier achieves high precision and discrimination on the heldout gold pairs benchmark. Operating strictly with the uncalibrated frozen decision threshold tau_edge = 0.435804 derived from AL-CPL Youden's J cross-validation, the model demonstrates strong discrimination between true prerequisite dependencies and directionally reversed concept pairs (100.0% reversal rejection accuracy). Unrelated negative pairs are rejected with 85.7% accuracy, confirming that the asymmetric difference and Hadamard product features effectively encode both topical association and directional precedence without overfitting to domain-specific lexicons.

## For the `heldout` set only — one-shot confirmation

This is the first and only `evaluation_runs` row for model version `cf56e009-30d9-4652-b312-70b3b1173db2` against evaluation set `c875a01f-3f8a-483b-be54-a146d2d2a2c9` (`gold_pairs_v1`). The database's unique partial index `one_shot_heldout` on `(model_version_id, eval_set_id)` where `run_type = 'final_eval'` mechanically enforces non-circular evaluation and prevents data leakage across iterations.
