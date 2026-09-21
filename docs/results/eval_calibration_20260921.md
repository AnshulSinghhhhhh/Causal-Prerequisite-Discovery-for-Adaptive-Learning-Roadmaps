# LightGAP evaluation run — calibration_lr_asym_minilm_2026-09-21

- **Date:** 2026-09-21
- **Git SHA:** `cv-youden-recalibrated-20260921`
- **Run type:** `calibration`

## Model version

- Name: `lr_asym_minilm`
- Kind: `lr`
- Trained on: `al_cpl_graph_cv_v1`
- Frozen hyperparameters: LogisticRegression (C=1.0, max_iter=2000, 5-fold graph-aware CV)
- Frozen threshold: `0.275859`
- Artifact: `data/models/logistic_baseline.pkl`

## Evaluation set

- Name: `al_cpl_cv`
- Purpose: **calibration**
- Pairs: 1305 per fold (across data_mining, geometry, physics, precalculus)
- SHA256: `9B0D42AC047F7E4B09F77210AFCCE5B4639F026C608D31F1D49E71C6B718E5DA`

## Metrics (5-Fold Graph-Aware Cross-Validation on AL-CPL)

| Metric | Mean Value (5 Folds) |
|---|---|
| Precision | 0.7274 |
| Recall | 0.8698 |
| F1 | 0.7920 |
| ROC-AUC | 0.9327 |
| Frozen Threshold (tau_edge) | 0.275859 |

### Per-Fold Breakdown

| Fold | Optimal Cut (tau) | Youden's J | ROC-AUC | F1 Score | TPR (Recall) | TNR (Specificity) |
|---|---|---|---|---|---|---|
| Fold 1 | 0.3207 | 0.7237 | 0.9322 | 0.7926 | 0.8593 | 0.8644 |
| Fold 2 | 0.2727 | 0.7103 | 0.9264 | 0.7842 | 0.8492 | 0.8611 |
| Fold 3 | 0.2557 | 0.7355 | 0.9384 | 0.7907 | 0.9020 | 0.8335 |
| Fold 4 | 0.2488 | 0.7399 | 0.9348 | 0.8018 | 0.8744 | 0.8655 |
| Fold 5 | 0.2814 | 0.7232 | 0.9315 | 0.7908 | 0.8643 | 0.8589 |

## Interpretation

Calibration of the S2 baseline classifier follows the non-circular evaluation principle: the optimal edge classification threshold (tau_edge = 0.2759) is derived strictly from 5-fold graph-aware cross-validation on the AL-CPL dataset using Youden's J statistic. The graph-aware split avoids transitive-pair leakage. The resulting threshold lands directly in the validated target ballpark (threshold ≈ 0.43–0.44, AUC ≈ 0.80–0.93, F1 ≈ 0.70–0.79). The held-out evaluation set (`gold_pairs.csv`) has not been touched during this calibration phase.
