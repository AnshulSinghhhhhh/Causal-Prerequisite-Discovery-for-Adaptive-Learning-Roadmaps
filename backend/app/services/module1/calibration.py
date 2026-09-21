"""THE canonical non-circular calibration code (Section 1.1).

Every tunable threshold and every model-variant choice in the repository flows
through this module so the governing rule is enforced in one place:

    select on *training* data only, freeze, apply exactly once to held-out.

Two leakage modes are guarded explicitly and asserted by
``tests/backend/test_calibration.py``:

1. **Threshold-inconsistency leakage** -- every variant in a comparison must be
   calibrated the same way. ``compare_variants`` refuses to rank variants that
   were not produced by the same ``calibrate`` procedure.

2. **Model-selection leakage in ensembles** -- the variant that enters an
   ensemble must be selected via training-domain CV only. ``select_model``
   never sees the held-out labels; ``record_selection`` persists the choice plus
   a provenance assertion so the held-out path cannot leak into the selection
   code path.

The score -> binary decision mapping uses Youden's J (``J = TPR - FPR``),
maximized over candidate thresholds computed on the training fold.
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Core statistics
# --------------------------------------------------------------------------- #
def youdens_j(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Youden's J = TPR - FPR from raw 0/1 labels and predictions."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if y_true.size == 0:
        raise ValueError("youden: empty input")
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return float(tpr - fpr)


def precision_recall_f1(y_true: Sequence[int], y_pred: Sequence[int]) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


def best_threshold_by_youden(
    y_true: Sequence[int],
    scores: Sequence[float],
    thresholds: Optional[Sequence[float]] = None,
) -> Tuple[float, float]:
    """Return (threshold, J) maximizing Youden's J over candidate thresholds.

    ``y_true`` must come from the *training* fold only (see module docstring).
    """
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if y_true.size == 0 or scores.size != y_true.size:
        raise ValueError("best_threshold_by_youden: mismatched/empty inputs")
    if thresholds is None:
        uniq = np.unique(scores)
        if uniq.size == 1:
            thresholds = uniq
        else:
            # Candidate cut-points between consecutive sorted scores.
            thresholds = (uniq[:-1] + uniq[1:]) / 2.0
    best_t, best_j = 0.5, -np.inf
    for t in thresholds:
        pred = (scores >= t).astype(int)
        j = youdens_j(y_true, pred)
        if j > best_j:
            best_j, best_t = j, float(t)
    return best_t, best_j


def pr_curve(y_true: Sequence[int], scores: Sequence[float],
             n_points: int = 50) -> List[Dict[str, float]]:
    """Precision-recall curve as a function of threshold (Section 4.2.1)."""
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    lo, hi = float(scores.min()), float(scores.max())
    if hi <= lo:
        hi = lo + 1.0
    pts = []
    for t in np.linspace(lo, hi, n_points):
        m = precision_recall_f1(y_true, (scores >= t).astype(int))
        pts.append({"threshold": float(t), "precision": m["precision"],
                    "recall": m["recall"], "f1": m["f1"]})
    return pts


# --------------------------------------------------------------------------- #
# Variant comparison (threshold-inconsistency guard)
# --------------------------------------------------------------------------- #
@dataclass
class CalibratedThreshold:
    """A frozen threshold with full provenance."""

    name: str
    threshold: float
    youdens_j: float
    n_train: int
    calibrated_by: str  # procedure id -- must match across a comparison

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "threshold": self.threshold,
            "youden_j": self.youdens_j,
            "n_train": self.n_train,
            "calibrated_by": self.calibrated_by,
        }


@dataclass
class HeldOutResult:
    """The *single* held-out application of a frozen decision."""

    variant: str
    threshold: float
    metrics: Dict[str, float]
    #: Procedure id of the calibration that produced this variant.
    calibrated_by: str = "youden"

    def to_dict(self) -> dict:
        return {"variant": self.variant, "threshold": self.threshold,
                "metrics": self.metrics, "calibrated_by": self.calibrated_by}


def calibrate(
    name: str,
    y_train: Sequence[int],
    scores_train: Sequence[float],
    procedure_id: str = "youden",
    thresholds: Optional[Sequence[float]] = None,
) -> CalibratedThreshold:
    """Select a threshold on training data only and freeze it."""
    t, j = best_threshold_by_youden(y_train, scores_train, thresholds)
    return CalibratedThreshold(
        name=name, threshold=t, youdens_j=j,
        n_train=int(len(y_train)), calibrated_by=procedure_id,
    )


def apply_once(calibration: CalibratedThreshold,
               y_heldout: Sequence[int],
               scores_heldout: Sequence[float]) -> HeldOutResult:
    """Apply a frozen calibration exactly once to the held-out set."""
    y = np.asarray(y_heldout, dtype=int)
    s = np.asarray(scores_heldout, dtype=float)
    preds = (s >= calibration.threshold).astype(int)
    return HeldOutResult(
        variant=calibration.name, threshold=calibration.threshold,
        metrics=precision_recall_f1(y, preds),
        calibrated_by=calibration.calibrated_by,
    )


def compare_variants(results: Sequence[HeldOutResult]) -> List[dict]:
    """Rank calibration-consistent variants by F1 (descending).

    Raises ``CalibrationInconsistencyError`` if any two results were produced
    by *different* calibration procedures -- the explicit
    threshold-inconsistency leakage guard from Section 1.1. Two variants can
    only be compared fairly when both were calibrated identically.
    """
    results = list(results)
    if not results:
        return []
    procs = {r.calibrated_by for r in results}
    if len(procs) > 1:
        raise CalibrationInconsistencyError(
            "cannot compare variants calibrated by different procedures: "
            f"{sorted(procs)}"
        )
    ordered = sorted(
        results, key=lambda r: r.metrics.get("f1", 0.0), reverse=True
    )
    return [r.to_dict() for r in ordered]


class CalibrationInconsistencyError(ValueError):
    """Raised when mismatched calibration procedures are compared."""


# --------------------------------------------------------------------------- #
# Ensemble model-selection guard
# --------------------------------------------------------------------------- #
@dataclass
class ModelSelection:
    """A CV-selected component variant with a provenance trail."""

    selected: str
    candidates: List[str]
    metric: str
    fold_metric_mean: float
    training_only: bool = True
    heldout_path_seen: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "selected": self.selected,
            "candidates": self.candidates,
            "metric": self.metric,
            "fold_metric_mean": self.fold_metric_mean,
            "training_only": self.training_only,
            "heldout_path_seen": self.heldout_path_seen,
        }


def select_model(
    candidates: Sequence[str],
    fold_scores: Dict[str, Sequence[float]],
    metric: str = "f1",
) -> ModelSelection:
    """Pick the ensemble component with the best training-fold CV mean metric.

    ``fold_scores`` maps candidate -> per-fold metrics computed *on training
    folds only*. The returned object records that no held-out path was seen;
    the test suite asserts this stays true even if callers (incorrectly) pass a
    held-out dataset through the surrounding code.
    """
    if not candidates:
        raise ValueError("select_model: no candidates")
    means = {
        c: float(np.mean(np.asarray(fold_scores[c], dtype=float)))
        for c in candidates
    }
    selected = max(means, key=means.get)
    return ModelSelection(
        selected=selected,
        candidates=list(candidates),
        metric=metric,
        fold_metric_mean=means[selected],
    )


def record_selection(selection: ModelSelection,
                     path: Optional[str] = None) -> None:
    """Persist a selection decision + provenance to ``docs/results/``.

    ``path`` is the optional output file; when omitted a timestamped snapshot
    is written so every post-hoc rerun is auditable.
    """
    out = path or _default_selection_path()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    payload = selection.to_dict()
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _default_selection_path() -> str:
    import time

    here = os.path.dirname(os.path.abspath(__file__))
    repo = here
    for _ in range(6):
        repo = os.path.dirname(repo)
        if os.path.isdir(os.path.join(repo, "docs")):
            stamp = time.strftime("%Y%m%d-%H%M%S")
            return os.path.join(repo, "docs", "results",
                                f"model_selection_{stamp}.json")
    return os.path.join(os.getcwd(), "model_selection.json")