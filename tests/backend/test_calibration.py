"""Calibration-leakage tests (Section 1.1) -- the day-one guards.

These assert the non-circular calibration rule is structurally enforced, not
just conventionally followed:

1. **Threshold-inconsistency leakage** -- variants calibrated by different
   procedures cannot be compared (``compare_variants`` raises).
2. **Model-selection leakage in ensembles** -- the variant-selection code path
   never receives the held-out file; only training-fold metric sequences are
   consumed, and ``select_model`` records ``training_only=True``.
3. **Youden's J** math is correct on a hand-checkable case.
"""

import numpy as np
import pytest

from backend.app.services.module1.calibration import (
    CalibrationInconsistencyError,
    calibrate,
    apply_once,
    best_threshold_by_youden,
    compare_variants,
    precision_recall_f1,
    select_model,
    youdens_j,
)


def test_youdens_j_hand_checkable():
    # Perfect separation: TPR=1, FPR=0 => J=1.
    assert youdens_j([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0
    # All-positive guesses on balanced labels: TPR=1, FPR=1 => J=0.
    assert youdens_j([1, 0], [1, 1]) == 0.0
    # All-negative: TPR=0, FPR=0 => J=0.
    assert youdens_j([1, 0], [0, 0]) == 0.0


def test_best_threshold_youden_finds_separating_cut():
    # Scores: positives high (0.9), negatives low (0.1). Best threshold separates.
    y = [1, 1, 0, 0]
    s = [0.9, 0.8, 0.2, 0.1]
    t, j = best_threshold_by_youden(y, s)
    assert j == 1.0
    # Any t in (0.2, 0.8) achieves J=1; the function returns a valid cut.
    assert 0.2 < t < 0.8


def test_precision_recall_f1_basic():
    m = precision_recall_f1([1, 0, 1], [1, 1, 0])
    # tp=1, fp=1, fn=1 => precision=0.5, recall=0.5, f1=0.5
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["f1"] == pytest.approx(0.5)


def test_apply_once_uses_frozen_threshold_exactly():
    cal = calibrate("lr", [0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    # Held-out applied once with the SAME threshold.
    held = apply_once(cal, [1, 0], [0.85, 0.15])
    assert held.threshold == cal.threshold
    # 0.85 >= t -> pred 1 (correct for label 1); 0.15 < t -> pred 0 (correct).
    assert held.metrics["tp"] == 1
    assert held.metrics["fp"] == 0


def test_compare_variants_rejects_inconsistent_calibration():
    # Two variants calibrated by DIFFERENT procedures must not be compared.
    a = calibrate("emb-only", [0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9],
                  procedure_id="youden")
    b = calibrate("emb-only", [0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9],
                  procedure_id="fixed-0.7")
    ra = apply_once(a, [1, 0], [0.9, 0.1])
    rb = apply_once(b, [1, 0], [0.9, 0.1])
    # Manually force differing procedures onto the results to exercise the guard.
    ra.calibrated_by = "youden"
    rb.calibrated_by = "fixed-0.7"
    with pytest.raises(CalibrationInconsistencyError):
        compare_variants([ra, rb])


def test_compare_variants_accepts_consistent_calibration():
    a = calibrate("lr", [0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    b = calibrate("dirgcn", [0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    ra = apply_once(a, [1, 0], [0.9, 0.1])
    rb = apply_once(b, [1, 0], [0.9, 0.1])
    # Same procedure id (default) => comparison allowed, sorted by F1.
    order = compare_variants([rb, ra])
    assert len(order) == 2


def test_select_model_never_sees_heldout(tmp_path):
    """The ensemble component-selection path consumes training-fold metrics only.

    We simulate a caller that (incorrectly) ALSO has a held-out file on disk;
    ``select_model`` takes only fold-score dicts (numbers), never a path, so the
    held-out file cannot enter the selection decision.
    """
    heldout_file = tmp_path / "heldout.csv"
    heldout_file.write_text("u,v,label\nA,B,1\n", encoding="utf-8")

    fold_scores = {
        "dirgcn_standard": [0.71, 0.72, 0.70],
        "dirgcn_attention": [0.74, 0.75, 0.73],
    }
    selection = select_model(list(fold_scores), fold_scores, metric="f1")

    assert selection.selected == "dirgcn_attention"
    assert selection.training_only is True
    # The selection result references no path at all -- by construction.
    assert selection.heldout_path_seen == []
    # And the held-out file's existence changes nothing about the outcome.
    assert selection.fold_metric_mean == pytest.approx(
        np.mean([0.74, 0.75, 0.73])
    )


def test_select_model_requires_candidates():
    with pytest.raises(ValueError):
        select_model([], {}, metric="f1")


def test_calibrate_freezes_youden_threshold():
    # Training data determines the threshold; it must be reproducible.
    cal = calibrate("lr", [0, 0, 0, 1, 1, 1], [0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert cal.n_train == 6
    assert cal.calibrated_by == "youden"
    assert 0.3 < cal.threshold < 0.7