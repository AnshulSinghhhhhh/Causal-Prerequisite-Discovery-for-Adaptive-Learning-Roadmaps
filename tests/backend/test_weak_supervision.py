"""Weak-supervision bootstrap tests (Module 1 fallback path, Section 4.1)."""

import numpy as np
import pytest

from backend.app.services.module1.weak_supervision_bootstrap import (
    labels_from_chapter_order,
    labels_from_link_asymmetry,
    bootstrap_train,
    fine_tune_on_gold,
)


def test_chapter_order_produces_directed_pairs():
    pairs = labels_from_chapter_order([["A", "B", "C"]])
    # A implies B and C; B implies C. Weights decay with distance.
    sources_targets = {(p.source, p.target) for p in pairs}
    assert ("A", "B") in sources_targets
    assert ("A", "C") in sources_targets
    assert ("B", "C") in sources_targets
    # No reverse pairs.
    assert ("C", "A") not in sources_targets
    # Nearer pair weighted higher than farther pair.
    w_ab = [p.weight for p in pairs if (p.source, p.target) == ("A", "B")][0]
    w_ac = [p.weight for p in pairs if (p.source, p.target) == ("A", "C")][0]
    assert w_ab > w_ac


def test_link_asymmetry_drops_invalid_and_self_loops():
    pairs = labels_from_link_asymmetry(
        [("A", "B"), ("B", "B"), ("A", "Z"), ("B", "A")],
        node_set=["A", "B"],
    )
    st = {(p.source, p.target) for p in pairs}
    assert ("A", "B") in st
    assert ("B", "B") not in st  # self-loop dropped
    assert ("A", "Z") not in st  # outside node set dropped
    assert ("B", "A") in st  # valid reverse asymmetry


def test_bootstrap_train_produces_fitted_model():
    rng = np.random.default_rng(0)
    embeddings = {c: rng.normal(size=384).astype(np.float32)
                  for c in ["A", "B", "C", "D"]}
    noisy = labels_from_chapter_order([["A", "B", "C", "D"]])
    model = bootstrap_train(embeddings, noisy, negatives_ratio=1.0, seed=0)
    assert model._model is not None


def test_fine_tune_on_gold_refits():
    rng = np.random.default_rng(0)
    embeddings = {c: rng.normal(size=384).astype(np.float32)
                  for c in ["A", "B", "C", "D"]}
    noisy = labels_from_chapter_order([["A", "B", "C", "D"]])
    model = bootstrap_train(embeddings, noisy, seed=0)
    gold = [("A", "B", 1), ("C", "D", 1), ("A", "C", 0)]
    refit = fine_tune_on_gold(model, embeddings, gold)
    assert refit._model is not None
    # fine_tune_on_gold re-fits in place and returns the same estimator.
    assert refit is model


def test_bootstrap_train_empty_embeddings_raises():
    with pytest.raises(ValueError):
        bootstrap_train({}, [], seed=0)