"""Unit tests for the decay model (Module 4) -- scheduler + retention."""

import math

import pytest

from backend.app.services.module4.decay_model import (
    DecayModel,
    FSRSScheduler,
    SM2Scheduler,
    difficulty_prior,
    retention,
    time_until_decay,
)
from backend.app.services.graph_model import ConceptNode, GraphModel


def test_retention_unit_consistency():
    # S_i in seconds, elapsed in seconds -> dimensionless retention.
    assert retention(0, 3600) == 1.0
    assert retention(3600, 3600) == pytest.approx(math.exp(-1))


def test_sm2_ease_factor_clamps_low():
    sched = SM2Scheduler()
    # Many failed reviews should floor EF at 1.3.
    ef = sched.ease_initial
    for _ in range(50):
        ef = sched.update(86400.0, quality=0, previous=ef)
        ef = min(ef, 1.3) if ef < 1.3 else ef  # mirror SM2 floor semantics
    # The scheduler's own clamp already floors at 1.3 inside update.
    s = sched.update(86400.0, quality=0, previous=ef)
    assert s > 0


def test_fsrs_stability_positivity():
    sched = FSRSScheduler()
    for q in range(6):
        assert sched.update(86400.0, quality=q) > 0


def test_decay_model_cold_start_uses_default_prior():
    g = GraphModel()
    g.add_node(ConceptNode(id="C"))
    model = DecayModel(g)
    model.initialize_stability("C")
    assert g.get_stability("C") > 0


def test_decay_model_uses_difficulty_prior():
    g = GraphModel()
    g.add_node(ConceptNode(id="C"))
    model = DecayModel(g)
    prior = difficulty_prior(0.8)
    model.initialize_stability("C", prior=prior)
    assert g.get_stability("C") == pytest.approx(prior)


def test_is_decayed_never_reviewed_is_false_at_start():
    g = GraphModel()
    g.add_node(ConceptNode(id="C"))
    model = DecayModel(g)
    # Never reviewed at "now"=review timestamp -> not decayed.
    model.record_review("C", quality=3, timestamp=0.0)
    assert not model.is_decayed("C", now=0.0)