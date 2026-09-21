"""Module 4 correctness tests -- decay curve crossing + remediation rewrite."""

import math

import pytest

from backend.app.services.module4.decay_model import (
    retention,
    time_until_decay,
    SM2Scheduler,
    FSRSScheduler,
    DecayModel,
    difficulty_prior,
)
from backend.app.services.module4.graph_rewriter import (
    GraphRewriter,
    remediation_node_id,
)
from backend.app.services.graph_model import (
    ConceptEdge,
    ConceptNode,
    EdgeType,
    GraphModel,
    NodeStatus,
)


# --------------------------------------------------------------------------- #
# Decay math
# --------------------------------------------------------------------------- #
def test_retention_exponential_decay():
    assert retention(0.0, 10.0) == pytest.approx(1.0)
    assert retention(10.0, 10.0) == pytest.approx(math.exp(-1.0))
    assert retention(100.0, 10.0) < 1e-4


def test_time_until_decay_crossing_point():
    # tau_decay = exp(-dt/S) => dt = S * ln(1/tau).
    tau = 0.5
    stability = 20.0
    dt = time_until_decay(stability, tau_decay=tau)
    # At exactly dt, retention == tau.
    assert retention(dt, stability) == pytest.approx(tau)
    # Just before dt, retention > tau; just after, < tau.
    assert retention(dt - 0.001, stability) > tau
    assert retention(dt + 0.001, stability) < tau


def test_retention_rejects_bad_inputs():
    with pytest.raises(ValueError):
        retention(-1.0, 10.0)
    with pytest.raises(ValueError):
        retention(1.0, 0.0)


def test_time_until_decay_rejects_bad_tau():
    with pytest.raises(ValueError):
        time_until_decay(10.0, tau_decay=1.5)


def test_sm2_update_grows_stability_on_success():
    sched = SM2Scheduler()
    s = sched.update(86400.0, quality=3, repetitions=0, interval_days=1.0)
    assert s > 0


def test_fsrs_update_grows_on_success_shrinks_on_fail():
    sched = FSRSScheduler()
    base = sched.default_stability
    up = sched.update(base, quality=4)
    down = sched.update(base, quality=1)
    assert up > base
    assert down < base


def test_decay_model_flags_crossing():
    from backend.app.services.graph_model import GraphModel

    g = GraphModel()
    g.add_node(ConceptNode(id="X"))
    model = DecayModel(g)
    model.initialize_stability("X", prior=86400.0)  # 1 day stability
    # Review now at t=0.
    t0 = 0.0
    model.record_review("X", quality=3, timestamp=t0)
    # At t0, retention ~1 -> not decayed.
    assert not model.is_decayed("X", now=t0)
    # Far future -> decayed.
    assert model.is_decayed("X", now=t0 + 86400.0 * 5)


def test_difficulty_prior_monotone():
    p_low = difficulty_prior(0.0)
    p_high = difficulty_prior(1.0)
    assert p_high > p_low
    assert 0.5 * 86400 <= p_low <= 3.0 * 86400


# --------------------------------------------------------------------------- #
# Graph rewriting
# --------------------------------------------------------------------------- #
def _graph_a_b_c():
    g = GraphModel()
    for nid in ["A", "B"]:
        g.add_node(ConceptNode(id=nid, label=nid))
    g.add_edge(ConceptEdge(source="A", target="B", confidence=0.8))
    return g


def test_remediation_node_id_canonical():
    assert remediation_node_id("M_7", "B") == "RM_7_B"
    assert remediation_node_id("M7", "B") == "RM_7_B"


def test_inject_remediation_reroutes_and_locks():
    g = _graph_a_b_c()
    g.add_node(ConceptNode(id="C"))
    g.add_edge(ConceptEdge(source="B", target="C", confidence=0.8))
    rewriter = GraphRewriter(g, resolution_successes=1)
    result = rewriter.inject_remediation("C", "M_7")

    assert result.injected_node == "RM_7_C"
    assert g.get_node("C").status == NodeStatus.LOCKED
    assert g.get_node("RM_7_C").is_dynamic_remediation
    assert g.get_node("RM_7_C").triggering_misconception == "M_7"
    # Rerouted: A->B stays, B->RM->C.
    assert g.has_edge("B", "RM_7_C")
    edge = g.get_edge("RM_7_C", "C")
    assert edge.type == EdgeType.REMEDIATION_LINK
    # No direct B->C prerequisite edge remains.
    assert not g.has_edge("B", "C")


def test_resolve_remediation_restores_structure():
    g = _graph_a_b_c()
    g.add_node(ConceptNode(id="C"))
    g.add_edge(ConceptEdge(source="B", target="C", confidence=0.8))
    rewriter = GraphRewriter(g, resolution_successes=1)
    rewriter.inject_remediation("C", "M_7")
    # Successful review resolves.
    assert rewriter.record_remediation_success("RM_7_C") is True
    result = rewriter.resolve_remediation("RM_7_C")

    assert result.resolved
    assert not g.has_node("RM_7_C")
    assert g.get_node("C").status == NodeStatus.IN_PROGRESS
    assert g.has_edge("B", "C")  # original edge restored
    assert g.get_edge("B", "C").type == EdgeType.PREREQUISITE


def test_remediation_requires_successes_before_resolve():
    g = _graph_a_b_c()
    g.add_node(ConceptNode(id="C"))
    g.add_edge(ConceptEdge(source="B", target="C", confidence=0.8))
    rewriter = GraphRewriter(g, resolution_successes=2)
    rewriter.inject_remediation("C", "M_7")
    assert rewriter.record_remediation_success("RM_7_C") is False
    assert g.get_node("C").status == NodeStatus.LOCKED
    assert rewriter.record_remediation_success("RM_7_C") is True


def test_inject_unknown_target_raises():
    g = GraphModel()
    rewriter = GraphRewriter(g)
    with pytest.raises(KeyError):
        rewriter.inject_remediation("NOPE", "M_1")


def test_resolve_unknown_raises():
    g = GraphModel()
    rewriter = GraphRewriter(g)
    with pytest.raises(KeyError):
        rewriter.resolve_remediation("RM_X")