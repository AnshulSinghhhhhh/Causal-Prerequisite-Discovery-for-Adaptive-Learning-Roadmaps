"""Canonical graph data model unit tests (Section 1.4 -- one implementation)."""

import pytest

from backend.app.services.graph_model import (
    ConceptEdge,
    ConceptNode,
    EdgeType,
    GraphModel,
    NodeStatus,
)


def test_add_and_get_node():
    g = GraphModel()
    g.add_node(ConceptNode(id="A", label="Concept A"))
    assert g.has_node("A")
    assert g.get_node("A").label == "Concept A"


def test_add_duplicate_node_raises():
    g = GraphModel()
    g.add_node(ConceptNode(id="A"))
    with pytest.raises(ValueError):
        g.add_node(ConceptNode(id="A"))


def test_edge_requires_existing_nodes():
    g = GraphModel()
    g.add_node(ConceptNode(id="A"))
    with pytest.raises(ValueError):
        g.add_edge(ConceptEdge(source="A", target="B"))


def test_add_duplicate_edge_raises():
    g = GraphModel()
    g.add_node(ConceptNode(id="A"))
    g.add_node(ConceptNode(id="B"))
    g.add_edge(ConceptEdge(source="A", target="B"))
    with pytest.raises(ValueError):
        g.add_edge(ConceptEdge(source="A", target="B"))


def test_neighbors():
    g = GraphModel()
    for n in ["A", "B", "C"]:
        g.add_node(ConceptNode(id=n))
    g.add_edge(ConceptEdge(source="A", target="B"))
    g.add_edge(ConceptEdge(source="A", target="C"))
    assert set(g.out_neighbors("A")) == {"B", "C"}
    assert g.in_neighbors("B") == ["A"]


def test_remove_node_removes_incident_edges():
    g = GraphModel()
    for n in ["A", "B", "C"]:
        g.add_node(ConceptNode(id=n))
    g.add_edge(ConceptEdge(source="A", target="B"))
    g.add_edge(ConceptEdge(source="B", target="C"))
    g.remove_node("B")
    assert not g.has_node("B")
    assert not g.has_edge("A", "B")
    assert not g.has_edge("B", "C")
    assert g.has_node("A") and g.has_node("C")


def test_review_and_stability_state():
    g = GraphModel()
    g.add_node(ConceptNode(id="A"))
    g.set_last_review("A", 123.0)
    g.set_stability("A", 86400.0)
    assert g.get_last_review("A") == 123.0
    assert g.get_stability("A") == 86400.0


def test_stability_must_be_positive():
    g = GraphModel()
    g.add_node(ConceptNode(id="A"))
    with pytest.raises(ValueError):
        g.set_stability("A", 0.0)


def test_edge_set_snapshot():
    g = GraphModel()
    for n in ["A", "B", "C"]:
        g.add_node(ConceptNode(id=n))
    g.add_edge(ConceptEdge(source="A", target="B", type=EdgeType.PREREQUISITE))
    g.add_edge(ConceptEdge(source="B", target="C",
                           type=EdgeType.REMEDIATION_LINK))
    snapshot = g.edge_set()
    # Only prerequisite edges are in the immutable snapshot.
    assert snapshot == (("A", "B"),)


def test_clear_remediation_state():
    g = GraphModel()
    for n in ["A", "B"]:
        g.add_node(ConceptNode(id=n))
    g.add_edge(ConceptEdge(source="A", target="B", type=EdgeType.PREREQUISITE))
    g.add_edge(ConceptEdge(source="B", target="A",
                           type=EdgeType.REMEDIATION_LINK))
    g.clear_remediation_state()
    assert g.has_edge("A", "B")
    assert not g.has_edge("B", "A")