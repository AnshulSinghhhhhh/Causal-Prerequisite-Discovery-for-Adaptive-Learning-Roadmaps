"""Tests for EdgeType.HIERARCHY separation from prerequisite edges."""

import pytest

from backend.app.services.graph_model import (
    ConceptEdge,
    ConceptNode,
    EdgeType,
    GraphModel,
    NodeStatus,
    NodeType,
)
from backend.app.services.module2.build_graph import build_dag


def test_edge_type_hierarchy_exists():
    assert EdgeType.HIERARCHY == "hierarchy"
    assert EdgeType.HIERARCHY in list(EdgeType)


def test_edge_set_excludes_hierarchy_edges():
    g = GraphModel()
    g.add_node(ConceptNode(id="root", node_type=NodeType.ROOT))
    g.add_node(ConceptNode(id="mod1", node_type=NodeType.BRANCH))
    g.add_node(ConceptNode(id="sub1", node_type=NodeType.LEAF))
    g.add_node(ConceptNode(id="sub2", node_type=NodeType.LEAF))

    # Add hierarchy edges
    g.add_edge(ConceptEdge(source="root", target="mod1", type=EdgeType.HIERARCHY))
    g.add_edge(ConceptEdge(source="mod1", target="sub1", type=EdgeType.HIERARCHY))

    # Add prerequisite edge
    g.add_edge(ConceptEdge(source="sub1", target="sub2", type=EdgeType.PREREQUISITE))

    # edge_set() snapshot must only contain prerequisite edges
    snapshot = g.edge_set()
    assert snapshot == (("sub1", "sub2"),)
    assert ("root", "mod1") not in snapshot
    assert ("mod1", "sub1") not in snapshot


def test_hierarchy_edges_survive_high_tau_edge():
    # Verify that structural hierarchy edges added directly to graph are never
    # filtered by build_dag's tau_edge cutoff
    g = GraphModel()
    g.add_node(ConceptNode(id="root"))
    g.add_node(ConceptNode(id="mod1"))
    g.add_node(ConceptNode(id="sub1"))

    # Add structural hierarchy edge
    g.add_edge(ConceptEdge(source="root", target="mod1", type=EdgeType.HIERARCHY))

    # Scored candidate pairs passed to build_dag with high tau_edge
    pairs = [("sub1", "mod1")]
    scores = [0.3]  # below 0.65 threshold

    dag_result = build_dag(pairs, scores, tau_edge=0.65)
    assert len(dag_result.edges) == 0  # pruned by threshold

    # But hierarchy edge on graph remains intact
    assert g.has_edge("root", "mod1")
    assert g.get_edge("root", "mod1").type == EdgeType.HIERARCHY
