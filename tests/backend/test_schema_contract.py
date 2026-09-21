"""Schema-contract tests (Section 10 / build order step 11).

Every endpoint response must validate against ``models/schema.py``. These tests
enforce that the contract and its GraphModel bridge round-trip cleanly, so a
schema drift between backend and frontend surfaces here rather than at
integration time.
"""

import pytest
from pydantic import ValidationError

from backend.app.models.schema import (
    GraphPayload,
    NodePayload,
    EdgePayload,
    StudentState,
    node_to_payload,
    edge_to_payload,
    graph_to_payload,
    payload_to_graph,
)
from backend.app.services.graph_model import (
    ConceptEdge,
    ConceptNode,
    EdgeType,
    GraphModel,
    NodeStatus,
)


def test_graph_payload_accepts_minimal_valid():
    g = GraphPayload(student_state={"goal_concept": "x",
                                    "time_budget_minutes": 30})
    assert g.student_state.goal_concept == "x"
    assert g.nodes == []
    assert g.edges == []


def test_graph_payload_rejects_invalid_status():
    with pytest.raises(ValidationError):
        GraphPayload(
            student_state={"goal_concept": "x", "time_budget_minutes": 30},
            nodes=[{"id": "A", "status": "bogus"}],
        )


def test_graph_payload_rejects_invalid_edge_type():
    with pytest.raises(ValidationError):
        GraphPayload(
            student_state={"goal_concept": "x", "time_budget_minutes": 30},
            edges=[{"source": "A", "target": "B", "type": "bogus"}],
        )


def test_confidence_bounds_enforced():
    with pytest.raises(ValidationError):
        EdgePayload(source="A", target="B", confidence=1.5)


def test_graph_to_payload_and_back_roundtrip():
    g = GraphModel()
    g.add_node(ConceptNode(id="A", status=NodeStatus.COMPLETED))
    g.add_node(ConceptNode(id="RM_1", is_dynamic_remediation=True,
                           triggering_misconception="M_1",
                           status=NodeStatus.LOCKED))
    g.add_edge(ConceptEdge(source="A", target="RM_1",
                           type=EdgeType.REMEDIATION_LINK, confidence=1.0))

    payload = graph_to_payload(g, goal_concept="D", time_budget_minutes=45)
    # Validate the payload against the schema (pydantic already did on build).
    assert isinstance(payload, GraphPayload)

    back = payload_to_graph(payload)
    assert set(back.node_ids()) == {"A", "RM_1"}
    assert back.get_node("RM_1").is_dynamic_remediation
    assert back.get_node("RM_1").triggering_misconception == "M_1"
    assert back.get_edge("A", "RM_1").type == EdgeType.REMEDIATION_LINK


def test_node_and_edge_payload_bridge():
    n = node_to_payload(ConceptNode(id="A", status=NodeStatus.LOCKED))
    assert n.status == "locked"
    e = edge_to_payload(ConceptEdge(source="A", target="B",
                                    confidence=0.7))
    assert e.type == "prerequisite"
    assert e.confidence == 0.7


def test_student_state_budget_nonnegative():
    with pytest.raises(ValidationError):
        StudentState(goal_concept="x", time_budget_minutes=-1)