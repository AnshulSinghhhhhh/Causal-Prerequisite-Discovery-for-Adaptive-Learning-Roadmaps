"""THE canonical JSON data contract (Section 10 / paper Section 6).

One schema, defined once, consumed by every module's endpoints and the
frontend's typed client (``frontend/src/api/client.ts``). Pydantic models here
are the ground truth for the wire format; ``to_dict`` / ``from_dict`` bridge to
the canonical ``GraphModel`` so there is never a second, drifting copy of the
contract.

Example payload (paper Section 6 / brief Section 10):

    {
      "student_state": {"goal_concept": "...", "time_budget_minutes": 0},
      "nodes": [
        {
          "id": "...",
          "status": "completed | in_progress | locked",
          "is_dynamic_remediation": false,
          "triggering_misconception": null
        }
      ],
      "edges": [
        {
          "source": "...", "target": "...",
          "type": "prerequisite | remediation_link",
          "confidence": 0.0
        }
      ]
    }
"""

from __future__ import annotations

from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field

from ..services.graph_model import (
    ConceptEdge,
    ConceptNode,
    EdgeType,
    GraphModel,
    NodeStatus,
    NodeType,
    PrerequisiteType,
)


# --------------------------------------------------------------------------- #
# Wire models
# --------------------------------------------------------------------------- #
NodeStatusLiteral = Literal["completed", "in_progress", "available", "locked"]
NodeTypeLiteral = Literal["root", "branch", "leaf"]
PrerequisiteTypeLiteral = Literal["required", "recommended", "optional"]
EdgeTypeLiteral = Literal["prerequisite", "remediation_link", "hierarchy"]


class StudentState(BaseModel):
    goal_concept: str
    time_budget_minutes: int = Field(ge=0)


class NodePayload(BaseModel):
    id: str
    label: str = ""
    status: NodeStatusLiteral = "in_progress"
    node_type: NodeTypeLiteral = "leaf"
    parent_id: Optional[str] = None
    children: List[str] = Field(default_factory=list)
    depth: int = 0                       # S5 longest-path layer, arbitrary range
    cluster_id: Optional[str] = None     # S5 Leiden community
    progress: float = 0.0
    is_dynamic_remediation: bool = False
    triggering_misconception: Optional[str] = None


class EdgePayload(BaseModel):
    source: str
    target: str
    type: EdgeTypeLiteral = "prerequisite"
    confidence: float = Field(ge=0.0, le=1.0)
    prerequisite_type: PrerequisiteTypeLiteral = "required"


class GraphPayload(BaseModel):
    """The single top-level response object every graph endpoint returns."""

    student_state: StudentState
    nodes: List[NodePayload] = Field(default_factory=list)
    edges: List[EdgePayload] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Quiz / remediation payloads (Section 7 of brief: response schema carries
# enough for Module 4 to act without a second lookup)
# --------------------------------------------------------------------------- #
class QuizQuestionPayload(BaseModel):
    stem: str
    options: List[str]
    correct_index: int
    target_node: str


class QuizResponsePayload(BaseModel):
    target_node: str
    selected_index: int
    is_correct: bool
    attributed_misconception: Optional[str] = None
    match_score: Optional[float] = None
    action: Optional[str] = None


# --------------------------------------------------------------------------- #
# Bridge: GraphModel <-> GraphPayload
# --------------------------------------------------------------------------- #
def node_to_payload(node: ConceptNode, computed_progress: Optional[float] = None) -> NodePayload:
    return NodePayload(
        id=node.id,
        label=node.label or node.id.replace("_", " ").title(),
        status=node.status.value,
        node_type=getattr(node.node_type, "value", str(node.node_type)),
        parent_id=node.parent_id,
        children=list(node.children_ids),
        depth=node.depth,
        cluster_id=node.cluster_id,
        progress=computed_progress if computed_progress is not None else node.progress,
        is_dynamic_remediation=node.is_dynamic_remediation,
        triggering_misconception=node.triggering_misconception,
    )


def edge_to_payload(edge: ConceptEdge) -> EdgePayload:
    prereq_type = getattr(edge, "prerequisite_type", PrerequisiteType.REQUIRED)
    prereq_str = getattr(prereq_type, "value", str(prereq_type))
    return EdgePayload(
        source=edge.source,
        target=edge.target,
        type=edge.type.value,
        confidence=edge.confidence,
        prerequisite_type=prereq_str if prereq_str in ("required", "recommended", "optional") else "required",
    )


def graph_to_payload(
    graph: GraphModel, goal_concept: str, time_budget_minutes: int
) -> GraphPayload:
    """Serialize a live ``GraphModel`` to the canonical wire object."""
    nodes = []
    for n in graph.nodes():
        prog = graph.compute_progress(n.id)
        nodes.append(node_to_payload(n, computed_progress=prog))

    return GraphPayload(
        student_state=StudentState(
            goal_concept=goal_concept,
            time_budget_minutes=time_budget_minutes,
        ),
        nodes=nodes,
        edges=[edge_to_payload(e) for e in graph.edges()],
    )


def payload_to_graph(payload: GraphPayload) -> GraphModel:
    """Rehydrate a ``GraphModel`` from a payload (used by tests / recovery)."""
    graph = GraphModel()
    for n in payload.nodes:
        graph.add_node(ConceptNode(
            id=n.id,
            label=n.label or n.id.replace("_", " ").title(),
            status=NodeStatus(n.status),
            node_type=NodeType(n.node_type) if n.node_type in ("root", "branch", "leaf") else NodeType.LEAF,
            parent_id=n.parent_id,
            children_ids=list(n.children),
            depth=n.depth,
            cluster_id=n.cluster_id,
            progress=n.progress,
            is_dynamic_remediation=n.is_dynamic_remediation,
            triggering_misconception=n.triggering_misconception,
        ))
    for e in payload.edges:
        graph.add_edge(ConceptEdge(
            source=e.source,
            target=e.target,
            type=EdgeType(e.type),
            confidence=e.confidence,
            prerequisite_type=PrerequisiteType(e.prerequisite_type) if hasattr(PrerequisiteType, e.prerequisite_type.upper()) else PrerequisiteType.REQUIRED,
        ))
    return graph