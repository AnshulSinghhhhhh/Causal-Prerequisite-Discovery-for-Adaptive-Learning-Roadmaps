"""Graph state endpoints: create session, fetch the current graph payload,
and build/attach a DAG + optimized path from Module 1/2 outputs."""
from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..models.schema import GraphPayload, graph_to_payload, payload_to_graph
from ..session_store import session_store
from ..services.graph_model import ConceptEdge, ConceptNode, NodeStatus
from ..services.module2.build_graph import build_dag, graph_from_dag

from ..services.content_store import complete_and_unlock_next
from ..services import pipeline

router = APIRouter(prefix="/graph", tags=["graph"])


class CreateSessionRequest(BaseModel):
    goal_concept: str
    time_budget_minutes: int = Field(default=0, ge=0)
    session_id: Optional[str] = None
    generate_roadmap: bool = False


class CreateSessionResponse(BaseModel):
    session_id: str
    graph: GraphPayload


class GenerateGraphRequest(BaseModel):
    goal_concept: str
    time_budget_minutes: int = Field(default=300, ge=0)
    model_version_id: Optional[str] = None
    session_id: Optional[str] = None


class CompleteNodeRequest(BaseModel):
    node_id: str


class BuildGraphRequest(BaseModel):
    """A scored edge list from Module 1 (or the pipeline) to assemble a DAG."""

    node_ids: List[str]
    pairs: List[tuple]  # (source, target)
    scores: List[float]
    tau_edge: Optional[float] = None


@router.post("", response_model=CreateSessionResponse)
def run_pipeline_graph(req: GenerateGraphRequest) -> CreateSessionResponse:
    """Run full LightGAP algorithmic pipeline (S0->S1->S2->S4->S5->S6) and return GraphPayload."""
    sid = session_store.create(req.goal_concept, req.time_budget_minutes, session_id=req.session_id)
    pipe_res = pipeline.run(
        req.goal_concept.strip(),
        model_version_id=req.model_version_id,
        time_budget_minutes=req.time_budget_minutes,
    )
    dynamic_graph = pipeline.pipeline_result_to_graph_model(pipe_res, req.goal_concept.strip(), session_id=sid)
    session_store.replace_graph(sid, dynamic_graph)
    payload = graph_to_payload(dynamic_graph, req.goal_concept, req.time_budget_minutes)
    return CreateSessionResponse(session_id=sid, graph=payload)


@router.post("/session", response_model=CreateSessionResponse)
def create_session(req: CreateSessionRequest) -> CreateSessionResponse:
    sid = session_store.create(req.goal_concept, req.time_budget_minutes,
                               session_id=req.session_id)
    if req.generate_roadmap and req.goal_concept.strip():
        pipe_res = pipeline.run(req.goal_concept.strip(), time_budget_minutes=req.time_budget_minutes)
        dynamic_graph = pipeline.pipeline_result_to_graph_model(pipe_res, req.goal_concept.strip(), session_id=sid)
        session_store.replace_graph(sid, dynamic_graph)

    payload = graph_to_payload(
        session_store.get_graph(sid), req.goal_concept, req.time_budget_minutes
    )
    return CreateSessionResponse(session_id=sid, graph=payload)


@router.post("/{session_id}/complete-node", response_model=GraphPayload)
def complete_node(session_id: str, req: CompleteNodeRequest) -> GraphPayload:
    """Mark a node as COMPLETED and unlock downstream dependent nodes."""
    try:
        graph = session_store.get_graph(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found")

    complete_and_unlock_next(graph, req.node_id)
    return graph_to_payload(
        graph, session_store.get_goal(session_id),
        session_store.get_budget(session_id),
    )


@router.get("/{session_id}/next-suggestion")
def get_next_suggestion(session_id: str) -> dict:
    """Find the next recommended concept node for the learner."""
    try:
        graph = session_store.get_graph(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found")
    next_id = graph.get_next_available()
    if next_id and graph.has_node(next_id):
        node = graph.get_node(next_id)
        return {"next_node_id": next_id, "label": node.label or next_id, "depth": node.depth}
    return {"next_node_id": None, "label": None, "depth": 0}


@router.get("/{session_id}/can-start/{node_id}")
def can_start_node(session_id: str, node_id: str) -> dict:
    """Check whether a node can be studied or if prerequisites are locked."""
    try:
        graph = session_store.get_graph(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found")
    if not graph.has_node(node_id):
        raise HTTPException(status_code=404, detail="node not found")
    
    node = graph.get_node(node_id)
    if node.status in (NodeStatus.IN_PROGRESS, NodeStatus.AVAILABLE, NodeStatus.COMPLETED):
        return {"can_start": True, "node_id": node_id, "status": node.status.value, "unmet_prerequisites": []}

    unmet = []
    for pid in graph.in_neighbors(node_id):
        if graph.has_node(pid) and graph.get_node(pid).status != NodeStatus.COMPLETED:
            p = graph.get_node(pid)
            unmet.append({"id": pid, "label": p.label or pid, "status": p.status.value})

    return {"can_start": False, "node_id": node_id, "status": node.status.value, "unmet_prerequisites": unmet}


@router.get("/{session_id}", response_model=GraphPayload)
def get_graph(session_id: str) -> GraphPayload:
    try:
        graph = session_store.get_graph(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found")
    return graph_to_payload(
        graph, session_store.get_goal(session_id),
        session_store.get_budget(session_id),
    )


@router.post("/{session_id}/build", response_model=GraphPayload)
def build_session_graph(session_id: str, req: BuildGraphRequest) -> GraphPayload:
    """Assemble a DAG + attach it to the session graph (Module 2)."""
    try:
        session_store.get_graph(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found")
    if len(req.pairs) != len(req.scores):
        raise HTTPException(status_code=422,
                            detail="pairs and scores must have equal length")

    result = build_dag(req.pairs, req.scores, tau_edge=req.tau_edge)
    if not result.is_dag:
        raise HTTPException(status_code=500,
                            detail="assembled graph is not a DAG")

    # Replace the session graph with the materialized DAG (plus any existing
    # dynamic remediation state, which we preserve by merging rather than
    # clobbering -- the live rewriter owns remediation nodes).
    existing = session_store.get_graph(session_id)
    new_graph = graph_from_dag(result, node_labels={n: n for n in req.node_ids})
    # Preserve runtime remediation nodes that may already be present.
    for node in existing.nodes():
        if node.is_dynamic_remediation and not new_graph.has_node(node.id):
            new_graph.add_node(ConceptNode(
                id=node.id, label=node.label, status=node.status,
                is_dynamic_remediation=node.is_dynamic_remediation,
                triggering_misconception=node.triggering_misconception,
            ))
    # Copy stability/review state forward.
    for node in existing.nodes():
        if new_graph.has_node(node.id):
            last = existing.get_last_review(node.id)
            stab = existing.get_stability(node.id)
            if last is not None:
                new_graph.set_last_review(node.id, last)
            if stab is not None:
                new_graph.set_stability(node.id, stab)

    session_store.replace_graph(session_id, new_graph)
    return graph_to_payload(
        new_graph, session_store.get_goal(session_id),
        session_store.get_budget(session_id),
    )