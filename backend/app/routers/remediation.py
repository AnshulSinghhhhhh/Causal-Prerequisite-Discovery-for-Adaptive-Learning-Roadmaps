"""Module 4 remediation/decay endpoints."""
from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..models.schema import GraphPayload, graph_to_payload
from ..session_store import session_store

router = APIRouter(prefix="/remediation", tags=["remediation"])


class RemediateRequest(BaseModel):
    session_id: str
    target_node: str
    misconception_id: str


class ResolveRequest(BaseModel):
    session_id: str
    remediation_node_id: str


class DecayCheckRequest(BaseModel):
    session_id: str
    now: Optional[float] = None


class RemediateResponse(BaseModel):
    injected_node: Optional[str]
    target: str
    locked: bool


class DecayCheckResponse(BaseModel):
    decayed_nodes: List[str]
    graph: GraphPayload


@router.post("/inject", response_model=RemediateResponse)
def inject_remediation(req: RemediateRequest) -> RemediateResponse:
    try:
        engine = session_store.get_engine(req.session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found")
    try:
        result = engine.rewriter.inject_remediation(
            req.target_node, req.misconception_id
        )
    except KeyError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return RemediateResponse(
        injected_node=result.injected_node, target=result.target,
        locked=result.locked,
    )


@router.post("/resolve", response_model=GraphPayload)
def resolve_remediation(req: ResolveRequest) -> GraphPayload:
    try:
        engine = session_store.get_engine(req.session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found")
    try:
        engine.rewriter.resolve_remediation(req.remediation_node_id)
    except KeyError as e:
        raise HTTPException(status_code=422, detail=str(e))
    graph = session_store.get_graph(req.session_id)
    return graph_to_payload(
        graph, session_store.get_goal(req.session_id),
        session_store.get_budget(req.session_id),
    )


@router.post("/decay-check", response_model=DecayCheckResponse)
def decay_check(req: DecayCheckRequest) -> DecayCheckResponse:
    try:
        engine = session_store.get_engine(req.session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found")
    decayed = engine.decayed_nodes(now=req.now)
    graph = session_store.get_graph(req.session_id)
    return DecayCheckResponse(
        decayed_nodes=decayed,
        graph=graph_to_payload(
            graph, session_store.get_goal(req.session_id),
            session_store.get_budget(req.session_id),
        ),
    )