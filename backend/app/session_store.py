"""Session store: one live ``GraphModel`` (+ engines) per learner session.

The paper requires Module 4 to mutate the *live* graph object the FastAPI
service holds per session -- not recompute from scratch. This module owns that
registry, so routers never create ad-hoc copies.
"""

from __future__ import annotations

from typing import Dict, Optional
from uuid import uuid4

from .services.graph_model import GraphModel
from .services.module4.engine import RemediationEngine


class SessionStore:
    def __init__(self) -> None:
        self._graphs: Dict[str, GraphModel] = {}
        self._engines: Dict[str, RemediationEngine] = {}
        self._goals: Dict[str, str] = {}
        self._budgets: Dict[str, int] = {}

    def create(self, goal_concept: str, time_budget_minutes: int,
               session_id: Optional[str] = None) -> str:
        sid = session_id or uuid4().hex
        self._graphs[sid] = GraphModel()
        self._engines[sid] = RemediationEngine(self._graphs[sid])
        self._goals[sid] = goal_concept
        self._budgets[sid] = time_budget_minutes
        return sid

    def get_graph(self, session_id: str) -> GraphModel:
        if session_id not in self._graphs:
            raise KeyError(f"unknown session: {session_id!r}")
        return self._graphs[session_id]

    def get_engine(self, session_id: str) -> RemediationEngine:
        if session_id not in self._engines:
            raise KeyError(f"unknown session: {session_id!r}")
        return self._engines[session_id]

    def get_goal(self, session_id: str) -> str:
        return self._goals[session_id]

    def get_budget(self, session_id: str) -> int:
        return self._budgets[session_id]

    def replace_graph(self, session_id: str, graph: GraphModel) -> None:
        """Swap in a rebuilt graph while keeping the engine bound to it."""
        if session_id not in self._graphs:
            raise KeyError(f"unknown session: {session_id!r}")
        self._graphs[session_id] = graph
        self._engines[session_id] = RemediationEngine(graph)

    def has(self, session_id: str) -> bool:
        return session_id in self._graphs


#: Application-wide (single-process) session store. A deployment behind a load
#: balancer would replace this with a shared store; the contract and engines
#: remain unchanged.
session_store = SessionStore()