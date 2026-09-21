"""Module 4 -- dynamic graph rewriter (live remediation-node injection).

Section 4.4 / brief Section 8: when a learner selects Option B (prerequisite
distractor) mapped to misconception ``M_k`` for target node ``v``:

    1. Inject a temporary remediation node ``RM_k`` immediately upstream of v.
    2. Reroute edges so incoming edges to v go through ``RM_k`` first:
       ``u -> RM_k -> v``.
    3. Lock node v (``status: locked``) until ``RM_k``'s remediation criteria
       are satisfied.
    4. Once satisfied, resolve ``RM_k`` and restore the original edge structure.

This is a *runtime graph mutation*, not a static recompute. The rewriter
operates on the live in-memory ``GraphModel`` the FastAPI service holds per
session, and every mutation is immediately visible in the next JSON-contract
response so the React Flow render updates without a full page reload.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from ..config import config
from ..graph_model import (
    ConceptEdge,
    ConceptNode,
    EdgeType,
    GraphModel,
    NodeStatus,
)


@dataclass
class RewriteResult:
    """Outcome of a remediation injection or resolution."""

    injected_node: Optional[str] = None
    target: str = ""
    misconception_id: str = ""
    locked: bool = False
    resolved: bool = False
    rerouted_edges: List[Tuple[str, str, str]] = field(default_factory=list)
    restored_edges: List[Tuple[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "injected_node": self.injected_node,
            "target": self.target,
            "misconception_id": self.misconception_id,
            "locked": self.locked,
            "resolved": self.resolved,
            "rerouted_edges": self.rerouted_edges,
            "restored_edges": self.restored_edges,
        }


def remediation_node_id(misconception_id: str, target: str) -> str:
    """Canonical id for a remediation node: ``RM_<M_k>_<target>``.

    Accepts both ``"M_7"`` and ``"M7"`` forms of the misconception id.
    """
    kid = misconception_id[2:] if misconception_id.startswith("M_") \
        else misconception_id[1:] if misconception_id.startswith("M") \
        else misconception_id
    return f"RM_{kid}_{target}"


class GraphRewriter:
    """Mutates the live ``GraphModel`` for misconception-driven remediation."""

    def __init__(self, graph: GraphModel,
                 resolution_successes: Optional[int] = None) -> None:
        self.graph = graph
        self.required_successes = (
            config.module4.resolution_successes
            if resolution_successes is None else resolution_successes
        )
        #: Per-remediation-node consecutive-success counters.
        self._success_counts: Dict[str, int] = {}
        #: Pre-rewrite structure stash: remediation -> (target, incoming, stability)
        self._stash: Dict[str, Tuple[str, FrozenIncoming]] = {}

    # -- injection ------------------------------------------------------ #
    def inject_remediation(self, target: str, misconception_id: str) -> RewriteResult:
        """Inject ``RM_k`` upstream of ``target``, reroute, and lock target."""
        if not self.graph.has_node(target):
            raise KeyError(f"cannot remediate unknown target: {target!r}")
        rm_id = remediation_node_id(misconception_id, target)
        if self.graph.has_node(rm_id):
            # Already remediating; re-lock and return idempotent result.
            target_node = self.graph.get_node(target)
            target_node.status = NodeStatus.LOCKED
            return RewriteResult(
                injected_node=rm_id, target=target,
                misconception_id=misconception_id, locked=True,
            )

        # 1. Snapshot incoming prerequisite edges for later restore.
        incoming = [
            (e.source, e.target, e.confidence)
            for e in self.graph.edges()
            if e.target == target and e.type == EdgeType.PREREQUISITE
        ]
        self._stash[rm_id] = (target, tuple(
            (s, t, c) for s, t, c in incoming
        ))

        # 2. Inject the remediation node.
        self.graph.add_node(ConceptNode(
            id=rm_id,
            label=f"Remediate {misconception_id}",
            status=NodeStatus.IN_PROGRESS,
            is_dynamic_remediation=True,
            triggering_misconception=misconception_id,
        ))

        # 3. Reroute incoming edges u -> target to u -> rm_id, then rm_id -> v.
        rerouted: List[Tuple[str, str, str]] = []
        for (u, _, c) in incoming:
            self.graph.remove_edge(u, target)
            self.graph.add_edge(ConceptEdge(
                source=u, target=rm_id, confidence=c,
            ))
            rerouted.append((u, rm_id, "prerequisite"))
        # Burden edge rm_id -> target (remediation_link).
        self.graph.add_edge(ConceptEdge(
            source=rm_id, target=target,
            type=EdgeType.REMEDIATION_LINK, confidence=1.0,
        ))
        rerouted.append((rm_id, target, "remediation_link"))

        # 4. Lock target.
        self.graph.get_node(target).status = NodeStatus.LOCKED
        self._success_counts[rm_id] = 0

        return RewriteResult(
            injected_node=rm_id, target=target,
            misconception_id=misconception_id, locked=True,
            rerouted_edges=rerouted,
        )

    # -- remediation progress ------------------------------------------ #
    def record_remediation_success(self, rm_id: str) -> bool:
        """Register a successful review of a remediation node.

        Returns True when the resolution criteria are met (all required
        consecutive successes achieved), False otherwise. Only the injected
        remediation node's successes count toward unlocking the target.
        """
        if not self.graph.has_node(rm_id):
            raise KeyError(f"unknown remediation node: {rm_id!r}")
        self._success_counts[rm_id] = self._success_counts.get(rm_id, 0) + 1
        return self._success_counts[rm_id] >= self.required_successes

    def is_resolved(self, rm_id: str) -> bool:
        return self._success_counts.get(rm_id, 0) >= self.required_successes

    # -- resolution ----------------------------------------------------- #
    def resolve_remediation(self, rm_id: str) -> RewriteResult:
        """Restore the original edge structure and unlock ``target``."""
        if rm_id not in self._stash:
            raise KeyError(f"no stashed structure for {rm_id!r} (not injected?)")
        target, incoming = self._stash.pop(rm_id)

        # Remove rerouted edges incident to rm_id.
        for e in list(self.graph.edges()):
            if e.source == rm_id or e.target == rm_id:
                self.graph.remove_edge(e.source, e.target)

        # Restore the original incoming prerequisite edges to target.
        restored: List[Tuple[str, str]] = []
        for (u, t, c) in incoming:
            if not self.graph.has_edge(u, t):
                self.graph.add_edge(ConceptEdge(
                    source=u, target=t, confidence=c,
                ))
            restored.append((u, t))

        # Unlock target and drop the remediation node entirely.
        if self.graph.has_node(target):
            self.graph.get_node(target).status = NodeStatus.IN_PROGRESS
        self.graph.remove_node(rm_id)
        self._success_counts.pop(rm_id, None)

        return RewriteResult(
            injected_node=rm_id, target=target,
            misconception_id="",
            resolved=True,
            restored_edges=restored,
        )

    # -- decay / refresh ------------------------------------------------ #
    def refresh_edge(self, source: str, target: str) -> None:
        """Mark an edge ``refresher_needed`` by re-locking the target.

        The JSON contract has no explicit ``refresher_needed`` field (Section
        6); the paper says the edge is *marked* refresher-needed. We surface the
        flag via the target node's status and by keeping the prerequisite edge
        in place, letting the frontend infer a micro-review for a decayed
        dependency. Consumers that need an explicit flag can extend
        ``ConceptEdge`` without touching the graph data model elsewhere.
        """
        if not self.graph.has_edge(source, target):
            raise KeyError(f"unknown edge: {source!r} -> {target!r}")
        # Mark the dependency as needing refresh by returning it to an
        # in-progress (not completed) state so the frontend surfaces a
        # micro-review before the learner proceeds to the dependent node.
        if self.graph.has_node(target):
            self.graph.get_node(target).status = NodeStatus.IN_PROGRESS


# Immutable alias for the stash type.
FrozenIncoming = Tuple[Tuple[str, str, float], ...]


def remediation_targets(graph: GraphModel) -> List[str]:
    """List currently-injected remediation nodes (for the API layer)."""
    return [
        n.id for n in graph.nodes()
        if n.is_dynamic_remediation
    ]