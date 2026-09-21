"""Canonical graph data model for LightGAP.

Per Section 1.4 of the project governing methodology, the in-memory graph
object (nodes, edges, statuses, misconceptions) must have *exactly one*
implementation imported everywhere. This module is that implementation:
Modules 1-4, the FastAPI routers, and -- via the JSON contract in
``models.schema`` -- the frontend all consume this single representation.
A fix made here propagates to every consumer; a duplicated copy would drift.
"""

from __future__ import annotations

import dataclasses
import itertools
from enum import Enum
from typing import Dict, Iterable, List, Optional, Set, Tuple


class NodeStatus(str, Enum):
    """Lifecycle state of a concept node (see Section 6 of the paper)."""

    COMPLETED = "completed"
    IN_PROGRESS = "in_progress"
    AVAILABLE = "available"
    LOCKED = "locked"


class NodeType(str, Enum):
    """Hierarchy kind for mind-map organization.

    Per spec 01 schema contract: 'branch' = any non-leaf, non-root node.
    Depth is an arbitrary int (from S5 longest-path layering), not a
    fixed enum level. Don't reintroduce a level cap anywhere downstream.
    """

    ROOT = "root"
    BRANCH = "branch"
    LEAF = "leaf"


class PrerequisiteType(str, Enum):
    """Relationship strength between concepts."""

    REQUIRED = "required"
    RECOMMENDED = "recommended"
    OPTIONAL = "optional"


class EdgeType(str, Enum):
    """Edge kind. ``remediation_link`` is introduced by Module 4 rewrites."""

    PREREQUISITE = "prerequisite"
    REMEDIATION_LINK = "remediation_link"
    HIERARCHY = "hierarchy"


@dataclasses.dataclass
class ConceptNode:
    """A single concept (or injected remediation node) in the live graph."""

    id: str
    #: Convention id for ordinary concepts; remediation nodes use ``RM_<k>``.
    label: str = ""
    status: NodeStatus = NodeStatus.IN_PROGRESS
    node_type: NodeType = NodeType.LEAF
    parent_id: Optional[str] = None
    children_ids: List[str] = dataclasses.field(default_factory=list)
    #: Depth from S5 longest-path layering — arbitrary range, not 0/1/2.
    depth: int = 0
    #: Cluster ID from S5 Leiden community detection.
    cluster_id: Optional[str] = None
    progress: float = 0.0
    #: Module 4 flags injected nodes so the contract can round-trip them.
    is_dynamic_remediation: bool = False
    #: Misconception id (``M_k``) that caused a remediation node to appear.
    triggering_misconception: Optional[str] = None
    #: Original (post-reduction) prerequisite edge set, restored on resolve.
    _locked_prereq: FrozenSetOfEdges = ()


@dataclasses.dataclass
class ConceptEdge:
    """A directed edge ``source -> target`` with an associated weight."""

    source: str
    target: str
    type: EdgeType = EdgeType.PREREQUISITE
    #: Prerequisite confidence in [0, 1] for prerequisite edges;
    #: 1.0 (placeholder) for remediation links.
    confidence: float = 1.0
    prerequisite_type: PrerequisiteType = PrerequisiteType.REQUIRED


# Immutable edge identity used when stashing the pre-rewrite structure.
FrozenSetOfEdges = Tuple[Tuple[str, str], ...]


class GraphModel:
    """Mutable, live graph object shared by the service and Modules 2/4.

    Thread-safety is deliberately out of scope: the FastAPI service holds one
    instance per learner session, and rewrites are serialised at the router
    layer. The model is intentionally storage-agnostic so it can be backed by
    a plain dict here and a real store later without changing consumers.
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, ConceptNode] = {}
        self._edges: Dict[Tuple[str, str], ConceptEdge] = {}
        #: Seconds since epoch when a concept was last reviewed.
        self._last_review: Dict[str, float] = {}
        #: Per-concept, per-learner stability S_i (Module 4 decay model).
        self._stability: Dict[str, float] = {}

    # ------------------------------------------------------------------ #
    # Node access
    # ------------------------------------------------------------------ #
    def add_node(self, node: ConceptNode) -> None:
        if node.id in self._nodes:
            raise ValueError(f"node already exists: {node.id!r}")
        self._nodes[node.id] = dataclasses.replace(node)

    def get_node(self, node_id: str) -> ConceptNode:
        try:
            return self._nodes[node_id]
        except KeyError:
            raise KeyError(f"unknown node: {node_id!r}") from None

    def has_node(self, node_id: str) -> bool:
        return node_id in self._nodes

    def node_ids(self) -> List[str]:
        return list(self._nodes)

    def nodes(self) -> Iterable[ConceptNode]:
        return iter(self._nodes.values())

    def remove_node(self, node_id: str) -> None:
        """Remove a node and every incident edge.

        Module 4 calls this to drop a resolved remediation node; consumers must
        never reach into ``_nodes`` / ``_edges`` directly.
        """
        if not self.has_node(node_id):
            raise KeyError(f"unknown node: {node_id!r}")
        for edge in list(self._edges.values()):
            if edge.source == node_id or edge.target == node_id:
                del self._edges[(edge.source, edge.target)]
        del self._nodes[node_id]
        self._last_review.pop(node_id, None)
        self._stability.pop(node_id, None)

    # ------------------------------------------------------------------ #
    # Edge access / mutation
    # ------------------------------------------------------------------ #
    def add_edge(self, edge: ConceptEdge) -> None:
        key = (edge.source, edge.target)
        if key in self._edges:
            raise ValueError(f"edge already exists: {key!r}")
        if not self.has_node(edge.source) or not self.has_node(edge.target):
            raise ValueError(f"edge references unknown node: {key!r}")
        self._edges[key] = dataclasses.replace(edge)

    def get_edge(self, source: str, target: str) -> ConceptEdge:
        try:
            return self._edges[(source, target)]
        except KeyError:
            raise KeyError(f"unknown edge: {source!r} -> {target!r}") from None

    def has_edge(self, source: str, target: str) -> bool:
        return (source, target) in self._edges

    def remove_edge(self, source: str, target: str) -> None:
        del self._edges[(source, target)]

    def edges(self) -> Iterable[ConceptEdge]:
        return iter(self._edges.values())

    def out_neighbors(self, node_id: str) -> List[str]:
        return [
            e.target for e in self._edges.values() if e.source == node_id
        ]

    def in_neighbors(self, node_id: str) -> List[str]:
        return [
            e.source for e in self._edges.values() if e.target == node_id
        ]

    def edge_set(self) -> FrozenSetOfEdges:
        """Immutable (u, v) snapshot of the current prerequisite edges."""
        return tuple(
            (s, t)
            for (s, t), e in self._edges.items()
            if e.type == EdgeType.PREREQUISITE
        )

    # ------------------------------------------------------------------ #
    # Review / stability state (Module 4)
    # ------------------------------------------------------------------ #
    def set_last_review(self, node_id: str, timestamp: float) -> None:
        self._last_review[node_id] = timestamp

    def get_last_review(self, node_id: str) -> Optional[float]:
        return self._last_review.get(node_id)

    def set_stability(self, node_id: str, stability: float) -> None:
        if stability <= 0:
            raise ValueError("stability must be positive")
        self._stability[node_id] = stability

    def get_stability(self, node_id: str) -> Optional[float]:
        return self._stability.get(node_id)

    def clear_remediation_state(self) -> None:
        """Drop runtime-only state; kept separate from structural data."""
        for edge in list(self._edges.values()):
            if edge.type == EdgeType.REMEDIATION_LINK:
                del self._edges[(edge.source, edge.target)]

    # ------------------------------------------------------------------ #
    # Hierarchy & tree traversal
    # ------------------------------------------------------------------ #
    def get_children(self, node_id: str) -> List[ConceptNode]:
        """Get ordered children of a node."""
        if not self.has_node(node_id):
            return []
        node = self.get_node(node_id)
        return [self.get_node(cid) for cid in node.children_ids if self.has_node(cid)]

    def get_parent(self, node_id: str) -> Optional[ConceptNode]:
        """Get parent node, or None if root."""
        if not self.has_node(node_id):
            return None
        node = self.get_node(node_id)
        if node.parent_id and self.has_node(node.parent_id):
            return self.get_node(node.parent_id)
        return None

    def get_root(self) -> Optional[ConceptNode]:
        """Get the root node of the tree."""
        for node in self.nodes():
            if node.node_type == NodeType.ROOT:
                return node
        # Fallback to depth 0 or first node
        for node in self.nodes():
            if node.depth == 0:
                return node
        nodes = list(self.nodes())
        return nodes[0] if nodes else None

    def compute_progress(self, node_id: str) -> float:
        """Recursively compute completion progress for a node (0.0 to 1.0)."""
        if not self.has_node(node_id):
            return 0.0
        node = self.get_node(node_id)
        if node.node_type == NodeType.LEAF or not node.children_ids:
            return 1.0 if node.status == NodeStatus.COMPLETED else 0.0
        children = self.get_children(node_id)
        if not children:
            return 1.0 if node.status == NodeStatus.COMPLETED else 0.0
        return sum(self.compute_progress(c.id) for c in children) / len(children)

    def get_leaves(self, node_id: Optional[str] = None) -> List[ConceptNode]:
        """Get all leaf descendants of a node (or all leaves if node_id is None)."""
        if node_id is None:
            return [n for n in self.nodes() if n.node_type == NodeType.LEAF]
        if not self.has_node(node_id):
            return []
        node = self.get_node(node_id)
        if node.node_type == NodeType.LEAF or not node.children_ids:
            return [node]
        result = []
        for child in self.get_children(node_id):
            result.extend(self.get_leaves(child.id))
        return result

    def get_next_available(self) -> Optional[str]:
        """Find the next recommended leaf node for the learner to study."""
        # Check available leaf nodes first
        for node in self.nodes():
            if node.status in (NodeStatus.AVAILABLE, NodeStatus.IN_PROGRESS) and (node.node_type == NodeType.LEAF or not node.children_ids):
                return node.id
        return None


def node_key(node_id: str) -> str:
    """Canonical string form of a node id (identity for dedup / hashing)."""
    return str(node_id)


def edge_key(source: str, target: str) -> Tuple[str, str]:
    """Canonical tuple form of a directed edge identity."""
    return (source, target)