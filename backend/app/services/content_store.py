"""Content and node lifecycle store for LightGAP.

Provides in-memory node content storage (rich learning packets, diagnostic quizzes)
and graph lifecycle unlocking (cascading state transitions when nodes complete).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .graph_model import GraphModel, NodeStatus

# In-memory storage for node learning materials & diagnostic quizzes keyed by (session_id, node_id)
_NODE_CONTENT_STORE: Dict[Tuple[str, str], Dict[str, Any]] = {}


def store_node_content(session_id: str, node_id: str, content: Dict[str, Any]) -> None:
    """Store enriched study pack and quiz data for a node in a session."""
    _NODE_CONTENT_STORE[(session_id, node_id)] = content


def get_node_content(session_id: str, node_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve stored study pack and quiz data for a node."""
    return _NODE_CONTENT_STORE.get((session_id, node_id))


def complete_and_unlock_next(graph: GraphModel, node_id: str) -> Dict[str, Any]:
    """Cascading completion and unlocking for hierarchical learning graph.
    
    1. Mark current node as COMPLETED.
    2. Unlock next sequential sibling to AVAILABLE/IN_PROGRESS.
    3. Update parent progress recursively; if parent reaches 100%, complete parent and unlock next branch.
    4. Unlock downstream DAG out-neighbors whose prerequisites are all satisfied.
    """
    if not graph.has_node(node_id):
        return {"completed": [], "unlocked": [], "next_suggestion": None}

    completed_nodes: List[str] = [node_id]
    unlocked_nodes: List[str] = []

    target_node = graph.get_node(node_id)
    target_node.status = NodeStatus.COMPLETED

    # 1. Check siblings within same parent
    parent = graph.get_parent(node_id)
    if parent:
        siblings = parent.children_ids
        if node_id in siblings:
            idx = siblings.index(node_id)
            if idx + 1 < len(siblings):
                next_sib_id = siblings[idx + 1]
                if graph.has_node(next_sib_id):
                    next_sib = graph.get_node(next_sib_id)
                    if next_sib.status == NodeStatus.LOCKED:
                        next_sib.status = NodeStatus.AVAILABLE
                        unlocked_nodes.append(next_sib_id)
                        if next_sib.children_ids:
                            first_child_id = next_sib.children_ids[0]
                            if graph.has_node(first_child_id):
                                fc = graph.get_node(first_child_id)
                                if fc.status == NodeStatus.LOCKED:
                                    fc.status = NodeStatus.AVAILABLE
                                    unlocked_nodes.append(first_child_id)

    # 2. Check parent completion cascade
    curr_parent = parent
    while curr_parent is not None:
        all_children_done = True
        for cid in curr_parent.children_ids:
            if graph.has_node(cid) and graph.get_node(cid).status != NodeStatus.COMPLETED:
                all_children_done = False
                break
        
        if all_children_done and curr_parent.status != NodeStatus.COMPLETED:
            curr_parent.status = NodeStatus.COMPLETED
            completed_nodes.append(curr_parent.id)

            grandparent = graph.get_parent(curr_parent.id)
            if grandparent:
                mod_siblings = grandparent.children_ids
                if curr_parent.id in mod_siblings:
                    m_idx = mod_siblings.index(curr_parent.id)
                    if m_idx + 1 < len(mod_siblings):
                        next_mod_id = mod_siblings[m_idx + 1]
                        if graph.has_node(next_mod_id):
                            next_mod = graph.get_node(next_mod_id)
                            if next_mod.status == NodeStatus.LOCKED:
                                next_mod.status = NodeStatus.AVAILABLE
                                unlocked_nodes.append(next_mod_id)
                                if next_mod.children_ids:
                                    first_c_id = next_mod.children_ids[0]
                                    if graph.has_node(first_c_id):
                                        fc_node = graph.get_node(first_c_id)
                                        if fc_node.status == NodeStatus.LOCKED:
                                            fc_node.status = NodeStatus.AVAILABLE
                                            unlocked_nodes.append(first_c_id)

            curr_parent = grandparent
        else:
            break

    # 3. Check general DAG out-neighbors
    for out_id in graph.out_neighbors(node_id):
        if not graph.has_node(out_id):
            continue
        out_node = graph.get_node(out_id)
        if out_node.status == NodeStatus.LOCKED:
            all_in_done = True
            for in_id in graph.in_neighbors(out_id):
                if graph.has_node(in_id) and graph.get_node(in_id).status != NodeStatus.COMPLETED:
                    all_in_done = False
                    break
            if all_in_done:
                out_node.status = NodeStatus.AVAILABLE
                if out_id not in unlocked_nodes:
                    unlocked_nodes.append(out_id)

    # 4. Find next recommended node
    next_suggestion = graph.get_next_available()

    return {
        "completed": completed_nodes,
        "unlocked": unlocked_nodes,
        "next_suggestion": next_suggestion,
    }
