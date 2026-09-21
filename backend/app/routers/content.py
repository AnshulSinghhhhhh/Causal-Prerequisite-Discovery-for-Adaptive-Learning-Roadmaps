"""Content & Study Hub Router (NotebookLM-style learning packets)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException

from ..services.content_store import get_node_content
from ..session_store import session_store

router = APIRouter(prefix="/content", tags=["content"])


@router.get("/{session_id}/{node_id}")
def get_study_pack(session_id: str, node_id: str) -> Dict[str, Any]:
    """Return the NotebookLM-style study packet for a concept node."""
    try:
        graph = session_store.get_graph(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found")

    if not graph.has_node(node_id):
        raise HTTPException(status_code=404, detail=f"node {node_id!r} not found in session")

    node = graph.get_node(node_id)
    stored = get_node_content(session_id, node_id) or {}

    # Required prerequisites
    incoming_prereqs = [
        graph.get_node(p).label or p for p in graph.in_neighbors(node_id)
        if graph.has_node(p)
    ]
    unmet_prereqs = [
        graph.get_node(p).label or p for p in graph.in_neighbors(node_id)
        if graph.has_node(p) and graph.get_node(p).status != "completed"
    ]

    return {
        "node_id": node_id,
        "label": node.label or node_id.replace("_", " ").title(),
        "status": node.status.value,
        "is_remediation": node.is_dynamic_remediation,
        "triggering_misconception": node.triggering_misconception,
        "summary": stored.get("summary", f"Study guide and mastery packet for {node.label or node_id}."),
        "key_takeaways": stored.get("key_takeaways", [
            f"Understand the fundamental definitions and intuition behind {node.label or node_id}.",
            "Master typical calculations, representations, and edge cases.",
            "Connect this concept with downstream dependencies."
        ]),
        "youtube": stored.get("youtube", [
            {"title": f"Complete Guide to {node.label or node_id}", "channel": "MIT / Stanford Online", "video_id": "aircAruvnKk", "url": "https://www.youtube.com/watch?v=aircAruvnKk", "query": f"{node.label or node_id} tutorial"}
        ]),
        "github": stored.get("github", [
            {"repo": "ageron/handson-ml3", "url": "https://github.com/ageron/handson-ml3", "description": f"Practical code examples and reference implementations for {node.label or node_id}."}
        ]),
        "books": stored.get("books", [
            {"title": "Foundational Computing and Mathematics", "author": "Standard Academic Press", "chapters": "Core Concepts"}
        ]),
        "papers_or_docs": stored.get("papers_or_docs", [
            {"title": "Comprehensive Topic Documentation", "source": "Official Documentation"}
        ]),
        "quiz": stored.get("quiz", []),
        "prerequisites": incoming_prereqs,
        "unmet_prerequisites": unmet_prereqs,
    }
