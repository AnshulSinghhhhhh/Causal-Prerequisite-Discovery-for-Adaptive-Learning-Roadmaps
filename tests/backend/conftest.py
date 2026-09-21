"""Shared pytest fixtures.

Ensures the repo root is on ``sys.path`` so ``backend.app.*`` imports resolve
regardless of where pytest is invoked from, and provides small synthetic
concept graphs reused across module tests.
"""

from __future__ import annotations

import os
import sys

import pytest

# Ensure `<repo>` and `<repo>/backend` are importable (conftest lives at
# `<repo>/tests/backend/conftest.py`, so three dirname() calls reach the root).
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (_REPO, os.path.join(_REPO, "backend")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


@pytest.fixture
def synthetic_embeddings():
    """Deterministic 384-dim embeddings keyed by concept name."""
    import numpy as np

    rng = np.random.default_rng(0)
    concepts = ["A", "B", "C", "D", "E"]
    return {c: rng.normal(size=384).astype(np.float32) for c in concepts}


@pytest.fixture
def simple_graph():
    """A small DAG: A->B, A->C, B->D, C->D."""
    from backend.app.services.graph_model import (
        ConceptEdge,
        ConceptNode,
        GraphModel,
    )

    g = GraphModel()
    for nid in ["A", "B", "C", "D"]:
        g.add_node(ConceptNode(id=nid, label=nid))
    for u, v in [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")]:
        g.add_edge(ConceptEdge(source=u, target=v, confidence=0.8))
    return g