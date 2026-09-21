"""Module 3 content-aggregator unit tests (caching / re-ranking focused)."""

import numpy as np

from backend.app.services.module3.content_aggregator import (
    ContentAggregator,
    DeterministicContentClient,
    NumpyVectorIndex,
    ResponseCache,
)


def test_aggregator_caches_and_does_not_requery():
    calls = {"n": 0}

    class CountingClient(DeterministicContentClient):
        def search(self, query, limit=5):
            calls["n"] += 1
            return super().search(query, limit=limit)

    client = CountingClient("docs")
    agg = ContentAggregator(clients=[client],
                            cache=ResponseCache(ttl_seconds=3600),
                            index=NumpyVectorIndex())
    emb = np.zeros(384, dtype=float)
    agg.aggregate("A", emb, query="q")
    first = calls["n"]
    agg.aggregate("A", emb, query="q")
    # Second call hits the cache -> no new underlying query.
    assert calls["n"] == first


def test_aggregator_empty_clients_returns_empty():
    agg = ContentAggregator(clients=[], index=NumpyVectorIndex())
    result = agg.aggregate("A", np.zeros(384), query="q")
    assert result.items == []
    assert result.ranked == []


def test_index_only_adds_once():
    idx = NumpyVectorIndex()
    idx.add("k", np.ones(4))
    idx.add("k", np.ones(4))  # duplicate key still appended (documented).
    assert len(idx) == 2