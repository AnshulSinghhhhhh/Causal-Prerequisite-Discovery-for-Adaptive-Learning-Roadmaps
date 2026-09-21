"""Module 3 -- content aggregation (API clients, caching, re-ranking, index).

Section 4.3: each node is populated by querying external content APIs
(YouTube Data API v3, GitHub REST API, arXiv, official docs) and re-ranking
candidates by cosine similarity between candidate *metadata* embeddings and the
node's concept embedding ``h_v``. API responses are cached (rate limits are
real; never re-query on every graph render), and candidates are indexed in an
in-memory vector index for fast re-ranking without re-embedding every request
-- the ~50MB RAM / <5ms query budget row (Section 7, Table 2).

ChromaDB is the paper's chosen backend, but it is an optional runtime
dependency (not importable offline). This module therefore defines a narrow
``VectorIndex`` interface with a pure-Python / NumPy in-memory implementation
as the canonical offline default, and a thin ``ChromaVectorIndex`` adapter so
the real backend can be swapped in without changing consumers. Re-ranking math
is identical either way.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..config import config


# --------------------------------------------------------------------------- #
# Item + client interfaces
# --------------------------------------------------------------------------- #
@dataclass
class ContentItem:
    """One retrieved resource candidate per node."""

    kind: str          # "youtube" | "github" | "arxiv" | "docs"
    url: str
    title: str
    description: str = ""
    #: Optional precomputed metadata embedding (kind-agnostic).
    embedding: Optional[np.ndarray] = None
    meta: Dict[str, str] = field(default_factory=dict)

    def metadata_text(self) -> str:
        return " ".join([self.title, self.description] +
                        [f"{k}:{v}" for k, v in sorted(self.meta.items())])


class ContentAPIClient(ABC):
    """A provider that returns candidate items for a query.

    Concrete clients make live HTTP calls; the offline environment wires a
    ``DeterministicContentClient`` instead so aggregation is testable today.
    """

    kind: str = "unknown"

    @abstractmethod
    def search(self, query: str, limit: int = 5) -> List[ContentItem]:
        raise NotImplementedError


class DeterministicContentClient(ContentAPIClient):
    """Offline client returning deterministic synthetic items per kind."""

    def __init__(self, kind: str, template: str = "{q}",
                 seed: int = 0) -> None:
        self.kind = kind
        self.template = template
        self.seed = seed

    def search(self, query: str, limit: int = 5) -> List[ContentItem]:
        rng = np.random.default_rng(self.seed)
        items = []
        for i in range(limit):
            t = self.template.format(q=query, i=i)
            items.append(ContentItem(
                kind=self.kind,
                url=f"{self.kind}://{query.replace(' ', '-')}/{i}",
                title=f"{t} #{i}",
                description=f"resource about {query}",
                meta={"rank": str(i)},
            ))
        return items


# --------------------------------------------------------------------------- #
# Caching
# --------------------------------------------------------------------------- #
class ResponseCache:
    """Simple TTL cache keyed by (kind, query)."""

    def __init__(self, ttl_seconds: Optional[int] = None,
                 store: Optional[Dict[str, Tuple[float, List[ContentItem]]]] = None):
        self.ttl = config.module3.cache_ttl_seconds if ttl_seconds is None \
            else ttl_seconds
        self.store: Dict[str, Tuple[float, List[ContentItem]]] = store or {}
        self._clock = time.time

    def _key(self, kind: str, query: str) -> str:
        return f"{kind}\x1f{query}"

    def get(self, kind: str, query: str) -> Optional[List[ContentItem]]:
        entry = self.store.get(self._key(kind, query))
        if entry is None:
            return None
        ts, items = entry
        if self._clock() - ts > self.ttl:
            del self.store[self._key(kind, query)]
            return None
        return items

    def set(self, kind: str, query: str, items: List[ContentItem]) -> None:
        self.store[self._key(kind, query)] = (self._clock(), items)


# --------------------------------------------------------------------------- #
# Vector index (canonical interface + NumPy default)
# --------------------------------------------------------------------------- #
class VectorIndex(ABC):
    """Narrow interface for similarity re-ranking, backended by Chroma in prod."""

    @abstractmethod
    def add(self, key: str, embedding: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def query(self, embedding: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
        """Return (key, cosine_similarity) descending."""
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class NumpyVectorIndex(VectorIndex):
    """In-memory NumPy index (~50MB footprint class, <5ms typical queries)."""

    def __init__(self) -> None:
        self._keys: List[str] = []
        self._vecs: List[np.ndarray] = []

    def add(self, key: str, embedding: np.ndarray) -> None:
        self._keys.append(key)
        self._vecs.append(np.asarray(embedding, dtype=float).ravel())

    def query(self, embedding: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
        if not self._vecs:
            return []
        q = np.asarray(embedding, dtype=float).ravel()
        sims = [(k, _cosine(q, v)) for k, v in zip(self._keys, self._vecs)]
        sims.sort(key=lambda kv: kv[1], reverse=True)
        return sims[:top_k]

    def __len__(self) -> int:
        return len(self._keys)


class ChromaVectorIndex(VectorIndex):
    """ChromaDB-backed adapter (optional runtime dependency).

    Imported lazily so the module (and its tests) never require chromadb to be
    installed. Falls back to ``NumpyVectorIndex`` when chromadb is unavailable.
    """

    def __init__(self, collection_name: str = "lightgap_content",
                 persistence_dir: Optional[str] = None) -> None:
        try:
            import chromadb  # noqa: F401
        except ImportError:
            self._backend = NumpyVectorIndex()
            self._chroma = False
            return
        import chromadb
        if persistence_dir:
            client = chromadb.PersistentClient(path=persistence_dir)
        else:
            client = chromadb.Client()
        self._col = client.get_or_create_collection(collection_name)
        self._backend = None  # type: ignore[assignment]
        self._chroma = True

    def add(self, key: str, embedding: np.ndarray) -> None:
        if not self._chroma:
            self._backend.add(key, embedding)
            return
        self._col.add(ids=[key], embeddings=[np.asarray(embedding).tolist()])

    def query(self, embedding: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
        if not self._chroma:
            return self._backend.query(embedding, top_k)
        res = self._col.query(
            query_embeddings=[np.asarray(embedding).tolist()], n_results=top_k
        )
        ids = res["ids"][0] if res["ids"] else []
        dists = res["distances"][0] if res["distances"] else []
        # chroma returns distances; convert to cosine-ish similarity (1 - d).
        return [(i, 1.0 - float(d)) for i, d in zip(ids, dists)]

    def __len__(self) -> int:
        if not self._chroma:
            return len(self._backend)
        return self._col.count()


# --------------------------------------------------------------------------- #
# Aggregator
# --------------------------------------------------------------------------- #
@dataclass
class AggregationResult:
    node_id: str
    items: List[ContentItem]
    #: Re-ranked (key, score) pairs from the vector index.
    ranked: List[Tuple[str, float]] = field(default_factory=list)


class ContentAggregator:
    """Fills each concept node with re-ranked external content.

    Uses ``ConceptEmbedder`` to embed candidate *metadata* so re-ranking uses
    the same canonical embedding space as the node's concept vector ``h_v``.
    """

    def __init__(
        self,
        clients: Optional[Sequence[ContentAPIClient]] = None,
        embedder=None,
        cache: Optional[ResponseCache] = None,
        index: Optional[VectorIndex] = None,
    ) -> None:
        self.clients: List[ContentAPIClient] = list(clients or [])
        self.embedder = embedder
        self.cache = cache or ResponseCache()
        self.index = index or NumpyVectorIndex()

    def _metadata_embedding(self, item: ContentItem) -> np.ndarray:
        if item.embedding is not None:
            return item.embedding
        if self.embedder is None:
            # No embedder: fall back to a deterministic bag-of-words embedding
            # so re-ranking still runs offline.
            return _bow_embedding(item.metadata_text())
        return self.embedder.embed([item.metadata_text()]).vectors[0]

    def aggregate(
        self,
        node_id: str,
        concept_embedding: np.ndarray,
        query: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> AggregationResult:
        """Query every client, cache, index, then re-rank by similarity."""
        limit = limit or config.module3.max_items_per_node
        query = query or node_id
        items: List[ContentItem] = []
        for client in self.clients:
            cached = self.cache.get(client.kind, query)
            if cached is not None:
                results = cached
            else:
                results = client.search(query, limit=limit)
                self.cache.set(client.kind, query, results)
            items.extend(results)

        # Index all retrieved items and re-rank against the node embedding.
        keyed: Dict[str, ContentItem] = {}
        for item in items:
            key = f"{item.kind}\x1f{item.url}"
            keyed[key] = item
            emb = self._metadata_embedding(item)
            if key not in {k for k, _ in self.index.query(concept_embedding,
                                                         len(self.index))}:
                self.index.add(key, emb)

        ranked = self.index.query(concept_embedding, top_k=limit)
        # Order items by rank, resolving to ContentItem objects.
        ordered = [keyed[k] for k, _ in ranked if k in keyed]
        return AggregationResult(node_id=node_id, items=ordered, ranked=ranked)


def _bow_embedding(text: str, dim: int = 384) -> np.ndarray:
    """Deterministic bag-of-words fallback embedding (dim matches MiniLM)."""
    vec = np.zeros(dim, dtype=float)
    for tok in text.lower().split():
        h = hashlib.md5(tok.encode("utf-8")).digest()
        idx = int.from_bytes(h[:4], "big") % dim
        vec[idx] += 1.0
    n = np.linalg.norm(vec)
    return vec / n if n else vec


def make_default_clients() -> List[ContentAPIClient]:
    """Wired clients. Without real API keys these are the deterministic stubs;
    with keys set via env vars, real HTTP clients are constructed instead."""
    enabled = config.module3.enabled_apis
    if not enabled:
        # Offline deterministic stubs for each canonical source.
        return [
            DeterministicContentClient("youtube", "video: {q}"),
            DeterministicContentClient("github", "repo: {q}"),
            DeterministicContentClient("arxiv", "paper: {q}"),
            DeterministicContentClient("docs", "doc: {q}"),
        ]
    # Real clients (constructed only when explicitly enabled).
    clients: List[ContentAPIClient] = []
    for name in enabled:
        if name == "arxiv":
            from .content_clients import ArxivClient
            clients.append(ArxivClient())
        elif name == "github":
            from .content_clients import GitHubClient
            clients.append(GitHubClient())
        elif name == "youtube":
            from .content_clients import YouTubeClient
            clients.append(YouTubeClient())
        elif name == "docs":
            from .content_clients import DocsClient
            clients.append(DocsClient())
    return clients