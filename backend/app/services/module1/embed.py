"""Module 1 -- concept-text embedding with on-disk caching (Step 2 of build order).

Encodes each concept's *text definition and learning-outcome summary* (not just
the bare concept name -- Section 4.1) with ``all-MiniLM-L6-v2`` (384-dim) and
caches results to ``data/embeddings/`` (``.npz``) so re-embedding is avoided
across runs. This is the ~80MB / <0.5s-per-100-concepts budget row in Table 2.

The cached ``.npz`` files are gitignored; the cache is keyed by a hash of the
embedding-model name plus the concept text so a model or text change silently
invalidates rather than returning stale vectors.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..config import config


def _hash_key(embedding_model: str, texts: Sequence[str]) -> str:
    """Stable cache key derived from model name + exact concept text."""
    joined = "\x1e".join(texts)
    digest = hashlib.sha256(
        f"{embedding_model}\x1f{joined}".encode("utf-8")
    ).hexdigest()[:16]
    return digest


def default_cache_dir() -> str:
    """Resolve the embeddings cache directory relative to this repo.

    Falls back to a temp directory when the repo layout is not present (tests).
    """
    here = os.path.dirname(os.path.abspath(__file__))
    repo = here
    for _ in range(6):
        repo = os.path.dirname(repo)
        if os.path.isdir(os.path.join(repo, "data")):
            return os.path.join(repo, "data", "embeddings")
    import tempfile

    return os.path.join(tempfile.gettempdir(), "lightgap", "embeddings")


@dataclass
class EmbeddingResult:
    """Embeddings plus provenance so consumers can detect stale contexts."""

    vectors: np.ndarray  # (n, d)
    model: str
    cache_key: str


class ConceptEmbedder:
    """Lazily loads the sentence-transformer and caches encoded vectors.

    The sentence-transformer is imported lazily so importing this module (and
    everything that depends on it) stays fast and offline-safe; only the
    ``embed`` call path pulls the (heavy) dependency in.
    """

    def __init__(self, model_name: Optional[str] = None,
                 cache_dir: Optional[str] = None) -> None:
        self.model_name = model_name or config.module1.embedding_model
        self.cache_dir = cache_dir or default_cache_dir()
        self._model = None

    # -- cache plumbing ------------------------------------------------- #
    def _cache_path(self, key: str) -> str:
        os.makedirs(self.cache_dir, exist_ok=True)
        return os.path.join(self.cache_dir, f"{key}.npz")

    def _load_cache(self, key: str) -> Optional[np.ndarray]:
        path = self._cache_path(key)
        if not os.path.exists(path):
            return None
        try:
            with np.load(path, allow_pickle=False) as data:
                return data["vectors"]
        except (KeyError, ValueError, OSError):
            return None

    def _store_cache(self, key: str, vectors: np.ndarray) -> None:
        path = self._cache_path(key)
        np.savez_compressed(path, vectors=vectors)

    # -- embedding ------------------------------------------------------ #
    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, texts: Sequence[str],
              use_cache: bool = True) -> EmbeddingResult:
        """Encode ``texts`` (n concepts) -> (n, d) vectors.

        ``texts[i]`` should be concept i's full definition + learning-outcome
        summary. Caching is keyed on the exact list, so concatenating the same
        concepts in the same order reuses the cache even across processes.
        """
        if not texts:
            return EmbeddingResult(
                vectors=np.zeros((0, config.module1.embedding_dim)),
                model=self.model_name,
                cache_key="",
            )
        key = _hash_key(self.model_name, list(texts))
        if use_cache:
            cached = self._load_cache(key)
            if cached is not None and cached.shape[0] == len(texts):
                return EmbeddingResult(
                    vectors=cached, model=self.model_name, cache_key=key
                )
        model = self._get_model()
        vectors = np.asarray(model.encode(list(texts)), dtype=np.float32)
        if use_cache:
            self._store_cache(key, vectors)
        return EmbeddingResult(
            vectors=vectors, model=self.model_name, cache_key=key
        )

    def embed_many(self, batches: Iterable[Sequence[str]],
                   use_cache: bool = True) -> Dict[Tuple[str, ...], np.ndarray]:
        """Convenience wrapper for embedding several concept sets at once."""
        out: Dict[Tuple[str, ...], np.ndarray] = {}
        for batch in batches:
            result = self.embed(batch, use_cache=use_cache)
            out[tuple(batch)] = result.vectors
        return out


def concept_text(name: str, definition: str = "",
                 outcomes: Sequence[str] = ()) -> str:
    """Canonical per-concept text fed to the embedder.

    Mirrors Section 4.1: the definition and learning-outcome summary are what
    gets encoded, not just the bare name. Callers with only a name still emit
    stable, explicit text so downstream stages behave identically.
    """
    pieces = [name.strip()]
    if definition.strip():
        pieces.append(definition.strip())
    for outcome in outcomes:
        if outcome.strip():
            pieces.append(outcome.strip())
    return " ".join(pieces)


def build_concept_texts(concepts: Iterable[object]) -> List[str]:
    """Map concept records (name/definition/outcomes or plain strings) to text.

    Accepts strings or object-like records with ``name`` / ``definition`` /
    ``outcomes`` attributes, so the same canonical text builder works for the
    gold set, dataset rows, and synthetic test concepts.
    """
    texts: List[str] = []
    for c in concepts:
        if isinstance(c, str):
            texts.append(concept_text(c))
        else:
            name = getattr(c, "name", getattr(c, "id", "?")).strip()
            definition = getattr(c, "definition", "") or ""
            outcomes = getattr(c, "outcomes", ()) or ()
            texts.append(concept_text(name, definition, outcomes))
    return texts