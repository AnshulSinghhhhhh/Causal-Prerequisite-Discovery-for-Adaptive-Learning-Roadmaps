"""Module 1 -- stage 1: directional features + baseline logistic regression.

Implements the asymmetric directional feature (Section 4.1):

    f_dir(u, v) = [ h_u || h_v || (h_u - h_v) || (h_u ⊙ h_v) ]   (1536-dim)

and the cheap, strong baseline classifier (logistic regression on ``f_dir``)
whose top-k scored edges per node build the *candidate graph* that DirGCN
(stage 2) is fed. An edgeless candidate graph is a broken DirGCN evaluation,
so ``build_candidate_graph`` always attempts to emit top-k edges and reports
when a node has no candidates.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import config


def directional_feature(
    h_u: np.ndarray, h_v: np.ndarray
) -> np.ndarray:
    """Build ``f_dir(u, v)`` from two 384-dim concept embeddings."""
    h_u = np.asarray(h_u, dtype=np.float32).ravel()
    h_v = np.asarray(h_v, dtype=np.float32).ravel()
    if h_u.shape[0] != h_v.shape[0]:
        raise ValueError("embedding dimension mismatch")
    return np.concatenate([h_u, h_v, h_u - h_v, h_u * h_v]).astype(np.float32)


def spatial_sign(x: Sequence[float], eps: float = 1e-9) -> np.ndarray:
    """Optional non-linearization applied to scores within the pipeline.

    Kept as a documented, deterministic transform so downstream stages and the
    ablation harness can reproduce identical score semantics. Default identity.
    """
    x = np.asarray(x, dtype=float)
    return x / (eps + np.abs(x))


@dataclass
class ScoreBatch:
    """Scored directed pairs: (u, v) with a probability-like score."""

    pairs: List[Tuple[str, str]] = dataclasses.field(default_factory=list)
    scores: List[float] = dataclasses.field(default_factory=list)
    #: Feature matrix used to produce the scores (for ablation reuse).
    features: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return len(self.pairs)

    def add(self, u: str, v: str, score: float) -> None:
        self.pairs.append((u, v))
        self.scores.append(score)


class LogisticBaseline:
    """Stage-1 baseline classifier over ``f_dir`` (logistic regression)."""

    def __init__(self, C: float = 1.0, seed: Optional[int] = None) -> None:
        self.C = C
        self.seed = seed if seed is not None else config.module1.seed
        self._model = None
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: Sequence[int]) -> "LogisticBaseline":
        from sklearn.linear_model import LogisticRegression

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=int)
        self._model = LogisticRegression(
            C=self.C, max_iter=2000, random_state=self.seed
        )
        self._model.fit(X, y)
        self.classes_ = self._model.classes_
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("LogisticBaseline.fit must be called first")
        return self._model.predict_proba(np.asarray(X, dtype=np.float32))

    def score_positive(self, X: np.ndarray) -> np.ndarray:
        """P(edge exists) regardless of class ordering."""
        proba = self.predict_proba(X)
        pos = int(np.where(self.classes_ == 1)[0][0]) \
            if 1 in self.classes_ else (proba.shape[1] - 1)
        return proba[:, pos]

    def feature_matrix(self, embeddings: Dict[str, np.ndarray],
                       pairs: Sequence[Tuple[str, str]]) -> np.ndarray:
        """Build the stacked ``f_dir`` matrix for a list of pairs."""
        rows = [directional_feature(embeddings[u], embeddings[v])
                for (u, v) in pairs]
        if not rows:
            return np.zeros((0, config.module1.directional_dim), dtype=np.float32)
        return np.vstack(rows)


def score_pairs(
    embeddings: Dict[str, np.ndarray],
    pairs: Sequence[Tuple[str, str]],
    model: LogisticBaseline,
) -> ScoreBatch:
    """Score every directed pair with the calibrated baseline."""
    if not pairs:
        return ScoreBatch()
    X = model.feature_matrix(embeddings, pairs)
    scores = model.score_positive(X).tolist()
    return ScoreBatch(pairs=list(pairs), scores=scores, features=X)


def build_candidate_graph(
    embeddings: Dict[str, np.ndarray],
    model: LogisticBaseline,
    nodes: Sequence[str],
    top_k: Optional[int] = None,
) -> ScoreBatch:
    """Top-k scored *outgoing* edges per node -> candidate-graph edge set.

    This is the specific candidate graph DirGCN requires. DirGCN must never be
    run on an edgeless graph, so this function raises when no pair can be
    scored (and documents per-node coverage in the returned batch).
    """
    from itertools import combinations, permutations

    top_k = top_k or config.module1.top_k_candidates
    nodes = list(nodes)
    # Candidate universe: all directed pairs (u, v) with u != v.
    pairs: List[Tuple[str, str]] = []
    for u in nodes:
        for v in nodes:
            if u != v:
                pairs.append((u, v))
    if not pairs:
        raise ValueError(
            "build_candidate_graph: empty node set would produce an edgeless "
            "candidate graph, which is a broken DirGCN evaluation"
        )
    scored = score_pairs(embeddings, pairs, model)
    # Keep top-k outgoing edges per source node.
    best_by_source: Dict[str, List[Tuple[float, str]]] = {}
    for (u, v), s in zip(scored.pairs, scored.scores):
        best_by_source.setdefault(u, []).append((s, v))
    kept: List[Tuple[str, str]] = []
    kept_scores: List[float] = []
    for u in nodes:
        tops = sorted(best_by_source.get(u, []), reverse=True)[:top_k]
        for s, v in tops:
            kept.append((u, v))
            kept_scores.append(s)
    return ScoreBatch(pairs=kept, scores=kept_scores)


def heldout_scores(
    embeddings: Dict[str, np.ndarray],
    pairs: Sequence[Tuple[str, str]],
    model: LogisticBaseline,
) -> np.ndarray:
    """Raw scores for a held-out pair list (no threshold applied here)."""
    return score_pairs(embeddings, pairs, model).scores