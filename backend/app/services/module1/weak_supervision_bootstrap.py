"""Module 1 -- weak-supervision bootstrap (documented fallback path).

For a target domain lacking labeled prerequisite data (Section 4.1), the paper
specifies a weak-supervision bootstrap rather than training from zero labels:

    1. Derive *noisy* pretraining labels from textbook chapter order and
       citation/hyperlink asymmetry (e.g. Wikipedia "See also" link direction).
    2. Fine-tune the stage-1 classifier on those noisy labels.
    3. Optionally refit on a small gold set that does exist for the domain.

This module is a first-class path, not an afterthought: it is imported by the
pipeline when a domain has no labeled pairs, and its noisy-label generation is
pure and unit-testable (chapter order + a directed adjacency of citations).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np

from .score_pairs import LogisticBaseline, directional_feature


@dataclass
class NoisyPair:
    """A weakly-labeled directed pair with an optional weight."""

    source: str
    target: str
    weight: float = 1.0


def labels_from_chapter_order(
    chapter_concepts: Sequence[Sequence[str]],
) -> List[NoisyPair]:
    """Concepts appearing earlier in a chapter imply later concepts.

    Only *within-chapter* ordering is used; cross-chapter inference is left to
    the caller so ordering strength can be controlled. Weight decays with
    distance to reflect lower confidence in far-apart pairs.
    """
    pairs: List[NoisyPair] = []
    for chapter in chapter_concepts:
        chapter = list(chapter)
        for i, u in enumerate(chapter):
            for j, v in enumerate(chapter[i + 1:], start=i + 1):
                weight = 1.0 / (j - i)  # nearer pairs weighted higher
                pairs.append(NoisyPair(u, v, weight))
    return pairs


def labels_from_link_asymmetry(
    links: Sequence[Tuple[str, str]],
    node_set: Sequence[str],
) -> List[NoisyPair]:
    """Directional hints from citation/hyperlink asymmetry.

    A "See also" / citation edge A -> B is treated as weak evidence that A
    precedes B. Self-loops and edges outside ``node_set`` are dropped.
    """
    allowed: Set[str] = set(node_set)
    pairs: List[NoisyPair] = []
    seen: Set[Tuple[str, str]] = set()
    for a, b in links:
        if a == b or a not in allowed or b not in allowed:
            continue
        if (a, b) in seen:
            continue
        seen.add((a, b))
        pairs.append(NoisyPair(a, b, 1.0))
    return pairs


def bootstrap_train(
    embeddings: Dict[str, np.ndarray],
    noisy_pairs: Sequence[NoisyPair],
    negatives_ratio: float = 1.0,
    seed: int = 42,
) -> LogisticBaseline:
    """Fit the stage-1 baseline on noisy (weak) labels.

    ``negatives_ratio`` controls how many random non-linked pairs are sampled
    as negatives per positive, so the classifier sees a balanced problem rather
    than only positive directional hints.
    """
    rng = np.random.default_rng(seed)
    nodes = list(embeddings.keys())
    if not nodes:
        raise ValueError("bootstrap_train: no nodes to embed")

    # Positive features from the noisy pairs.
    positive: List[Tuple[str, str]] = [(p.source, p.target) for p in noisy_pairs
                                       if p.source in embeddings
                                       and p.target in embeddings]
    # Negative samples: random directed pairs not in the positive set.
    pos_set = set(positive)
    negatives: List[Tuple[str, str]] = []
    n_needed = int(len(positive) * negatives_ratio)
    attempts = 0
    while len(negatives) < n_needed and attempts < n_needed * 50:
        u = nodes[rng.integers(len(nodes))]
        v = nodes[rng.integers(len(nodes))]
        if u != v and (u, v) not in pos_set and (u, v) not in negatives:
            negatives.append((u, v))
        attempts += 1

    all_pairs = positive + negatives
    y = [1] * len(positive) + [0] * len(negatives)
    X = np.vstack([directional_feature(embeddings[u], embeddings[v])
                   for u, v in all_pairs])
    model = LogisticBaseline(seed=seed)
    model.fit(X, y)
    return model


def fine_tune_on_gold(
    model: LogisticBaseline,
    embeddings: Dict[str, np.ndarray],
    gold_pairs: Sequence[Tuple[str, str, int]],
) -> LogisticBaseline:
    """Re-fit (fine-tune) the baseline on a small domain gold set.

    ``gold_pairs`` are (u, v, label) triples with label in {0, 1}. This is the
    second half of the paper's bootstrap: noisy pretrain, then gold refit.
    """
    pairs = [(u, v) for (u, v, _) in gold_pairs]
    y = [int(lbl) for (_, _, lbl) in gold_pairs]
    X = model.feature_matrix(embeddings, pairs)
    model.fit(X, y)
    return model