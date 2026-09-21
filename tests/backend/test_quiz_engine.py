"""Unit tests for Module 3 -- content aggregation (caching/index/re-rank) and
quiz engine (misconception-tagged distractors + match threshold)."""

import numpy as np
import pytest

from backend.app.services.module3.content_aggregator import (
    ContentAggregator,
    DeterministicContentClient,
    NumpyVectorIndex,
    ResponseCache,
    _cosine,
)
from backend.app.services.module3.quiz_engine import (
    DeterministicQuestionGenerator,
    Misconception,
    OptionRole,
    QuizEngine,
    attribute_selection,
    match_score,
)
from backend.app.services.config import config


def test_cosine_similarity():
    assert _cosine(np.array([1.0, 0]), np.array([1.0, 0])) == pytest.approx(1.0)
    assert _cosine(np.array([1.0, 0]), np.array([0.0, 1.0])) == pytest.approx(0.0)
    assert _cosine(np.zeros(3), np.zeros(3)) == 0.0


def test_response_cache_ttl():
    cache = ResponseCache(ttl_seconds=1)
    client_items = [DeterministicContentClient("y").search("q")[0]]
    cache.set("y", "q", client_items)
    assert cache.get("y", "q") is not None
    # Force expiry by manipulating the stored timestamp.
    key = list(cache.store)[0]
    cache.store[key] = (cache._clock() - 2.0, client_items)
    assert cache.get("y", "q") is None


def test_numpy_vector_index_query():
    idx = NumpyVectorIndex()
    idx.add("k1", np.array([1.0, 0.0]))
    idx.add("k2", np.array([0.0, 1.0]))
    res = idx.query(np.array([1.0, 0.0]), top_k=2)
    assert res[0][0] == "k1"
    assert res[0][1] > res[1][1]


def test_content_aggregator_reranks(synthetic_embeddings):
    client = DeterministicContentClient("docs", template="about {q}")
    agg = ContentAggregator(clients=[client], embedder=None,
                            cache=ResponseCache(ttl_seconds=0),
                            index=NumpyVectorIndex())
    result = agg.aggregate("A", synthetic_embeddings["A"], query="A", limit=3)
    assert result.node_id == "A"
    assert len(result.items) <= 3
    # All items come from the deterministic client.
    assert all(i.kind == "docs" for i in result.items)


def test_quiz_generation_has_three_roles():
    gen = DeterministicQuestionGenerator()
    qs = gen.generate("A", "linear algebra", prerequisites=["arithmetic"])
    assert len(qs) == 3
    for q in qs:
        roles = [o.role for o in q.options]
        assert roles.count(OptionRole.CORRECT) == 1
        assert roles.count(OptionRole.PREREQUISITE_DISTRACTOR) == 1
        assert roles.count(OptionRole.CONCEPTUAL_DISTRACTOR) == 1


def test_quiz_grade_correct_and_distractor():
    engine = QuizEngine(generator=DeterministicQuestionGenerator())
    m = Misconception.from_text("M_prereq", "matrix dimension mismatch")
    engine.add_misconception(m)
    q = engine.generate("A", "linear algebra",
                        prerequisites=["arithmetic"])[0]
    # Option A correct.
    r = engine.grade(q, 0)
    assert r.is_correct
    # Option B is the prerequisite distractor.
    rb = engine.grade(q, 1)
    assert not rb.is_correct
    assert rb.attributed_misconception is not None


def test_match_score_threshold_attribution():
    m = Misconception.from_text("M_k", "correlation vs causation")
    # A vector very dissimilar to M_k => low cosine => attributed (score<th).
    dissimilar = m.vector * -1.0
    score = match_score(m, dissimilar)
    assert score < config.module3.gamma_misconception
    attributed, s = attribute_selection(
        type("Opt", (), {"misconception_vector": dissimilar})(), m
    )
    assert attributed is True
    assert s == pytest.approx(score)