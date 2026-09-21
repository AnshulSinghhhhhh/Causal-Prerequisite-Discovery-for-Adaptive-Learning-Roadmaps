"""Unit tests for Module 1 -- directional feature, embedding, DirGCN guards."""

import numpy as np
import pytest

from backend.app.services.module1.score_pairs import (
    directional_feature,
    LogisticBaseline,
    build_candidate_graph,
    score_pairs,
)
from backend.app.services.module1.dirgcn import (
    DirGCN,
    AttentionDirGCN,
    build_edge_index,
    run_dirgcn,
)
from backend.app.services.module1.counterfactual_probe import (
    DeterministicProbeBackend,
    ambiguous_mask,
    counterfactual_probe,
    refine_logits,
    STANDARD_TEMPLATES,
    WITHOUT_TEMPLATES,
)
from backend.app.services.config import config


def test_directional_feature_shape_and_symmetry():
    h_u = np.arange(4, dtype=np.float32)
    h_v = np.arange(4, dtype=np.float32) * 2
    f = directional_feature(h_u, h_v)
    assert f.shape[0] == 4 * 4
    # Asymmetry: f(u,v) != f(v,u) because (h_u - h_v) flips sign.
    f_rev = directional_feature(h_v, h_u)
    assert not np.allclose(f, f_rev)


def test_directional_feature_dimension_mismatch_raises():
    with pytest.raises(ValueError):
        directional_feature(np.zeros(3), np.zeros(4))


def test_logistic_baseline_fit_and_score(synthetic_embeddings):
    X = np.vstack([
        directional_feature(synthetic_embeddings[u], synthetic_embeddings[v])
        for (u, v) in [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")]
    ])
    y = [1, 1, 0, 0]
    model = LogisticBaseline().fit(X, y)
    scores = model.score_positive(X)
    assert scores.shape == (4,)
    assert ((scores >= 0) & (scores <= 1)).all()


def test_build_candidate_graph_creates_topk_edges(synthetic_embeddings):
    X = np.vstack([
        directional_feature(synthetic_embeddings[u], synthetic_embeddings[v])
        for u in synthetic_embeddings for v in synthetic_embeddings if u != v
    ])
    y = np.random.default_rng(1).integers(0, 2, size=X.shape[0])
    model = LogisticBaseline().fit(X, y)
    nodes = list(synthetic_embeddings)
    batch = build_candidate_graph(synthetic_embeddings, model, nodes, top_k=2)
    # Each of the 5 nodes should have <=2 outgoing candidate edges.
    from collections import Counter
    out_degree = Counter(u for u, v in batch.pairs)
    assert all(d <= 2 for d in out_degree.values())
    assert len(batch.pairs) >= 1


def test_build_candidate_graph_empty_raises(synthetic_embeddings):
    model = LogisticBaseline()
    model.fit(np.zeros((2, config.module1.directional_dim), dtype=np.float32),
              [0, 1])
    from backend.app.services.module1.score_pairs import build_candidate_graph
    with pytest.raises(ValueError):
        build_candidate_graph(synthetic_embeddings, model, [], top_k=2)


def test_dirgcn_empty_edge_index_raises():
    model = DirGCN(in_dim=4, hidden=8)
    x = np.random.default_rng(0).normal(size=(4, 4)).astype(np.float32)
    import torch
    empty = torch.empty((2, 0), dtype=torch.long)
    with pytest.raises(ValueError):
        model.encode(torch.as_tensor(x), empty)


def test_dirgcn_attention_variant_runs():
    import torch
    model = AttentionDirGCN(in_dim=4, hidden=8)
    x = np.random.default_rng(0).normal(size=(4, 4)).astype(np.float32)
    ei = build_edge_index([(0, 1), (1, 2), (2, 3)]).numpy()
    out = run_dirgcn(x, ei, [(0, 1), (1, 2)], attention=True)
    assert out.shape == (2,)
    assert ((out >= 0) & (out <= 1)).all()


def test_ambiguous_mask_band():
    scores = np.array([0.2, 0.5, 0.8, 0.4, 0.7])
    mask = ambiguous_mask(scores)
    assert list(mask) == [False, True, False, True, True]


def test_refine_logits_identity_when_d_zero():
    p = np.array([0.9, 0.1])
    refined = refine_logits(p, [0.0, 0.0], alpha=0.5)
    assert np.allclose(refined, p, atol=1e-6)


def test_counterfactual_probe_deterministic_and_asymmetric():
    backend = DeterministicProbeBackend()
    r = counterfactual_probe("A", "B", backend)
    # Deterministic backend => symmetric input yields d_asym ~ 0.
    assert r.d_asym == pytest.approx(0.0, abs=1e-6)
    # Templates are paraphrased (>=3).
    assert len(STANDARD_TEMPLATES) >= 3
    assert len(WITHOUT_TEMPLATES) >= 3