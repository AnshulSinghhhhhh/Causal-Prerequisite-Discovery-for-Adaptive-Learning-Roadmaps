"""Module 1 pipeline orchestration tests."""

import numpy as np
import pytest

from backend.app.services.module1.pipeline import Module1Pipeline
from backend.app.services.module1.score_pairs import (
    LogisticBaseline,
    directional_feature,
)
from backend.app.services.module1.counterfactual_probe import (
    DeterministicProbeBackend,
)


def _fit_baseline(embeddings):
    nodes = list(embeddings)
    pairs = [(u, v) for u in nodes for v in nodes if u != v]
    X = np.vstack([directional_feature(embeddings[u], embeddings[v])
                   for u, v in pairs])
    y = np.random.default_rng(5).integers(0, 2, size=X.shape[0])
    return LogisticBaseline(seed=0).fit(X, y), pairs


def test_pipeline_run_requires_stage1():
    rng = np.random.default_rng(0)
    embeddings = {c: rng.normal(size=384).astype(np.float32)
                  for c in ["A", "B", "C"]}
    pipeline = Module1Pipeline()
    with pytest.raises(RuntimeError):
        pipeline.run(embeddings, [("A", "B")])


def test_pipeline_run_stage1_only(synthetic_embeddings):
    model, pairs = _fit_baseline(synthetic_embeddings)
    pipeline = Module1Pipeline(stage1=model)
    out = pipeline.run(synthetic_embeddings, pairs[:4])
    assert len(out.pairs) == 4
    assert out.scores.shape == (4,)
    assert out.stage2_scores is None  # no DirGCN wired


def test_pipeline_run_with_probe_stage3(synthetic_embeddings):
    model, pairs = _fit_baseline(synthetic_embeddings)
    backend = DeterministicProbeBackend()
    pipeline = Module1Pipeline(stage1=model, probe_backend=backend)
    out = pipeline.run(synthetic_embeddings, pairs[:4])
    # Probe refines scores but leaves them in (0, 1).
    assert ((out.scores > 0) & (out.scores < 1)).all()


def test_pipeline_edge_records(synthetic_embeddings):
    model, pairs = _fit_baseline(synthetic_embeddings)
    pipeline = Module1Pipeline(stage1=model)
    out = pipeline.run(synthetic_embeddings, pairs[:4])
    records = out.edge_records()
    assert len(records) == 4
    assert set(records[0]) == {"source", "target", "confidence"}