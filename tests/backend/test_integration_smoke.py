"""Integration smoke test -- a tiny end-to-end run through all four modules
and the FastAPI layer, asserting no errors and a schema-valid final response.

This does NOT require a GPU, network, or downloaded data: it uses synthetic
concepts and the deterministic module backends, exactly as the offline unit
suite promises. The full three-track protocol (Section 11) is exercised by
``scripts/run_full_pipeline.py`` once real data/models are available.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.models.schema import GraphPayload
from backend.app.session_store import session_store
from backend.app.services.graph_model import NodeStatus
from backend.app.services.module1.score_pairs import (
    LogisticBaseline,
    directional_feature,
)
from backend.app.services.module3.quiz_engine import (
    QuizEngine,
    DeterministicQuestionGenerator,
    Misconception,
)


@pytest.fixture
def client():
    return TestClient(app)


def _synthetic_concepts():
    rng = np.random.default_rng(7)
    concepts = ["A", "B", "C", "D", "E"]
    embeddings = {c: rng.normal(size=384).astype(np.float32)
                  for c in concepts}
    return concepts, embeddings


def test_full_pipeline_smoke(client):
    """
    1. Create a session.
    2. Build a DAG from synthetic Module-1-style scored edges (Module 2).
    3. Generate + grade a quiz (Module 3) -> exercise remediation (Module 4).
    4. Fetch the graph and assert the final response is schema-valid.
    """
    concepts, embeddings = _synthetic_concepts()

    # Module 1: train a trivial baseline on synthetic pairs and score.
    pairs = [(u, v) for u in concepts for v in concepts if u != v]
    X = np.vstack([directional_feature(embeddings[u], embeddings[v])
                   for u, v in pairs])
    y = np.random.default_rng(3).integers(0, 2, size=X.shape[0])
    model = LogisticBaseline(seed=0).fit(X, y)
    scores = model.score_positive(X).tolist()

    # Create session (Module 2's graph will be attached here).
    resp = client.post("/graph/session", json={
        "goal_concept": "E", "time_budget_minutes": 60,
    })
    assert resp.status_code == 200
    sid = resp.json()["session_id"]

    # Build DAG from the scored pairs.
    resp2 = client.post(f"/graph/{sid}/build", json={
        "node_ids": concepts,
        "pairs": [list(p) for p in pairs],
        "scores": scores,
        "tau_edge": 0.4,  # lenient so some edges survive
    })
    assert resp2.status_code == 200
    # The response must validate against the contract.
    GraphPayload(**resp2.json())

    # Module 3: generate a quiz for one node and grade a wrong (distractor)
    # answer to trigger Module 4 remediation.
    target = next(iter(session_store.get_graph(sid).node_ids()), "A")
    gen_resp = client.post("/quiz/generate", json={
        "session_id": sid, "node_id": target, "concept_text": target,
    })
    assert gen_resp.status_code == 200
    qs = gen_resp.json()
    assert len(qs) == 3

    # Grade option index 1 (prerequisite distractor) -> remediation.
    grade_resp = client.post("/quiz/grade", json={
        "session_id": sid, "node_id": target,
        "stem": qs[0]["stem"], "selected_index": 1,
    })
    assert grade_resp.status_code == 200
    grade = grade_resp.json()

    # Final graph fetch and schema validation.
    final = client.get(f"/graph/{sid}")
    assert final.status_code == 200
    payload = GraphPayload(**final.json())

    # If a remediation node was injected, it must be flagged correctly.
    rem_nodes = [n for n in payload.nodes if n.is_dynamic_remediation]
    if grade.get("attributed_misconception"):
        assert len(rem_nodes) >= 1
        for rm in rem_nodes:
            assert rm.triggering_misconception is not None
    # And its edges must be of remediation_link type.
    for e in payload.edges:
        assert e.type in ("prerequisite", "remediation_link")


def test_health_endpoint(client):
    assert client.get("/health").json() == {"status": "ok"}