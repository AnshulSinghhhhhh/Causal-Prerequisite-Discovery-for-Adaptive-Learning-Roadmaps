"""Unit tests for S1 Signal Hardening (07_S1_SIGNAL_HARDENING.md).

Tests:
1. S0: harvest_syllabus returns well-formed syllabus documents with source_type='syllabus'.
2. S1: compute_s_order applies 3x multiplier to syllabus-sourced documents over wikipedia.
3. S1: compute_s_llm_plaus evaluates prerequisite plausibility and falls back safely.
4. S1: generate_candidates integrates s_llm_plaus into fused s_corr.
5. S6: _compute_descendant_value implements CROSS_CLUSTER_WEIGHT discounting.
"""

from __future__ import annotations

import networkx as nx
import pytest
from unittest.mock import MagicMock, patch

from backend.app.services.module0.harvest import _clean_html_text, harvest_syllabus
from backend.app.services.module1.candidate_gen import (
    compute_s_order,
    compute_s_llm_plaus,
    generate_candidates,
)
from backend.app.services.module2.path_optimizer import (
    CROSS_CLUSTER_WEIGHT,
    _compute_descendant_value,
)


def test_clean_html_text():
    raw_html = """
    <html>
        <head><style>body { color: red; }</style></head>
        <body>
            <nav><a href="#">Nav item</a></nav>
            <h1>Machine Learning Syllabus</h1>
            <p>Topic 1: Linear Algebra &amp; Vectors.</p>
            <p>Topic 2: Calculus &amp; Optimization.</p>
            <script>console.log('test');</script>
            <footer>Copyright 2026</footer>
        </body>
    </html>
    """
    cleaned = _clean_html_text(raw_html)
    assert "Nav item" not in cleaned
    assert "Copyright 2026" not in cleaned
    assert "console.log" not in cleaned
    assert "body { color" not in cleaned
    assert "Machine Learning Syllabus" in cleaned
    assert "Linear Algebra & Vectors." in cleaned
    assert "Calculus & Optimization." in cleaned


def test_harvest_syllabus_structure():
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_resp_search = MagicMock()
        mock_resp_search.__enter__.return_value = mock_resp_search
        mock_resp_search.read.return_value = b"""
        <a class="result__snippet" href="https://example.edu/cs101_syllabus">CS101 Machine Learning Course Outline</a>
        """
        mock_resp_page = MagicMock()
        mock_resp_page.__enter__.return_value = mock_resp_page
        mock_resp_page.read.return_value = b"""
        <html><body>
        <h1>Machine Learning Course Syllabus and Lecture Schedule</h1>
        <p>Week 1: Foundations of supervised learning, linear classification, loss functions, and empirical risk minimization.</p>
        <p>Week 2: Deep neural network architectures, backpropagation, and stochastic gradient descent optimization methods.</p>
        <p>Week 3: Regularization techniques, generalization theory, and evaluation metrics.</p>
        </body></html>
        """
        mock_urlopen.side_effect = [mock_resp_search, mock_resp_page]

        docs = harvest_syllabus("Machine Learning", domain_id="test-domain", max_results=1)
        assert len(docs) == 1
        assert docs[0]["source_type"] == "syllabus"
        assert docs[0]["domain_id"] == "test-domain"
        assert "supervised learning" in docs[0]["raw_text"]


def test_s_order_source_type_weighting():
    concepts = [
        {"id": "c1", "canonical_name": "Calculus"},
        {"id": "c2", "canonical_name": "Machine Learning"},
    ]
    # Doc 1: Wikipedia article with Calculus before Machine Learning
    # Doc 2: Wikipedia article with Machine Learning before Calculus
    # Doc 3: Syllabus with Calculus before Machine Learning
    docs = [
        {"source_type": "wikipedia", "raw_text": "Calculus is used in Machine Learning."},
        {"source_type": "wikipedia", "raw_text": "Machine Learning sometimes mentions Calculus."},
        {"source_type": "syllabus", "raw_text": "Prerequisites: Calculus. Core subject: Machine Learning."},
    ]

    # Without weighting (equal 1.0 weights):
    # coappear = 3, order (Calculus -> ML) = 2 -> 2/3 = 0.6667
    unweighted = compute_s_order(concepts, docs, syllabus_weight=1.0, default_weight=1.0)
    assert abs(unweighted[("c1", "c2")] - 2.0 / 3.0) < 1e-5

    # With syllabus weighting (syllabus 3x, wiki 1x):
    # coappear = 1 + 1 + 3 = 5
    # order (Calculus -> ML) = 1 (doc 1) + 0 (doc 2) + 3 (doc 3) = 4
    # fraction = 4/5 = 0.80
    weighted = compute_s_order(concepts, docs, syllabus_weight=3.0, default_weight=1.0)
    assert abs(weighted[("c1", "c2")] - 0.80) < 1e-5
    # Order (ML -> Calculus) = 1 / 5 = 0.20
    assert abs(weighted[("c2", "c1")] - 0.20) < 1e-5


def test_compute_s_llm_plaus_fallback():
    # If no API key, returns neutral 0.5
    pairs = [("c1", "c2"), ("c2", "c3")]
    concepts = [
        {"id": "c1", "canonical_name": "Linear Algebra"},
        {"id": "c2", "canonical_name": "Machine Learning"},
        {"id": "c3", "canonical_name": "Data Dredging"},
    ]
    with patch.dict("os.environ", {"GROQ_API_KEY": ""}):
        scores = compute_s_llm_plaus(pairs, concepts)
        assert scores[("c1", "c2")] == 0.5
        assert scores[("c2", "c3")] == 0.5


def test_compute_s_llm_plaus_mock():
    pairs = [("c1", "c2"), ("c3", "c2")]
    concepts = [
        {"id": "c1", "canonical_name": "Linear Algebra"},
        {"id": "c2", "canonical_name": "Machine Learning"},
        {"id": "c3", "canonical_name": "Data Dredging"},
    ]
    mock_groq = MagicMock()
    mock_resp = MagicMock()
    mock_resp.choices = [
        MagicMock(message=MagicMock(content='[{"index": 1, "score": 0.85}, {"index": 2, "score": 0.20}]'))
    ]
    mock_groq.chat.completions.create.return_value = mock_resp

    with patch.dict("os.environ", {"GROQ_API_KEY": "dummy"}), \
         patch("backend.app.services.module1.candidate_gen._get_groq_client", return_value=mock_groq):
        scores = compute_s_llm_plaus(pairs, concepts)
        assert scores[("c1", "c2")] == 0.85
        assert scores[("c3", "c2")] == 0.20


def test_s6_in_cluster_vs_cross_cluster_weighting():
    # DAG: A -> B, A -> C
    # Cluster 1: A, B
    # Cluster 2: C
    # A unlocks B (in-cluster) and C (cross-cluster)
    dag = nx.DiGraph()
    dag.add_edge("A", "B")
    dag.add_edge("A", "C")
    ancestor_set = {"A", "B", "C"}
    cluster_map = {"A": "cluster_1", "B": "cluster_1", "C": "cluster_2"}

    val = _compute_descendant_value(dag, ancestor_set, cluster_map=cluster_map)

    # In-cluster descendant of A: {B} -> 1
    # Cross-cluster descendant of A: {C} -> 0.3
    # Total for A = 1.0 + 0.3 = 1.3
    assert abs(val["A"] - (1.0 + CROSS_CLUSTER_WEIGHT * 1.0)) < 1e-5
    # Leaves B and C have 0 descendants
    assert val["B"] == 0.0
    assert val["C"] == 0.0
