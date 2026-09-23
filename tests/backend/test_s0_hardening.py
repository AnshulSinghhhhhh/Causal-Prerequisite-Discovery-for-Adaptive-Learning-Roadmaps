"""Unit tests for S0 / S6 hardening (06_S0_S6_HARDENING.md).

Tests:
  A1: Strip reference sections in harvest.py
  A2: POS well-formedness filter in extract.py
  A3: Normalization rules in canonicalize.py
  A4: Union-Find transitive merge in canonicalize.py
  B1: Descendant-count value function in path_optimizer.py
"""

import pytest
import networkx as nx

from backend.app.services.module0.harvest import _strip_reference_sections
from backend.app.services.module0.extract import is_wellformed, extract_noun_phrases
from backend.app.services.module0.canonicalize import normalize, UnionFind, deduplicate_concepts
from backend.app.services.module2.path_optimizer import _compute_descendant_value


# ---------------------------------------------------------------------------
# A1: Reference section stripping
# ---------------------------------------------------------------------------

def test_strip_reference_sections():
    doc_with_refs = (
        "Machine learning is a field of study.\n"
        "It involves neural networks and decision trees.\n\n"
        "== References ==\n"
        "1. Smith, J. (2020). Stony Brook Collected Algorithms.\n"
        "2. Doe, A. (2021). Intro to AI.\n"
    )
    stripped = _strip_reference_sections(doc_with_refs)
    assert "Machine learning is a field of study." in stripped
    assert "neural networks" in stripped
    assert "References" not in stripped
    assert "Stony Brook" not in stripped

    doc_with_see_also = (
        "Linear regression models continuous variables.\n\n"
        "=== See also ===\n"
        "* Logistic regression\n"
        "* Polynomial regression\n"
    )
    stripped_see = _strip_reference_sections(doc_with_see_also)
    assert "Linear regression models" in stripped_see
    assert "See also" not in stripped_see
    assert "Logistic regression" not in stripped_see

    doc_clean = "Deep learning uses multilayer artificial neural networks."
    assert _strip_reference_sections(doc_clean) == doc_clean


# ---------------------------------------------------------------------------
# A2: POS well-formedness filter
# ---------------------------------------------------------------------------

def test_pos_wellformedness():
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        pytest.skip("spaCy en_core_web_sm not available")

    # Valid noun phrases
    doc1 = nlp("neural networks")
    span1 = doc1[:]
    assert is_wellformed(span1) is True

    doc2 = nlp("gradient descent")
    span2 = doc2[:]
    assert is_wellformed(span2) is True

    # Legitimate gerund / participle noun phrases (Issue 3)
    for phrase in ["supervised learning", "reinforcement learning", "deep learning", "gradient descent", "machine learning"]:
        doc = nlp(phrase)
        assert is_wellformed(doc[:]) is True, f"Expected {phrase} to be well-formed"

    # Spans with finite verbs or clause verbs (Issue 3 / A2 hardening)
    for clause_span in [
        "bacteriochlorophyll conduct photosynthesis",
        "bacteriochlorophyll conducts photosynthesis",
        "bacteria conduct photosynthesis",
        "plants synthesize glucose",
        "algorithms learn patterns",
    ]:
        doc = nlp(clause_span)
        assert is_wellformed(doc[:]) is False, f"Expected {clause_span} to be rejected by finite verb filter"

    # Starting with coordinating conjunction (e.g. "and classification rule")
    doc3 = nlp("and classification rule")
    span3 = doc3[:]
    assert is_wellformed(span3) is False

    # Starting with determiner alone (e.g. "a binary search")
    doc4 = nlp("a binary search")
    span4 = doc4[:]
    assert is_wellformed(span4) is False

    # Starting with preposition (e.g. "in neural networks")
    doc5 = nlp("in neural networks")
    span5 = doc5[:]
    assert is_wellformed(span5) is False


# ---------------------------------------------------------------------------
# A3: Normalization rules
# ---------------------------------------------------------------------------

def test_normalization():
    # Lowercase & strip leading determiners
    assert normalize("A Neural Network") == "neural network"
    assert normalize("The Support Vector Machine") == "support vector machine"
    assert normalize("an activation function") == "activation function"

    # Strip possessives
    assert normalize("Bayes' theorem") == "bayes theorem"
    assert normalize("Student's t-distribution") == "student t-distribution"

    # Collapse whitespace
    assert normalize("  convolutional   neural    network  ") == "convolutional neural network"

    # Plural to singular collapse on head noun
    # "Neural networks" -> "neural network"
    assert normalize("Neural networks") == "neural network"
    assert normalize("Decision trees") == "decision tree"


# ---------------------------------------------------------------------------
# A4: Union-find & Transitive merge
# ---------------------------------------------------------------------------

def test_union_find_transitivity():
    uf = UnionFind(4)
    # A (0) merges with B (1)
    uf.union(0, 1)
    # B (1) merges with C (2)
    uf.union(1, 2)

    # All three must share the same root
    assert uf.find(0) == uf.find(1) == uf.find(2)
    # D (3) is separate
    assert uf.find(3) != uf.find(0)


def test_deduplicate_concepts_exact_and_plural():
    raw = [
        {"name": "a neural network", "definition": "A network of artificial neurons."},
        {"name": "Neural networks", "definition": "Computational models inspired by biological neural networks."},
        {"name": "neural network", "definition": ""},
        {"name": "support vector machines", "definition": "Supervised learning models with associated learning algorithms."},
    ]

    canonical, aliases = deduplicate_concepts(raw, similarity_threshold=0.85)

    # "a neural network", "Neural networks", "neural network" should all merge into 1 canonical concept
    canon_names = [c["name"].lower() for c in canonical]
    assert any("neural" in name for name in canon_names)
    assert len(canonical) == 2  # neural network + support vector machines

    # Aliases should record the merged names
    alias_map = {alias: canon for canon, alias in aliases}
    assert "Neural networks" in alias_map or "a neural network" in alias_map


# ---------------------------------------------------------------------------
# B1: Descendant-count value function
# ---------------------------------------------------------------------------

def test_descendant_value_computation():
    """
    DAG:
       A -> B -> D
       A -> C -> D
    Goal is D.
    Ancestors of D: {A, B, C, D}.
    Descendants within ancestor set:
       A unlocks B, C, D -> 3
       B unlocks D       -> 1
       C unlocks D       -> 1
       D unlocks nothing -> 0
    """
    dag = nx.DiGraph()
    dag.add_edges_from([("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")])
    ancestors = {"A", "B", "C", "D"}

    val = _compute_descendant_value(dag, ancestors)
    assert val["A"] == 3.0
    assert val["B"] == 1.0
    assert val["C"] == 1.0
    assert val["D"] == 0.0


# ---------------------------------------------------------------------------
# S0 Structural Filters Regression Tests (Part B.1)
# ---------------------------------------------------------------------------

def test_numeric_fragment_filter_regression():
    from backend.app.services.module0.extract import is_numeric_fragment

    # Confirmed noise artifacts from Photosynthesis output
    assert is_numeric_fragment("0.1% to 8%") is True
    assert is_numeric_fragment("3–6%") is True
    assert is_numeric_fragment("3-6%") is True

    # Other numeric and percentage fragments
    assert is_numeric_fragment("10-20%") is True
    assert is_numeric_fragment("50%") is True
    assert is_numeric_fragment("1.5 to 3.0") is True
    assert is_numeric_fragment("0.1 to 0.8") is True
    assert is_numeric_fragment("12345") is True

    # Legitimate concepts containing numbers/symbols must NOT be dropped
    assert is_numeric_fragment("C4 photosynthesis") is False
    assert is_numeric_fragment("k-nearest neighbors") is False
    assert is_numeric_fragment("3D convolutional network") is False
    assert is_numeric_fragment("F1 score") is False
    assert is_numeric_fragment("supervised learning") is False


def test_heading_metatoken_filter_regression():
    from backend.app.services.module0.extract import is_heading_or_metatoken

    # Confirmed heading/meta-token artifacts from Photosynthesis and Machine Learning outputs
    assert is_heading_or_metatoken("QUESTIONS") is True
    assert is_heading_or_metatoken("Introduction") is True
    assert is_heading_or_metatoken("problems") is True
    assert is_heading_or_metatoken("difficulty") is True
    assert is_heading_or_metatoken("study") is True
    assert is_heading_or_metatoken("types") is True
    assert is_heading_or_metatoken("some fields") is True

    # Common syllabus / textbook structural headings
    assert is_heading_or_metatoken("Syllabus") is True
    assert is_heading_or_metatoken("EXERCISES") is True
    assert is_heading_or_metatoken("Chapter 1") is True
    assert is_heading_or_metatoken("Module 3") is True
    assert is_heading_or_metatoken("Homework") is True
    assert is_heading_or_metatoken("Grading") is True

    # Legitimate concepts must NOT be dropped
    assert is_heading_or_metatoken("machine learning") is False
    assert is_heading_or_metatoken("neural network") is False
    assert is_heading_or_metatoken("photosynthesis") is False
    assert is_heading_or_metatoken("gradient descent") is False
    assert is_heading_or_metatoken("Statistical Learning Theory") is False


def test_structural_filters_in_deduplicate_concepts():
    raw = [
        {"name": "0.1% to 8%", "definition": "Photosynthetic efficiency."},
        {"name": "QUESTIONS", "definition": "Review questions."},
        {"name": "Introduction", "definition": "Course overview."},
        {"name": "gradient descent", "definition": "Optimization algorithm."},
        {"name": "Gradient descent", "definition": "First-order iterative optimization."},
    ]
    canonical, aliases = deduplicate_concepts(raw, similarity_threshold=0.85)

    names = [c["name"] for c in canonical]
    assert "0.1% to 8%" not in names
    assert "QUESTIONS" not in names
    assert "Introduction" not in names
    assert len(canonical) == 1
    assert "gradient descent" in canonical[0]["name"].lower()
