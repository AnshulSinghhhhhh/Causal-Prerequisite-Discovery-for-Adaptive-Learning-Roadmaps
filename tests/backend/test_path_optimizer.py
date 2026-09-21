"""Unit tests for the path optimizer (Module 2 ILP)."""

import pytest

from backend.app.services.module2.path_optimizer import (
    solve_path,
    hand_solvable_instance,
)


def test_solve_path_requires_all_inputs():
    nodes, prereq, importance, time, budget = hand_solvable_instance()
    with pytest.raises(ValueError):
        solve_path(nodes, prereq, {k: v for k, v in importance.items()
                                   if k != "A"}, time, budget)
    with pytest.raises(ValueError):
        solve_path(nodes, prereq, importance,
                   {k: v for k, v in time.items() if k != "A"}, budget)


def test_solve_path_rejects_unknown_prereq_node():
    nodes, _, importance, time, budget = hand_solvable_instance()
    with pytest.raises(ValueError):
        solve_path(nodes, [("A", "Z")], importance, time, budget)


def test_solve_path_nonpositive_budget():
    nodes, prereq, importance, time, _ = hand_solvable_instance()
    sol = solve_path(nodes, prereq, importance, time, 0.0)
    assert not sol.feasible
    assert sol.selected == []


def test_solve_path_empty_nodes():
    with pytest.raises(ValueError):
        solve_path([], [], {}, {}, 10.0)


def test_hand_solvable_instance():
    nodes, prereq, importance, time, budget = hand_solvable_instance()
    sol = solve_path(nodes, prereq, importance, time, budget)
    assert sol.feasible
    assert set(sol.selected) == {"A", "B", "C"}
    assert sol.total_time == 6.0
    assert sol.objective == 6.0


def test_b1_descendant_value_weighting():
    """Verify foundational prerequisites receive higher weight under B1."""
    import networkx as nx
    from backend.app.services.module2.path_optimizer import _compute_descendant_value

    # Graph: Foundation -> Intermediate -> Advanced -> Goal
    # Leaf trivia also connects to Advanced, but unlocks nothing
    g = nx.DiGraph()
    g.add_edges_from([
        ("foundational", "intermediate"),
        ("intermediate", "advanced"),
        ("advanced", "goal"),
        ("trivia", "advanced"),
    ])
    ancestors = {"foundational", "intermediate", "advanced", "trivia", "goal"}
    val = _compute_descendant_value(g, ancestors)

    assert val["foundational"] > val["intermediate"]
    assert val["intermediate"] > val["advanced"]
    assert val["advanced"] > val["goal"]
    assert val["foundational"] > val["trivia"]
    assert val["goal"] == 0.0