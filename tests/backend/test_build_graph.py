"""Module 2 correctness tests -- synthetic cycles/redundant edges + ILP."""

import networkx as nx
import pytest

from backend.app.services.module2.build_graph import (
    build_dag,
    filter_by_threshold,
    prune_cycles,
    transitive_reduction_edges,
    audit_collateral_damage,
    graph_from_dag,
)
from backend.app.services.module2.path_optimizer import (
    solve_path,
    hand_solvable_instance,
)


def test_filter_by_threshold_splits():
    kept, dropped = filter_by_threshold(
        [("A", "B"), ("B", "C")], [0.9, 0.4], tau_edge=0.65
    )
    assert kept == [("A", "B", 0.9)]
    assert dropped == [("B", "C", 0.4)]


def test_prune_cycles_removes_lowest_confidence_edge():
    # Cycle A->B->C->A; lowest-confidence edge is C->A (0.2).
    edges = [("A", "B", 0.9), ("B", "C", 0.8), ("C", "A", 0.2)]
    acyclic, pruned = prune_cycles(edges)
    assert ("C", "A", 0.2) in pruned
    # Result is acyclic.
    g = nx.DiGraph()
    g.add_edges_from([(u, v) for u, v, _ in acyclic])
    assert nx.is_directed_acyclic_graph(g)


def test_prune_cycles_handles_self_loop():
    edges = [("A", "A", 0.5), ("A", "B", 0.9)]
    acyclic, pruned = prune_cycles(edges)
    assert ("A", "A", 0.5) in pruned
    g = nx.DiGraph()
    g.add_edges_from([(u, v) for u, v, _ in acyclic])
    assert nx.is_directed_acyclic_graph(g)


def test_transitive_reduction_removes_redundant_hop():
    # A->B, B->C, A->C: direct A->C is redundant.
    edges = [("A", "B", 0.9), ("B", "C", 0.9), ("A", "C", 0.9)]
    kept, dropped = transitive_reduction_edges(edges)
    assert ("A", "C", 0.9) in dropped
    assert ("A", "B", 0.9) in kept and ("B", "C", 0.9) in kept


def test_build_dag_end_to_end_is_dag():
    pairs = [("A", "B"), ("B", "C"), ("A", "C"), ("C", "A"), ("B", "D")]
    scores = [0.9, 0.9, 0.9, 0.2, 0.9]
    result = build_dag(pairs, scores, tau_edge=0.65)
    assert result.is_dag
    g = nx.DiGraph()
    g.add_edges_from([(u, v) for u, v, _ in result.edges])
    assert nx.is_directed_acyclic_graph(g)


def test_audit_collateral_damage_classifies():
    # Gold pairs: A->B (survives), A->C (subsumed via A->B->C), C->A (reverse
    # filtered out under threshold), D->E (absent -> threshold).
    pairs = [("A", "B"), ("B", "C"), ("A", "C"), ("C", "B")]
    scores = [0.9, 0.9, 0.9, 0.3]
    result = build_dag(pairs, scores, tau_edge=0.65)
    gold = [("A", "B"), ("A", "C"), ("C", "B"), ("D", "E")]
    audit = audit_collateral_damage(result, gold)
    # A->B survives.
    assert ("A", "B") in audit.survived
    # A->C is subsumed by A->B->C path (transitive reduction removed it).
    assert ("A", "C") in audit.subsumed_by_transitive_path or \
           ("A", "C") in audit.survived
    # C->B was filtered by threshold (0.3 < 0.65).
    assert ("C", "B") in audit.lost_to_threshold or \
           ("C", "B") in audit.lost_to_cycle_pruning
    # D->E never existed -> threshold.
    assert ("D", "E") in audit.lost_to_threshold


def test_graph_from_dag_materializes():
    pairs = [("A", "B"), ("B", "C")]
    result = build_dag(pairs, [0.9, 0.9])
    g = graph_from_dag(result)
    assert set(g.node_ids()) == {"A", "B", "C"}
    assert g.has_edge("A", "B") and g.has_edge("B", "C")


def test_solve_path_hand_solvable():
    nodes, prereq, importance, time, budget = hand_solvable_instance()
    sol = solve_path(nodes, prereq, importance, time, budget)
    assert sol.feasible
    # Optimal subset: A, B, C (value 6, time 6); D unreachable (needs 10).
    assert set(sol.selected) == {"A", "B", "C"}
    assert sol.objective == pytest.approx(6.0)
    assert sol.total_time <= budget


def test_solve_path_respects_budget_and_precedence():
    nodes, prereq, importance, time, budget = hand_solvable_instance()
    sol = solve_path(nodes, prereq, importance, time, 4.0)
    # Budget 4: A(1)+C(3)=4 value 4, or A+B=3 value 3. Best is A+C=4.
    selected = set(sol.selected)
    # A must precede C -> if C selected, A selected.
    if "C" in selected:
        assert "A" in selected
    assert sol.total_time <= 4.0


def test_solve_path_infeasible_goal():
    nodes, prereq, importance, time, budget = hand_solvable_instance()
    sol = solve_path(nodes, prereq, importance, time, budget, goal="D")
    assert not sol.feasible


def test_solve_path_goal_reachable_with_bigger_budget():
    nodes, prereq, importance, time, _ = hand_solvable_instance()
    sol = solve_path(nodes, prereq, importance, time, 10.0, goal="D")
    assert sol.feasible
    assert "D" in sol.selected
    assert "A" in sol.selected and "B" in sol.selected and "C" in sol.selected