"""Module 2 -- precedence-constrained knapsack path optimizer (ILP).

Section 4.2 formalization:

    maximize    sum_v  x_v * Importance(v)
    subject to  sum_v  x_v * EstimatedTime(v) <= T_budget
                x_v <= x_u            for every (u, v) in E_prereq
                x_v in {0, 1}

Solved *exactly* with ``scipy.optimize.milp`` (the paper explicitly allows
SciPy's MILP as an alternative to PuLP). Fast at course scale (tens to low
hundreds of nodes; < 0.1s CPU per Table 2), so no heuristic or RL solver is
needed. ``Importance`` and ``EstimatedTime`` are *inputs*, not derived here --
estimation is a separate future concern (Section 9 limitations).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class PathSolution:
    """Optimal (or infeasible) study-path selection."""

    selected: List[str]
    objective: float
    total_time: float
    feasible: bool
    message: str = ""

    def to_dict(self) -> dict:
        return {
            "selected": self.selected,
            "objective": self.objective,
            "total_time": self.total_time,
            "feasible": self.feasible,
            "message": self.message,
        }


def solve_path(
    nodes: Sequence[str],
    prerequisites: Sequence[Tuple[str, str]],
    importance: Dict[str, float],
    time: Dict[str, float],
    budget: float,
    goal: Optional[str] = None,
) -> PathSolution:
    """Solve the precedence-constrained knapsack ILP exactly.

    ``prerequisites`` are directed edges (u -> v) meaning "u must be studied
    before v". ``goal`` is optional; when given it is FORCED into the solution
    (x_goal = 1), because a learner's goal concept must be reached, which
    transitively pulls in its prerequisites via the precedence constraints.
    """
    nodes = list(nodes)
    if not nodes:
        raise ValueError("solve_path: no nodes")
    if budget <= 0:
        return PathSolution([], 0.0, 0.0, feasible=False,
                            message="non-positive budget")

    n = len(nodes)
    idx = {node: i for i, node in enumerate(nodes)}
    for u, v in prerequisites:
        if u not in idx or v not in idx:
            raise ValueError(f"prerequisite references unknown node: {(u, v)}")
    for node in nodes:
        if node not in importance:
            raise ValueError(f"missing importance for {node!r}")
        if node not in time:
            raise ValueError(f"missing time for {node!r}")

    # scipy milp minimizes c^T x with bounds and linear constraints.
    from scipy.optimize import LinearConstraint, milp, Bounds

    c = np.array([-float(importance[node]) for node in nodes])

    # Build A_ub x <= b_ub: time constraint + precedence constraints.
    rows: List[Tuple[List[float], float]] = []
    # Time budget: sum_v x_v * time(v) <= budget
    rows.append(([float(time[node]) for node in nodes], float(budget)))
    # Precedence: x_v - x_u <= 0  =>  x_v <= x_u
    for (u, v) in prerequisites:
        coeff = [0.0] * n
        coeff[idx[v]] = 1.0
        coeff[idx[u]] = -1.0
        rows.append((coeff, 0.0))

    A = np.array([r[0] for r in rows], dtype=float) if rows else None
    ub = np.array([r[1] for r in rows], dtype=float)

    integrality = np.ones(n)  # binary: 1 = integer
    bounds = Bounds(0, 1)

    constraints = []
    if A is not None and A.size:
        constraints.append(LinearConstraint(A, -np.inf, ub))

    # Goal forcing: x_goal == 1 (add as an equality constraint).
    if goal is not None:
        if goal not in idx:
            raise ValueError(f"unknown goal concept: {goal!r}")
        g_idx = idx[goal]
        row = np.zeros(n)
        row[g_idx] = 1.0
        constraints.append(LinearConstraint(row, 1.0, 1.0))

    res = milp(
        c=c,
        integrality=integrality,
        bounds=bounds,
        constraints=constraints,
    )

    if not res.success:
        return PathSolution(
            selected=[], objective=0.0, total_time=0.0,
            feasible=False, message=res.message,
        )

    x = np.round(res.x).astype(int)
    selected = [node for node, xi in zip(nodes, x) if xi == 1]
    total_time = float(sum(time[node] for node in selected))
    objective = float(sum(importance[node] for node in selected))
    # Validate feasibility flag from the solver's own success.
    feasible = bool(res.success) and total_time <= budget + 1e-9
    return PathSolution(
        selected=selected, objective=objective, total_time=total_time,
        feasible=feasible, message=res.message,
    )


def hand_solvable_instance() -> Tuple[
    List[str], List[Tuple[str, str]], Dict[str, float], Dict[str, float], float
]:
    """A small, hand-checkable instance used by the unit test.

    Concepts: A (prereq of B and C), B (prereq of D), C (prereq of D), D.
    Importance = [1, 2, 3, 4]; time = [1, 2, 3, 4]; budget = 6.
    Optimal: without a goal, pick C (3) + A (1) + B(2) = time 6 value 6
    (D is unreachable within budget because D needs A+B+C = time 10).
    """
    nodes = ["A", "B", "C", "D"]
    prereq = [("A", "B"), ("B", "D"), ("C", "D"), ("A", "C")]
    importance = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0}
    time = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0}
    return nodes, prereq, importance, time, 6.0


class InfeasibleBudgetError(ValueError):
    """Raised by helpers that require a feasible solution."""


# Structural default placeholder for S6 path planning (07_S1_SIGNAL_HARDENING.md Part C).
# Note: Unlike tau_edge or S1 signal weights, CROSS_CLUSTER_WEIGHT is not loss-calibrated
# due to lack of ground-truth pedagogical ordering datasets; 0.3 is a structural default
# that discounts diffuse cross-cluster connections relative to in-cluster prerequisites.
CROSS_CLUSTER_WEIGHT: float = 0.3


def _compute_descendant_value(
    dag: "nx.DiGraph",
    ancestor_set: set,
    cluster_map: Optional[Dict[str, Any]] = None,
    cross_cluster_weight: float = CROSS_CLUSTER_WEIGHT,
) -> Dict[str, float]:
    """B1/C: Value = in-cluster unlock count + discounted cross-cluster count.

    Nodes that are deeply load-bearing within their own S5 cluster rank higher,
    while diffuse cross-cluster connections (caveats, broad associations)
    receive a structural discount (CROSS_CLUSTER_WEIGHT = 0.3).
    """
    import networkx as nx

    cluster_map = cluster_map or {}
    value: Dict[str, float] = {}
    subdag = dag.subgraph(ancestor_set)
    for n in ancestor_set:
        # descendants of n within the ancestor subgraph (excludes self)
        desc = nx.descendants(subdag, n) & ancestor_set
        n_cluster = cluster_map.get(n)
        if n_cluster is not None:
            in_cluster = {m for m in desc if cluster_map.get(m) == n_cluster}
            cross_cluster = desc - in_cluster
            value[n] = float(len(in_cluster) + cross_cluster_weight * len(cross_cluster))
        else:
            value[n] = float(len(desc))
    return value


def optimize_path(
    snapshot_id: str,
    goal_node_id: Optional[str] = None,
    mastery: Optional[Dict[str, float]] = None,
    time_budget_minutes: float = 300.0,
) -> List[str]:
    """Solve optimal learning path for a graph snapshot using the ILP solver.

    B1: value[n] = in-cluster unlock count + CROSS_CLUSTER_WEIGHT * cross-cluster count.
    B2: cost[n]  = concepts.est_minutes (default 15, from DB).
    """
    import networkx as nx
    from ...db import table, fetch_all

    # Query admitted dag edges
    edges_res = fetch_all(table("dag_edges").select("src_id, dst_id").eq(
        "snapshot_id", snapshot_id
    ).eq("admitted", True))
    prereqs = [(e["src_id"], e["dst_id"]) for e in edges_res]

    # Build full DAG
    g = nx.DiGraph()
    for u, v in prereqs:
        g.add_edge(u, v)
    if goal_node_id:
        g.add_node(goal_node_id)

    nodes = list(g.nodes)
    if not nodes:
        return []

    mastery = mastery or {}
    MASTERY_THRESHOLD = 0.8

    # Query cluster_ids for in-cluster vs cross-cluster value weighting (Part C)
    cluster_map: Dict[str, Any] = {}
    try:
        all_nodes_list = list(nodes)
        for i in range(0, len(all_nodes_list), 80):
            chunk = all_nodes_list[i:i + 80]
            rows = table("concepts").select("id, cluster_id").in_(
                "id", chunk
            ).execute().data
            for r in rows:
                if r.get("cluster_id"):
                    cluster_map[r["id"]] = r["cluster_id"]
    except Exception:
        pass

    # --- B1: Ancestor set & descendant-count value ---
    if goal_node_id and goal_node_id in g:
        ancestors = nx.ancestors(g, goal_node_id)
        if len(ancestors) > 0:
            ancestor_set = ancestors | {goal_node_id}
        else:
            # When goal is foundational (0 ancestors), learner studies goal + key topics it unlocks
            descendants = nx.descendants(g, goal_node_id)
            if len(descendants) >= 3:
                ancestor_set = descendants | {goal_node_id}
            else:
                ancestor_set = set(nodes)
    else:
        ancestor_set = set(nodes)

    value = _compute_descendant_value(g, ancestor_set, cluster_map=cluster_map)

    # Remove already-mastered nodes
    mastered = {n for n in ancestor_set if mastery.get(n, 0.0) >= MASTERY_THRESHOLD}
    decide = ancestor_set - mastered

    if not decide:
        return []

    # --- B2: Per-node cost from est_minutes ---
    decide_list = list(decide)
    time_est: Dict[str, float] = {n: 15.0 for n in decide_list}  # default
    try:
        # Fetch est_minutes for all concepts in the decide set
        # Batch in chunks to avoid URL-length limits
        for i in range(0, len(decide_list), 80):
            chunk = decide_list[i:i + 80]
            rows = table("concepts").select("id, est_minutes").in_(
                "id", chunk
            ).execute().data
            for r in rows:
                if r.get("est_minutes"):
                    time_est[r["id"]] = float(r["est_minutes"])
    except Exception:
        pass  # fall back to defaults

    # Filter prerequisites to only those within the decide set
    decide_prereqs = [(u, v) for u, v in prereqs if u in decide and v in decide]

    # Importance = value for unmastered, but ensure non-zero
    importance = {n: max(1.0, value.get(n, 0.0)) for n in decide_list}

    solution = solve_path(
        nodes=decide_list,
        prerequisites=decide_prereqs,
        importance=importance,
        time=time_est,
        budget=float(time_budget_minutes),
        goal=goal_node_id if goal_node_id in decide else None,
    )

    if not solution.feasible or not solution.selected:
        # Fallback to topological traversal of prerequisites
        if goal_node_id and goal_node_id in g:
            subg = g.subgraph(decide)
            try:
                topo = list(nx.topological_sort(subg))
                return topo[:max(1, int(time_budget_minutes // 15))]
            except Exception:
                return list(decide)[:max(1, int(time_budget_minutes // 15))]
        return decide_list[:max(1, int(time_budget_minutes // 15))]

    # --- Topological ordering with tie-breaking ---
    # Build subgraph over selected nodes
    sg = nx.DiGraph()
    sg.add_nodes_from(solution.selected)
    for u, v in decide_prereqs:
        if u in sg and v in sg:
            sg.add_edge(u, v)

    # Fetch depth for tie-breaking
    depth_map: Dict[str, int] = {}
    try:
        for i in range(0, len(solution.selected), 80):
            chunk = solution.selected[i:i + 80]
            rows = table("concepts").select("id, depth").in_(
                "id", chunk
            ).execute().data
            for r in rows:
                depth_map[r["id"]] = r.get("depth", 0) or 0
    except Exception:
        pass

    try:
        # Topological sort with tie-breaking: ascending depth, descending value
        topo = list(nx.lexicographical_topological_sort(
            sg,
            key=lambda n: (depth_map.get(n, 0), -value.get(n, 0.0)),
        ))
        return topo
    except Exception:
        try:
            return list(nx.topological_sort(sg))
        except Exception:
            return solution.selected