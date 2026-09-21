"""Module 2 -- DAG construction (threshold filter, Tarjan SCC, transitive reduction).

Section 4.2 pipeline:

    1. Filter candidate edges at ``tau_edge``.
    2. Tarjan's SCC detects cycles; prune the lowest-confidence edge in each.
    3. ``networkx.transitive_reduction`` removes redundant hops.

Also provides ``audit_collateral_damage`` (Section 4.2 / build order step 9): for
every true prerequisite pair in the gold set, check whether it survives as a
direct edge, and if not classify the loss as threshold filtering, cycle pruning
(a correct edge dropped because a false reversed edge scored higher), or
(non-issue) subsumption into a valid transitive path. This tells us whether a
DAG quality problem belongs to Module 2 or was inherited from Module 1's scores.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import networkx as nx

from ..config import config
from ..graph_model import ConceptEdge, EdgeType, GraphModel, NodeStatus, ConceptNode


@dataclass
class DAGBuildResult:
    """The assembled DAG plus provenance of every dropped/kept edge."""

    edges: List[Tuple[str, str, float]] = field(default_factory=list)
    dropped_threshold: List[Tuple[str, str, float]] = field(default_factory=list)
    dropped_cycle: List[Tuple[str, str, float]] = field(default_factory=list)
    dropped_transitive: List[Tuple[str, str, float]] = field(default_factory=list)
    is_dag: bool = True

    @property
    def edge_set(self) -> Set[Tuple[str, str]]:
        return {(u, v) for (u, v, _) in self.edges}


@dataclass
class CollateralAudit:
    """Per-gold-pair survival status after DAG construction."""

    survived: List[Tuple[str, str]] = field(default_factory=list)
    lost_to_threshold: List[Tuple[str, str]] = field(default_factory=list)
    lost_to_cycle_pruning: List[Tuple[str, str]] = field(default_factory=list)
    subsumed_by_transitive_path: List[Tuple[str, str]] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "total": len(self.survived) + len(self.lost_to_threshold)
            + len(self.lost_to_cycle_pruning)
            + len(self.subsumed_by_transitive_path),
            "survived": len(self.survived),
            "lost_to_threshold": len(self.lost_to_threshold),
            "lost_to_cycle_pruning": len(self.lost_to_cycle_pruning),
            "subsumed_by_transitive_path": len(self.subsumed_by_transitive_path),
        }


def filter_by_threshold(
    pairs: Sequence[Tuple[str, str]],
    scores: Sequence[float],
    tau_edge: Optional[float] = None,
) -> Tuple[List[Tuple[str, str, float]], List[Tuple[str, str, float]]]:
    """Split scored pairs at ``tau_edge`` into kept / dropped."""
    tau = config.module2.tau_edge if tau_edge is None else tau_edge
    kept, dropped = [], []
    for (u, v), s in zip(pairs, scores):
        entry = (u, v, float(s))
        (kept if s >= tau else dropped).append(entry)
    return kept, dropped


def prune_cycles(
    edges: Sequence[Tuple[str, str, float]],
) -> Tuple[List[Tuple[str, str, float]], List[Tuple[str, str, float]]]:
    """Detect SCC cycles and prune the lowest-confidence edge in each.

    Returns (acyclic_edges, pruned_edges). Iterates: within every SCC of size
    > 1 (or self-loop), drop the minimum-confidence edge, then recompute until
    the graph is acyclic. Deterministic tie-break on (confidence, u, v) so runs
    are reproducible.
    """
    remaining: List[Tuple[str, str, float]] = list(edges)
    pruned: List[Tuple[str, str, float]] = []
    changed = True
    while changed:
        changed = False
        g = nx.DiGraph()
        g.add_weighted_edges_from(
            [(u, v, {"c": c}) for (u, v, c) in remaining]
        )
        # Find a cycle via SCCs with size > 1 or any self loop.
        sccs = [s for s in nx.strongly_connected_components(g)
                if len(s) > 1]
        # Also catch self-loops.
        self_loops = [u for u in g.nodes if g.has_edge(u, u)]
        if not sccs and not self_loops:
            break
        if self_loops:
            sc_nodes = {self_loops[0]}
        else:
            sc_nodes = sccs[0]
        # Candidate edges fully inside the SCC (or self-loop).
        candidates = [
            e for e in remaining
            if e[0] in sc_nodes and e[1] in sc_nodes
        ]
        if not candidates:
            break
        # Lowest confidence; deterministic tie-break.
        worst = min(candidates, key=lambda e: (e[2], e[0], e[1]))
        remaining.remove(worst)
        pruned.append(worst)
        changed = True
    return remaining, pruned


def transitive_reduction_edges(
    edges: Sequence[Tuple[str, str, float]],
) -> Tuple[List[Tuple[str, str, float]], List[Tuple[str, str, float]]]:
    """Drop edges implying a longer path (A->C when A->B->C exists)."""
    g = nx.DiGraph()
    g.add_weighted_edges_from([(u, v, {"c": c}) for (u, v, c) in edges])
    tr = nx.transitive_reduction(g)
    kept: List[Tuple[str, str, float]] = []
    dropped: List[Tuple[str, str, float]] = []
    kept_set = {(u, v) for u, v in tr.edges}
    for (u, v, c) in edges:
        (kept if (u, v) in kept_set else dropped).append((u, v, c))
    return kept, dropped


def build_dag(
    pairs: Optional[Sequence[Tuple[str, str]]] = None,
    scores: Optional[Sequence[float]] = None,
    tau_edge: Optional[float] = None,
    do_transitive_reduction: Optional[bool] = None,
    domain_id: Optional[str] = None,
    model_version_id: Optional[str] = None,
) -> Any:
    """Run the full DAG construction pipeline from scored edge candidates.

    If domain_id and model_version_id are provided, delegates to build_dag_snapshot
    to load from and persist to Supabase tables.
    """
    if domain_id is not None and model_version_id is not None:
        tau = tau_edge if tau_edge is not None else 0.435804
        return build_dag_snapshot(domain_id, model_version_id, tau)

    if pairs is None or scores is None:
        raise ValueError("pairs and scores are required when not using domain_id/model_version_id")

    do_tr = config.module2.transitive_reduce \
        if do_transitive_reduction is None else do_transitive_reduction
    kept, dropped_t = filter_by_threshold(pairs, scores, tau_edge=tau_edge)
    acyclic, dropped_c = prune_cycles(kept)
    dropped_tr: List[Tuple[str, str, float]] = []
    if do_tr:
        acyclic, dropped_tr = transitive_reduction_edges(acyclic)

    # Final sanity check: the result must be a DAG.
    g = nx.DiGraph()
    g.add_edges_from([(u, v) for (u, v, _) in acyclic])
    is_dag = nx.is_directed_acyclic_graph(g)
    return DAGBuildResult(
        edges=acyclic,
        dropped_threshold=dropped_t,
        dropped_cycle=dropped_c,
        dropped_transitive=dropped_tr,
        is_dag=is_dag,
    )


def build_dag_snapshot(
    domain_id: str,
    model_version_id: str,
    tau_edge: float,
) -> Dict[str, Any]:
    """Assemble DAG from candidate_edges and edge_scores in Supabase,
    then persist graph_snapshots and dag_edges with dropped_reason.
    """
    from ...db import table, fetch_all

    # Fetch candidate edges with pagination to avoid PostgREST 1000-row limit
    cands = fetch_all(table("candidate_edges").select("id, src_id, dst_id").eq("domain_id", domain_id))
    cand_map = {c["id"]: (c["src_id"], c["dst_id"]) for c in cands}
    cand_ids = list(cand_map.keys())

    if not cand_ids:
        snap_res = table("graph_snapshots").insert({
            "domain_id": domain_id,
            "model_version_id": model_version_id,
            "tau_edge": round(tau_edge, 6),
            "n_nodes": 0,
            "n_edges": 0,
            "build_stats": {},
        }).execute()
        snap_id = snap_res.data[0]["id"]
        return {"snapshot_id": snap_id, "edges": [], "dag_result": DAGBuildResult()}

    # Fetch edge scores for these candidates in batches of 200
    scores_res = []
    batch_size = 200
    for i in range(0, len(cand_ids), batch_size):
        batch = cand_ids[i:i + batch_size]
        chunk = table("edge_scores").select(
            "candidate_edge_id, s_fused, s_lr_forward, s_lr_reverse"
        ).eq("model_version_id", model_version_id).in_("candidate_edge_id", batch).execute().data
        scores_res.extend(chunk)

    pairs = []
    score_list = []
    for s in scores_res:
        cid = s["candidate_edge_id"]
        if cid in cand_map:
            u, v = cand_map[cid]
            s_fwd = s.get("s_lr_forward") or 0.0
            s_rev = s.get("s_lr_reverse") or 0.0
            # Orient edge in direction of higher probability
            if s_fwd >= s_rev:
                oriented_u, oriented_v = u, v
                sc = s_fwd
            else:
                oriented_u, oriented_v = v, u
                sc = s_rev
            pairs.append((oriented_u, oriented_v))
            score_list.append(float(sc))

    dag_result = build_dag(pairs, score_list, tau_edge=tau_edge)


    nodes = set()
    for u, v, _ in dag_result.edges:
        nodes.add(u)
        nodes.add(v)

    build_stats = {
        "dropped_threshold_count": len(dag_result.dropped_threshold),
        "dropped_cycle_count": len(dag_result.dropped_cycle),
        "dropped_transitive_count": len(dag_result.dropped_transitive),
        "admitted_edge_count": len(dag_result.edges),
    }

    snap_res = table("graph_snapshots").insert({
        "domain_id": domain_id,
        "model_version_id": model_version_id,
        "tau_edge": round(tau_edge, 6),
        "n_nodes": len(nodes),
        "n_edges": len(dag_result.edges),
        "build_stats": build_stats,
    }).execute()
    snap_id = snap_res.data[0]["id"]

    dag_edge_rows = []
    for u, v, conf in dag_result.edges:
        dag_edge_rows.append({
            "snapshot_id": snap_id,
            "src_id": u,
            "dst_id": v,
            "confidence": conf,
            "admitted": True,
            "dropped_reason": None,
        })
    for u, v, conf in dag_result.dropped_threshold:
        dag_edge_rows.append({
            "snapshot_id": snap_id,
            "src_id": u,
            "dst_id": v,
            "confidence": conf,
            "admitted": False,
            "dropped_reason": "threshold",
        })
    for u, v, conf in dag_result.dropped_cycle:
        dag_edge_rows.append({
            "snapshot_id": snap_id,
            "src_id": u,
            "dst_id": v,
            "confidence": conf,
            "admitted": False,
            "dropped_reason": "cycle_prune",
        })
    for u, v, conf in dag_result.dropped_transitive:
        dag_edge_rows.append({
            "snapshot_id": snap_id,
            "src_id": u,
            "dst_id": v,
            "confidence": conf,
            "admitted": False,
            "dropped_reason": "transitive",
        })

    if dag_edge_rows:
        for i in range(0, len(dag_edge_rows), 200):
            table("dag_edges").upsert(dag_edge_rows[i:i+200], on_conflict="snapshot_id,src_id,dst_id").execute()

    return {
        "snapshot_id": snap_id,
        "edges": [{"src_id": u, "dst_id": v, "confidence": c, "admitted": True} for (u, v, c) in dag_result.edges],
        "dag_result": dag_result,
    }


def has_path(g: nx.DiGraph, u: str, v: str) -> bool:
    """True if a (possibly indirect) path u -> ... -> v exists."""
    return nx.has_path(g, u, v) if u in g and v in g else False


def audit_collateral_damage(
    result: DAGBuildResult,
    gold_pairs: Sequence[Tuple[str, str]],
) -> CollateralAudit:
    """Classify the fate of every gold prerequisite pair after DAG build.

    A gold pair is ``subsumed_by_transitive_path`` when it is not a direct edge
    but a valid multi-hop path still connects u -> v (a non-issue). Otherwise
    it is ``lost_to_threshold`` (edge filtered at tau_edge), or
    ``lost_to_cycle_pruning`` (a correct edge pruned because a false reversed
    edge scored higher), or survived as a direct edge.
    """
    audit = CollateralAudit()
    kept_set = result.edge_set
    # Build a graph of the *final* edges to test transitive paths.
    g = nx.DiGraph()
    g.add_edges_from([(u, v) for (u, v, _) in result.edges])
    # Also graph of pre-cycle edges to distinguish threshold vs cycle loss.
    pre_cycle = nx.DiGraph()
    pre_cycle.add_edges_from([(u, v) for (u, v, _) in result.edges])
    for (u, v, _) in result.dropped_cycle:
        pre_cycle.add_edge(u, v)

    threshold_set = {(u, v) for (u, v, _) in result.dropped_threshold}
    cycle_set = {(u, v) for (u, v, _) in result.dropped_cycle}

    for (u, v) in gold_pairs:
        if (u, v) in kept_set:
            audit.survived.append((u, v))
        elif has_path(g, u, v):
            audit.subsumed_by_transitive_path.append((u, v))
        elif (u, v) in cycle_set:
            audit.lost_to_cycle_pruning.append((u, v))
        elif (u, v) in threshold_set:
            audit.lost_to_threshold.append((u, v))
        else:
            # Edge absent everywhere: treat as threshold-filtered.
            audit.lost_to_threshold.append((u, v))
    return audit


def graph_from_dag(
    result: DAGBuildResult,
    node_labels: Optional[Dict[str, str]] = None,
) -> GraphModel:
    """Materialize a ``GraphModel`` from a DAG build result."""
    model = GraphModel()
    labels = node_labels or {}
    node_ids: Set[str] = set()
    for (u, v, _) in result.edges:
        node_ids.add(u)
        node_ids.add(v)
    for nid in sorted(node_ids):
        model.add_node(ConceptNode(id=nid, label=labels.get(nid, nid)))
    for (u, v, c) in result.edges:
        model.add_edge(ConceptEdge(source=u, target=v, confidence=c))
    return model