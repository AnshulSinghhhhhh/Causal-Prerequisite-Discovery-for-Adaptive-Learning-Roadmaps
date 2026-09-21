"""S5 — Tree induction.

Converts a verified DAG into a hierarchical tree with:
1. Depth from the DAG (longest-path layering)
2. Grouping from structure (Leiden community detection)
3. Naming from LLM (one Groq call per cluster)

This is the stage that actually produces the mind map.
Depth and grouping are algorithmic; only naming uses an LLM.
"""

from __future__ import annotations

import os
import json
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
import networkx as nx
import numpy as np

from ...db import table

load_dotenv()


def compute_depths(G: nx.DiGraph) -> Dict[str, int]:
    """Compute depth for each node via longest-path layering.
    
    A concept's depth is the length of the longest prerequisite
    chain beneath it. This makes depth vary naturally by subject
    instead of being fixed.
    
    Uses topological DP:
    depth[v] = max(depth[u] + 1 for u in predecessors(v)), or 0 if no predecessors.
    """
    if not G.nodes():
        return {}
    
    depths: Dict[str, int] = {}
    
    # Topological sort for DP
    try:
        topo_order = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        # If there are cycles (shouldn't happen after S4), fall back
        topo_order = list(G.nodes())
    
    for node in topo_order:
        preds = list(G.predecessors(node))
        if not preds:
            depths[node] = 0
        else:
            depths[node] = max(depths.get(p, 0) for p in preds) + 1
    
    return depths


def detect_communities(
    G: nx.DiGraph,
    edge_weight_key: str = "confidence",
    seed: int = 42,
) -> Tuple[Dict[str, int], Optional[int]]:
    """Detect communities using Leiden algorithm with fixed random seed.
    
    Uses the undirected projection of the DAG, edge-weighted
    by confidence. Connected nodes (degree > 0) are partitioned into
    dense communities. Isolated nodes (degree == 0) are grouped into
    a single consolidated cluster to prevent fragmentation.
    
    Returns:
        (membership dict, isolated_cluster_idx)
    """
    try:
        import igraph as ig
        import leidenalg
    except ImportError:
        return _fallback_louvain(G), None
    
    if len(G.nodes()) == 0:
        return {}, None
    
    undirected = G.to_undirected()
    connected_nodes = [n for n in G.nodes() if undirected.degree(n) > 0]
    isolated_nodes = [n for n in G.nodes() if undirected.degree(n) == 0]
    
    membership: Dict[str, int] = {}
    next_idx = 0
    
    if connected_nodes:
        node_index = {n: i for i, n in enumerate(connected_nodes)}
        edges = []
        weights = []
        for u, v, data in G.edges(data=True):
            if u in node_index and v in node_index:
                edges.append((node_index[u], node_index[v]))
                weights.append(data.get(edge_weight_key, 1.0))
        
        ig_graph = ig.Graph(n=len(connected_nodes), edges=edges, directed=False)
        ig_graph.es["weight"] = weights
        
        # Run Leiden with deterministic random seed
        partition = leidenalg.find_partition(
            ig_graph,
            leidenalg.ModularityVertexPartition,
            weights=weights,
            seed=seed,
        )
        
        for i, node in enumerate(connected_nodes):
            membership[node] = partition.membership[i]
            
        next_idx = max(partition.membership) + 1 if partition.membership else 0

    isolated_cluster_idx = None
    if isolated_nodes:
        isolated_cluster_idx = next_idx
        for node in isolated_nodes:
            membership[node] = isolated_cluster_idx

    return membership, isolated_cluster_idx


def _fallback_louvain(G: nx.DiGraph) -> Dict[str, int]:
    """Fallback community detection using NetworkX Louvain."""
    undirected = G.to_undirected()
    try:
        from networkx.algorithms.community import louvain_communities
        communities = louvain_communities(undirected, weight="confidence", seed=42)
        membership = {}
        for i, community in enumerate(communities):
            for node in community:
                membership[node] = i
        return membership
    except Exception:
        # Last resort: each node is its own cluster
        return {n: i for i, n in enumerate(G.nodes())}


def name_clusters(
    clusters: Dict[int, List[Dict]],
    isolated_cluster_idx: Optional[int] = None,
) -> Dict[int, str]:
    """Name each cluster using a single Groq LLM call per cluster.
    
    This is a labelling task: naming a set the algorithm already
    determined. The LLM does not decide the set membership.
    """
    from groq import Groq
    
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    model = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
    
    labels: Dict[int, str] = {}
    assigned_names_lower = set()
    
    for cluster_id, members in clusters.items():
        if isolated_cluster_idx is not None and cluster_id == isolated_cluster_idx:
            labels[cluster_id] = "Foundational & Independent Concepts"
            assigned_names_lower.add("foundational & independent concepts")
            continue

        concept_list = "\n".join(
            f"- {m['canonical_name']}: {m.get('definition', 'N/A')}"
            for m in members[:15]  # Cap to avoid token limits
        )
        
        already_assigned = [lbl for lbl in labels.values() if lbl != "Foundational & Independent Concepts"]
        sibling_clause = ""
        if already_assigned:
            sibling_clause = (
                f"\nThe following cluster names have already been assigned to siblings:\n"
                f"{already_assigned}\n"
                f"Choose a distinct, descriptive name that does not duplicate any of these.\n"
            )

        prompt = f"""Given these concepts and their definitions, write a 2-4 word label for this group.
Just the label, nothing else.{sibling_clause}
Concepts:
{concept_list}"""
        
        models_to_try = [model]
        for cand in ["qwen/qwen3.8-27b", "openai/gpt-oss-20b", "openai/gpt-oss-120b", "groq/compound-mini", "groq/compound"]:
            if cand not in models_to_try:
                models_to_try.append(cand)

        named = False
        for current_model in models_to_try:
            try:
                response = client.chat.completions.create(
                    model=current_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=20,
                )
                raw_label = response.choices[0].message.content.strip().strip('"').strip("'").rstrip(".")
                # Parse label if model returned "**Label:** XYZ" or multiline reasoning
                if "**Label:**" in raw_label:
                    m = re.search(r'\*\*Label:\*\*\s*([^\n\r]+)', raw_label)
                    label = m.group(1).strip() if m else raw_label.split("\n")[0].strip()
                elif "\n" in raw_label:
                    label = raw_label.split("\n")[0].strip()
                else:
                    label = raw_label
                label = re.sub(r'^(label|cluster\s*label)[:\s]*', '', label, flags=re.IGNORECASE).strip().strip('*').strip('"').strip("'")
                
                if label and label.lower() not in assigned_names_lower:
                    labels[cluster_id] = label
                    assigned_names_lower.add(label.lower())
                    named = True
                    break
            except Exception as e:
                if "429" in str(e) or "rate_limit" in str(e):
                    continue
                break
        if not named:
            fallback = f"Cluster {cluster_id + 1}"
            suffix = 1
            while fallback.lower() in assigned_names_lower:
                suffix += 1
                fallback = f"Cluster {cluster_id + 1} ({suffix})"
            labels[cluster_id] = fallback
            assigned_names_lower.add(fallback.lower())
    
    return labels



def build_tree_paths(
    G: nx.DiGraph,
    depths: Dict[str, int],
    root_id: str,
) -> List[Dict]:
    """Build closure table entries for the tree.
    
    For each ancestor-descendant pair reachable in the DAG,
    create a tree_paths entry with the path depth.
    """
    paths = []
    
    # Self-paths (depth 0)
    for node in G.nodes():
        paths.append({
            "ancestor_id": node,
            "descendant_id": node,
            "depth": 0,
        })
    
    # Ancestor-descendant paths via BFS from each node
    for node in G.nodes():
        # Find all descendants
        visited = set()
        queue = [(node, 0)]
        while queue:
            current, d = queue.pop(0)
            for succ in G.successors(current):
                if succ not in visited:
                    visited.add(succ)
                    paths.append({
                        "ancestor_id": node,
                        "descendant_id": succ,
                        "depth": d + 1,
                    })
                    queue.append((succ, d + 1))
    
    return paths


def induce_tree(
    snapshot_id: str,
    dag_edges: List[Dict],
    concepts: List[Dict],
    domain_id: str,
) -> Dict:
    """Full tree induction pipeline: depth + clustering + naming.
    
    Args:
        snapshot_id: the graph_snapshots row ID
        dag_edges: admitted edges from S4
        concepts: concept dicts with id, canonical_name, definition
        domain_id: the domain UUID
    
    Returns:
        Dict with tree stats: {depths, clusters, tree_paths}
    """
    # Build networkx DAG from admitted edges
    G = nx.DiGraph()
    for c in concepts:
        G.add_node(c["id"], **c)
    for e in dag_edges:
        if e.get("admitted", True) and not e.get("dropped_reason"):
            G.add_edge(
                e["src_id"], e["dst_id"],
                confidence=e.get("confidence", 1.0),
            )
    
    # Step 1: Compute depths (longest-path layering)
    depths = compute_depths(G)
    
    # Step 2: Detect communities (Leiden)
    membership, isolated_cluster_idx = detect_communities(G)
    
    # Step 3: Name clusters (LLM — labelling only)
    cluster_members: Dict[int, List[Dict]] = {}
    for c in concepts:
        cluster_idx = membership.get(c["id"], 0)
        if cluster_idx not in cluster_members:
            cluster_members[cluster_idx] = []
        cluster_members[cluster_idx].append(c)
    
    cluster_labels = name_clusters(cluster_members, isolated_cluster_idx=isolated_cluster_idx)

    
    # Persist clusters
    cluster_id_map: Dict[int, str] = {}  # cluster_idx -> DB uuid
    for cluster_idx, label in cluster_labels.items():
        result = table("clusters").insert({
            "snapshot_id": snapshot_id,
            "label": label,
        }).execute()
        cluster_id_map[cluster_idx] = result.data[0]["id"]
    
    # Update concepts with depth and cluster_id
    for c in concepts:
        concept_id = c["id"]
        depth = depths.get(concept_id, 0)
        cluster_idx = membership.get(concept_id, 0)
        cluster_db_id = cluster_id_map.get(cluster_idx)
        
        table("concepts").update({
            "depth": depth,
            "cluster_id": cluster_db_id,
        }).eq("id", concept_id).execute()
    
    # Build and persist tree_paths
    # Find root (node with no predecessors, or lowest depth)
    roots = [n for n in G.nodes() if G.in_degree(n) == 0]
    root_id = roots[0] if roots else (list(G.nodes())[0] if G.nodes() else None)
    
    if root_id:
        tree_paths = build_tree_paths(G, depths, root_id)
        # Add snapshot_id
        for tp in tree_paths:
            tp["snapshot_id"] = snapshot_id
        
        # Batch insert
        if tree_paths:
            batch_size = 500
            for i in range(0, len(tree_paths), batch_size):
                batch = tree_paths[i:i + batch_size]
                table("tree_paths").insert(batch).execute()
    else:
        tree_paths = []
    
    return {
        "depths": depths,
        "n_clusters": len(cluster_labels),
        "cluster_labels": cluster_labels,
        "membership": membership,
        "cluster_id_map": cluster_id_map,
        "n_tree_paths": len(tree_paths),
        "depth_range": (min(depths.values()), max(depths.values())) if depths else (0, 0),
    }
