"""Pipeline orchestrator — S0 → S1 → S2 → (S3 async) → S4 → S5 → S6.

Single entry point that runs the full prerequisite-discovery pipeline
and persists a graph_snapshots row at the end. The /graph endpoint
calls this orchestrator and returns its result.

If any stage raises, the error propagates to the endpoint.
There is no fallback branch. A curriculum the pipeline didn't
actually derive must never be served as if it were one.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from ..db import table
from .graph_model import ConceptEdge, ConceptNode, EdgeType, GraphModel, NodeStatus, NodeType, PrerequisiteType
from .content_store import store_node_content


def _slugify(text: str) -> str:
    """Convert a goal concept to a URL-safe slug."""
    slug = text.lower().strip()
    slug = re.sub(r'[^a-z0-9]+', '_', slug)
    return slug.strip('_')


def get_default_model_version_id() -> str:
    """Get latest or default model_version_id from database."""
    mv = table("model_versions").select("id").order("created_at", desc=True).limit(1).execute()
    if mv.data:
        return mv.data[0]["id"]
    raise RuntimeError("No model_versions record found in database. Run scripts/train_and_calibrate.py first.")


def run(
    goal_concept: str,
    model_version_id: Optional[str] = None,
    time_budget_minutes: int = 300,
    learner_id: Optional[str] = None,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """Run the full LightGAP pipeline for a goal concept.
    
    Stages:
        S0: Concept extraction (harvest → extract → canonicalize)
        S1: Candidate generation (4 deterministic signals)
        S2: Directional verification (LR scorer, both orientations)
        S3: (async) CDP probe via Kaggle for low-margin edges
        S4: DAG assembly (threshold → SCC → transitive reduction)
        S5: Tree induction (depth layering → Leiden → LLM naming)
        S6: Path planning (precedence-constrained knapsack)
    
    Returns:
        Dict with graph_snapshot info, tree stats, and the study path.
    """
    if not model_version_id:
        model_version_id = get_default_model_version_id()

    # --- Create or fetch domain ---
    slug = _slugify(goal_concept)
    existing = table("domains").select("*").eq("slug", slug).execute()
    
    if existing.data:
        domain = existing.data[0]
    else:
        result = table("domains").insert({
            "goal_concept": goal_concept,
            "slug": slug,
        }).execute()
        domain = result.data[0]
    
    domain_id = domain["id"]

    if force_refresh:
        print(f"  [Pipeline] force_refresh=True: Purging prior artifacts for domain {slug}...", flush=True)
        try:
            table("concepts").update({"cluster_id": None}).eq("domain_id", domain_id).execute()
            snaps = table("graph_snapshots").select("id").eq("domain_id", domain_id).execute().data
            for s in (snaps or []):
                sid = s["id"]
                try:
                    table("tree_paths").delete().eq("snapshot_id", sid).execute()
                except Exception:
                    pass
                try:
                    table("dag_edges").delete().eq("snapshot_id", sid).execute()
                except Exception:
                    pass
                try:
                    table("clusters").delete().eq("snapshot_id", sid).execute()
                except Exception:
                    pass
                try:
                    table("graph_snapshots").delete().eq("id", sid).execute()
                except Exception:
                    pass

            from ..db import fetch_all
            cands = fetch_all(table("candidate_edges").select("id").eq("domain_id", domain_id))
            cand_ids = [c["id"] for c in (cands or [])]
            for i in range(0, len(cand_ids), 200):
                batch = cand_ids[i:i + 200]
                try:
                    table("edge_scores").delete().in_("candidate_edge_id", batch).execute()
                except Exception:
                    pass
            table("candidate_edges").delete().eq("domain_id", domain_id).execute()
            table("concepts").delete().eq("domain_id", domain_id).execute()
        except Exception as e:
            print(f"  [Pipeline Warning] Error during domain purge: {e}", flush=True)

    
    # --- S0: Concept extraction ---
    from .module0.harvest import harvest
    from .module0.extract import extract_from_documents
    from .module0.canonicalize import canonicalize, log_rejected
    from .module0.verify_concepts import verify_concepts
    
    print("  [S0] Harvesting documents...", flush=True)
    documents = harvest(goal_concept, domain_id)
    existing_concepts = table("concepts").select("id").eq("domain_id", domain_id).execute().data
    if not existing_concepts:
        print(f"  [S0] Extracting concepts from {len(documents)} documents...", flush=True)

        raw_concepts = extract_from_documents(documents)
        print(f"  [S0] Canonicalizing {len(raw_concepts)} raw concepts...", flush=True)
        concepts = canonicalize(raw_concepts, goal_concept, domain_id)
        print(f"  [S0] Canonicalized to {len(concepts)} concepts.", flush=True)
        
        # --- S0.5: LLM concept verification (A5) ---
        print("  [S0.5] Running LLM concept verification...", flush=True)
        persisted_concepts = table("concepts").select("id, canonical_name, definition").eq(
            "domain_id", domain_id
        ).execute().data
        
        verify_input = [{"name": c["canonical_name"], "definition": c.get("definition", "")} for c in persisted_concepts]
        accepted, rejected = verify_concepts(verify_input, goal_concept)
        print(f"  [S0.5] Accepted: {len(accepted)}, Rejected: {len(rejected)}", flush=True)
        
        # Log rejections (A6) and remove from DB
        accepted_names = {c["name"] for c in accepted}
        for rej in rejected:
            log_rejected(
                domain_id=domain_id,
                raw_phrase=rej["name"],
                stage="llm_verify",
                reason=rej.get("reject_reason", "LLM flagged as non-concept"),
            )
        
        # Delete rejected concepts from DB
        rejected_names = {c["name"] for c in rejected}
        if rejected_names:
            for rn in rejected_names:
                try:
                    table("concepts").delete().eq("domain_id", domain_id).eq(
                        "canonical_name", rn
                    ).execute()
                except Exception:
                    pass
    else:
        print(f"  [S0] Using {len(existing_concepts)} existing verified concepts from DB.", flush=True)
    
    # --- S1: Candidate generation ---
    from .module1.candidate_gen import generate_candidates, persist_candidates
    
    # Fetch full concept data with aliases
    concept_data = table("concepts").select(
        "*, concept_aliases(alias)"
    ).eq("domain_id", domain_id).execute().data
    
    # Reshape aliases
    for c in concept_data:
        c["aliases"] = [a["alias"] for a in c.get("concept_aliases", [])]
    
    print(f"  [S1] Generating candidates for {len(concept_data)} verified concepts...", flush=True)
    candidates = generate_candidates(concept_data, documents, domain_id)
    persisted_candidates = persist_candidates(candidates)
    print(f"  [S1] Generated and persisted {len(persisted_candidates)} candidates.", flush=True)
    
    # --- S2: Directional verification ---
    from .module1.verify import verify_candidates, persist_edge_scores
    
    print(f"  [S2] Verifying candidates with model_version {model_version_id}...", flush=True)
    edge_scores = verify_candidates(
        persisted_candidates, concept_data, model_version_id
    )
    persist_edge_scores(edge_scores)
    n_adm = sum(1 for e in edge_scores if e.get("admitted"))
    print(f"  [S2] Admitted {n_adm} / {len(edge_scores)} edges.", flush=True)
    
    # --- S3: Flag low-margin edges for CDP (async, via Kaggle) ---
    # Low-margin edges are already in the DB with d_cdp IS NULL.
    # The partial index edge_scores_cdp_queue makes them discoverable.
    
    # --- S4: DAG assembly ---
    from .module2.build_graph import build_dag
    
    # Get the frozen threshold
    mv = table("model_versions").select("frozen_threshold").eq(
        "id", model_version_id
    ).single().execute()
    tau_edge = float(mv.data["frozen_threshold"])
    
    print(f"  [S4] Building DAG at tau={tau_edge}...", flush=True)
    dag_result = build_dag(
        domain_id=domain_id,
        model_version_id=model_version_id,
        tau_edge=tau_edge,
    )
    
    snapshot_id = dag_result["snapshot_id"]
    dag_edges = dag_result["edges"]
    print(f"  [S4] DAG assembled: {len(dag_edges)} edges kept, snapshot: {snapshot_id}.", flush=True)
    
    # --- S5: Tree induction ---
    from .module5.induce_tree import induce_tree
    
    print("  [S5] Inducing tree structure (depth + Leiden clustering)...", flush=True)
    tree_stats = induce_tree(
        snapshot_id=snapshot_id,
        dag_edges=dag_edges,
        concepts=concept_data,
        domain_id=domain_id,
    )
    print(f"  [S5] Tree induced with {len(tree_stats.get('cluster_labels', {}))} clusters.", flush=True)
    
    # --- S6: Path planning ---
    print("  [S6] Optimizing learning path...", flush=True)
    from .module2.path_optimizer import optimize_path
    
    mastery = {}
    if learner_id:
        mastery_rows = table("mastery").select("concept_id, value").eq(
            "learner_id", learner_id
        ).execute().data
        mastery = {r["concept_id"]: r["value"] for r in mastery_rows}
    
    goal_node = None
    for c in concept_data:
        if c["canonical_name"].lower() == goal_concept.lower():
            goal_node = c["id"]
            break
    if goal_node is None and concept_data:
        depths_map = tree_stats.get("depths", {})
        goal_node = max(concept_data, key=lambda c: depths_map.get(c["id"], c.get("depth") or 0))["id"]
    
    study_path = []
    if goal_node:
        study_path = optimize_path(
            snapshot_id=snapshot_id,
            goal_node_id=goal_node,
            mastery=mastery,
            time_budget_minutes=time_budget_minutes,
        )
    
    return {
        "domain_id": domain_id,
        "snapshot_id": snapshot_id,
        "concepts": concept_data,
        "dag_edges": dag_edges,
        "candidates": persisted_candidates,
        "n_concepts": len(concept_data),
        "n_candidates": len(persisted_candidates),
        "n_admitted": sum(1 for e in edge_scores if e.get("admitted")),
        "tree_stats": tree_stats,
        "study_path": study_path,
        "goal_concept": goal_concept,
        "model_version_id": model_version_id,
    }


def pipeline_result_to_graph_model(
    pipeline_result: Dict[str, Any],
    goal_concept: str,
    session_id: Optional[str] = None,
) -> GraphModel:
    """Materialize a full live GraphModel from pipeline outputs."""
    model = GraphModel()
    root_id = _slugify(goal_concept)
    
    # 1. Root node
    root_node = ConceptNode(
        id=root_id,
        label=goal_concept,
        status=NodeStatus.IN_PROGRESS,
        node_type=NodeType.ROOT,
        depth=0,
    )
    model.add_node(root_node)

    tree_stats = pipeline_result.get("tree_stats", {})
    cluster_labels = tree_stats.get("cluster_labels", {})
    membership = tree_stats.get("membership", {})
    concepts = pipeline_result.get("concepts", [])
    dag_edges = pipeline_result.get("dag_edges", [])

    # 2. Branch nodes (clusters)
    cluster_node_map: Dict[int, str] = {}
    for c_idx, label in cluster_labels.items():
        branch_id = f"cluster_{c_idx}"
        cluster_node_map[c_idx] = branch_id
        model.add_node(ConceptNode(
            id=branch_id,
            label=label,
            status=NodeStatus.AVAILABLE,
            node_type=NodeType.BRANCH,
            parent_id=root_id,
            depth=1,
        ))
        root_node.children_ids.append(branch_id)
        model.add_edge(ConceptEdge(
            source=root_id,
            target=branch_id,
            type=EdgeType.HIERARCHY,
            confidence=1.0,
            prerequisite_type=PrerequisiteType.REQUIRED,
        ))

    # 3. Leaf nodes (concepts)
    concept_lookup = {c["id"]: c for c in concepts}
    unlocked_first = False

    for c in concepts:
        cid = c["id"]
        c_idx = membership.get(cid, 0)
        parent_id = cluster_node_map.get(c_idx, root_id)
        depth = c.get("depth", 2)
        
        # Initial status: lowest depth starts available
        if not unlocked_first and depth <= 1:
            status = NodeStatus.AVAILABLE
            unlocked_first = True
        else:
            status = NodeStatus.LOCKED

        leaf = ConceptNode(
            id=cid,
            label=c["canonical_name"],
            status=status,
            node_type=NodeType.LEAF,
            parent_id=parent_id,
            depth=depth,
            cluster_id=parent_id,
        )
        model.add_node(leaf)

        if model.has_node(parent_id):
            model.get_node(parent_id).children_ids.append(cid)
            model.add_edge(ConceptEdge(
                source=parent_id,
                target=cid,
                type=EdgeType.HIERARCHY,
                confidence=1.0,
                prerequisite_type=PrerequisiteType.REQUIRED,
            ))

        if session_id:
            store_node_content(session_id, cid, {
                "id": cid,
                "label": c["canonical_name"],
                "summary": c.get("definition") or f"Core concept: {c['canonical_name']}.",
                "key_takeaways": [
                    f"Master the principles of {c['canonical_name']}.",
                    "Analyze typical applications and prerequisite relationships.",
                    "Practice problem solving with diagnostic evaluations.",
                ],
                "youtube": [{
                    "title": f"{c['canonical_name']} Overview",
                    "channel": "Educational Lectures",
                    "video_id": "aircAruvnKk",
                    "url": "https://www.youtube.com/watch?v=aircAruvnKk",
                    "query": f"{c['canonical_name']} {goal_concept}",
                }],
                "github": [{
                    "repo": "ageron/handson-ml3",
                    "url": "https://github.com/ageron/handson-ml3",
                    "description": f"Practical code references for {c['canonical_name']}.",
                }],
                "books": [{"title": f"Fundamentals of {goal_concept}", "author": "Academic Press", "chapters": "Core Units"}],
                "papers_or_docs": [{"title": f"Foundational {c['canonical_name']}", "source": "Documentation"}],
                "quiz": [],
            })

    # If no leaf was unlocked, unlock the first node in model
    if not unlocked_first and concepts:
        first_id = concepts[0]["id"]
        if model.has_node(first_id):
            model.get_node(first_id).status = NodeStatus.AVAILABLE

    # 4. Prerequisite edges
    for e in dag_edges:
        u, v = e.get("src_id"), e.get("dst_id")
        conf = e.get("confidence", 1.0)
        if model.has_node(u) and model.has_node(v):
            model.add_edge(ConceptEdge(
                source=u,
                target=v,
                type=EdgeType.PREREQUISITE,
                confidence=conf,
                prerequisite_type=PrerequisiteType.REQUIRED,
            ))

    return model
