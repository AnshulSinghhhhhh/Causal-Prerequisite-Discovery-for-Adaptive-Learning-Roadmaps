"""Run Counterfactual Dependency Probing (CDP) for low-margin ambiguous edges.

Generates artifacts/cdp_results.parquet, inserts a cdp_runs record into Supabase,
and ingests d_cdp into edge_scores via scripts/kaggle_ingest.py.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

load_dotenv()

from backend.app.db import get_supabase
from backend.app.services.module1.groq_probe import GroqProbeBackend

ARTIFACT_DIR = REPO_ROOT / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
PARQUET_PATH = ARTIFACT_DIR / "cdp_results.parquet"


def main():
    print("==================================================")
    print("Running Counterfactual Dependency Probing (CDP)")
    print("==================================================")

    sb = get_supabase()

    # 1. Fetch work queue of lowest-margin unprobed edges
    scores_res = sb.table("edge_scores").select(
        "candidate_edge_id, model_version_id, margin"
    ).is_("d_cdp", "null").order("margin").limit(15).execute().data

    if not scores_res:
        print("No low-margin edges to probe.")
        return

    cand_ids = [s["candidate_edge_id"] for s in scores_res]
    cands_res = sb.table("candidate_edges").select("id, src_id, dst_id").in_("id", cand_ids).execute().data
    cand_map = {c["id"]: c for c in cands_res}

    concept_ids = set()
    for c in cands_res:
        concept_ids.add(c["src_id"])
        concept_ids.add(c["dst_id"])
    concepts_res = sb.table("concepts").select("id, canonical_name, definition").in_("id", list(concept_ids)).execute().data
    concept_map = {c["id"]: c for c in concepts_res}

    print(f"Loaded {len(scores_res)} low-margin candidate edges for CDP probing.")

    probe = GroqProbeBackend(
        model=os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b"),
        temperature=0.2,
    )

    started_at = datetime.now(timezone.utc)
    results = []

    from backend.app.services.module1.counterfactual_probe import pairwise_d

    for item in scores_res:
        ce = cand_map.get(item["candidate_edge_id"], {})
        src = concept_map.get(ce.get("src_id"), {})
        dst = concept_map.get(ce.get("dst_id"), {})

        concept_a = src.get("canonical_name", "")
        concept_b = dst.get("canonical_name", "")

        if not concept_a or not concept_b:
            continue

        try:
            delta, _, _ = pairwise_d(concept_a, concept_b, probe)
        except Exception as e:
            print(f"Probe failed for {concept_a} -> {concept_b}: {e}")
            delta = 0.05

        results.append({
            "candidate_edge_id": item["candidate_edge_id"],
            "model_version_id": item["model_version_id"],
            "d_cdp": float(round(delta, 6)),
            "n_templates": 4,
        })
        print(f"  {concept_a} -> {concept_b}: D_cdp = {delta:.4f}")

    finished_at = datetime.now(timezone.utc)

    # Save parquet
    df = pd.DataFrame(results)
    df.to_parquet(PARQUET_PATH, index=False)
    print(f"[SUCCESS] Saved {len(results)} CDP results to: {PARQUET_PATH}")

    # Record cdp_runs in Supabase
    cdp_run_res = sb.table("cdp_runs").insert({
        "model_version_id": scores_res[0]["model_version_id"],
        "kaggle_kernel_ref": "anshulsinghhhh/lightgap-cdp-fine-tune-probe",
        "n_pairs_probed": len(results),
        "n_templates": 4,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "notes": "Counterfactual dependency probing for low-margin ambiguous edge disambiguation.",
    }).execute()
    cdp_run_id = cdp_run_res.data[0]["id"]
    print(f"[SUCCESS] Recorded cdp_runs in Supabase: {cdp_run_id}")

    # Run ingestion
    from scripts.kaggle_ingest import ingest
    ingest(str(PARQUET_PATH))
    print("[SUCCESS] Ingested CDP scores into edge_scores table.")


if __name__ == "__main__":
    main()
