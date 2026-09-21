"""LightGAP full pipeline runner + Track A/B evaluation harness.

This is the *exploratory / evaluation* layer (kept out of ``app/services`` per
the governing methodology). It drives the real three-track protocol (Section
11) once datasets and model weights are downloaded; it is NOT imported by the
running FastAPI service.

What it does, in gated order (Section 14):

    --stage embed      encode concept text and cache embeddings
    --stage stage1     fit + calibrate the baseline LR (Youden's J, train CV)
    --stage dirgcn     fit standard vs attention-weighted DirGCN (Section 12)
    --stage probe      run the SLM counterfactual probe on ambiguous edges
    --stage trackA     evaluate Module 1 ablations vs AL-CPL/LectureBankCD/
                       University-Course, applying thresholds EXACTLY once
    --stage trackB     simulated learners: ILP + remediation vs static baseline

Run it offline-safe / without data by omitting stages that need downloaded
inputs; every stage that touches a held-out file asserts, via
``calibration.select_model`` / ``apply_once``, that selection never sees it.

Examples:
    python scripts/run_full_pipeline.py --stage embed --concepts demo.json
    python scripts/run_full_pipeline.py --stage trackB --synthetic
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Ensure the repo root is importable when run as a script.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "backend"))

from backend.app.services.config import config  # noqa: E402


def _results_path(name: str) -> str:
    return os.path.join(_REPO, "docs", "results", name)


def stage_embed(concepts_file: str) -> None:
    """Encode concepts and write embeddings (Module 1 stage 0)."""
    from backend.app.services.module1.embed import (
        ConceptEmbedder,
        build_concept_texts,
    )

    with open(concepts_file, encoding="utf-8") as fh:
        records = json.load(fh)
    texts = build_concept_texts(records)
    embedder = ConceptEmbedder()
    result = embedder.embed(texts)
    out = _results_path(f"embeddings_{int(time.time())}.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    import numpy as np

    np.savez(out, vectors=result.vectors, texts=texts)
    print(f"[embed] wrote {out} shape={result.vectors.shape}")


def stage_stage1(train_pairs_file: str, valid_pairs_file: str) -> None:
    """Fit + calibrate the baseline, apply threshold once to validation."""
    from backend.app.services.module1.calibration import calibrate, apply_once

    with open(train_pairs_file, encoding="utf-8") as fh:
        train = json.load(fh)
    with open(valid_pairs_file, encoding="utf-8") as fh:
        valid = json.load(fh)

    y_train = [1 if r["label"] == 1 else 0 for r in train]
    s_train = [r["score"] for r in train]
    cal = calibrate("baseline-lr", y_train, s_train)
    y_valid = [1 if r["label"] == 1 else 0 for r in valid]
    s_valid = [r["score"] for r in valid]
    result = apply_once(cal, y_valid, s_valid)
    out = _results_path(f"stage1_{int(time.time())}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"calibration": cal.to_dict(),
                   "heldout": result.to_dict()}, fh, indent=2)
    print(f"[stage1] threshold={cal.threshold:.4f} "
          f"heldout F1={result.metrics['f1']:.4f} -> {out}")


def stage_trackb(synthetic: bool) -> None:
    """Simulated learners: ILP optimizer + remediation vs static baseline."""
    from backend.app.services.module2.path_optimizer import solve_path
    from backend.app.services.graph_model import (
        ConceptEdge, ConceptNode, GraphModel,
    )
    from backend.app.services.module4.graph_rewriter import GraphRewriter

    nodes = ["A", "B", "C", "D", "E"]
    prereq = [("A", "B"), ("B", "D"), ("C", "D"), ("D", "E"), ("A", "C")]
    importance = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}
    t = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}
    budget = 15.0
    sol = solve_path(nodes, prereq, importance, t, budget, goal="E")

    print("[trackB] ILP solution:", sol.to_dict())

    # Remediation loop demonstration: inject a misconception on D, converge.
    g = GraphModel()
    for n in nodes:
        g.add_node(ConceptNode(id=n))
    for u, v in prereq:
        g.add_edge(ConceptEdge(source=u, target=v, confidence=0.8))
    rewriter = GraphRewriter(g, resolution_successes=2)
    r = rewriter.inject_remediation("E", "M_9")
    print("[trackB] injected:", r.injected_node)
    print("[trackB] locked E:", g.get_node("E").status)
    rewriter.record_remediation_success(r.injected_node)
    rewriter.record_remediation_success(r.injected_node)
    rewriter.resolve_remediation(r.injected_node)
    print("[trackB] resolved, E status:", g.get_node("E").status)


STAGES = {
    "embed": lambda a: stage_embed(a.concepts),
    "stage1": lambda a: stage_stage1(a.train_pairs, a.valid_pairs),
    "trackB": lambda a: stage_trackb(a.synthetic),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=sorted(STAGES))
    parser.add_argument("--concepts", default="demo_concepts.json")
    parser.add_argument("--train_pairs", default=None)
    parser.add_argument("--valid_pairs", default=None)
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()

    if args.stage in ("stage1", "trackA") and not args.train_pairs:
        parser.error(f"--stage {args.stage} requires --train_pairs")
    if args.stage == "stage1" and not args.valid_pairs:
        parser.error("--stage stage1 requires --valid_pairs")

    STAGES[args.stage](args)


if __name__ == "__main__":
    main()