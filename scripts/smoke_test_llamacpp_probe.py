"""Smoke test for LocalLlamaCppProbeBackend on 29 ambiguous-band pairs from gold_pairs.csv.

Evaluates:
  1. Process safety & peak RAM utilization under memory constraint.
  2. Per-pair inference latency on CPU.
  3. Raw D_asym values across all 29 pairs.
  4. Directional accuracy on prerequisite vs reversal relationships.
  5. Checkpoints detailed metrics to docs/results/smoke_test_llamacpp_29pairs.json.
"""
from __future__ import annotations

import csv
import json
import os
import pickle
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import psutil

# Root path bootstrap
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "backend"))

from backend.app.services.module1.counterfactual_probe import (
    STANDARD_TEMPLATES,
    WITHOUT_TEMPLATES,
    pairwise_d,
    refine_logits,
)
from backend.app.services.module1.embed import ConceptEmbedder
from backend.app.services.module1.llamacpp_probe import LocalLlamaCppProbeBackend
from backend.app.services.module1.score_pairs import directional_feature


def get_process_rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)


def get_available_ram_gb() -> float:
    return psutil.virtual_memory().available / (1024 ** 3)


def main():
    print("=" * 70)
    print("LightGAP: Local 3B SLM Probe Smoke Test (29 Ambiguous Gold Pairs)")
    print("=" * 70)

    # 1. Hardware baseline
    rss_start = get_process_rss_mb()
    avail_ram_start = get_available_ram_gb()
    print(f"[PRE-FLIGHT] Initial Process RSS: {rss_start:.1f} MB")
    print(f"[PRE-FLIGHT] System Available RAM: {avail_ram_start:.2f} GB / {psutil.virtual_memory().total / (1024**3):.2f} GB")

    # 2. Load LR model and gold pairs
    model_path = os.path.join(REPO_ROOT, "data", "models", "logistic_baseline.pkl")
    gold_path = os.path.join(REPO_ROOT, "data", "labeled", "gold_pairs.csv")
    with open(model_path, "rb") as f:
        lr_model = pickle.load(f)

    gold_pairs: List[Tuple[str, str, int]] = []
    with open(gold_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            src = row.get("concept_a") or row.get("source")
            tgt = row.get("concept_b") or row.get("target")
            lbl = int(row.get("label") or row.get("label_prereq"))
            gold_pairs.append((src.strip(), tgt.strip(), lbl))

    # Embed unique concepts
    concepts = sorted(list({p[0] for p in gold_pairs} | {p[1] for p in gold_pairs}))
    embedder = ConceptEmbedder()
    emb_res = embedder.embed(concepts)
    emb_dict = {c: emb_res.vectors[i] for i, c in enumerate(concepts)}

    # Filter to ambiguous band [0.40, 0.70]
    ambiguous_pairs = []
    for src, tgt, lbl in gold_pairs:
        feat = directional_feature(emb_dict[src], emb_dict[tgt]).reshape(1, -1)
        p = float(lr_model.predict_proba(feat)[0, 1])
        if 0.40 <= p <= 0.70:
            ambiguous_pairs.append({
                "source": src,
                "target": tgt,
                "label": lbl,
                "lr_prob": p,
            })

    print(f"[DATA] Identified {len(ambiguous_pairs)} ambiguous pairs in gold set.")

    # 3. Initialize Local llama.cpp backend
    print("\n[LLAMA.CPP] Initializing LocalLlamaCppProbeBackend (n_ctx=512)...")
    t_load_start = time.perf_counter()
    backend = LocalLlamaCppProbeBackend(n_ctx=512)
    load_time = time.perf_counter() - t_load_start
    rss_after_load = get_process_rss_mb()
    print(f"[LLAMA.CPP] Loaded weights in {load_time:.2f}s.")
    print(f"[LLAMA.CPP] Process RSS after load: {rss_after_load:.1f} MB (Delta: +{rss_after_load - rss_start:.1f} MB)")
    print(f"[LLAMA.CPP] Available RAM remaining: {get_available_ram_gb():.2f} GB")

    # 4. Evaluate the 29 pairs
    print("\n" + "-" * 70)
    print(f"{'#':2s} | {'Pair (u -> v)':<40s} | {'Lbl':3s} | {'P_LR':5s} | {'D_asym':7s} | {'P_ref':5s} | {'Time':5s}")
    print("-" * 70)

    results = []
    pair_latencies = []
    peak_rss = rss_after_load
    directional_correct = 0
    directional_total = 0
    t_eval_start = time.perf_counter()

    for idx, item in enumerate(ambiguous_pairs):
        u, v = item["source"], item["target"]
        lbl = item["label"]
        p_lr = item["lr_prob"]

        t0 = time.perf_counter()
        # Compute D(u -> v) and D(v -> u)
        d_fwd, sp_f, wp_f = pairwise_d(u, v, backend)
        d_bwd, sp_b, wp_b = pairwise_d(v, u, backend)
        d_asym = d_fwd - d_bwd
        dur = time.perf_counter() - t0
        pair_latencies.append(dur)

        # Update peak RSS
        curr_rss = get_process_rss_mb()
        peak_rss = max(peak_rss, curr_rss)

        # Refine logit with alpha=0.5
        p_refined = float(refine_logits([p_lr], [d_asym], alpha=0.5)[0])

        # Directional accuracy test:
        # If label == 1: prerequisite holds => D(u->v) should be positive (D_asym > 0)
        # If label == 0: reversal => D_asym should be negative (D_asym < 0)
        # If label == -1: unrelated pair (excluded from directional accuracy)
        is_dir_correct = None
        if lbl == 1:
            is_dir_correct = (d_asym > 0)
            directional_total += 1
            if is_dir_correct:
                directional_correct += 1
        elif lbl == 0:
            is_dir_correct = (d_asym < 0)
            directional_total += 1
            if is_dir_correct:
                directional_correct += 1

        pair_str = f"{u} -> {v}"
        if len(pair_str) > 40:
            pair_str = pair_str[:37] + "..."
        print(f"{idx+1:2d} | {pair_str:<40s} | {lbl:3d} | {p_lr:.3f} | {d_asym:+6.3f} | {p_refined:.3f} | {dur:4.1f}s")

        results.append({
            "pair_index": idx + 1,
            "source": u,
            "target": v,
            "label": lbl,
            "lr_prob": p_lr,
            "d_forward": d_fwd,
            "d_backward": d_bwd,
            "d_asym": d_asym,
            "standard_ppl_fwd": sp_f,
            "without_ppl_fwd": wp_f,
            "standard_ppl_bwd": sp_b,
            "without_ppl_bwd": wp_b,
            "refined_prob": p_refined,
            "latency_sec": dur,
            "directional_correct": is_dir_correct,
        })

    total_eval_time = time.perf_counter() - t_eval_start
    avg_latency = float(np.mean(pair_latencies))
    dir_acc = (directional_correct / directional_total) if directional_total > 0 else 0.0

    print("-" * 70)
    print(f"[SUMMARY] Evaluated {len(ambiguous_pairs)} pairs in {total_eval_time:.1f}s (avg: {avg_latency:.2f}s/pair).")
    print(f"[SUMMARY] Peak Process RSS: {peak_rss:.1f} MB (Headroom remaining: {get_available_ram_gb():.2f} GB)")
    print(f"[SUMMARY] Directional Accuracy (sign of D_asym): {dir_acc * 100:.1f}% ({directional_correct}/{directional_total})")

    # Save artifact
    out_dir = os.path.join(REPO_ROOT, "docs", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "smoke_test_llamacpp_29pairs.json")

    report = {
        "backend": "LocalLlamaCppProbeBackend",
        "model": "Qwen2.5-3B-Instruct (Q4_K_M GGUF)",
        "hardware_telemetry": {
            "start_process_rss_mb": rss_start,
            "post_load_process_rss_mb": rss_after_load,
            "peak_process_rss_mb": peak_rss,
            "available_ram_start_gb": avail_ram_start,
            "available_ram_end_gb": get_available_ram_gb(),
            "n_ctx": 512,
            "n_threads": backend._n_threads,
        },
        "performance_telemetry": {
            "total_pairs": len(ambiguous_pairs),
            "total_latency_sec": total_eval_time,
            "avg_latency_sec": avg_latency,
            "min_latency_sec": float(np.min(pair_latencies)),
            "max_latency_sec": float(np.max(pair_latencies)),
        },
        "directional_evaluation": {
            "directional_pairs_evaluated": directional_total,
            "directional_correct": directional_correct,
            "directional_accuracy": dir_acc,
        },
        "pairs_detail": results,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[OUTPUT] Full smoke test report saved to {out_file}")


if __name__ == "__main__":
    main()
