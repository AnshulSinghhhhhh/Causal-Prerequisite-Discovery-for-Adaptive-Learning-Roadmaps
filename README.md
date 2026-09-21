# LightGAP — Lightweight Graph-based Adaptive Pathway system

An Intelligent Tutoring System that replaces static, human-curated concept
roadmaps with an **automatically-built, adaptive one**, running end-to-end on a
free-tier cloud GPU (single Colab T4) or a consumer laptop CPU with a total
inference footprint **under 3GB**.

Four sequential, iterative modules (Section 14 build order — they are gated,
not independent tools):

1. **Module 1 — Prerequisite Determination Engine** (`backend/app/services/module1/`)
   predicts directed prerequisite relations from concept text via
   `all-MiniLM-L6-v2` embeddings → logistic-regression baseline → DirGCN
   (standard **and** attention-weighted variants) → SLM counterfactual
   perplexity probe.
2. **Module 2 — Graph Construction & Constraint Optimization**
   (`module2/`) assembles a valid DAG (threshold → Tarjan SCC → transitive
   reduction) and solves a precedence-constrained knapsack via
   `scipy.optimize.milp`.
3. **Module 3 — Content Aggregator & Diagnostic Quiz** (`module3/`) attaches
   re-ranked external resources and misconception-tagged three-option quizzes.
4. **Module 4 — Dynamic Graph Rewriter & Remediation** (`module4/`) tracks
   exponential memory decay (SM-2/FSRS) and mutates the live graph when a
   learner reveals a misconception.

The interactive output is a self-healing roadmap rendered as a **React Flow**
graph, backed by a **FastAPI** service exposing one JSON contract
(`backend/app/models/schema.py`).

---

## Repository layout

```
lightgap/
  backend/
    app/
      main.py                 # FastAPI app (JSON contract, Section 10)
      models/schema.py        # THE canonical contract (pydantic)
      routers/                # graph.py / quiz.py / remediation.py
      services/
        graph_model.py        # THE canonical in-memory graph (Section 1.4)
        config.py             # THE canonical hyperparameters
        module1/ ... module4/
      session_store.py        # one live graph per learner session
      requirements.txt
  frontend/                   # React Flow + typed client
  data/                       # gitignored artifacts; gold_pairs.csv committed
  tests/backend/              # unit + leakage + contract + integration
  scripts/                    # download_datasets.sh + run_full_pipeline.py
  docs/                       # labeling guide + results snapshots
```

## Quick start (offline-safe — no data or model weights required)

```bash
# Backend tests (all 92 pass with only CPU + no network)
cd lightgap
python -m pytest tests/backend -q

# Import smoke test of every service module
python -c "from backend.app.main import app; print('ok')"

# Run the FastAPI service (optional)
cd backend && uvicorn app.main:app --reload --port 8000
```

The full unit suite exercises every module with **synthetic data**: no GPU, no
downloaded dataset, no model download. The real three-track protocol (Section
11) runs via `scripts/run_full_pipeline.py` once datasets/models are available.

## Running the real pipeline (requires network + model weights)

```bash
# 1. Download datasets (AL-CPL, PnPR-GCN split, LectureBankCD, Univ. Course)
bash scripts/download_datasets.sh            # add --refd only to re-test RefD

# 2. Encode concepts (downloads all-MiniLM-L6-v2, caches to data/embeddings)
python scripts/run_full_pipeline.py --stage embed --concepts my_concepts.json

# 3. Fit + calibrate the baseline, apply the threshold once to validation
python scripts/run_full_pipeline.py --stage stage1 \
  --train_pairs train.json --valid_pairs valid.json

# 4. Simulated learners (Track B)
python scripts/run_full_pipeline.py --stage trackB --synthetic
```

## Governing methodology (read before touching thresholds)

- **Non-circular calibration (Section 1.1):** every tunable threshold and model
  variant is selected via training-domain CV (Youden's J), frozen, then applied
  **exactly once** to the held-out / gold set. `calibration.py` is the single
  implementation; its tests assert the two leakage modes (threshold
  inconsistency; ensemble model-selection) cannot see the held-out file.
- **One implementation per shared concern (Section 1.4):** the graph data model
  (`graph_model.py`), the JSON contract (`schema.py`), and the calibration logic
  each have exactly one canonical copy.
- **Report negative results plainly (Section 1.2).** A documented negative
  result is a deliverable.

## Open question (Section 12) — resolve, don't assume

Whether a GNN component helps is unresolved across lineages. Both DirGCN
variants (standard and attention-weighted) are built; run the comparison under
the full non-circular protocol before deciding what ships.

## Hardware budget (Section 9/Table 2)

| Component | Footprint | Time |
|---|---|---|
| Embeddings (MiniLM) | ~80MB | <0.5s / 100 concepts |
| DirGCN (2-layer) | <300MB | <15s train (T4) |
| SLM probe (4-bit 3B) | ~2.2GB | ~0.8s / pair (ambiguous only) |
| ILP (scipy milp) | system | <0.1s CPU |
| Vector index (in-mem) | ~50MB | <5ms query |

Total target **< 3GB**.