# LightGAP — Master Results Summary & Conference Evidence Compilation

This document provides the consolidated master summary of all validated empirical results, architecture ablations, domain walkthroughs, and architectural limitations for the LightGAP conference paper. 

---

## 1. Document Index & Navigation

The complete empirical and methodological record is organized across the following detailed reports:

* **Part A — Core Validations & Methodological Audits:**  
  [`existing_results_writeup.md`](file:///c:/Users/anshu/Documents/newstart/aadfadfadsfadsf/lightgap/docs/results/conference_data/existing_results_writeup.md)  
  *Contains: Module 1 baseline metrics on AL-CPL and held-out gold sets, six architecture ablations, Module 2 correctness audits, PostgreSQL schema-level non-circular evaluation trigger, Counterfactual Dependency Probing (CDP) case study, and postmortems of two resolved data leakage bugs.*
* **Part B — Clean Worked Example (Benchmark Domain):**  
  [`result_data_mining.md`](file:///c:/Users/anshu/Documents/newstart/aadfadfadsfadsf/lightgap/docs/results/result_data_mining.md)  
  *Contains: Full end-to-end pipeline run (S0–S7) on the native AL-CPL "data mining" domain, verifying structural noise elimination, unpinned candidate plausibility scoring ($0.000 \le s_{\text{llm\_plaus}} \le 1.000$), strict partial order induction ($0$ cycle drops), 20-node optimal ILP learning path, and diagnostic assessment generation.*
* **Part C — Diagnostic Assessment & Runtime Graph Remediation:**  
  [`module3_module4_writeup.md`](file:///c:/Users/anshu/Documents/newstart/aadfadfadsfadsf/lightgap/docs/results/conference_data/module3_module4_writeup.md)  
  *Contains: Scope declaration for Modules 3 & 4, three illustrated diagnostic quiz items (Data Mining, Machine Learning, Photosynthesis), step-by-step before/after graph mutation walkthrough of the live remediation cycle (`GraphRewriter`), and mathematical formulation of the exponential memory decay model.*
* **Part D — Limitations & Findings: The Broad-Domain Failure Mode:**  
  [`limitations_and_findings.md`](file:///c:/Users/anshu/Documents/newstart/aadfadfadsfadsf/lightgap/docs/results/conference_data/limitations_and_findings.md)  
  *Contains: Formal writeup of the core limitation finding: "naive candidate filtering, applied in sequence, starves output on broad domains even when each individual filter is independently correct", analyzing the ML vs. Data Mining breadth contrast, numeric/syllabus artifact survival, near-duplicate merge trade-offs, and cycle pruning rate swings.*

---

## 2. Paper-Ready Module Summaries

### Module 1: Pairwise Prerequisite Link Prediction
Module 1 predicts directed pedagogical dependencies between concept pairs using an 80 MB sentence transformer (`all-MiniLM-L6-v2`) coupled with a frozen Logistic Regression classifier operating on a 1536-dimensional asymmetric feature representation:
$$f_{\text{dir}}(u, v) = \big[\, h_u \;\parallel\; h_v \;\parallel\; (h_u - h_v) \;\parallel\; (h_u \odot h_v) \,\big] \in \mathbb{R}^{1536}$$
Decision threshold $\tau_{\text{edge}} = 0.275859$ was calibrated strictly on training data via Youden's $J$ statistic across 5-fold graph-aware cross-validation on the AL-CPL benchmark, yielding a mean ROC-AUC of **0.9327**, Precision of **0.7274**, Recall of **0.8698**, and F1 of **0.7920**. Evaluated one-shot against the held-out gold benchmark (`gold_pairs_v1`, 105 pairs) protected by a PostgreSQL schema-level anti-leakage trigger, the frozen classifier achieved an ROC-AUC of **0.8022**, Accuracy of **0.6381**, Precision of **0.8421**, Recall of **0.5000**, and F1 of **0.6275**. Subclass breakdown confirmed strong directional discrimination: the model rejected **90.0%** of reversed pairs (18/20) and **81.0%** of unrelated pairs (17/21). Ambiguous directional margins are arbitrated by secondary Counterfactual Dependency Probing (CDP) via normalized teacher-forced SLM perplexity differences, correctly rejecting near-synonyms without corrupting global graph topology.

### Module 2: DAG Induction, Topological Stratification & Constrained Path Planning
Module 2 converts continuous pairwise edge predictions into a topologically valid, pedagogical Directed Acyclic Graph (DAG) and solves for an optimal learning sequence under user constraints. Cyclical dependencies are resolved via Tarjan’s Strongly Connected Components (SCC) algorithm by iteratively severing the edge of minimum confidence within detected cycles. Transitive reduction (`networkx.transitive_reduction`) eliminates redundant shortcut edges to maximize structural readability while preserving complete reachability (e.g., pruning 2,204 transitive links in *Data Mining* to leave 232 irreducible edges). Concepts are stratified into pedagogical tiers via topological longest-path leveling and grouped into thematic clusters using Leiden community detection. Finally, path planning is formulated as a precedence-constrained knapsack problem and solved via `scipy.optimize.milp` in under 0.05 seconds, maximizing cumulative downstream concept enablement under a strict time budget (guaranteeing 100% adherence to precedence constraints).

### Module 3: Diagnostic Misconception-Tagged Assessment Engine
Module 3 attaches targeted diagnostic questions to concept nodes in the induced DAG to localize learning failure. Unlike generic multiple-choice items, each question enforces a 3-role distractor schema: Option A confirms target mastery; Option B presents a distractor pre-mapped to an upstream prerequisite misconception vector $M_k$; and Option C isolates within-concept confusion. Selecting Option B signals an unmastered upstream dependency, supplying Module 4 with the exact $(target, misconception\_id)$ tuple needed for intervention without secondary database lookups. Illustrated across three diverse corpora (*Data Mining*, *Machine Learning*, and *Photosynthesis*), the engine reliably identifies specific cognitive errors—such as conflating relational data retrieval with exploratory pattern discovery or confusing electron transport chemiosmosis with direct substrate-level phosphorylation.

### Module 4: Adaptive Memory Decay & Runtime Graph Rewriting
Module 4 achieves closed-loop curriculum adaptation through spaced memory decay tracking and real-time graph rewriting. Concept retention is modeled via exponential decay $R(\Delta t) = \exp(-\Delta t / S_v)$ parameterized by SM-2 and FSRS stability updates, automatically flagging concepts whose retention drops below $\tau_{\text{decay}} = 0.5$. When a learner triggers a prerequisite distractor (Option B) on target $v$, the dynamic graph rewriter (`GraphRewriter`) executes a live, non-destructive mutation on the active session graph: it snapshots existing incoming prerequisite links, injects a temporary remediation node $RM_k$, reroutes incoming upstream edges through $RM_k$, inserts a burden edge $RM_k \to v$, and locks $v$. Once the learner demonstrates remediation mastery (e.g., 2 consecutive successful reviews), the rewriter removes $RM_k$, restores original edge topology from the snapshot, and unlocks $v$ for progression.

---

## 3. Architecture Ablation Table: Negative & Inconclusive Results

All candidate architectures evaluated against the frozen asymmetric Logistic Regression baseline on identical 5-fold cross-validation splits:

| Architecture Variant | Metric Delta vs. Baseline | Causal Explanation for Rejection |
|---|---|---|
| **DirGCN Standard** (2-layer directed graph convolutional network) | $\Delta\text{F1} \approx -0.052$ (CV F1 = 0.740 vs. 0.792 baseline; mean raw score 0.4837) | Candidate graphs generated prior to topological pruning contain substantial sparsity and false-positive candidate edges. Spatial graph convolutions aggregate representations across unverified incoming and outgoing edges, diluting the sharp pairwise directional gradient $(h_u - h_v)$ with neighborhood noise and over-smoothing representations across dense concept communities. |
| **DirGCN Attention-Weighted** (AttnDirGCN with learnable edge attention) | $\Delta\text{F1} \approx -0.002$ (CV F1 = 0.790 vs. 0.792 baseline; +196 params, attention entropy 2.9953) | Attention mechanisms designed to dynamically downweight spurious candidate edges degraded toward near-uniform weighting across dense subgraphs or overfitted to training domain topologies. The marginal capacity gain failed to provide statistically meaningful directional separation out of domain while adding parameter overhead and inference latency. |
| **Definition-Enrichment** (Concatenating extracted definition prose into embedding) | $\Delta\text{F1} \approx -0.041$ (CV F1 = 0.751 vs. 0.792 baseline; cross-domain recall dropped ~12%) | Definitions harvested from heterogeneous web and syllabus documents vary widely in length, style, and lexical quality. Concatenating full definitions introduced domain-specific terminology and descriptive distractors that diluted the geometric distance of canonical concept titles, reducing cosine gradient stability. |
| **RefD Augmentation** (Data augmentation from Wikipedia Reference Distance dataset) | Severe cross-domain transfer regression ($\Delta\text{F1} < -0.100$ on held-out gold) | RefD labels dependencies based on Wikipedia hyperlink asymmetry and citation proximity rather than pedagogical prerequisites. The resulting training signal introduced conflicting directional labels for educational concepts, shifting the learned decision hyperplane away from instructional dependency. |
| **GNN + Logistic Regression Ensemble** | $\Delta\text{F1} \approx +0.003$ on CV, but $\Delta\text{F1} \approx -0.035$ on held-out gold transfer | Ensembling soft probabilities from DirGCN and Logistic Regression added hyperparameter tuning fragility without providing orthogonal signal. Because the GNN component suffered from out-of-domain topological sensitivity, ensembling eroded the high precision (0.8421) of the standalone linear classifier on transfer sets. |
| **SLM Perplexity Probe** (Dense edge prediction via 4-bit Qwen2.5-3B teacher-forcing) | $\Delta\text{F1} \approx -0.021$ as primary scorer; latency prohibitive (~34.5s/batch, ~0.8s/pair) | Normalized teacher-forced perplexity differences $D(u \to v)$ are heavily confounded by the base language model's pretraining exposure to concept $v$, conflating term familiarity with true prerequisite dependency. Due to noise and latency, it is unviable as a primary scorer and only justifiable as a sparse secondary arbitrator for ambiguous pairs. |

---

## 4. What is Still Missing for the Paper (Explicit Gap List)

To preserve scientific rigor, we explicitly enumerate the remaining empirical and experimental gaps prior to camera-ready submission:

1. **Track C Human Cohort Study (Pedagogical Evaluation):**  
   Modules 3 and 4 are currently validated at the systems, software, and unit-test levels, with concrete before/after walkthroughs. However, they lack empirical evaluation on a human learner cohort (e.g., randomized A/B trial measuring learning gain, time-to-mastery, or retention curves against standard linear curricula).
2. **Multi-Domain Psychometric Item Analysis:**  
   While Module 3 quiz items were generated and qualitatively audited across three domains (*Data Mining*, *Machine Learning*, and *Photosynthesis*), large-scale psychometric evaluation (e.g., Item Response Theory discrimination parameters, empirical distractor selection frequency across diverse student cohorts) has not been performed.
3. **Joint Pruning Optimization (Resolving Cascading Starvation):**  
   As documented in Part D, sequential feedforward filtering (S0 $\to$ S1 $\to$ S2 $\to$ S4) starves output on broad, uncurated domains. Formulating concept selection, directional prediction, and cycle elimination as a joint constrained optimization problem remains future algorithmic work.
4. **Multi-Domain Held-Out Gold Expansion:**  
   The held-out evaluation set (`gold_pairs_v1`, 105 pairs) is currently concentrated in Machine Learning curricula. Expanding non-circular, schema-locked gold sets to non-CS STEM disciplines (e.g., Organic Chemistry, Cellular Biology) is necessary to measure cross-domain transfer bounds across disparate scientific ontologies.

---

## 5. Verification & Test Discipline

Prior to finalizing this compilation, the entire backend verification suite was executed to confirm that all hardening changes, structural filters, and module integrations preserve 100% test passing status:

* **Test Execution Command:** `python -m pytest tests/backend -q`
* **Test Suite Status:** **115 passed** (100% passing across 17 test modules, 0 failures, 0 regressions)
* **Coverage Scope:**
  - S0 concept extraction and structural noise suppression (`test_s0_hardening.py`: 9 tests)
  - S1 plausibility scoring and asymmetric fallback floor resolution (`test_s1_hardening.py`: 6 tests)
  - M1 directional classifier and calibration contracts (`test_calibration.py`, `test_weak_supervision.py`, `test_dirgcn.py`, `test_llamacpp_probe.py`: 31 tests)
  - M2 graph induction, transitive reduction, and ILP path optimization (`test_build_graph.py`, `test_path_optimizer.py`, `test_hierarchy_edge.py`: 20 tests)
  - M3 diagnostic assessment engine and distractor mappings (`test_quiz_engine.py`, `test_content_aggregator.py`: 10 tests)
  - M4 memory decay scheduling and dynamic graph rewriting (`test_decay_model.py`, `test_graph_rewriter.py`: 20 tests)
  - Database schema contracts and integration smoke runs (`test_schema_contract.py`, `test_graph_model.py`, `test_pipeline.py`, `test_integration_smoke.py`: 19 tests)
