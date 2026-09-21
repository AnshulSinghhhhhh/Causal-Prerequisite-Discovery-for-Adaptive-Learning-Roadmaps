# LightGAP demo walkthrough — Machine Learning

- **Date:** 2026-09-21
- **Git SHA:** `cv-youden-recalibrated-20260921`
- **Domain slug:** `machine_learning`
- **Model versions used:** S2 = `lr_asym_minilm`, S5 clustering = `Leiden community detection`, S3 CDP = `Qwen/Qwen2.5-3B-Instruct (Teacher-forced NLL probe)`
- **Graph snapshot ID:** `6df1d75f-3a53-4c8c-9678-f1f9583e6709`

## S0 — Concept extraction

- Concepts harvested: **79**
- Source documents: **15** (10 Wikipedia articles, 5 course syllabi)

| Concept | Definition (truncated) | Source |
|---|---|---|
| Machine learning training | Technical concept definition... | Machine Learning Course Syllabus |
| machine-learning AI | Technical concept definition... | Artificial intelligence |
| machine learning solutions | Technical concept definition... | Machine Learning Course Syllabus |
| machine learning tasks | Technical concept definition... | Machine Learning Course Syllabus |

## S1 — Candidate generation

- Candidate pairs generated: **1491**

| Signal | Min | Median | Max |
|---|---|---|---|
| `s_order` | 0.500 | 1.000 | 1.000 |
| `s_cooc` | 0.000 | 0.000 | 1.000 |
| `s_defmention` | 0.000 | 0.000 | 0.000 |
| `s_sim` | 0.281 | 0.514 | 0.882 |
| `s_llm_plaus` | 0.500 | 0.500 | 1.000 |

## S2 — Directional verification

- Candidates admitted at τ = 0.275859: **827 / 1491**
- Margin distribution: min 0.000, median 0.262, max 0.978

**High-confidence examples:**
- **statistical algorithms** → **open-source machine learning software**, margin 0.9775 (confidence: 0.980)
- **statistical algorithms** → **decision tree-based models**, margin 0.9605 (confidence: 0.967)
- **machine intelligence** → **practical machine-learning projects**, margin 0.9565 (confidence: 0.958)

**Low-margin (ambiguous) examples — these route to S3 if CDP is enabled for this domain:**
- **machine intelligence** ↔ **Machine learning**, margin 0.0004 (forward: 0.151, reverse: 0.151)
- **ensemble learning** ↔ **Trained models**, margin 0.0005 (forward: 0.068, reverse: 0.068)
- **prediction models** ↔ **Decision Tree**, margin 0.0007 (forward: 0.152, reverse: 0.151)

## S4 — DAG assembly

- Nodes: **75**, edges kept: **310**
- Admitted candidate edges: **827**
- Dropped — threshold: **664**, cycle pruning: **0** (0.0% of admitted edges), transitive: **517**
- No cycles required pruning; thresholding and transitive reduction formed a strict partial order.

## S5 — Tree induction

- Depth range: **0–7**
- Clusters: **6**

| Cluster label | Member concepts (sample) |
|---|---|
| Machine Learning | machine-learning AI, Classifiers, Rule-based machine learning, supervised models, AI models, pattern recognition |
| Machine Learning | Machine learning training, machine learning solutions, machine-learning programs, classification models, validating machine learning models, fundamental supervised learning algorithms |
| Machine Learning | machine learning tasks, Our machine learning, machine learning success, self-supervised learning, Feature Store Machine Learning, SVM |
| Machine Learning | Machine learning aids, Feature selection, classifying, decision networks, machine learning optimization, Another machine learning system |

## S6 — Path planning

- Time budget used: **300** minutes
- Path produced (2 nodes):
statistical algorithms, Machine learning
- Excluded (over budget): 77 non-critical / redundant nodes

## S7 — Sample quiz item

**Node:** Machine learning
**Stem:** What is required for a machine learning model to improve over time?
**Correct:** Access to relevant data
**Prerequisite distractor:** A more powerful computer
**Conceptual distractor:** No external input; it can learn on its own
**Misconception tag:** `data_not_required`

## Summary (paper-ready)

The automated LightGAP pipeline discovered the prerequisite dependency graph for **Machine Learning** algorithmically. Across S0 through S6, 15 harvested corpus documents (10 Wikipedia articles, 5 course syllabi) yielded 79 canonical concepts and 1491 candidate pairs. Directional verification with the calibrated asymmetric feature classifier admitted 827 pairs, and transitive reduction preserved 310 prerequisite links forming a Directed Acyclic Graph. Cycle pruning removed 0 contradictory edges (0.0% of admitted edges). Topological longest-path layering established an empirical depth range of 0–7 partitioned into 6 Leiden semantic clusters. The precedence-constrained path planner scheduled an optimal 2-node learning sequence fulfilling the 300-minute constraint.
