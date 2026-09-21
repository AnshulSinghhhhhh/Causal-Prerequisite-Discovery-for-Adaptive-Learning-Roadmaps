# LightGAP demo walkthrough — Machine Learning

- **Date:** 2026-09-21
- **Git SHA:** `50e403ee61a9ce82f38803feed8aff9fa8d127c2`
- **Domain slug:** `machine_learning`
- **Model versions used:** S2 = `lr_asym_minilm`, S5 clustering = `Leiden community detection`, S3 CDP = `Qwen/Qwen2.5-3B-Instruct (Teacher-forced NLL probe)`
- **Graph snapshot ID:** `97a8ddb3-8a22-4db4-8f12-466dbb503497`

## S0 — Concept extraction

- Concepts harvested: **100**
- Source documents: **15** (10 Wikipedia articles, 5 course syllabi)

| Concept | Definition (truncated) | Source |
|---|---|---|
| Machine Learning Articles | Machine Learning Articles.... | Machine Learning Course Syllabus |
| machine-learning programs | Although machine learning has been transformative in some fields, machine-learning program... | Machine learning |
| Machine learning (ML | A field of AI that develops statistical algorithms capable of learning from data, generali... | Machine learning |
| learning theory | The computational analysis of machine learning algorithms and their performance is a branc... | Machine learning |

## S1 — Candidate generation

- Candidate pairs generated: **2153**
- `s_llm_plaus` evaluated **75** of **2153** surviving pairs (top-ranked by heuristic strength); remaining **2078** pairs use a neutral 0.5 prior.

| Signal | Min | Median | Max |
|---|---|---|---|
| `s_order` | 0.000 | 1.000 | 1.000 |
| `s_cooc` | 0.000 | 0.683 | 1.000 |
| `s_defmention` | 0.000 | 0.000 | 1.000 |
| `s_sim` | 0.325 | 0.527 | 0.993 |
| `s_llm_plaus` | 0.500 | 0.500 | 1.000 |

## S2 — Directional verification

- Candidates admitted at τ = 0.275859: **1359 / 2153**
- Margin distribution: min 0.000, median 0.353, max 0.993

**High-confidence examples:**
- **open-source machine learning software** → **study**, margin 0.9929 (confidence: 0.993)
- **real-world machine learning projects** → **study**, margin 0.9928 (confidence: 0.993)
- **machine learning software applications** → **study**, margin 0.9904 (confidence: 0.991)

**Low-margin (ambiguous) examples — these route to S3 if CDP is enabled for this domain:**
- **their applications** ↔ **large amounts**, margin 0.0004 (forward: 0.055, reverse: 0.055)
- **previously unknown properties** ↔ **Machine learning (ML**, margin 0.0005 (forward: 0.058, reverse: 0.059)
- **Generalization** ↔ **Classification**, margin 0.0009 (forward: 0.122, reverse: 0.123)

## S4 — DAG assembly

- Nodes: **95**, edges kept: **375**
- Admitted candidate edges: **1359**
- Dropped — threshold: **794**, cycle pruning: **0** (0.0% of admitted edges), transitive: **984**
- No cycles required pruning; thresholding and transitive reduction formed a strict partial order.

## S5 — Tree induction

- Depth range: **0–12**
- Clusters: **6**

| Cluster label | Member concepts (sample) |
|---|---|
| Foundational & Independent Concepts | Machine Learning Design, Machine Learning Bachelor’s Degree Techno India University B.Sc, Machine Learning Master’s Degree, Artificial Intelligence PG, Machine Learning AI Engineering Artificial Intelligence |
| Cluster 4 | Machine Learning Articles, Classification, machine learning (and other artificial intelligence) methods, optimization, Intellipaat, machine learning techniques |
| Cluster 3 | machine-learning programs, all advanced techniques, machine learning software applications, their desired outputs, Feature learning, machine learning's vulnerability |
| Cluster 5 | Machine learning (ML, basic algorithms, data mining and data management, many practical machine learning applications, Bayesian methods, supervised models |

## S6 — Path planning

- Time budget used: **300** minutes
- Path produced (20 nodes):
commerce, customer behaviour, problems, study, Statistical Learning Theory, difficulty, some fields, decision rules, minority populations, types, Introduction, biased data ...
- Excluded (over budget): 80 non-critical / redundant nodes

## S7 — Sample quiz item

**Node:** Machine Learning Articles
**Stem:** A data scientist is writing a technical article explaining the difference between Supervised and Unsupervised Learning. Which of the following statements correctly distinguishes the two based on the nature of the training data?
**Correct:** Supervised learning uses labeled data where the input is paired with the correct output, while unsupervised learning uses unlabeled data to find hidden structures or patterns.
**Prerequisite distractor:** Supervised learning requires a human to manually correct the model's errors during the training process, whereas unsupervised learning relies entirely on the model's internal feedback loop without any human intervention.
**Conceptual distractor:** Supervised learning is used to predict continuous values (regression), while unsupervised learning is used to predict discrete categories (classification).
**Misconception tag:** `Confusing algorithm types with data labeling status`

## Summary (paper-ready)

The automated LightGAP pipeline discovered the prerequisite dependency graph for **Machine Learning** algorithmically. Across S0 through S6, 15 harvested corpus documents (10 Wikipedia articles, 5 course syllabi) yielded 100 canonical concepts and 2153 candidate pairs. Directional verification with the calibrated asymmetric feature classifier admitted 1359 pairs, and transitive reduction preserved 375 prerequisite links forming a Directed Acyclic Graph. Cycle pruning removed 0 contradictory edges (0.0% of admitted edges). Topological longest-path layering established an empirical depth range of 0–12 partitioned into 6 Leiden semantic clusters. The precedence-constrained path planner scheduled an optimal 20-node learning sequence fulfilling the 300-minute constraint.
