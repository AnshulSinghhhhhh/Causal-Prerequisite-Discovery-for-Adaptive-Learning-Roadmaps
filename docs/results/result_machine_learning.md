# LightGAP demo walkthrough — Machine Learning

- **Date:** 2026-09-22
- **Git SHA:** `8a02b0de89d2ad4c3db7ee0f731b6dc0365321bf`
- **Domain slug:** `machine_learning`
- **Model versions used:** S2 = `lr_asym_minilm`, S5 clustering = `Leiden community detection`, S3 CDP = `Qwen/Qwen2.5-3B-Instruct (Teacher-forced NLL probe)`
- **Graph snapshot ID:** `b76988c7-82cb-4894-b5a1-db77bb6a82d3`

## S0 — Concept extraction

- Concepts harvested: **100**
- Source documents: **15** (10 Wikipedia articles, 5 course syllabi)

| Concept | Definition (truncated) | Source |
|---|---|---|
| Machine Learning Articles | Machine Learning Articles.... | Machine Learning Course Syllabus |
| machine-learning programs | Although machine learning has been transformative in some fields, machine-learning program... | Machine learning |
| Supervised Learning | A category of machine learning algorithms that includes techniques like linear regression,... | Machine Learning Course Syllabus |
| statistical algorithms | Machine learning (ML) is a field of study in artificial intelligence concerned with the de... | Machine learning |

## S1 — Candidate generation

- Candidate pairs generated: **2193**
- `s_llm_plaus` evaluated **75** of **2193** surviving pairs (top-ranked by heuristic strength); remaining **2118** pairs use a neutral 0.5 prior.

| Signal | Min | Median | Max |
|---|---|---|---|
| `s_order` | 0.000 | 1.000 | 1.000 |
| `s_cooc` | 0.000 | 0.687 | 1.000 |
| `s_defmention` | 0.000 | 0.000 | 1.000 |
| `s_sim` | 0.304 | 0.522 | 0.993 |
| `s_llm_plaus` | 0.500 | 0.500 | 1.000 |

## S2 — Directional verification

- Candidates admitted at τ = 0.275859: **1390 / 2193**
- Margin distribution: min 0.000, median 0.355, max 0.993

**High-confidence examples:**
- **open-source machine learning software** → **study**, margin 0.9929 (confidence: 0.993)
- **machine learning software applications** → **study**, margin 0.9904 (confidence: 0.991)
- **open-source machine learning software** → **types**, margin 0.9859 (confidence: 0.986)

**Low-margin (ambiguous) examples — these route to S3 if CDP is enabled for this domain:**
- **Machine Learning Language** ↔ **business patterns**, margin 0.0001 (forward: 0.074, reverse: 0.075)
- **large amounts** ↔ **their applications**, margin 0.0004 (forward: 0.055, reverse: 0.055)
- **previously unknown properties** ↔ **Machine learning (ML**, margin 0.0005 (forward: 0.058, reverse: 0.059)

## S4 — DAG assembly

- Nodes: **95**, edges kept: **363**
- Admitted candidate edges: **1390**
- Dropped — threshold: **803**, cycle pruning: **0** (0.0% of admitted edges), transitive: **1027**
- No cycles required pruning; thresholding and transitive reduction formed a strict partial order.

## S5 — Tree induction

- Depth range: **0–13**
- Clusters: **6**

| Cluster label | Member concepts (sample) |
|---|---|
| Foundational & Independent Concepts | Machine Learning Master’s Degree, Machine Learning Design, Machine Learning Bachelor’s Degree Techno India University B.Sc, Artificial Intelligence PG, Machine Learning AI Engineering Artificial Intelligence |
| Cluster 1 | Machine Learning Articles, standard machine learning approach, machine learning (and other artificial intelligence) methods, chosen examples, their performance, its mathematical and computational nature |
| Cluster 4 | machine-learning programs, Training models, Many learning problems, Feature learning, Probably approximately correct learning, inductive machine learning |
| Cluster 2 | Supervised Learning, machine learning applications, Advanced Machine Learning, various domains, their applications, essential machine learning concepts |

## S6 — Path planning

- Time budget used: **300** minutes
- Path produced (20 nodes):
commerce, customer behaviour, problems, study, Statistical Learning Theory, difficulty, some fields, Introduction, types, decision rules, minority populations, biased data ...
- Excluded (over budget): 80 non-critical / redundant nodes

## S7 — Sample quiz item

**Node:** Machine Learning Articles
**Stem:** A data scientist is writing a technical article explaining the difference between supervised and unsupervised learning. Which of the following statements correctly distinguishes the two based on the nature of the training data?
**Correct:** Supervised learning uses labeled data where the input is paired with the correct output, whereas unsupervised learning uses unlabeled data to find hidden structures or patterns without predefined output labels.
**Prerequisite distractor:** Supervised learning requires a large amount of data to be computationally efficient, while unsupervised learning can operate effectively with very small datasets because it does not need to calculate error gradients.
**Conceptual distractor:** Supervised learning is used to predict continuous values (regression), while unsupervised learning is used to predict categorical classes (classification), regardless of whether labels are present.
**Misconception tag:** `Confusing algorithm types (regression/classification) with learning paradigms (supervised/unsupervised)`

## Summary (paper-ready)

The automated LightGAP pipeline discovered the prerequisite dependency graph for **Machine Learning** algorithmically. Across S0 through S6, 15 harvested corpus documents (10 Wikipedia articles, 5 course syllabi) yielded 100 canonical concepts and 2193 candidate pairs. Directional verification with the calibrated asymmetric feature classifier admitted 1390 pairs, and transitive reduction preserved 363 prerequisite links forming a Directed Acyclic Graph. Cycle pruning removed 0 contradictory edges (0.0% of admitted edges). Topological longest-path layering established an empirical depth range of 0–13 partitioned into 6 Leiden semantic clusters. The precedence-constrained path planner scheduled an optimal 20-node learning sequence fulfilling the 300-minute constraint.
