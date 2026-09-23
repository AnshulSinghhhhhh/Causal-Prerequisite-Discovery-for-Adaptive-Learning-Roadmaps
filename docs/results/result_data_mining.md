# LightGAP demo walkthrough — data mining

- **Date:** 2026-09-23
- **Git SHA:** `25ff2fddd6a1421783e491b849c5d9aaa0df3f5a`
- **Domain slug:** `data_mining`
- **Model versions used:** S2 = `lr_asym_minilm`, S5 clustering = `Leiden community detection`, S3 CDP = `Qwen/Qwen2.5-3B-Instruct (Teacher-forced NLL probe)`
- **Graph snapshot ID:** `2c31e7b2-a58a-4887-850a-470bc7ed4531`

## S0 — Concept extraction

- Concepts harvested: **100**
- Source documents: **11** (10 Wikipedia articles, 1 course syllabi)

| Concept | Definition (truncated) | Source |
|---|---|---|
| ACM SIGKDD Conference | KDD Conference – ACM SIGKDD Conference on Knowledge Discovery and Data Mining.... | Data mining |
| Association rule learning (dependency modeling | A technique that searches for relationships between variables, often used for market baske... | Data mining |
| Association rule mining | A method for discovering dependencies among data items.... | Data mining |
| automatic or automatic analysis | The actual data mining task is the semi-automatic or automatic analysis of massive quantit... | Data mining |

## S1 — Candidate generation

- Candidate pairs generated: **3421**
- `s_llm_plaus` evaluated **300** of **3421** surviving pairs (top-ranked by heuristic strength); remaining **3121** pairs use a neutral 0.5 prior.

| Signal | Min | Median | Max |
|---|---|---|---|
| `s_order` | 0.000 | 1.000 | 1.000 |
| `s_cooc` | 0.000 | 0.000 | 1.000 |
| `s_defmention` | 0.000 | 0.000 | 1.000 |
| `s_sim` | 0.272 | 0.532 | 0.986 |
| `s_llm_plaus` | 0.000 | 0.500 | 1.000 |

## S2 — Directional verification

- Candidates admitted at τ = 0.275859: **2436 / 3421**
- Margin distribution: min 0.000, median 0.476, max 1.000

**High-confidence examples:**
- **Proprietary data-mining software** → **statistics**, margin 0.9999 (confidence: 1.000)
- **Proprietary data-mining software** → **science**, margin 0.9998 (confidence: 1.000)
- **Proprietary data-mining software** → **information**, margin 0.9997 (confidence: 1.000)

**Low-margin (ambiguous) examples — these route to S3 if CDP is enabled for this domain:**
- **CRISP-DM methodology** ↔ **more accurate prediction results**, margin 0.0000 (forward: 0.053, reverse: 0.053)
- **data sets** ↔ **Pre**, margin 0.0001 (forward: 0.041, reverse: 0.041)
- **Subspace clustering** ↔ **warehousing**, margin 0.0004 (forward: 0.178, reverse: 0.179)

## S4 — DAG assembly

- Nodes: **100**, edges kept: **232**
- Admitted candidate edges: **2436**
- Dropped — threshold: **985**, cycle pruning: **0** (0.0% of admitted edges), transitive: **2204**
- No cycles required pruning; thresholding and transitive reduction formed a strict partial order.

## S5 — Tree induction

- Depth range: **0–22**
- Clusters: **6**

| Cluster label | Member concepts (sample) |
|---|---|
| Cluster 2 | ACM SIGKDD Conference, bioinformatics, certain groups, Classification analysis, Cluster analysis, Computer science conferences |
| Cluster 3 | Association rule learning (dependency modeling, CRISP-DM methodology, CRISP‑DM, Data dredging, Data mining and statistics software, data mining companies |
| Cluster 4 | Association rule mining, automatic or automatic analysis, intelligent methods, K-optimal pattern discovery, many different data mining techniques, massive data sets |
| Cluster 6 | businesses, continuous production, Data mining, Data Preprocessing, essential concepts, improved strategies |

## S6 — Path planning

- Time budget used: **300** minutes
- Path produced (20 nodes):
statistics, science, information, customers, trouble, knowledge, concise, further use, use, Ideation, finance, medicine ...
- Excluded (over budget): 80 non-critical / redundant nodes

## S7 — Sample quiz item

**Node:** Data mining
**Stem:** A company has a database containing 10 million customer transaction records. They want to identify hidden patterns, such as which products are frequently purchased together, to optimize inventory. Which of the following best describes the primary technical approach they should employ?
**Correct:** Applying algorithms from machine learning and statistics to scan the massive dataset for non-trivial patterns and correlations.
**Prerequisite distractor:** Writing a simple SQL query to retrieve all transactions where the total amount exceeds $100, as this directly filters the relevant data.
**Conceptual distractor:** Manually reviewing a random sample of 100 transactions to guess the purchasing trends, as this is the most accurate way to understand customer behavior.
**Misconception tag:** `Confusing data retrieval with pattern discovery`

## Summary (paper-ready)

The automated LightGAP pipeline discovered the prerequisite dependency graph for **data mining** algorithmically. Across S0 through S6, 11 harvested corpus documents (10 Wikipedia articles, 1 course syllabi) yielded 100 canonical concepts and 3421 candidate pairs. Directional verification with the calibrated asymmetric feature classifier admitted 2436 pairs, and transitive reduction preserved 232 prerequisite links forming a Directed Acyclic Graph. Cycle pruning removed 0 contradictory edges (0.0% of admitted edges). Topological longest-path layering established an empirical depth range of 0–22 partitioned into 6 Leiden semantic clusters. The precedence-constrained path planner scheduled an optimal 20-node learning sequence fulfilling the 300-minute constraint.
