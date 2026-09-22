# LightGAP demo walkthrough — Photosynthesis

- **Date:** 2026-09-22
- **Git SHA:** `8a02b0de89d2ad4c3db7ee0f731b6dc0365321bf`
- **Domain slug:** `photosynthesis`
- **Model versions used:** S2 = `lr_asym_minilm`, S5 clustering = `Leiden community detection`, S3 CDP = `Qwen/Qwen2.5-3B-Instruct (Teacher-forced NLL probe)`
- **Graph snapshot ID:** `19693fc3-dbf1-4312-9a89-fb2bd9da47a2`

## S0 — Concept extraction

- Concepts harvested: **100**
- Source documents: **11** (10 Wikipedia articles, 1 course syllabi)

| Concept | Definition (truncated) | Source |
|---|---|---|
| 0.1% to 8% | Actual plants' photosynthetic efficiency varies with the frequency of the light being conv... | Photosynthesis |
| 3–6% | Plants usually convert light into chemical energy with a photosynthetic efficiency of 3–6%... | Photosynthesis |
| Actual plants' photosynthetic efficiency | Actual plants' photosynthetic efficiency varies with the frequency of the light being conv... | Photosynthesis |
| Alarm photosynthesis | Alarm photosynthesis represents a photosynthetic variant to be added to the well-known C4 ... | Photosynthesis |

## S1 — Candidate generation

- Candidate pairs generated: **3832**
- `s_llm_plaus` evaluated **75** of **3832** surviving pairs (top-ranked by heuristic strength); remaining **3757** pairs use a neutral 0.5 prior.

| Signal | Min | Median | Max |
|---|---|---|---|
| `s_order` | 0.500 | 1.000 | 1.000 |
| `s_cooc` | 0.000 | 0.000 | 1.000 |
| `s_defmention` | 0.000 | 0.000 | 1.000 |
| `s_sim` | 0.322 | 0.537 | 0.998 |
| `s_llm_plaus` | 0.500 | 0.500 | 1.000 |

## S2 — Directional verification

- Candidates admitted at τ = 0.275859: **1538 / 3832**
- Margin distribution: min 0.000, median 0.180, max 0.988

**High-confidence examples:**
- **phytoglycogen** → **food**, margin 0.9879 (confidence: 0.988)
- **Some shade-loving plants** → **food**, margin 0.9813 (confidence: 0.981)
- **food** → **Alarm photosynthesis**, margin 0.9771 (confidence: 0.977)

**Low-margin (ambiguous) examples — these route to S3 if CDP is enabled for this domain:**
- **Photoautotrophs** ↔ **carbon dioxide fixation**, margin 0.0001 (forward: 0.023, reverse: 0.023)
- **Plant species** ↔ **photosynthesis**, margin 0.0001 (forward: 0.051, reverse: 0.051)
- **gas diffusion** ↔ **fructose**, margin 0.0001 (forward: 0.026, reverse: 0.026)

## S4 — DAG assembly

- Nodes: **100**, edges kept: **756**
- Admitted candidate edges: **1538**
- Dropped — threshold: **2294**, cycle pruning: **0** (0.0% of admitted edges), transitive: **782**
- No cycles required pruning; thresholding and transitive reduction formed a strict partial order.

## S5 — Tree induction

- Depth range: **0–5**
- Clusters: **4**

| Cluster label | Member concepts (sample) |
|---|---|
| Cluster 3 | 0.1% to 8%, Alarm photosynthesis, biological processes, C4 photosynthesis research, carbohydrates, Chloracidobacterium |
| Cluster 4 | 3–6%, all living creatures, Artificial Photosynthesis, carbon dioxide fixation, chemical energy, CO2 fixation |
| Cluster 2 | Actual plants' photosynthetic efficiency, algae, anaerobic photosynthesis, cyanobacteria, depth, environmental factors |
| Cluster 1 | all aspects, analysis, Chemolithoautotrophy, dependency, differences, Earth |

## S6 — Path planning

- Time budget used: **300** minutes
- Path produced (6 nodes):
food, QUESTIONS, light, differences, analysis, photosynthesis
- Excluded (over budget): 94 non-critical / redundant nodes

## S7 — Sample quiz item

**Node:** photosynthesis
**Stem:** During the light-dependent reactions of photosynthesis, what is the primary role of the electron transport chain (ETC) in the thylakoid membrane?
**Correct:** To create a proton gradient across the membrane that drives ATP synthase to produce ATP.
**Prerequisite distractor:** To directly convert light energy into chemical energy stored in glucose molecules.
**Conceptual distractor:** To reduce NADP+ to NADPH by accepting electrons from water, which is the sole source of ATP production.
**Misconception tag:** `Confusing the ETC's role in chemiosmosis with direct substrate-level phosphorylation or glucose synthesis`

## Summary (paper-ready)

The automated LightGAP pipeline discovered the prerequisite dependency graph for **Photosynthesis** algorithmically. Across S0 through S6, 11 harvested corpus documents (10 Wikipedia articles, 1 course syllabi) yielded 100 canonical concepts and 3832 candidate pairs. Directional verification with the calibrated asymmetric feature classifier admitted 1538 pairs, and transitive reduction preserved 756 prerequisite links forming a Directed Acyclic Graph. Cycle pruning removed 0 contradictory edges (0.0% of admitted edges). Topological longest-path layering established an empirical depth range of 0–5 partitioned into 4 Leiden semantic clusters. The precedence-constrained path planner scheduled an optimal 6-node learning sequence fulfilling the 300-minute constraint.
