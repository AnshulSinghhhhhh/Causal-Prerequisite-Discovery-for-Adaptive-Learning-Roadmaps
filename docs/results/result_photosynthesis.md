# LightGAP demo walkthrough — Photosynthesis

- **Date:** 2026-09-21
- **Git SHA:** `cv-youden-recalibrated-20260921`
- **Domain slug:** `photosynthesis`
- **Model versions used:** S2 = `lr_asym_minilm`, S5 clustering = `Leiden community detection`, S3 CDP = `Qwen/Qwen2.5-3B-Instruct (Teacher-forced NLL probe)`
- **Graph snapshot ID:** `71de82ad-3db6-45b8-88c9-0953ad5296ff`

## S0 — Concept extraction

- Concepts harvested: **78**
- Source documents: **11** (10 Wikipedia articles, 1 course syllabi)

| Concept | Definition (truncated) | Source |
|---|---|---|
| Actual plants' photosynthetic efficiency | Technical concept definition... | Photosynthesis |
| Alarm photosynthesis | Technical concept definition... | Photosynthesis |
| algae | A large and diverse group of photosynthetic organisms.... | Photosynthesis |
| all photosynthetic protists | Technical concept definition... | Algae |

## S1 — Candidate generation

- Candidate pairs generated: **1818**

| Signal | Min | Median | Max |
|---|---|---|---|
| `s_order` | 0.500 | 1.000 | 1.000 |
| `s_cooc` | 0.000 | 0.000 | 1.000 |
| `s_defmention` | 0.000 | 0.000 | 1.000 |
| `s_sim` | 0.151 | 0.437 | 0.916 |
| `s_llm_plaus` | 0.500 | 0.500 | 1.000 |

## S2 — Directional verification

- Candidates admitted at τ = 0.275859: **137 / 1818**
- Margin distribution: min 0.000, median 0.049, max 0.882

**High-confidence examples:**
- **other Calvin cycle enzymes** → **Carbon dioxide**, margin 0.8823 (confidence: 0.883)
- **non-carbon-fixing anoxygenic photosynthesis** → **Carbon dioxide**, margin 0.8539 (confidence: 0.854)
- **anoxygenic phototrophs** → **Carbon dioxide**, margin 0.8237 (confidence: 0.824)

**Low-margin (ambiguous) examples — these route to S3 if CDP is enabled for this domain:**
- **Photoautotroph** ↔ **Total photosynthesis**, margin 0.0000 (forward: 0.022, reverse: 0.022)
- **eight photosynthetic lineages** ↔ **photosynthetic cells**, margin 0.0000 (forward: 0.012, reverse: 0.012)
- **C2 photosynthesis** ↔ **algae**, margin 0.0001 (forward: 0.020, reverse: 0.020)

## S4 — DAG assembly

- Nodes: **72**, edges kept: **119**
- Admitted candidate edges: **137**
- Dropped — threshold: **1681**, cycle pruning: **0** (0.0% of admitted edges), transitive: **18**
- No cycles required pruning; thresholding and transitive reduction formed a strict partial order.

## S5 — Tree induction

- Depth range: **0–2**
- Clusters: **4**

| Cluster label | Member concepts (sample) |
|---|---|
| Photosynthesis | Actual plants' photosynthetic efficiency, algae, all plants, anaerobic photosynthesis, anaerobic photosynthetic electron transport chains, archaea use sunlight |
| Photosynthesis | Alarm photosynthesis, all photosynthetic protists, Anoxygenic photosynthesis, anoxygenic photosynthetic bacteria, C4 photosynthesis research, C4 plants |
| Photosynthesis | anoxygenic phototrophs, bacteriochlorophyll, biosynthesis, Calvin cycle, Chloroplast, eight photosynthetic lineages |
| Foundational & Independent Concepts | bacteriochlorophyll conduct photosynthesis, Mixotroph, photosynthetic capacities, photosynthetic cells, photosynthetic eukaryotes, Plant respiration |

## S6 — Path planning

- Time budget used: **300** minutes
- Path produced (13 nodes):
Photosynthesis, chlorophyllide, anaerobic photosynthesis, Anoxygenic photosynthesis, Alarm photosynthesis, hydrogen-based lithoautotrophy, Chlorella, non-carbon-fixing anoxygenic photosynthesis, other Calvin cycle enzymes, anoxygenic phototrophs, all photosynthetic protists, anoxygenic photosynthetic bacteria ...
- Excluded (over budget): 65 non-critical / redundant nodes

## S7 — Sample quiz item

**Node:** Photosynthesis
**Stem:** A student claims that the primary role of the light-dependent reactions in photosynthesis is to directly fix carbon dioxide into glucose. Which of the following statements best identifies the error in this claim?
**Correct:** The claim is incorrect because light-dependent reactions produce ATP and NADPH, which are then used in the Calvin cycle to fix carbon dioxide; they do not directly fix carbon.
**Prerequisite distractor:** The claim is incorrect because light-dependent reactions occur in the mitochondria, not the chloroplasts, so they cannot interact with carbon fixation pathways.
**Conceptual distractor:** The claim is incorrect because light-dependent reactions consume carbon dioxide to generate oxygen, leaving no carbon available for glucose synthesis.
**Misconception tag:** `Confusion between light-dependent and light-independent reaction functions`

## Summary (paper-ready)

The automated LightGAP pipeline discovered the prerequisite dependency graph for **Photosynthesis** algorithmically. Across S0 through S6, 11 harvested corpus documents (10 Wikipedia articles, 1 course syllabi) yielded 78 canonical concepts and 1818 candidate pairs. Directional verification with the calibrated asymmetric feature classifier admitted 137 pairs, and transitive reduction preserved 119 prerequisite links forming a Directed Acyclic Graph. Cycle pruning removed 0 contradictory edges (0.0% of admitted edges). Topological longest-path layering established an empirical depth range of 0–2 partitioned into 4 Leiden semantic clusters. The precedence-constrained path planner scheduled an optimal 13-node learning sequence fulfilling the 300-minute constraint.
