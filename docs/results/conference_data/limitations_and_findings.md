# LightGAP — Architectural Limitations & Empirical Findings: The Broad-Domain Failure Mode

This document formalizes the primary empirical limitation and architectural discovery identified during the development and evaluation of LightGAP: **the sensitivity of multi-stage prerequisite discovery pipelines to corpus breadth and sequential candidate filtering**. 

Rather than treating these phenomena as engineering anomalies to be concealed, we report them as concrete methodological findings that illuminate the trade-offs of learning-path induction from unstructured corpora.

---

## 1. Executive Finding: The Cascading Starvation Thesis

> **Core Finding**: *"Naive candidate filtering, applied in sequence, starves output on broad domains even when each individual filter is independently correct."*

In automated curriculum discovery, individual filtering stages are typically designed and evaluated in isolation:
1. Lexical and morphological filters in extraction (S0).
2. Semantic similarity clustering for synonym canonicalization (S0).
3. Embedding cosine proximity and LLM plausibility pruning in candidate generation (S1).
4. Directional classifier thresholding $\tau_{\text{edge}}$ (S2).
5. Strongly Connected Component (SCC) cycle pruning (S4).
6. Transitive reduction (S4).
7. Precedence-constrained knapsack path planning (S6).

While each filter exhibits defensible local precision when evaluated independently, their feedforward sequential composition creates a severe **cascading attenuation effect**. On diffuse, multi-subdomain topics (e.g., open-domain *"Machine Learning"*), this sequential bottleneck starves the downstream graph of legitimate prerequisite edges. Conversely, attempting to widen individual filter thresholds to preserve recall introduces non-conceptual noise that propagates uncontrollably into candidate scoring.

---

## 2. Corpus Breadth as the Binding Constraint: Machine Learning vs. Data Mining

The central experimental contrast in LightGAP demonstrates that **corpus breadth—not classifier signal quality—is the primary binding constraint on graph coherence**.

### 2.1 Comparative Analysis

| Dimension | Domain A: Open-Domain "Machine Learning" | Domain B: Grounded "Data Mining" |
|---|---|---|
| **Corpus Nature** | Broad web scrape + academic overviews + promotional syllabus brochures | Curated Wikipedia articles + structured course syllabus |
| **Harvested Concepts (S0)** | 95 canonical concepts | 100 canonical concepts |
| **Initial Candidate Pairs (S1)** | 2,193 candidate pairs | 3,421 candidate pairs |
| **Admitted Candidates (S2)** | 1,390 edges ($\tau = 0.275859$) | 2,436 edges ($\tau = 0.275859$) |
| **Cycle Pruning Drops (S4)** | **137 edges severed** (early run: ~35% of admitted; swings up to 72% on unhardened subsets) | **0 edges severed** (**0.0%** cycle drops; strict partial order formed directly) |
| **Kept DAG Edges (S4)** | 363 edges (after 1,027 transitive drops) | 232 edges (after 2,204 transitive drops) |
| **Path Planning (S6)** | Included peripheral and syllabus artifacts (`commerce`, `problems`, `difficulty`, `Introduction`) | Strictly conceptual progression (`statistics`, `science`, `information`, `knowledge`, `Ideation`...) |

### 2.2 Why Signal Quality Was Not the Culprit
The asymmetric feature classifier ($f_{\text{dir}} = [h_u \parallel h_v \parallel (h_u - h_v) \parallel (h_u \odot h_v)]$) was held completely constant across both domains, operating with the identical frozen calibration threshold $\tau_{\text{edge}} = 0.275859$. 

The stark divergence in structural quality was entirely driven by corpus concentration:
- In *Data Mining* (one of AL-CPL's native benchmark domains), concepts occupy an established, hierarchical taxonomy (e.g., *data preprocessing* $\to$ *frequent pattern mining* $\to$ *association rules* $\to$ *FP-growth*). Edges naturally flow in one direction, yielding **zero cycles** requiring SCC destruction.
- In *Machine Learning*, documents harvested from general web queries encompass distinct sub-disciplines (reinforcement learning, deep vision, statistical learning, introductory brochures) written at conflicting levels of granularity. Because textbook introductions discuss *supervised learning* and *neural networks* reciprocally, the classifier detected valid local directional evidence in both orientations depending on document context, inducing massive cyclic cliques that degraded downstream topological layering.

---

## 3. Survivor Artifacts: Structural Leakage from Real-World Curricula

Unstructured pedagogical corpora (specifically university syllabi and Wikipedia reference tables) consistently injected non-conceptual text spans that evaded conventional noun-phrase extraction:

### 3.1 Numeric Fragments
In the scientific domain (*Photosynthesis*), empirical rate measurements and chemical percentages embedded in article summaries survived S0 extraction:
- Extracted spans: `"0.1% to 8%"`, `"3–6%"`, `"50%"`.
- **Causal Failure**: Standard spaCy/Stanza noun-chunk extractors recognize alphanumeric compounds and percentage modifiers as noun phrases or adjectival heads. Without an explicit numeric-fragment suppression filter, these measurements entered the candidate pipeline as purported prerequisite concepts, forming nonsensical candidate edges such as `"0.1% to 8%" \to "photosynthesis"`.

### 3.2 Syllabus & Meta-Token Header Noise
In course syllabi (*Machine Learning* and *Data Mining*), organizational markers and pedagogical headings survived into the canonical concept set:
- Extracted spans: `"QUESTIONS"`, `"Introduction"`, `"problems"`, `"Course Overview"`, `"Techno India University B.Sc"`.
- **Causal Failure**: Syllabus documents structure content using numbered headings and administrative sections. When parsed as flat text, these headers are lexically prominent, high-frequency noun chunks. In S6 path planning, the ILP optimizer frequently prioritized tokens like `"Introduction"` or `"problems"` because their topological in-degree was artificially inflated by syllabus cross-referencing.

---

## 4. The Near-Duplicate Merge Threshold: The Precision/Recall Dilemma

In Step S0 (canonicalization), candidate noun phrases are embedded using `all-MiniLM-L6-v2` and clustered using agglomerative clustering at cosine similarity threshold $\tau_{\text{similarity}}$:

$$\text{sim}(u, v) = \frac{h_u \cdot h_v}{\|h_u\|_2 \|h_v\|_2} \ge \tau_{\text{similarity}}$$

Our calibration audits (`calibration_similarity_threshold.json`) revealed an intractable Pareto frontier between synonym collapse and semantic conflation:

1. **At Conservative Thresholds ($\tau_{\text{similarity}} \ge 0.85$):**
   - *Precision*: High semantic purity. Distinct concepts remain separated.
   - *Failure Mode (Graph Inflation)*: Lexical near-synonyms survive as distinct entities. In *Data Mining*, both `"CRISP-DM methodology"` and `"CRISP‑DM"`, as well as `"association rule learning"` and `"association rule mining"`, persisted as independent nodes. This artificially squared the downstream candidate evaluation space ($O(N^2)$) and created near-parallel duplicate paths in S5 tree induction.

2. **At Aggressive Thresholds ($\tau_{\text{similarity}} \le 0.78$):**
   - *Recall*: Effectively collapses grammatical and stylistic variants.
   - *Failure Mode (Semantic Conflation)*: As demonstrated in our Counterfactual Dependency Probing (CDP) audit, distinct concepts with high semantic proximity—such as `"Deep learning"` and `"Deep neural networks"`—are erroneously collapsed into a single node. This prematurely eliminates genuine hierarchical dependencies where one concept is an architectural prerequisite for the other.

---

## 5. Cycle-Pruning Rate Swings as a Measure of Topological Coherence

Tarjan's Strongly Connected Components (SCC) algorithm resolves cyclical dependencies by finding the cycle $\mathcal{C}$ and deleting the edge with minimum confidence $\min_{(u,v) \in \mathcal{C}} s(u, v)$. 

Tracking the **cycle pruning drop rate** ($\rho_{\text{cycle}} = |E_{\text{pruned}}| / |E_{\text{admitted}}|$) provides a rigorous, unsupervised indicator of corpus health:
- $\rho_{\text{cycle}} \approx 0.0\%$: Indicates high corpus coherence with clear directional hierarchy (observed in *Data Mining*).
- $\rho_{\text{cycle}} > 20.0\%$: Indicates diffuse, conflicting multi-source corpora with semantic overlap (observed in early *Machine Learning* runs at 35%).
- When $\rho_{\text{cycle}}$ spikes, the greedy edge-removal strategy risks breaking valid transitive chains, creating disconnected graph components and shallow learning paths.

---

## 6. Recommendations for Future Research

LightGAP's findings indicate that closing the gap between algorithmic prerequisite discovery and human-grade curricula requires rethinking feedforward filtering:

1. **Curricular Grounding over Open Harvesting**: Prerequisite pipelines should ground concept extraction within authoritative curricular taxonomies (e.g., ACM/IEEE curricula or official course prerequisites) rather than open web crawling.
2. **Joint Iterative Inference**: Rather than cascading S0 $\to$ S1 $\to$ S2 $\to$ S4 sequentially, candidate admission and cycle elimination should be solved as a joint constrained optimization problem, preventing the compound attenuation of true dependencies.
3. **Multi-Scale Concept Resolution**: Incorporating explicit parent-child ontology levels (e.g., distinguishing between a 40-hour course topic vs. a 15-minute formula derivation) would prevent granular formula variables from competing with broad curricular subjects.
