# LightGAP demo walkthrough — {{GOAL_CONCEPT}}

- **Date:** {{YYYY-MM-DD}}
- **Git SHA:** {{git_sha}}
- **Domain slug:** {{goal_slug}}
- **Model versions used:** S2 = `{{lr_model_version_name}}`, S5 clustering = `{{clustering_method}}`, S3 CDP (if used) = `{{cdp_model_version_name or "not run for this domain"}}`
- **Graph snapshot ID:** {{graph_snapshots.id}}

## S0 — Concept extraction

- Concepts harvested: **{{n_concepts}}**
- Source documents: **{{n_documents}}** ({{source_type_breakdown, e.g. "6 Wikipedia articles, 2 syllabi"}})

| Concept | Definition (truncated) | Source |
|---|---|---|
| {{concept_1}} | {{definition_1}} | {{source_1}} |
| {{concept_2}} | {{definition_2}} | {{source_2}} |
| {{concept_3}} | {{definition_3}} | {{source_3}} |

## S1 — Candidate generation

- Candidate pairs generated: **{{n_candidates}}**

| Signal | Min | Median | Max |
|---|---|---|---|
| `s_order` | {{}} | {{}} | {{}} |
| `s_cooc` | {{}} | {{}} | {{}} |
| `s_defmention` | {{}} | {{}} | {{}} |
| `s_sim` | {{}} | {{}} | {{}} |

## S2 — Directional verification

- Candidates admitted at τ = {{tau_edge}}: **{{n_admitted}} / {{n_candidates}}**
- Margin distribution: min {{}}, median {{}}, max {{}}

**High-confidence examples:**
- {{concept_A}} → {{concept_B}}, margin {{}}

**Low-margin (ambiguous) examples — these route to S3 if CDP is enabled for this domain:**
- {{concept_C}} ↔ {{concept_D}}, margin {{}}

## S4 — DAG assembly

- Nodes: {{n_nodes}}, edges kept: {{n_edges_kept}}
- Dropped — threshold: {{n_dropped_threshold}}, cycle pruning: {{n_dropped_cycle}}, transitive: {{n_dropped_transitive}}
- {{If a cycle was pruned: describe which edge lost and why, same format as the project's existing collateral-damage audit.}}

## S5 — Tree induction

- Depth range: {{min_depth}}–{{max_depth}}
- Clusters: {{n_clusters}}

| Cluster label | Member concepts (sample) |
|---|---|
| {{cluster_1_label}} | {{concepts}} |
| {{cluster_2_label}} | {{concepts}} |

## S6 — Path planning

- Time budget used: {{minutes}} minutes
- Path produced ({{n_nodes_in_path}} nodes): {{ordered list}}
- Excluded (over budget): {{list, if any}}

## S7 — Sample quiz item

**Node:** {{concept}}
**Stem:** {{quiz_stem}}
**Correct:** {{correct_option}}
**Prerequisite distractor:** {{prereq_distractor}}
**Conceptual distractor:** {{conceptual_distractor}}
**Misconception tag:** {{tag}}

## Rendered UI

![{{goal_concept}} roadmap]({{screenshot_path}})

## Summary (paper-ready)

{{One paragraph, plain English, suitable for the evaluation section — what this walkthrough demonstrates about the pipeline for this specific goal concept.}}
