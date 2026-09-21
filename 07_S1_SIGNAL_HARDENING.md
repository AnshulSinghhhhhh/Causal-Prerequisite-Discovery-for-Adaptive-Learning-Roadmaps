# LightGAP — S1 Signal Hardening

Read `00_ANTIGRAVITY_BUILD_BRIEF.md`, `01_ARCHITECTURE_SPEC.md`, and `06_S0_S6_HARDENING.md` first. This addendum covers three things found after re-running the Machine Learning demo on the hardened S0/S6 pipeline: a blocking check on S2's reliability, a broadened S0 source set, a fifth S1 signal, and a small S6 refinement. Same constraint as `06`: nothing here is keyed to "Machine Learning," "Data Dredging," or any other specific phrase — every fix is a source, a signal, or a weighting rule that applies identically to any goal concept.

## Do this first — it's blocking, not optional

The hardened re-run showed the cycle-pruning rate among S2-admitted edges roughly double relative to the first run (35% → 72% of admitted edges). That's consistent with S2 being much less reliable outside the four domains it was calibrated on (data mining, geometry, physics, precalculus) than inside them — which is exactly what `gold_pairs.csv` exists to test, since it was hand-built specifically to evaluate cross-domain transfer to ML-adjacent content.

**Before doing anything else in this file:** check whether a `final_eval` row exists in `evaluation_runs` for the current `model_versions` row against `gold_pairs_v1`. If one doesn't exist yet, run it now — it's a one-shot measurement per model version (the schema trigger in `02_DATABASE_SUPABASE_SPEC.md` enforces that), so it needs to happen before any further tuning changes what "the current model version" even means. Write the result to `docs/results/eval_gold_e2e_<date>.md` per `05_TESTING_RESULTS_PROTOCOL.md`.

- If the numbers land near the target ballpark (F1 ≈ 0.70, AUC ≈ 0.80), S2 itself is fine and the cycle-pruning regression is more likely a graph-density effect (a smaller, more topically concentrated node set has more opportunities for two candidate edges to disagree) — proceed with Parts A–C below.
- If the numbers show a real regression, S2 needs more or different training signal before Parts A–C will help much — flag that back rather than continuing to tune S0/S1/S6 around a scorer that's guessing.

---

## Part A — S0: broaden harvested sources beyond Wikipedia

`s_order` (candidate signal 1 of 4 in the original spec) is supposed to capture real pedagogical sequencing, but Wikipedia prose ordering is a weak proxy for it — an article can mention a caveat topic in the same breath as a core one with no ordering information at all. Real course syllabi encode actual sequencing decisions made by instructors, which is exactly the signal `s_order` was designed to use.

**Add a second harvest query type in `harvest.py`,** alongside the existing Wikipedia pull: a general web search for `"<goal concept> syllabus OR course outline OR curriculum"`, tagging the results `source_type = 'syllabus'` in `corpus_documents`. This is a query pattern, not a hardcoded list of sites — it generalizes to any goal concept the same way the Wikipedia harvest already does.

**Weight `s_order` by source type when combining across documents:** a syllabus-sourced ordering observation should count for more than a Wikipedia-prose one, since it's evidence of real sequencing rather than incidental proximity. A simple fixed multiplier (e.g. 3x for `syllabus`, 1x for `wikipedia`) in the weighted average is enough — this is a structural rule about source reliability, not something tuned per domain.

---

## Part B — S1: a fifth signal, `s_llm_plaus`

The four existing signals are all forms of association (order, co-occurrence, definitional mention, similarity) — none of them can distinguish "A is necessary to understand B" from "A is commonly mentioned near B as a caveat, a historical aside, or an unrelated sense of the same word." That gap is what let a real-but-irrelevant concept score high enough to reach position 2 in a study path.

**Add one batched Groq call per domain** (chunked the same way `verify_concepts.py` already chunks the A5 pass), asking, for each candidate pair `(A, B)`: *does knowing A plausibly help you understand B — as a real prerequisite, not just something commonly discussed nearby? Answer per item, with a 0–1 plausibility score.* Fixed rubric, no domain-specific content in the prompt.

**This is a fusion input, not a decision.** Combine `s_llm_plaus` into `s_corr` alongside the other four, weight calibrated on AL-CPL cross-validation like every other weight in this pipeline, frozen the same way. It never directly admits or rejects an edge on its own — same relationship to the pipeline that the CDP signal already has, just roughly three orders of magnitude cheaper, since it's a single chat completion instead of a Kaggle GPU job.

**This should also shrink the CDP queue.** A pair `s_llm_plaus` scores as clearly implausible is likely to end up with a low fused `s_corr` and either fail admission outright or land somewhere S3 doesn't need to spend a GPU pass resolving — worth checking after the fact whether the low-margin queue that reaches S3 gets smaller once this signal is in place, since that's real compute saved, not just quality gained.

**Database:** add the raw per-pair signal to `candidate_edges` for the same auditability reason the other three signals are stored individually rather than only as the combined `s_corr`:

```sql
-- migration 0010_llm_plausibility_signal.sql
alter table candidate_edges add column if not exists s_llm_plaus real;
```

---

## Part C — S6: in-cluster vs. cross-cluster unlock weighting

A small refinement to the value function from `06_S0_S6_HARDENING.md` Part B1. Right now `value[n]` counts every descendant in the ancestor-of-goal subgraph equally. Nodes that connect broadly across many different S5 clusters — caveats, historical asides, cross-cutting topics — tend to rack up descendant count without being genuinely foundational to any one of them, while a node that's deeply load-bearing *within* its own cluster is exactly what you want ranked higher.

```python
value[n] = len(descendants(n, dag) & A & same_cluster(n)) \
         + CROSS_CLUSTER_WEIGHT * len(descendants(n, dag) & A - same_cluster(n))
```

Uses `cluster_id`, already computed by S5 — no new data needed.

**Be honest about `CROSS_CLUSTER_WEIGHT`: it doesn't have a calibration story the way `tau_edge` or the S1 signal weights do.** There's no ground-truth pedagogical ordering dataset to fit it against automatically. Pick a reasonable structural default (0.3 is a defensible starting point — cross-cluster connections still count, just at a discount) and treat it as a placeholder to revisit once either a real calibration signal exists for path quality or enough manual sanity reads (per `06`'s B5) accumulate to justify adjusting it. Don't present it in the paper as calibrated the same way the other frozen constants are — it isn't, and saying so is more useful than letting it blend in with the numbers that are.

---

## Re-validation

Re-run both demo walkthroughs again after Parts A–C. Specifically check, and record in the updated `result_<goal_slug>.md` files:

- The cycle-pruned / admitted percentage from `02_DATABASE_SUPABASE_SPEC.md`'s edge accounting — it should move back toward (or below) the 35% seen in the first hardening pass, not stay near 72%.
- Whether concepts that scored low on `s_llm_plaus` correspond to the kind of entries that shouldn't have ranked highly before (check this by inspecting the signal directly, not by eyeballing the final path — the path alone doesn't show you why something was included or excluded).
- The same plain human read from `06`'s B5: does the path look like a coherent sequence, not just a technically valid one.
