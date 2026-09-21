# LightGAP — S0 / S6 Hardening

Read `00_ANTIGRAVITY_BUILD_BRIEF.md` and `01_ARCHITECTURE_SPEC.md` first. This addendum replaces the S0 (concept extraction) and S6 (path planning) implementation details in that spec with a hardened version, written after the first real end-to-end run (goal concept "Machine Learning") surfaced concrete problems.

**Design constraint that shapes every fix below: nothing here is keyed to "Machine Learning," "binary search," or any other specific example.** Every fix is a structural, algorithmic, or POS/section-level rule that applies identically to any goal concept. If a fix can only be justified by "this specific phrase looked wrong," it doesn't belong in this file — the point of this pass is that the *next* domain's specific noise, whatever it turns out to be, gets caught by the same general rules without anyone writing a new special case for it.

## What the first run actually showed

Three independent signals in the "Machine Learning" demo run, not one:

- Concept names included extraction fragments — determiner-prefixed phrases ("a binary search," "a computational process"), a conjunction fragment ("and classification rule"), and what looks like a citation/reference-list artifact ("Stony Brook Collected Algorithms").
- 35% of the edges that passed the directional-verification threshold (137 of 393) were later removed as cycle participants — high enough to suggest near-duplicate nodes ("a neural network" / "Neural networks") were getting inconsistent edges to shared neighbors, not genuine directional disagreement.
- Three of four sampled clusters shared an identical LLM-assigned label ("Machine Learning"), suggesting either redundant communities or a naming call with no visibility into what its siblings were already named.

All three point at S0's canonicalization pass under-filtering and under-merging. S6's path-planning objective is a separate, real problem, but it's optimizing over whatever node set S0 hands it — fix S0 first, or a better S6 objective is just ranking noise more confidently.

---

## Part A — S0: concept extraction and canonicalization

### A1. Strip non-body sections before extraction (`harvest.py`)

Before returning a harvested document's text, drop everything from the first occurrence of a "References," "See also," "External links," "Notes," "Further reading," or "Bibliography" heading onward (Wikipedia's API returns section structure — use it, don't regex the rendered text). This alone likely accounts for citation-style noise like the Stony Brook example — it's a structural fix, not a content filter, so it applies to every domain's source documents equally.

### A2. Well-formedness filter at extraction (`extract.py`)

After spaCy chunking and the LLM extraction pass, reject any candidate span whose syntactic root (via spaCy's dependency parse) is not a `NOUN` or `PROPN`, and reject any span whose first token is a coordinating conjunction, determiner alone, or preposition. Concretely:

```python
def is_wellformed(span: spacy.tokens.Span) -> bool:
    if span.root.pos_ not in ("NOUN", "PROPN"):
        return False
    first = span[0]
    if first.pos_ in ("CCONJ", "SCONJ", "ADP") or first.lemma_.lower() in ("a", "an", "the"):
        return False
    return True
```

This is a POS-structural rule, not a keyword list — it will catch "and classification rule" and "a binary search" the same way it would catch the equivalent noise in a linear algebra or reinforcement learning corpus, without anyone needing to have seen those specific phrases first.

### A3. Normalize before embedding, not after (`canonicalize.py`)

Build a `normalize()` function applied before the dedup embedding step — lowercase, strip a leading determiner (`a`/`an`/`the`), lemmatize the head noun (singular/plural collapse via spaCy), strip possessives, collapse whitespace. Embed the *normalized* string for the dedup comparison in A4, not the raw extracted surface form. Keep the raw form too — store it in `concept_aliases`, and select a display-quality canonical name in A4 rather than defaulting to whichever variant was extracted first.

This directly targets the "a neural network" / "Neural networks" split — after normalization both become `neural network`, which should merge on exact-string match alone, before cosine similarity is even needed.

### A4. Union-find merge, not pairwise (`canonicalize.py`)

Two-tier merge, applied as a transitive closure rather than independent pairwise decisions:

1. **Exact match on the normalized form** → auto-merge, no threshold involved.
2. **Cosine similarity above a calibrated threshold** on the MiniLM embedding of the normalized form → merge candidates.

Use union-find (disjoint-set) across both tiers so that if A merges with B and B merges with C, all three land in one cluster even if A and C weren't compared directly or didn't individually clear the threshold — independent pairwise merging is exactly what lets a chain like this fall apart into two nodes instead of one.

**Calibrate the cosine threshold, don't pick a round number.** Same discipline as `tau_edge`: take a small sample of near-duplicate and genuinely-distinct pairs from AL-CPL's own concept set (it already has enough concepts to hand-check a sample against), pick the threshold that best separates them, freeze it, record it the same way `tau_edge` is recorded. This is the one place in this file that needs a number chosen by evidence rather than by structure — don't skip the calibration step to save time.

**Canonical name selection within a merged cluster:** prefer, in order, (1) a form that exactly matches a Wikipedia article or redirect title, (2) the most frequently occurring raw surface form across source documents, (3) the shortest well-formed form. Store the rest of the cluster's raw forms in `concept_aliases`.

### A5. A batched second-opinion pass (new: `verify_concepts.py`)

After A1–A4, run **one batched Groq call per domain** (chunk into batches of 30–50 if needed) over the deduped candidate list: *"Which of these are genuine named concepts or topics, as opposed to fragments, citations, or overly generic phrases? Answer per item."* This is a labelling task over a list the pipeline already produced — it doesn't decide structure, it flags well-formedness, which is squarely inside what an LLM is allowed to do per the governing principle in `00_ANTIGRAVITY_BUILD_BRIEF.md`. It's also the cheapest way to catch whatever A1–A4's structural rules miss, since a rule-based filter alone can't perfectly distinguish "Empirical risk minimization" (fine) from something that's syntactically well-formed but still not a real concept for this domain.

Only concepts that pass A1–A5 get persisted to `concepts`.

### A6. Keep the audit trail (database addition, see below)

Every one of A1–A5 rejects or merges something. Log it, using the same pattern `dag_edges.dropped_reason` already established — a rejected or merged candidate is data, not noise to discard silently. This also gives you a genuinely useful methods-section table for free: "N raw extractions → filtered by stage → M final concepts," the same kind of ablation reporting this project already does for edges.

---

## Part B — S6: a real pedagogical objective

### B1. The fix: weight nodes by how much they unlock, not just include them

The original spec said "maximize coverage of unmastered ancestors of the goal node" without saying every ancestor is worth the same — so a knapsack maximizing raw count under a time budget will happily spend budget on cheap, tangential nodes. That's an underspecification in the original architecture spec, not just an implementation gap, and it's what actually explains the "binary search" / "Hebbian theory" style path.

The fix stays inside standard operations-research terms — no hand-picked bonus/penalty terms, no new constants beyond the cost estimate in B2:

```python
A = ancestors(goal, dag) | {goal}          # unchanged from the original spec

# value: how many other required concepts does learning this one help unlock
value = {n: len(descendants(n, dag) & A) for n in A}

cost = {n: concepts[n].est_minutes for n in A}    # see B2

mastered = {n for n in A if mastery.get(n, 0) >= MASTERY_THRESHOLD}
decide = A - mastered

# maximize sum(value[n] * x[n] for n in decide)
# subject to:
#   sum(cost[n] * x[n] for n in decide) <= budget
#   x[p] >= x[n]  for every n in decide and every direct prerequisite p of n that's also in decide
#   x[n] in {0, 1}
```

`value[n]` is a plain descendant count within the ancestor-of-goal subgraph — nodes near the foundational end of the DAG naturally have more things depending on them and score higher automatically; leaf trivia that don't unlock anything else score at or near zero and get starved out under a tight budget without needing a separate "noise penalty" term. This is graph structure only, computed fresh for every domain — nothing here is specific to any one subject.

Once the ILP returns the selected set, order it for the frontend with a topological sort over the selected subgraph, breaking ties by ascending `depth` (more foundational first) then descending `value`. The knapsack tells you *which* nodes; this step is what turns that set into the sequential path the UI actually shows.

**If descendant count proves too coarse once you see it on real output** (e.g., ties are common, or it under-weights nodes whose value is mostly indirect), the natural escalation is personalized PageRank over the reversed ancestor subgraph instead of a flat count — same inputs, no new hand-tuned weights, just a different centrality measure. Don't reach for it preemptively; try the plain count first and only escalate if B5's re-validation shows it's insufficient.

### B2. Per-node cost estimate

The knapsack needs a cost per node and nothing in the current spec supplies one. Use a placeholder until S7's resource curation is wired in, then let real data replace it:

- Default: a fixed `est_minutes` per concept (see schema addition below), assigned at creation time.
- Once S7 attaches real resources to a node, update `est_minutes` from the resource's actual duration/estimated reading time, so the budget constraint gets more accurate over time without any change to B1's formulation.

### B3. Database addition

```sql
-- migration 0009_hardening.sql

alter table concepts add column if not exists est_minutes int not null default 15;

create table rejected_concepts (
  id           uuid primary key default gen_random_uuid(),
  domain_id    uuid not null references domains(id) on delete cascade,
  raw_phrase   text not null,
  stage        text not null,   -- 'section_strip' | 'pos_filter' | 'llm_verify' | 'merged_duplicate'
  reason       text,
  merged_into  uuid references concepts(id),   -- set only when stage = 'merged_duplicate'
  created_at   timestamptz not null default now()
);
```

Apply through the same Supabase MCP flow as `0001`–`0008` in `02_DATABASE_SUPABASE_SPEC.md`.

### B4. What NOT to do — guardrails against re-specializing this later

- No blocklist of rejected phrases, ever. A1–A5 must be the only mechanism; if a new domain produces a new kind of noise, extend the *rule* (a new section heading to strip, a POS pattern to reject), not a list of banned strings.
- No per-domain stopword list. If a filter needs to know it's looking at "Machine Learning" specifically to work, it's the wrong filter.
- No manual reweighting of individual node types ("always deprioritize algorithm nodes," etc.) in B1. `value[n]` must stay a pure function of graph structure — the moment it starts branching on what kind of concept `n` is, it stops generalizing and starts requiring maintenance per domain, which is the exact failure mode this whole redesign exists to get away from.

### B5. Re-validation

Once A1–A5 and B1–B2 are implemented, re-run both existing demo walkthroughs — "Machine Learning" and the structurally different second domain from `05_TESTING_RESULTS_PROTOCOL.md` — and update their `result_<goal_slug>.md` files. Specifically check, and say so explicitly in the updated files: the raw-extraction-to-final-concept-count ratio (from A6's audit trail), the cycle-pruning rate as a fraction of admitted edges (expect it to drop well below the 35% seen in the first run), and whether the new S6 path reads as a coherent sequence on a plain human read rather than a technically-valid-but-wandering one. There's no ground-truth pedagogical ordering to score against automatically — this last check is a manual sanity read, and it's fine for it to stay that way rather than inventing a synthetic metric for it.
