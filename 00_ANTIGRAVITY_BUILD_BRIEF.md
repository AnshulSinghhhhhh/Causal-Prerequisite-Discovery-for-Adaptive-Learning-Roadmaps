# LightGAP — Build Brief for Antigravity

This is the entry point. Read this file first, then the others in order.

**Working assumption:** this is a brand-new Antigravity workspace with no prior LightGAP code, no `.env`, no database, nothing carried over. Everything below describes what to build from an empty repository, not what to edit in an existing one. Where earlier research already produced a validated result, that result is given as a **target and a methodology to reproduce**, not as code to reuse — there is no code to reuse.

## Read order

1. **This file** — principles, non-negotiables, phases, secrets, definition of done.
2. `01_ARCHITECTURE_SPEC.md` — the eight pipeline stages, built fresh, following the methodology below.
3. `02_DATABASE_SUPABASE_SPEC.md` — full schema and migration files. Supabase MCP is already connected — use it directly.
4. `03_KAGGLE_FINETUNE_PIPELINE.md` — fine-tuning and running the 3B model on a Kaggle T4; nothing 3B-scale runs locally.
5. `04_UI_DESIGN_BRIEF.md` — the knowledge-tree + DAG renderer, built with 21st.dev, Google Stitch, and the impeccable skill.
6. `05_TESTING_RESULTS_PROTOCOL.md` — how to test, and the exact `.md` report every run must produce for the paper.
7. `templates/` — fill-in templates the two results files above point to.

## What this project is

LightGAP is a research-grade Intelligent Tutoring System headed for a conference paper. It automatically discovers prerequisite relationships between concepts and turns them into an adaptive learning roadmap, instead of a hand-authored or LLM-guessed curriculum. A separate, earlier build of this system already validated a prerequisite scorer (asymmetric-feature logistic regression, F1 0.704 / AUC 0.805 on a held-out set) and diagnosed why an LLM-invented curriculum produces generic-looking mind maps — because deciding structure (how many branches, what depth, what order) is exactly the part an LLM is worst at and a discovered dependency graph is best at. This build reproduces the validated pieces from scratch and adds the stages the paper specifies but were never actually implemented before: multi-signal candidate generation, counterfactual dependency probing, and tree induction.

## Two assets you have to supply yourself — I cannot regenerate these

- **`gold_pairs.csv`** — a hand-built, 105-row held-out evaluation set (true prerequisite pairs, deliberate reversal pairs, and unrelated pairs) covering the linear-algebra-to-ML pathway. This was hand-labeled by the paper's authors; there is no way to reconstruct it from public data. If you still have the file, add it to the new repo at `data/labeled/gold_pairs.csv` in phase P0 below, before anything downstream needs it. If you don't have it anymore, this build can still proceed through candidate generation and the LR baseline's cross-validation numbers, but nothing gets a one-shot held-out evaluation until you have a replacement set — flag that explicitly rather than substituting something else quietly.
- **The paper draft** — useful for the S0 concept-extraction seed list and terminology, and for cross-checking the final architecture against what's already written. Drop it into `docs/` if you want Antigravity to reference it directly; not required for the build to proceed.

## Governing principle — do not violate this anywhere in the build

**Language models label; algorithms structure.**

- An LLM *may*: extract candidate concept mentions from source text, write a one-sentence definition, name a cluster of concepts, write a quiz stem and distractors, classify a learner's misconception from free text.
- An LLM *may never*: decide how many branches or modules exist, decide which concept is a prerequisite of which, decide depth or ordering, decide cluster membership.

The earlier build of this project violated this in exactly one place — asking an LLM to invent a whole curriculum's structure directly — and that single violation is why its output looked generic regardless of how the UI was styled. Don't reintroduce it. If you find yourself about to prompt an LLM for "a hierarchical curriculum" or "4 to 6 modules," stop.

## Non-circular evaluation — do not violate this anywhere in the build

- Every threshold, hyperparameter, or model/prompt-selection decision is chosen via cross-validation on a freshly downloaded AL-CPL dataset only.
- `gold_pairs.csv`, once you have it in the repo, is touched by a `final_eval` exactly once per model version, and only after that model version's configuration is frozen. The database schema in `02_DATABASE_SUPABASE_SPEC.md` enforces this — a `model_selection` run against gold is a rejected INSERT, not a code-review comment.
- This exact discipline is why the earlier build's validated numbers are trustworthy: two real leakage bugs were caught there specifically because of this protocol (one in a model-capacity comparison at inconsistently-derived thresholds, one in an ensemble variant selected by comparing candidates against the held-out set directly). Keep the same discipline here from the start, enforced at the schema level this time instead of relying on remembering to follow it.

## Methodology to reproduce, as a target — not code to reuse

There's no LR model to import; train one, following this exact recipe, because it's already validated at this specification:

- **Feature construction:** for a directed candidate pair (u, v), embed both with `all-MiniLM-L6-v2`, then build `f_dir(u,v) = [h_u ‖ h_v ‖ (h_u−h_v) ‖ (h_u⊙h_v)]` — concatenation, difference, and elementwise product, in that order. The asymmetry (difference and product don't commute the same way forward vs. reversed) is what lets a plain logistic regression detect direction at all.
- **Model:** logistic regression on those features. Simpler architectures were deliberately *not* the starting hypothesis here — five more complex alternatives (a directed GNN, definition-enriched embeddings, training-data augmentation from a second dataset, a GNN+LR ensemble, and a small-LLM counterfactual probe) were tried in the earlier build and every one of them either failed to beat this baseline or made cross-domain transfer worse. You don't need to re-run all five to arrive at the same baseline — train the LR directly — but if the paper's ablation table needs those comparisons reproduced, they're listed in `01_ARCHITECTURE_SPEC.md` for reference.
- **Calibration protocol:** Youden's J statistic via cross-validation on AL-CPL, using a *graph-aware* split (folds must avoid transitive-pair leakage — if A→B and B→C are both in the training fold, don't let A→C leak into a different fold as if it were independent evidence). Freeze the resulting threshold before it ever touches `gold_pairs.csv`.
- **Target ballpark (for sanity-checking your fresh run, not a number to force):** F1 ≈ 0.70, ROC-AUC ≈ 0.80, threshold ≈ 0.43–0.44. A fresh AL-CPL pull and a fresh random seed won't reproduce these to the fourth decimal, and that's fine — what matters is landing in the same regime with the same qualitative profile (precision noticeably higher than recall, near-perfect accuracy on deliberately-reversed pairs). If your fresh run lands far outside this range, treat that as a signal to check the feature construction or the CV split before proceeding, not as a new result to report uncritically.

## Secrets — the user will provide the values directly to you; do not ask for them in chat and do not print them anywhere, including in the results `.md` files from `05_TESTING_RESULTS_PROTOCOL.md`

| Secret | Used by | Where it lives |
|---|---|---|
| `GROQ_API_KEY` | S0 concept extraction, S5 cluster naming, S7 quiz authoring | local `.env`, untracked from the very first commit — add `.env` to `.gitignore` before the first key ever gets pasted in |
| `SUPABASE_URL`, `SUPABASE_SERVICE_KEY` | Migrations, server-side app client | local `.env`; Supabase MCP already has its own auth, prefer MCP calls over psql where possible |
| `SUPABASE_ANON_KEY` | Frontend RLS-scoped client (learner-facing tables only) | frontend env |
| `HF_TOKEN` | Kaggle kernel: model download, private adapter upload | Kaggle → Add-ons → Secrets |
| Kaggle API token (`kaggle.json`) | Pushing/polling/pulling the Kaggle kernel from your machine | `~/.kaggle/kaggle.json`, never inside the kernel itself |

## Build phases and gates

Work in this order. Each gate is a measurement, not a vibe check — if it fails, stop and fix that phase before moving on.

| Phase | Work | Gate |
|---|---|---|
| P0 | Bootstrap: FastAPI + React/Vite skeleton, `.gitignore` covering `.env`, apply the full schema from `02_DATABASE_SUPABASE_SPEC.md` via Supabase MCP, download AL-CPL, add `gold_pairs.csv` to the repo if you have it. | Repo exists, migrations applied, AL-CPL present locally, and either `gold_pairs.csv` is in place or its absence is explicitly flagged. |
| P1 | Train and calibrate the S2 baseline (methodology above) on the fresh AL-CPL pull. Freeze the threshold. Insert the `model_versions` row. | Metrics land in the target ballpark above. Frozen threshold recorded in the database, not just in code. `gold_pairs.csv` has not been touched yet. |
| P2 | Build S0 (concept extraction) and S1 (candidate generation), offline. | **Candidate recall on `gold_pairs.csv` ≥ 0.95** — of the 64 true prerequisite pairs, S1 must surface at least 61 of them as candidates. Below ~0.85, stop and rework S1's signals before writing anything downstream. This is a one-time diagnostic on the candidate generator, not a `final_eval` — it doesn't touch the leakage guard. |
| P3 | Build the pipeline orchestrator (S2 wired to real candidates), the `/graph` endpoint, and run the one-shot `final_eval` against `gold_pairs.csv`. | A `final_eval` row exists, inserted exactly once, and its numbers are in the same regime as the target ballpark. |
| P4 | Build S5 (tree induction: longest-path layering + clustering) and the frontend renderer. | Two different goal concepts (see `05_TESTING_RESULTS_PROTOCOL.md`) produce visibly different tree shapes — different depth, different branch counts. |
| P5 | Kaggle: fine-tune per `03_KAGGLE_FINETUNE_PIPELINE.md`, then run the CDP probe on the low-margin edge band. | A calibration run shows what weight (if any) the fused score gives the CDP signal, frozen via AL-CPL CV, evaluated once against gold. |
| P6 | Modules 3/4 (resources, quizzes, remediation, decay) built and evaluated with the same rigor as 1/2. Results assembled for the paper. | Every phase above has its `.md` report in `docs/results/` per `05_TESTING_RESULTS_PROTOCOL.md`. |

## Definition of done

- No path through the application can serve a curriculum the pipeline didn't actually derive — no LLM-invented shortcut, no placeholder standing in for a real result.
- The threshold enforced at request time is the one your own fresh calibration run produced and froze, verifiable by reading it back from the database — never a default placeholder.
- A request for "Machine Learning" and a request for a structurally different subject produce trees with different depth and branch counts, both traceable to real DAG structure, not to two different LLM prompts.
- The 3B model has never run on the user's laptop. Its outputs exist as rows in `edge_scores`, produced by a Kaggle kernel.
- Every phase gate above has a corresponding `.md` file in `docs/results/`, written to the template in `05_TESTING_RESULTS_PROTOCOL.md`, ready to paste into the paper.
