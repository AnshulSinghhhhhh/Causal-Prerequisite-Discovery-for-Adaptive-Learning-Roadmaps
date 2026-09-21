# LightGAP — Testing & Results Protocol

Read the other files first. This one governs how every test is run and — the part that matters for the paper — how its output gets written down. **Every run listed below produces its own `.md` file in `docs/results/`. Never fold two runs' numbers into one file, and never skip writing the file because a run "just confirms what we expected."**

## Run taxonomy and naming

| Run kind | When | Filename | Template |
|---|---|---|---|
| Candidate recall diagnostic | End of P1 | `docs/results/eval_candidate_recall_<YYYYMMDD>.md` | `templates/eval_report_template.md` |
| End-to-end gold evaluation | End of P3, and again any time `model_versions` changes for S2's fusion | `docs/results/eval_gold_e2e_<model_version_name>_<YYYYMMDD>.md` | `templates/eval_report_template.md` |
| CDP fine-tune calibration | End of P5 | `docs/results/eval_cdp_finetune_<YYYYMMDD>.md` | `templates/eval_report_template.md` |
| Demo walkthrough (a specific goal concept, full pipeline) | End of P4, and whenever a demo is needed for the paper | `docs/results/result_<goal_slug>.md` | `templates/result_demo_template.md` |
| Any other model-version comparison or ablation you run for the paper | Whenever | `docs/results/eval_<short_name>_<YYYYMMDD>.md` | `templates/eval_report_template.md` |

`goal_slug` is the same `_clean_id`-style snake_case slug already used elsewhere in the codebase, e.g. `machine_learning`, `linear_algebra`.

## The two required demo walkthroughs

Run the full S0→S6 pipeline (not the old template path) for at least these two goal concepts, and write each as its own `result_<goal_slug>.md`:

1. **`Machine Learning`** → `docs/results/result_machine_learning.md`
2. **A structurally different subject** — pick one with a genuinely different dependency shape, e.g. `Linear Algebra` (much shallower, more parallel) or `Reinforcement Learning` (deeper chains through MDPs → value functions → policy gradients) → `docs/results/result_<that_slug>.md`

The point of the second one isn't coverage, it's the P4 gate from `00_ANTIGRAVITY_BUILD_BRIEF.md`: these two files, read side by side, should show visibly different depth and branch counts. If they come out looking like the same shape with different labels, the redesign hasn't actually fixed F6 — go back to S5, not to the UI.

## What "run the pipeline" captures for a demo walkthrough

For each goal concept, capture the actual intermediate output at every stage — not just the final tree. This is what makes the `.md` usable as a paper figure/table source rather than a screenshot description:

- **S0** — how many concepts were harvested, from how many source documents, 3-5 example `(concept, definition, source)` rows.
- **S1** — candidate pair count, and the per-signal score distribution (min/median/max for each of `s_order`, `s_cooc`, `s_defmention`, `s_sim`).
- **S2** — how many candidates were admitted at the frozen `tau_edge`, the margin distribution, 2-3 example pairs where the model was confident and 2-3 where it was ambiguous (low margin — these are the ones that would route to S3).
- **S4** — DAG stats: nodes, edges kept, edges dropped and by which reason (`threshold` / `cycle_prune` / `transitive`), and if any cycle was actually pruned, show which edge lost and why (the same collateral-damage audit the project already has, now against a live snapshot).
- **S5** — the induced tree: depth range, number of clusters, cluster labels, 1-2 example clusters with their member concepts.
- **S6** — the planner's output path for a stated time budget (pick a round number, e.g. 300 minutes), showing which nodes were included/excluded and why.
- **S7** — one example quiz item for one node, with its misconception tag.
- A screenshot of the rendered UI for this goal concept (reference the image file; don't try to describe pixels in prose).
- A short narrative paragraph in plain English suitable for dropping into the paper's evaluation section, summarizing what this walkthrough demonstrates.

## What every evaluation report captures

Mirror the `evaluation_runs` table row this run corresponds to, plus enough narrative that it reads as a paper subsection on its own:

- Run name, `run_type`, timestamp, git SHA.
- Model version: name, kind, frozen hyperparameters, frozen threshold if applicable.
- Eval set: name, `purpose` (calibration vs heldout), pair count, sha256.
- Full metrics block: accuracy, precision, recall, F1, ROC-AUC where applicable, confusion matrix, subclass breakdown (prerequisite / reversal / unrelated, matching the existing gold-set convention).
- One paragraph of interpretation — what this number means, and if it's worse or better than a prior run, why (a causal explanation, not just "improved/regressed" — matching the standard already set by the five rejected architectures documented in the existing `eval_output/evaluation_report.md`).
- If this run is a `final_eval` against `gold_pairs.csv`: an explicit note confirming it's the first and only time this model version has touched that set (the database trigger from `02_DATABASE_SUPABASE_SPEC.md` guarantees this mechanically, but say so in the file too, since this is the number the paper will cite).

## Non-negotiable while writing these files

- Never write a secret value, API key, or connection string into any `docs/results/*.md` file, even redacted-looking. Reference "Supabase" or "Kaggle kernel `<slug>`" by name; never a URL with credentials in it.
- Never back-fill a results file for a run that wasn't actually executed — if a phase gate hasn't been reached yet, don't write a placeholder file that looks like a real result. Missing is honest; a fabricated-looking number is not.
- If a run's outcome is a negative result (candidate recall too low, CDP's γ calibrates to ~0), write the file anyway, with the same rigor as a positive result. The existing project's five rejected architectures are already treated this way in `eval_output/evaluation_report.md` — keep that standard.
