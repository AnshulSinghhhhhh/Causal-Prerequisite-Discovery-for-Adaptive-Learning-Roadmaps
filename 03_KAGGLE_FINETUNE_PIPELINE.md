# LightGAP — Kaggle Fine-Tune + CDP Pipeline

Read `00_ANTIGRAVITY_BUILD_BRIEF.md` and `01_ARCHITECTURE_SPEC.md` (S3) first.

## A note on scope, stated as an assumption

"Fine-tune from scratch" here means a full QLoRA fine-tuning run starting from the pretrained `Qwen2.5-3B-Instruct` checkpoint — not pretraining a language model from random weights, which isn't feasible on a single T4 and isn't what the paper's design calls for. If that's not what was meant, say so before Job B below is built; everything else in this file works unchanged with the base model alone (Job B becomes optional rather than required).

## The load-bearing fact that shapes this whole file

**CDP is inference over a fixed per-edge quantity, not something computed per request.** `D(A→B)` doesn't change between learners, sessions, or requests — it's a property of the concept pair. So it's computed once in a batch job and written to `edge_scores.d_cdp`; the running application never loads a 3B model, it reads a float out of Postgres. The user's 8GB local RAM constraint is not a limitation this architecture works around — it's irrelevant to it, because the 3B model was never going to belong in the request path even on capable hardware.

## What gets fine-tuned, and why

Fine-tuning a classifier to predict prerequisite direction would duplicate the already-validated LR model's job (F1 0.704, precision 0.864) with a much more expensive model, for likely no gain — the reversal-pair accuracy (0.90) shows the LR model is already good at direction. That's not where fine-tuning helps.

What the paper's Equation 1 actually needs is a model that produces **consistent, well-grounded explanations of these specific domain concepts**, because `D(A→B)` is only a meaningful causal-necessity signal if the model's explanation quality is stable enough that withholding a real prerequisite produces a measurable perplexity delta, and withholding an unrelated concept doesn't. An off-the-shelf instruction-tuned model explaining "gradient descent" has no domain grounding beyond its pretraining; a model fine-tuned on this project's own harvested concept definitions (from S0) should produce sharper, more consistent explanations of exactly the concepts being probed — which is what makes the delta measurement trustworthy rather than noisy.

**So: Job B fine-tunes Qwen2.5-3B-Instruct via QLoRA on domain-explanation generation** (concept → grounded explanation), not on directional classification. Job A then uses the fine-tuned checkpoint to run the actual CDP probe from `01_ARCHITECTURE_SPEC.md` S3.

## Secrets

Set these once, before the first kernel push:

| Secret | Where | Used for |
|---|---|---|
| `HF_TOKEN` | Kaggle → Add-ons → Secrets, attached to the kernel | Downloading the base model, uploading the LoRA adapter to a private HF repo |
| `SUPABASE_URL` | Kaggle secret | Reading the work queue, writing results |
| `SUPABASE_SERVICE_KEY` | Kaggle secret | Same — service role, bypasses RLS (fine, this table has none) |

Nothing above goes in notebook source or the repo. Read them at runtime:

```python
from kaggle_secrets import UserSecretsClient
sec = UserSecretsClient()
HF_TOKEN = sec.get_secret("HF_TOKEN")
SB_URL   = sec.get_secret("SUPABASE_URL")
SB_KEY   = sec.get_secret("SUPABASE_SERVICE_KEY")
```

Locally (your machine, driving the kernel, not inside it), `~/.kaggle/kaggle.json` authenticates the CLI:

```bash
kaggle kernels push   -p kaggle/
kaggle kernels status -k <user>/lightgap-cdp-finetune
kaggle kernels output -k <user>/lightgap-cdp-finetune -p artifacts/
```

## Repo layout

```
lightgap/
  kaggle/
    kernel-metadata.json     # kernel id, GPU accelerator = "gpu-t4-x1" or "gpu-t4-x2", dataset deps
    finetune.py               # Job B
    cdp_probe.py               # Job A
    requirements.txt           # pin: transformers, peft, bitsandbytes, torch, supabase-py, pyarrow
    common/
      probe.py                 # teacher-forced NLL — the actual Equation 1 computation
      templates.py              # the 4 paraphrase templates already defined in config.py
  scripts/
    kaggle_submit.py            # push / poll / pull, run from your machine
    kaggle_ingest.py             # parquet output -> edge_scores.d_cdp, updates s_fused
```

## Job B — QLoRA fine-tune (the "from scratch" run)

**Training data:** For the very first run, before S0's harvester has produced a large corpus, use the definitions already present in `data/external/al-cpl/` (AL-CPL's four domains) as the source concept→explanation pairs — this is training data, not evaluation data, so using AL-CPL content here doesn't touch the leakage guard. As S0 harvests more domains over time, feed its `concepts.definition` rows in too. Format as instruction pairs:

```json
{"instruction": "Explain the concept: {concept_name}", "output": "{reference_explanation}"}
```

Where `reference_explanation` is either the AL-CPL/S0 definition directly (if it's already a few sentences of prose) or a Groq-generated expansion of it, lightly checked for factual consistency against the source.

**Hyperparameter search, and the freeze discipline — this is not optional:**

1. Grid over a small set of QLoRA configs on **cross-validation splits of the training data only** — never touch `gold_pairs.csv` at this stage:

   | Param | Candidates |
   |---|---|
   | LoRA rank | 8, 16, 32 |
   | Learning rate | 1e-4, 2e-4 |
   | Epochs | 2, 3 |

   Score each config by validation-split perplexity on held-out training-domain explanations (not by anything downstream of CDP yet — that comes later, and separately, in Job A's own calibration).

2. **Freeze the winning config.** Record it in `model_versions.hyperparams` for the row you're about to create.

3. Fine-tune once more on the full training corpus with the frozen config. This is the checkpoint that ships.

4. Export the LoRA adapter (tens of MB, not the full model) to a private HF Hub repo or directly to the `model-artifacts` Supabase Storage bucket from `02_DATABASE_SUPABASE_SPEC.md`. Insert a `model_versions` row: `kind = 'cdp_probe'`, `artifact_uri` pointing at the export, `hyperparams` recording the frozen config, `git_sha` of the kernel code.

```python
# kaggle/finetune.py — shape, not full implementation
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct", token=HF_TOKEN,
    load_in_4bit=True, device_map="auto",
)
lora_cfg = LoraConfig(r=FROZEN_RANK, lora_alpha=FROZEN_RANK*2,
                       target_modules=["q_proj","k_proj","v_proj","o_proj"],
                       lora_dropout=0.05, task_type="CAUSAL_LM")
model = get_peft_model(base, lora_cfg)
# ... Trainer with the frozen lr/epochs, then model.save_pretrained(...) and push to Hub
```

**Smoke test before the full run:** train on a tiny slice (a few dozen pairs, 1 epoch) first, confirm the kernel doesn't OOM on a T4, confirm checkpoint save/load round-trips, confirm the HF Hub / Storage push succeeds — the same "don't burn a full run on a broken pipeline" instinct already used for the AL-CPL calibration run elsewhere in this project.

## Job A — the CDP probe

Runs against the fine-tuned checkpoint from Job B (or the base model, if Job B hasn't landed yet — the probe code doesn't care which).

1. Pull the work queue directly from Supabase: `edge_scores` rows with `margin` in the bottom quartile and `d_cdp is null` (the partial index from `02_DATABASE_SUPABASE_SPEC.md` makes this a cheap query).
2. Load the model, `torch.manual_seed(42)`, `model.eval()`.
3. For each edge and each of the 4 paraphrase templates: build the standard prompt ("Explain concept B") and the constrained prompt ("Explain concept B without using or assuming any knowledge of concept A"), teacher-force the fixed reference explanation of B under both, take mean negative log-likelihood, compute:

   ```
   D(A→B) = (nll_constrained - nll_standard) / nll_standard
   ```

   This is Equation 1 from the paper, implemented as actual perplexity — not the word-count proxy the current repo's Groq-based probe uses, and not something achievable through the Groq API at all (every model on Groq rejects `logprobs=True`, already confirmed empirically in this project — that's exactly why this runs on Kaggle instead).

4. Average `D` across the 4 templates, write `(candidate_edge_id, d_cdp, n_templates, model_version_id)` to a parquet in `/kaggle/working/`.
5. Insert a `cdp_runs` row recording the kernel slug+version, pair count, start/finish times.

**Pin the model by commit revision**, not `main` — `AutoModelForCausalLM.from_pretrained(..., revision="<commit sha>")` — or results drift silently on a future run without anyone changing config.

## Score fusion and freeze

After Job A's first full pass, calibrate `γ` (the CDP term's weight in `s_fused = α·s_corr + β·s_lr + γ·norm(d_cdp)`) the same way `tau_edge` was calibrated: grid search on AL-CPL cross-validation, freeze, then one `final_eval` row against `gold_pairs.csv` for this specific model version — inserted once, enforced by the trigger in `02_DATABASE_SUPABASE_SPEC.md`.

**If `γ` calibrates to near zero, that is a valid and useful result, not a failed pipeline.** It would mean CDP doesn't move the fused score beyond what S1+S2 already provide — worth reporting in the paper exactly as the five already-rejected architectures are, with the same causal explanation for why. Don't force a non-zero weight to make the addition look like it worked.

## Ingestion back into the app

`scripts/kaggle_ingest.py` downloads the kernel's parquet output (`kaggle kernels output`), upserts into `edge_scores.d_cdp`, recomputes `s_fused` for the affected rows, and triggers a new `graph_snapshots` build via `01_ARCHITECTURE_SPEC.md` S4-S5 for any domain whose edges changed. The FastAPI service has no GPU dependency and no knowledge that Kaggle exists — it queries a table that happens to have been populated by a batch job.
