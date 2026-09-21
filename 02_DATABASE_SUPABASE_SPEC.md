# LightGAP — Database Spec (Supabase / Postgres)

Read `00_ANTIGRAVITY_BUILD_BRIEF.md` and `01_ARCHITECTURE_SPEC.md` first. Supabase MCP is already connected — use its migration/SQL-execution tool directly rather than a local `psql` session. Put every migration below in `supabase/migrations/` under version control so the schema is part of the project's reproducibility story, then apply each one through the MCP tool in the order given.

## Design principle

This is a research record first, an application store second. Two things fall out of that:

1. **Every scored edge is tied to the model version that scored it.** Scores from different models coexist (`edge_scores` is keyed on `(candidate_edge_id, model_version_id)`), never overwritten.
2. **A held-out evaluation set can only be used for a `final_eval`, enforced by the schema, not by discipline.** This project has already had two leakage incidents (see `00_ANTIGRAVITY_BUILD_BRIEF.md`). The `evaluation_runs` constraint below makes the second class of that mistake a rejected INSERT.

## Extensions

```sql
-- migration 0001_extensions.sql
create extension if not exists vector;      -- pgvector, for concept embeddings
create extension if not exists pgcrypto;    -- gen_random_uuid()
```

## Core domain and corpus tables

```sql
-- migration 0002_domain_and_corpus.sql

create table domains (
  id           uuid primary key default gen_random_uuid(),
  goal_concept text not null,
  slug         text not null unique,
  created_at   timestamptz not null default now()
);

create table corpus_documents (
  id          uuid primary key default gen_random_uuid(),
  domain_id   uuid not null references domains(id) on delete cascade,
  source_type text not null,        -- 'wikipedia' | 'syllabus' | 'al_cpl' | 'other'
  source_url  text,
  title       text,
  raw_text    text not null,
  fetched_at  timestamptz not null default now()
);

create table concepts (
  id             uuid primary key default gen_random_uuid(),
  domain_id      uuid not null references domains(id) on delete cascade,
  canonical_name text not null,
  definition     text,
  source_doc_id  uuid references corpus_documents(id),
  source_span    int4range,
  embedding      vector(384),
  created_at     timestamptz not null default now(),
  unique (domain_id, canonical_name)
);
create index on concepts using hnsw (embedding vector_cosine_ops);

create table concept_aliases (
  concept_id uuid not null references concepts(id) on delete cascade,
  alias      text not null,
  primary key (concept_id, alias)
);
```

## Candidate and scoring layer

```sql
-- migration 0003_candidates_and_scores.sql

create table candidate_edges (
  id            uuid primary key default gen_random_uuid(),
  domain_id     uuid not null references domains(id) on delete cascade,
  src_id        uuid not null references concepts(id) on delete cascade,
  dst_id        uuid not null references concepts(id) on delete cascade,
  s_order       real,
  s_cooc        real,
  s_defmention  real,
  s_sim         real,       -- pruning only, per 01_ARCHITECTURE_SPEC.md S1 — never used to score direction
  s_corr        real not null,
  generated_at  timestamptz not null default now(),
  unique (src_id, dst_id),
  check (src_id <> dst_id)
);

create table model_versions (
  id               uuid primary key default gen_random_uuid(),
  name             text not null,             -- e.g. 'lr_asym_minilm', 'qwen25_3b_cdp_ft_v1'
  kind             text not null,             -- 'lr' | 'gnn' | 'cdp_probe' | 'fusion'
  git_sha          text not null,
  trained_on       text not null,             -- e.g. 'al_cpl_graph_cv_v2'
  hyperparams      jsonb not null default '{}',
  frozen_threshold numeric(8,6),
  artifact_uri     text,                      -- Supabase Storage or HF Hub, for anything with weights
  created_at       timestamptz not null default now(),
  unique (name, git_sha)
);

create table cdp_runs (
  id                uuid primary key default gen_random_uuid(),
  model_version_id  uuid not null references model_versions(id),
  kaggle_kernel_ref  text not null,           -- kernel slug + version, for provenance
  n_pairs_probed    int not null,
  n_templates       int not null,
  started_at        timestamptz not null,
  finished_at       timestamptz,
  notes             text
);

create table edge_scores (
  candidate_edge_id uuid not null references candidate_edges(id) on delete cascade,
  model_version_id  uuid not null references model_versions(id),
  s_lr_forward      real,
  s_lr_reverse      real,
  margin            real generated always as (abs(coalesce(s_lr_forward,0) - coalesce(s_lr_reverse,0))) stored,
  admitted          boolean,
  d_cdp             real,
  cdp_run_id        uuid references cdp_runs(id),
  s_fused           real,
  scored_at         timestamptz not null default now(),
  primary key (candidate_edge_id, model_version_id)
);

-- the S3 work queue: low-margin, unprobed edges, ordered for pickup
create index edge_scores_cdp_queue
  on edge_scores (margin)
  where d_cdp is null;
```

## Graph, tree, and cluster layer

```sql
-- migration 0004_graph_and_tree.sql

create table graph_snapshots (
  id                uuid primary key default gen_random_uuid(),
  domain_id         uuid not null references domains(id),
  model_version_id  uuid not null references model_versions(id),
  tau_edge          numeric(8,6) not null,
  n_nodes           int not null,
  n_edges           int not null,
  build_stats       jsonb not null default '{}',  -- dropped_threshold / cycle / transitive counts
  created_at        timestamptz not null default now()
);

create table dag_edges (
  snapshot_id    uuid not null references graph_snapshots(id) on delete cascade,
  src_id         uuid not null references concepts(id),
  dst_id         uuid not null references concepts(id),
  confidence     real not null,
  admitted       boolean not null,
  dropped_reason text,    -- 'threshold' | 'cycle_prune' | 'transitive' | null
  primary key (snapshot_id, src_id, dst_id)
);

create table clusters (
  id           uuid primary key default gen_random_uuid(),
  snapshot_id  uuid not null references graph_snapshots(id) on delete cascade,
  label        text,                -- LLM-authored, per 01_ARCHITECTURE_SPEC.md S5 step 3
  created_at   timestamptz not null default now()
);

alter table concepts add column if not exists cluster_id uuid references clusters(id);
alter table concepts add column if not exists depth int;   -- per-snapshot depth is really a property of
                                                             -- (snapshot, concept) — see note below

-- tree_paths as a closure table: arbitrary depth at O(1) ancestor/descendant queries,
-- which is what the frontend's recursive renderer needs (see 04_UI_DESIGN_BRIEF.md)
create table tree_paths (
  snapshot_id   uuid not null references graph_snapshots(id) on delete cascade,
  ancestor_id   uuid not null references concepts(id),
  descendant_id uuid not null references concepts(id),
  depth         int not null,
  primary key (snapshot_id, ancestor_id, descendant_id)
);
```

**Note on `concepts.depth`/`cluster_id` vs snapshots:** a concept's depth and cluster membership are technically properties of a specific `graph_snapshots` run, not of the concept globally — a concept can appear in multiple domains/snapshots over time. The columns above are a pragmatic denormalization for the *current* snapshot only. If the project ends up needing multiple live snapshots per domain simultaneously, move `depth`/`cluster_id` into a `snapshot_concepts` join table instead. Don't over-build this before it's needed — start with the denormalized version above.

## Evaluation layer — the leakage guard

```sql
-- migration 0005_evaluation.sql

create type eval_purpose as enum ('calibration','heldout');
create type run_type     as enum ('calibration','model_selection','final_eval');

create table eval_sets (
  id           uuid primary key default gen_random_uuid(),
  name         text not null unique,     -- 'al_cpl_cv', 'gold_pairs_v1'
  purpose      eval_purpose not null,
  n_pairs      int not null,
  sha256       text not null,            -- content hash; pins the exact file version
  created_at   timestamptz not null default now()
);

create table evaluation_runs (
  id               uuid primary key default gen_random_uuid(),
  model_version_id uuid not null references model_versions(id),
  eval_set_id      uuid not null references eval_sets(id),
  run_type         run_type not null,
  metrics          jsonb not null,       -- accuracy, precision, recall, f1, auc, confusion, subclass breakdown
  threshold_used   numeric(8,6) not null,
  git_sha          text not null,
  created_at       timestamptz not null default now()
);

-- Postgres CHECK constraints can't subquery, so enforce with a trigger:
create or replace function forbid_selection_on_heldout()
returns trigger as $$
declare
  p eval_purpose;
begin
  select purpose into p from eval_sets where id = new.eval_set_id;
  if p = 'heldout' and new.run_type <> 'final_eval' then
    raise exception 'eval_set % is heldout — only run_type=final_eval may touch it (got %)',
      new.eval_set_id, new.run_type;
  end if;
  return new;
end;
$$ language plpgsql;

create trigger trg_forbid_selection_on_heldout
  before insert or update on evaluation_runs
  for each row execute function forbid_selection_on_heldout();

-- a heldout set may be scored at most once per model version
create unique index one_shot_heldout
  on evaluation_runs (model_version_id, eval_set_id)
  where run_type = 'final_eval';
```

**This is the constraint that matters most in the whole schema.** It means an agent (human or LLM) cannot accidentally repeat the ensemble-selection leakage bug from earlier in this project — the database itself refuses the insert.

## Learner layer

```sql
-- migration 0006_learners.sql

create table learners (
  id          uuid primary key default gen_random_uuid(),
  auth_uid    uuid,                       -- Supabase auth.uid(), nullable for anonymous/demo sessions
  created_at  timestamptz not null default now()
);

create table mastery (
  learner_id  uuid not null references learners(id) on delete cascade,
  concept_id  uuid not null references concepts(id) on delete cascade,
  value       real not null default 0 check (value between 0 and 1),
  updated_at  timestamptz not null default now(),
  primary key (learner_id, concept_id)
);

create table quiz_items (
  id             uuid primary key default gen_random_uuid(),
  concept_id     uuid not null references concepts(id) on delete cascade,
  stem           text not null,
  correct_option text not null,
  prereq_distractor      text not null,
  conceptual_distractor  text not null,
  misconception_tag      text,
  created_at     timestamptz not null default now()
);

create table quiz_responses (
  id            uuid primary key default gen_random_uuid(),
  learner_id    uuid not null references learners(id) on delete cascade,
  quiz_item_id  uuid not null references quiz_items(id),
  chosen_option text not null,
  correct       boolean not null,
  answered_at   timestamptz not null default now()
);   -- append-only by convention: never update or delete rows here.
     -- this is the dataset that eventually lets edge confidences be refined from real
     -- learner behaviour, per the paper's closed-loop design (§4.6).

create table misconceptions (
  id            uuid primary key default gen_random_uuid(),
  learner_id    uuid not null references learners(id) on delete cascade,
  concept_id    uuid not null references concepts(id),
  tag           text not null,
  detected_from uuid references quiz_responses(id),
  detected_at   timestamptz not null default now()
);

create table remediation_nodes (
  id                uuid primary key default gen_random_uuid(),
  misconception_id  uuid not null references misconceptions(id) on delete cascade,
  snapshot_id       uuid not null references graph_snapshots(id),
  inserted_before   uuid not null references concepts(id),
  content           text not null,
  created_at        timestamptz not null default now()
);
```

## Row Level Security

```sql
-- migration 0007_rls.sql

alter table learners           enable row level security;
alter table mastery             enable row level security;
alter table quiz_responses      enable row level security;
alter table misconceptions      enable row level security;

create policy learner_owns_self on learners
  for all using (auth_uid = auth.uid());

create policy learner_owns_mastery on mastery
  for all using (learner_id in (select id from learners where auth_uid = auth.uid()));

create policy learner_owns_responses on quiz_responses
  for all using (learner_id in (select id from learners where auth_uid = auth.uid()));

create policy learner_owns_misconceptions on misconceptions
  for all using (learner_id in (select id from learners where auth_uid = auth.uid()));

-- Everything above the learner layer (domains, concepts, candidate_edges, edge_scores,
-- graph_snapshots, evaluation_runs, ...) is research/application data, not per-user —
-- leave RLS off and access it only with the service role key from the backend.
```

## Storage

Create a Supabase Storage bucket for anything with weights:

```sql
insert into storage.buckets (id, name, public) values ('model-artifacts', 'model-artifacts', false);
```

`model_versions.artifact_uri` points here for the LoRA adapter export from `03_KAGGLE_FINETUNE_PIPELINE.md` (small — tens of MB). The base Qwen2.5-3B weights themselves are not re-uploaded here; they're pulled from Hugging Face Hub by the Kaggle kernel each run.

## Seed migration — register the eval sets, nothing else

This is a fresh repo, so there is no prior evaluation run to backfill. The only thing this migration does is register the two eval sets once you have their files in hand, so P1's `model_versions` insert and P3's `final_eval` insert (both happen for real, later, from actual runs) have something to reference:

```sql
-- migration 0008_seed_eval_sets.sql

insert into eval_sets (name, purpose, n_pairs, sha256) values
  ('al_cpl_cv',     'calibration', <actual count after download>, '<sha256 of the AL-CPL file you trained on>'),
  ('gold_pairs_v1', 'heldout',     105,                             '<sha256 of data/labeled/gold_pairs.csv>');
```

Compute the `sha256` values with `sha256sum` against the files actually in the repo — don't leave them as placeholder text in the live database. If `gold_pairs.csv` isn't in the repo yet (see `00_ANTIGRAVITY_BUILD_BRIEF.md`), insert only the `al_cpl_cv` row and add `gold_pairs_v1` once the file exists.

An earlier build of this same project reached F1 0.7037 / AUC 0.8053 on an equivalent held-out set, at threshold 0.435804 — useful as the target ballpark named in `00_ANTIGRAVITY_BUILD_BRIEF.md`, but don't insert it as a row here. It isn't a result this repo produced. P1 and P3 in the build brief will insert the real `model_versions` and `evaluation_runs` rows once your own fresh calibration and evaluation actually run.

## Hardening addition (0009) — see `06_S0_S6_HARDENING.md`

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

`est_minutes` backs S6's per-node cost estimate; `rejected_concepts` is the audit trail for S0's canonicalization filters, mirroring the `dropped_reason` pattern already used on `dag_edges`. Apply this after `0008`, through the same MCP flow as everything else.

## Signal addition (0010) — see `07_S1_SIGNAL_HARDENING.md`

```sql
-- migration 0010_llm_plausibility_signal.sql
alter table candidate_edges add column if not exists s_llm_plaus real;
```

The fifth S1 signal — a cheap batched plausibility check that catches what pure association signals can't (word-sense collisions, topically-adjacent-but-non-prerequisite concepts). Apply after `0009`.

## Applying migrations

Apply `0001` through `0008` in order through the Supabase MCP tool (its SQL-execution or migration-apply action — check what it's named in your current MCP tool list, it varies by MCP server version). Keep each file under `supabase/migrations/NNNN_name.sql` in the repo regardless of how it's applied, so the schema history is reviewable in git.