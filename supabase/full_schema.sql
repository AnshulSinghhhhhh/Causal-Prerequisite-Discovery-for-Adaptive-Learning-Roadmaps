-- migration 0001_extensions.sql
create extension if not exists vector;      -- pgvector, for concept embeddings
create extension if not exists pgcrypto;    -- gen_random_uuid()
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
-- migration 0003_candidates_and_scores.sql

create table candidate_edges (
  id            uuid primary key default gen_random_uuid(),
  domain_id     uuid not null references domains(id) on delete cascade,
  src_id        uuid not null references concepts(id) on delete cascade,
  dst_id        uuid not null references concepts(id) on delete cascade,
  s_order       real,
  s_cooc        real,
  s_defmention  real,
  s_sim         real,
  s_llm_plaus   real,
  s_corr        real not null,
  generated_at  timestamptz not null default now(),
  unique (src_id, dst_id),
  check (src_id <> dst_id)
);

create table model_versions (
  id               uuid primary key default gen_random_uuid(),
  name             text not null,
  kind             text not null,
  git_sha          text not null,
  trained_on       text not null,
  hyperparams      jsonb not null default '{}',
  frozen_threshold numeric(8,6),
  artifact_uri     text,
  created_at       timestamptz not null default now(),
  unique (name, git_sha)
);

create table cdp_runs (
  id                uuid primary key default gen_random_uuid(),
  model_version_id  uuid not null references model_versions(id),
  kaggle_kernel_ref  text not null,
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

create index edge_scores_cdp_queue
  on edge_scores (margin)
  where d_cdp is null;
-- migration 0004_graph_and_tree.sql

create table graph_snapshots (
  id                uuid primary key default gen_random_uuid(),
  domain_id         uuid not null references domains(id),
  model_version_id  uuid not null references model_versions(id),
  tau_edge          numeric(8,6) not null,
  n_nodes           int not null,
  n_edges           int not null,
  build_stats       jsonb not null default '{}',
  created_at        timestamptz not null default now()
);

create table dag_edges (
  snapshot_id    uuid not null references graph_snapshots(id) on delete cascade,
  src_id         uuid not null references concepts(id),
  dst_id         uuid not null references concepts(id),
  confidence     real not null,
  admitted       boolean not null,
  dropped_reason text,
  primary key (snapshot_id, src_id, dst_id)
);

create table clusters (
  id           uuid primary key default gen_random_uuid(),
  snapshot_id  uuid not null references graph_snapshots(id) on delete cascade,
  label        text,
  created_at   timestamptz not null default now()
);

alter table concepts add column if not exists cluster_id uuid references clusters(id);
alter table concepts add column if not exists depth int;

create table tree_paths (
  snapshot_id   uuid not null references graph_snapshots(id) on delete cascade,
  ancestor_id   uuid not null references concepts(id),
  descendant_id uuid not null references concepts(id),
  depth         int not null,
  primary key (snapshot_id, ancestor_id, descendant_id)
);
-- migration 0005_evaluation.sql

create type eval_purpose as enum ('calibration','heldout');
create type run_type     as enum ('calibration','model_selection','final_eval');

create table eval_sets (
  id           uuid primary key default gen_random_uuid(),
  name         text not null unique,
  purpose      eval_purpose not null,
  n_pairs      int not null,
  sha256       text not null,
  created_at   timestamptz not null default now()
);

create table evaluation_runs (
  id               uuid primary key default gen_random_uuid(),
  model_version_id uuid not null references model_versions(id),
  eval_set_id      uuid not null references eval_sets(id),
  run_type         run_type not null,
  metrics          jsonb not null,
  threshold_used   numeric(8,6) not null,
  git_sha          text not null,
  created_at       timestamptz not null default now()
);

create or replace function forbid_selection_on_heldout()
returns trigger as $$
declare
  p eval_purpose;
begin
  select purpose into p from eval_sets where id = new.eval_set_id;
  if p = 'heldout' and new.run_type <> 'final_eval' then
    raise exception 'eval_set % is heldout â€” only run_type=final_eval may touch it (got %)',
      new.eval_set_id, new.run_type;
  end if;
  return new;
end;
$$ language plpgsql;

create trigger trg_forbid_selection_on_heldout
  before insert or update on evaluation_runs
  for each row execute function forbid_selection_on_heldout();

create unique index one_shot_heldout
  on evaluation_runs (model_version_id, eval_set_id)
  where run_type = 'final_eval';
-- migration 0006_learners.sql

create table learners (
  id          uuid primary key default gen_random_uuid(),
  auth_uid    uuid,
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
);

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
-- migration 0008_seed_eval_sets.sql

insert into eval_sets (name, purpose, n_pairs, sha256) values
  ('al_cpl_cv',     'calibration', 844, '9B0D42AC047F7E4B09F77210AFCCE5B4639F026C608D31F1D49E71C6B718E5DA'),
  ('gold_pairs_v1', 'heldout',     105, '735BB8691B6F42097A0AC30670D045305B979A97B49B4A866D8C1522B3ADE5FE');

insert into storage.buckets (id, name, public) values ('model-artifacts', 'model-artifacts', false);
