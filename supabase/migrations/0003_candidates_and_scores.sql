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
