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
