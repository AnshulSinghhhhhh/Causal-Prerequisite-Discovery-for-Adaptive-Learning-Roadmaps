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
