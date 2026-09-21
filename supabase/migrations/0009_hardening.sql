-- migration 0009_hardening.sql

alter table concepts add column if not exists est_minutes int not null default 15;

create table if not exists rejected_concepts (
  id           uuid primary key default gen_random_uuid(),
  domain_id    uuid not null references domains(id) on delete cascade,
  raw_phrase   text not null,
  stage        text not null,   -- 'section_strip' | 'pos_filter' | 'llm_verify' | 'merged_duplicate'
  reason       text,
  merged_into  uuid references concepts(id),   -- set only when stage = 'merged_duplicate'
  created_at   timestamptz not null default now()
);
