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
