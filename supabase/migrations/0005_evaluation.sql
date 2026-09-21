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
    raise exception 'eval_set % is heldout — only run_type=final_eval may touch it (got %)',
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
