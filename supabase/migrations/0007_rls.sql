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
