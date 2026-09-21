-- migration 0008_seed_eval_sets.sql

insert into eval_sets (name, purpose, n_pairs, sha256) values
  ('al_cpl_cv',     'calibration', 844, '9B0D42AC047F7E4B09F77210AFCCE5B4639F026C608D31F1D49E71C6B718E5DA'),
  ('gold_pairs_v1', 'heldout',     105, '735BB8691B6F42097A0AC30670D045305B979A97B49B4A866D8C1522B3ADE5FE');

insert into storage.buckets (id, name, public) values ('model-artifacts', 'model-artifacts', false);
