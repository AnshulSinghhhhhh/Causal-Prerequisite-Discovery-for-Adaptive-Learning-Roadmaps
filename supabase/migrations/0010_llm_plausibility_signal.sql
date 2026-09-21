-- migration 0010_llm_plausibility_signal.sql

alter table candidate_edges add column if not exists s_llm_plaus real;
