-- migration 0001_extensions.sql
create extension if not exists vector;      -- pgvector, for concept embeddings
create extension if not exists pgcrypto;    -- gen_random_uuid()
