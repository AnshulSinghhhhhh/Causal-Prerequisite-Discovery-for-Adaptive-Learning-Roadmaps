"""Supabase client singleton for LightGAP.

All database access goes through this module. Reads connection details
from environment variables. Prefer Supabase MCP calls for migrations;
use this client for runtime application queries.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv

load_dotenv()  # reads .env at project root


@lru_cache(maxsize=1)
def get_supabase():
    """Return a cached Supabase client (service-role, bypasses RLS)."""
    from supabase import create_client, Client

    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_KEY"]
    return create_client(url, key)


@lru_cache(maxsize=1)
def get_supabase_anon():
    """Return a cached Supabase client (anon key, respects RLS)."""
    from supabase import create_client, Client

    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_ANON_KEY"]
    return create_client(url, key)


def table(name: str):
    """Shorthand: get_supabase().table(name)."""
    return get_supabase().table(name)


def fetch_all(query_builder, chunk_size: int = 1000) -> list:
    """Fetch all rows from a Supabase PostgREST query builder using range pagination."""
    rows = []
    start = 0
    while True:
        res = query_builder.range(start, start + chunk_size - 1).execute()
        data = res.data or []
        rows.extend(data)
        if len(data) < chunk_size:
            break
        start += chunk_size
    return rows

