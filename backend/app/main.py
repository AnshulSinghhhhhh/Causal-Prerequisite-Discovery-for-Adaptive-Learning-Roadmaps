"""FastAPI application exposing the single JSON contract (Section 10).

Wires the three routers (graph / quiz / remediation) behind the canonical
schema in ``models/schema.py`` and the per-session live graph in
``session_store.py``. Every endpoint response validates against ``schema.py``,
enforced by ``tests/backend/test_schema_contract.py``.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import content, graph, quiz, remediation
from .models.schema import GraphPayload

app = FastAPI(
    title="LightGAP",
    description=(
        "Lightweight Graph-based Adaptive Pathway system: prerequisite-aware "
        "intelligent tutoring with budget-constrained path optimization and "
        "misconception-driven live graph rewriting."
    ),
    version="0.1.0",
)

# The React Flow frontend is served from a Vite dev server on a different
# origin; allow it during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(graph.router)
app.include_router(quiz.router)
app.include_router(remediation.router)
app.include_router(content.router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/openapi.json")
def openapi_schema() -> dict:
    """Expose the OpenAPI-derived contract for the frontend to type against."""
    return app.openapi()