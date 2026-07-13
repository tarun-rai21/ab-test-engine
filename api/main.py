"""
api/main.py — FastAPI app entrypoint. Wires api/routers/experiments.py and
api/routers/analysis.py together, initializes the schema on startup, and
gets automatic interactive docs at /docs for free (FastAPI's built-in
OpenAPI generation) — per the spec's own reasoning for choosing FastAPI:
"interviewers can literally open /docs and try it."

Run (from project root):
    uvicorn api.main:app --reload

DATABASE_URL is read the same way every other part of this project reads
it (via db.connection.get_engine()) — defaults to a local SQLite file if
not set, exactly like running any test or db/seed.py script locally.
init_schema() is idempotent (proven in tests/test_connection.py since
Phase 0) — safe to call on every startup without wiping existing data on
a restart, unlike the in-memory, wiped-per-test databases used throughout
this project's own test suite.

Uses the `lifespan` context-manager pattern (not the older
@app.on_event("startup") decorator, which FastAPI has deprecated) for
startup initialization.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.routers import analysis, experiments
from db.connection import get_engine, init_schema


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = get_engine()
    init_schema(engine)
    yield
    # No teardown needed — nothing to close; get_engine()'s connection
    # pooling is managed by SQLAlchemy itself.


app = FastAPI(
    title="A/B Test Analysis Engine API",
    description=(
        "Thin HTTP layer over core/pipeline.py's analyze_experiment() — "
        "SRM detection, power/MDE context, raw + CUPED effect estimation, "
        "sequential-peeking risk, and segment analysis with Simpson's-"
        "paradox detection, all backed by code independently validated "
        "in Phases 1-6."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(experiments.router)
app.include_router(analysis.router)


@app.get("/")
def root() -> dict:
    return {"status": "ok", "docs": "/docs"}