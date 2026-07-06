"""
Single point of truth for how the app talks to the database.

Nothing else in the codebase should construct an engine, know the schema file's
path, or care whether we're on SQLite or Postgres. That isolation is the point:
core/, api/, and app/ all import get_engine()/get_connection() from here and never
touch a connection string directly.

Env var:
    DATABASE_URL  — defaults to sqlite:///local.db (zero-setup local dev, NFR5).
                    For Postgres parity: postgresql+psycopg2://user:pass@host/db
"""

from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.pool import StaticPool

SCHEMA_PATH = Path(__file__).parent / "schema.sql"

_engine: Engine | None = None


def get_database_url() -> str:
    return os.environ.get("DATABASE_URL", "sqlite:///local.db")


def get_engine(echo: bool = False) -> Engine:
    """
    Process-wide singleton engine. Re-creating engines per call is a common
    SQLAlchemy footgun (connection pool churn); guarded against here.

    in-memory SQLite (":memory:") gets StaticPool so the same connection —
    and therefore the same in-memory DB — is reused across calls; without this,
    each new connection sees an empty, freshly-created database.
    """
    global _engine
    if _engine is not None:
        return _engine

    url = get_database_url()
    connect_args: dict = {}
    engine_kwargs: dict = {"echo": echo}

    if url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
        if ":memory:" in url:
            engine_kwargs["poolclass"] = StaticPool

    _engine = create_engine(url, connect_args=connect_args, **engine_kwargs)
    return _engine


def reset_engine() -> None:
    """Test isolation helper: forces get_engine() to build a fresh engine next call."""
    global _engine
    if _engine is not None:
        _engine.dispose()
    _engine = None


def init_schema(engine: Engine | None = None) -> None:
    """
    Executes schema.sql against the target DB. Idempotent (CREATE TABLE IF NOT
    EXISTS), safe to call on every app startup / test setup.
    """
    engine = engine or get_engine()
    ddl = SCHEMA_PATH.read_text()

    with engine.begin() as conn:
        for statement in _split_statements(ddl):
            conn.execute(text(statement))


def _split_statements(ddl: str) -> list[str]:
    """
    Strips SQL line comments first, then splits on ';'.

    Splitting on ';' before removing comments is unsafe: a comment containing
    a semicolon in ordinary prose (we hit this — a comment read "...directly;
    do not redefine...") gets treated as a statement boundary, and the
    trailing comment fragment then gets executed as literal SQL.

    Still naive in one respect: this does not handle a semicolon or '--'
    appearing inside a string literal (e.g. a DEFAULT 'a;b' value). Not a risk
    for this schema (no such literals exist), but don't extend this function's
    trust boundary without checking that assumption still holds.
    """
    lines = []
    for line in ddl.splitlines():
        comment_start = line.find("--")
        if comment_start != -1:
            line = line[:comment_start]
        lines.append(line)

    ddl_no_comments = "\n".join(lines)
    statements = [s.strip() for s in ddl_no_comments.split(";")]
    return [s for s in statements if s]
