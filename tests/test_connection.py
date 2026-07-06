
import pytest
from sqlalchemy import text

from db.connection import get_engine, init_schema, reset_engine


@pytest.fixture(autouse=True)
def isolated_engine(monkeypatch):
    """
    Forces every test in this file onto a fresh in-memory DB, isolated from
    whatever local.db might exist on disk and from any engine another test
    file already created in this process.
    """
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    reset_engine()
    yield
    reset_engine()


def test_init_schema_creates_all_tables():
    engine = get_engine()
    init_schema(engine)

    with engine.connect() as conn:
        tables = {
            row[0]
            for row in conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='table'")
            ).fetchall()
        }

    expected = {
        "users", "experiments", "variants", "assignments",
        "events", "experiment_results", "sequential_checkpoints", "srm_checks",
    }
    assert expected.issubset(tables)


def test_init_schema_is_idempotent():
    """Calling init_schema twice must not error — CREATE TABLE IF NOT EXISTS guarantees this."""
    engine = get_engine()
    init_schema(engine)
    init_schema(engine)  # should not raise
