
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


def test_get_engine_rebuilds_when_database_url_changes(monkeypatch, tmp_path):
    """
    Review issue L2: get_engine() used to return the cached engine even
    after DATABASE_URL changed, so seed_database(database_url=...) silently
    wrote to whichever DB the first engine pointed at.
    """
    memory_engine = get_engine()
    assert get_engine() is memory_engine  # unchanged URL -> same singleton

    file_url = f"sqlite:///{(tmp_path / 'other.db').as_posix()}"
    monkeypatch.setenv("DATABASE_URL", file_url)
    file_engine = get_engine()
    assert file_engine is not memory_engine
    assert file_engine.url.render_as_string(hide_password=False) == file_url
    file_engine.dispose()


def test_create_memory_engine_returns_isolated_databases():
    from db.connection import create_memory_engine

    a, b = create_memory_engine(), create_memory_engine()
    with a.begin() as conn:
        conn.execute(text(
            "INSERT INTO experiments (experiment_id, name, start_date) "
            "VALUES ('e1', 'x', '2025-01-01')"
        ))
    with b.connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM experiments")).scalar() == 0


def test_init_schema_adds_columns_missing_from_an_older_database():
    """
    CREATE TABLE IF NOT EXISTS never alters an existing table, so a
    database created before the planned-schedule columns existed must get
    them from init_schema's column migrations.
    """
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE experiments (experiment_id TEXT PRIMARY KEY, name TEXT NOT NULL, "
            "hypothesis TEXT, start_date DATE NOT NULL, end_date DATE, "
            "status TEXT NOT NULL DEFAULT 'running')"
        ))
        conn.execute(text(
            "INSERT INTO experiments (experiment_id, name, start_date) "
            "VALUES ('old', 'x', '2025-01-01')"
        ))

    init_schema(engine)
    init_schema(engine)  # idempotent: second run must not try to re-add

    with engine.connect() as conn:
        columns = {row[1] for row in conn.execute(text("PRAGMA table_info(experiments)"))}
        assert {"n_checkpoints_planned", "checkpoint_n"} <= columns
        assert conn.execute(text("SELECT COUNT(*) FROM experiments")).scalar() == 1
