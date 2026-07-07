"""
tests/test_persistence.py — verifies the Option B contract: trust-tagging
is computed from SRMResult, never independently settable, and persistence
is idempotent.
"""

import pytest
from sqlalchemy import text

from core.inference import InferenceResult
from core.persistence import persist_inference_result
from core.validity import SRMResult
from db.connection import get_engine, init_schema, reset_engine


@pytest.fixture(autouse=True)
def isolated_db(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    reset_engine()
    engine = get_engine()
    init_schema(engine)
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO experiments (experiment_id, name, start_date, status) "
            "VALUES ('exp_p', 'test', '2025-01-01', 'running')"
        ))
    yield engine
    reset_engine()


def _make_inference_result(method="raw_ttest"):
    return InferenceResult(
        method=method, point_estimate=0.02, ci_lower=0.01, ci_upper=0.03,
        alpha=0.05, p_value=0.001, standard_error=0.005, degrees_freedom=1998.0,
        n_control=1000, n_treatment=1000,
    )


def _make_srm_result(flagged: bool):
    return SRMResult(
        observed_counts=(10000, 10000), expected_counts=(10000.0, 10000.0),
        chi_sq_stat=0.0 if not flagged else 500.0,
        p_value=1.0 if not flagged else 1e-50,
        flagged=flagged,
    )


def test_trusted_true_when_srm_not_flagged(isolated_db):
    engine = isolated_db
    rid = persist_inference_result(
        engine, "exp_p", "conversion", _make_inference_result(), _make_srm_result(flagged=False)
    )
    row = engine.connect().execute(
        text("SELECT trusted FROM experiment_results WHERE result_id = :r"), {"r": rid}
    ).fetchone()
    assert row.trusted == 1


def test_trusted_false_when_srm_flagged(isolated_db):
    """
    THE core contract: an SRM-flagged experiment must NEVER produce a
    trusted=True row, regardless of how good the effect estimate looks.
    """
    engine = isolated_db
    rid = persist_inference_result(
        engine, "exp_p", "conversion", _make_inference_result(), _make_srm_result(flagged=True)
    )
    row = engine.connect().execute(
        text("SELECT trusted FROM experiment_results WHERE result_id = :r"), {"r": rid}
    ).fetchone()
    assert row.trusted == 0


def test_requires_both_arguments_present(isolated_db):
    with pytest.raises(ValueError):
        persist_inference_result(
            isolated_db, "exp_p", "conversion", None, _make_srm_result(flagged=False)
        )
    with pytest.raises(ValueError):
        persist_inference_result(
            isolated_db, "exp_p", "conversion", _make_inference_result(), None
        )


def test_persistence_is_idempotent_not_duplicating(isolated_db):
    """
    Re-running the same analysis slice must OVERWRITE, not accumulate.
    Same idempotency principle as db/seed.py's delete-then-insert.
    """
    engine = isolated_db
    persist_inference_result(engine, "exp_p", "conversion", _make_inference_result(), _make_srm_result(False))
    persist_inference_result(engine, "exp_p", "conversion", _make_inference_result(), _make_srm_result(False))

    count = engine.connect().execute(
        text("SELECT COUNT(*) FROM experiment_results WHERE experiment_id = 'exp_p'")
    ).fetchone()[0]
    assert count == 1, f"Expected 1 row after two identical persists, got {count}"


def test_different_methods_produce_separate_rows(isolated_db):
    """
    raw_ttest and cuped for the SAME experiment/metric must NOT collide —
    result_id includes method, so both should coexist as separate rows.
    """
    engine = isolated_db
    persist_inference_result(
        engine, "exp_p", "conversion", _make_inference_result("raw_ttest"), _make_srm_result(False)
    )
    persist_inference_result(
        engine, "exp_p", "conversion", _make_inference_result("cuped"), _make_srm_result(False)
    )
    count = engine.connect().execute(
        text("SELECT COUNT(*) FROM experiment_results WHERE experiment_id = 'exp_p'")
    ).fetchone()[0]
    assert count == 2