"""
tests/test_api.py — end-to-end tests for the Phase 7 FastAPI layer
(api/main.py, api/routers/experiments.py, api/routers/analysis.py).

Uses FastAPI's TestClient, which calls the app in-process via ASGI —
no real server or uvicorn process needed to run these. TestClient is used
as a context manager (`with TestClient(app) as client`) specifically so
the app's startup event (schema initialization) actually fires against
the freshly-isolated in-memory database, not the default one.

Same isolated_db discipline used throughout this project since Phase 2:
each test gets a fresh in-memory SQLite database, so tests never see each
other's data.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from api.main import app
from db.connection import get_engine, reset_engine


@pytest.fixture(autouse=True)
def isolated_db(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    reset_engine()
    yield
    reset_engine()


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def _create_experiment(client, name="Test experiment"):
    response = client.post(
        "/experiments",
        json={
            "name": name,
            "hypothesis": "Treatment increases conversion",
            "variants": [
                {"name": "control", "split_pct": 0.5},
                {"name": "treatment", "split_pct": 0.5},
            ],
        },
    )
    assert response.status_code == 200
    return response.json()["experiment_id"]


def _assign(client, experiment_id, **overrides):
    body = {
        "mode": "simulate",
        "n_users": 2000,
        "baseline_rate": 0.10,
        "true_effect": 0.02,
        "seed": 42,
    }
    body.update(overrides)
    return client.post(f"/experiments/{experiment_id}/assign", json=body)


# --------------------------------------------------------------------- #
# POST /experiments
# --------------------------------------------------------------------- #


def test_create_experiment_returns_id_and_status(client):
    response = client.post(
        "/experiments",
        json={
            "name": "My experiment",
            "hypothesis": "Something",
            "variants": [
                {"name": "control", "split_pct": 0.5},
                {"name": "treatment", "split_pct": 0.5},
            ],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["experiment_id"].startswith("exp_")
    assert body["status"] == "created"


def test_create_experiment_variants_must_sum_to_one(client):
    """
    Real validation added at the API boundary — split_pct values that
    don't sum to 1.0 must be rejected BEFORE anything is written to the
    database, not discovered later as a confusing downstream SRM result.
    """
    response = client.post(
        "/experiments",
        json={
            "name": "Bad split",
            "variants": [
                {"name": "control", "split_pct": 0.5},
                {"name": "treatment", "split_pct": 0.6},  # sums to 1.1
            ],
        },
    )
    assert response.status_code == 422


# --------------------------------------------------------------------- #
# POST /experiments/{id}/assign
# --------------------------------------------------------------------- #


def test_assign_generates_data_and_returns_healthy_srm(client):
    experiment_id = _create_experiment(client)
    response = _assign(client, experiment_id)

    assert response.status_code == 200
    body = response.json()
    assert body["n_assigned"] == 2000
    assert body["srm_check"]["flagged"] is False


def test_assign_nonexistent_experiment_returns_404(client):
    response = _assign(client, "exp_does_not_exist")
    assert response.status_code == 404


def test_assign_unsupported_mode_returns_400(client):
    experiment_id = _create_experiment(client)
    response = _assign(client, experiment_id, mode="ingest")
    assert response.status_code == 400


def test_assign_rejects_experiment_without_control_and_treatment_variants(client):
    """
    Real gap caught during manual testing (Swagger UI's own auto-filled
    example created a single variant named "string"): the simulator ALWAYS
    generates variant_ids using the hardcoded _control/_treatment suffixes,
    completely independent of whatever variant names the experiment was
    actually created with. Without this check, mismatched variant names
    would silently produce orphaned assignment rows referencing variant_ids
    that don't exist. Must be caught here, loudly, not discovered later.
    """
    response = client.post(
        "/experiments",
        json={
            "name": "Mismatched variants",
            "variants": [
                {"name": "A", "split_pct": 0.5},
                {"name": "B", "split_pct": 0.5},
            ],
        },
    )
    experiment_id = response.json()["experiment_id"]

    assign_response = _assign(client, experiment_id)
    assert assign_response.status_code == 400
    assert "control" in assign_response.json()["detail"]


def test_assign_srm_flagged_on_corrupted_split(client):
    """
    Mirrors this project's own established SRM-detection tests
    (test_full_pipeline_detects_corrupted_split) — proving the API surface
    correctly reports a REAL randomization problem, not just the happy path.
    """
    experiment_id = _create_experiment(client)
    response = _assign(client, experiment_id, n_users=20000, corrupted_split=0.45)

    assert response.status_code == 200
    assert response.json()["srm_check"]["flagged"] is True


# --------------------------------------------------------------------- #
# POST /experiments/{id}/analyze  and  GET /experiments/{id}/report
# --------------------------------------------------------------------- #


def test_analyze_returns_full_report_shape(client):
    experiment_id = _create_experiment(client)
    _assign(client, experiment_id)

    response = client.post(
        f"/experiments/{experiment_id}/analyze",
        json={"metric": "conversion", "use_cuped": True},
    )
    assert response.status_code == 200
    body = response.json()

    assert body["experiment_id"] == experiment_id
    assert body["srm"]["flagged"] is False
    assert body["achievable_mde"] is not None
    assert body["raw_effect"]["method"] == "raw_ttest"
    assert body["cuped_effect"]["method"] == "cuped"
    assert body["cuped_variance_reduction_pct"] is not None
    assert body["sequential_risk"] is None  # not requested
    assert body["segments"] is None  # not requested


def test_analyze_with_use_cuped_false_omits_cuped_effect(client):
    experiment_id = _create_experiment(client)
    _assign(client, experiment_id)

    response = client.post(
        f"/experiments/{experiment_id}/analyze",
        json={"metric": "conversion", "use_cuped": False},
    )
    body = response.json()
    assert body["cuped_effect"] is None
    assert body["cuped_variance_reduction_pct"] is None


def test_analyze_with_segment_columns_returns_segment_analysis(client):
    experiment_id = _create_experiment(client)
    _assign(client, experiment_id, n_users=8000)

    response = client.post(
        f"/experiments/{experiment_id}/analyze",
        json={"metric": "conversion", "segment_columns": ["device_type"]},
    )
    body = response.json()
    assert body["segments"] is not None
    assert len(body["segments"]) == 1
    assert body["segments"][0]["segment_column"] == "device_type"


def test_analyze_nonexistent_experiment_returns_404(client):
    response = client.post(
        "/experiments/exp_does_not_exist/analyze", json={"metric": "conversion"}
    )
    assert response.status_code == 404


def test_report_matches_analyze_and_does_not_double_persist(client):
    """
    Proves the specific design decision documented in
    api/routers/analysis.py's module docstring: GET /report recomputes via
    persist=False and must NOT write additional rows to experiment_results
    beyond what POST /analyze already persisted, even though it returns an
    equally complete report.
    """
    experiment_id = _create_experiment(client)
    _assign(client, experiment_id)

    analyze_response = client.post(
        f"/experiments/{experiment_id}/analyze", json={"metric": "conversion"}
    )
    assert analyze_response.status_code == 200

    engine = get_engine()
    count_after_analyze = engine.connect().execute(
        text("SELECT COUNT(*) FROM experiment_results WHERE experiment_id = :eid"),
        {"eid": experiment_id},
    ).fetchone()[0]
    assert count_after_analyze == 2  # raw + cuped

    report_response = client.get(f"/experiments/{experiment_id}/report")
    assert report_response.status_code == 200
    assert report_response.json() == analyze_response.json()

    count_after_report = engine.connect().execute(
        text("SELECT COUNT(*) FROM experiment_results WHERE experiment_id = :eid"),
        {"eid": experiment_id},
    ).fetchone()[0]
    assert count_after_report == count_after_analyze  # unchanged — no double-persist


def test_report_nonexistent_experiment_returns_404(client):
    response = client.get("/experiments/exp_does_not_exist/report")
    assert response.status_code == 404
