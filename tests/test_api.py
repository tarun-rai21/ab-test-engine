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
    GET /report serves the report stored by POST /analyze — identical
    content — and writes nothing (no extra experiment_results rows).
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


@pytest.mark.parametrize("overrides", [
    {"covariate_correlation": -0.6},          # infeasible for the simulator's skewed covariate
    {"corrupted_split": 1.5},                 # outside (0, 1)
    {"segment_heterogeneity": {"device_type": {"mobile": 0.1}, "region": {"US": 0.1}}},
    {"segment_heterogeneity": {"browser": {"chrome": 0.1}}},  # not a simulated column
])
def test_assign_invalid_simulator_input_returns_422_not_500(client, overrides):
    """Review issue L1: these all used to surface as 500 Internal Server Error."""
    experiment_id = _create_experiment(client)
    response = _assign(client, experiment_id, **overrides)
    assert response.status_code == 422


def test_create_experiment_duplicate_variant_names_returns_422(client):
    response = client.post("/experiments", json={
        "name": "dup",
        "variants": [
            {"name": "control", "split_pct": 0.5},
            {"name": "control", "split_pct": 0.5},
        ],
    })
    assert response.status_code == 422


def test_create_experiment_zero_split_variant_returns_422(client):
    response = client.post("/experiments", json={
        "name": "zero",
        "variants": [
            {"name": "control", "split_pct": 0.5},
            {"name": "treatment", "split_pct": 0.5},
            {"name": "treatment_b", "split_pct": 0.0},
        ],
    })
    assert response.status_code == 422


def test_analyze_rejects_unknown_spending_method(client):
    experiment_id = _create_experiment(client)
    _assign(client, experiment_id)
    response = client.post(
        f"/experiments/{experiment_id}/analyze", json={"spending": "pocock"}
    )
    assert response.status_code == 422


def test_planned_schedule_can_be_stored_and_drives_the_peeking_check(client):
    """
    A schedule stored at creation (or via PUT /schedule) is used by
    POST /analyze without being re-supplied.
    """
    response = client.post("/experiments", json={
        "name": "scheduled",
        "schedule": {"n_checkpoints_planned": 10, "checkpoint_n": 200},
    })
    assert response.status_code == 200
    experiment_id = response.json()["experiment_id"]
    _assign(client, experiment_id)

    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO sequential_checkpoints (checkpoint_id, experiment_id, checked_at, "
            "cumulative_n, p_value_at_check, alpha_threshold_at_check) "
            "VALUES ('cp1', :eid, '2025-01-02', 1000, 0.2, 0.05)"
        ), {"eid": experiment_id})

    report = client.post(f"/experiments/{experiment_id}/analyze", json={}).json()
    assert report["sequential_risk"]["information_fraction"] == pytest.approx(0.5)

    put = client.put(
        f"/experiments/{experiment_id}/schedule",
        json={"n_checkpoints_planned": 5, "checkpoint_n": 200},
    )
    assert put.status_code == 200
    report = client.post(f"/experiments/{experiment_id}/analyze", json={}).json()
    assert report["sequential_risk"]["information_fraction"] == pytest.approx(1.0)

    missing = client.put(
        "/experiments/does_not_exist/schedule",
        json={"n_checkpoints_planned": 5, "checkpoint_n": 200},
    )
    assert missing.status_code == 404


def test_report_is_read_from_storage_not_recomputed(client):
    """
    GET /report must return the STORED analysis — including options such
    as segment columns that a recomputation with defaults would drop — and
    404 for a metric that was never analysed.
    """
    experiment_id = _create_experiment(client)
    _assign(client, experiment_id)

    missing = client.get(f"/experiments/{experiment_id}/report")
    assert missing.status_code == 404
    assert "POST" in missing.json()["detail"]

    analyzed = client.post(
        f"/experiments/{experiment_id}/analyze",
        json={"segment_columns": ["device_type"], "use_cuped": False},
    ).json()
    report = client.get(f"/experiments/{experiment_id}/report")
    assert report.status_code == 200
    assert report.json() == analyzed
    assert report.json()["segments"] is not None and report.json()["cuped_effect"] is None
    assert report.headers["X-Report-Computed-At"]

    # Change the underlying data: the stored report must NOT change until
    # the experiment is re-analysed.
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM events WHERE experiment_id = :eid"), {"eid": experiment_id})
    assert client.get(f"/experiments/{experiment_id}/report").json() == analyzed

    other_metric = client.get(f"/experiments/{experiment_id}/report?metric=revenue_per_user")
    assert other_metric.status_code == 404


def test_persisted_results_include_test_statistics(client):
    experiment_id = _create_experiment(client)
    _assign(client, experiment_id)
    report = client.post(f"/experiments/{experiment_id}/analyze", json={}).json()

    with get_engine().connect() as conn:
        row = conn.execute(text(
            "SELECT p_value, standard_error, degrees_freedom, n_control, n_treatment "
            "FROM experiment_results WHERE experiment_id = :eid AND method = 'raw_ttest'"
        ), {"eid": experiment_id}).fetchone()
    raw = report["raw_effect"]
    assert row.p_value == pytest.approx(raw["p_value"])
    assert row.standard_error == pytest.approx(raw["standard_error"])
    assert row.n_control + row.n_treatment == raw["n_control"] + raw["n_treatment"]


def test_api_key_is_required_when_configured(client, monkeypatch):
    """With API_KEY set, data endpoints need a matching X-API-Key header;
    '/' and '/health' stay open."""
    monkeypatch.setenv("API_KEY", "s3cret-key")

    assert client.post("/experiments", json={"name": "x"}).status_code == 401
    wrong = client.post("/experiments", json={"name": "x"}, headers={"X-API-Key": "nope"})
    assert wrong.status_code == 401
    ok = client.post("/experiments", json={"name": "x"}, headers={"X-API-Key": "s3cret-key"})
    assert ok.status_code == 200

    assert client.get("/").status_code == 200
    assert client.get("/health").json() == {"status": "ok", "database": "ok"}


def test_api_is_open_when_no_key_is_configured(client, monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    assert client.post("/experiments", json={"name": "x"}).status_code == 200
