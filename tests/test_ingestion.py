"""
tests/test_ingestion.py — mapping real-world data onto the schema
(core.ingestion) and the API's POST /experiments/{id}/ingest.
"""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from api.main import app
from core.ingestion import (
    create_experiment_with_variants,
    ingest_canonical,
    parse_binary,
    to_canonical,
    validate_canonical,
)
from core.pipeline import analyze_experiment
from db.connection import create_memory_engine, get_engine, init_schema, reset_engine

# A spreadsheet in someone else's shape: own column names, A/B/QA labels,
# yes/no conversion flags.
RAW = pd.DataFrame({
    "customer": [f"c{i}" for i in range(12)],
    "bucket": ["A", "B"] * 5 + ["QA", "QA"],
    "bought": ["yes", "no", "no", "yes", "yes", "yes", "no", "no", "yes", "no", "yes", "no"],
    "order_total": [20.0, 0, 0, 35.5, 18.0, 22.0, 0, 0, 40.0, 0, 15.0, 0],
    "pre_spend": [1.0, 2.0, 1.5, 3.0, 0.5, 2.5, 1.0, 1.2, 2.2, 0.8, 1.0, 1.0],
    "device": ["mobile", "desktop"] * 6,
})
COLUMNS = {
    "variant": "bucket", "converted": "bought", "revenue": "order_total",
    "user_id": "customer", "pre_period_covariate": "pre_spend", "device_type": "device",
}
VARIANTS = {"A": "control", "B": "treatment", "QA": None}


def test_parse_binary_accepts_common_encodings():
    parsed = parse_binary(pd.Series(["Yes", "no", "TRUE", "0", "1", "maybe"]))
    assert parsed.iloc[:5].tolist() == [1, 0, 1, 0, 1]
    assert pd.isna(parsed.iloc[5])
    assert parse_binary(pd.Series([True, False])).tolist() == [1, 0]
    assert pd.isna(parse_binary(pd.Series([0, 1, 2])).iloc[2])


def test_mapping_renames_columns_relabels_variants_and_drops_ignored_rows():
    canonical = to_canonical(RAW, COLUMNS, VARIANTS)

    assert len(canonical) == 10  # the two QA rows are dropped
    assert set(canonical["variant_name"]) == {"control", "treatment"}
    assert canonical["converted"].tolist()[:4] == [1, 0, 0, 1]
    assert canonical["user_id"].iloc[0] == "c0"
    assert validate_canonical(canonical) == []


def test_converted_can_be_derived_from_revenue():
    columns = {k: v for k, v in COLUMNS.items() if k != "converted"}
    canonical = to_canonical(RAW, columns, VARIANTS)
    assert canonical["converted"].tolist() == (RAW["order_total"][:10] > 0).astype(int).tolist()


def test_unmapped_variant_values_are_reported():
    canonical = to_canonical(RAW, COLUMNS, {"A": "control", "B": "treatment"})  # QA unmapped
    assert any("no recognised variant" in e for e in validate_canonical(canonical))


@pytest.mark.parametrize("columns, message", [
    ({"converted": "bought"}, "'variant' column must be mapped"),
    ({"variant": "bucket"}, "Map a 'converted' column"),
    ({"variant": "missing_col", "converted": "bought"}, "not found"),
    ({"variant": "bucket", "converted": "bought", "colour": "device"}, "Unknown column roles"),
])
def test_invalid_mappings_raise(columns, message):
    with pytest.raises(ValueError, match=message):
        to_canonical(RAW, columns, VARIANTS)


def test_default_convention_needs_no_mapping():
    df = pd.DataFrame({"variant_id": ["x_control", "x_treatment_b"], "converted": [1, 0]})
    canonical = to_canonical(df)
    assert canonical["variant_name"].tolist() == ["control", "treatment_b"]


def test_ingest_into_existing_experiment_replaces_earlier_data():
    engine = create_memory_engine()
    create_experiment_with_variants(engine, "exp_real", ["treatment"], treatment_split=0.5)
    canonical = to_canonical(RAW, COLUMNS, VARIANTS)

    assert ingest_canonical(engine, "exp_real", canonical) == 10
    assert ingest_canonical(engine, "exp_real", canonical.iloc[:6]) == 6  # re-ingest replaces

    with engine.connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM assignments")).scalar() == 6
        assert conn.execute(text("SELECT COUNT(*) FROM users")).scalar() == 6
    report = analyze_experiment(engine, "exp_real", metric_name="revenue_per_user",
                                persist=False)
    assert report.raw_effect.n_control + report.raw_effect.n_treatment == 6


def test_ingest_rejects_variants_the_experiment_does_not_have():
    engine = create_memory_engine()
    create_experiment_with_variants(engine, "exp_two_arm", ["treatment"])
    canonical = to_canonical(RAW, COLUMNS, {"A": "control", "B": "treatment_b", "QA": None})
    with pytest.raises(ValueError, match="not defined"):
        ingest_canonical(engine, "exp_two_arm", canonical)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    reset_engine()
    init_schema(get_engine())
    with TestClient(app) as test_client:
        yield test_client
    reset_engine()


def test_api_ingest_maps_real_data_and_checks_srm(client):
    experiment_id = client.post("/experiments", json={"name": "real"}).json()["experiment_id"]
    response = client.post(f"/experiments/{experiment_id}/ingest", json={
        "rows": RAW.to_dict(orient="records"),
        "columns": COLUMNS,
        "variant_values": VARIANTS,
    })
    assert response.status_code == 200
    body = response.json()
    assert body["n_ingested"] == 10
    assert body["srm_check"]["flagged"] is False

    report = client.post(f"/experiments/{experiment_id}/analyze",
                         json={"metric": "revenue_per_conversion"})
    assert report.status_code == 200


def test_api_ingest_reports_data_problems_as_422(client):
    experiment_id = client.post("/experiments", json={"name": "bad"}).json()["experiment_id"]
    bad = RAW.assign(bought="sometimes").to_dict(orient="records")
    response = client.post(f"/experiments/{experiment_id}/ingest", json={
        "rows": bad, "columns": COLUMNS, "variant_values": VARIANTS,
    })
    assert response.status_code == 422
    assert "converted" in response.json()["detail"]

    missing = client.post("/experiments/nope/ingest", json={"rows": [{"a": 1}]})
    assert missing.status_code == 404


def test_assign_ingest_mode_points_to_the_ingest_endpoint(client):
    experiment_id = client.post("/experiments", json={"name": "x"}).json()["experiment_id"]
    response = client.post(f"/experiments/{experiment_id}/assign", json={
        "mode": "ingest", "n_users": 10, "baseline_rate": 0.1,
    })
    assert response.status_code == 400
    assert "/ingest" in response.json()["detail"]
