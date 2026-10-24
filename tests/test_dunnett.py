"""
tests/test_dunnett.py — the step-down Dunnett-type adjustment for A/B/n
experiments and its wiring through the pipeline and API.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy.stats import norm
from sqlalchemy import text

from api.main import app
from core.multiple_testing import dunnett_adjust, holm_adjust
from core.pipeline import analyze_experiment
from db.connection import get_engine, init_schema, reset_engine
from db.seed import seed_database


@pytest.fixture
def isolated_db(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    reset_engine()
    engine = get_engine()
    init_schema(engine)
    yield engine
    reset_engine()


ESTIMATES = [0.02, 0.015, 0.004]
SES = [0.008, 0.008, 0.008]


def _raw(estimates, ses):
    return [2 * norm.sf(abs(e) / s) for e, s in zip(estimates, ses)]


def test_independent_comparisons_reduce_to_step_down_sidak():
    """With no shared-control covariance, max-|Z| probabilities factorise:
    the most significant arm's adjusted p is 1 - (1 - p)^3."""
    raw = _raw(ESTIMATES, SES)
    adjusted = dunnett_adjust(ESTIMATES, SES, control_variance=0.0)
    assert adjusted[0] == pytest.approx(1 - (1 - raw[0]) ** 3, abs=1e-4)
    assert adjusted[1] == pytest.approx(1 - (1 - raw[1]) ** 2, abs=1e-4)
    assert adjusted[2] == pytest.approx(raw[2], abs=1e-9)


def test_shared_control_makes_dunnett_less_conservative_than_holm():
    raw = _raw(ESTIMATES, SES)
    control_variance = 0.5 * SES[0] ** 2  # control contributes half of each variance
    dunnett = dunnett_adjust(ESTIMATES, SES, control_variance, raw_p_values=raw)
    holm = holm_adjust(raw)

    assert all(d <= h + 1e-12 for d, h in zip(dunnett, holm))
    assert dunnett[0] < holm[0]
    assert all(d >= r for d, r in zip(dunnett, raw))  # never below the raw p-value


def test_dunnett_is_monotone_in_significance_order():
    adjusted = dunnett_adjust(ESTIMATES, SES, 0.3 * SES[0] ** 2)
    order = np.argsort(-np.abs(np.array(ESTIMATES) / np.array(SES)))
    sorted_adjusted = [adjusted[i] for i in order]
    assert sorted_adjusted == sorted(sorted_adjusted)


def test_single_arm_returns_raw_p_value():
    assert dunnett_adjust([0.02], [0.01], 0.00005, raw_p_values=[0.0461]) == [0.0461]


def test_dunnett_invalid_inputs():
    with pytest.raises(ValueError):
        dunnett_adjust([], [], 0.0)
    with pytest.raises(ValueError):
        dunnett_adjust([0.1, 0.2], [0.05, 0.0], 0.0)
    with pytest.raises(ValueError):
        dunnett_adjust([0.1, 0.2], [0.05], 0.0)


def test_pipeline_defaults_to_dunnett_and_supports_holm(isolated_db):
    seed_database({"simulator": {
        "n_users": 30000, "baseline_rate": 0.10, "true_effect": 0.0,
        "covariate_correlation": 0.3, "seed": 161, "corrupted_split": None,
        "treatment_arms": {"treatment": 0.012, "treatment_b": 0.008, "treatment_c": 0.0},
    }}, database_url="sqlite:///:memory:")

    dunnett = analyze_experiment(isolated_db, "exp_seed161", persist=False)
    holm = analyze_experiment(
        isolated_db, "exp_seed161", persist=False, multiple_comparison="holm"
    )

    assert dunnett.multiple_comparison == "dunnett"
    assert holm.multiple_comparison == "holm"
    for d, h in zip(dunnett.comparisons, holm.comparisons):
        assert d.raw_effect == h.raw_effect
        assert d.raw_p_value_adjusted <= h.raw_p_value_adjusted + 1e-9
        assert d.raw_p_value_adjusted >= d.raw_effect.p_value
        assert d.cuped_p_value_adjusted <= h.cuped_p_value_adjusted + 1e-9

    with pytest.raises(ValueError):
        analyze_experiment(isolated_db, "exp_seed161", persist=False,
                           multiple_comparison="bonferroni")


def test_api_still_serves_reports_stored_with_old_holm_field_names(isolated_db):
    """Reports persisted before the Dunnett option used *_holm keys and had
    no multiple_comparison field; GET /report must still return them."""
    with TestClient(app) as client:
        experiment_id = client.post("/experiments", json={"name": "old"}).json()["experiment_id"]
        client.post(f"/experiments/{experiment_id}/assign",
                    json={"n_users": 2000, "baseline_rate": 0.1})
        report = client.post(f"/experiments/{experiment_id}/analyze", json={}).json()

        legacy = dict(report)
        legacy.pop("multiple_comparison")
        legacy["comparisons"] = [
            {**{k: v for k, v in c.items() if not k.endswith("_adjusted")},
             "raw_p_value_holm": c["raw_p_value_adjusted"],
             "cuped_p_value_holm": c["cuped_p_value_adjusted"]}
            for c in report["comparisons"]
        ]
        with get_engine().begin() as conn:
            conn.execute(text("UPDATE analysis_reports SET report_json = :r "
                              "WHERE experiment_id = :eid"),
                         {"r": json.dumps(legacy), "eid": experiment_id})

        served = client.get(f"/experiments/{experiment_id}/report")
        assert served.status_code == 200
        body = served.json()
        assert body["multiple_comparison"] == "holm"
        assert body["comparisons"][0]["raw_p_value_adjusted"] == pytest.approx(
            report["comparisons"][0]["raw_p_value_adjusted"]
        )
