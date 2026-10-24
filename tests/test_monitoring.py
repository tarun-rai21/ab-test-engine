"""
tests/test_monitoring.py — sequential monitoring per metric and per arm,
recording looks, and changing the planned sample size mid-test.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy.stats import norm
from sqlalchemy import text

from api.main import app
from core.data_access import get_sequential_checkpoints, save_checkpoint, save_planned_schedule
from core.monitoring import record_look
from core.pipeline import analyze_experiment
from core.sequential import lan_demets_boundary, sequential_check
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


def _seed_three_arms(seed: int = 171) -> str:
    seed_database({"simulator": {
        "n_users": 9000, "baseline_rate": 0.20, "true_effect": 0.0,
        "covariate_correlation": 0.0, "seed": seed, "corrupted_split": None,
        "revenue_mean": 40.0, "treatment_arms": {"treatment": 0.03, "treatment_b": 0.0},
    }}, database_url="sqlite:///:memory:")
    return f"exp_seed{seed}"


# --------------------------------------------------------------------- #
# Re-planning: lan_demets_boundary with fixed past thresholds
# --------------------------------------------------------------------- #

def test_boundary_without_fixed_looks_is_unchanged():
    fractions = [0.2, 0.45, 0.7, 1.0]
    assert lan_demets_boundary(fractions) == lan_demets_boundary(
        fractions, fixed_thresholds=[None] * 4
    )


def test_replanned_boundary_keeps_past_thresholds_and_controls_type_one_error():
    """
    Plan was 2000; looks at 400 and 800 used thresholds under that plan.
    The plan grows to 4000 and two more looks follow. Past thresholds must
    be kept exactly and the overall type-I error must still be ~5%
    (independent Monte Carlo on null Brownian paths at the actual looks).
    """
    old = lan_demets_boundary([400 / 2000, 800 / 2000])
    looks = np.array([400, 800, 2000, 4000])
    replanned = lan_demets_boundary(
        looks / 4000, fixed_thresholds=[old[0], old[1], None, None]
    )
    assert replanned[:2] == old

    t = looks / looks[-1]
    z_bounds = norm.isf(np.array(replanned) / 2)
    rng = np.random.default_rng(0)
    paths = np.cumsum(rng.standard_normal((200_000, 4)) * np.sqrt(np.diff(np.r_[0, t])), axis=1)
    crossed = (np.abs(paths / np.sqrt(t)) >= z_bounds).any(axis=1).mean()
    assert crossed == pytest.approx(0.05, abs=0.002)


def test_shrinking_plan_mid_test_also_controls_type_one_error():
    old = lan_demets_boundary([1000 / 4000, 2000 / 4000])
    looks = np.array([1000, 2000, 2500, 3000])
    replanned = lan_demets_boundary(looks / 3000, fixed_thresholds=[old[0], old[1], None, None])

    t = looks / looks[-1]
    z_bounds = norm.isf(np.array(replanned) / 2)
    rng = np.random.default_rng(1)
    paths = np.cumsum(rng.standard_normal((200_000, 4)) * np.sqrt(np.diff(np.r_[0, t])), axis=1)
    crossed = (np.abs(paths / np.sqrt(t)) >= z_bounds).any(axis=1).mean()
    assert crossed == pytest.approx(0.05, abs=0.002)


def test_sequential_check_fixes_thresholds_recorded_under_a_different_plan():
    old_thresholds = lan_demets_boundary([0.2, 0.4])
    history = [
        {"cumulative_n": 400, "p_value_at_check": 0.3,
         "alpha_threshold_at_check": old_thresholds[0], "planned_total_n": 2000},
        {"cumulative_n": 800, "p_value_at_check": 0.2,
         "alpha_threshold_at_check": old_thresholds[1], "planned_total_n": 2000},
        {"cumulative_n": 2000, "p_value_at_check": 0.004,
         "alpha_threshold_at_check": 0.05, "planned_total_n": 4000},
    ]
    result = sequential_check(history, n_checkpoints_planned=10, checkpoint_n=400)
    expected = lan_demets_boundary(
        [0.1, 0.2, 0.5], fixed_thresholds=[old_thresholds[0], old_thresholds[1], None]
    )[-1]

    assert result.replanned is True
    assert result.corrected_threshold == pytest.approx(expected)
    assert result.information_fraction == pytest.approx(0.5)

    # Under the original 2000 plan, the past looks were made under the plan
    # now in force (the latest look is always computed fresh), so nothing
    # is fixed and the plain boundary applies.
    same_plan = sequential_check(history, n_checkpoints_planned=5, checkpoint_n=400)
    assert same_plan.replanned is False
    assert same_plan.corrected_threshold == pytest.approx(lan_demets_boundary([0.2, 0.4, 1.0])[-1])


# --------------------------------------------------------------------- #
# Recording looks and per-arm / per-metric checks
# --------------------------------------------------------------------- #

def test_record_look_stores_one_row_per_arm_with_corrected_threshold(isolated_db):
    experiment_id = _seed_three_arms()
    save_planned_schedule(isolated_db, experiment_id, n_checkpoints_planned=10, checkpoint_n=900)

    looks = record_look(isolated_db, experiment_id, metric_name="conversion",
                        users_per_variant=600)

    assert [look.treatment_arm for look in looks] == ["treatment", "treatment_b"]
    assert all(look.cumulative_n == 1800 for look in looks)
    assert all(look.planned_total_n == 9000 for look in looks)
    expected_threshold = lan_demets_boundary([1800 / 9000])[0]
    assert all(look.alpha_threshold == pytest.approx(expected_threshold) for look in looks)

    rows = get_sequential_checkpoints(isolated_db, experiment_id, "conversion", "treatment_b")
    assert len(rows) == 1 and rows[0]["planned_total_n"] == 9000
    assert get_sequential_checkpoints(isolated_db, experiment_id, "revenue_per_user") == []


def test_record_look_without_plan_records_naive_threshold(isolated_db):
    experiment_id = _seed_three_arms(172)
    looks = record_look(isolated_db, experiment_id, users_per_variant=500)
    assert all(look.alpha_threshold == 0.05 and look.planned_total_n is None for look in looks)


def test_pipeline_checks_every_arm_on_the_analysed_metric(isolated_db):
    experiment_id = _seed_three_arms(173)
    save_planned_schedule(isolated_db, experiment_id, n_checkpoints_planned=10, checkpoint_n=900)
    for users in (500, 1500, 2900):
        record_look(isolated_db, experiment_id, metric_name="conversion", users_per_variant=users)
    record_look(isolated_db, experiment_id, metric_name="revenue_per_user", users_per_variant=900)

    conversion = analyze_experiment(isolated_db, experiment_id, persist=False)
    revenue = analyze_experiment(
        isolated_db, experiment_id, metric_name="revenue_per_user", persist=False
    )

    assert [r.treatment_arm for r in conversion.sequential_risks] == ["treatment", "treatment_b"]
    assert all(r.metric_name == "conversion" for r in conversion.sequential_risks)
    assert conversion.sequential_risk == conversion.sequential_risks[0]
    assert conversion.sequential_risks[0].latest_checkpoint_n == 8700
    assert revenue.sequential_risks[0].latest_checkpoint_n == 2700  # its own history


def test_legacy_checkpoint_rows_apply_to_every_metric_and_arm(isolated_db):
    """Rows logged before per-metric/per-arm monitoring have NULL metric and
    arm; they must still be picked up so old histories keep working."""
    experiment_id = _seed_three_arms(174)
    save_checkpoint(isolated_db, experiment_id, 3000, 0.04, 0.05)
    for metric in ("conversion", "revenue_per_user"):
        for arm in ("treatment", "treatment_b"):
            assert len(get_sequential_checkpoints(isolated_db, experiment_id, metric, arm)) == 1


def test_changing_the_plan_mid_test_is_reflected_in_the_report(isolated_db):
    experiment_id = _seed_three_arms(175)
    record_look(isolated_db, experiment_id, users_per_variant=600,
                n_checkpoints_planned=5, checkpoint_n=900)
    record_look(isolated_db, experiment_id, users_per_variant=1200,
                n_checkpoints_planned=5, checkpoint_n=900)
    save_planned_schedule(isolated_db, experiment_id, n_checkpoints_planned=10, checkpoint_n=900)
    record_look(isolated_db, experiment_id, users_per_variant=2900)

    report = analyze_experiment(isolated_db, experiment_id, persist=False)
    assert all(r.replanned for r in report.sequential_risks)
    with isolated_db.connect() as conn:
        plans = set(conn.execute(text(
            "SELECT DISTINCT planned_total_n FROM sequential_checkpoints"
        )).scalars())
    assert plans == {4500, 9000}


def test_api_records_checkpoints_for_every_arm(isolated_db):
    with TestClient(app) as client:
        experiment_id = client.post("/experiments", json={
            "name": "monitored",
            "variants": [
                {"name": "control", "split_pct": 0.34},
                {"name": "treatment", "split_pct": 0.33},
                {"name": "treatment_b", "split_pct": 0.33},
            ],
            "schedule": {"n_checkpoints_planned": 4, "checkpoint_n": 1500},
        }).json()["experiment_id"]
        client.post(f"/experiments/{experiment_id}/assign",
                    json={"n_users": 3000, "baseline_rate": 0.1})

        looks = client.post(f"/experiments/{experiment_id}/checkpoints",
                            json={"metric": "conversion"})
        assert looks.status_code == 200
        body = looks.json()
        assert [look["treatment_arm"] for look in body] == ["treatment", "treatment_b"]
        assert all(look["planned_total_n"] == 6000 for look in body)

        report = client.post(f"/experiments/{experiment_id}/analyze", json={}).json()
        assert len(report["sequential_risks"]) == 2
        assert report["sequential_risks"][1]["treatment_arm"] == "treatment_b"
