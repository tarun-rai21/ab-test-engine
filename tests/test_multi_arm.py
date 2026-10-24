"""
tests/test_multi_arm.py — A/B/n experiments: Holm adjustment, arm
splitting, multi-arm simulation, the pipeline comparing every arm against
control, and the API end to end.
"""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from api.main import app
from core.data_access import split_arms, split_by_variant
from core.multiple_testing import holm_adjust
from core.pipeline import analyze_experiment
from data_sim.simulator import ExperimentSimulator
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


THREE_ARMS = {"treatment": 0.04, "treatment_b": 0.0}


# --------------------------------------------------------------------- #
# holm_adjust
# --------------------------------------------------------------------- #

def test_holm_adjust_known_values_in_input_order():
    # sorted 0.01, 0.03, 0.04 -> 3*0.01, max(0.03, 2*0.03), max(0.06, 1*0.04)
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_holm_adjust_single_p_value_unchanged_and_capped_at_one():
    assert holm_adjust([0.2]) == [0.2]
    assert holm_adjust([0.6, 0.7]) == pytest.approx([1.0, 1.0])


def test_holm_adjust_invalid_inputs():
    with pytest.raises(ValueError):
        holm_adjust([])
    with pytest.raises(ValueError):
        holm_adjust([1.5])


# --------------------------------------------------------------------- #
# split_arms
# --------------------------------------------------------------------- #

def test_split_arms_uses_variant_name_and_sorts_arms():
    df = pd.DataFrame({
        "variant_id": ["e_control", "e_treatment_b", "e_treatment", "e_control"],
        "variant_name": ["control", "treatment_b", "treatment", "control"],
        "converted": [0, 1, 1, 0],
    })
    control, arms = split_arms(df)
    assert len(control) == 2
    assert list(arms) == ["treatment", "treatment_b"]
    with pytest.raises(ValueError):
        split_by_variant(df)  # more than one arm


def test_split_arms_requires_control_and_an_arm():
    with pytest.raises(ValueError):
        split_arms(pd.DataFrame({"variant_id": ["e_control"], "variant_name": ["control"]}))
    with pytest.raises(ValueError):
        split_arms(pd.DataFrame({"variant_id": ["e_a"], "variant_name": ["a"]}))


# --------------------------------------------------------------------- #
# simulator
# --------------------------------------------------------------------- #

def test_simulator_multi_arm_allocation_and_effects():
    sim = ExperimentSimulator(
        n_users=60_000, baseline_rate=0.10, true_effect=0.0, covariate_correlation=0.0,
        seed=5, treatment_arms=THREE_ARMS,
        allocation={"control": 0.5, "treatment": 0.25, "treatment_b": 0.25},
    )
    _, assignments, _, gt = sim.generate()
    shares = assignments["variant_id"].value_counts(normalize=True)

    assert shares["exp_seed5_control"] == pytest.approx(0.5, abs=0.01)
    assert shares["exp_seed5_treatment_b"] == pytest.approx(0.25, abs=0.01)
    assert gt.arm_effects_configured == THREE_ARMS
    assert gt.arm_effects_realized["treatment"] == pytest.approx(0.04, abs=1e-9)
    assert sim.intended_allocation()["treatment"] == 0.25


@pytest.mark.parametrize("kwargs", [
    {"treatment_arms": {"control": 0.1}},
    {"treatment_arms": THREE_ARMS, "corrupted_split": 0.4},
    {"treatment_arms": THREE_ARMS, "allocation": {"control": 0.5, "treatment": 0.5}},
    {"allocation": {"control": 0.7, "treatment": 0.7}},
])
def test_simulator_rejects_invalid_arm_configuration(kwargs):
    with pytest.raises(ValueError):
        ExperimentSimulator(n_users=100, baseline_rate=0.1, true_effect=0.0, **kwargs)


# --------------------------------------------------------------------- #
# pipeline
# --------------------------------------------------------------------- #

def test_pipeline_compares_every_arm_against_control(isolated_db):
    seed_database({"simulator": {
        "n_users": 30000, "baseline_rate": 0.10, "true_effect": 0.0,
        "covariate_correlation": 0.3, "seed": 31, "corrupted_split": None,
        "treatment_arms": THREE_ARMS,
    }}, database_url="sqlite:///:memory:")

    report = analyze_experiment(isolated_db, "exp_seed31", segment_columns=["device_type"])

    assert report.srm.flagged is False
    assert len(report.srm.observed_counts) == 3
    by_arm = {c.treatment_arm: c for c in report.comparisons}
    assert list(by_arm) == ["treatment", "treatment_b"]
    assert report.raw_effect == by_arm["treatment"].raw_effect  # primary = first arm

    strong, null = by_arm["treatment"], by_arm["treatment_b"]
    assert strong.raw_effect.ci_lower < 0.04 < strong.raw_effect.ci_upper
    assert strong.raw_p_value_holm < 0.05
    assert null.raw_p_value_holm >= null.raw_effect.p_value
    assert strong.cuped_effect is not None and strong.cuped_p_value_holm is not None

    assert {s.treatment_arm for s in report.segments} == {"treatment", "treatment_b"}

    with isolated_db.connect() as conn:
        variant_ids = set(conn.execute(text(
            "SELECT DISTINCT variant_id FROM experiment_results "
            "WHERE experiment_id = 'exp_seed31' AND method = 'raw_ttest' AND segment IS NULL"
        )).scalars())
    assert variant_ids == {"exp_seed31_treatment", "exp_seed31_treatment_b"}


def test_two_arm_report_has_single_comparison_with_unadjusted_p(isolated_db):
    seed_database({"simulator": {
        "n_users": 2000, "baseline_rate": 0.10, "true_effect": 0.02,
        "covariate_correlation": 0.0, "seed": 32, "corrupted_split": None,
    }}, database_url="sqlite:///:memory:")
    report = analyze_experiment(isolated_db, "exp_seed32", persist=False)

    assert len(report.comparisons) == 1
    only = report.comparisons[0]
    assert only.treatment_arm == "treatment"
    assert only.raw_p_value_holm == pytest.approx(only.raw_effect.p_value)


# --------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------- #

def test_api_multi_arm_experiment_end_to_end(isolated_db):
    with TestClient(app) as client:
        created = client.post("/experiments", json={
            "name": "three arms",
            "variants": [
                {"name": "control", "split_pct": 0.5},
                {"name": "treatment", "split_pct": 0.25},
                {"name": "treatment_b", "split_pct": 0.25},
            ],
        })
        experiment_id = created.json()["experiment_id"]

        assigned = client.post(f"/experiments/{experiment_id}/assign", json={
            "n_users": 8000, "baseline_rate": 0.1, "true_effect": 0.0,
            "arm_effects": {"treatment": 0.05},
        })
        assert assigned.status_code == 200
        # unequal 50/25/25 design simulated as declared -> no SRM
        assert assigned.json()["srm_check"]["flagged"] is False

        report = client.post(f"/experiments/{experiment_id}/analyze", json={}).json()
        assert [c["treatment_arm"] for c in report["comparisons"]] == ["treatment", "treatment_b"]

        bad = client.post(f"/experiments/{experiment_id}/assign", json={
            "n_users": 1000, "baseline_rate": 0.1, "arm_effects": {"treatment_z": 0.1},
        })
        assert bad.status_code == 422


def test_api_unequal_two_arm_design_is_not_falsely_srm_flagged(isolated_db):
    """
    The simulator used to split 50/50 regardless of the declared design, so
    an experiment created as 90/10 was SRM-flagged on its own healthy data.
    """
    with TestClient(app) as client:
        created = client.post("/experiments", json={
            "name": "90/10",
            "variants": [
                {"name": "control", "split_pct": 0.9},
                {"name": "treatment", "split_pct": 0.1},
            ],
        })
        experiment_id = created.json()["experiment_id"]
        assigned = client.post(f"/experiments/{experiment_id}/assign", json={
            "n_users": 5000, "baseline_rate": 0.1,
        })
        assert assigned.json()["srm_check"]["flagged"] is False
