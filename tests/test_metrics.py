"""
tests/test_metrics.py — revenue and ratio metrics: the delta-method
estimator, winsorization, metric dispatch, simulator order values, and the
pipeline running each metric end to end.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.inference import delta_method_ratio_ci, raw_ttest_ci
from core.metrics import (
    METRICS,
    estimate_effect,
    metric_outcome,
    validate_metric,
    winsorization_cap,
)
from core.pipeline import analyze_experiment
from core.validity import achievable_mde_from_se
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


def _arm(revenue, conversions, variant):
    return pd.DataFrame({
        "variant_id": variant,
        "revenue": revenue,
        "n_conversions": conversions,
        "converted": (np.asarray(conversions) > 0).astype(int),
    })


# --------------------------------------------------------------------- #
# delta_method_ratio_ci
# --------------------------------------------------------------------- #

def test_delta_method_reduces_to_welch_se_when_every_denominator_is_one():
    """
    With N=1 for every user the ratio is just mean(Y), so the delta-method
    variance must equal s_Y^2/n per arm — the Welch standard error.
    """
    rng = np.random.default_rng(0)
    y_c, y_t = rng.normal(10, 3, 400), rng.normal(11, 4, 300)

    delta = delta_method_ratio_ci(y_c, np.ones(400), y_t, np.ones(300))
    welch = raw_ttest_ci(y_c, y_t)

    assert delta.point_estimate == pytest.approx(welch.point_estimate)
    assert delta.standard_error == pytest.approx(welch.standard_error, rel=1e-9)
    assert delta.degrees_freedom is None
    assert delta.method == "delta_ratio"


def test_delta_method_matches_bootstrap_standard_error():
    """
    Users with 0+ orders of random value: the delta-method SE of the
    difference in revenue per order must agree with a user-level bootstrap.
    """
    rng = np.random.default_rng(1)

    def arm(n, scale):
        orders = rng.poisson(0.8, n)
        revenue = np.array([rng.exponential(scale, k).sum() for k in orders])
        return revenue, orders.astype(float)

    y_c, n_c = arm(3000, 40.0)
    y_t, n_t = arm(3000, 44.0)
    result = delta_method_ratio_ci(y_c, n_c, y_t, n_t)

    boot = []
    for _ in range(1000):
        ic = rng.integers(0, 3000, 3000)
        it = rng.integers(0, 3000, 3000)
        boot.append(y_t[it].sum() / n_t[it].sum() - y_c[ic].sum() / n_c[ic].sum())
    assert result.standard_error == pytest.approx(np.std(boot, ddof=1), rel=0.1)


def test_delta_method_rejects_arm_without_conversions():
    with pytest.raises(ValueError):
        delta_method_ratio_ci(np.zeros(10), np.zeros(10), np.ones(10), np.ones(10))


# --------------------------------------------------------------------- #
# metric dispatch and winsorization
# --------------------------------------------------------------------- #

def test_validate_metric():
    for metric in METRICS:
        validate_metric(metric)
    with pytest.raises(ValueError):
        validate_metric("clicks")


def test_winsorization_cap_is_pooled_quantile_and_only_for_revenue_per_user():
    df = pd.concat([
        _arm(np.arange(100, dtype=float), np.ones(100), "x_control"),
        _arm(np.arange(100, 200, dtype=float), np.ones(100), "x_treatment"),
    ])
    assert winsorization_cap(df, "revenue_per_user", 0.99) == pytest.approx(
        np.quantile(np.arange(200), 0.99)
    )
    assert winsorization_cap(df, "conversion", 0.99) is None
    assert winsorization_cap(df, "revenue_per_user", None) is None
    with pytest.raises(ValueError):
        winsorization_cap(df, "revenue_per_user", 0.3)

    assert metric_outcome(df, "revenue_per_user", cap=50.0).max() == 50.0


def test_estimate_effect_dispatches_by_metric():
    control = _arm([0.0, 10.0, 20.0, 0.0], [0, 1, 2, 0], "x_control")
    treatment = _arm([5.0, 0.0, 30.0, 15.0], [1, 0, 1, 1], "x_treatment")

    conversion = estimate_effect(control, treatment, "conversion")
    revenue = estimate_effect(control, treatment, "revenue_per_user")
    ratio = estimate_effect(control, treatment, "revenue_per_conversion")

    assert conversion.point_estimate == pytest.approx(0.75 - 0.5)
    assert revenue.point_estimate == pytest.approx(12.5 - 7.5)
    assert ratio.point_estimate == pytest.approx(50 / 3 - 30 / 3)
    assert ratio.method == "delta_ratio"


def test_achievable_mde_from_se():
    assert achievable_mde_from_se(1.0) == pytest.approx(1.959964 + 0.841621, rel=1e-5)
    with pytest.raises(ValueError):
        achievable_mde_from_se(0.0)


# --------------------------------------------------------------------- #
# simulator order values
# --------------------------------------------------------------------- #

def test_simulator_order_values_have_configured_mean_and_lift():
    sim = ExperimentSimulator(
        n_users=200_000, baseline_rate=0.10, true_effect=0.0, covariate_correlation=0.0,
        seed=3, revenue_mean=50.0, revenue_lift=0.10, revenue_sigma=1.0,
    )
    _, assignments, events, gt = sim.generate()
    merged = events.merge(assignments[["user_id", "variant_id"]], on="user_id")
    is_treatment = merged["variant_id"].str.endswith("_treatment")

    assert merged.loc[~is_treatment, "value"].mean() == pytest.approx(50.0, rel=0.03)
    assert merged.loc[is_treatment, "value"].mean() == pytest.approx(55.0, rel=0.03)
    assert gt.revenue_mean == 50.0 and gt.revenue_lift == 0.10


def test_simulator_without_revenue_keeps_unit_values_and_same_data():
    """
    revenue_mean=None must not consume random draws: existing seeds must
    reproduce exactly the same assignments and conversions.
    """
    base = ExperimentSimulator(n_users=2000, baseline_rate=0.1, true_effect=0.02, seed=9)
    _, a1, e1, _ = base.generate()
    assert (e1["value"] == 1.0).all()

    with_revenue = ExperimentSimulator(
        n_users=2000, baseline_rate=0.1, true_effect=0.02, seed=9, revenue_mean=20.0
    )
    _, a2, e2, _ = with_revenue.generate()
    pd.testing.assert_frame_equal(a1, a2)
    assert list(e1["user_id"]) == list(e2["user_id"])


# --------------------------------------------------------------------- #
# pipeline end to end
# --------------------------------------------------------------------- #

def test_pipeline_runs_every_metric(isolated_db):
    seed_database({"simulator": {
        "n_users": 8000, "baseline_rate": 0.20, "true_effect": 0.0,
        "covariate_correlation": 0.3, "seed": 21, "corrupted_split": None,
        "revenue_mean": 50.0, "revenue_lift": 0.2,
    }}, database_url="sqlite:///:memory:")

    conversion = analyze_experiment(isolated_db, "exp_seed21", persist=False)
    rpu = analyze_experiment(
        isolated_db, "exp_seed21", metric_name="revenue_per_user",
        winsorize_quantile=0.99, segment_columns=["device_type"], persist=False,
    )
    aov = analyze_experiment(
        isolated_db, "exp_seed21", metric_name="revenue_per_conversion",
        segment_columns=["device_type"], persist=False,
    )

    assert conversion.raw_effect.method == "raw_ttest"
    assert rpu.winsorization_cap is not None and rpu.cuped_effect is not None
    assert rpu.achievable_mde is not None and rpu.segments is not None
    assert aov.raw_effect.method == "delta_ratio"
    assert aov.cuped_effect is None  # CUPED does not apply to ratio metrics
    # configured order-value lift: 50 * 0.2 = +10 per conversion
    assert aov.raw_effect.ci_lower < 10.0 < aov.raw_effect.ci_upper
    assert aov.segments[0].segments[0].inference.method == "delta_ratio"


def test_pipeline_rejects_unknown_metric(isolated_db):
    seed_database({"simulator": {
        "n_users": 500, "baseline_rate": 0.2, "true_effect": 0.0,
        "covariate_correlation": 0.0, "seed": 22, "corrupted_split": None,
    }}, database_url="sqlite:///:memory:")
    with pytest.raises(ValueError):
        analyze_experiment(isolated_db, "exp_seed22", metric_name="clicks", persist=False)
