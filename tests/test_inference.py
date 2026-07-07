"""
tests/test_inference.py — closed-form and cross-consistency tests for
core/inference.py. Zero SRM/database dependency, per the module's own
purity constraint.
"""
import numpy as np
import pytest

from core.inference import cuped_adjust, raw_ttest_ci, variance_reduction_pct


def test_raw_ttest_recovers_known_effect():
    """
    Large n, large true effect, tight tolerance -> CI should be narrow and
    clearly exclude zero. Not a p-hacked toy: uses a FIXED seed so the exact
    numbers are reproducible and can be hand-verified once, permanently.
    """
    rng = np.random.default_rng(0)
    control = rng.normal(10, 2, 1000)
    treatment = rng.normal(12, 2, 1000)

    result = raw_ttest_ci(control, treatment)

    assert result.point_estimate == pytest.approx(2.0, abs=0.3)
    assert result.ci_lower < result.point_estimate < result.ci_upper
    assert result.ci_upper < 0.5 + 2.0  # sanity bound, not a tight assertion
    assert result.p_value < 0.001
    assert result.degrees_freedom != 1998  # Welch's df must NOT equal pooled df exactly


def test_raw_ttest_null_effect_ci_contains_zero():
    """
    No true difference -> CI should contain 0 with high probability. This is
    a probabilistic test (not deterministic), so it checks CI COVERAGE
    behavior conceptually via a single well-seeded draw, not a guarantee —
    genuine frequentist coverage validation belongs in Phase 4's harness
    (1000+ repeated draws), not here.
    """
    rng = np.random.default_rng(1)
    control = rng.normal(10, 2, 500)
    treatment = rng.normal(10, 2, 500)  # IDENTICAL true mean

    result = raw_ttest_ci(control, treatment)
    assert result.ci_lower < 0 < result.ci_upper


def test_raw_ttest_unequal_variance_does_not_crash_and_uses_welch_df():
    """
    Deliberately unequal variances -> Welch's df should differ substantially
    from the naive pooled df (n1+n2-2), proving equal_var=False is actually
    taking effect, not silently defaulting to pooled behavior.
    """
    rng = np.random.default_rng(2)
    control = rng.normal(10, 1, 200)      # low variance
    treatment = rng.normal(10, 10, 200)   # high variance, same mean

    result = raw_ttest_ci(control, treatment)
    pooled_df = 200 + 200 - 2  # 398
    # Welch's df shrinks hard under variance imbalance
    assert result.degrees_freedom < pooled_df * 0.7


def test_raw_ttest_insufficient_observations_raises():
    with pytest.raises(ValueError):
        raw_ttest_ci([1.0], [1.0, 2.0])


def test_raw_ttest_zero_variance_both_groups_raises():
    with pytest.raises(ValueError):
        raw_ttest_ci([5.0, 5.0, 5.0], [5.0, 5.0, 5.0])


def test_cuped_theta_recovers_known_slope():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, 2000)
    y = 5 + 0.8 * x + rng.normal(0, 1, 2000)

    adj = cuped_adjust(y, x)
    assert adj.theta == pytest.approx(0.8, abs=0.05)


def test_cuped_adjustment_is_mean_preserving():
    """
    THE unbiasedness property, tested directly rather than trusted from the
    docstring's algebra: E[Y_cuped] should equal E[Y] up to sampling noise,
    for ANY theta (this is an algebraic identity, not dependent on theta
    being well-chosen) — if this fails, cuped_adjust has a real bug, since
    this must hold unconditionally.
    """
    rng = np.random.default_rng(4)
    x = rng.normal(0, 1, 5000)
    y = 5 + 0.8 * x + rng.normal(0, 1, 5000)

    adj = cuped_adjust(y, x)
    assert adj.y_adjusted.mean() == pytest.approx(y.mean(), abs=1e-9)


def test_cuped_zero_variance_covariate_raises():
    y = np.array([1.0, 2.0, 3.0])
    x = np.array([5.0, 5.0, 5.0])  # constant -> Var(x) = 0
    with pytest.raises(ValueError):
        cuped_adjust(y, x)


def test_variance_reduction_matches_correlation_squared_at_scale():
    """
    Cross-consistency check: the INDEPENDENTLY computed variance_reduction_pct
    should match the theoretical 100*corr(y,x)^2 prediction within sampling
    tolerance, for a LARGE n where noise is small. This is the test that
    would catch a theta-computation bug the direct formula alone might hide —
    exactly the scenario reasoned through by hand above (36.75% vs 39.0%
    theoretical, confirmed within ~1 SE).
    """
    rng = np.random.default_rng(5)
    n = 20000
    x = rng.normal(0, 1, n)
    y = 5 + 0.8 * x + rng.normal(0, 1, n)

    adj = cuped_adjust(y, x)
    vr = variance_reduction_pct(y, adj.y_adjusted)

    corr = np.corrcoef(y, x)[0, 1]
    theoretical_vr = 100 * corr**2

    assert vr == pytest.approx(theoretical_vr, abs=2.0)  # generous, n=20000 tightens noise


def test_variance_reduction_zero_raw_variance_raises():
    y = np.array([5.0, 5.0, 5.0])
    y_cuped = np.array([5.0, 5.0, 5.0])
    with pytest.raises(ValueError):
        variance_reduction_pct(y, y_cuped)
