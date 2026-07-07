"""
tests/test_validity.py — closed-form and structural tests for core/validity.py.
"""
import pytest

from core.validity import mde_curve, required_sample_size, srm_check


def test_srm_perfectly_balanced_not_flagged():
    r = srm_check([10000, 10000], [0.5, 0.5])
    assert r.chi_sq_stat == 0.0
    assert r.p_value == 1.0
    assert not r.flagged


def test_srm_known_closed_form_chi_square():
    """Hand-computed: chi_sq = 50 + 50 = 100 exactly."""
    r = srm_check([4500, 5500], [0.5, 0.5])
    assert r.chi_sq_stat == pytest.approx(100.0, abs=1e-6)
    assert r.flagged


def test_srm_borderline_between_005_and_0001_thresholds():
    """
    A split that WOULD be flagged at alpha=0.05 but should NOT be flagged
    at the project's actual threshold (0.001) — this is the test that
    actually validates WHY the stricter threshold matters, not just that
    the function runs. If this doesn't exist, nothing proves the 0.001
    vs 0.05 design decision has an observable effect.
    """
    # Engineer counts giving p-value between 0.001 and 0.05.
    # 5100/4900 split at n=10000 total:
    r = srm_check([5100, 4900], [0.5, 0.5])
    assert 0.001 < r.p_value < 0.05
    assert not r.flagged  # would be flagged under naive 0.05, but isn't here


def test_srm_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        srm_check([100, 100, 100], [0.5, 0.5])


def test_srm_ratios_not_summing_to_one_raises():
    with pytest.raises(ValueError):
        srm_check([100, 100], [0.5, 0.6])


def test_required_sample_size_matches_hand_derivation():
    """Cross-checked by hand: n=3839 for baseline=0.10, mde=0.02, alpha=0.05, power=0.80."""
    result = required_sample_size(baseline_rate=0.10, mde=0.02)
    assert result.required_n_per_variant == 3839


def test_mde_curve_is_monotonically_decreasing():
    mdes = mde_curve(baseline_rate=0.10, n_values=[1000, 5000, 20000, 100000])
    assert all(mdes[i] > mdes[i + 1] for i in range(len(mdes) - 1))


def test_mde_curve_consistent_with_required_sample_size():
    """
    Cross-check between the two functions: the n required to detect a given
    mde, fed back through mde_curve, should recover approximately that mde.
    This is the strongest test in the file — it validates that
    required_sample_size and mde_curve are actual inverses of each other,
    not two independently-plausible-looking but inconsistent functions.
    """
    target_mde = 0.02
    n = required_sample_size(baseline_rate=0.10, mde=target_mde).required_n_per_variant
    recovered_mde = mde_curve(baseline_rate=0.10, n_values=[n])[0]
    assert recovered_mde == pytest.approx(target_mde, rel=0.01)


def test_invalid_baseline_rate_raises():
    with pytest.raises(ValueError):
        required_sample_size(baseline_rate=1.5, mde=0.02)


def test_mde_exceeding_ceiling_raises():
    with pytest.raises(ValueError):
        required_sample_size(baseline_rate=0.95, mde=0.10)  # p2 would exceed 1

