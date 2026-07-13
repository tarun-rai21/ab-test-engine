"""
tests/test_segments.py — closed-form and structural tests for
core/segments.py. Fast, no database dependency — segment_breakdown() takes
a DataFrame directly, matching the pure-module pattern already established
for core/inference.py, core/validity.py, core/sequential.py.
"""

import numpy as np
import pandas as pd
import pytest

from core.segments import benjamini_hochberg, segment_breakdown, simpsons_paradox_flag

# --------------------------------------------------------------------- #
# benjamini_hochberg — closed-form checks
# --------------------------------------------------------------------- #


def test_benjamini_hochberg_known_textbook_example():
    """
    Hand-computed example, m=5, fdr=0.05:
      sorted p:      0.005   0.01   0.03   0.04   0.5
      rank:            1      2      3      4      5
      threshold:     0.01   0.02   0.03   0.04   0.05
      p <= threshold: yes    yes    yes    yes    no
    Largest rank satisfying the inequality is 4 -> the four smallest
    p-values are significant, the largest (0.5) is not.
    """
    p_values = [0.01, 0.04, 0.03, 0.005, 0.5]  # deliberately NOT pre-sorted
    result = benjamini_hochberg(p_values, fdr=0.05)
    assert result == [True, True, True, True, False]


def test_benjamini_hochberg_preserves_input_order_not_sorted_order():
    """
    THE detail most BH implementations get wrong: the returned list must
    align with the INPUT order, not the internally-sorted order. Uses a
    case where sorted order is the reverse of input order to make any
    order bug obvious rather than accidentally passing.
    """
    p_values = [0.5, 0.001]  # index 0 is the LARGE one, index 1 is the SMALL one
    result = benjamini_hochberg(p_values, fdr=0.05)
    assert result == [False, True]


def test_benjamini_hochberg_nothing_significant_when_all_p_values_large():
    p_values = [0.5, 0.6, 0.7]
    result = benjamini_hochberg(p_values, fdr=0.05)
    assert result == [False, False, False]


def test_benjamini_hochberg_everything_significant_when_all_p_values_tiny():
    p_values = [0.001, 0.002, 0.003]
    result = benjamini_hochberg(p_values, fdr=0.05)
    assert result == [True, True, True]


def test_benjamini_hochberg_single_pvalue_reduces_to_plain_alpha_comparison():
    """At m=1, rank/m*fdr == fdr exactly — BH degenerates to a plain
    p <= fdr check, the same as an uncorrected single test."""
    assert benjamini_hochberg([0.03], fdr=0.05) == [True]
    assert benjamini_hochberg([0.06], fdr=0.05) == [False]


def test_benjamini_hochberg_invalid_inputs_raise():
    with pytest.raises(ValueError):
        benjamini_hochberg([])
    with pytest.raises(ValueError):
        benjamini_hochberg([0.01, 0.02], fdr=1.5)
    with pytest.raises(ValueError):
        benjamini_hochberg([0.01, 1.5])  # p-value out of [0,1]


# --------------------------------------------------------------------- #
# simpsons_paradox_flag — closed-form checks
# --------------------------------------------------------------------- #


def test_simpsons_paradox_flag_detects_sign_flip():
    assert simpsons_paradox_flag(pooled_estimate=0.05, segment_estimate=-0.02) is True
    assert simpsons_paradox_flag(pooled_estimate=-0.05, segment_estimate=0.02) is True


def test_simpsons_paradox_flag_no_flag_when_signs_agree():
    assert simpsons_paradox_flag(pooled_estimate=0.05, segment_estimate=0.02) is False
    assert simpsons_paradox_flag(pooled_estimate=-0.05, segment_estimate=-0.02) is False


def test_simpsons_paradox_flag_zero_pooled_or_segment_never_flags():
    """
    Convention documented in the function itself: a comparison against an
    exact zero is not a meaningful sign-flip signal either direction.
    """
    assert simpsons_paradox_flag(pooled_estimate=0.0, segment_estimate=-0.02) is False
    assert simpsons_paradox_flag(pooled_estimate=0.05, segment_estimate=0.0) is False


def test_simpsons_paradox_flag_returns_native_bool_even_with_numpy_scalar_inputs():
    """
    Real gap this test closes: pandas .mean() returns numpy.float64, not a
    native Python float. (a > b) != (c > d) on numpy scalars produces
    numpy.bool_, and numpy.bool_(True) is NOT the same object as Python's
    True — an `is True` check downstream (a natural thing to write) would
    silently fail even though the VALUE is correct.
    """
    pooled = np.float64(0.1)
    segment = np.float64(-0.05)
    result = simpsons_paradox_flag(pooled_estimate=pooled, segment_estimate=segment)
    assert result is True  # identity check — this is the whole point of the test
    assert type(result) is bool  # not numpy.bool_


# --------------------------------------------------------------------- #
# segment_breakdown — hand-built DataFrame tests
# --------------------------------------------------------------------- #


def _make_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_segment_breakdown_known_answer():
    """
    Hand-built: two segments (mobile/desktop), each with a clean, distinct
    control/treatment conversion split. Verifies segment_breakdown's wiring
    (correct per-segment slicing, correct pooled-vs-segment comparison) —
    not raw_ttest_ci's own math, already proven correct in isolation.
    """
    rows = []
    # mobile: control 5/10 convert, treatment 8/10 convert -> positive effect
    for i in range(10):
        rows.append({
            "variant_id": "exp_control", "device_type": "mobile",
            "converted": 1 if i % 2 == 0 else 0,
        })
        rows.append({
            "variant_id": "exp_treatment", "device_type": "mobile",
            "converted": 1 if i < 8 else 0,
        })
    # desktop: control 5/10 convert, treatment 5/10 convert -> no effect
    for i in range(10):
        rows.append({
            "variant_id": "exp_control", "device_type": "desktop",
            "converted": 1 if i % 2 == 0 else 0,
        })
        rows.append({
            "variant_id": "exp_treatment", "device_type": "desktop",
            "converted": 1 if i % 2 == 0 else 0,
        })
    df = _make_df(rows)

    result = segment_breakdown(df, "device_type", pooled_point_estimate=0.15)

    assert result.segment_column == "device_type"
    assert len(result.segments) == 2
    assert result.excluded_segments == ()

    by_value = {s.segment_value: s for s in result.segments}
    assert by_value["mobile"].inference.point_estimate == pytest.approx(0.3, abs=1e-9)
    assert by_value["desktop"].inference.point_estimate == pytest.approx(0.0, abs=1e-9)


def test_segment_breakdown_excludes_insufficient_data_segment_not_fatal():
    """
    A segment with fewer than 2 observations in either arm must be
    EXCLUDED, not crash the whole analysis — matches the "normal edge
    case, not a fatal error" philosophy applied elsewhere in this project.
    """
    rows = []
    for i in range(10):
        rows.append({"variant_id": "exp_control", "device_type": "mobile",
                      "converted": 1 if i % 2 == 0 else 0})
        rows.append({"variant_id": "exp_treatment", "device_type": "mobile",
                      "converted": 1 if i < 8 else 0})
    # "tablet" segment: only ONE treatment user -> insufficient for raw_ttest_ci
    rows.append({"variant_id": "exp_control", "device_type": "tablet", "converted": 1})
    rows.append({"variant_id": "exp_control", "device_type": "tablet", "converted": 0})
    rows.append({"variant_id": "exp_treatment", "device_type": "tablet", "converted": 1})

    df = _make_df(rows)
    result = segment_breakdown(df, "device_type", pooled_point_estimate=0.15)

    assert len(result.segments) == 1
    assert result.segments[0].segment_value == "mobile"
    assert result.excluded_segments == ("tablet",)


def test_segment_breakdown_all_segments_excluded_raises():
    rows = [
        {"variant_id": "exp_control", "device_type": "tablet", "converted": 1},
        {"variant_id": "exp_treatment", "device_type": "tablet", "converted": 0},
    ]
    df = _make_df(rows)
    with pytest.raises(ValueError):
        segment_breakdown(df, "device_type", pooled_point_estimate=0.0)


def test_segment_breakdown_missing_column_raises():
    df = _make_df([{"variant_id": "exp_control", "converted": 1}])
    with pytest.raises(ValueError):
        segment_breakdown(df, "nonexistent_column", pooled_point_estimate=0.0)


def test_segment_breakdown_detects_simpsons_paradox():
    """
    Constructs a deliberate Simpson's-paradox scenario: the POOLED estimate
    is positive, but one segment's effect is clearly, strongly negative.
    simpsons_flag must be True for that segment, False for the one that
    agrees in sign with the pooled estimate.

    Segment SIZES are deliberately UNEQUAL (30 vs 10) — the classic
    real-world shape of Simpson's paradox, where a confounding variable
    (segment membership here) correlates with both the group split and the
    outcome. Equal-sized, exactly-opposite-magnitude segments would cancel
    to a pooled estimate of precisely 0.0, which this module deliberately
    declines to flag either way (see simpsons_paradox_flag's zero-value
    convention) — that would test the wrong thing here.
    """
    rows = []
    # "mobile": strong positive effect, LARGER segment (30 users/arm) ->
    # dominates the pooled estimate
    for i in range(30):
        rows.append({"variant_id": "exp_control", "device_type": "mobile",
                      "converted": 1 if i < 6 else 0})        # 6/30 = 0.20
        rows.append({"variant_id": "exp_treatment", "device_type": "mobile",
                      "converted": 1 if i < 24 else 0})       # 24/30 = 0.80
    # "desktop": strong negative effect, SMALLER segment (10 users/arm)
    for i in range(10):
        rows.append({"variant_id": "exp_control", "device_type": "desktop",
                      "converted": 1 if i < 8 else 0})        # 8/10 = 0.80
        rows.append({"variant_id": "exp_treatment", "device_type": "desktop",
                      "converted": 1 if i < 2 else 0})        # 2/10 = 0.20

    df = _make_df(rows)
    # Pooled estimate computed directly rather than assumed, to keep the
    # test's own premise honest and self-checking.
    pooled_control_rate = df[df["variant_id"] == "exp_control"]["converted"].mean()
    pooled_treatment_rate = df[df["variant_id"] == "exp_treatment"]["converted"].mean()
    pooled_estimate = pooled_treatment_rate - pooled_control_rate
    assert pooled_estimate != 0.0, (
        "Test construction produced an exactly-zero pooled estimate — "
        "adjust segment sizes/rates so this test actually exercises a "
        "nonzero-vs-nonzero sign comparison."
    )

    result = segment_breakdown(df, "device_type", pooled_point_estimate=pooled_estimate)
    by_value = {s.segment_value: s for s in result.segments}

    assert by_value["mobile"].inference.point_estimate > 0
    assert by_value["desktop"].inference.point_estimate < 0

    # Whichever segment disagrees in sign with the (computed) pooled
    # estimate must be flagged; the one that agrees must not be.
    if pooled_estimate > 0:
        assert by_value["desktop"].simpsons_flag is True
        assert by_value["mobile"].simpsons_flag is False
    else:
        assert by_value["mobile"].simpsons_flag is True
        assert by_value["desktop"].simpsons_flag is False