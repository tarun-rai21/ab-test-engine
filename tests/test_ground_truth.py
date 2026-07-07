"""
tests/test_ground_truth.py — unit tests for the GroundTruth dataclass in
isolation, independent of the simulator that produces it.

This was flagged as an explicitly deferred gap in Phase 1: GroundTruth's
immutability was assumed to hold because of frozen=True, and summary()'s
conditional branches (corrupted_split present/absent, segment info
present/absent) were never directly exercised. Deferred at the time as
low-risk since frozen=True is language-enforced, not custom logic — but
per this project's own standard, "low risk" is not the same as "verified,"
so this closes the gap rather than leaving it as an assumption.
"""

from dataclasses import FrozenInstanceError

import pytest

from data_sim.ground_truth import GroundTruth


def _make_ground_truth(**overrides) -> GroundTruth:
    """
    Minimal valid GroundTruth with sensible defaults, so each test only
    needs to override the field(s) actually relevant to it.
    """
    defaults = dict(
        n_users=1000,
        baseline_rate=0.10,
        true_effect_configured=0.02,
        covariate_correlation_target=0.5,
        seed=42,
        corrupted_split=None,
        segment_column=None,
        segment_effects_configured=None,
        calibrated_intercept=-2.2,
        calibrated_slope=0.8,
        baseline_rate_realized=0.1003,
        covariate_correlation_realized=0.4981,
        true_effect_realized=0.0197,
    )
    defaults.update(overrides)
    return GroundTruth(**defaults)


# --------------------------------------------------------------------- #
# Immutability — frozen=True is language-enforced, but must be PROVEN,
# not assumed, per this project's own standard established since Phase 1.
# --------------------------------------------------------------------- #

def test_ground_truth_is_immutable():
    """
    Attempting to mutate any field after construction must raise
    FrozenInstanceError. If this test starts passing without raising,
    frozen=True was removed or bypassed somehow — a real regression,
    since every later validation claim depends on GroundTruth being a
    tamper-proof record of what was actually configured/realized.
    """
    gt = _make_ground_truth()
    with pytest.raises(FrozenInstanceError):
        gt.true_effect_configured = 0.99


def test_ground_truth_equality_is_value_based():
    """
    Two GroundTruth instances with identical field values must compare
    equal (dataclass-generated __eq__), and differing on even one field
    must compare unequal. test_simulator.py's determinism test relies on
    this (g1 == g2 for identical seeds) — this test isolates that
    assumption instead of only exercising it indirectly.
    """
    gt1 = _make_ground_truth()
    gt2 = _make_ground_truth()
    assert gt1 == gt2

    gt3 = _make_ground_truth(seed=99)
    assert gt1 != gt3


# --------------------------------------------------------------------- #
# summary() — conditional branches, each exercised directly rather than
# only incidentally via whatever config a simulator test happens to use.
# --------------------------------------------------------------------- #

def test_summary_includes_core_fields_always():
    """
    n_users, baseline_rate (configured vs realized), true_effect
    (configured vs realized), and covariate_correlation must always
    appear — these are unconditional, not gated on optional fields.
    """
    gt = _make_ground_truth()
    summary = gt.summary()

    assert "n_users=1000" in summary
    assert "baseline_rate" in summary
    assert "true_effect" in summary
    assert "covariate_correlation" in summary


def test_summary_omits_corrupted_split_line_when_none():
    """
    corrupted_split=None (the healthy-experiment case) must NOT produce
    a 'corrupted_split' line — that line exists specifically to flag a
    deliberately broken randomization scenario, and its presence when
    nothing is actually corrupted would be misleading.
    """
    gt = _make_ground_truth(corrupted_split=None)
    assert "corrupted_split" not in gt.summary()


def test_summary_includes_corrupted_split_line_when_set():
    """
    corrupted_split=0.45 (a deliberately broken SRM scenario) must
    surface visibly in the summary — this is the line an analyst reading
    validation output depends on to know the data was intentionally
    mis-randomized for testing purposes.
    """
    gt = _make_ground_truth(corrupted_split=0.45)
    summary = gt.summary()
    assert "corrupted_split" in summary
    assert "0.45" in summary


def test_summary_omits_segment_line_when_no_heterogeneity_configured():
    """
    segment_effects_configured=None (the pooled/no-segment-heterogeneity
    case) must NOT produce a segment line.
    """
    gt = _make_ground_truth(segment_column=None, segment_effects_configured=None)
    assert "segment_column" not in gt.summary()


def test_summary_includes_segment_line_when_heterogeneity_configured():
    """
    When segment heterogeneity IS configured, the summary must surface
    the segment column name plus both configured and realized per-segment
    effects — this is the evidence a reviewer needs to confirm Phase 6's
    Simpson's-paradox scenario was actually set up as intended.
    """
    gt = _make_ground_truth(
        segment_column="device_type",
        segment_effects_configured={"mobile": 0.05, "desktop": 0.0},
        segment_effects_realized={"mobile": 0.0512, "desktop": 0.0021},
    )
    summary = gt.summary()
    assert "segment_column=device_type" in summary
    assert "segment_effects_configured" in summary
    assert "segment_effects_realized" in summary


# --------------------------------------------------------------------- #
# Defaults — the two fields with default values must actually default
# correctly when omitted, not just when explicitly passed.
# --------------------------------------------------------------------- #

def test_segment_effects_realized_defaults_to_none():
    gt = _make_ground_truth()
    assert gt.segment_effects_realized is None


def test_composition_defaults_to_additive():
    """
    Locks in the current single-supported composition mode. If a future
    phase adds a non-additive segment-heterogeneity mode, this test
    should be updated deliberately, not silently left passing by
    accident while the actual default drifts.
    """
    gt = _make_ground_truth()
    assert gt.composition == "additive"