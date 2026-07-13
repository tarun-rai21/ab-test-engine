"""
validation/test_segment_recovery.py — proves the LAST of the six target
metrics originally listed in this project's spec Section 7.6 / Phase 4's
validation report: "Segment recovery (deliberately heterogeneous segment):
Recovered segment effect sign matches ground truth; magnitude within
simulation noise of the configured true segment effect. Wrong sign or
magnitude indicates a bug in the segment query or in how
segment_heterogeneity is applied in the simulator." This could not be
built until core/segments.py existed (Phase 6) — the other five metrics
(CI coverage, CUPED variance reduction x2, naive/corrected peeking FPR)
were already proven in Phases 4-5.

MARKED SLOW — release gate, not per-commit, same rationale as every other
file in this directory.

SCOPE NOTE: unlike CI coverage or peeking FPR, this is NOT a repeated-
simulation RATE measurement (there is no "recovery rate" being estimated
across many independent trials) — it is a single, large-n ground-truth
recovery check: does segment_breakdown() correctly recover the SIGN and
approximate MAGNITUDE of a deliberately configured heterogeneous segment
effect, exactly as the spec's own target metric describes. One
well-powered scenario (n=20000, matching the scenario already proven in
tests/test_pipeline.py's fast integration test) is the right scope here,
not N=500 repetitions of an already-deterministic-at-this-scale recovery.
"""

import pytest

from core.data_access import get_inference_data
from core.inference import raw_ttest_ci
from core.segments import segment_breakdown
from db.connection import get_engine, reset_engine
from db.seed import seed_database

# Same scenario already validated in tests/test_pipeline.py's fast
# integration test — repeated here at validation-harness scale/status
# because the spec explicitly lists this as one of the six release-gate
# target metrics, not because the underlying numbers are expected to
# differ from what the fast test already showed.
SEGMENT_EFFECTS_CONFIGURED = {"mobile": 0.15, "desktop": -0.08}
MAGNITUDE_TOLERANCE = 0.02  # absolute; generous given n=20000 shrinks noise heavily


@pytest.mark.slow
def test_segment_recovery_matches_configured_ground_truth():
    """
    Configures a KNOWN, deliberately heterogeneous per-segment effect via
    the simulator (mobile: +0.15, desktop: -0.08 — same construction as
    Phase 1's segment_heterogeneity support, unused for this purpose until
    now), then confirms segment_breakdown() recovers both the correct SIGN
    and the correct approximate MAGNITUDE for each segment — the ground-
    truth check the original spec's Section 7.6 target metrics table
    describes, closing the last of the six originally-planned validation
    checks.
    """
    config = {"simulator": {
        "n_users": 20000, "baseline_rate": 0.10, "true_effect": 0.0,
        "covariate_correlation": 0.0, "seed": 500, "corrupted_split": None,
        "segment_heterogeneity": {"device_type": SEGMENT_EFFECTS_CONFIGURED},
    }}
    reset_engine()
    seed_database(config, database_url="sqlite:///:memory:")
    engine = get_engine()

    df = get_inference_data(engine, "exp_seed500", segment_columns=["device_type"])

    pooled_result = raw_ttest_ci(
        df[df["variant_id"].str.endswith("_control")]["converted"].to_numpy(dtype=float),
        df[df["variant_id"].str.endswith("_treatment")]["converted"].to_numpy(dtype=float),
    )

    analysis = segment_breakdown(
        df, "device_type", pooled_point_estimate=pooled_result.point_estimate
    )
    by_value = {s.segment_value: s for s in analysis.segments}

    print(
        f"\npooled={pooled_result.point_estimate:.4f}  "
        f"mobile={by_value['mobile'].inference.point_estimate:.4f} "
        f"(configured={SEGMENT_EFFECTS_CONFIGURED['mobile']})  "
        f"desktop={by_value['desktop'].inference.point_estimate:.4f} "
        f"(configured={SEGMENT_EFFECTS_CONFIGURED['desktop']})"
    )

    for segment_value, configured_effect in SEGMENT_EFFECTS_CONFIGURED.items():
        recovered = by_value[segment_value].inference.point_estimate

        # SIGN check — the more critical failure mode. Wrong sign means the
        # segment query or the simulator's effect-application is broken,
        # not merely imprecise.
        assert (recovered > 0) == (configured_effect > 0), (
            f"Segment {segment_value!r}: recovered effect {recovered:.4f} has the "
            f"WRONG SIGN vs configured {configured_effect} — this indicates a bug "
            f"in the segment query (core/segments.py) or in how "
            f"segment_heterogeneity is applied in the simulator "
            f"(data_sim/simulator.py), per this exact failure-mode note in the "
            f"original spec's target metrics table."
        )

        # MAGNITUDE check — should be close given n=20000 shrinks sampling
        # noise heavily (per-segment n is roughly 12000/8000 given
        # DEVICE_PROBS=(0.6, 0.4), still large).
        assert abs(recovered - configured_effect) <= MAGNITUDE_TOLERANCE, (
            f"Segment {segment_value!r}: recovered effect {recovered:.4f} deviates "
            f"from configured {configured_effect} by "
            f"{abs(recovered - configured_effect):.4f}, exceeding the "
            f"{MAGNITUDE_TOLERANCE} tolerance."
        )

    # The Simpson's-paradox flag itself, as the spec's own Phase 6 expected
    # output describes: whichever segment disagrees in sign with the
    # pooled estimate must be flagged.
    pooled_sign_positive = pooled_result.point_estimate > 0
    disagreeing_segment = "desktop" if pooled_sign_positive else "mobile"
    agreeing_segment = "mobile" if pooled_sign_positive else "desktop"
    assert by_value[disagreeing_segment].simpsons_flag is True
    assert by_value[agreeing_segment].simpsons_flag is False
