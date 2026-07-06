import numpy as np
import pytest

from data_sim.simulator import ExperimentSimulator


def test_determinism():
    """
    Same seed, same inputs -> byte-identical output across all DataFrames and
    GroundTruth. Foundation for NFR2 — Phase 4's validation harness is
    meaningless if a single seed doesn't reproduce itself.
    """
    sim1 = ExperimentSimulator(n_users=5000, baseline_rate=0.1, true_effect=0.02, seed=42)
    sim2 = ExperimentSimulator(n_users=5000, baseline_rate=0.1, true_effect=0.02, seed=42)

    u1, a1, e1, g1 = sim1.generate()
    u2, a2, e2, g2 = sim2.generate()

    assert u1.equals(u2), "users_df differs across identical seeds"
    assert a1.equals(a2), "assignments_df differs across identical seeds"
    assert e1.equals(e2), "events_df differs across identical seeds"
    assert g1 == g2, "GroundTruth differs across identical seeds"


@pytest.mark.parametrize(
    "n_users,baseline_rate,target_corr",
    [
        (5000, 0.10, 0.5),
        (5000, 0.10, 0.3),
        (5000, 0.30, 0.5),
        (5000, 0.10, -0.25),   # NOTE: was -0.4, found empirically unreachable at
                                 # baseline_rate=0.10 given this covariate's skew —
                                 # see ExperimentSimulator module docstring. This
                                 # value is a first attempt at a reachable negative
                                 # target, NOT independently pre-verified — if this
                                 # ALSO fails or raises, reduce magnitude further
                                 # (e.g. -0.15) rather than re-fighting the solver.
    ],
)
def test_covariate_correlation_within_sampling_noise(n_users, baseline_rate, target_corr):
    """
    Encodes the SE-based tolerance band, not an arbitrary threshold.
    SE(r) ≈ sqrt((1 - r^2) / (n - 2)). 4*SE pass band: generous enough to avoid
    flaking on ordinary resampling variance, tight enough to catch structural bias.
    """
    sim = ExperimentSimulator(
        n_users=n_users, baseline_rate=baseline_rate, true_effect=0.0, covariate_correlation=target_corr, seed=7
    )
    _, _, _, gt = sim.generate()

    se = np.sqrt((1 - target_corr**2) / (n_users - 2))
    tolerance = 4 * se

    gap = abs(gt.covariate_correlation_realized - target_corr)
    assert gap <= tolerance, (
        f"Realized correlation {gt.covariate_correlation_realized:.4f} deviates from "
        f"target {target_corr:.4f} by {gap:.4f}, exceeding 4*SE tolerance ({tolerance:.4f}). "
        f"This gap-to-noise ratio ({gap/se:.1f} SE) suggests a STRUCTURAL bug, not noise."
    )


def test_negative_correlation_infeasible_target_raises_cleanly():
    """
    Documents and locks in the KNOWN LIMITATION rather than leaving it as an
    undiscovered landmine: target_corr=-0.4 at baseline_rate=0.10 is unreachable
    for this covariate's skew, and the joint calibration correctly raises
    (ValueError or RuntimeError, depending on which guard fires first) instead
    of silently returning a miscalibrated result.

    If this test starts FAILING (i.e. -0.4 stops raising and starts returning a
    value), that means either the solver's behavior changed or the covariate
    distribution changed — investigate, don't just delete this test.
    """
    sim = ExperimentSimulator(
        n_users=5000, baseline_rate=0.10, true_effect=0.0, covariate_correlation=-0.4, seed=7
    )
    with pytest.raises((ValueError, RuntimeError)):
        sim.generate()


def test_zero_correlation_is_exact_shortcut():
    sim = ExperimentSimulator(n_users=2000, baseline_rate=0.1, true_effect=0.0, covariate_correlation=0.0, seed=1)
    _, _, _, gt = sim.generate()
    assert gt.calibrated_slope == 0.0


def test_corrupted_split_produces_expected_imbalance():
    sim = ExperimentSimulator(
        n_users=20000, baseline_rate=0.1, true_effect=0.0, corrupted_split=0.45, seed=3
    )
    _, assignments_df, _, _ = sim.generate()
    treatment_share = (assignments_df["variant_id"].str.endswith("_treatment")).mean()
    assert abs(treatment_share - 0.45) < 0.02


def test_baseline_rate_within_sampling_noise():
    n_users = 10000
    baseline_rate = 0.10
    sim = ExperimentSimulator(
        n_users=n_users, baseline_rate=baseline_rate, true_effect=0.0,
        covariate_correlation=0.5, seed=42,
    )
    _, _, _, gt = sim.generate()

    se = np.sqrt(baseline_rate * (1 - baseline_rate) / n_users)
    tolerance = 4 * se
    gap = abs(gt.baseline_rate_realized - baseline_rate)
    assert gap <= tolerance, (
        f"Realized baseline rate {gt.baseline_rate_realized:.4f} vs configured "
        f"{baseline_rate:.4f}, gap={gap:.4f} exceeds 4*SE={tolerance:.4f} ({gap/se:.1f} SE)."
    )