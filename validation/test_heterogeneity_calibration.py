"""
validation/test_segment_heterogeneity.py — calibration and power of the
Cochran's Q heterogeneity test (core.segments.heterogeneity_test) as used by
segment_breakdown().

  - Calibration: with the SAME true effect in every region, the test should
    reject at about its nominal 5%.
  - Power: with a configured device-type difference, it should reject
    almost always.

Data comes straight from ExperimentSimulator (no database) to keep 1000
repetitions fast; the claim is about the test, not the SQL layer.

MARKED SLOW — release gate, not per-commit.
"""

import pandas as pd
import pytest

from core.segments import segment_breakdown
from data_sim.simulator import ExperimentSimulator


def _inference_frame(sim: ExperimentSimulator) -> pd.DataFrame:
    users, assignments, events, _ = sim.generate()
    df = users.merge(assignments[["user_id", "variant_id"]], on="user_id")
    df["converted"] = df["user_id"].isin(events["user_id"]).astype(int)
    return df


def _rejection_rate(segment_column: str, heterogeneity, n_sims: int, base_seed: int) -> float:
    rejections = 0
    for i in range(n_sims):
        sim = ExperimentSimulator(
            n_users=4000, baseline_rate=0.20, true_effect=0.02, covariate_correlation=0.0,
            seed=base_seed + i, segment_heterogeneity=heterogeneity,
        )
        analysis = segment_breakdown(_inference_frame(sim), segment_column, 0.02)
        rejections += analysis.heterogeneity_p_value < 0.05
    return rejections / n_sims


@pytest.mark.slow
def test_heterogeneity_test_is_calibrated_under_a_homogeneous_effect():
    rate = _rejection_rate("region", None, n_sims=1000, base_seed=80000)
    print(f"\n[homogeneous] Q-test rejection rate={rate:.4f}")
    assert 0.035 <= rate <= 0.07


@pytest.mark.slow
def test_heterogeneity_test_detects_a_real_segment_difference():
    """
    Effects: mobile 0.02+0.04, desktop 0.02-0.04, a difference of 0.08.
    With n=4000 (60% mobile), baseline 0.20 and 50/50 arms, the segment
    SEs are ~sqrt(2*0.16/1200)=0.0163 and ~sqrt(2*0.16/800)=0.0200, so the
    difference has SE ~0.0258 and theoretical power is
    Phi(0.08/0.0258 - 1.96) ~= 0.87. The measured rate must match that,
    within simulation noise (SE ~2.4pp at 200 sims).
    """
    rate = _rejection_rate(
        "device_type", {"device_type": {"mobile": 0.04, "desktop": -0.04}},
        n_sims=200, base_seed=90000,
    )
    print(f"\n[heterogeneous] Q-test rejection rate={rate:.4f} (theory ~0.87)")
    assert rate == pytest.approx(0.87, abs=0.07)
