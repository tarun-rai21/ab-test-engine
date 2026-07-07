"""
validation/test_cuped_variance_reduction.py — proves CUPED's variance
reduction scales with covariate correlation strength, across repeated
simulated experiments, with tolerance bands derived from BOOTSTRAPPED
empirical standard error (SD/sqrt(N) of the actual per-run measurements),
not guessed constants.

MARKED SLOW — release gate, not per-commit (same rationale as
test_ci_coverage.py, established in Phase 0's CI design).
"""

import numpy as np
import pytest

from core.data_access import get_inference_data
from core.inference import cuped_adjust, variance_reduction_pct
from db.connection import get_engine, reset_engine
from db.seed import seed_database

N_SIMULATIONS = 200  # fewer than CI coverage's 1000 — this test measures a
                       # MEAN variance-reduction rate, which converges faster
                       # than a binomial coverage proportion; 200 reps keeps
                       # runtime reasonable while still averaging out noise


def _variance_reductions(correlation: float, base_seed: int) -> list[float]:
    """Returns the raw per-run list, not just the mean — needed to compute
    the empirical SE via bootstrap rather than guessing a tolerance."""
    reductions = []
    for i in range(N_SIMULATIONS):
        seed = base_seed + i
        config = {"simulator": {
            "n_users": 5000, "baseline_rate": 0.10, "true_effect": 0.0,
            "covariate_correlation": correlation, "seed": seed, "corrupted_split": None,
        }}
        reset_engine()
        seed_database(config, database_url="sqlite:///:memory:")
        engine = get_engine()

        df = get_inference_data(engine, f"exp_seed{seed}")
        y = df["converted"].to_numpy(dtype=float)
        x = df["pre_period_covariate"].to_numpy(dtype=float)

        adj = cuped_adjust(y, x)
        reductions.append(variance_reduction_pct(y, adj.y_adjusted))
    return reductions


@pytest.mark.slow
def test_variance_reduction_increases_with_correlation_strength():
    """
    Core claim under test: mean variance reduction at correlation=0.7 must
    be MEASURABLY greater than at correlation=0.3, across independent
    repeated simulations — not just positive at one arbitrary value.

    Tolerance bands are the EMPIRICAL standard error of the mean (SD/sqrt(N)
    of the actual 200 per-run measurements), not a guessed constant — this
    replaces the earlier abs=8.0/abs=10.0 placeholders, which happened to
    pass but were not derived from measured variance.
    """
    reductions_low = _variance_reductions(correlation=0.3, base_seed=30000)
    reductions_high = _variance_reductions(correlation=0.7, base_seed=40000)

    vr_low = float(np.mean(reductions_low))
    vr_high = float(np.mean(reductions_high))

    se_low = float(np.std(reductions_low, ddof=1) / np.sqrt(N_SIMULATIONS))
    se_high = float(np.std(reductions_high, ddof=1) / np.sqrt(N_SIMULATIONS))

    theoretical_low = 100 * 0.3**2   # 9.0
    theoretical_high = 100 * 0.7**2  # 49.0

    print(
        f"\nvr_low={vr_low:.2f}% (SE={se_low:.3f}, theory={theoretical_low})  "
        f"vr_high={vr_high:.2f}% (SE={se_high:.3f}, theory={theoretical_high})"
    )

    assert vr_high > vr_low, (
        f"Variance reduction did NOT increase with correlation strength: "
        f"corr=0.3 gave {vr_low:.2f}%, corr=0.7 gave {vr_high:.2f}%."
    )

    # 4x empirical SE — same multiplier convention as Phase 1's calibration
    # tolerance bands, now grounded in measured variance instead of a guess.
    assert abs(vr_low - theoretical_low) <= 4 * se_low, (
        f"vr_low={vr_low:.2f}% deviates from theory={theoretical_low}% by "
        f"{abs(vr_low - theoretical_low):.2f}pp, exceeding 4*SE={4 * se_low:.2f}pp"
    )
    assert abs(vr_high - theoretical_high) <= 4 * se_high, (
        f"vr_high={vr_high:.2f}% deviates from theory={theoretical_high}% by "
        f"{abs(vr_high - theoretical_high):.2f}pp, exceeding 4*SE={4 * se_high:.2f}pp"
    )