"""
validation/test_ci_coverage.py — proves, rather than asserts, that
raw_ttest_ci()'s 95% CI actually contains the true effect ~95% of the time
across repeated simulated experiments.

MARKED SLOW: excluded from the default CI run (pytest -m "not slow").
Run explicitly via scripts/run_validation_suite.sh before any release/
portfolio-freeze. This is a release gate, not a per-commit check — see
Phase 0's CI design rationale.

This is fundamentally different from every prior test in this project:
previous tests checked ONE number against ONE target. This checks a RATE
(coverage) across MANY independent repetitions, because "95% CI" is a claim
about long-run procedure behavior, not about any single interval.
"""

import numpy as np
import pytest

from core.data_access import get_inference_data, split_by_variant
from core.inference import raw_ttest_ci
from data_sim.simulator import ExperimentSimulator
from db.connection import get_engine, init_schema, reset_engine
from db.seed import seed_database

N_SIMULATIONS = 1000
NOMINAL_COVERAGE = 0.95
# Project spec's target band: 93%-97%. Wider than the pure binomial 2*SE
# band (~[0.936, 0.964]) to tolerate additional slack from Welch's df being
# an asymptotic approximation, not exact, at moderate sample sizes.
COVERAGE_LOWER_BOUND = 0.93
COVERAGE_UPPER_BOUND = 0.97


@pytest.mark.slow
def test_ci_coverage_at_nominal_rate():
    """
    Runs N_SIMULATIONS=1000 independent simulated experiments, each with a
    KNOWN true_effect, computes raw_ttest_ci() for each, and checks what
    fraction of the resulting 95% CIs actually contain the true effect.

    Each simulation uses a DIFFERENT seed (base_seed + i) — independence
    across repetitions is what makes the coverage RATE meaningful; reusing
    one seed 1000 times would just measure the same draw 1000 times.
    """
    true_effect = 0.02
    baseline_rate = 0.10
    n_users_per_sim = 2000  # moderate n: large enough for Welch's df approx to
                             # be reasonable, small enough that 1000 sims run
                             # in a tractable time (NFR3: target <2 minutes)

    contained_count = 0
    base_seed = 10000  # offset from other seeds used elsewhere in the project,
                         # to avoid any accidental collision with seeds used
                         # in tests/ during a shared test session

    for i in range(N_SIMULATIONS):
        seed = base_seed + i
        config = {"simulator": {
            "n_users": n_users_per_sim, "baseline_rate": baseline_rate,
            "true_effect": true_effect, "covariate_correlation": 0.0,
            "seed": seed, "corrupted_split": None,
        }}
        reset_engine()
        seed_database(config, database_url="sqlite:///:memory:")
        engine = get_engine()

        df = get_inference_data(engine, f"exp_seed{seed}")
        control_df, treatment_df = split_by_variant(df)

        result = raw_ttest_ci(
            control_df["converted"].to_numpy(), treatment_df["converted"].to_numpy()
        )

        # THE actual check: does this CI contain the TRUE effect (known,
        # because we built the simulator and configured true_effect above)?
        if result.ci_lower <= true_effect <= result.ci_upper:
            contained_count += 1

    observed_coverage = contained_count / N_SIMULATIONS
    print(f"\nobserved_coverage={observed_coverage:.4f} ({contained_count}/{N_SIMULATIONS})")
    assert COVERAGE_LOWER_BOUND <= observed_coverage <= COVERAGE_UPPER_BOUND, (
        f"Observed CI coverage {observed_coverage:.4f} across {N_SIMULATIONS} "
        f"simulated experiments falls outside the target band "
        f"[{COVERAGE_LOWER_BOUND}, {COVERAGE_UPPER_BOUND}]. Per the project's "
        f"validation harness target metrics: outside this band indicates a "
        f"bug in the CI formula or estimator bias, NOT sampling noise — the "
        f"band already accounts for the binomial noise in the coverage "
        f"estimate itself plus Welch's df approximation slack."
    )