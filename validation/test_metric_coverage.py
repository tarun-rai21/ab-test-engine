"""
validation/test_metric_coverage.py — proves the revenue metrics' 95%
confidence intervals contain the TRUE effect ~95% of the time, the same way
validation/test_ci_coverage.py does for conversion.

The simulator's order values are log-normal (sigma=1: heavily right-skewed),
which is exactly where a naive interval would be most at risk. True effects
are known in closed form because log-normal means are exact and
covariate_correlation=0 makes every user's conversion probability equal to
baseline_rate (control) or baseline_rate + true_effect (treatment):

  - revenue_per_conversion (delta method):  revenue_mean * revenue_lift
  - revenue_per_user (Welch):  (b + e) * m * (1 + lift) - b * m

MARKED SLOW — release gate, not per-commit.
"""

import pytest

from core.data_access import get_inference_data, split_by_variant
from core.metrics import estimate_effect
from db.connection import get_engine, reset_engine
from db.seed import seed_database

COVERAGE_LOWER_BOUND = 0.93
COVERAGE_UPPER_BOUND = 0.97
N_SIMULATIONS = 500

BASELINE, TRUE_EFFECT = 0.20, 0.02
REVENUE_MEAN, REVENUE_LIFT = 50.0, 0.10

TRUE_EFFECTS = {
    "revenue_per_conversion": REVENUE_MEAN * REVENUE_LIFT,
    "revenue_per_user": (BASELINE + TRUE_EFFECT) * REVENUE_MEAN * (1 + REVENUE_LIFT)
    - BASELINE * REVENUE_MEAN,
}


@pytest.mark.slow
@pytest.mark.parametrize("metric,base_seed", [
    ("revenue_per_conversion", 60000),
    ("revenue_per_user", 70000),
])
def test_revenue_metric_ci_coverage(metric, base_seed):
    true_effect = TRUE_EFFECTS[metric]
    contained = 0
    for i in range(N_SIMULATIONS):
        seed = base_seed + i
        reset_engine()
        seed_database({"simulator": {
            "n_users": 4000, "baseline_rate": BASELINE, "true_effect": TRUE_EFFECT,
            "covariate_correlation": 0.0, "seed": seed, "corrupted_split": None,
            "revenue_mean": REVENUE_MEAN, "revenue_lift": REVENUE_LIFT, "revenue_sigma": 1.0,
        }}, database_url="sqlite:///:memory:")

        control, treatment = split_by_variant(get_inference_data(get_engine(), f"exp_seed{seed}"))
        result = estimate_effect(control, treatment, metric)
        if result.ci_lower <= true_effect <= result.ci_upper:
            contained += 1

    coverage = contained / N_SIMULATIONS
    print(f"\n[{metric}] coverage={coverage:.4f} ({contained}/{N_SIMULATIONS}), "
          f"true_effect={true_effect:.3f}")
    assert COVERAGE_LOWER_BOUND <= coverage <= COVERAGE_UPPER_BOUND
