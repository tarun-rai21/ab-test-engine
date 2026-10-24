"""
validation/test_multi_arm_fwer.py — proves the Holm and step-down Dunnett
adjustments control the family-wise error rate when several treatment arms
are compared against one control, and that NOT adjusting inflates it.

Global null: three treatment arms, none with any real effect. The FWER is
the share of simulated experiments in which at least one arm is declared
significant. Unadjusted, three looks at p < 0.05 give roughly 1 - 0.95^3
(~13%, a little less because the arms share a control group); both
adjustments must hold it at or below 5%, with Dunnett — which uses the
shared-control correlation — closer to 5% than the conservative Holm.

Uses the same Welch test and adjustment functions as core/pipeline.py, on
arrays generated directly (no database), since the claim is about the
multiple-testing procedure, not the data pipeline.

MARKED SLOW — release gate, not per-commit.
"""

import numpy as np
import pytest

from core.inference import raw_ttest_ci
from core.multiple_testing import dunnett_adjust, holm_adjust

N_SIMULATIONS = 2000
N_PER_ARM = 2000
N_ARMS = 3
BASELINE = 0.10


@pytest.mark.slow
def test_adjustments_control_family_wise_error_across_null_arms():
    rng = np.random.default_rng(2025)
    naive_hits = holm_hits = dunnett_hits = 0
    for _ in range(N_SIMULATIONS):
        control = rng.binomial(1, BASELINE, N_PER_ARM).astype(float)
        results = [
            raw_ttest_ci(control, rng.binomial(1, BASELINE, N_PER_ARM).astype(float))
            for _ in range(N_ARMS)
        ]
        p_values = [r.p_value for r in results]
        naive_hits += min(p_values) < 0.05
        holm_hits += min(holm_adjust(p_values)) < 0.05
        dunnett_hits += min(dunnett_adjust(
            [r.point_estimate for r in results],
            [r.standard_error for r in results],
            control.var(ddof=1) / len(control),
            raw_p_values=p_values,
        )) < 0.05

    naive_fwer = naive_hits / N_SIMULATIONS
    holm_fwer = holm_hits / N_SIMULATIONS
    dunnett_fwer = dunnett_hits / N_SIMULATIONS
    print(f"\nnaive FWER={naive_fwer:.4f}  holm FWER={holm_fwer:.4f}  "
          f"dunnett FWER={dunnett_fwer:.4f} ({N_ARMS} null arms, {N_SIMULATIONS} sims)")

    assert naive_fwer > 0.09  # inflation is real
    assert holm_fwer <= 0.065  # 5% plus ~3 SE of simulation noise
    assert dunnett_fwer <= 0.065
    assert dunnett_fwer >= holm_fwer  # same data: Dunnett rejects at least as often
