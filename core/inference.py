"""
core/inference.py — raw two-sample inference and CUPED variance reduction.

DESIGN CONSTRAINT (Phase 2 architectural decision): this module is PURE and
has ZERO awareness of SRMResult or any 'trusted' concept. It does not know
whether SRM passed, does not accept an SRM result as an argument, and does
not tag its own output as trusted/untrusted. Trust-tagging happens exclusively
at persistence time (core/persistence.py, not yet built), which will require
both an InferenceResult and an SRMResult as separate explicit arguments before
writing to experiment_results. This preserves the spec's four-independent-
modules architecture (Section 4.1) — core/inference.py's correctness is
therefore completely testable without ever touching core/validity.py.

HARDENING NOTE (added while building core/pipeline.py's tests): all three
"is this degenerate" guards below now check for NaN explicitly, not just
exact zero. A NaN variance/standard-error (e.g. from an all-NULL covariate
column read back from SQL as NaN) previously slipped past `if x == 0:`
entirely, since NaN == 0 is False in Python/NumPy — the NaN then propagated
silently through cuped_adjust() -> variance_reduction_pct() -> raw_ttest_ci()
-> persist_inference_result(), surfacing only as a confusing SQLite
IntegrityError three function calls away from the actual cause. Caught by a
test that (correctly) exercised a NULL pre_period_covariate value; fixed
here at the source rather than only in the test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class InferenceResult:
    method: str  # "raw_ttest" | "cuped"
    point_estimate: float       # treatment_mean - control_mean
    ci_lower: float
    ci_upper: float
    alpha: float
    p_value: float
    standard_error: float
    degrees_freedom: float
    n_control: int
    n_treatment: int


def raw_ttest_ci(
    control: np.ndarray,
    treatment: np.ndarray,
    alpha: float = 0.05,
) -> InferenceResult:
    """
    Welch's two-sample t-test (unequal variances assumed) — NOT the pooled/
    Student's t-test. Chosen because treatment frequently changes outcome
    variance, not just the mean (e.g. a revenue-increasing feature often
    creates more high-value outliers, inflating treatment-group variance).
    Assuming equal variance when it doesn't hold inflates false-positive
    rate; Welch's costs a small amount of power when variances ARE equal, in
    exchange for validity when they aren't. This is a DEVIATION from scipy's
    own default (ttest_ind defaults to equal_var=True) — deliberate, not an
    oversight.

    Welch-Satterthwaite degrees of freedom (non-integer, unlike pooled df):

        df = (v1/n1 + v2/n2)^2 / [ (v1/n1)^2/(n1-1) + (v2/n2)^2/(n2-1) ]

    where v1, v2 are the sample variances. This approximates the true
    sampling distribution of the Welch t-statistic, which is not exactly a
    t-distribution with any fixed df — it's an approximation, not exact,
    but standard and well-validated practice.
    """
    control = np.asarray(control, dtype=float)
    treatment = np.asarray(treatment, dtype=float)

    n1, n2 = len(control), len(treatment)
    if n1 < 2 or n2 < 2:
        raise ValueError(f"Need at least 2 observations per group, got n1={n1}, n2={n2}")

    mean1, mean2 = control.mean(), treatment.mean()
    var1, var2 = control.var(ddof=1), treatment.var(ddof=1)  # ddof=1: sample variance

    diff = mean2 - mean1  # treatment - control: positive = treatment lifted the metric

    se_sq = var1 / n1 + var2 / n2
    se = np.sqrt(se_sq)

    if se == 0 or np.isnan(se):  # HARDENED: NaN now caught, not just exact zero
        raise ValueError(
            "Standard error is zero or NaN — both groups may have zero variance, "
            "or the input contains missing/NaN values. Cannot compute a "
            "meaningful CI; check for degenerate/constant/missing input."
        )

    df = se_sq**2 / ((var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1))

    t_stat, p_value = stats.ttest_ind(treatment, control, equal_var=False)
    t_crit = stats.t.ppf(1 - alpha / 2, df)

    ci_lower = diff - t_crit * se
    ci_upper = diff + t_crit * se

    return InferenceResult(
        method="raw_ttest",
        point_estimate=float(diff),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        alpha=alpha,
        p_value=float(p_value),
        standard_error=float(se),
        degrees_freedom=float(df),
        n_control=n1,
        n_treatment=n2,
    )


@dataclass(frozen=True)
class CupedAdjustment:
    y_adjusted: np.ndarray
    theta: float


def cuped_adjust(y: np.ndarray, x: np.ndarray) -> CupedAdjustment:
    """
    Y_cuped = Y - theta * (X - mean(X)),  theta = Cov(Y,X) / Var(X)

    theta is the OLS slope of Y on X — computed directly via the covariance/
    variance ratio rather than calling statsmodels.OLS, since for a single
    predictor these are mathematically identical and the direct formula
    avoids the overhead and API surface of a full OLS call for one number.

    UNBIASEDNESS: E[Y_cuped] = E[Y] - theta*E[X - mean(X)] = E[Y] - theta*0
    = E[Y]. The adjustment is mean-zero by construction (X - mean(X) has
    mean zero over the sample), so this does NOT change the expected value
    of the effect estimate — only its variance. This is NOT an assumption,
    it's an algebraic identity that holds for ANY theta, which is why CUPED
    cannot introduce bias regardless of how well-chosen theta is.

    CRITICAL PRECONDITION, not enforced by this function's signature (see
    caller responsibility below): X MUST be a pre-treatment covariate,
    measured before assignment. If X is post-treatment, it can be causally
    downstream of the treatment itself, and "adjusting for" it risks
    removing part of the actual treatment effect (a collider/post-treatment-
    bias problem) — this function has no way to verify that from the data
    alone, since a post-treatment covariate and a pre-treatment covariate
    look identical numerically. This must be enforced by the CALLER's data
    pipeline (core/data_access.py should only ever pull
    users.pre_period_covariate, which is generated before assignment in the
    simulator — verified in Phase 1 — never an events-table column).
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)

    if len(y) != len(x):
        raise ValueError(f"y and x must have equal length, got {len(y)} and {len(x)}")

    var_x = x.var(ddof=1)
    if var_x == 0 or np.isnan(var_x):  # HARDENED: NaN now caught, not just exact zero
        raise ValueError(
            "Covariate x has zero variance or contains NaN — cannot compute theta. "
            "Check for a constant covariate or missing pre_period_covariate values."
        )

    cov_xy = np.cov(y, x, ddof=1)[0, 1]
    theta = cov_xy / var_x

    y_adjusted = y - theta * (x - x.mean())

    return CupedAdjustment(y_adjusted=y_adjusted, theta=float(theta))


def variance_reduction_pct(y_raw: np.ndarray, y_cuped: np.ndarray) -> float:
    """
    Percent reduction in variance from CUPED adjustment:
        100 * (1 - Var(Y_cuped) / Var(Y_raw))

    At the OLS-optimal theta, this equals 100 * correlation(Y, X)^2 exactly
    (a separate, derivable identity — Var(Y_cuped) = Var(Y)*(1-rho^2) when
    theta is OLS-optimal). NOT re-derived from correlation here; computed
    directly from the two variances, which is the more honest measurement —
    it will match the rho^2 prediction only if theta was actually
    OLS-optimal, and computing it independently is what lets a test catch a
    theta-computation bug that the rho^2 shortcut would hide.
    """
    var_raw = np.asarray(y_raw, dtype=float).var(ddof=1)
    var_cuped = np.asarray(y_cuped, dtype=float).var(ddof=1)

    if var_raw == 0 or np.isnan(var_raw):  # HARDENED: NaN now caught, not just exact zero
        raise ValueError(
            "Raw variance is zero or NaN — variance reduction is undefined. "
            "Check for constant or missing input."
        )

    return float(100 * (1 - var_cuped / var_raw))
