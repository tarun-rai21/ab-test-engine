"""
core/multiple_testing.py — family-wise error control for comparing several
treatment arms against one control: Holm (any dependence) and a step-down
Dunnett adjustment that exploits the shared control group.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import multivariate_normal, norm

ADJUSTMENTS = ("dunnett", "holm")


def holm_adjust(p_values: list[float]) -> list[float]:
    """
    Holm step-down adjusted p-values, in the same order as the input.

    With m p-values sorted ascending p_(1) <= ... <= p_(m), the adjusted
    value for rank i is max over j <= i of min(1, (m - j + 1) * p_(j)).
    Rejecting every hypothesis whose adjusted p-value is below alpha
    controls the family-wise error rate (the chance of ANY false positive
    among the arms) at alpha, under any dependence between the tests —
    and is uniformly more powerful than Bonferroni.

    Chosen for the many-arms-vs-control comparison because each arm is a
    separate shipping decision: a false "winner" is costly, so FWER (not
    the FDR that segment analysis uses) is the appropriate target.
    dunnett_adjust() is more powerful when the comparisons share a control.

    A single p-value is returned unchanged.
    """
    if not p_values:
        raise ValueError("p_values must be non-empty")
    if any(not (0.0 <= p <= 1.0) for p in p_values):
        raise ValueError(f"All p_values must be in [0,1], got {p_values}")

    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, index in enumerate(order):
        running_max = max(running_max, min(1.0, (m - rank) * p_values[index]))
        adjusted[index] = running_max
    return adjusted


def dunnett_adjust(
    estimates: list[float],
    standard_errors: list[float],
    control_variance: float,
    raw_p_values: list[float] | None = None,
) -> list[float]:
    """
    Step-down Dunnett-type adjusted p-values for k treatment arms each
    compared with ONE shared control, in input order.

    Every comparison reuses the control mean, so the estimates are
    positively correlated: Cov(D_i, D_j) = Var(control mean) =
    control_variance, giving Corr(Z_i, Z_j) = control_variance /
    (SE_i * SE_j). Holm ignores that correlation (Bonferroni at each
    step); using it gives exact family-wise error control and smaller
    adjusted p-values. Unlike scipy.stats.dunnett (pooled variance), this
    works from each comparison's own standard error, so it matches the
    Welch, CUPED and delta-method estimators used in this project.

    Step-down: order arms by |z| descending; for the r-th, the adjusted
    p-value is P(max over the arms not yet rejected of |Z_j| >= |z_r|)
    under the joint normal null, made monotone with a running maximum.
    Uses the large-sample normal approximation (the Welch df here are in
    the hundreds or more), floored at each arm's raw p-value when given.
    Multivariate-normal probabilities are computed
    with a fixed seed so results are reproducible.

    With a single arm there is nothing to adjust: raw_p_values (if given)
    are returned unchanged, so two-arm reports keep the t-test p-value.
    """
    k = len(estimates)
    if k == 0 or len(standard_errors) != k:
        raise ValueError("estimates and standard_errors must be non-empty and equal length")
    se = np.asarray(standard_errors, dtype=float)
    if np.any(se <= 0) or control_variance < 0:
        raise ValueError("standard errors must be positive and control_variance non-negative")
    z = np.abs(np.asarray(estimates, dtype=float) / se)
    if k == 1:
        return list(raw_p_values) if raw_p_values is not None else [float(2 * norm.sf(z[0]))]

    correlation = np.clip(control_variance / np.outer(se, se), 0.0, 1.0)
    np.fill_diagonal(correlation, 1.0)

    order = np.argsort(-z)
    adjusted = np.empty(k)
    running_max = 0.0
    for rank, index in enumerate(order):
        remaining = order[rank:]
        bound = np.full(len(remaining), z[index])
        if len(remaining) == 1:
            p_value = float(2 * norm.sf(z[index]))
        else:
            inside = multivariate_normal(
                mean=np.zeros(len(remaining)),
                cov=correlation[np.ix_(remaining, remaining)],
            ).cdf(bound, lower_limit=-bound, rng=np.random.default_rng(0))
            p_value = float(min(1.0, max(0.0, 1.0 - inside)))
        if raw_p_values is not None:
            # The normal approximation can land a hair below a t-test's own
            # p-value; an adjusted p-value must never be smaller than raw.
            p_value = max(p_value, raw_p_values[index])
        running_max = max(running_max, p_value)
        adjusted[index] = running_max
    return adjusted.tolist()
