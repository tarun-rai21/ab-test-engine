"""
core/multiple_testing.py — family-wise error control for comparing several
treatment arms against one control.
"""

from __future__ import annotations


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
    the FDR that segment analysis uses) is the appropriate target. Dunnett's
    test would be slightly more powerful by exploiting the shared control
    group, at the cost of a multivariate-t computation.

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
