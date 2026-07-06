"""
core/validity.py — SRM detection and power/MDE analysis.

Both functions here are PURE: no database dependency, no simulator dependency.
They take plain counts/rates as arguments and return plain results. This is
deliberate — it means these functions are testable against known closed-form
inputs (NFR6) without any DB or simulator machinery in the loop, and it means
Phase 2's wiring into real data (core/data_access.py, next file) is a thin,
separately-testable layer on top of logic that's already proven correct in
isolation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.stats import chisquare, norm

SRM_FLAG_THRESHOLD = 0.001  # stricter than 0.05 — see srm_check docstring


@dataclass(frozen=True)
class SRMResult:
    observed_counts: tuple[int, ...]
    expected_counts: tuple[float, ...]
    chi_sq_stat: float
    p_value: float
    flagged: bool


def srm_check(
    observed_counts: list[int],
    expected_ratios: list[float],
    threshold: float = SRM_FLAG_THRESHOLD,
) -> SRMResult:
    """
    Chi-square goodness-of-fit test: are observed variant counts consistent
    with the intended allocation ratio?

    threshold=0.001, not the conventional 0.05: SRM is checked on EVERY
    analysis run, not once per study. At alpha=0.05, roughly 1 in 20
    perfectly-healthy experiments would be falsely flagged just from routine
    chance — intolerable at the frequency this check actually runs. This is
    documented, standard practice (Microsoft's ExP platform uses the same
    reasoning). It is a deliberate trade: raising the bar for "flag as broken"
    reduces false alarms at the cost of slightly reduced sensitivity to small
    real imbalances — acceptable here because SRM breaks in practice tend to
    be large (redirect bugs, bot filtering), not subtle 1% skews.

    Raises ValueError on malformed input rather than silently producing a
    meaningless statistic — e.g. counts and ratios of different lengths, or
    ratios that don't sum to ~1, are almost certainly caller bugs (wrong
    variant matched to wrong ratio), not valid inputs to quietly tolerate.
    """
    if len(observed_counts) != len(expected_ratios):
        raise ValueError(
            f"observed_counts (len={len(observed_counts)}) and expected_ratios "
            f"(len={len(expected_ratios)}) must have the same length."
        )
    if not np.isclose(sum(expected_ratios), 1.0, atol=1e-6):
        raise ValueError(f"expected_ratios must sum to 1.0, got {sum(expected_ratios)}")
    if any(c < 0 for c in observed_counts):
        raise ValueError(f"observed_counts must be non-negative, got {observed_counts}")

    total_n = sum(observed_counts)
    expected_counts = [total_n * r for r in expected_ratios]

    chi_sq_stat, p_value = chisquare(f_obs=observed_counts, f_exp=expected_counts)

    return SRMResult(
        observed_counts=tuple(observed_counts),
        expected_counts=tuple(expected_counts),
        chi_sq_stat=float(chi_sq_stat),
        p_value=float(p_value),
        flagged=bool(p_value < threshold),
    )


@dataclass(frozen=True)
class PowerResult:
    baseline_rate: float
    mde: float
    alpha: float
    power: float
    required_n_per_variant: int


def required_sample_size(
    baseline_rate: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> PowerResult:
    """
    Two-proportion z-test power formula (exact, closed-form — no root-finding
    needed here, unlike Phase 1's calibration, because n is being SOLVED FOR
    directly given mde, not the reverse):

        n = (z_(alpha/2) + z_beta)^2 * (p1(1-p1) + p2(1-p2)) / (p1 - p2)^2

    where p1 = baseline_rate, p2 = p1 + mde, z_(alpha/2) is the two-sided
    critical value for alpha, z_beta is the critical value for the target
    power. Uses scipy.stats.norm.ppf directly rather than hardcoding
    1.96/0.84 — transparent, and correct for any alpha/power the caller
    passes, not just the conventional 0.05/0.80 pair.

    NOTE: this uses the UNPOOLED variance (p1(1-p1) + p2(1-p2)), matching the
    project spec exactly. Some practical calculators use a POOLED
    approximation (2*p_bar*(1-p_bar)) assuming p1≈p2 for small effects — that
    approximation is slightly conservative (predicts marginally larger n) and
    is why mde_curve() below needs root-finding rather than a closed-form
    inverse: the exact formula has mde on both sides of the equation once you
    solve for mde given n instead of n given mde.
    """
    if not (0.0 < baseline_rate < 1.0):
        raise ValueError(f"baseline_rate must be in (0,1), got {baseline_rate}")
    if mde <= 0:
        raise ValueError(f"mde must be positive, got {mde}")
    p2 = baseline_rate + mde
    if not (0.0 < p2 < 1.0):
        raise ValueError(
            f"baseline_rate + mde = {p2} must be in (0,1) — mde too large for this baseline_rate."
        )
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0,1), got {alpha}")
    if not (0.0 < power < 1.0):
        raise ValueError(f"power must be in (0,1), got {power}")

    z_alpha_half = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)

    p1 = baseline_rate
    numerator = (z_alpha_half + z_beta) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2))
    denominator = (p1 - p2) ** 2
    n = numerator / denominator

    return PowerResult(
        baseline_rate=baseline_rate,
        mde=mde,
        alpha=alpha,
        power=power,
        required_n_per_variant=int(np.ceil(n)),
    )


def mde_curve(
    baseline_rate: float,
    n_values: list[int],
    alpha: float = 0.05,
    power: float = 0.80,
) -> list[float]:
    """
    Inverse of required_sample_size(): given a sample size, what's the
    smallest MDE this experiment could reliably detect?

    This is genuinely implicit — mde appears both in the numerator (via
    p2 = p1+mde affecting variance) and squared in the denominator — so unlike
    required_sample_size() above, there's no direct algebraic inverse. Solved
    via brentq per n value.

    Monotonicity justification (required for brentq's bisection to be valid):
    as mde increases, the effect becomes easier to detect, so required_n
    strictly decreases. Equivalently, required_n(mde) is a strictly decreasing
    function of mde over (0, 1-baseline_rate) — meaning it has a well-defined,
    unique inverse. This is a genuinely well-behaved monotonic relationship
    (unlike Phase 1's correlation calibration, which had asymmetric,
    occasionally-infeasible regions) — verified by construction of the
    formula itself, not merely assumed the way Phase 1's early bracket
    assumptions were.
    """
    if not (0.0 < baseline_rate < 1.0):
        raise ValueError(f"baseline_rate must be in (0,1), got {baseline_rate}")

    max_mde = (1.0 - baseline_rate) * 0.999  # p2 must stay < 1

    def n_gap(mde: float, target_n: int) -> float:
        return required_sample_size(baseline_rate, mde, alpha, power).required_n_per_variant - target_n

    mdes = []
    for target_n in n_values:
        if target_n <= 0:
            raise ValueError(f"n_values must all be positive, got {target_n}")
        # Bracket: mde just above 0 (huge required n) to max_mde (smallest n).
        lo, hi = 1e-6, max_mde
        try:
            mde = brentq(n_gap, lo, hi, args=(target_n,), xtol=1e-6)
        except ValueError as exc:
            raise ValueError(
                f"Could not solve for MDE at n={target_n}: even mde={max_mde:.4f} "
                f"(near-maximal) requires more/fewer samples than {target_n}. "
                f"This n may be too small for ANY detectable effect at this "
                f"baseline_rate, alpha, power combination."
            ) from exc
        mdes.append(mde)

    return mdes
