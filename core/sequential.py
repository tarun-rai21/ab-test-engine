"""
core/sequential.py — empirical false-positive-rate simulation under naive
repeated significance testing ("peeking"), an O'Brien-Fleming-style
alpha-spending correction (verified empirically against the same simulation,
not trusted from the closed-form formula alone), and a live-experiment check
that applies the correction to a real, in-progress experiment's stored
checkpoint history.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from core.inference import raw_ttest_ci


@dataclass(frozen=True)
class PeekingSimulationResult:
    n_simulations: int
    n_checkpoints: int
    threshold_schedule: tuple[float, ...]
    empirical_fpr: float
    checkpoint_trigger_counts: tuple[int, ...]


@dataclass(frozen=True)
class SequentialCheckResult:
    latest_checkpoint_n: int
    checkpoint_position: int      # e.g. 4, meaning "the 4th check out of the planned total"
    naive_p_value: float
    naive_significant: bool
    corrected_threshold: float
    corrected_significant: bool
    disagreement: bool             # True = naive says yes, corrected says no (the dangerous case)


def alpha_spending_schedule(n_checkpoints: int, total_alpha: float = 0.05) -> tuple[float, ...]:
    """
    O'Brien-Fleming-STYLE boundary via the standard closed-form approximation:

        z_k = z_(alpha/2) * sqrt(K/k),   alpha_k = 2*(1 - Phi(z_k))

    NOT the exact Lan-DeMets spending function (which requires numerical
    integration to guarantee cumulative type-I error sums exactly to
    total_alpha across all looks) — this is the standard, widely-used
    closed-form approximation. Its correction quality MUST be verified
    empirically (see simulate_peeking_fpr with this schedule substituted for
    a flat threshold), not trusted because the formula is textbook-standard.

    z_1 (first look) is very large -> alpha_1 near zero -> essentially
    impossible to trigger on early noise. z_K (final look) equals z_(alpha/2)
    exactly -> full nominal alpha at the pre-registered endpoint.
    """
    if n_checkpoints < 1:
        raise ValueError(f"n_checkpoints must be >= 1, got {n_checkpoints}")
    if not (0.0 < total_alpha < 1.0):
        raise ValueError(f"total_alpha must be in (0,1), got {total_alpha}")

    z_final = norm.ppf(1 - total_alpha / 2)
    K = n_checkpoints

    thresholds = []
    for k in range(1, K + 1):
        z_k = z_final * np.sqrt(K / k)
        alpha_k = 2 * (1 - norm.cdf(z_k))
        thresholds.append(alpha_k)

    return tuple(thresholds)


def simulate_peeking_fpr(
    n_simulations: int = 500,
    n_checkpoints: int = 10,
    checkpoint_n: int = 200,
    baseline_rate: float = 0.10,
    naive_alpha: float = 0.05,
    threshold_schedule: tuple[float, ...] | None = None,
    seed: int = 2024,
) -> PeekingSimulationResult:
    """
    Simulates n_simulations independent NULL-EFFECT experiments, checking at
    each of n_checkpoints cumulative sample sizes whether the checkpoint's
    p-value crosses its threshold.

    threshold_schedule=None -> NAIVE behavior: flat naive_alpha at every
    checkpoint (the original, uncorrected simulation).
    threshold_schedule=<tuple> -> CORRECTED behavior: per-checkpoint
    thresholds from alpha_spending_schedule(), typically stricter early,
    relaxing to naive_alpha at the final checkpoint.

    Both cases share this ONE loop deliberately — two separate
    implementations (one naive, one corrected) risk silently diverging over
    time, the same failure shape as this project's Phase 2 split_pct bug,
    where two conceptually-linked quantities lived in separate code paths.
    """
    if n_checkpoints < 1:
        raise ValueError(f"n_checkpoints must be >= 1, got {n_checkpoints}")
    if not (0.0 < baseline_rate < 1.0):
        raise ValueError(f"baseline_rate must be in (0,1), got {baseline_rate}")

    if threshold_schedule is None:
        threshold_schedule = tuple([naive_alpha] * n_checkpoints)
    elif len(threshold_schedule) != n_checkpoints:
        raise ValueError(
            f"threshold_schedule length ({len(threshold_schedule)}) must equal "
            f"n_checkpoints ({n_checkpoints})"
        )

    rng = np.random.default_rng(seed)
    total_n = n_checkpoints * checkpoint_n

    triggered_count = 0
    trigger_checkpoint_counts = [0] * n_checkpoints

    for sim_idx in range(n_simulations):
        control = rng.binomial(1, baseline_rate, total_n).astype(float)
        treatment = rng.binomial(1, baseline_rate, total_n).astype(float)

        triggered_this_sim = False
        for checkpoint_idx in range(n_checkpoints):
            cumulative_n = (checkpoint_idx + 1) * checkpoint_n
            control_slice = control[:cumulative_n]
            treatment_slice = treatment[:cumulative_n]

            threshold = threshold_schedule[checkpoint_idx]
            result = raw_ttest_ci(control_slice, treatment_slice, alpha=threshold)

            if result.p_value < threshold:
                triggered_this_sim = True
                trigger_checkpoint_counts[checkpoint_idx] += 1
                break

        if triggered_this_sim:
            triggered_count += 1

    empirical_fpr = triggered_count / n_simulations

    return PeekingSimulationResult(
        n_simulations=n_simulations,
        n_checkpoints=n_checkpoints,
        threshold_schedule=threshold_schedule,
        empirical_fpr=empirical_fpr,
        checkpoint_trigger_counts=tuple(trigger_checkpoint_counts),
    )


def sequential_check(
    checkpoints: list[dict],
    n_checkpoints_planned: int,
    checkpoint_n: int,
    naive_alpha: float = 0.05,
    total_alpha: float = 0.05,
) -> SequentialCheckResult:
    """
    Applies the alpha-spending schedule to a REAL, live experiment's stored
    checkpoint history, rather than a simulation. Flags disagreement between
    the naive (flat 5%) rule and the corrected (position-aware) rule at the
    LATEST checkpoint only — earlier checkpoints are historical record, not
    the current decision point.

    checkpoints: list of dicts, each representing one saved row from the
    sequential_checkpoints table, e.g.
        {"cumulative_n": 400, "p_value_at_check": 0.031}
    Must contain at least one entry. The LATEST checkpoint is identified as
    the one with the largest cumulative_n — not by list order, since a
    caller should never be trusted to have passed them in order.

    n_checkpoints_planned / checkpoint_n: the SAME values used when the
    experiment's schedule was originally planned (e.g. 10 checkpoints of
    200 users each) — needed to know which position in the alpha-spending
    schedule this checkpoint corresponds to.
    """
    if not checkpoints:
        raise ValueError("checkpoints list must contain at least one entry.")

    latest = max(checkpoints, key=lambda c: c["cumulative_n"])
    latest_n = latest["cumulative_n"]
    p_value = latest["p_value_at_check"]

    # Position in the schedule: cumulative_n=200 with checkpoint_n=200 -> position 1;
    # cumulative_n=800 -> position 4. Clamp to the planned range so a
    # checkpoint slightly off-grid doesn't index out of bounds.
    position = max(1, min(n_checkpoints_planned, round(latest_n / checkpoint_n)))

    schedule = alpha_spending_schedule(n_checkpoints_planned, total_alpha)
    corrected_threshold = schedule[position - 1]

    naive_significant = p_value < naive_alpha
    corrected_significant = p_value < corrected_threshold

    return SequentialCheckResult(
        latest_checkpoint_n=latest_n,
        checkpoint_position=position,
        naive_p_value=p_value,
        naive_significant=naive_significant,
        corrected_threshold=corrected_threshold,
        corrected_significant=corrected_significant,
        disagreement=(naive_significant and not corrected_significant),
    )