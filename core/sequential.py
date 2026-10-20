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
from functools import lru_cache

import numpy as np
from scipy.optimize import brentq
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
    beyond_plan: bool = False      # True = latest look is past the planned final sample size


def _obf_crossing_probability(c: float, n_checkpoints: int, grid_step: float = 0.005) -> float:
    """
    Exact (up to grid discretization) null probability that a two-sided
    O'Brien-Fleming boundary z_k = c*sqrt(K/k) is crossed at ANY of K
    equally spaced looks.

    With equal increments, the cumulative sum S_k = sqrt(k)*Z_k is a
    Gaussian random walk with N(0,1) steps, and the OBF boundary on that
    scale is FLAT: |S_k| >= c*sqrt(K). So the no-crossing probability is
    the mass of a random walk that stays inside (-c*sqrt(K), c*sqrt(K)) for
    K steps — computed by repeatedly convolving the in-strip density with
    the N(0,1) step density (Armitage-McPherson-Rowe recursion), using a
    midpoint grid.
    """
    bound = c * np.sqrt(n_checkpoints)
    n_grid = int(np.ceil(2 * bound / grid_step))
    h = 2 * bound / n_grid
    x = -bound + h * (np.arange(n_grid) + 0.5)
    density = norm.pdf(x)
    step_kernel = norm.pdf(np.arange(-n_grid + 1, n_grid) * h) * h
    for _ in range(n_checkpoints - 1):
        density = np.convolve(density, step_kernel, mode="valid")
    return float(1.0 - density.sum() * h)


@lru_cache(maxsize=64)
def alpha_spending_schedule(n_checkpoints: int, total_alpha: float = 0.05) -> tuple[float, ...]:
    """
    O'Brien-Fleming group-sequential boundary for K equally spaced looks:

        z_k = c * sqrt(K/k),   alpha_k = 2*(1 - Phi(z_k))

    where the constant c is SOLVED FOR so that the probability of crossing
    the boundary at any look under the null equals total_alpha exactly
    (see _obf_crossing_probability). For K=10, alpha=0.05: c = 2.0865,
    matching published O'Brien-Fleming tables (2.087).

    BUG FIX (review issue H2): this previously used c = z_(alpha/2) = 1.96,
    i.e. the full nominal alpha at the final look ON TOP OF the alpha spent
    at earlier looks. Its exact type-I error at K=10 is ~6.9%, not 5% — the
    "6.6%" validation result was systematic overshoot, not noise. With a
    calibrated c the final look's nominal alpha is below total_alpha
    (~0.0369 at K=10), which is the price of having looked early.

    z_1 (first look) is very large -> alpha_1 near zero -> essentially
    impossible to trigger on early noise.

    Cached: the result depends only on (n_checkpoints, total_alpha), and
    the root-find costs ~0.2s for K=10.
    """
    if n_checkpoints < 1:
        raise ValueError(f"n_checkpoints must be >= 1, got {n_checkpoints}")
    if not (0.0 < total_alpha < 1.0):
        raise ValueError(f"total_alpha must be in (0,1), got {total_alpha}")

    K = n_checkpoints
    z_nominal = norm.ppf(1 - total_alpha / 2)
    if K == 1:
        c = z_nominal  # one look: no correction needed, exact closed form
    else:
        c = brentq(
            lambda c: _obf_crossing_probability(c, K) - total_alpha,
            z_nominal, z_nominal + 3.0, xtol=1e-6,
        )

    return tuple(float(2 * (1 - norm.cdf(c * np.sqrt(K / k)))) for k in range(1, K + 1))


def simulate_peeking_fpr(
    n_simulations: int = 500,
    n_checkpoints: int = 10,
    checkpoint_n: int = 200,
    baseline_rate: float = 0.10,
    naive_alpha: float = 0.05,
    threshold_schedule: tuple[float, ...] | None = None,
    seed: int = 2024,
    checkpoint_ns: list[int] | None = None,
) -> PeekingSimulationResult:
    """
    Simulates n_simulations independent NULL-EFFECT experiments, checking at
    each of n_checkpoints cumulative sample sizes whether the checkpoint's
    p-value crosses its threshold.

    threshold_schedule=None -> NAIVE behavior: flat naive_alpha at every
    checkpoint (the original, uncorrected simulation).
    threshold_schedule=<tuple> -> CORRECTED behavior: per-checkpoint
    thresholds from alpha_spending_schedule(), typically stricter early,
    relaxing toward (but, once calibrated, staying below) naive_alpha at
    the final checkpoint.

    UNITS: checkpoint_n and checkpoint_ns count users PER ARM (each arm is
    sliced to that length). This differs from sequential_check() and the
    sequential_checkpoints table, whose cumulative_n is the TOTAL across
    both arms — so checkpoint_n=200 here corresponds to checkpoint_n=400
    there.

    Both cases share this ONE loop deliberately — two separate
    implementations (one naive, one corrected) risk silently diverging over
    time, the same failure shape as this project's Phase 2 split_pct bug,
    where two conceptually-linked quantities lived in separate code paths.

    checkpoint_ns: optional list of EXPLICIT cumulative sample sizes for
    each checkpoint (e.g. [120, 300, 550, 700, 1000, 1250, 1500, 1650, 1800,
    2000] for irregular, real-world-style spacing — daily checks with
    uneven traffic), overriding the default UNIFORM spacing derived from
    checkpoint_n * (index+1). Must be strictly increasing with length
    n_checkpoints. When omitted, behavior is byte-for-byte identical to
    every prior call site (fully backward-compatible) — this closes the
    "checkpoints are evenly spaced... irregular real-world checkpoint
    timing is not validated" limitation flagged since Phase 5, without
    changing default behavior for anything already calling this function.

    NOTE ON THE SCHEDULE ITSELF: alpha_spending_schedule()'s z_k =
    z_final*sqrt(K/k) formula is INDEX-based (checkpoint COUNT k out of K),
    not information-based (actual cumulative_n ratio). The classical
    O'Brien-Fleming design technically uses the information-accrual ratio,
    which equals the look-count ratio only under EQUAL spacing. Under
    irregular spacing (checkpoint_ns provided), this function still applies
    the same index-based schedule — it does not attempt an information-
    based redesign. This measures whether that existing approximation still
    controls FPR adequately under irregular spacing, not whether a more
    sophisticated schedule would do better (out of scope here, and
    documented as such in validation/test_peeking_inflation.py).
    """
    if n_checkpoints < 1:
        raise ValueError(f"n_checkpoints must be >= 1, got {n_checkpoints}")
    if not (0.0 < baseline_rate < 1.0):
        raise ValueError(f"baseline_rate must be in (0,1), got {baseline_rate}")

    if checkpoint_ns is not None:
        if len(checkpoint_ns) != n_checkpoints:
            raise ValueError(
                f"checkpoint_ns length ({len(checkpoint_ns)}) must equal "
                f"n_checkpoints ({n_checkpoints})"
            )
        if list(checkpoint_ns) != sorted(set(checkpoint_ns)):
            raise ValueError(
                f"checkpoint_ns must be strictly increasing with no duplicates, "
                f"got {checkpoint_ns}"
            )
        cumulative_sizes = list(checkpoint_ns)
        total_n = cumulative_sizes[-1]
    else:
        cumulative_sizes = [(k + 1) * checkpoint_n for k in range(n_checkpoints)]
        total_n = n_checkpoints * checkpoint_n

    if threshold_schedule is None:
        threshold_schedule = tuple([naive_alpha] * n_checkpoints)
    elif len(threshold_schedule) != n_checkpoints:
        raise ValueError(
            f"threshold_schedule length ({len(threshold_schedule)}) must equal "
            f"n_checkpoints ({n_checkpoints})"
        )

    rng = np.random.default_rng(seed)

    triggered_count = 0
    trigger_checkpoint_counts = [0] * n_checkpoints

    for sim_idx in range(n_simulations):
        control = rng.binomial(1, baseline_rate, total_n).astype(float)
        treatment = rng.binomial(1, baseline_rate, total_n).astype(float)

        triggered_this_sim = False
        for checkpoint_idx in range(n_checkpoints):
            cumulative_n = cumulative_sizes[checkpoint_idx]
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
    400 users each) — needed to know which position in the alpha-spending
    schedule this checkpoint corresponds to.

    UNITS: cumulative_n and checkpoint_n are TOTAL users across both arms
    (matching the sequential_checkpoints table). See checkpoint_position().
    """
    if not checkpoints:
        raise ValueError("checkpoints list must contain at least one entry.")

    latest = max(checkpoints, key=lambda c: c["cumulative_n"])
    latest_n = latest["cumulative_n"]
    p_value = latest["p_value_at_check"]

    position, beyond_plan = checkpoint_position(latest_n, n_checkpoints_planned, checkpoint_n)

    schedule = alpha_spending_schedule(n_checkpoints_planned, total_alpha)
    corrected_threshold = schedule[position - 1]

    naive_significant = bool(p_value < naive_alpha)
    corrected_significant = bool(p_value < corrected_threshold)

    return SequentialCheckResult(
        latest_checkpoint_n=latest_n,
        checkpoint_position=position,
        naive_p_value=p_value,
        naive_significant=naive_significant,
        corrected_threshold=corrected_threshold,
        corrected_significant=corrected_significant,
        disagreement=(naive_significant and not corrected_significant),
        beyond_plan=beyond_plan,
    )


def checkpoint_position(
    cumulative_n: int, n_checkpoints_planned: int, checkpoint_n: int
) -> tuple[int, bool]:
    """
    Maps a checkpoint's cumulative sample size onto its 1-based position in
    the planned schedule: cumulative_n=400 with checkpoint_n=400 -> 1;
    cumulative_n=1600 -> 4. Both arguments are TOTAL users across arms.

    Returns (position, beyond_plan). The position is clamped to
    [1, n_checkpoints_planned] so an off-grid checkpoint never indexes out
    of bounds — but a look past the planned final sample size is reported
    via beyond_plan=True instead of being silently clamped (review issue
    M1): the schedule spends all of alpha by the planned final look, so a
    look beyond it is outside what the correction was designed for.
    """
    if n_checkpoints_planned < 1 or checkpoint_n < 1:
        raise ValueError(
            f"n_checkpoints_planned and checkpoint_n must be >= 1, got "
            f"{n_checkpoints_planned} and {checkpoint_n}"
        )
    raw_position = round(cumulative_n / checkpoint_n)
    position = max(1, min(n_checkpoints_planned, raw_position))
    return position, raw_position > n_checkpoints_planned
