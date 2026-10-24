"""
core/sequential.py — empirical false-positive-rate simulation under naive
repeated significance testing ("peeking"), two group-sequential corrections
(verified empirically against the same simulation, not trusted from the
formulas alone), and a live-experiment check that applies a correction to a
real, in-progress experiment's stored checkpoint history.

The two corrections:
  - alpha_spending_schedule(): classical O'Brien-Fleming for K EQUALLY
    spaced looks (index-based, calibrated constant).
  - lan_demets_boundary(): Lan-DeMets alpha spending with an
    O'Brien-Fleming-type spending function, driven by each look's actual
    INFORMATION FRACTION (cumulative_n / planned total). Valid for
    irregular, unplanned look times — the default for live checks.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.optimize import brentq
from scipy.signal import fftconvolve
from scipy.stats import norm
from scipy.stats import t as t_dist


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
    spending: str = "lan_demets"   # "lan_demets" (information-based) | "obrien_fleming" (index)
    information_fraction: float | None = None  # latest cumulative_n / planned total
    treatment_arm: str | None = None  # which arm-vs-control comparison was monitored
    metric_name: str | None = None
    replanned: bool = False  # True = some past looks were made under a different plan


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


def lan_demets_obf_spending(information_fraction: float, total_alpha: float = 0.05) -> float:
    """
    Cumulative two-sided alpha spent by information fraction t, using the
    Lan-DeMets O'Brien-Fleming-type spending function applied symmetrically
    (alpha/2 per side, the convention of R's ldbounds/gsDesign):

        alpha(t) = 4 * (1 - Phi(z_(alpha/4) / sqrt(t))),   0 < t <= 1

    alpha(1) = total_alpha exactly; alpha(t) is vanishingly small early,
    so — like O'Brien-Fleming — almost nothing is spent on early looks.
    t > 1 (looking past the planned sample size) is capped at 1: the
    budget is fully spent by then. Uses norm.sf, not 1 - cdf, so tiny
    early values don't underflow to exactly zero.
    """
    if information_fraction <= 0:
        raise ValueError(f"information_fraction must be > 0, got {information_fraction}")
    t = min(information_fraction, 1.0)
    return float(4.0 * norm.sf(norm.isf(total_alpha / 4) / np.sqrt(t)))


def lan_demets_boundary(
    information_fractions: Sequence[float],
    total_alpha: float = 0.05,
    grid_step: float = 0.001,
    fixed_thresholds: Sequence[float | None] | None = None,
) -> tuple[float, ...]:
    """
    Nominal two-sided p-value thresholds for looks taken at the given
    information fractions t_1 < t_2 < ... (t_k = cumulative_n_k / planned
    total n), such that the null probability of crossing at look k for the
    FIRST time equals alpha(t_k) - alpha(t_{k-1}) (lan_demets_obf_spending).
    The overall type-I error is therefore total_alpha whenever the final
    look reaches t >= 1, for ANY spacing of looks — unlike
    alpha_spending_schedule(), whose calibration assumes equal spacing.

    Each look's threshold depends only on the looks up to and including it,
    so thresholds for past looks never change when new looks are added.

    METHOD (Armitage-McPherson-Rowe recursion on the Brownian scale): under
    the null, W(t) = Z(t) * sqrt(t) has independent N(0, t_k - t_{k-1})
    increments. Starting from W(t_1) ~ N(0, t_1), repeatedly convolve the
    sub-density of paths that have NOT yet crossed with the next increment's
    normal density, then choose the boundary c_k so the mass beyond +/-c_k
    equals this look's alpha increment. z_k = c_k / sqrt(t_k).

    Verified against published ldbounds values for 5 equally spaced looks
    (z = 4.877, 3.357, 2.680, 2.290, 2.031) in tests/test_sequential.py.
    The first look at or past t = 1 is the final analysis: it spends
    whatever alpha remains (so it CAN reject). Any look after that has
    nothing left to spend and gets threshold 0.0 (it can never reject).

    fixed_thresholds (optional, one entry per look, None = compute): looks
    whose boundary was ALREADY USED under an earlier plan keep that exact
    recorded threshold. The recursion measures the alpha those looks
    actually spent and later looks spend alpha(t) minus that amount, with
    t measured against the CURRENT planned total — the standard Lan-DeMets
    way to change the planned sample size mid-test without rewriting
    history or exceeding total_alpha. If the plan grew so much that the
    spending function has not yet caught up with what was spent, the next
    looks get threshold 0 until it does. With no fixed looks the result is
    exactly as before.
    """
    fractions = [float(t) for t in information_fractions]
    if not fractions:
        raise ValueError("information_fractions must be non-empty")
    if fractions[0] <= 0 or any(b <= a for a, b in zip(fractions, fractions[1:])):
        raise ValueError(
            f"information_fractions must be positive and strictly increasing, got {fractions}"
        )
    if not (0.0 < total_alpha < 1.0):
        raise ValueError(f"total_alpha must be in (0,1), got {total_alpha}")
    fixed = list(fixed_thresholds) if fixed_thresholds is not None else [None] * len(fractions)
    if len(fixed) != len(fractions):
        raise ValueError("fixed_thresholds must have one entry per look")

    thresholds: list[float] = []
    spent_so_far = 0.0
    # Grid on the W scale, symmetric about 0: x = h * (-n..n).
    h = grid_step
    n = int(np.ceil(8 * np.sqrt(fractions[0]) / h))
    x = h * np.arange(-n, n + 1)
    density = norm.pdf(x, scale=np.sqrt(fractions[0]))
    previous_t = fractions[0]

    for k, t in enumerate(fractions):
        if k > 0:
            sd = np.sqrt(t - previous_t)
            half = int(np.ceil(8 * sd / h))
            kernel = norm.pdf(h * np.arange(-half, half + 1), scale=sd) * h
            density = np.clip(fftconvolve(density, kernel, mode="full"), 0.0, None)
            n += half
            x = h * np.arange(-n, n + 1)
            previous_t = t

        # Mass strictly inside |W| < x_j, accumulated outward from the centre.
        centre = n
        pair_mass = density[centre + 1:] + density[centre - 1::-1]
        inside = (density[centre] + np.concatenate([[0.0], np.cumsum(pair_mass)])) * h

        if fixed[k] is not None:
            # Boundary already used at this look: keep it, and book the
            # alpha it actually spent given the earlier looks.
            threshold = float(fixed[k])
            thresholds.append(threshold)
            if threshold <= 0.0:
                continue  # could never reject: spends nothing, cuts nothing
            c = norm.isf(threshold / 2) * np.sqrt(t)
            if k == 0:
                spent_so_far += threshold  # W(t_1) is exactly N(0, t_1)
            else:
                radius = min(int(c / h), len(inside) - 1)
                spent_so_far += float(max(0.0, inside[-1] - inside[radius]))
            density[np.abs(x) >= c] = 0.0
            continue

        target_spent = lan_demets_obf_spending(t, total_alpha)
        increment = target_spent - spent_so_far
        if increment <= 0.0:
            thresholds.append(0.0)  # nothing left to spend: this look cannot reject
            continue
        spent_so_far = target_spent

        if k == 0:
            # First look: W(t_1) ~ N(0, t_1), so the boundary is closed-form.
            z_1 = norm.isf(increment / 2)
            thresholds.append(float(increment))
            density[np.abs(x) >= z_1 * np.sqrt(t)] = 0.0
            continue

        target_inside = inside[-1] - increment
        j = int(np.searchsorted(inside, target_inside))
        if j >= len(inside):
            c = x[-1]
        elif j == 0:
            c = 0.0
        else:
            # Linear interpolation between grid radii (j-1)*h and j*h.
            frac = (target_inside - inside[j - 1]) / (inside[j] - inside[j - 1])
            c = (j - 1 + frac) * h

        z_k = c / np.sqrt(t)
        thresholds.append(float(2 * norm.sf(z_k)))
        density[np.abs(x) >= c] = 0.0

    return tuple(thresholds)


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

    # Draw every simulation's data first, in exactly the same order as the
    # original per-simulation loop (control then treatment, per sim), so a
    # given seed reproduces the same experiments.
    control = np.empty((n_simulations, total_n))
    treatment = np.empty((n_simulations, total_n))
    for sim_idx in range(n_simulations):
        control[sim_idx] = rng.binomial(1, baseline_rate, total_n)
        treatment[sim_idx] = rng.binomial(1, baseline_rate, total_n)

    p_values = _welch_p_values_at_checkpoints(control, treatment, cumulative_sizes)
    thresholds = np.asarray(threshold_schedule, dtype=float)

    # A simulation stops at its FIRST crossing; count where that happens.
    crossed = p_values < thresholds  # shape (n_simulations, n_checkpoints)
    triggered = crossed.any(axis=1)
    first_crossing = crossed.argmax(axis=1)
    triggered_count = int(triggered.sum())
    trigger_checkpoint_counts = np.bincount(
        first_crossing[triggered], minlength=n_checkpoints
    ).tolist()

    empirical_fpr = triggered_count / n_simulations

    return PeekingSimulationResult(
        n_simulations=n_simulations,
        n_checkpoints=n_checkpoints,
        threshold_schedule=threshold_schedule,
        empirical_fpr=empirical_fpr,
        checkpoint_trigger_counts=tuple(trigger_checkpoint_counts),
    )


def _welch_p_values_at_checkpoints(
    control: np.ndarray, treatment: np.ndarray, cumulative_sizes: list[int]
) -> np.ndarray:
    """
    Welch t-test p-values for every simulation (rows) at every checkpoint
    (columns), computed together from cumulative sums of 0/1 outcomes —
    the same statistic as core.inference.raw_ttest_ci() (equal n per arm at
    each checkpoint, sample variance with ddof=1, Welch-Satterthwaite df),
    without one scipy call per test. This made the 2000-simulation
    validation runs several times faster.

    For 0/1 data with s successes in n draws, the ddof=1 sample variance is
    (s - s^2/n) / (n - 1). A checkpoint where both arms are constant (zero
    standard error) gets p = 1: no evidence, never a trigger (raw_ttest_ci
    would raise instead).
    """
    n = np.asarray(cumulative_sizes, dtype=float)
    idx = np.asarray(cumulative_sizes) - 1
    s_c = np.cumsum(control, axis=1)[:, idx]
    s_t = np.cumsum(treatment, axis=1)[:, idx]

    var_c = (s_c - s_c**2 / n) / (n - 1)
    var_t = (s_t - s_t**2 / n) / (n - 1)
    se_sq = var_c / n + var_t / n

    with np.errstate(divide="ignore", invalid="ignore"):
        t_stat = (s_t / n - s_c / n) / np.sqrt(se_sq)
        df = se_sq**2 / ((var_c / n) ** 2 / (n - 1) + (var_t / n) ** 2 / (n - 1))
        p = 2 * t_dist.sf(np.abs(t_stat), df)
    return np.where(se_sq > 0, p, 1.0)


def sequential_check(
    checkpoints: list[dict],
    n_checkpoints_planned: int,
    checkpoint_n: int,
    naive_alpha: float = 0.05,
    total_alpha: float = 0.05,
    spending: str = "lan_demets",
) -> SequentialCheckResult:
    """
    Applies an alpha-spending correction to a REAL, live experiment's stored
    checkpoint history, rather than a simulation. Flags disagreement between
    the naive (flat 5%) rule and the corrected rule at the LATEST checkpoint
    only — earlier checkpoints are historical record, not the current
    decision point.

    spending:
      - "lan_demets" (default): information-based. Every stored look's
        information fraction t = cumulative_n / (n_checkpoints_planned *
        checkpoint_n) feeds lan_demets_boundary(), so irregular or
        unplanned look times are handled correctly. If the plan changed
        mid-test, looks recorded under the old plan (planned_total_n on the
        row) keep the threshold that was actually used.
      - "obrien_fleming": the classical index-based schedule from
        alpha_spending_schedule(), mapping the latest look to its nearest
        planned position. Exact only when looks follow the planned equal
        spacing; kept for comparison.

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
    planned_total_n = n_checkpoints_planned * checkpoint_n
    information_fraction = latest_n / planned_total_n

    replanned = False
    if spending == "lan_demets":
        # One look per distinct sample size (the latest-logged row wins).
        by_n = {c["cumulative_n"]: c for c in checkpoints}
        look_ns = sorted(by_n)
        # Looks whose threshold was set under a DIFFERENT planned total keep
        # that recorded threshold (it was the rule actually in force); the
        # rest are computed under the current plan. See lan_demets_boundary.
        fixed: list[float | None] = []
        for n in look_ns[:-1]:
            row = by_n[n]
            recorded_plan = row.get("planned_total_n")
            if recorded_plan is not None and recorded_plan != planned_total_n:
                fixed.append(row["alpha_threshold_at_check"])
                replanned = True
            else:
                fixed.append(None)
        thresholds = lan_demets_boundary(
            [n / planned_total_n for n in look_ns], total_alpha,
            fixed_thresholds=fixed + [None],
        )
        corrected_threshold = thresholds[-1]
        # Information-based: any look past the planned total is beyond plan.
        beyond_plan = latest_n > planned_total_n
    elif spending == "obrien_fleming":
        schedule = alpha_spending_schedule(n_checkpoints_planned, total_alpha)
        corrected_threshold = schedule[position - 1]
    else:
        raise ValueError(
            f"spending must be 'lan_demets' or 'obrien_fleming', got {spending!r}"
        )

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
        spending=spending,
        information_fraction=information_fraction,
        replanned=replanned,
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
