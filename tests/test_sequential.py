"""
tests/test_sequential.py — closed-form and structural tests for
core/sequential.py.

Fast, unit-scale only. The full 500-simulation naive-vs-corrected FPR
comparison (the 22.0% -> 6.6% result) is NOT re-run here — that belongs in
validation/test_peeking_inflation.py, marked @pytest.mark.slow, since it
takes real wall-clock time and is a release gate, not a per-commit check
(same reasoning as Phase 4's validation/ split).
"""

import numpy as np
import pytest
from scipy.stats import norm

from core.sequential import (
    alpha_spending_schedule,
    checkpoint_position,
    sequential_check,
    simulate_peeking_fpr,
)

# --------------------------------------------------------------------- #
# alpha_spending_schedule — closed-form checks
# --------------------------------------------------------------------- #

def test_schedule_matches_published_obrien_fleming_tables():
    """
    Calibrated O'Brien-Fleming constants for two-sided alpha=0.05 are
    tabulated (Jennison & Turnbull): K=2 -> c=1.977, K=5 -> c=2.040,
    K=10 -> c=2.087. The nominal per-look alphas are checked against values
    computed from those published constants.

    Review issue H2: the old schedule used c=1.96, and this test's
    predecessor asserted schedule[-1] == 0.05 exactly — locking in a
    boundary whose true type-I error was ~6.9%.
    """
    k2 = alpha_spending_schedule(n_checkpoints=2, total_alpha=0.05)
    assert k2[0] == pytest.approx(0.0052, abs=2e-4)
    assert k2[1] == pytest.approx(0.0480, abs=3e-4)

    k5 = alpha_spending_schedule(n_checkpoints=5, total_alpha=0.05)
    assert k5[-1] == pytest.approx(2 * (1 - norm.cdf(2.040)), rel=0.01)

    k10 = alpha_spending_schedule(n_checkpoints=10, total_alpha=0.05)
    assert k10[-1] == pytest.approx(2 * (1 - norm.cdf(2.087)), rel=0.01)
    assert k10[-1] < 0.05  # the price of having looked early


def test_schedule_controls_overall_type_one_error_at_total_alpha():
    """
    Independent Monte Carlo check (does not reuse the integration routine):
    simulate null z-statistic paths at 10 equally spaced looks and count how
    often ANY look crosses its threshold. Must be ~5%, not the ~6.9% the old
    c=1.96 boundary produced. 200k paths -> SE ~0.05pp.
    """
    K = 10
    schedule = alpha_spending_schedule(n_checkpoints=K, total_alpha=0.05)
    z_bounds = norm.ppf(1 - np.array(schedule) / 2)

    rng = np.random.default_rng(0)
    paths = np.cumsum(rng.standard_normal((200_000, K)), axis=1)
    z = paths / np.sqrt(np.arange(1, K + 1))
    crossed = (np.abs(z) >= z_bounds).any(axis=1).mean()

    assert crossed == pytest.approx(0.05, abs=0.002)


def test_schedule_first_checkpoint_is_vanishingly_small():
    """
    At k=1, z_1 = c*sqrt(K) — for K=10 (c=2.0865) that is ~6.6, giving
    alpha_1 on the order of 1e-11.
    """
    schedule = alpha_spending_schedule(n_checkpoints=10, total_alpha=0.05)
    assert schedule[0] == pytest.approx(2 * (1 - norm.cdf(2.0865 * np.sqrt(10))), rel=0.05)
    assert schedule[0] < 1e-6  # generously loose bound: must be MUCH smaller than naive 0.05


def test_schedule_is_strictly_increasing():
    """
    sqrt(K/k) strictly decreases as k increases -> z_k strictly decreases ->
    alpha_k strictly increases. Must hold for ANY n_checkpoints, not just 10.
    """
    schedule = alpha_spending_schedule(n_checkpoints=10, total_alpha=0.05)
    assert all(schedule[i] < schedule[i + 1] for i in range(len(schedule) - 1))


def test_schedule_length_matches_n_checkpoints():
    schedule = alpha_spending_schedule(n_checkpoints=7, total_alpha=0.05)
    assert len(schedule) == 7


def test_schedule_single_checkpoint_equals_total_alpha():
    """
    Degenerate case: K=1 means there's only one look, so alpha_1 should
    equal total_alpha exactly (sqrt(1/1)=1, no correction needed at all).
    """
    schedule = alpha_spending_schedule(n_checkpoints=1, total_alpha=0.05)
    assert schedule[0] == pytest.approx(0.05, abs=1e-9)


def test_schedule_invalid_inputs_raise():
    with pytest.raises(ValueError):
        alpha_spending_schedule(n_checkpoints=0)
    with pytest.raises(ValueError):
        alpha_spending_schedule(n_checkpoints=5, total_alpha=1.5)


# --------------------------------------------------------------------- #
# simulate_peeking_fpr — small-scale wiring test, NOT the full validation run
# --------------------------------------------------------------------- #

def test_simulate_peeking_fpr_naive_wiring():
    """
    Small n_simulations (20, not 500) — this is a FAST wiring check
    (does the function run, return correct shapes, produce a sensible
    range), not a precision measurement. The precise 22.0% figure is
    validated separately and slowly in validation/test_peeking_inflation.py.
    """
    result = simulate_peeking_fpr(n_simulations=20, n_checkpoints=3, checkpoint_n=50, seed=1)

    assert result.n_simulations == 20
    assert result.n_checkpoints == 3
    assert len(result.threshold_schedule) == 3
    assert result.threshold_schedule == (0.05, 0.05, 0.05)  # naive: flat schedule
    assert 0.0 <= result.empirical_fpr <= 1.0
    assert len(result.checkpoint_trigger_counts) == 3
    assert sum(result.checkpoint_trigger_counts) <= 20  # can't exceed total simulations


def test_simulate_peeking_fpr_corrected_uses_provided_schedule():
    schedule = alpha_spending_schedule(n_checkpoints=3, total_alpha=0.05)
    result = simulate_peeking_fpr(
        n_simulations=20, n_checkpoints=3, checkpoint_n=50,
        threshold_schedule=schedule, seed=1,
    )
    assert result.threshold_schedule == schedule


def test_simulate_peeking_fpr_mismatched_schedule_length_raises():
    with pytest.raises(ValueError):
        simulate_peeking_fpr(
            n_simulations=10, n_checkpoints=5, checkpoint_n=50,
            threshold_schedule=(0.05, 0.05),  # wrong length
        )


def test_simulate_peeking_fpr_determinism():
    """Same seed -> identical result, same reproducibility standard as every
    other stochastic function in this project since Phase 1."""
    r1 = simulate_peeking_fpr(n_simulations=20, n_checkpoints=3, checkpoint_n=50, seed=7)
    r2 = simulate_peeking_fpr(n_simulations=20, n_checkpoints=3, checkpoint_n=50, seed=7)
    assert r1.empirical_fpr == r2.empirical_fpr
    assert r1.checkpoint_trigger_counts == r2.checkpoint_trigger_counts


def test_simulate_peeking_fpr_checkpoint_ns_matches_equivalent_uniform_case():
    """
    checkpoint_ns=[200, 400, 600] describes EXACTLY the same cumulative
    sample sizes as the default uniform checkpoint_n=200, n_checkpoints=3
    case — the two calling styles must produce byte-identical results at
    the same seed, proving checkpoint_ns doesn't silently change anything
    when it happens to describe a uniform schedule.
    """
    uniform = simulate_peeking_fpr(n_simulations=20, n_checkpoints=3, checkpoint_n=200, seed=11)
    explicit = simulate_peeking_fpr(
        n_simulations=20, n_checkpoints=3, checkpoint_n=200,
        checkpoint_ns=[200, 400, 600], seed=11,
    )
    assert uniform.empirical_fpr == explicit.empirical_fpr
    assert uniform.checkpoint_trigger_counts == explicit.checkpoint_trigger_counts


def test_simulate_peeking_fpr_checkpoint_ns_wrong_length_raises():
    with pytest.raises(ValueError):
        simulate_peeking_fpr(
            n_simulations=10, n_checkpoints=5, checkpoint_ns=[100, 200, 300],
        )


def test_simulate_peeking_fpr_checkpoint_ns_not_strictly_increasing_raises():
    with pytest.raises(ValueError):
        simulate_peeking_fpr(
            n_simulations=10, n_checkpoints=3, checkpoint_ns=[200, 200, 600],
        )
    with pytest.raises(ValueError):
        simulate_peeking_fpr(
            n_simulations=10, n_checkpoints=3, checkpoint_ns=[400, 200, 600],  # out of order
        )

# --------------------------------------------------------------------- #
# sequential_check — structural tests
# --------------------------------------------------------------------- #

def test_sequential_check_detects_disagreement():
    """
    Reproduces the exact hand-verified scenario: checkpoint 4 of 10,
    naive p=0.03 says significant, corrected threshold (~0.00097) says no.
    """
    checkpoints = [
        {"cumulative_n": 200, "p_value_at_check": 0.40},
        {"cumulative_n": 400, "p_value_at_check": 0.15},
        {"cumulative_n": 600, "p_value_at_check": 0.06},
        {"cumulative_n": 800, "p_value_at_check": 0.03},
    ]
    result = sequential_check(checkpoints, n_checkpoints_planned=10, checkpoint_n=200)

    assert result.checkpoint_position == 4
    assert result.naive_significant is True
    # corrected_threshold comes from alpha_spending_schedule(), which uses
    # scipy's norm.ppf internally -> corrected_significant is a numpy.bool_,
    # not a plain Python bool. Truthiness ('not x' / 'if x') is safe here
    # since numpy.bool_ implements __bool__ correctly; only IDENTITY checks
    # (is True / is False) would be unsafe, since numpy.bool_(False) is not
    # the same object as Python's False.
    assert not result.corrected_significant
    assert result.disagreement is True


def test_sequential_check_no_disagreement_when_genuinely_significant():
    """
    At the FINAL checkpoint (position 10), the corrected threshold is ~0.0369
    (below the naive 0.05 after calibration) -> a p-value clearly
    below both should show naive and corrected agreeing, no disagreement.
    """
    checkpoints = [{"cumulative_n": 2000, "p_value_at_check": 0.001}]
    result = sequential_check(checkpoints, n_checkpoints_planned=10, checkpoint_n=200)

    assert result.checkpoint_position == 10
    assert result.naive_significant is True
    assert result.corrected_significant  # see truthiness-vs-identity note above
    assert result.disagreement is False


def test_sequential_check_uses_latest_checkpoint_regardless_of_list_order():
    """
    Checkpoints passed OUT OF ORDER must still correctly identify the
    LATEST (largest cumulative_n) one — not just take the last list item.
    """
    checkpoints = [
        {"cumulative_n": 800, "p_value_at_check": 0.03},
        {"cumulative_n": 200, "p_value_at_check": 0.40},  # out of order on purpose
        {"cumulative_n": 400, "p_value_at_check": 0.15},
    ]
    result = sequential_check(checkpoints, n_checkpoints_planned=10, checkpoint_n=200)
    assert result.latest_checkpoint_n == 800
    assert result.checkpoint_position == 4


def test_sequential_check_empty_checkpoints_raises():
    with pytest.raises(ValueError):
        sequential_check([], n_checkpoints_planned=10, checkpoint_n=200)


def test_sequential_check_flags_look_beyond_planned_schedule():
    """
    Review issue M1: a look past the planned final sample size used to be
    silently clamped to the final position. It still uses the final
    threshold (clamped) but must now report beyond_plan=True.
    """
    checkpoints = [{"cumulative_n": 4000, "p_value_at_check": 0.02}]
    result = sequential_check(checkpoints, n_checkpoints_planned=10, checkpoint_n=200)
    assert result.checkpoint_position == 10
    assert result.beyond_plan is True

    on_plan = sequential_check(
        [{"cumulative_n": 2000, "p_value_at_check": 0.02}],
        n_checkpoints_planned=10, checkpoint_n=200,
    )
    assert on_plan.beyond_plan is False


def test_checkpoint_position_maps_total_n_onto_schedule():
    assert checkpoint_position(400, 10, 400) == (1, False)
    assert checkpoint_position(1600, 10, 400) == (4, False)
    assert checkpoint_position(50, 10, 400) == (1, False)   # below first look: clamp up
    assert checkpoint_position(8000, 10, 400) == (10, True)
    with pytest.raises(ValueError):
        checkpoint_position(400, 10, 0)
