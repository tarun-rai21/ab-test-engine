"""
validation/test_peeking_inflation.py — commits the naive-vs-corrected
peeking false-positive-rate comparison as permanent, reproducible evidence.

Naive and corrected runs share seed=2024 (common random numbers), so any
difference is attributable ONLY to the threshold schedule, not different
random draws.

REVISED (review issue H2): the original version ran 500 simulations and
accepted a corrected FPR of 6.6% against a band widened to 7.5%. With 500
simulations the SE is ~1.1pp, too coarse to distinguish 5% from 7% — and
the old schedule's exact type-I error was in fact ~6.9%. The schedule now
uses a calibrated O'Brien-Fleming constant, simulations are raised to 2000
(SE ~0.5pp), and the corrected band is tightened to [3.5%, 7%].

MARKED SLOW — release gate, not per-commit (same rationale as Phase 4's
validation/ files: 2000 simulations x 10 checkpoints = 20000 t-test calls,
real wall-clock cost, not appropriate for routine CI).
"""

import pytest

from core.sequential import alpha_spending_schedule, lan_demets_boundary, simulate_peeking_fpr

# Naive band: the exact null crossing probability of a flat |z| >= 1.96 at
# 10 equally spaced looks is ~19.3% (Armitage-McPherson-Rowe), so the
# spec's original 20% floor sat ABOVE theory — 2000 sims measured 19.95%.
# The band is centred on theory instead; any value here still means
# roughly 4x inflation over nominal.
NAIVE_FPR_LOWER_BOUND = 0.15
NAIVE_FPR_UPPER_BOUND = 0.30
# Nominal 5% +/- ~3 SE at 2000 sims, with slight allowance for the t-test on
# binary data not being exactly the Brownian-motion z-statistic the
# boundary is calibrated for.
CORRECTED_FPR_LOWER_BOUND = 0.035
CORRECTED_FPR_UPPER_BOUND = 0.07

N_SIMULATIONS = 2000

SEED = 2024  # SAME seed for naive and corrected runs — common random numbers,
              # so any FPR difference is attributable ONLY to the threshold
              # schedule, not to different underlying data.


@pytest.mark.slow
def test_naive_peeking_inflates_fpr_above_nominal():
    """
    Reproduces the exact interactively-measured result: with a FLAT 0.05
    threshold at every one of 10 checkpoints, the empirical false-positive
    rate on TRUE-NULL data should land materially above the nominal 5%,
    near the ~19% theoretical value for 10 equally spaced looks.
    """
    result = simulate_peeking_fpr(
        n_simulations=N_SIMULATIONS, n_checkpoints=10, checkpoint_n=200, seed=SEED
    )

    print(
        f"\nnaive_fpr={result.empirical_fpr:.4f} "
        f"({int(round(result.empirical_fpr * N_SIMULATIONS))}/{N_SIMULATIONS}), "
        f"trigger_counts={result.checkpoint_trigger_counts}"
    )

    assert NAIVE_FPR_LOWER_BOUND <= result.empirical_fpr <= NAIVE_FPR_UPPER_BOUND, (
        f"Naive empirical FPR {result.empirical_fpr:.4f} falls outside the "
        f"expected inflation band [{NAIVE_FPR_LOWER_BOUND}, {NAIVE_FPR_UPPER_BOUND}]. "
        f"This is the core claim of Phase 5 — if this fails, either the "
        f"simulation's null-data generation or the checkpoint logic has "
        f"regressed."
    )


@pytest.mark.slow
def test_alpha_spending_correction_restores_fpr_near_nominal():
    """
    Reproduces the exact interactively-measured correction: applying the
    O'Brien-Fleming-style schedule to the SAME underlying null data (same
    seed) should bring the empirical FPR back down to the nominal 5%, within
    sampling noise.
    """
    schedule = alpha_spending_schedule(n_checkpoints=10, total_alpha=0.05)

    result = simulate_peeking_fpr(
        n_simulations=N_SIMULATIONS, n_checkpoints=10, checkpoint_n=200,
        threshold_schedule=schedule, seed=SEED,
    )

    print(
        f"\ncorrected_fpr={result.empirical_fpr:.4f} "
        f"({int(round(result.empirical_fpr * N_SIMULATIONS))}/{N_SIMULATIONS}), "
        f"trigger_counts={result.checkpoint_trigger_counts}"
    )

    assert CORRECTED_FPR_LOWER_BOUND <= result.empirical_fpr <= CORRECTED_FPR_UPPER_BOUND, (
        f"Corrected empirical FPR {result.empirical_fpr:.4f} falls outside "
        f"the target band [{CORRECTED_FPR_LOWER_BOUND}, {CORRECTED_FPR_UPPER_BOUND}]. "
        f"Investigate whether alpha_spending_schedule() or the checkpoint-"
        f"threshold wiring in simulate_peeking_fpr() has changed."
    )


@pytest.mark.slow
def test_correction_produces_lower_fpr_than_naive_on_identical_data():
    """
    THE core comparative claim, tested directly rather than inferred from
    two separate band checks: on the EXACT SAME underlying null data (same
    seed), the corrected schedule must produce a STRICTLY lower FPR than
    the naive flat threshold. This is stronger evidence than both individual
    band checks passing independently, since it directly proves the
    correction WORKED on this data, not just that both numbers happen to
    separately fall in their respective expected ranges.
    """
    schedule = alpha_spending_schedule(n_checkpoints=10, total_alpha=0.05)

    naive = simulate_peeking_fpr(
        n_simulations=N_SIMULATIONS, n_checkpoints=10, checkpoint_n=200, seed=SEED
    )
    corrected = simulate_peeking_fpr(
        n_simulations=N_SIMULATIONS, n_checkpoints=10, checkpoint_n=200,
        threshold_schedule=schedule, seed=SEED,
    )

    assert corrected.empirical_fpr < naive.empirical_fpr, (
        f"Corrected FPR ({corrected.empirical_fpr:.4f}) is NOT lower than "
        f"naive FPR ({naive.empirical_fpr:.4f}) on identical underlying "
        f"data — the alpha-spending correction failed to improve anything, "
        f"a severe regression regardless of whether either number "
        f"individually falls in its expected band."
    )

# --------------------------------------------------------------------- #
# Irregular checkpoint spacing — closes the limitation stated directly in
# Phase 5's own documentation: "Checkpoints occur at fixed increments...
# not irregular real-world-style intervals... this configuration's
# behavior under irregular spacing has not been validated."
# --------------------------------------------------------------------- #

# Deliberately uneven — mimics daily checks under uneven traffic, same
# final total_n=2000 as the fixed-spacing tests above for comparability.
IRREGULAR_CHECKPOINT_NS = [120, 300, 550, 700, 1000, 1250, 1500, 1650, 1800, 2000]


@pytest.mark.slow
def test_correction_still_reduces_fpr_under_irregular_spacing():
    """
    THE question this test exists to answer, stated plainly: does the
    existing INDEX-based alpha-spending schedule (z_k depends on checkpoint
    COUNT k out of K, not on how far apart checkpoints actually are) still
    meaningfully reduce the false-positive rate when checkpoints are
    unevenly spaced, or does the fixed-spacing assumption quietly break it?

    This does NOT claim the correction is still calibrated to exactly 5% —
    only that it must still produce a MEANINGFULLY lower FPR than naive
    peeking on the same irregularly-spaced data, mirroring the weaker but
    still decisive comparative claim already proven for fixed spacing in
    test_correction_produces_lower_fpr_than_naive_on_identical_data above.
    A precise nominal-alpha band is intentionally NOT asserted here, since
    the classical O'Brien-Fleming design's exactness assumption (index
    ratio == information ratio) is known to not strictly hold once spacing
    is irregular — see core.sequential.simulate_peeking_fpr()'s own
    docstring note on this.

    Measured results for the current schedule are recorded in
    validation/validation_report.md.
    """
    schedule = alpha_spending_schedule(n_checkpoints=10, total_alpha=0.05)

    naive = simulate_peeking_fpr(
        n_simulations=N_SIMULATIONS, n_checkpoints=10,
        checkpoint_ns=IRREGULAR_CHECKPOINT_NS, seed=SEED,
    )
    corrected = simulate_peeking_fpr(
        n_simulations=N_SIMULATIONS, n_checkpoints=10, checkpoint_ns=IRREGULAR_CHECKPOINT_NS,
        threshold_schedule=schedule, seed=SEED,
    )

    print(
        f"\n[irregular spacing] naive_fpr={naive.empirical_fpr:.4f} "
        f"({int(round(naive.empirical_fpr * N_SIMULATIONS))}/{N_SIMULATIONS}), "
        f"corrected_fpr={corrected.empirical_fpr:.4f} "
        f"({int(round(corrected.empirical_fpr * N_SIMULATIONS))}/{N_SIMULATIONS})"
    )

    assert corrected.empirical_fpr < naive.empirical_fpr, (
        f"Under irregular checkpoint spacing, corrected FPR "
        f"({corrected.empirical_fpr:.4f}) is NOT lower than naive FPR "
        f"({naive.empirical_fpr:.4f}) — the index-based alpha-spending "
        f"schedule provides no protection at all once spacing is uneven, "
        f"a real finding that would mean this correction should not be "
        f"trusted for real-world irregular checkpoint timing without "
        f"further work (an information-based schedule, not index-based)."
    )
    # A materially looser sanity band than the fixed-spacing case — this is
    # deliberately NOT the tight [0.04, 0.075] band used above, since we
    # are explicitly not claiming exactness under an assumption (equal
    # spacing) the classical formula depends on.
    assert corrected.empirical_fpr <= 0.15, (
        f"Corrected FPR under irregular spacing ({corrected.empirical_fpr:.4f}) "
        f"is still well above nominal even accounting for the approximation "
        f"— the correction is providing only weak protection under uneven "
        f"checkpoint timing, worth flagging even though it is directionally "
        f"still better than naive."
    )


# --------------------------------------------------------------------- #
# Lan-DeMets (information-based) boundary — the correction used by default
# for live experiments, where look times are rarely the planned ones.
# --------------------------------------------------------------------- #

@pytest.mark.slow
@pytest.mark.parametrize("checkpoint_ns", [
    pytest.param([200 * (k + 1) for k in range(10)], id="equal_spacing"),
    pytest.param(IRREGULAR_CHECKPOINT_NS, id="irregular_spacing"),
])
def test_lan_demets_restores_nominal_fpr(checkpoint_ns):
    """
    Thresholds come from each look's information fraction
    (cumulative_n / final n), so the boundary is valid for ANY spacing —
    unlike alpha_spending_schedule(), which is calibrated for equal
    spacing only. Held to the same tight [3.5%, 7%] band as the
    equal-spacing O'Brien-Fleming test above, under BOTH spacings.
    """
    fractions = [n / checkpoint_ns[-1] for n in checkpoint_ns]
    schedule = lan_demets_boundary(fractions, total_alpha=0.05)

    result = simulate_peeking_fpr(
        n_simulations=N_SIMULATIONS, n_checkpoints=len(checkpoint_ns),
        checkpoint_ns=checkpoint_ns, threshold_schedule=schedule, seed=SEED,
    )
    print(
        f"\n[lan-demets] fpr={result.empirical_fpr:.4f} "
        f"({int(round(result.empirical_fpr * N_SIMULATIONS))}/{N_SIMULATIONS}), "
        f"trigger_counts={result.checkpoint_trigger_counts}"
    )
    assert CORRECTED_FPR_LOWER_BOUND <= result.empirical_fpr <= CORRECTED_FPR_UPPER_BOUND
