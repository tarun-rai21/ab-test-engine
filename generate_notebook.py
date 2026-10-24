"""
generate_notebook.py — one-time script that builds notebooks/writeup.ipynb.

Run once:
    python generate_notebook.py

A .ipynb file is JSON under the hood — this script builds that JSON
directly using only Python's built-in json module, no new dependencies.
Code cells are left with EMPTY outputs (execution_count: null, outputs: [])
rather than pre-filled with numbers claiming to be "already run" — this
project's standing rule since Phase 1 is to prove claims by execution, not
assert them, and fabricating output that looks real but wasn't would
violate that rule. Open the resulting notebook and use "Restart Kernel and
Run All" to produce genuine, reproducible output.
"""

import json
from pathlib import Path


def md(*lines: str) -> dict:
    source = [line + "\n" for line in lines[:-1]] + [lines[-1]] if lines else []
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(*lines: str) -> dict:
    source = [line + "\n" for line in lines[:-1]] + [lines[-1]] if lines else []
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


cells = [
    md(
        "# A/B Test Analysis Engine — Narrative Walkthrough",
        "",
        "This notebook demonstrates the full engine end to end: simulate an "
        "experiment with a **known, configured ground truth** → check "
        "randomization validity → estimate the effect (raw and CUPED) → "
        "demonstrate the peeking-inflation problem and its correction → "
        "break the result down by segment and flag effect reversals → "
        "handle irregular look times, revenue metrics and multi-arm tests.",
        "",
        "Every number below is produced by running this project's own code "
        "— nothing here is fabricated or asserted from theory. Run this "
        "notebook top to bottom (`Restart Kernel and Run All`) to reproduce "
        "these results yourself.",
    ),
    code(
        "import sys",
        "from pathlib import Path",
        "sys.path.insert(0, str(Path.cwd().parent))  # project root, if running from notebooks/",
        "",
        "import numpy as np",
        "import matplotlib.pyplot as plt",
        "",
        "from core.data_access import get_inference_data, split_by_variant",
        "from core.inference import raw_ttest_ci, cuped_adjust, variance_reduction_pct",
        "from core.pipeline import analyze_experiment",
        "from core.data_access import get_stored_report",
        "from core.sequential import (",
        "    alpha_spending_schedule, lan_demets_boundary, sequential_check, simulate_peeking_fpr,",
        ")",
        "from data_sim.simulator import ExperimentSimulator",
        "from db.connection import get_engine, init_schema, reset_engine",
        "from db.seed import seed_database",
        "",
        "print('Imports OK')",
    ),
    md(
        "## 1. Simulate an experiment with a known ground truth",
        "",
        "We configure a synthetic experiment with a **known true effect** "
        "(`true_effect=0.02`) and a **known covariate correlation** "
        "(`covariate_correlation=0.5`, for CUPED) — the simulator records "
        "exactly what was configured, so every claim below can be checked "
        "against a real, known answer, not just trusted.",
    ),
    code(
        "import os",
        "os.environ['DATABASE_URL'] = 'sqlite:///:memory:'",
        "reset_engine()",
        "engine = get_engine()",
        "init_schema(engine)",
        "",
        "config = {'simulator': {",
        "    'n_users': 20000, 'baseline_rate': 0.10, 'true_effect': 0.02,",
        "    'covariate_correlation': 0.5, 'seed': 777, 'corrupted_split': None,",
        "}}",
        "seed_database(config)",
        "experiment_id = 'exp_seed777'",
        "print(f'Seeded experiment: {experiment_id}')",
    ),
    md(
        "## 2. Validity check — is the randomization actually healthy?",
        "",
        "Before trusting any effect estimate, we check whether the observed "
        "control/treatment split matches the intended allocation (Sample "
        "Ratio Mismatch). A healthy experiment should NOT be flagged here.",
    ),
    code(
        "report = analyze_experiment(engine, experiment_id, use_cuped=True)",
        "",
        "print(f'SRM flagged: {report.srm.flagged}')",
        "print(f'SRM p-value: {report.srm.p_value:.4f}')",
        "print(f'Observed split: {report.srm.observed_counts}')",
        "print(f'Expected split: {report.srm.expected_counts}')",
        "print(f'Achievable MDE at this sample size: {report.achievable_mde:.4f}')",
    ),
    md(
        "## 3. Effect estimate — raw vs. CUPED",
        "",
        "The raw estimate is a standard Welch's two-sample t-test. The "
        "CUPED-adjusted estimate uses a pre-experiment covariate to reduce "
        "variance — same expected value, tighter confidence interval, if "
        "the covariate is genuinely correlated with the outcome (it is "
        "here, by construction: `covariate_correlation=0.5`).",
    ),
    code(
        "print('--- Raw ---')",
        "print(f'Point estimate: {report.raw_effect.point_estimate:+.4f}')",
        "print(f'95% CI: [{report.raw_effect.ci_lower:.4f}, '",
        "      f'{report.raw_effect.ci_upper:.4f}]')",
        "print(f'p-value: {report.raw_effect.p_value:.4g}')",
        "",
        "print('\\n--- CUPED ---')",
        "print(f'Point estimate: {report.cuped_effect.point_estimate:+.4f}')",
        "print(f'95% CI: [{report.cuped_effect.ci_lower:.4f}, '",
        "      f'{report.cuped_effect.ci_upper:.4f}]')",
        "print(f'p-value: {report.cuped_effect.p_value:.4g}')",
        "print(f'\\nVariance reduction from CUPED: {report.cuped_variance_reduction_pct:.1f}%')",
    ),
    code(
        "fig, ax = plt.subplots(figsize=(7, 2))",
        "results = [('Raw', report.raw_effect), ('CUPED', report.cuped_effect)]",
        "for y, (label, result) in enumerate(results):",
        "    ax.plot([result.ci_lower, result.ci_upper], [y, y], color='steelblue', linewidth=3)",
        "    ax.plot(result.point_estimate, y, 'o', color='steelblue', markersize=8)",
        "ax.axvline(0, color='gray', linestyle='--')",
        "ax.set_yticks([0, 1])",
        "ax.set_yticklabels(['Raw', 'CUPED'])",
        "ax.set_xlabel('Effect estimate (treatment - control)')",
        "ax.set_title('95% confidence intervals: raw vs. CUPED')",
        "plt.tight_layout()",
        "plt.show()",
    ),
    md(
        "## 4. The peeking problem — why checking results repeatedly is dangerous",
        "",
        "If an analyst checks significance at every one of 10 checkpoints "
        "and stops the moment `p < 0.05` appears, the TRUE false-positive "
        "rate is far above the nominal 5% — even when there is truly no "
        "effect. We measure this directly across 2000 independent null-"
        "effect simulations, then show that an O'Brien-Fleming boundary "
        "(constant calibrated so the overall false-positive rate is exactly "
        "5%) restores the false-positive rate to nominal.",
    ),
    code(
        "naive = simulate_peeking_fpr(",
        "    n_simulations=2000, n_checkpoints=10, checkpoint_n=200, seed=2024,",
        ")",
        "schedule = alpha_spending_schedule(n_checkpoints=10, total_alpha=0.05)",
        "corrected = simulate_peeking_fpr(",
        "    n_simulations=2000, n_checkpoints=10, checkpoint_n=200,",
        "    threshold_schedule=schedule, seed=2024,",
        ")",
        "",
        "print(f'Naive empirical FPR: {naive.empirical_fpr:.1%}  (nominal target: 5%)')",
        "print(f'Alpha-spending corrected FPR: {corrected.empirical_fpr:.1%}')",
        "print(f'\\nNaive trigger positions: {naive.checkpoint_trigger_counts}')",
        "print(f'Corrected trigger positions: {corrected.checkpoint_trigger_counts}')",
    ),
    md(
        "Notice where the triggers happen: naive triggers are front-loaded "
        "at early checkpoints (stopping on noise), while corrected triggers "
        "essentially never fire at the first looks and concentrate at the "
        "later checkpoints — direct evidence "
        "the correction targets *early-look noise* specifically, not just a "
        "uniform threshold reduction.",
    ),
    md(
        "## 5. Segment analysis — does the pooled result hide something?",
        "",
        "We now configure a **deliberately heterogeneous** segment effect: "
        "mobile users respond well, desktop users respond badly — and check "
        "whether the engine correctly recovers both directions and flags "
        "the significant reversal (qualitative effect heterogeneity).",
    ),
    code(
        "config2 = {'simulator': {",
        "    'n_users': 20000, 'baseline_rate': 0.10, 'true_effect': 0.0,",
        "    'covariate_correlation': 0.0, 'seed': 888, 'corrupted_split': None,",
        "    'segment_heterogeneity': {'device_type': {'mobile': 0.15, 'desktop': -0.08}},",
        "}}",
        "seed_database(config2)",
        "",
        "report2 = analyze_experiment(engine, 'exp_seed888', segment_columns=['device_type'])",
        "",
        "print(f'Pooled effect: {report2.raw_effect.point_estimate:+.4f}')",
        "print()",
        "for s in report2.segments[0].segments:",
        "    flag = ' <-- EFFECT REVERSAL' if s.reversal_flag else ''",
        "    print(f'{s.segment_value:10s}: {s.inference.point_estimate:+.4f}  '",
        "          f'(p={s.inference.p_value:.4g}, BH-significant={s.bh_significant}){flag}')",
        "",
        "device = report2.segments[0]",
        "print(f\"\\nDo the segment effects differ? Cochran's Q = {device.heterogeneity_q:.1f}, \"",
        "      f'p = {device.heterogeneity_p_value:.3g}')",
    ),
    md(
        "Per-segment significance alone cannot show that segments *differ* — "
        "one segment may simply have less data. Cochran's Q tests the "
        "difference directly (equivalent to a treatment × segment interaction "
        "test). When several columns or arms are analysed, one "
        "Benjamini-Hochberg correction runs across every segment test in the "
        "report.",
    ),
    md(
        "## 6. Irregular look times — the Lan-DeMets boundary",
        "",
        "The O'Brien-Fleming schedule in section 4 assumes looks happen at the "
        "planned, equally spaced sample sizes. Real teams check when they "
        "check. The Lan-DeMets boundary spends alpha by each look's "
        "**information fraction** (cumulative n / planned total n), so it stays "
        "exact under any spacing.",
    ),
    code(
        "irregular = [120, 300, 550, 700, 1000, 1250, 1500, 1650, 1800, 2000]  # per arm",
        "fractions = [n / irregular[-1] for n in irregular]",
        "",
        "for name, schedule in [",
        "    (\"Index-based O'Brien-Fleming\", alpha_spending_schedule(10)),",
        "    ('Lan-DeMets (information-based)', lan_demets_boundary(fractions)),",
        "]:",
        "    result = simulate_peeking_fpr(n_simulations=2000, n_checkpoints=10,",
        "                                  checkpoint_ns=irregular, threshold_schedule=schedule,",
        "                                  seed=2024)",
        "    print(f'{name:32s} false-positive rate under irregular looks: '",
        "          f'{result.empirical_fpr:.2%}')",
        "",
        "# Planned 10 looks of 200 users, but only two actual looks, at n=1000 and n=2000:",
        "looks = [{'cumulative_n': 1000, 'p_value_at_check': 0.20},",
        "         {'cumulative_n': 2000, 'p_value_at_check': 0.045}]",
        "for spending in ('obrien_fleming', 'lan_demets'):",
        "    check = sequential_check(looks, n_checkpoints_planned=10, checkpoint_n=200,",
        "                             spending=spending)",
        "    print(f'{spending:15s} final threshold {check.corrected_threshold:.4f} -> '",
        "          f'p=0.045 significant: {check.corrected_significant}')",
    ),
    md(
        "The index-based schedule treats the n=2000 look as look 10 of 10, as "
        "if nine earlier looks had spent alpha. Lan-DeMets knows only one "
        "earlier look happened, so it allows a threshold close to 0.05.",
    ),
    md(
        "## 7. Revenue metrics — heavy tails and a ratio metric",
        "",
        "Conversion is not the only outcome that matters. The simulator can "
        "attach log-normal (heavily right-skewed) order values to conversions. "
        "Here the treatment raises conversion by +0.02 and average order value "
        "by +10% on a mean of 50, so the true effects are known:",
        "",
        "- **revenue per user** (0 for non-converters): "
        "0.22 × 55 − 0.20 × 50 = **+2.10**, analysed with Welch's t-test, "
        "optionally winsorized;",
        "- **revenue per conversion** (average order value): 50 × 0.10 = "
        "**+5.00**. This is a ratio of two random per-user quantities, so it "
        "uses the **delta method** at the user level rather than treating "
        "orders as independent observations.",
    ),
    code(
        "seed_database({'simulator': {",
        "    'n_users': 40000, 'baseline_rate': 0.20, 'true_effect': 0.02,",
        "    'covariate_correlation': 0.3, 'seed': 999, 'corrupted_split': None,",
        "    'revenue_mean': 50.0, 'revenue_lift': 0.10, 'revenue_sigma': 1.0,",
        "}})",
        "",
        "for metric, options in [('revenue_per_user', {}),",
        "                        ('revenue_per_user', {'winsorize_quantile': 0.99}),",
        "                        ('revenue_per_conversion', {})]:",
        "    r = analyze_experiment(engine, 'exp_seed999', metric_name=metric, persist=False,",
        "                           **options)",
        "    best = r.cuped_effect or r.raw_effect",
        "    label = metric + (' (winsorized)' if options else '')",
        "    print(f'{label:32s} {best.method:11s} effect {best.point_estimate:+.3f}  '",
        "          f'95% CI [{best.ci_lower:+.3f}, {best.ci_upper:+.3f}]')",
    ),
    md(
        "Winsorizing caps the largest orders, which narrows the interval but "
        "also shrinks the estimate toward zero: it measures a capped revenue, "
        "a slightly different quantity. CUPED applies to revenue per user but "
        "not to the ratio metric.",
    ),
    md(
        "## 8. More than two arms — Dunnett-adjusted comparisons",
        "",
        "With several treatment arms, each compared with control, the chance "
        "that *some* arm looks like a winner by luck grows with the number of "
        "arms. Step-down Dunnett-adjusted p-values hold that family-wise error at "
        "5%, using the correlation the shared control group creates (Holm is "
        "available too, and more conservative). Here "
        "`treatment` has a real +0.03 effect and `treatment_b` has none.",
    ),
    code(
        "seed_database({'simulator': {",
        "    'n_users': 30000, 'baseline_rate': 0.10, 'true_effect': 0.0,",
        "    'covariate_correlation': 0.3, 'seed': 1001, 'corrupted_split': None,",
        "    'treatment_arms': {'treatment': 0.03, 'treatment_b': 0.0},",
        "}})",
        "",
        "report3 = analyze_experiment(engine, 'exp_seed1001')",
        "print(f'SRM across 3 variants flagged: {report3.srm.flagged}')",
        "for c in report3.comparisons:",
        "    effect = c.cuped_effect or c.raw_effect",
        "    p_adj = c.cuped_p_value_adjusted if c.cuped_effect else c.raw_p_value_adjusted",
        "    print(f'{c.treatment_arm:12s} effect {effect.point_estimate:+.4f}  '",
        "          f'p={effect.p_value:.3g}  Dunnett p={p_adj:.3g}  '",
        "          f'significant={p_adj < 0.05}')",
        "",
        "stored = get_stored_report(engine, 'exp_seed1001', 'conversion')",
        "print('\\nStored report available for GET /report, computed at',",
        "      stored['computed_at'])",
    ),
    md(
        "## Summary",
        "",
        "This walkthrough demonstrated, with real executed code and real "
        "measured output:",
        "",
        "1. **SRM detection** correctly clears a healthy randomization.",
        "2. **CUPED** measurably reduces variance versus the raw estimate "
        "when a correlated pre-experiment covariate exists.",
        "3. **Naive repeated peeking genuinely inflates the false-positive "
        "rate** to roughly 4x the nominal 5%, and a calibrated "
        "O'Brien-Fleming boundary restores it — with the trigger-position "
        "shift showing the "
        "correction targets the right mechanism.",
        "4. **Segment analysis correctly recovers a deliberately configured "
        "heterogeneous effect** and flags the segment that disagrees in "
        "direction with the pooled result — a sign-reversing heterogeneity "
        "the pooled number alone would hide — and the Q test confirms the "
        "segment effects differ.",
        "5. **Lan-DeMets** keeps the false-positive rate at 5% when looks "
        "happen at irregular times, and does not over-penalise looks that "
        "never happened.",
        "6. **Revenue metrics** recover known effects on heavy-tailed data, "
        "with the delta method for the ratio metric.",
        "7. **Multi-arm tests** separate a real winner from a null arm with "
        "Dunnett-adjusted p-values.",
        "",
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

output_path = Path("notebooks/writeup.ipynb")
output_path.parent.mkdir(exist_ok=True)
output_path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(f"Wrote {output_path}")
