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
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines[:-1]] + [lines[-1]] if lines else []}


def code(*lines: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [l + "\n" for l in lines[:-1]] + [lines[-1]] if lines else [],
    }


cells = [
    md(
        "# A/B Test Analysis Engine — Narrative Walkthrough",
        "",
        "This notebook demonstrates the full engine end to end: simulate an "
        "experiment with a **known, configured ground truth** → check "
        "randomization validity → estimate the effect (raw and CUPED) → "
        "demonstrate the peeking-inflation problem and its correction → "
        "break the result down by segment and flag Simpson's paradox.",
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
        "from core.sequential import alpha_spending_schedule, simulate_peeking_fpr",
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
        "print(f'95% CI: [{report.raw_effect.ci_lower:.4f}, {report.raw_effect.ci_upper:.4f}]')",
        "print(f'p-value: {report.raw_effect.p_value:.4g}')",
        "",
        "print('\\n--- CUPED ---')",
        "print(f'Point estimate: {report.cuped_effect.point_estimate:+.4f}')",
        "print(f'95% CI: [{report.cuped_effect.ci_lower:.4f}, {report.cuped_effect.ci_upper:.4f}]')",
        "print(f'p-value: {report.cuped_effect.p_value:.4g}')",
        "print(f'\\nVariance reduction from CUPED: {report.cuped_variance_reduction_pct:.1f}%')",
    ),
    code(
        "fig, ax = plt.subplots(figsize=(7, 2))",
        "for y, (label, result) in enumerate([('Raw', report.raw_effect), ('CUPED', report.cuped_effect)]):",
        "    ax.plot([result.ci_lower, result.ci_upper], [y, y], color='steelblue', linewidth=3)",
        "    ax.plot(result.point_estimate, y, 'o', color='steelblue', markersize=8)",
        "ax.axvline(0, color='gray', linestyle='--')",
        "ax.set_yticks([0, 1]); ax.set_yticklabels(['Raw', 'CUPED'])",
        "ax.set_xlabel('Effect estimate (treatment - control)')",
        "ax.set_title('95% confidence intervals: raw vs. CUPED')",
        "plt.tight_layout(); plt.show()",
    ),
    md(
        "## 4. The peeking problem — why checking results repeatedly is dangerous",
        "",
        "If an analyst checks significance at every one of 10 checkpoints "
        "and stops the moment `p < 0.05` appears, the TRUE false-positive "
        "rate is far above the nominal 5% — even when there is truly no "
        "effect. We measure this directly across 500 independent null-"
        "effect simulations, then show that a simple alpha-spending "
        "correction restores the false-positive rate back near nominal.",
    ),
    code(
        "naive = simulate_peeking_fpr(",
        "    n_simulations=500, n_checkpoints=10, checkpoint_n=200, seed=2024,",
        ")",
        "schedule = alpha_spending_schedule(n_checkpoints=10, total_alpha=0.05)",
        "corrected = simulate_peeking_fpr(",
        "    n_simulations=500, n_checkpoints=10, checkpoint_n=200,",
        "    threshold_schedule=schedule, seed=2024,",
        ")",
        "",
        "print(f'Naive empirical false-positive rate:     {naive.empirical_fpr:.1%}  (nominal target: 5%)')",
        "print(f'Alpha-spending corrected FPR:             {corrected.empirical_fpr:.1%}')",
        "print(f'\\nNaive trigger positions (by checkpoint):     {naive.checkpoint_trigger_counts}')",
        "print(f'Corrected trigger positions (by checkpoint): {corrected.checkpoint_trigger_counts}')",
    ),
    md(
        "Notice where the triggers happen: naive triggers are front-loaded "
        "at early checkpoints (stopping on noise), while corrected triggers "
        "shift almost entirely to the final checkpoint — direct evidence "
        "the correction targets *early-look noise* specifically, not just a "
        "uniform threshold reduction.",
    ),
    md(
        "## 5. Segment analysis — does the pooled result hide something?",
        "",
        "We now configure a **deliberately heterogeneous** segment effect: "
        "mobile users respond well, desktop users respond badly — and check "
        "whether the engine correctly recovers both directions and flags "
        "the disagreement (Simpson's paradox).",
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
        "    flag = ' <-- SIMPSON\\'S PARADOX' if s.simpsons_flag else ''",
        "    print(f'{s.segment_value:10s}: {s.inference.point_estimate:+.4f}  '",
        "          f'(p={s.inference.p_value:.4g}, BH-significant={s.bh_significant}){flag}')",
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
        "rate** well above 5%, and a simple alpha-spending correction "
        "restores it — with the trigger-position shift proving the "
        "correction targets the right mechanism.",
        "4. **Segment analysis correctly recovers a deliberately configured "
        "heterogeneous effect** and flags the segment that disagrees in "
        "direction with the pooled result — a real Simpson's-paradox catch.",
        "",
        "Full technical documentation for every phase, every design "
        "decision, and every bug found along the way is available in the "
        "project's `Phase_*_Documentation.md` files and `README.md`.",
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