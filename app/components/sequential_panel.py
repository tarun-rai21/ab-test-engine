"""
app/components/sequential_panel.py — peeking-risk display, including a
p-value trace plot against both naive and corrected thresholds.

Looks are recorded with core.monitoring.record_look(), once per treatment
arm on the metric being analysed, so every arm-vs-control comparison has
its own history and its own verdict.

REAL GAP FOUND DURING MANUAL TESTING, fixed here: the first version of
this panel always logged a checkpoint at the CURRENT TOTAL sample size —
meaning repeated clicks without regenerating a larger dataset logged
identical (non-informative) checkpoints, not genuine sequential accrual.
Fixed by letting the user choose an EARLIER point within the data already
generated, and computing the p-value that would ACTUALLY have applied at
that smaller sample size — simulating realistic peeking against a single
already-generated dataset.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from core.data_access import get_sequential_checkpoints
from core.monitoring import record_look
from core.sequential import SequentialCheckResult


def _render_trace_plot(checkpoints: list[dict], arm: str) -> None:
    """
    Observed p-values against the naive 0.05 and the corrected threshold
    RECORDED at each look (the rule actually in force then, including
    after any mid-test change of plan).
    """
    rows = sorted(checkpoints, key=lambda c: c["cumulative_n"])
    xs = [c["cumulative_n"] for c in rows]
    ps = [c["p_value_at_check"] for c in rows]

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(xs, ps, "o-", color="steelblue", label="Observed p-value")
    ax.axhline(0.05, color="firebrick", linestyle="--", label="Naive threshold (0.05)")
    # Looks past the planned total can never reject (threshold 0), which a
    # log axis cannot draw; looks logged without a plan have no correction.
    corrected = [
        (c["cumulative_n"], c["alpha_threshold_at_check"]) for c in rows
        if c["planned_total_n"] is not None and c["alpha_threshold_at_check"] > 0
    ]
    if corrected:
        ax.plot(*zip(*corrected), "x--", color="darkorange",
                label="Corrected threshold (Lan-DeMets)")
        ax.set_yscale("log")
    ax.set_xlabel("Cumulative sample size (all variants)")
    ax.set_ylabel("p-value")
    ax.set_title(f"Peeking history — {arm} vs control")
    ax.legend(fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)


def _verdict_table(risks: tuple[SequentialCheckResult, ...]) -> None:
    st.dataframe(pd.DataFrame([{
        "Arm": r.treatment_arm,
        "Latest look (n)": r.latest_checkpoint_n,
        "p-value": f"{r.naive_p_value:.4g}",
        "Corrected threshold": f"{r.corrected_threshold:.4g}",
        "Naive says significant": "Yes" if r.naive_significant else "No",
        "Corrected says significant": "Yes" if r.corrected_significant else "No",
        "Peeking risk": "⚠️ Yes" if r.disagreement else "No",
    } for r in risks]), hide_index=True, width="stretch")


def render_sequential_panel(
    engine,
    experiment_id: str,
    sequential_risks: tuple[SequentialCheckResult, ...],
    max_users_per_variant: int,
    arm_names: list[str],
    metric_name: str = "conversion",
    n_checkpoints_planned: int | None = None,
    checkpoint_n: int | None = None,
) -> None:
    st.subheader("Sequential Testing / Peeking Risk")
    st.caption(
        "Checking significance repeatedly as data accrues inflates the true "
        "false-positive rate above the nominal 5% — this panel tracks that "
        f"risk across repeated looks, for every arm, on `{metric_name}`."
    )

    if max_users_per_variant < 2:
        st.info("Not enough data generated to simulate a checkpoint history.")
        return

    st.markdown(
        "**Simulate an earlier checkpoint** — pretend you'd only collected "
        "data up to this point, and log what the p-values actually would "
        "have been at that size:"
    )
    users_per_variant = st.slider(
        "Users per variant at this checkpoint",
        min_value=2,
        max_value=max_users_per_variant,
        value=max_users_per_variant,
    )

    if st.button("📌 Log this checkpoint"):
        try:
            looks = record_look(
                engine, experiment_id, metric_name=metric_name,
                users_per_variant=users_per_variant,
                n_checkpoints_planned=n_checkpoints_planned, checkpoint_n=checkpoint_n,
            )
        except ValueError as exc:
            st.error(f"Cannot compute a p-value at this sample size: {exc}")
            return
        summary = ", ".join(f"{look.treatment_arm}: p={look.p_value:.4g}" for look in looks)
        st.success(
            f"Logged checkpoint at n={looks[0].cumulative_n} ({summary}). "
            f"Re-run analysis to see the updated risk."
        )
        st.rerun()

    primary_arm = arm_names[0]
    checkpoints = get_sequential_checkpoints(engine, experiment_id, metric_name, primary_arm)
    if not checkpoints:
        st.info(
            "No checkpoint logged yet for this metric. Use the slider above to "
            "log checkpoints at a few different sample sizes."
        )
        return

    if not sequential_risks:
        st.warning(
            "Checkpoint history exists, but the peeking-risk check wasn't "
            "enabled for this analysis run — tick 'Enable peeking-risk check' "
            "above and re-run analysis to see it evaluated."
        )
    else:
        if len(sequential_risks) > 1:
            _verdict_table(sequential_risks)
        flagged = [r.treatment_arm for r in sequential_risks if r.disagreement]
        if flagged:
            st.error(
                f"⚠️ **Naive and corrected conclusions disagree** for {', '.join(flagged)}. "
                "At the naive 0.05 threshold this looks significant, but the "
                "peeking-corrected threshold says it is not — the result may be a "
                "false positive produced by checking too many times."
            )
        else:
            st.success(
                "✅ Naive and corrected conclusions agree — no peeking-related "
                "risk detected at the latest checkpoint."
            )
        if any(r.replanned for r in sequential_risks):
            st.info(
                "The planned sample size changed during the test. Looks taken under "
                "the earlier plan keep the thresholds that were in force; the alpha "
                "they spent is carried forward under the new plan."
            )
        if any(r.beyond_plan for r in sequential_risks):
            st.warning(
                "The latest checkpoint is beyond the planned total sample size, so "
                "the Lan-DeMets budget is fully spent and this look cannot produce a "
                "significant result. To keep going, increase the planned schedule: "
                "past looks keep their thresholds and the remaining alpha is spread "
                "over the new plan."
            )

    _render_trace_plot(checkpoints, primary_arm)
