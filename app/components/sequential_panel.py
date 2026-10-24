"""
app/components/sequential_panel.py — peeking-risk display, including a
p-value trace plot against both naive and corrected thresholds.

REAL GAP FOUND DURING MANUAL TESTING, fixed here: the first version of
this panel always logged a checkpoint at the CURRENT TOTAL sample size —
meaning repeated clicks without regenerating a larger dataset logged
identical (non-informative) checkpoints, not genuine sequential accrual.
Fixed by letting the user choose an EARLIER point within the data already
generated, and computing the p-value that would ACTUALLY have applied at
that smaller sample size — simulating realistic peeking against a single
already-generated dataset, rather than requiring new data-generation
infrastructure to produce genuinely growing data over time.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sqlalchemy import text

from core.data_access import get_sequential_checkpoints
from core.metrics import estimate_effect
from core.sequential import SequentialCheckResult, lan_demets_boundary


def _insert_checkpoint(
    engine, experiment_id: str, cumulative_n: int, p_value: float, alpha_threshold: float
) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO sequential_checkpoints
                    (checkpoint_id, experiment_id, checked_at, cumulative_n,
                     p_value_at_check, alpha_threshold_at_check)
                VALUES
                    (:cid, :eid, :checked_at, :cum_n, :pval, :alpha)
                """
            ),
            {
                "cid": f"cp_{uuid.uuid4().hex[:12]}",
                "eid": experiment_id,
                "checked_at": datetime.now(timezone.utc).isoformat(),
                "cum_n": cumulative_n,
                "pval": p_value,
                "alpha": alpha_threshold,
            },
        )


def _render_trace_plot(
    checkpoints: list[dict],
    n_checkpoints_planned: int | None,
    checkpoint_n: int | None,
) -> None:
    """
    The corrected threshold at each logged checkpoint is the Lan-DeMets
    boundary for the looks actually taken, at their information fractions
    (cumulative_n / planned total) — the same computation sequential_check()
    uses for the verdict, so the plotted line and the verdict always agree.
    Without a planned total, only the naive threshold is drawn.
    """
    checkpoints_sorted = sorted(checkpoints, key=lambda c: c["cumulative_n"])

    xs = [c["cumulative_n"] for c in checkpoints_sorted]
    ps = [c["p_value_at_check"] for c in checkpoints_sorted]

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(xs, ps, "o-", color="steelblue", label="Observed p-value")
    ax.axhline(0.05, color="firebrick", linestyle="--", label="Naive threshold (0.05)")
    if n_checkpoints_planned is not None and checkpoint_n is not None:
        planned_total = n_checkpoints_planned * checkpoint_n
        look_ns = sorted(set(xs))
        by_n = dict(zip(look_ns, lan_demets_boundary([n / planned_total for n in look_ns])))
        # Looks past the planned total can never reject (threshold 0), which
        # a log axis cannot draw — they are left off the threshold line.
        drawn = [(n, by_n[n]) for n in look_ns if by_n[n] > 0]
        if drawn:
            ax.plot(*zip(*drawn), "x--", color="darkorange",
                    label="Corrected threshold (Lan-DeMets)")
        ax.set_yscale("log")
    ax.set_xlabel("Cumulative sample size")
    ax.set_ylabel("p-value")
    ax.set_title("Peeking history: p-value vs. significance thresholds")
    ax.legend(fontsize=8)
    fig.tight_layout()

    st.pyplot(fig)


def render_sequential_panel(
    engine,
    experiment_id: str,
    sequential_risk: SequentialCheckResult | None,
    control_df: pd.DataFrame,
    treatment_df: pd.DataFrame,
    metric_name: str = "conversion",
    winsorization_cap: float | None = None,
    n_checkpoints_planned: int | None = None,
    checkpoint_n: int | None = None,
) -> None:
    st.subheader("Sequential Testing / Peeking Risk")
    st.caption(
        "Checking significance repeatedly as data accrues inflates the true "
        "false-positive rate above the nominal 5% — this panel tracks that "
        "risk across repeated looks at this experiment."
    )

    max_n_per_arm = min(len(control_df), len(treatment_df))
    if max_n_per_arm < 2:
        st.info("Not enough data generated to simulate a checkpoint history.")
        return

    st.markdown(
        "**Simulate an earlier checkpoint** — pretend you'd only collected "
        "data up to this point, and log what the p-value actually would "
        "have been at that size:"
    )
    checkpoint_n_per_arm = st.slider(
        "Users per arm at this checkpoint",
        min_value=2,
        max_value=max_n_per_arm,
        value=max_n_per_arm,
    )

    if st.button("📌 Log this checkpoint"):
        # Same metric and estimator as the main analysis, on the first
        # n users per arm.
        try:
            prefix_result = estimate_effect(
                control_df.iloc[:checkpoint_n_per_arm],
                treatment_df.iloc[:checkpoint_n_per_arm],
                metric_name, winsorization_cap,
            )
        except ValueError as exc:
            st.error(f"Cannot compute a p-value at this sample size: {exc}")
            return
        cumulative_n = checkpoint_n_per_arm * 2
        # Record the threshold that actually applied at this look: the
        # Lan-DeMets boundary given every look up to and including this one
        # if a plan is set, otherwise the naive 0.05.
        alpha_threshold = 0.05
        if n_checkpoints_planned is not None and checkpoint_n is not None:
            planned_total = n_checkpoints_planned * checkpoint_n
            prior_ns = {
                c["cumulative_n"] for c in get_sequential_checkpoints(engine, experiment_id)
                if c["cumulative_n"] < cumulative_n
            }
            look_ns = sorted(prior_ns | {cumulative_n})
            alpha_threshold = lan_demets_boundary([n / planned_total for n in look_ns])[-1]
        _insert_checkpoint(
            engine, experiment_id, cumulative_n, prefix_result.p_value, alpha_threshold
        )
        st.success(
            f"Logged checkpoint at n={cumulative_n} (n={checkpoint_n_per_arm}/arm), "
            f"p={prefix_result.p_value:.4g}. Re-run analysis to see the updated risk."
        )
        st.rerun()

    checkpoints = get_sequential_checkpoints(engine, experiment_id)

    if not checkpoints:
        st.info(
            "No checkpoint logged yet. Use the slider above to log checkpoints "
            "at a few different sample sizes to build up a peeking history."
        )
        return

    if sequential_risk is None:
        st.warning(
            "Checkpoint history exists, but the peeking-risk check wasn't "
            "enabled for this analysis run — tick 'Enable peeking-risk check' "
            "above and re-run analysis to see it evaluated."
        )
    elif sequential_risk.disagreement:
        st.error(
            "⚠️ **Naive and corrected conclusions disagree.** At the naive "
            "0.05 threshold this looks significant, but the peeking-corrected "
            "threshold says it is not — this result may be a false positive "
            "produced by checking too many times. Consider waiting for more data."
        )
    if sequential_risk is not None and sequential_risk.beyond_plan:
        st.warning(
            f"The latest checkpoint (n={sequential_risk.latest_checkpoint_n}) is "
            f"beyond the planned final sample size — the Lan-DeMets budget is "
            f"fully spent at the planned total, so this look cannot produce a "
            f"significant result. Extending past the plan needs a new design."
        )
    if sequential_risk is not None and not sequential_risk.disagreement:
        st.success(
            "✅ Naive and corrected conclusions agree — no peeking-related "
            "risk detected at this checkpoint."
        )

    _render_trace_plot(checkpoints, n_checkpoints_planned, checkpoint_n)
