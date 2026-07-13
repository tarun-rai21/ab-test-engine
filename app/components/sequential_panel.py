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
import numpy as np
import streamlit as st
from sqlalchemy import text

from core.data_access import get_sequential_checkpoints
from core.inference import raw_ttest_ci
from core.sequential import SequentialCheckResult, alpha_spending_schedule


def _insert_checkpoint(engine, experiment_id: str, cumulative_n: int, p_value: float) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO sequential_checkpoints
                    (checkpoint_id, experiment_id, checked_at, cumulative_n,
                     p_value_at_check, alpha_threshold_at_check)
                VALUES
                    (:cid, :eid, :checked_at, :cum_n, :pval, 0.05)
                """
            ),
            {
                "cid": f"cp_{uuid.uuid4().hex[:12]}",
                "eid": experiment_id,
                "checked_at": datetime.now(timezone.utc).isoformat(),
                "cum_n": cumulative_n,
                "pval": p_value,
            },
        )


def _render_trace_plot(checkpoints: list[dict]) -> None:
    checkpoints_sorted = sorted(checkpoints, key=lambda c: c["cumulative_n"])
    schedule = alpha_spending_schedule(n_checkpoints=len(checkpoints_sorted), total_alpha=0.05)

    xs = [c["cumulative_n"] for c in checkpoints_sorted]
    ps = [c["p_value_at_check"] for c in checkpoints_sorted]

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(xs, ps, "o-", color="steelblue", label="Observed p-value")
    ax.axhline(0.05, color="firebrick", linestyle="--", label="Naive threshold (0.05)")
    ax.plot(xs, schedule, "x--", color="darkorange", label="Corrected threshold")
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
    control_converted: np.ndarray,
    treatment_converted: np.ndarray,
) -> None:
    st.subheader("Sequential Testing / Peeking Risk")
    st.caption(
        "Checking significance repeatedly as data accrues inflates the true "
        "false-positive rate above the nominal 5% — this panel tracks that "
        "risk across repeated looks at this experiment."
    )

    max_n_per_arm = min(len(control_converted), len(treatment_converted))
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
        prefix_result = raw_ttest_ci(
            control_converted[:checkpoint_n_per_arm],
            treatment_converted[:checkpoint_n_per_arm],
        )
        cumulative_n = checkpoint_n_per_arm * 2
        _insert_checkpoint(engine, experiment_id, cumulative_n, prefix_result.p_value)
        st.success(
            f"Logged checkpoint at n={cumulative_n} (n={checkpoint_n_per_arm}/arm), "
            f"p={prefix_result.p_value:.4g}. Re-run analysis to see the updated risk."
        )
        st.rerun()

    checkpoints = get_sequential_checkpoints(engine, experiment_id)

    if len(checkpoints) <= 1:
        st.info(
            f"{'No' if not checkpoints else 'Only one'} checkpoint logged yet — "
            f"peeking risk can only be assessed once an experiment has been "
            f"checked more than once. Use the slider above to log checkpoints "
            f"at a few different sample sizes to build up history."
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
    else:
        st.success(
            "✅ Naive and corrected conclusions agree — no peeking-related "
            "risk detected at this checkpoint."
        )

    _render_trace_plot(checkpoints)