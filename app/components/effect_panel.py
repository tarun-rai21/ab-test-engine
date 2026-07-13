"""
app/components/effect_panel.py — raw vs CUPED effect display, side by side,
with confidence intervals shown visually.

Renders core.inference.InferenceResult objects (raw_effect, optionally
cuped_effect) and the CUPED variance-reduction percentage, all already
computed by core.pipeline.analyze_experiment() — no new statistical logic.

Uses matplotlib for the CI comparison bar (per the original spec's tech
stack table: "matplotlib for static report plots... CI comparison bars").
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st

from core.inference import InferenceResult


def _render_ci_bar(raw_effect: InferenceResult, cuped_effect: InferenceResult | None) -> None:
    fig, ax = plt.subplots(figsize=(6, 1.8 if cuped_effect else 1.0))

    rows = [("Raw", raw_effect)]
    if cuped_effect is not None:
        rows.append(("CUPED", cuped_effect))

    y_positions = list(range(len(rows)))
    for y, (label, result) in zip(y_positions, rows):
        ax.plot([result.ci_lower, result.ci_upper], [y, y], color="steelblue", linewidth=3)
        ax.plot(result.point_estimate, y, "o", color="steelblue", markersize=8)

    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([label for label, _ in rows])
    ax.set_xlabel("Effect estimate (treatment - control)")
    ax.set_title("95% confidence interval — dashed line marks zero effect")
    fig.tight_layout()

    st.pyplot(fig)


def render_effect_panel(
    raw_effect: InferenceResult,
    cuped_effect: InferenceResult | None,
    cuped_variance_reduction_pct: float | None,
) -> None:
    st.subheader("Effect Estimate")

    _render_ci_bar(raw_effect, cuped_effect)

    if cuped_effect is not None:
        col1, col2 = st.columns(2)
    else:
        col1 = st.container()
        col2 = None

    with col1:
        st.markdown("**Raw estimate**")
        st.metric("Point estimate", f"{raw_effect.point_estimate:+.4f}")
        st.caption(
            f"95% CI: [{raw_effect.ci_lower:.4f}, {raw_effect.ci_upper:.4f}]  "
            f"·  p = {raw_effect.p_value:.4g}"
        )
        if raw_effect.ci_lower <= 0 <= raw_effect.ci_upper:
            st.caption("⚠️ This interval crosses zero — not statistically significant.")

    if col2 is not None and cuped_effect is not None:
        with col2:
            st.markdown("**CUPED-adjusted estimate**")
            st.metric("Point estimate", f"{cuped_effect.point_estimate:+.4f}")
            st.caption(
                f"95% CI: [{cuped_effect.ci_lower:.4f}, {cuped_effect.ci_upper:.4f}]  "
                f"·  p = {cuped_effect.p_value:.4g}"
            )
            if cuped_effect.ci_lower <= 0 <= cuped_effect.ci_upper:
                st.caption("⚠️ This interval crosses zero — not statistically significant.")

    if cuped_variance_reduction_pct is not None:
        st.info(
            f"CUPED reduced variance by **{cuped_variance_reduction_pct:.1f}%** "
            f"compared to the raw estimate, using pre-experiment data as a "
            f"noise-reduction covariate — this typically produces a narrower, "
            f"more precise confidence interval above without changing what "
            f"the estimate is actually measuring."
        )
    else:
        st.caption(
            "CUPED not available for this dataset (either disabled, or no "
            "usable pre-period covariate was found)."
        )
