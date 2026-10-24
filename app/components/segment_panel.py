"""
app/components/segment_panel.py — segment table + effect-reversal
banner.

Renders core.segments.SegmentAnalysisResult objects, already computed by
core.pipeline.analyze_experiment() when segment_columns is requested — no
new statistical logic, purely a rendering layer over Phase 6's already-
proven output.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from core.segments import SegmentAnalysisResult


def render_segment_panel(segments: tuple[SegmentAnalysisResult, ...] | None) -> None:
    st.subheader("Segment Analysis")

    if not segments:
        st.caption(
            "No segment breakdown requested for this analysis. Select one or "
            "more segment columns above and re-run to see per-segment effects."
        )
        return

    for analysis in segments:
        arm_label = f" — `{analysis.treatment_arm}` vs control" if analysis.treatment_arm else ""
        st.markdown(f"**By `{analysis.segment_column}`{arm_label}**")

        any_reversal = any(s.reversal_flag for s in analysis.segments)
        if any_reversal:
            flagged_names = [s.segment_value for s in analysis.segments if s.reversal_flag]
            st.error(
                f"🔀 **Effect reversal:** the segment(s) "
                f"{', '.join(flagged_names)} show a statistically significant "
                f"(BH-corrected) effect in the OPPOSITE direction from the "
                f"overall pooled result "
                f"({analysis.pooled_point_estimate:+.4f}). The aggregate number "
                f"alone would hide this — investigate before shipping based on "
                f"the pooled result."
            )

        rows = []
        for s in analysis.segments:
            rows.append({
                "Segment": s.segment_value,
                "Effect estimate": f"{s.inference.point_estimate:+.4f}",
                "95% CI": f"[{s.inference.ci_lower:.4f}, {s.inference.ci_upper:.4f}]",
                "p-value": f"{s.inference.p_value:.4g}",
                "Significant (BH-corrected)": "✅ Yes" if s.bh_significant else "—",
                "Significant reversal?": "⚠️ Yes" if s.reversal_flag else "No",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

        if analysis.excluded_segments:
            st.caption(
                f"Excluded from analysis (insufficient data): "
                f"{', '.join(analysis.excluded_segments)}"
            )

        if analysis.heterogeneity_p_value is not None:
            differs = analysis.heterogeneity_p_value < 0.05
            st.markdown(
                f"**Do the segment effects differ?** Cochran's Q = "
                f"{analysis.heterogeneity_q:.2f}, p = {analysis.heterogeneity_p_value:.4g} — "
                + ("evidence that the effect varies across segments."
                   if differs else
                   "no evidence that the effect varies across segments (this test has "
                   "low power with few or small segments).")
            )

        st.caption(
            "'Significant (BH-corrected)' uses one Benjamini-Hochberg correction "
            "across every segment tested in this report (all columns and arms), "
            "controlling the false-discovery rate. A segment being significant "
            "while another is not does not by itself show their effects differ — "
            "the Q test above answers that."
        )
        st.divider()
