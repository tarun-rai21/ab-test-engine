"""
app/components/segment_panel.py — segment table + Simpson's-paradox
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
        st.markdown(f"**By `{analysis.segment_column}`**")

        any_paradox = any(s.simpsons_flag for s in analysis.segments)
        if any_paradox:
            flagged_names = [s.segment_value for s in analysis.segments if s.simpsons_flag]
            st.error(
                f"🔀 **Simpson's-paradox warning:** the segment(s) "
                f"{', '.join(flagged_names)} show an effect in the OPPOSITE "
                f"direction from the overall pooled result "
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
                "Disagrees with pooled?": "⚠️ Yes" if s.simpsons_flag else "No",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        if analysis.excluded_segments:
            st.caption(
                f"Excluded from analysis (insufficient data): "
                f"{', '.join(analysis.excluded_segments)}"
            )

        st.caption(
            "'Significant (BH-corrected)' uses the Benjamini-Hochberg "
            "correction across all segments tested in this column, "
            "controlling the false-discovery rate from testing multiple "
            "segments at once."
        )
        st.divider()
