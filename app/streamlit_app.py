"""
app/streamlit_app.py — Phase 8 entrypoint. Assembles all five components
into one single-page report, per the original spec's own stated layout
("assemble into a single-page report").

Run:
    streamlit run app/streamlit_app.py

Calls core.pipeline.analyze_experiment() DIRECTLY (not via the Phase 7
API) — a deliberate architecture decision: for a single-user local demo
tool, running one process (streamlit) is simpler and more reliable than
requiring uvicorn + streamlit running simultaneously, with no real benefit
from the added HTTP round-trip in this context. Phase 7's API remains a
separate, independently-proven piece of this project either way.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from app.components.effect_panel import render_effect_panel
from app.components.segment_panel import render_segment_panel
from app.components.sequential_panel import render_sequential_panel
from app.components.upload_or_generate import render_data_input
from app.components.validity_panel import render_validity_panel
from core.data_access import ALLOWED_SEGMENT_COLUMNS, get_inference_data, split_by_variant
from core.pipeline import analyze_experiment
from db.connection import get_engine

st.set_page_config(page_title="A/B Test Analysis Engine", layout="wide")

st.title("A/B Test Analysis Engine")
st.caption(
    "Structural enforcement of correct experimentation methodology — "
    "SRM detection, CUPED variance reduction, peeking correction, and "
    "Simpson's-paradox-aware segment analysis."
)

# --------------------------------------------------------------------- #
# 1. Data input
# --------------------------------------------------------------------- #

new_experiment_id = render_data_input()
if new_experiment_id is not None and new_experiment_id != st.session_state.get("experiment_id"):
    st.session_state["experiment_id"] = new_experiment_id
    st.session_state["report"] = None  # clear any stale report from a previous dataset

experiment_id = st.session_state.get("experiment_id")

if experiment_id is None:
    st.info("Generate or upload data above to begin.")
    st.stop()

st.divider()

# --------------------------------------------------------------------- #
# 2. Analysis configuration + trigger
# --------------------------------------------------------------------- #

st.subheader("Run Analysis")
col1, col2, col3 = st.columns(3)

with col1:
    use_cuped = st.checkbox("Use CUPED variance reduction", value=True)

with col2:
    segment_columns = st.multiselect(
        "Segment analysis by", options=list(ALLOWED_SEGMENT_COLUMNS)
    )

with col3:
    enable_sequential = st.checkbox("Enable peeking-risk check", value=False)
    n_checkpoints_planned = st.number_input(
        "Planned checkpoints", min_value=1, value=10, disabled=not enable_sequential
    )
    checkpoint_n = st.number_input(
        "Users per checkpoint", min_value=1, value=200, disabled=not enable_sequential
    )

if st.button("▶️ Run analysis", type="primary"):
    engine = get_engine()
    try:
        report = analyze_experiment(
            engine,
            experiment_id,
            use_cuped=use_cuped,
            segment_columns=segment_columns or None,
            n_checkpoints_planned=int(n_checkpoints_planned) if enable_sequential else None,
            checkpoint_n=int(checkpoint_n) if enable_sequential else None,
        )
        st.session_state["report"] = report
    except ValueError as exc:
        st.error(f"Analysis failed: {exc}")
        st.session_state["report"] = None

report = st.session_state.get("report")
if report is None:
    st.info("Configure the options above and click 'Run analysis' to see results.")
    st.stop()

st.divider()

# --------------------------------------------------------------------- #
# 3. Report — all five panels, single page
# --------------------------------------------------------------------- #

render_validity_panel(report.srm, report.achievable_mde)
st.divider()

render_effect_panel(report.raw_effect, report.cuped_effect, report.cuped_variance_reduction_pct)
st.divider()

control_df, treatment_df = split_by_variant(get_inference_data(get_engine(), experiment_id))

render_sequential_panel(
    get_engine(),
    experiment_id,
    report.sequential_risk,
    control_converted=control_df["converted"].to_numpy(dtype=float),
    treatment_converted=treatment_df["converted"].to_numpy(dtype=float),
)
st.divider()

render_segment_panel(report.segments)
