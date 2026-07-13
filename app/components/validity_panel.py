"""
app/components/validity_panel.py — SRM check + achievable-MDE display.

Renders core.validity.SRMResult and the achievable_mde float already
computed by core.pipeline.analyze_experiment() — no new statistical
logic, purely a rendering layer over already-proven values.

Per spec NFR4 ("plain-language flags... in addition to the underlying
statistics"): an SRM-flagged result gets a prominent, unambiguous warning
BEFORE any numbers are shown — this is the one failure mode serious
enough that a non-technical reader should stop and not trust anything
else on the page until it's resolved.
"""

from __future__ import annotations

import streamlit as st

from core.validity import SRMResult


def render_validity_panel(srm: SRMResult, achievable_mde: float | None) -> None:
    st.subheader("Validity Checks")

    if srm.flagged:
        st.error(
            "🚫 **Do not trust this result: SRM detected.** "
            "The observed split between control and treatment does not "
            "match the intended allocation — this usually means the "
            "randomization itself is broken (a redirect bug, bot-filtering "
            "asymmetry, or a logging gap), not that the treatment doesn't work. "
            "Every number below is still shown, but should not be acted on "
            "until this is investigated and fixed."
        )
    else:
        st.success("✅ Randomization looks healthy — no sample ratio mismatch detected.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Observed split", " / ".join(str(c) for c in srm.observed_counts))
    with col2:
        st.metric("Expected split", " / ".join(f"{c:.0f}" for c in srm.expected_counts))
    with col3:
        st.metric("SRM p-value", f"{srm.p_value:.4g}")

    with st.expander("SRM statistical detail"):
        st.write(f"Chi-square statistic: `{srm.chi_sq_stat:.4f}`")
        st.write(
            "Flag threshold: p < 0.001 (stricter than the conventional 0.05, "
            "since this check runs on every single analysis — see "
            "core/validity.py's own docstring for why)."
        )

    st.divider()

    st.subheader("Was this test capable of detecting a meaningful effect?")
    if achievable_mde is None:
        st.warning(
            "Achievable MDE could not be computed for this dataset "
            "(e.g. the observed baseline rate was exactly 0% or 100%)."
        )
    else:
        st.metric(
            "Smallest effect this sample size could reliably detect",
            f"{achievable_mde:.2%}",
        )
        st.caption(
            "If your observed effect estimate is smaller than this number, "
            "a 'no significant difference' result is inconclusive, not proof "
            "the treatment doesn't work — the test may simply not have had "
            "enough data to detect an effect this small."
        )