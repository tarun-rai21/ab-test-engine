"""
core/pipeline.py — the orchestration layer implementing spec Section 4.4's
full analysis sequence end to end: SRM -> power/MDE context -> raw effect
estimate -> optional CUPED effect estimate -> optional sequential-peeking
check -> optional segment analysis -> one assembled AnalysisReport.

WHY THIS FILE EXISTS: every module it calls (core/validity.py,
core/inference.py, core/sequential.py, core/segments.py,
core/persistence.py, core/data_access.py) has been independently,
thoroughly tested in isolation. None of them had ever been run together,
end to end, for a single experiment, until this file — a real gap
explicitly identified while auditing Phases 1-5: nothing in the codebase
owned the actual product surface (what a future Streamlit app or API
endpoint would call).

DESIGN DECISIONS, made explicit here rather than left implicit in code:

1. SRM-flagged experiments are NOT skipped. Every downstream step still
   runs — matches spec 4.4 step 2 ("pipeline still runs downstream modules
   but the report renders a blocking warning ahead of any effect estimate").
   Trust-tagging is already handled correctly by persist_inference_result(),
   which computes `trusted` from the SRMResult itself, never from caller
   intent — nothing extra needed here to keep that contract.

2. Achievable MDE uses mde_curve(), not required_sample_size(). Spec step 3
   asks "was this test even capable of detecting a meaningful effect given
   the sample size it actually got" — that is the INVERSE direction
   mde_curve() already solves, and needs no extra input from the caller
   beyond what the realized data already provides.

3. CUPED's theta is estimated on the POOLED dataset (both arms combined),
   then the SAME adjustment is applied to both arms before splitting for
   the two-sample test. This is not a new convention invented for this
   file — it exactly mirrors the pattern already established and verified
   in tests/test_data_access.py's
   test_full_pipeline_cuped_variance_reduction_matches_theory.

4. Sequential-peeking check is genuinely optional. The schema has no column
   anywhere that stores an experiment's PLANNED checkpoint schedule
   (n_checkpoints_planned, checkpoint_n) — so these are accepted as
   optional caller-supplied parameters. If omitted, or if fewer than 2
   checkpoint rows exist yet, sequential_risk is simply None in the report.
   This is a real, honest schema gap surfaced here, not papered over.

5. Segment analysis (Phase 6) is optional, driven by an explicit
   segment_columns parameter — one SegmentAnalysisResult per requested
   column, or None if no columns were requested. Uses the RAW (not CUPED)
   pooled point estimate as the Simpson's-paradox reference — matches the
   spec exactly (core/segments.py's segment_breakdown() reuses
   raw_ttest_ci() per segment, mirroring the pooled raw effect, not the
   CUPED-adjusted one). Each segment's persisted row uses
   "{segment_column}:{segment_value}" as its segment key (e.g.
   "device_type:mobile"), not just the bare value — this avoids result_id
   collisions if two different segment_columns happen to share a value
   string (e.g. a device_type and a region that happen to both be "US").

   segment_columns is also passed straight through to
   get_inference_data()'s own segment_columns parameter (Step 1's change)
   — if no segments were requested, we pass an empty list so the query
   doesn't fetch columns nobody asked for; if specific segments were
   requested, only those are fetched.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from sqlalchemy import Engine

from core.data_access import (
    get_inference_data,
    get_sequential_checkpoints,
    get_variant_counts,
    split_by_variant,
)
from core.inference import InferenceResult, cuped_adjust, raw_ttest_ci, variance_reduction_pct
from core.persistence import persist_inference_result
from core.segments import SegmentAnalysisResult, segment_breakdown
from core.sequential import SequentialCheckResult, sequential_check
from core.validity import SRMResult, mde_curve, srm_check


@dataclass(frozen=True)
class AnalysisReport:
    experiment_id: str
    metric_name: str
    srm: SRMResult
    achievable_mde: float | None       # None only if mde_curve() couldn't solve (see docstring)
    raw_effect: InferenceResult
    cuped_effect: InferenceResult | None
    cuped_variance_reduction_pct: float | None
    sequential_risk: SequentialCheckResult | None
    segments: tuple[SegmentAnalysisResult, ...] | None  # None if segment_columns not requested


def analyze_experiment(
    engine: Engine,
    experiment_id: str,
    metric_name: str = "conversion",
    use_cuped: bool = True,
    n_checkpoints_planned: int | None = None,
    checkpoint_n: int | None = None,
    segment_columns: list[str] | None = None,
    persist: bool = True,
) -> AnalysisReport:
    """
    Runs the full spec Section 4.4 sequence for one experiment and returns a
    single AnalysisReport. Persists the raw, (if computed) CUPED, and (if
    requested) per-segment effect estimates via
    core.persistence.persist_inference_result(), unless persist=False
    (useful for pure computation in tests, or a caller that wants to
    inspect a report before committing it).

    n_checkpoints_planned / checkpoint_n: pass BOTH to enable the sequential-
    peeking check against this experiment's real sequential_checkpoints
    rows. Omit either (or both) to skip it — see design decision 4 above for
    why this can't be inferred automatically from the schema as it stands.

    segment_columns: pass a list of column names from
    core.data_access.ALLOWED_SEGMENT_COLUMNS (e.g. ["device_type",
    "region"]) to run Benjamini-Hochberg-corrected, Simpson's-paradox-
    flagged segment analysis for each. Omit (None) to skip entirely
    (segments=None in the report, and no segment columns are even fetched
    from the database). A failure analyzing one requested column (e.g.
    every segment in that column had insufficient data) does not take down
    the others or the rest of the report — same fault-tolerance pattern as
    achievable_mde and CUPED above.

    Raises ValueError if the experiment has no seeded variants or no
    assignment data at all (propagated from get_variant_counts /
    get_inference_data) — this is intentional, matching this project's
    established "fail loudly" convention rather than returning a
    partially-populated report for a nonexistent or empty experiment.
    """
    observed_counts, expected_ratios = get_variant_counts(engine, experiment_id)
    srm_result = srm_check(observed_counts, expected_ratios)

    # Only fetch the segment columns actually requested — an empty list
    # when none are, per get_inference_data()'s own segment_columns
    # contract (Step 1).
    df = get_inference_data(
        engine, experiment_id, segment_columns=segment_columns or []
    )
    control_df, treatment_df = split_by_variant(df)

    raw_control = control_df["converted"].to_numpy(dtype=float)
    raw_treatment = treatment_df["converted"].to_numpy(dtype=float)
    raw_effect = raw_ttest_ci(raw_control, raw_treatment)

    if persist:
        persist_inference_result(engine, experiment_id, metric_name, raw_effect, srm_result)

    # --- Achievable MDE: context, not a gate. A failure here (e.g. observed
    # baseline_rate is exactly 0 or 1) should not take down the whole report.
    achievable_mde: float | None
    try:
        baseline_rate_observed = float(control_df["converted"].mean())
        n_per_variant = min(len(control_df), len(treatment_df))
        achievable_mde = mde_curve(baseline_rate_observed, [n_per_variant])[0]
    except ValueError:
        achievable_mde = None

    # --- CUPED: optional, and itself fault-tolerant for the same reason.
    cuped_effect: InferenceResult | None = None
    cuped_vr: float | None = None
    if use_cuped:
        try:
            y = df["converted"].to_numpy(dtype=float)
            x = df["pre_period_covariate"].to_numpy(dtype=float)
            adj = cuped_adjust(y, x)  # theta estimated on the POOLED sample — see decision 3

            df_adjusted = df.copy()
            df_adjusted["converted_cuped"] = adj.y_adjusted
            control_adj, treatment_adj = split_by_variant(df_adjusted)

            cuped_raw = raw_ttest_ci(
                control_adj["converted_cuped"].to_numpy(dtype=float),
                treatment_adj["converted_cuped"].to_numpy(dtype=float),
            )
            cuped_effect = replace(cuped_raw, method="cuped")
            cuped_vr = variance_reduction_pct(y, adj.y_adjusted)

            if persist:
                persist_inference_result(
                    engine, experiment_id, metric_name, cuped_effect, srm_result
                )
        except ValueError:
            cuped_effect = None
            cuped_vr = None

    # --- Sequential-peeking check: optional, see design decision 4.
    sequential_risk: SequentialCheckResult | None = None
    if n_checkpoints_planned is not None and checkpoint_n is not None:
        checkpoints = get_sequential_checkpoints(engine, experiment_id)
        if len(checkpoints) > 1:
            sequential_risk = sequential_check(
                checkpoints, n_checkpoints_planned, checkpoint_n
            )

    # --- Segment analysis (Phase 6): optional, see design decision 5.
    segments: tuple[SegmentAnalysisResult, ...] | None = None
    if segment_columns:
        segment_results = []
        for column in segment_columns:
            try:
                analysis = segment_breakdown(
                    df, column, pooled_point_estimate=raw_effect.point_estimate
                )
            except ValueError:
                # This ONE column's segment analysis failed (e.g. every
                # segment value had insufficient data) — skip it, don't
                # take down the rest of the report or the other columns.
                continue

            segment_results.append(analysis)

            if persist:
                for segment_result in analysis.segments:
                    persist_inference_result(
                        engine, experiment_id, metric_name, segment_result.inference,
                        srm_result, segment=f"{column}:{segment_result.segment_value}",
                    )

        segments = tuple(segment_results) if segment_results else None

    return AnalysisReport(
        experiment_id=experiment_id,
        metric_name=metric_name,
        srm=srm_result,
        achievable_mde=achievable_mde,
        raw_effect=raw_effect,
        cuped_effect=cuped_effect,
        cuped_variance_reduction_pct=cuped_vr,
        sequential_risk=sequential_risk,
        segments=segments,
    )
