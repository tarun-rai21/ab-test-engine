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

4. Sequential-peeking check is optional. The planned schedule
   (n_checkpoints_planned, checkpoint_n) is read from the experiments
   table when stored there (core.data_access.save_planned_schedule), and
   explicit caller-supplied values override it. With no schedule from
   either source, or no checkpoint rows yet, sequential_risk is None. A
   single checkpoint IS evaluated: the first look is the most noise-prone
   one, so skipping it would hide exactly the risk this check exists for.
   checkpoint_n and the stored cumulative_n are TOTAL users across arms.

5. Segment analysis (Phase 6) is optional, driven by an explicit
   segment_columns parameter — one SegmentAnalysisResult per requested
   column, or None if no columns were requested. Uses the RAW (not CUPED)
   pooled point estimate as the effect-reversal reference — matches the
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

import pandas as pd
from sqlalchemy import Engine

from core.data_access import (
    get_inference_data,
    get_planned_schedule,
    get_sequential_checkpoints,
    get_variant_counts,
    split_arms,
)
from core.inference import InferenceResult, cuped_adjust, raw_ttest_ci, variance_reduction_pct
from core.metrics import (
    MEAN_METRICS,
    estimate_effect,
    metric_outcome,
    validate_metric,
    winsorization_cap,
)
from core.multiple_testing import holm_adjust
from core.persistence import persist_inference_result, persist_srm_result
from core.segments import SegmentAnalysisResult, segment_breakdown
from core.sequential import SequentialCheckResult, sequential_check
from core.validity import SRMResult, achievable_mde_from_se, mde_curve, srm_check


@dataclass(frozen=True)
class ArmComparison:
    """One treatment arm vs control. Holm-adjusted p-values control the
    family-wise error rate across all arms (equal to the raw p-value when
    there is a single arm)."""
    treatment_arm: str
    raw_effect: InferenceResult
    raw_p_value_holm: float
    cuped_effect: InferenceResult | None
    cuped_p_value_holm: float | None


@dataclass(frozen=True)
class AnalysisReport:
    """
    raw_effect / cuped_effect are the PRIMARY comparison: the only arm in a
    two-arm experiment, or the first treatment arm by name in an A/B/n
    experiment. `comparisons` holds every arm vs control.
    """
    experiment_id: str
    metric_name: str
    srm: SRMResult
    achievable_mde: float | None       # None only if mde_curve() couldn't solve (see docstring)
    raw_effect: InferenceResult
    cuped_effect: InferenceResult | None
    cuped_variance_reduction_pct: float | None
    sequential_risk: SequentialCheckResult | None
    segments: tuple[SegmentAnalysisResult, ...] | None  # None if segment_columns not requested
    winsorization_cap: float | None = None  # revenue_per_user cap applied, if any
    comparisons: tuple[ArmComparison, ...] = ()  # every treatment arm vs control


def analyze_experiment(
    engine: Engine,
    experiment_id: str,
    metric_name: str = "conversion",
    use_cuped: bool = True,
    n_checkpoints_planned: int | None = None,
    checkpoint_n: int | None = None,
    segment_columns: list[str] | None = None,
    persist: bool = True,
    spending: str = "lan_demets",
    winsorize_quantile: float | None = None,
) -> AnalysisReport:
    """
    Runs the full spec Section 4.4 sequence for one experiment and returns a
    single AnalysisReport. Persists the raw, (if computed) CUPED, and (if
    requested) per-segment effect estimates via
    core.persistence.persist_inference_result(), unless persist=False
    (useful for pure computation in tests, or a caller that wants to
    inspect a report before committing it).

    n_checkpoints_planned / checkpoint_n: pass BOTH to run the sequential-
    peeking check with this schedule. Omit both to use the schedule stored
    on the experiment, if any (see design decision 4); passing only one is
    a ValueError.

    spending: "lan_demets" (default, information-based — valid for any
    look timing) or "obrien_fleming" (index-based, exact only for the
    planned equal spacing). See core.sequential.sequential_check().

    segment_columns: pass a list of column names from
    core.data_access.ALLOWED_SEGMENT_COLUMNS (e.g. ["device_type",
    "region"]) to run Benjamini-Hochberg-corrected, effect-reversal-
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
    validate_metric(metric_name)
    df = get_inference_data(
        engine, experiment_id, segment_columns=segment_columns or []
    )
    control_df, arms = split_arms(df)
    multi_arm = len(arms) > 1
    # Two-arm results keep their original persisted keys (variant "pooled");
    # with several arms each comparison is keyed by its arm's variant_id.
    arm_variant_ids = {
        arm: (arm_df["variant_id"].iloc[0] if multi_arm else None)
        for arm, arm_df in arms.items()
    }

    cap = winsorization_cap(df, metric_name, winsorize_quantile)
    raw_by_arm = {
        arm: estimate_effect(control_df, arm_df, metric_name, cap)
        for arm, arm_df in arms.items()
    }
    raw_holm = dict(zip(raw_by_arm, holm_adjust([r.p_value for r in raw_by_arm.values()])))

    if persist:
        persist_srm_result(engine, experiment_id, srm_result)
        for arm, result in raw_by_arm.items():
            persist_inference_result(
                engine, experiment_id, metric_name, result, srm_result,
                variant_id=arm_variant_ids[arm],
            )

    # --- Achievable MDE: context, not a gate. A failure here (e.g. observed
    # baseline_rate is exactly 0 or 1) should not take down the whole report.
    # Conversion uses the exact two-proportion inversion; continuous and
    # ratio metrics use the realized standard error. With several arms the
    # least-powered comparison (smallest arm / largest SE) is reported.
    achievable_mde: float | None
    try:
        if metric_name == "conversion":
            baseline_rate_observed = float(control_df["converted"].mean())
            n_per_variant = min(len(control_df), *(len(a) for a in arms.values()))
            achievable_mde = mde_curve(baseline_rate_observed, [n_per_variant])[0]
        else:
            achievable_mde = achievable_mde_from_se(
                max(r.standard_error for r in raw_by_arm.values())
            )
    except ValueError:
        achievable_mde = None

    # --- CUPED: optional, and itself fault-tolerant for the same reason.
    cuped_by_arm: dict[str, InferenceResult] = {}
    cuped_vr: float | None = None
    # Ratio metrics have no per-user additive outcome to adjust, so CUPED
    # applies to mean metrics only.
    if use_cuped and metric_name in MEAN_METRICS:
        try:
            y = metric_outcome(df, metric_name, cap)
            x = df["pre_period_covariate"].to_numpy(dtype=float)
            adj = cuped_adjust(y, x)  # theta estimated on the POOLED sample — see decision 3

            df_adjusted = df.copy()
            df_adjusted["outcome_cuped"] = adj.y_adjusted
            control_adj, arms_adj = split_arms(df_adjusted)
            for arm, arm_adj in arms_adj.items():
                cuped_raw = raw_ttest_ci(
                    control_adj["outcome_cuped"].to_numpy(dtype=float),
                    arm_adj["outcome_cuped"].to_numpy(dtype=float),
                )
                cuped_by_arm[arm] = replace(cuped_raw, method="cuped")
            cuped_vr = variance_reduction_pct(y, adj.y_adjusted)

            if persist:
                for arm, result in cuped_by_arm.items():
                    persist_inference_result(
                        engine, experiment_id, metric_name, result, srm_result,
                        variant_id=arm_variant_ids[arm],
                    )
        except ValueError:
            cuped_by_arm = {}
            cuped_vr = None
    cuped_holm = (
        dict(zip(cuped_by_arm, holm_adjust([r.p_value for r in cuped_by_arm.values()])))
        if cuped_by_arm else {}
    )

    # --- Sequential-peeking check: optional, see design decision 4.
    sequential_risk: SequentialCheckResult | None = None
    if (n_checkpoints_planned is None) != (checkpoint_n is None):
        raise ValueError(
            "Pass both n_checkpoints_planned and checkpoint_n, or neither (to use "
            "the schedule stored on the experiment)."
        )
    if n_checkpoints_planned is None:
        stored_schedule = get_planned_schedule(engine, experiment_id)
        if stored_schedule is not None:
            n_checkpoints_planned, checkpoint_n = stored_schedule
    if n_checkpoints_planned is not None and checkpoint_n is not None:
        checkpoints = get_sequential_checkpoints(engine, experiment_id)
        if checkpoints:
            sequential_risk = sequential_check(
                checkpoints, n_checkpoints_planned, checkpoint_n, spending=spending
            )

    # --- Segment analysis (Phase 6): optional, see design decision 5. Run
    # for every treatment arm against control.
    segments: tuple[SegmentAnalysisResult, ...] | None = None
    if segment_columns:
        segment_results = []
        for arm, arm_df in arms.items():
            arm_vs_control = pd.concat([control_df, arm_df])
            for column in segment_columns:
                try:
                    analysis = segment_breakdown(
                        arm_vs_control, column,
                        pooled_point_estimate=raw_by_arm[arm].point_estimate,
                        estimator=lambda c, t: estimate_effect(c, t, metric_name, cap),
                    )
                except ValueError:
                    # This ONE column's segment analysis failed (e.g. every
                    # segment value had insufficient data) — skip it, don't
                    # take down the rest of the report or the other columns.
                    continue

                analysis = replace(analysis, treatment_arm=arm)
                segment_results.append(analysis)

                if persist:
                    for segment_result in analysis.segments:
                        persist_inference_result(
                            engine, experiment_id, metric_name, segment_result.inference,
                            srm_result, segment=f"{column}:{segment_result.segment_value}",
                            variant_id=arm_variant_ids[arm],
                        )

        segments = tuple(segment_results) if segment_results else None

    comparisons = tuple(
        ArmComparison(
            treatment_arm=arm,
            raw_effect=raw_by_arm[arm],
            raw_p_value_holm=raw_holm[arm],
            cuped_effect=cuped_by_arm.get(arm),
            cuped_p_value_holm=cuped_holm.get(arm),
        )
        for arm in arms
    )
    primary = comparisons[0]

    return AnalysisReport(
        experiment_id=experiment_id,
        metric_name=metric_name,
        srm=srm_result,
        achievable_mde=achievable_mde,
        raw_effect=primary.raw_effect,
        cuped_effect=primary.cuped_effect,
        cuped_variance_reduction_pct=cuped_vr,
        sequential_risk=sequential_risk,
        segments=segments,
        winsorization_cap=cap,
        comparisons=comparisons,
    )
