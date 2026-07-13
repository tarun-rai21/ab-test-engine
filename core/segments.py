"""
core/segments.py — Phase 6: per-segment effect estimation, Benjamini-
Hochberg multiple-testing correction, and Simpson's-paradox sign-flip
detection.

ARCHITECTURE: this module reuses core.inference.raw_ttest_ci() directly,
called once per segment slice of the data — the SAME already-validated
statistical primitive used for the pooled effect estimate, not a new
statistical method. The two genuinely new pieces here are
benjamini_hochberg() and simpsons_paradox_flag(), both pure, closed-form,
and independently testable with hand-verifiable inputs.

This module takes a DataFrame directly (shaped like
core.data_access.get_inference_data()'s output) rather than an engine +
experiment_id — matching the "core modules stay pure, only
core/pipeline.py talks to the database" pattern already established by
core/inference.py, core/validity.py, and core/sequential.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from core.data_access import split_by_variant
from core.inference import InferenceResult, raw_ttest_ci

DEFAULT_FDR = 0.05


def benjamini_hochberg(p_values: list[float], fdr: float = DEFAULT_FDR) -> list[bool]:
    """
    Benjamini-Hochberg procedure for controlling the false discovery rate
    across m simultaneous hypothesis tests (here: m segment-level t-tests).

    Given p_values sorted ascending as p_(1) <= ... <= p_(m), find the
    LARGEST rank k such that p_(k) <= (k/m) * fdr. Reject (mark significant)
    every hypothesis with rank <= k — not merely the ones individually
    satisfying the inequality at their own rank, which is the single most
    common implementation mistake with this procedure.

    Chosen over a Bonferroni correction (per spec Section 7.5): Bonferroni
    controls family-wise error rate (probability of ANY false positive),
    which is unnecessarily conservative for exploratory segment analysis
    and rewards not looking for effects over correctly finding them; BH
    controls the expected PROPORTION of false discoveries among rejections,
    the standard choice for this kind of exploratory analysis in industry
    experimentation platforms.

    Returns a list of booleans in the SAME ORDER as the input p_values (not
    sorted order) — callers can zip this directly against their original
    segment list without tracking a separate index mapping.
    """
    if not p_values:
        raise ValueError("p_values must be non-empty")
    if not (0.0 < fdr < 1.0):
        raise ValueError(f"fdr must be in (0,1), got {fdr}")
    if any(not (0.0 <= p <= 1.0) for p in p_values):
        raise ValueError(f"All p_values must be in [0,1], got {p_values}")

    m = len(p_values)
    # (original_index, p_value), sorted ascending by p_value
    indexed = sorted(enumerate(p_values), key=lambda pair: pair[1])

    threshold_rank = 0
    for rank, (_, p) in enumerate(indexed, start=1):
        if p <= (rank / m) * fdr:
            threshold_rank = rank  # keep the LARGEST rank satisfying the inequality

    significant = [False] * m
    for rank, (original_index, _p) in enumerate(indexed, start=1):
        if rank <= threshold_rank:
            significant[original_index] = True

    return significant


def simpsons_paradox_flag(pooled_estimate: float, segment_estimate: float) -> bool:
    """
    Flags whether a segment's point estimate has the OPPOSITE sign from the
    pooled estimate — the hallmark of Simpson's paradox: an aggregate
    effect that masks a genuinely different (or reversed) effect within a
    subgroup.

    CONVENTION: if either value is exactly 0.0, this returns False (no
    flag). A sign comparison against a true zero is not a meaningful
    paradox signal — treating 0.0 as arbitrarily "positive" or "negative"
    would produce a worse default than simply declining to flag a
    genuinely null comparison.

    Explicitly cast to a native Python bool before returning: if either
    input arrives as a numpy scalar (e.g. from a pandas .mean()), the
    (a > b) != (c > d) comparison naturally produces numpy.bool_, not
    Python's bool. numpy.bool_(True) is NOT the same object as Python's
    True, so an `is True` check downstream would silently fail — the same
    type-purity issue already documented and cast around in this project's
    core/persistence.py (int(trusted)) and db/seed.py (coerce_for_sqlite).
    """
    if pooled_estimate == 0.0 or segment_estimate == 0.0:
        return False
    return bool((pooled_estimate > 0) != (segment_estimate > 0))


@dataclass(frozen=True)
class SegmentResult:
    segment_value: str
    inference: InferenceResult
    bh_significant: bool
    simpsons_flag: bool


@dataclass(frozen=True)
class SegmentAnalysisResult:
    segment_column: str
    pooled_point_estimate: float
    segments: tuple[SegmentResult, ...]
    excluded_segments: tuple[str, ...]  # segment values skipped for insufficient data


def segment_breakdown(
    df: pd.DataFrame,
    segment_column: str,
    pooled_point_estimate: float,
    fdr: float = DEFAULT_FDR,
) -> SegmentAnalysisResult:
    """
    Computes a per-segment effect estimate for every distinct value of
    segment_column, applies Benjamini-Hochberg across all segment-level
    p-values tested, and flags any segment whose sign disagrees with the
    pooled estimate.

    df must be shaped like core.data_access.get_inference_data()'s output —
    at minimum 'variant_id', 'converted', and segment_column present.

    SEGMENTS WITH INSUFFICIENT DATA ARE EXCLUDED, NOT FATAL: a segment
    whose control or treatment slice has fewer than 2 observations (the
    hard minimum raw_ttest_ci() requires) is skipped and recorded in
    excluded_segments, rather than crashing the entire analysis. A rare
    segment value having too little data is a normal, expected situation
    (e.g. a device_type with very few users) — the same "don't treat a
    normal edge case as fatal" philosophy already applied to
    get_sequential_checkpoints() returning an empty list rather than
    raising when an experiment simply hasn't been peeked at yet.

    Raises ValueError only if segment_column doesn't exist, has no non-null
    values, or if EVERY segment ends up excluded (meaning no segment-level
    analysis was possible at all) — that IS a real failure worth surfacing
    loudly, unlike a single excluded segment among several.
    """
    if segment_column not in df.columns:
        raise ValueError(
            f"segment_column {segment_column!r} not found in DataFrame columns: "
            f"{list(df.columns)}"
        )

    segment_values = sorted(df[segment_column].dropna().unique().tolist())
    if not segment_values:
        raise ValueError(f"No non-null values found in segment_column {segment_column!r}")

    per_segment: list[tuple[str, InferenceResult]] = []
    excluded: list[str] = []

    for value in segment_values:
        segment_df = df[df[segment_column] == value]
        try:
            control_df, treatment_df = split_by_variant(segment_df)
        except ValueError:
            excluded.append(str(value))
            continue

        try:
            inference = raw_ttest_ci(
                control_df["converted"].to_numpy(dtype=float),
                treatment_df["converted"].to_numpy(dtype=float),
            )
        except ValueError:
            excluded.append(str(value))
            continue

        per_segment.append((str(value), inference))

    if not per_segment:
        raise ValueError(
            f"No segment in {segment_column!r} had sufficient data for inference "
            f"(all {len(segment_values)} segment(s) excluded: {excluded}). "
            f"Cannot perform segment analysis on this column."
        )

    p_values = [inference.p_value for _, inference in per_segment]
    bh_flags = benjamini_hochberg(p_values, fdr=fdr)

    segments = tuple(
        SegmentResult(
            segment_value=value,
            inference=inference,
            bh_significant=bh_flags[i],
            simpsons_flag=simpsons_paradox_flag(pooled_point_estimate, inference.point_estimate),
        )
        for i, (value, inference) in enumerate(per_segment)
    )

    return SegmentAnalysisResult(
        segment_column=segment_column,
        pooled_point_estimate=pooled_point_estimate,
        segments=segments,
        excluded_segments=tuple(excluded),
    )