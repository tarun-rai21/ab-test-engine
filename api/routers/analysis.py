"""
api/routers/analysis.py — POST /experiments/{id}/analyze and
GET /experiments/{id}/report.

KNOWN SPEC DEVIATION, documented here rather than silently worked around:
the original spec describes GET /report as served "from experiment_results
without recomputation." The current experiment_results schema (Phase 0)
only stores point_estimate, ci_lower, ci_upper, method, and trusted — it
does not store p_value, standard_error, degrees_freedom, n_control/
n_treatment, or CUPED's variance_reduction_pct. A full AnalysisReport
genuinely cannot be reconstructed from what's persisted today.

Rather than either (a) silently returning an incomplete report that LOOKS
complete, or (b) extending the schema and persistence layer as scope creep
inside what's supposed to be a thin API wrapper (Phase 7) around already-
tested Phase 3 code — GET /report here RECOMPUTES via
analyze_experiment(..., persist=False), giving a genuinely complete,
correct report at the cost of not matching the spec's literal wording.
Extending the schema to truly avoid recomputation remains a legitimate
future improvement, out of scope for this phase.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from sqlalchemy import text

from api.schemas import AnalysisReportSchema, AnalyzeRequest
from core.pipeline import analyze_experiment
from db.connection import get_engine

router = APIRouter(prefix="/experiments", tags=["analysis"])


def _require_experiment_exists(engine, experiment_id: str) -> None:
    with engine.connect() as conn:
        exists = conn.execute(
            text("SELECT 1 FROM experiments WHERE experiment_id = :eid"),
            {"eid": experiment_id},
        ).fetchone()
    if not exists:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id!r} not found")


@router.post("/{experiment_id}/analyze", response_model=AnalysisReportSchema)
def analyze(experiment_id: str, request: AnalyzeRequest) -> AnalysisReportSchema:
    """
    Runs the full analysis pipeline (core.pipeline.analyze_experiment) and
    PERSISTS the results — this is the "commit" step. See GET /report below
    for the read-only equivalent.
    """
    engine = get_engine()
    _require_experiment_exists(engine, experiment_id)

    try:
        report = analyze_experiment(
            engine,
            experiment_id,
            metric_name=request.metric,
            use_cuped=request.use_cuped,
            n_checkpoints_planned=request.n_checkpoints_planned,
            checkpoint_n=request.checkpoint_n,
            segment_columns=request.segment_columns,
            persist=True,
        )
    except ValueError as exc:
        # Matches this project's established "fail loudly" convention from
        # analyze_experiment() itself — surfaced here as a proper HTTP error
        # rather than an opaque 500.
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return AnalysisReportSchema.model_validate(report)


@router.get("/{experiment_id}/report", response_model=AnalysisReportSchema)
def get_report(
    experiment_id: str,
    metric: str = "conversion",
    use_cuped: bool = True,
) -> AnalysisReportSchema:
    """
    Returns a full analysis report WITHOUT persisting anything new
    (persist=False) — see this file's module docstring for why this
    recomputes rather than reading purely from experiment_results, and
    what a true no-recomputation implementation would require.

    segment_columns/sequential-check parameters are intentionally NOT
    exposed here (unlike POST /analyze) — this endpoint is meant for a
    quick "what does this experiment look like right now" read, not a
    full reconfigurable analysis; callers wanting segment or sequential
    detail should use POST /analyze directly.
    """
    engine = get_engine()
    _require_experiment_exists(engine, experiment_id)

    try:
        report = analyze_experiment(
            engine, experiment_id, metric_name=metric, use_cuped=use_cuped, persist=False
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return AnalysisReportSchema.model_validate(report)