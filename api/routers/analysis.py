"""
api/routers/analysis.py — POST /experiments/{id}/analyze and
GET /experiments/{id}/report.

POST /analyze runs the full pipeline and PERSISTS it: per-comparison rows
in experiment_results (including p-value, standard error, degrees of
freedom and arm sizes) and the complete report, with the parameters that
produced it, in analysis_reports.

GET /report serves that stored report WITHOUT recomputation, as the
original spec describes. It used to re-run the analysis because the schema
could not hold a full report; it returns 404 for a metric that has never
been analysed, rather than silently computing a fresh one with default
settings that may differ from the analysis the caller ran.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Response
from sqlalchemy import text

from api.schemas import AnalysisReportSchema, AnalyzeRequest
from core.data_access import get_stored_report
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
            spending=request.spending,
            winsorize_quantile=request.winsorize_quantile,
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
    response: Response,
    metric: str = "conversion",
) -> AnalysisReportSchema:
    """
    Returns the report stored by the latest POST /analyze for this metric,
    exactly as that call returned it. Nothing is recomputed or written.
    The X-Report-Computed-At header says when the analysis ran.
    """
    engine = get_engine()
    _require_experiment_exists(engine, experiment_id)

    stored = get_stored_report(engine, experiment_id, metric)
    if stored is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No stored report for experiment {experiment_id!r}, metric {metric!r}. "
                f"Run POST /experiments/{experiment_id}/analyze first."
            ),
        )
    response.headers["X-Report-Computed-At"] = stored["computed_at"]
    return AnalysisReportSchema.model_validate(stored["report"])
