"""
core/persistence.py — writes InferenceResult + SRMResult into experiment_results.

This is the ONLY place experiment_results.trusted is ever set. Per the
Option B architectural decision (Phase 2/3): core/inference.py stays pure
and SRM-unaware; core/validity.py stays pure and inference-unaware. This
file is the single point where the two independently-correct modules meet,
and it is designed so that meeting cannot happen incorrectly:

  - trusted is COMPUTED from srm_result.flagged inside this function, never
    accepted as a caller-supplied boolean — a caller cannot pass
    trusted=True alongside a flagged SRMResult, because trusted isn't a
    parameter at all.
  - Both inference_result and srm_result are REQUIRED (no defaults) —
    calling this function without an SRM check in hand is a TypeError/
    ValueError, not a silent gap.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone

from sqlalchemy import Connection, Engine, text

from core.inference import InferenceResult
from core.validity import SRMResult


@contextmanager
def _transaction(bind: Engine | Connection) -> Iterator[Connection]:
    """
    Every writer below accepts either an Engine (its own transaction, as
    before) or an open Connection (joins the caller's transaction). The
    pipeline passes one Connection for a whole analysis so that results,
    the SRM row and the stored report are committed together or not at
    all — a failure midway can no longer leave a half-written analysis.
    """
    if isinstance(bind, Connection):
        yield bind
    else:
        with bind.begin() as conn:
            yield conn

def clear_metric_results(
    engine: Engine | Connection, experiment_id: str, metric_name: str
) -> None:
    """
    Deletes every experiment_results row for (experiment, metric). The
    pipeline runs this first, in the same transaction as the new rows, so
    the table holds exactly the latest analysis of each metric — the same
    rule as analysis_reports. Overwriting by result_id alone left rows the
    new analysis did not produce (e.g. segment rows from an earlier run
    with segments, computed on data that has since been replaced).
    """
    with _transaction(engine) as conn:
        conn.execute(
            text(
                "DELETE FROM experiment_results "
                "WHERE experiment_id = :eid AND metric_name = :metric"
            ),
            {"eid": experiment_id, "metric": metric_name},
        )


def persist_inference_result(
    engine: Engine | Connection,
    experiment_id: str,
    metric_name: str,
    inference_result: InferenceResult,
    srm_result: SRMResult,
    variant_id: str | None = None,
    segment: str | None = None,
) -> str:
    """
    Writes one row to experiment_results. Returns the result_id written.

    result_id is DETERMINISTIC — a pure function of
    (experiment_id, metric_name, method, segment, variant_id) — matching this
    project's established convention (Phase 1: no uuid4 for reproducible
    entities). Re-running the same analysis slice OVERWRITES the prior row
    via delete-then-insert, the same idempotency pattern as db/seed.py, for
    the same reason: re-analysis regenerates a fixed artifact rather than
    accumulating history.

    trusted is cast to int(), not left as a Python bool, before insertion —
    this project already hit a real SQLite/Python-version type-coercion
    surprise once (Phase 1's coerce_for_sqlite, driven by Python 3.12+
    deprecating implicit bool/date adapters). Casting explicitly here avoids
    depending on whichever adapter behavior the local sqlite3 driver
    happens to have.
    """
    if inference_result is None or srm_result is None:
        raise ValueError(
            "Both inference_result and srm_result are REQUIRED. This function "
            "exists specifically to make persisting an inference result "
            "without an accompanying SRM check structurally impossible."
        )

    trusted = not srm_result.flagged

    segment_key = segment or "pooled"
    variant_key = variant_id or "pooled"
    result_id = (
        f"{experiment_id}_{metric_name}_{inference_result.method}_"
        f"{segment_key}_{variant_key}"
    )
    computed_at = datetime.now(timezone.utc).isoformat()

    with _transaction(engine) as conn:
        conn.execute(
            text("DELETE FROM experiment_results WHERE result_id = :rid"),
            {"rid": result_id},
        )
        conn.execute(
            text(
                """
                INSERT INTO experiment_results
                    (result_id, experiment_id, metric_name, variant_id, segment,
                     point_estimate, ci_lower, ci_upper, trusted, method, computed_at,
                     p_value, standard_error, degrees_freedom, n_control, n_treatment)
                VALUES
                    (:result_id, :experiment_id, :metric_name, :variant_id, :segment,
                     :point_estimate, :ci_lower, :ci_upper, :trusted, :method, :computed_at,
                     :p_value, :standard_error, :degrees_freedom, :n_control, :n_treatment)
                """
            ),
            {
                "result_id": result_id,
                "experiment_id": experiment_id,
                "metric_name": metric_name,
                "variant_id": variant_id,
                "segment": segment,
                "point_estimate": inference_result.point_estimate,
                "ci_lower": inference_result.ci_lower,
                "ci_upper": inference_result.ci_upper,
                "p_value": inference_result.p_value,
                "standard_error": inference_result.standard_error,
                "degrees_freedom": inference_result.degrees_freedom,
                "n_control": inference_result.n_control,
                "n_treatment": inference_result.n_treatment,
                "trusted": int(trusted),
                "method": inference_result.method,
                "computed_at": computed_at,
            },
        )

    return result_id


def persist_srm_result(
    engine: Engine | Connection, experiment_id: str, srm_result: SRMResult
) -> None:
    """
    Writes the experiment's latest SRM check to srm_checks (one row per
    experiment, replaced on re-analysis — same delete-then-insert
    idempotency as persist_inference_result). The table existed in the
    schema from Phase 0 but nothing ever wrote to it (review issue L4).
    """
    with _transaction(engine) as conn:
        conn.execute(
            text("DELETE FROM srm_checks WHERE experiment_id = :eid"),
            {"eid": experiment_id},
        )
        conn.execute(
            text(
                """
                INSERT INTO srm_checks (experiment_id, chi_sq_stat, p_value, flagged, checked_at)
                VALUES (:eid, :chi_sq, :p_value, :flagged, :checked_at)
                """
            ),
            {
                "eid": experiment_id,
                "chi_sq": srm_result.chi_sq_stat,
                "p_value": srm_result.p_value,
                "flagged": int(srm_result.flagged),
                "checked_at": datetime.now(timezone.utc).isoformat(),
            },
        )


def persist_report(
    engine: Engine | Connection,
    experiment_id: str,
    metric_name: str,
    report_payload: dict,
    parameters: dict,
) -> None:
    """
    Stores the complete report (already converted to plain JSON-compatible
    data, e.g. via dataclasses.asdict) and the analysis parameters that
    produced it, replacing any earlier report for the same (experiment,
    metric). GET /experiments/{id}/report serves this row directly, so a
    report can be read back without re-running the analysis.
    """
    with _transaction(engine) as conn:
        conn.execute(
            text(
                "DELETE FROM analysis_reports "
                "WHERE experiment_id = :eid AND metric_name = :metric"
            ),
            {"eid": experiment_id, "metric": metric_name},
        )
        conn.execute(
            text(
                """
                INSERT INTO analysis_reports
                    (experiment_id, metric_name, report_json, parameters_json, computed_at)
                VALUES (:eid, :metric, :report, :parameters, :computed_at)
                """
            ),
            {
                "eid": experiment_id,
                "metric": metric_name,
                "report": json.dumps(report_payload),
                "parameters": json.dumps(parameters),
                "computed_at": datetime.now(timezone.utc).isoformat(),
            },
        )
