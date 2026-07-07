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

from datetime import datetime, timezone

from sqlalchemy import Engine, text

from core.inference import InferenceResult
from core.validity import SRMResult


def persist_inference_result(
    engine: Engine,
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

    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM experiment_results WHERE result_id = :rid"),
            {"rid": result_id},
        )
        conn.execute(
            text(
                """
                INSERT INTO experiment_results
                    (result_id, experiment_id, metric_name, variant_id, segment,
                     point_estimate, ci_lower, ci_upper, trusted, method, computed_at)
                VALUES
                    (:result_id, :experiment_id, :metric_name, :variant_id, :segment,
                     :point_estimate, :ci_lower, :ci_upper, :trusted, :method, :computed_at)
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
                "trusted": int(trusted),
                "method": inference_result.method,
                "computed_at": computed_at,
            },
        )

    return result_id
