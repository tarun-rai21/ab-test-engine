"""
core/data_access.py — the ONLY place raw SQL lives in this codebase (per the
project's SQLAlchemy Core design decision: SQL stays visible here, not hidden
inside an ORM).

Every function here is a thin query + shape-into-plain-Python-types layer.
No statistical logic lives here — that stays in core/validity.py, core/inference.py,
etc. This separation means core/validity.py's correctness (proven in
tests/test_validity.py) is completely independent of whether the SQL wiring
is correct — a bug in one cannot masquerade as a bug in the other.
"""

from __future__ import annotations

from sqlalchemy import Engine, text


def get_variant_counts(engine: Engine, experiment_id: str) -> tuple[list[int], list[float]]:
    """
    Returns (observed_counts, expected_ratios), ordered consistently with each
    other by variant_id, ready to pass directly into core.validity.srm_check().

    Query joins variants (which holds the INTENDED split_pct) against a
    COUNT of assignments per variant, using LEFT JOIN so a variant with ZERO
    assignments still appears with observed_n=0 rather than silently
    vanishing from the result — a variant that received no traffic at all is
    itself a serious SRM-relevant signal, not a row to drop.

    Ordering: explicitly ORDER BY variant_id so the two returned lists are
    positionally aligned by construction, not by hoping dict/set iteration
    order happens to match — do not rely on insertion order or database
    default ordering, which is NOT guaranteed by SQL semantics in general.
    """
    query = text(
        """
        SELECT v.variant_id, v.split_pct AS expected_pct,
               COUNT(a.user_id) AS observed_n
        FROM variants v
        LEFT JOIN assignments a
            ON v.variant_id = a.variant_id AND v.experiment_id = a.experiment_id
        WHERE v.experiment_id = :experiment_id
        GROUP BY v.variant_id, v.split_pct
        ORDER BY v.variant_id
        """
    )

    with engine.connect() as conn:
        rows = conn.execute(query, {"experiment_id": experiment_id}).fetchall()

    if not rows:
        raise ValueError(
            f"No variants found for experiment_id={experiment_id!r}. "
            f"Either the experiment doesn't exist or variants weren't seeded."
        )

    observed_counts = [int(row.observed_n) for row in rows]
    expected_ratios = [float(row.expected_pct) for row in rows]

    return observed_counts, expected_ratios