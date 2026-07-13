"""
core/data_access.py — the ONLY file in this codebase permitted to contain raw
SQL, per the project's SQLAlchemy Core design decision (Phase 0): SQL stays
visible here, not hidden inside an ORM, since this project explicitly targets
demonstrating SQL fluency.

Every function here is a thin query + shape-into-plain-Python-types layer.
No statistical logic lives here — that stays in core/validity.py,
core/inference.py. This separation means a bug in one cannot masquerade as a
bug in the other; each has been proven correct independently.
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy import Engine, text

# The ONLY column names get_inference_data() will ever accept as a segment
# dimension. This exists specifically as a security boundary: SQL column
# names CANNOT be safely bound as query parameters the way values can
# (SQLAlchemy's :param binding only escapes VALUES, never identifiers), so
# any caller-supplied column name must be checked against this fixed list
# BEFORE it is ever interpolated into a query string. Extending segment
# analysis to a new dimension (e.g. "marketing_channel") means adding it
# here — and nowhere else needs to change.
ALLOWED_SEGMENT_COLUMNS = ("device_type", "region", "existing_customer")


def get_variant_counts(engine: Engine, experiment_id: str) -> tuple[list[int], list[float]]:
    """
    Returns (observed_counts, expected_ratios), ordered consistently with each
    other by variant_id, ready to pass directly into core.validity.srm_check().

    LEFT JOIN: a variant with ZERO assignments must still appear with
    observed_n=0 rather than silently vanishing — a variant that received no
    traffic at all is itself a serious SRM-relevant signal, not a row to drop.

    ORDER BY variant_id: the two returned lists are positionally aligned by
    construction, not by hoping iteration order happens to match — SQL does
    not guarantee row order without an explicit ORDER BY.
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


def get_inference_data(
    engine: Engine,
    experiment_id: str,
    segment_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Returns one row per user: user_id, variant_id, pre_period_covariate,
    converted, plus whichever segment columns were requested.

    segment_columns controls which optional segment dimensions (from
    ALLOWED_SEGMENT_COLUMNS) are included:
      - None (default): include ALL allow-listed columns — this preserves
        the function's original behavior exactly, so every existing caller
        that doesn't pass this argument sees no change at all.
      - [] (empty list): include NONE of them — used when no segment
        analysis was requested for this call, avoiding an unnecessary
        fetch of columns nobody asked for.
      - a specific list, e.g. ["device_type"]: include only those columns.

    SECURITY NOTE: segment_columns feeds into the SQL SELECT and GROUP BY
    clauses as column NAMES, not values — something SQLAlchemy's :param
    binding cannot protect, since bind parameters only escape values, never
    identifiers. Every requested name is validated against
    ALLOWED_SEGMENT_COLUMNS and rejected immediately if it isn't a member,
    BEFORE anything is interpolated into the query string. This is the same
    "validate untrusted input before use" discipline already applied
    elsewhere in this project (core.validity.srm_check's length/sum
    checks, ExperimentSimulator's covariate_correlation bounds check) —
    applied here specifically because dynamic column names are a genuine
    SQL-injection surface that a naive implementation would open up.

    NAMING/SCOPING NOTE: this function is deliberately scoped to expose ONLY
    pre_period_covariate as the COVARIATE column — never a post-treatment
    events-table value. core.inference.cuped_adjust() has an unenforceable
    precondition that its covariate input must be pre-treatment; this
    function is the structural (not just documented) defense against that —
    any future need for a DIFFERENT covariate must be a separately-named,
    separately-reviewed function, not a new column silently added here.
    The segment columns above are NOT covariates in that sense — they are
    static, pre-treatment SEGMENT dimensions (for Phase 6's
    core/segments.py), not values CUPED adjusts against, so their presence
    here does not weaken that guarantee.

    FAN-OUT DEFENSE: aggregation via GROUP BY + CASE WHEN COUNT(...)>0
    collapses any number of matching event rows per user into a single
    binary flag. The current simulator only ever writes one conversion event
    per converting user, so today there is no fan-out risk in practice — but
    the schema itself places no constraint preventing multiple event rows
    per user (e.g. a future revenue-event extension). This query is written
    to be correct regardless of event cardinality, not correct-by-luck given
    today's simulator behavior. A raw, non-aggregated JOIN would silently
    multiply a user's row for each matching event, corrupting every
    downstream count and effect estimate without any error being raised.
    """
    if segment_columns is None:
        columns_to_fetch = list(ALLOWED_SEGMENT_COLUMNS)
    else:
        invalid = [c for c in segment_columns if c not in ALLOWED_SEGMENT_COLUMNS]
        if invalid:
            raise ValueError(
                f"Invalid segment_columns {invalid} — must be a subset of "
                f"ALLOWED_SEGMENT_COLUMNS {ALLOWED_SEGMENT_COLUMNS}."
            )
        columns_to_fetch = list(segment_columns)

    # Safe to build this way: columns_to_fetch is now GUARANTEED to be a
    # subset of the fixed ALLOWED_SEGMENT_COLUMNS tuple above, never raw
    # caller input — the validation above is what makes this f-string safe.
    segment_select = "".join(f", u.{col}" for col in columns_to_fetch)
    segment_group_by = "".join(f", u.{col}" for col in columns_to_fetch)

    query = text(
        f"""
        SELECT u.user_id, a.variant_id, u.pre_period_covariate{segment_select},
               CASE WHEN COUNT(e.event_id) > 0 THEN 1 ELSE 0 END AS converted
        FROM assignments a
        JOIN users u ON a.user_id = u.user_id
        LEFT JOIN events e
            ON e.user_id = a.user_id AND e.experiment_id = a.experiment_id
            AND e.event_type = 'conversion'
        WHERE a.experiment_id = :experiment_id
        GROUP BY u.user_id, a.variant_id, u.pre_period_covariate{segment_group_by}
        ORDER BY u.user_id
        """
    )
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"experiment_id": experiment_id})

    if df.empty:
        raise ValueError(f"No assignment data found for experiment_id={experiment_id!r}.")

    return df


def split_by_variant(
    df: pd.DataFrame,
    control_suffix: str = "_control",
    treatment_suffix: str = "_treatment",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits get_inference_data()'s output into (control_df, treatment_df) by
    variant_id suffix — matches the naming convention fixed in Phase 1's
    ExperimentSimulator._build_assignments() (f"{experiment_id}_control" /
    f"{experiment_id}_treatment").

    Raises if either group is empty rather than letting an empty DataFrame
    propagate silently into raw_ttest_ci(), which requires n>=2 per group —
    failing here gives a clear, specific error message instead of a confusing
    downstream ValueError with no context about WHY the group was empty.
    """
    control_df = df[df["variant_id"].str.endswith(control_suffix)]
    treatment_df = df[df["variant_id"].str.endswith(treatment_suffix)]

    if control_df.empty or treatment_df.empty:
        raise ValueError(
            f"Split produced an empty group: control={len(control_df)}, "
            f"treatment={len(treatment_df)}. Check variant_id naming convention "
            f"matches '{{experiment_id}}{control_suffix}' / '{{experiment_id}}{treatment_suffix}'."
        )
    return control_df, treatment_df


def get_sequential_checkpoints(engine: Engine, experiment_id: str) -> list[dict]:
    """
    Returns every stored checkpoint row for this experiment, shaped exactly
    as core.sequential.sequential_check() expects: a list of dicts, each
    with 'cumulative_n' and 'p_value_at_check' keys.

    This is the query nothing in the codebase used before Phase 5's
    orchestration layer (core/pipeline.py) — sequential_check() had only
    ever been exercised against hand-built dicts in tests, never a real
    sequential_checkpoints row, a gap explicitly flagged (and left open) in
    Phase 5's own documentation.

    Returns an EMPTY list, not an error, when no checkpoints exist yet.
    Unlike get_variant_counts/get_inference_data — where zero rows signals a
    missing or misconfigured experiment — zero sequential_checkpoints rows
    is the NORMAL state for most experiments (they simply haven't been
    peeked at yet). Raising here would make the common case an error.
    Ordered by cumulative_n for readability only; sequential_check() never
    trusts list order regardless, since it always finds the latest
    checkpoint explicitly by cumulative_n.
    """
    query = text(
        """
        SELECT cumulative_n, p_value_at_check
        FROM sequential_checkpoints
        WHERE experiment_id = :experiment_id
        ORDER BY cumulative_n
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(query, {"experiment_id": experiment_id}).fetchall()

    return [
        {"cumulative_n": int(row.cumulative_n), "p_value_at_check": float(row.p_value_at_check)}
        for row in rows
    ]