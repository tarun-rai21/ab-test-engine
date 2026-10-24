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

import json

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
    Returns one row per user: user_id, variant_id, variant_name (the
    variants table's name: 'control' or a treatment arm), pre_period_covariate,
    converted (0/1), n_conversions (count of conversion events) and revenue
    (sum of their values, 0 for non-converters), plus whichever segment
    columns were requested. converted feeds the conversion metric; revenue
    and n_conversions feed the revenue metrics (core.metrics).

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
        SELECT u.user_id, a.variant_id, v.name AS variant_name,
               u.pre_period_covariate{segment_select},
               CASE WHEN COUNT(e.event_id) > 0 THEN 1 ELSE 0 END AS converted,
               COUNT(e.event_id) AS n_conversions,
               COALESCE(SUM(e.value), 0.0) AS revenue
        FROM assignments a
        JOIN users u ON a.user_id = u.user_id
        JOIN variants v ON v.variant_id = a.variant_id
        LEFT JOIN events e
            ON e.user_id = a.user_id AND e.experiment_id = a.experiment_id
            AND e.event_type = 'conversion'
        WHERE a.experiment_id = :experiment_id
        GROUP BY u.user_id, a.variant_id, v.name, u.pre_period_covariate{segment_group_by}
        ORDER BY u.user_id
        """
    )
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"experiment_id": experiment_id})

    if df.empty:
        raise ValueError(f"No assignment data found for experiment_id={experiment_id!r}.")

    return df


def split_arms(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Splits get_inference_data()'s output into the control group and one
    DataFrame per treatment arm: (control_df, {arm_name: arm_df}), arms in
    sorted name order.

    Arms are identified by the variants table's NAME (variant_name column):
    "control" is the control group, every other name is a treatment arm —
    so A/B/n experiments (e.g. control, treatment, treatment_b) work. A
    DataFrame without variant_name (hand-built in tests) falls back to the
    variant_id suffix convention: ids ending "_control" are control, every
    other distinct variant_id is an arm.

    Raises if control or every arm is empty, rather than letting an empty
    group propagate into raw_ttest_ci(), which requires n>=2 per group.
    """
    if "variant_name" in df.columns:
        is_control = df["variant_name"] == "control"
        arm_key = df["variant_name"]
    else:
        is_control = df["variant_id"].str.endswith("_control")
        arm_key = df["variant_id"]

    control_df = df[is_control]
    treated = df[~is_control]
    arms = {
        str(name): group
        for name, group in sorted(treated.groupby(arm_key[~is_control]), key=lambda kv: kv[0])
    }
    if control_df.empty or not arms:
        raise ValueError(
            f"Split produced an empty group: control={len(control_df)}, "
            f"treatment arms={list(arms)}. A variant named 'control' (or a "
            f"variant_id ending in '_control') and at least one other variant "
            f"are required."
        )
    return control_df, arms


def split_by_variant(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Two-arm convenience wrapper around split_arms(): returns
    (control_df, treatment_df). Raises ValueError if the data has more
    than one treatment arm — use split_arms() for A/B/n experiments.
    """
    control_df, arms = split_arms(df)
    if len(arms) > 1:
        raise ValueError(
            f"Data has {len(arms)} treatment arms ({list(arms)}); use split_arms() "
            f"for multi-arm experiments."
        )
    return control_df, next(iter(arms.values()))


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


def get_planned_schedule(engine: Engine, experiment_id: str) -> tuple[int, int] | None:
    """
    Returns the experiment's stored (n_checkpoints_planned, checkpoint_n),
    or None if no schedule was stored (either column NULL) or the
    experiment doesn't exist. checkpoint_n is TOTAL users across arms.
    """
    query = text(
        """
        SELECT n_checkpoints_planned, checkpoint_n
        FROM experiments
        WHERE experiment_id = :experiment_id
        """
    )
    with engine.connect() as conn:
        row = conn.execute(query, {"experiment_id": experiment_id}).fetchone()

    if row is None or row.n_checkpoints_planned is None or row.checkpoint_n is None:
        return None
    return int(row.n_checkpoints_planned), int(row.checkpoint_n)


def save_planned_schedule(
    engine: Engine, experiment_id: str, n_checkpoints_planned: int, checkpoint_n: int
) -> None:
    """
    Stores the experiment's planned sequential-monitoring schedule, so later
    analyses can run the peeking check without the caller re-supplying it.
    Raises ValueError for non-positive values or an unknown experiment.
    """
    if n_checkpoints_planned < 1 or checkpoint_n < 1:
        raise ValueError(
            f"n_checkpoints_planned and checkpoint_n must be >= 1, got "
            f"{n_checkpoints_planned} and {checkpoint_n}"
        )
    query = text(
        """
        UPDATE experiments
        SET n_checkpoints_planned = :n_planned, checkpoint_n = :checkpoint_n
        WHERE experiment_id = :experiment_id
        """
    )
    with engine.begin() as conn:
        result = conn.execute(query, {
            "n_planned": n_checkpoints_planned,
            "checkpoint_n": checkpoint_n,
            "experiment_id": experiment_id,
        })
    if result.rowcount == 0:
        raise ValueError(f"No experiment found with experiment_id={experiment_id!r}.")


def get_stored_report(engine: Engine, experiment_id: str, metric_name: str) -> dict | None:
    """
    Returns the latest persisted report for (experiment, metric) as
    {"report": ..., "parameters": ..., "computed_at": ...}, or None if that
    metric has never been analysed with persist=True.
    """
    query = text(
        """
        SELECT report_json, parameters_json, computed_at
        FROM analysis_reports
        WHERE experiment_id = :experiment_id AND metric_name = :metric_name
        """
    )
    with engine.connect() as conn:
        row = conn.execute(
            query, {"experiment_id": experiment_id, "metric_name": metric_name}
        ).fetchone()
    if row is None:
        return None
    return {
        "report": json.loads(row.report_json),
        "parameters": json.loads(row.parameters_json),
        "computed_at": str(row.computed_at),
    }
