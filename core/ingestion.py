"""
core/ingestion.py — turning real data into this schema.

Real data arrives as one row per user in whatever shape the source system
produced: its own column names, variant labels such as "A"/"B", conversion
flags as yes/no or true/false. Ingestion has three steps, shared by the
Streamlit upload and the API's POST /experiments/{id}/ingest:

  1. to_canonical(): map the source columns onto the canonical ones —
     variant_name ('control' or a treatment arm name), converted (0/1) and
     the optional user_id, revenue, pre_period_covariate and segment
     columns. Rows whose variant is mapped to None are dropped (e.g. QA
     traffic). Without an explicit mapping, the original convention is
     used: a 'variant_id' column ending in '_control' / '_treatment...'.
  2. validate_canonical(): human-readable problems, or [] if valid.
  3. ingest_canonical(): write users, assignments and events for an
     experiment whose variants already exist (create_experiment_with_variants()
     makes them for a fresh upload), replacing any earlier ingested data.
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy import Engine, text

SEGMENT_COLUMNS = ("device_type", "region", "existing_customer")
OPTIONAL_COLUMNS = ("user_id", "revenue", "pre_period_covariate", *SEGMENT_COLUMNS)
MAPPABLE_ROLES = ("variant", "converted", *OPTIONAL_COLUMNS)

_TRUE = {"1", "1.0", "true", "t", "yes", "y", "converted"}
_FALSE = {"0", "0.0", "false", "f", "no", "n", ""}


def arm_name(variant_id: str) -> str | None:
    """
    Variant name implied by the default variant_id convention: '..._control'
    -> 'control'; otherwise the text from the LAST '_treatment' onward, so
    'exp_treatment' -> 'treatment' and 'exp_treatment_b' -> 'treatment_b'.
    None if the id matches neither pattern.
    """
    if variant_id.endswith("_control"):
        return "control"
    position = variant_id.rfind("_treatment")
    return variant_id[position + 1:] if position != -1 else None


def parse_binary(values: pd.Series) -> pd.Series:
    """
    Reads a conversion flag written as 0/1, true/false, yes/no (any case).
    Unrecognised values (including blanks in a numeric column) become NaN,
    which validate_canonical() then reports.
    """
    if pd.api.types.is_bool_dtype(values):
        return values.astype(int)
    if pd.api.types.is_numeric_dtype(values):
        return values.where(values.isin([0, 1]))
    lowered = values.astype(str).str.strip().str.lower()
    return lowered.map(lambda v: 1 if v in _TRUE else 0 if v in _FALSE else float("nan"))


def to_canonical(
    raw: pd.DataFrame,
    columns: dict[str, str] | None = None,
    variant_values: dict[str, str | None] | None = None,
) -> pd.DataFrame:
    """
    Maps a source DataFrame onto the canonical columns.

    columns: {role: source column}, roles from MAPPABLE_ROLES. 'variant' is
    required when mapping; 'converted' may be omitted if 'revenue' is
    mapped (a user converted if their revenue is above zero).
    variant_values: {raw variant value: 'control' | arm name | None}; None
    drops those rows. Values missing from the mapping are left unmapped and
    reported by validate_canonical().

    With columns=None the default convention applies (variant_id suffixes
    and canonical column names already in the file).
    """
    if columns is None:
        df = raw.copy()
        if "variant_id" not in df.columns:
            raise ValueError(
                "No column mapping given and no 'variant_id' column found — map the "
                "variant column explicitly."
            )
        df["variant_name"] = [arm_name(str(v)) for v in df["variant_id"]]
        if "converted" in df.columns:
            df["converted"] = parse_binary(df["converted"])
        return df.drop(columns="variant_id")

    unknown = set(columns) - set(MAPPABLE_ROLES)
    if unknown:
        raise ValueError(f"Unknown column roles {sorted(unknown)}; use {MAPPABLE_ROLES}.")
    missing_sources = [src for src in columns.values() if src not in raw.columns]
    if missing_sources:
        raise ValueError(f"Mapped columns not found in the data: {missing_sources}")
    if "variant" not in columns:
        raise ValueError("A 'variant' column must be mapped.")
    if "converted" not in columns and "revenue" not in columns:
        raise ValueError("Map a 'converted' column, or a 'revenue' column to derive it from.")

    variants = raw[columns["variant"]].astype(str).str.strip()
    if variant_values is not None:
        mapped = variants.map(lambda v: variant_values.get(v, "__unmapped__"))
        keep = mapped.notna()
        raw, mapped = raw[keep], mapped[keep]
        variant_name = mapped.where(mapped != "__unmapped__")
    else:
        variant_name = variants

    df = pd.DataFrame({"variant_name": variant_name.values}, index=raw.index)
    for role in OPTIONAL_COLUMNS:
        if role in columns:
            df[role] = raw[columns[role]].values
    if "converted" in columns:
        df["converted"] = parse_binary(raw[columns["converted"]]).values
    else:
        df["converted"] = (pd.to_numeric(df["revenue"], errors="coerce") > 0).astype(int)
    return df.reset_index(drop=True)


def validate_canonical(df: pd.DataFrame) -> list[str]:
    """Human-readable problems with canonical data ([] = valid)."""
    if "converted" not in df.columns:
        return ["Missing required column: converted."]
    errors = []
    names = df["variant_name"]
    if names.isna().any():
        errors.append(
            "Some rows have no recognised variant — map every variant value (or set it "
            "to be ignored), or use variant_id values ending in '_control' / containing "
            "'_treatment'."
        )
    named = set(names.dropna())
    if "control" not in named or named == {"control"}:
        errors.append("The data needs a control group and at least one treatment arm.")
    converted = pd.to_numeric(df["converted"], errors="coerce")
    if converted.isna().any() or not converted.isin([0, 1]).all():
        errors.append("Every 'converted' value must be 0/1, true/false or yes/no (no blanks).")
    if "user_id" in df.columns and df["user_id"].duplicated().any():
        errors.append("user_id values must be unique — this format is one row per user.")
    if "revenue" in df.columns:
        revenue = pd.to_numeric(df["revenue"], errors="coerce")
        if revenue.isna().any() or (revenue < 0).any():
            errors.append("revenue must be numeric, non-negative, with no blanks.")
        elif ((revenue > 0) & (converted != 1)).any():
            errors.append("Rows with revenue > 0 must have converted = 1.")
    # Unrecognised values would otherwise become NaN, and bool(NaN) is True:
    # "unknown" would be silently stored as an existing customer.
    if "existing_customer" in df.columns and parse_binary(
        df["existing_customer"].dropna()
    ).isna().any():
        errors.append(
            "existing_customer must be 0/1, true/false or yes/no (blanks are allowed)."
        )
    if "pre_period_covariate" in df.columns and pd.to_numeric(
        df["pre_period_covariate"], errors="coerce"
    ).isna().any():
        errors.append(
            "pre_period_covariate must be numeric with no blanks (or leave it unmapped "
            "to run without CUPED)."
        )
    return errors


def create_experiment_with_variants(
    engine: Engine,
    experiment_id: str,
    arm_names: list[str],
    treatment_split: float = 0.5,
    name: str = "Uploaded dataset",
) -> None:
    """
    Creates an experiment with a control and the given arms. treatment_split
    is the INTENDED share of users across all arms (split equally between
    them); control gets the rest. It becomes variants.split_pct, which the
    SRM check compares against.
    """
    if not (0.0 < treatment_split < 1.0):
        raise ValueError(f"treatment_split must be in (0,1), got {treatment_split}")
    arms = sorted(set(arm_names) - {"control"})
    if not arms:
        raise ValueError("At least one treatment arm is required.")
    variants = pd.DataFrame(
        [{"variant_id": f"{experiment_id}_control", "experiment_id": experiment_id,
          "name": "control", "split_pct": 1 - treatment_split}]
        + [{"variant_id": f"{experiment_id}_{arm}", "experiment_id": experiment_id,
            "name": arm, "split_pct": treatment_split / len(arms)} for arm in arms]
    )
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO experiments (experiment_id, name, start_date, status) "
            "VALUES (:eid, :name, '2025-01-01', 'analyzed')"
        ), {"eid": experiment_id, "name": name})
        variants.to_sql("variants", conn, if_exists="append", index=False)


def ingest_canonical(engine: Engine, experiment_id: str, df: pd.DataFrame) -> int:
    """
    Writes validated canonical data (see validate_canonical) for an existing
    experiment: users, assignments and one conversion event per converter
    (value = the user's revenue, or 1.0 without a revenue column). Returns
    the number of users written.

    Every variant name must be one of the experiment's variants. Earlier
    ingested data for this experiment is replaced, in the same transaction.
    Stored user_ids are prefixed with experiment_id, so separate datasets
    in one database never share user rows.
    """
    errors = validate_canonical(df)
    if errors:
        raise ValueError(" ".join(errors))

    with engine.connect() as conn:
        variant_ids = dict(conn.execute(
            text("SELECT name, variant_id FROM variants WHERE experiment_id = :eid"),
            {"eid": experiment_id},
        ).fetchall())
    unknown = sorted(set(df["variant_name"]) - set(variant_ids))
    if unknown:
        raise ValueError(
            f"Variants {unknown} are not defined for experiment {experiment_id!r} "
            f"(defined: {sorted(variant_ids)})."
        )

    df = df.copy()
    if "user_id" not in df.columns:
        df["user_id"] = [f"{i:07d}" for i in range(len(df))]
    df["user_id"] = [f"u_{experiment_id}_{uid}" for uid in df["user_id"]]
    for col in ("pre_period_covariate", *SEGMENT_COLUMNS):
        if col not in df.columns:
            df[col] = None
    df["converted"] = pd.to_numeric(df["converted"]).astype(int)
    df["revenue"] = (
        pd.to_numeric(df["revenue"]) if "revenue" in df.columns
        else df["converted"].astype(float)
    )

    users = pd.DataFrame({
        "user_id": df["user_id"],
        "signup_date": "2025-01-01",
        "device_type": df["device_type"],
        "region": df["region"],
        "existing_customer": [
            bool(parse_binary(pd.Series([v])).iloc[0]) if pd.notna(v) else None
            for v in df["existing_customer"]
        ],
        "pre_period_covariate": pd.to_numeric(df["pre_period_covariate"], errors="coerce"),
    })
    assignments = pd.DataFrame({
        "user_id": df["user_id"],
        "experiment_id": experiment_id,
        "variant_id": df["variant_name"].map(variant_ids),
        "assigned_at": "2025-01-01",
    })
    converters = df.loc[df["converted"] == 1].reset_index(drop=True)
    events = pd.DataFrame({
        "event_id": [f"ev_{experiment_id}_{i:07d}" for i in range(len(converters))],
        "user_id": converters["user_id"],
        "experiment_id": experiment_id,
        "event_type": "conversion",
        "event_timestamp": "2025-01-02",
        "value": converters["revenue"].astype(float),
    })

    with engine.begin() as conn:
        for statement in (
            "DELETE FROM users WHERE user_id IN "
            "(SELECT user_id FROM assignments WHERE experiment_id = :eid)",
            "DELETE FROM events WHERE experiment_id = :eid",
            "DELETE FROM assignments WHERE experiment_id = :eid",
        ):
            conn.execute(text(statement), {"eid": experiment_id})
        users.to_sql("users", conn, if_exists="append", index=False)
        assignments.to_sql("assignments", conn, if_exists="append", index=False)
        events.to_sql("events", conn, if_exists="append", index=False)
    return len(df)
