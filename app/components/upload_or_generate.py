"""
app/components/upload_or_generate.py — Phase 8's data-input widget.

Two modes, both ending in the SAME place: a real, schema-compliant
in-memory SQLite database, populated with exactly one experiment, ready
for core.pipeline.analyze_experiment() to run against — unmodified, the
exact same function already proven correct in Phases 1-7. Neither mode
introduces a second, divergent analysis code path.

MODE 1 — "Generate simulated data": wraps db.seed.seed_database() (Phase
1/7, already tested) with Streamlit input widgets.

MODE 2 — "Upload CSV": SCOPE DECISION, stated directly rather than
silently assumed — the CSV must already be shaped like
core.data_access.get_inference_data()'s OUTPUT (one row per user), not an
arbitrary raw multi-table schema. Required columns: variant_id (ending in
"_control"/"_treatment"), converted (0/1). Optional: pre_period_covariate
(needed for CUPED; without it, CUPED is unavailable for this dataset),
device_type/region/existing_customer (segment columns). Reconstructing a
full multi-table schema from an arbitrary spreadsheet is a much bigger,
fragile guessing problem than the rest of this app — out of scope here,
same reasoning already applied to Phase 7's mode="ingest" deferral.
"""

from __future__ import annotations

import uuid

import pandas as pd
import streamlit as st
from sqlalchemy import Engine, text

from db.connection import create_memory_engine
from db.seed import seed_database

REQUIRED_CSV_COLUMNS = ("variant_id", "converted")
OPTIONAL_SEGMENT_COLUMNS = ("device_type", "region", "existing_customer")


def get_session_engine() -> Engine:
    """
    One private, in-memory SQLite database PER browser session, stored in
    st.session_state so it survives Streamlit's rerun-the-whole-script
    model.

    BUG FIX (review issue L2): this used to set DATABASE_URL and call
    reset_engine()/get_engine(), i.e. the PROCESS-WIDE singleton — so a
    second browser session's reset disposed the first session's in-memory
    database. Each session now owns its own engine object, and callers
    pass it explicitly instead of reaching for get_engine().

    (History: the original version of this function also forgot to set an
    in-memory URL at all and silently wrote to the persistent local.db,
    where stale users from earlier runs broke CUPED. That symptom's root
    cause — user_ids not scoped to an experiment — is fixed in the
    simulator; see review issue H1.)
    """
    if "engine" not in st.session_state:
        st.session_state["engine"] = create_memory_engine()
    return st.session_state["engine"]


def _generate_experiment_id() -> str:
    return f"exp_{uuid.uuid4().hex[:12]}"


def validate_upload(df: pd.DataFrame) -> list[str]:
    """
    Returns a list of human-readable problems with an uploaded DataFrame
    (empty list = valid). Pure, so it is testable without Streamlit.
    """
    missing = [c for c in REQUIRED_CSV_COLUMNS if c not in df.columns]
    if missing:
        return [f"Missing required column(s): {missing}"]

    errors = []
    variant_ids = df["variant_id"].astype(str)
    if not variant_ids.str.endswith(("_control", "_treatment")).all():
        errors.append(
            "Every variant_id value must end in '_control' or '_treatment' "
            "(e.g. 'myexp_control'). Found values that don't match this pattern."
        )
    converted = pd.to_numeric(df["converted"], errors="coerce")
    if converted.isna().any() or not converted.isin([0, 1]).all():
        errors.append("Every 'converted' value must be 0 or 1 (no blanks).")
    if "user_id" in df.columns and df["user_id"].duplicated().any():
        errors.append("user_id values must be unique — this format is one row per user.")
    if "pre_period_covariate" in df.columns and pd.to_numeric(
        df["pre_period_covariate"], errors="coerce"
    ).isna().any():
        errors.append(
            "pre_period_covariate must be numeric with no blanks (or omit the "
            "column entirely to run without CUPED)."
        )
    return errors


def ingest_upload(
    engine: Engine, df: pd.DataFrame, experiment_id: str, treatment_split: float = 0.5
) -> None:
    """
    Writes a validated upload (see validate_upload) into the schema as one
    experiment. Pure database code, testable without Streamlit.

    treatment_split is the INTENDED allocation, stored in variants.split_pct
    for the SRM check. It used to be hard-coded to 0.5, so a real 90/10
    design was falsely flagged as SRM (review issue M3).

    Stored user_ids are prefixed with experiment_id: the session DB
    persists across uploads, so a second upload reusing the same ids
    (generated or user-supplied) would otherwise collide on users.user_id
    or join to a previous upload's attributes (review issues H1/M3).
    """
    if not (0.0 < treatment_split < 1.0):
        raise ValueError(f"treatment_split must be in (0,1), got {treatment_split}")

    df = df.copy()
    if "user_id" not in df.columns:
        df["user_id"] = [f"{i:07d}" for i in range(len(df))]
    df["user_id"] = [f"u_{experiment_id}_{uid}" for uid in df["user_id"]]
    if "pre_period_covariate" not in df.columns:
        df["pre_period_covariate"] = None
    for col in OPTIONAL_SEGMENT_COLUMNS:
        if col not in df.columns:
            df[col] = None

    # Whatever prefix the uploaded file's variant_id values originally had
    # doesn't need to match experiment_id — every downstream query filters
    # by the assignments table's own experiment_id column.
    control_variant_id = f"{experiment_id}_control"
    treatment_variant_id = f"{experiment_id}_treatment"
    df["variant_id"] = [
        control_variant_id if str(v).endswith("_control") else treatment_variant_id
        for v in df["variant_id"]
    ]
    df["converted"] = pd.to_numeric(df["converted"]).astype(int)

    users = pd.DataFrame({
        "user_id": df["user_id"],
        "signup_date": "2025-01-01",
        "device_type": df["device_type"],
        "region": df["region"],
        "existing_customer": [
            bool(v) if pd.notna(v) else None for v in df["existing_customer"]
        ],
        "pre_period_covariate": pd.to_numeric(df["pre_period_covariate"], errors="coerce"),
    })
    assignments = pd.DataFrame({
        "user_id": df["user_id"],
        "experiment_id": experiment_id,
        "variant_id": df["variant_id"],
        "assigned_at": "2025-01-01",
    })
    converted_ids = df.loc[df["converted"] == 1, "user_id"].reset_index(drop=True)
    events = pd.DataFrame({
        "event_id": [f"ev_{experiment_id}_{i:07d}" for i in range(len(converted_ids))],
        "user_id": converted_ids,
        "experiment_id": experiment_id,
        "event_type": "conversion",
        "event_timestamp": "2025-01-02",
        "value": 1.0,
    })

    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO experiments (experiment_id, name, start_date, status) "
            "VALUES (:eid, 'Uploaded dataset', '2025-01-01', 'analyzed')"
        ), {"eid": experiment_id})
        conn.execute(text(
            "INSERT INTO variants (variant_id, experiment_id, name, split_pct) VALUES "
            "(:cvid, :eid, 'control', :cpct), (:tvid, :eid, 'treatment', :tpct)"
        ), {
            "cvid": control_variant_id, "tvid": treatment_variant_id, "eid": experiment_id,
            "cpct": 1 - treatment_split, "tpct": treatment_split,
        })
        users.to_sql("users", conn, if_exists="append", index=False)
        assignments.to_sql("assignments", conn, if_exists="append", index=False)
        events.to_sql("events", conn, if_exists="append", index=False)


def render_generate_data_form() -> str | None:
    """
    Renders the 'generate simulated data' form. Returns the experiment_id
    once data has been generated, or None if the form hasn't been
    submitted yet this run.
    """
    st.subheader("Generate simulated experiment data")
    st.caption(
        "Uses the same ExperimentSimulator already proven correct in "
        "Phases 1-6 — nothing here is new statistical logic, only the "
        "input widgets around it."
    )

    with st.form("generate_form"):
        col1, col2 = st.columns(2)
        with col1:
            n_users = st.number_input(
                "Number of users", min_value=100, max_value=200000, value=5000, step=100
            )
            baseline_rate = st.slider(
                "Baseline conversion rate", min_value=0.01, max_value=0.99, value=0.10, step=0.01
            )
            true_effect = st.slider(
                "True treatment effect", min_value=-0.20, max_value=0.20, value=0.02, step=0.005
            )
        with col2:
            covariate_correlation = st.slider(
                "Covariate correlation (for CUPED)", min_value=0.0, max_value=0.9,
                value=0.5, step=0.05,
            )
            seed = st.number_input("Random seed", min_value=0, value=42, step=1)
            simulate_srm_bug = st.checkbox("Simulate a broken randomization (SRM bug)", value=False)

        corrupted_split = st.slider(
            "Corrupted treatment split (if SRM bug enabled)",
            min_value=0.30, max_value=0.70, value=0.45, step=0.01,
        ) if simulate_srm_bug else None

        submitted = st.form_submit_button("Generate data")

    if not submitted:
        return None

    config = {"simulator": {
        "n_users": int(n_users),
        "baseline_rate": float(baseline_rate),
        "true_effect": float(true_effect),
        "covariate_correlation": float(covariate_correlation),
        "seed": int(seed),
        "corrupted_split": corrupted_split,
    }}

    # seed_database() derives its OWN experiment_id from the seed
    # (f"exp_seed{seed}"), per its Phase 1 contract — we use the same
    # deterministic naming convention to know which id it wrote.
    sim_experiment_id = f"exp_seed{int(seed)}"
    try:
        seed_database(config, engine=get_session_engine())
    except (ValueError, RuntimeError) as exc:
        st.error(f"Could not generate data for this configuration: {exc}")
        return None

    st.success(f"Generated experiment: {sim_experiment_id}")
    return sim_experiment_id


def render_upload_form() -> str | None:
    """
    Renders the 'upload a CSV' form. Returns the experiment_id once a
    valid file has been uploaded and ingested, or None otherwise.
    """
    st.subheader("Upload your own data")
    st.caption(
        f"CSV must have one row per user, with columns: "
        f"{', '.join(REQUIRED_CSV_COLUMNS)} (required), plus optionally "
        f"user_id, pre_period_covariate (needed for CUPED) and/or "
        f"{', '.join(OPTIONAL_SEGMENT_COLUMNS)} (for segment analysis). "
        f"variant_id values must end in '_control' or '_treatment'."
    )

    treatment_split = st.number_input(
        "Intended treatment allocation (used by the SRM check)",
        min_value=0.01, max_value=0.99, value=0.50, step=0.01,
    )
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file is None:
        return None

    try:
        df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read this file as a CSV: {exc}")
        return None

    errors = validate_upload(df)
    if errors:
        for error in errors:
            st.error(error)
        return None

    if "pre_period_covariate" not in df.columns:
        st.warning(
            "No pre_period_covariate column found — CUPED will not be "
            "available for this dataset."
        )

    experiment_id = _generate_experiment_id()
    ingest_upload(get_session_engine(), df, experiment_id, float(treatment_split))

    st.success(f"Ingested {len(df)} rows as experiment: {experiment_id}")
    return experiment_id


def render_data_input() -> str | None:
    """
    Top-level entry point for this component — renders a mode selector,
    then whichever form applies. Returns the active experiment_id once
    data is ready, or None if nothing has been submitted yet this run.
    """
    mode = st.radio("Data source", ["Generate simulated data", "Upload CSV"], horizontal=True)

    if mode == "Generate simulated data":
        return render_generate_data_form()
    return render_upload_form()
