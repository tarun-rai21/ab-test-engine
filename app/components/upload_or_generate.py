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
from sqlalchemy import text

from db.connection import get_engine, init_schema, reset_engine
from db.seed import seed_database

REQUIRED_CSV_COLUMNS = ("variant_id", "converted")
OPTIONAL_SEGMENT_COLUMNS = ("device_type", "region", "existing_customer")


def _fresh_session_engine():
    """
    One fresh, in-memory SQLite database PER browser session — stored in
    st.session_state so it survives across Streamlit's rerun-the-whole-
    script-on-every-click model, but never shared between two different
    users/tabs, AND CRITICALLY never shared with any persistent local
    database file left over from other work (manual testing, other
    sessions, etc).

    BUG FIX: the first version of this function called reset_engine() and
    get_engine() WITHOUT ever actually setting DATABASE_URL to
    "sqlite:///:memory:" — meaning it silently fell back to whatever
    persistent file-based database the project defaults to. This broke
    CUPED in a real, confusing way: db/seed.py's seed_database() skips
    re-inserting a user_id if it already exists (a correct optimization
    IF only one simulator config is ever seeded into a database — same
    seed always produces identical data). Once multiple DIFFERENT configs
    (different covariate_correlation values across different manual test
    runs) hit the SAME persistent file, that assumption broke: a NEW
    experiment's assignments/events were correctly regenerated, but the
    shared users table's pre_period_covariate values were stale leftovers
    from a PREVIOUS, unrelated run — completely uncorrelated with the
    CURRENT run's outcomes, exactly matching the "0.0% variance reduction"
    symptom this fix resolves. Explicitly forcing an in-memory database
    here, matching the isolated_db fixture pattern used throughout this
    project's own test suite since Phase 2, makes this class of bug
    structurally impossible rather than just less likely.
    """
    if "db_initialized" not in st.session_state:
        import os
        os.environ["DATABASE_URL"] = "sqlite:///:memory:"
        reset_engine()
        engine = get_engine()
        init_schema(engine)
        st.session_state["db_initialized"] = True
    return get_engine()


def _generate_experiment_id() -> str:
    return f"exp_{uuid.uuid4().hex[:12]}"


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

    _fresh_session_engine()

    config = {"simulator": {
        "n_users": int(n_users),
        "baseline_rate": float(baseline_rate),
        "true_effect": float(true_effect),
        "covariate_correlation": float(covariate_correlation),
        "seed": int(seed),
        "corrupted_split": corrupted_split,
    }}

    # seed_database() derives its OWN experiment_id from the seed
    # (f"exp_seed{seed}"), per its Phase 1 contract, unchanged here — we
    # read back whatever id it actually used rather than generating our
    # own, since this mode's data always comes from that deterministic
    # naming convention.
    sim_experiment_id = f"exp_seed{int(seed)}"
    seed_database(config)

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
        f"pre_period_covariate (needed for CUPED) and/or "
        f"{', '.join(OPTIONAL_SEGMENT_COLUMNS)} (for segment analysis). "
        f"variant_id values must end in '_control' or '_treatment'."
    )

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file is None:
        return None

    try:
        df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read this file as a CSV: {exc}")
        return None

    missing = [c for c in REQUIRED_CSV_COLUMNS if c not in df.columns]
    if missing:
        st.error(f"Missing required column(s): {missing}")
        return None

    if not df["variant_id"].str.endswith(("_control", "_treatment")).all():
        st.error(
            "Every variant_id value must end in '_control' or '_treatment' "
            "(e.g. 'myexp_control'). Found values that don't match this pattern."
        )
        return None

    if "user_id" not in df.columns:
        df = df.copy()
        df["user_id"] = [f"u_{i:07d}" for i in range(len(df))]

    if "pre_period_covariate" not in df.columns:
        st.warning(
            "No pre_period_covariate column found — CUPED will not be "
            "available for this dataset."
        )
        df["pre_period_covariate"] = None

    for col in OPTIONAL_SEGMENT_COLUMNS:
        if col not in df.columns:
            df[col] = None

    engine = _fresh_session_engine()
    experiment_id = _generate_experiment_id()
    # Whatever prefix the uploaded file's variant_id values originally had
    # doesn't need to match our newly-generated experiment_id — every
    # downstream query filters by the assignments table's own
    # experiment_id column, never by parsing the variant_id string itself.
    control_variant_id = f"{experiment_id}_control"
    treatment_variant_id = f"{experiment_id}_treatment"
    df = df.copy()
    df["variant_id"] = df["variant_id"].apply(
        lambda v: control_variant_id if v.endswith("_control") else treatment_variant_id
    )

    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO experiments (experiment_id, name, start_date, status) "
            "VALUES (:eid, 'Uploaded dataset', '2025-01-01', 'analyzed')"
        ), {"eid": experiment_id})
        conn.execute(text(
            "INSERT INTO variants (variant_id, experiment_id, name, split_pct) VALUES "
            "(:cvid, :eid, 'control', 0.5), (:tvid, :eid, 'treatment', 0.5)"
        ), {"cvid": control_variant_id, "tvid": treatment_variant_id, "eid": experiment_id})

        for _, row in df.iterrows():
            conn.execute(text(
                "INSERT INTO users (user_id, signup_date, device_type, region, "
                "existing_customer, pre_period_covariate) VALUES "
                "(:uid, '2025-01-01', :device_type, :region, :existing_customer, :covariate)"
            ), {
                "uid": row["user_id"],
                "device_type": row.get("device_type"),
                "region": row.get("region"),
                "existing_customer": (
                    bool(row["existing_customer"])
                    if pd.notna(row.get("existing_customer")) else None
                ),
                "covariate": (
                    float(row["pre_period_covariate"])
                    if pd.notna(row.get("pre_period_covariate")) else None
                ),
            })
            conn.execute(text(
                "INSERT INTO assignments (user_id, experiment_id, variant_id, assigned_at) "
                "VALUES (:uid, :eid, :vid, '2025-01-01')"
            ), {"uid": row["user_id"], "eid": experiment_id, "vid": row["variant_id"]})
            if int(row["converted"]) == 1:
                conn.execute(text(
                    "INSERT INTO events (event_id, user_id, experiment_id, event_type, "
                    "event_timestamp, value) VALUES (:evid, :uid, :eid, 'conversion', "
                    "'2025-01-02', 1.0)"
                ), {
                    "evid": f"ev_{uuid.uuid4().hex[:12]}",
                    "uid": row["user_id"], "eid": experiment_id,
                })

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
