"""
app/components/upload_or_generate.py — Phase 8's data-input widget.

Two modes, both ending in the SAME place: a real, schema-compliant
in-memory SQLite database, populated with exactly one experiment, ready
for core.pipeline.analyze_experiment() to run against — unmodified, the
exact same function already proven correct in Phases 1-7. Neither mode
introduces a second, divergent analysis code path.

MODE 1 — "Generate simulated data": wraps db.seed.seed_database() (Phase
1/7, already tested) with Streamlit input widgets.

MODE 2 — "Upload CSV": one row per user, in the source system's own shape.
Columns are mapped onto the canonical ones (variant, converted, and the
optional user_id, revenue, pre_period_covariate and segment columns), and
raw variant labels such as "A"/"B" onto control / treatment arms — see
core.ingestion, which the API's POST /experiments/{id}/ingest shares.
Files already using the canonical names need no mapping.
"""

from __future__ import annotations

import hashlib
import io
import json
import uuid

import pandas as pd
import streamlit as st
from sqlalchemy import Engine

from core.ingestion import (
    create_experiment_with_variants,
    ingest_canonical,
    to_canonical,
    validate_canonical,
)
from db.connection import create_memory_engine
from db.seed import seed_database


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


def validate_upload(
    df: pd.DataFrame,
    columns: dict[str, str] | None = None,
    variant_values: dict[str, str | None] | None = None,
) -> list[str]:
    """
    Problems with an uploaded DataFrame under the given column mapping
    ([] = valid). See core.ingestion for the mapping rules.
    """
    try:
        canonical = to_canonical(df, columns, variant_values)
    except ValueError as exc:
        return [str(exc)]
    return validate_canonical(canonical)


def ingest_upload(
    engine: Engine,
    df: pd.DataFrame,
    experiment_id: str,
    treatment_split: float = 0.5,
    columns: dict[str, str] | None = None,
    variant_values: dict[str, str | None] | None = None,
) -> None:
    """
    Creates a new experiment for an uploaded file and ingests it.
    treatment_split is the intended share of users across all treatment
    arms (used by the SRM check).
    """
    canonical = to_canonical(df, columns, variant_values)
    errors = validate_canonical(canonical)
    if errors:
        raise ValueError(" ".join(errors))
    create_experiment_with_variants(
        engine, experiment_id, list(canonical["variant_name"].unique()), treatment_split
    )
    ingest_canonical(engine, experiment_id, canonical)


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

        add_arm_b = st.checkbox("Add a second treatment arm (treatment_b)", value=False)
        arm_b_effect = st.slider(
            "True effect of treatment_b (if added)",
            min_value=-0.20, max_value=0.20, value=0.0, step=0.005,
        )

        col3, col4 = st.columns(2)
        with col3:
            revenue_mean = st.number_input(
                "Average order value (0 = conversions only, no revenue)",
                min_value=0.0, value=50.0, step=5.0,
            )
        with col4:
            revenue_lift = st.slider(
                "Treatment lift in order value", min_value=-0.5, max_value=0.5,
                value=0.0, step=0.01,
            )

        submitted = st.form_submit_button("Generate data")

    if not submitted:
        return None

    config = {"simulator": {
        "n_users": int(n_users),
        "baseline_rate": float(baseline_rate),
        "true_effect": float(true_effect),
        "covariate_correlation": float(covariate_correlation),
        "seed": int(seed),
        "revenue_mean": float(revenue_mean) if revenue_mean > 0 else None,
        "revenue_lift": float(revenue_lift),
        "treatment_arms": (
            {"treatment": float(true_effect), "treatment_b": float(arm_b_effect)}
            if add_arm_b else None
        ),
        # The SRM-bug knob is two-arm only.
        "corrupted_split": corrupted_split if not add_arm_b else None,
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


def _column_mapping_widgets(df: pd.DataFrame) -> tuple[dict[str, str], dict[str, str | None]]:
    """Lets the user say which column plays which role, and what each raw
    variant value means."""
    none = "— not in this file —"
    options = [none, *df.columns]

    def pick(label: str, role: str, guesses: tuple[str, ...], required: bool = False):
        default = next((c for c in df.columns if c.lower() in guesses), None)
        choices = list(df.columns) if required else options
        index = choices.index(default) if default in choices else 0
        return st.selectbox(label, choices, index=index, key=f"map_{role}")

    col1, col2 = st.columns(2)
    with col1:
        variant_col = pick("Variant / group column", "variant",
                           ("variant", "variant_id", "group", "arm", "bucket"), required=True)
        converted_col = pick("Converted column (0/1, true/false, yes/no)", "converted",
                             ("converted", "conversion", "purchased", "converted_flag"))
        revenue_col = pick("Revenue per user", "revenue", ("revenue", "spend", "amount"))
        user_col = pick("User id", "user_id", ("user_id", "user", "uid", "customer",
                                               "customer_id", "visitor_id", "id"))
    with col2:
        covariate_col = pick("Pre-period covariate (for CUPED)", "pre_period_covariate",
                             ("pre_period_covariate", "pre_spend", "baseline"))
        device_col = pick("Device type segment", "device_type", ("device_type", "device"))
        region_col = pick("Region segment", "region", ("region", "country", "market"))
        customer_col = pick("Existing-customer segment", "existing_customer",
                            ("existing_customer", "returning", "is_existing"))

    columns = {"variant": variant_col}
    for role, col in (("converted", converted_col), ("revenue", revenue_col),
                      ("user_id", user_col), ("pre_period_covariate", covariate_col),
                      ("device_type", device_col), ("region", region_col),
                      ("existing_customer", customer_col)):
        if col != none:
            columns[role] = col

    raw_values = sorted(df[variant_col].astype(str).str.strip().unique())
    if len(raw_values) > 10:
        st.error(f"The variant column has {len(raw_values)} distinct values — is it the "
                 f"right column?")
        return columns, {}
    st.markdown("**What does each variant value mean?**")
    choices = ["control", "treatment", "treatment_b", "treatment_c", "ignore these rows"]
    variant_values: dict[str, str | None] = {}
    value_cols = st.columns(min(len(raw_values), 4))
    for i, value in enumerate(raw_values):
        default = "control" if "control" in value.lower() or i == 0 else (
            choices[min(i, 3)] if i > 0 else "treatment"
        )
        with value_cols[i % len(value_cols)]:
            meaning = st.selectbox(f"`{value}`", choices, index=choices.index(default),
                                   key=f"variant_value_{value}")
        variant_values[value] = None if meaning == "ignore these rows" else meaning
    return columns, variant_values


def render_upload_form() -> str | None:
    """
    Renders the 'upload a CSV' form. Returns the experiment_id once a
    valid file has been uploaded and ingested, or None otherwise.

    Ingestion is remembered per (file contents, mapping, split): Streamlit
    re-runs this script on every click, and re-ingesting the still-attached
    file each time used to create a NEW experiment on every interaction,
    discarding the report and piling copies into the database.
    """
    st.subheader("Upload your own data")
    st.caption(
        "One row per user. If your columns already use this project's names "
        "(variant_id ending in '_control' / containing '_treatment', converted, "
        "and optionally user_id, revenue, pre_period_covariate, device_type, region, "
        "existing_customer), they are used directly; otherwise map them below."
    )

    treatment_split = st.number_input(
        "Intended share of users in treatment arms (used by the SRM check)",
        min_value=0.01, max_value=0.99, value=0.50, step=0.01,
    )
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file is None:
        return None

    content = uploaded_file.getvalue()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        st.error(f"Could not read this file as a CSV: {exc}")
        return None

    uses_convention = {"variant_id", "converted"} <= set(df.columns)
    map_columns = st.checkbox(
        "Map my columns (different column names or variant labels)",
        value=not uses_convention,
    )
    columns, variant_values = (
        _column_mapping_widgets(df) if map_columns else (None, None)
    )

    errors = validate_upload(df, columns, variant_values)
    if errors:
        for error in errors:
            st.error(error)
        return None

    key = hashlib.sha256(
        content + json.dumps([columns, variant_values, treatment_split],
                             sort_keys=True, default=str).encode()
    ).hexdigest()
    ingested = st.session_state.setdefault("ingested_uploads", {})
    if key in ingested:
        return ingested[key]

    if columns is None and "pre_period_covariate" not in df.columns or (
        columns is not None and "pre_period_covariate" not in columns
    ):
        st.warning("No pre-period covariate — CUPED will not be available for this dataset.")

    experiment_id = _generate_experiment_id()
    ingest_upload(get_session_engine(), df, experiment_id, float(treatment_split),
                  columns, variant_values)
    ingested[key] = experiment_id
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
