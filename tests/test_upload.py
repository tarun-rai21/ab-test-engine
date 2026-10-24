"""
tests/test_upload.py — the Streamlit upload flow's validation and ingestion
logic (app/components/upload_or_generate.py), tested without a running
Streamlit server: validate_upload() and ingest_upload() are plain functions.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.components.upload_or_generate import ingest_upload, validate_upload
from core.pipeline import analyze_experiment
from db.connection import create_memory_engine


def _valid_df(n_per_arm: int = 50, treatment_n: int | None = None) -> pd.DataFrame:
    treatment_n = n_per_arm if treatment_n is None else treatment_n
    return pd.DataFrame({
        "variant_id": ["x_control"] * n_per_arm + ["x_treatment"] * treatment_n,
        "converted": ([1, 0, 0, 0, 0] * n_per_arm)[:n_per_arm]
        + ([1, 1, 0, 0, 0] * treatment_n)[:treatment_n],
        "pre_period_covariate": [float(i % 7) for i in range(n_per_arm + treatment_n)],
        "device_type": ["mobile", "desktop"] * ((n_per_arm + treatment_n) // 2),
    })


def test_validate_upload_accepts_valid_data():
    assert validate_upload(_valid_df()) == []


@pytest.mark.parametrize("mutate, fragment", [
    (lambda df: df.drop(columns="converted"), "Missing required"),
    (lambda df: df.assign(variant_id="x_other"), "_control"),
    (lambda df: df.assign(converted=2), "0/1"),
    (lambda df: df.assign(converted=[None] + [0] * (len(df) - 1)), "0/1"),
    (lambda df: df.assign(user_id="same"), "unique"),
    (lambda df: df.assign(pre_period_covariate=[None] + [1.0] * (len(df) - 1)), "numeric"),
])
def test_validate_upload_reports_problems(mutate, fragment):
    errors = validate_upload(mutate(_valid_df()))
    assert any(fragment in e for e in errors), errors


def test_two_uploads_in_one_session_do_not_collide():
    """
    Review issue M3/H1: a second upload in the same session DB regenerated
    the same user_ids and crashed with an IntegrityError.
    """
    engine = create_memory_engine()
    ingest_upload(engine, _valid_df(), "exp_a")
    ingest_upload(engine, _valid_df(), "exp_b")

    report = analyze_experiment(engine, "exp_b", persist=False)
    assert report.raw_effect.n_control == 50
    assert report.raw_effect.n_treatment == 50


def test_upload_intended_split_drives_srm_check():
    """
    A healthy 90/10 design must NOT be SRM-flagged once the intended split
    is supplied (it used to be hard-coded to 50/50).
    """
    engine = create_memory_engine()
    df = _valid_df(n_per_arm=900, treatment_n=100)

    ingest_upload(engine, df, "exp_9010", treatment_split=0.10)
    assert analyze_experiment(engine, "exp_9010", persist=False).srm.flagged is False

    ingest_upload(engine, df, "exp_5050", treatment_split=0.50)
    assert analyze_experiment(engine, "exp_5050", persist=False).srm.flagged is True


def test_upload_revenue_column_feeds_revenue_metrics():
    df = _valid_df()
    is_treatment = df["variant_id"] == "x_treatment"
    order_value = [30.0 + 10 * (i % 3) for i in range(len(df))] + 20.0 * is_treatment
    df["revenue"] = df["converted"] * order_value
    assert validate_upload(df) == []

    engine = create_memory_engine()
    ingest_upload(engine, df, "exp_rev")
    report = analyze_experiment(
        engine, "exp_rev", metric_name="revenue_per_conversion", persist=False
    )
    converters = df[df["converted"] == 1]
    expected = (
        converters.loc[converters["variant_id"] == "x_treatment", "revenue"].mean()
        - converters.loc[converters["variant_id"] == "x_control", "revenue"].mean()
    )
    assert report.raw_effect.point_estimate == pytest.approx(expected)


def test_validate_upload_rejects_bad_revenue():
    df = _valid_df()
    df["revenue"] = 10.0  # revenue on non-converting rows
    assert any("converted = 1" in e for e in validate_upload(df))
    df["revenue"] = -1.0
    assert any("non-negative" in e for e in validate_upload(df))


def test_upload_with_extra_treatment_arm():
    df = pd.concat([_valid_df(), _valid_df().assign(variant_id="x_treatment_b").iloc[50:]])
    assert validate_upload(df) == []

    engine = create_memory_engine()
    ingest_upload(engine, df, "exp_abn", treatment_split=0.5)
    report = analyze_experiment(engine, "exp_abn", persist=False)

    assert [c.treatment_arm for c in report.comparisons] == ["treatment", "treatment_b"]
    assert report.srm.expected_counts == pytest.approx((75.0, 37.5, 37.5))


def test_validate_upload_requires_control_and_an_arm():
    df = _valid_df().assign(variant_id="x_control")
    assert any("control group and at least one" in e for e in validate_upload(df))
