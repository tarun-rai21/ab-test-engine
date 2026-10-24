"""
tests/test_segment_heterogeneity.py — Cochran's Q heterogeneity test and the
single Benjamini-Hochberg correction applied across every segment test in a
report.
"""

from __future__ import annotations

import pytest
from scipy.stats import chi2

from core.inference import InferenceResult
from core.pipeline import analyze_experiment
from core.segments import (
    SegmentAnalysisResult,
    SegmentResult,
    apply_global_bh,
    heterogeneity_test,
)
from db.connection import get_engine, init_schema, reset_engine
from db.seed import seed_database


@pytest.fixture
def isolated_db(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    reset_engine()
    engine = get_engine()
    init_schema(engine)
    yield engine
    reset_engine()


def _inference(estimate: float, p_value: float, se: float = 0.05) -> InferenceResult:
    return InferenceResult(
        method="raw_ttest", point_estimate=estimate, ci_lower=estimate - 2 * se,
        ci_upper=estimate + 2 * se, alpha=0.05, p_value=p_value, standard_error=se,
        degrees_freedom=100.0, n_control=100, n_treatment=100,
    )


def _analysis(column: str, p_values: list[float], estimate: float = 0.1) -> SegmentAnalysisResult:
    return SegmentAnalysisResult(
        segment_column=column,
        pooled_point_estimate=0.05,
        segments=tuple(
            SegmentResult(f"{column}_{i}", _inference(estimate, p), True, False)
            for i, p in enumerate(p_values)
        ),
        excluded_segments=(),
    )


# --------------------------------------------------------------------- #
# heterogeneity_test
# --------------------------------------------------------------------- #

def test_heterogeneity_test_known_value():
    # w = 400 each, pooled estimate 0, Q = 400 * 0.01 * 2 = 8 on 1 df
    q, p = heterogeneity_test([0.1, -0.1], [0.05, 0.05])
    assert q == pytest.approx(8.0)
    assert p == pytest.approx(chi2.sf(8.0, 1))


def test_heterogeneity_test_identical_effects_give_q_zero():
    q, p = heterogeneity_test([0.03, 0.03, 0.03], [0.01, 0.02, 0.05])
    assert q == pytest.approx(0.0)
    assert p == pytest.approx(1.0)


def test_heterogeneity_test_weights_by_precision():
    """A precise segment pulls the pooled estimate toward itself, so the
    noisy segment contributes most of Q."""
    q, _ = heterogeneity_test([0.0, 0.2], [0.01, 0.1])
    pooled = (0.0 / 0.01**2 + 0.2 / 0.1**2) / (1 / 0.01**2 + 1 / 0.1**2)
    expected = (0.0 - pooled) ** 2 / 0.01**2 + (0.2 - pooled) ** 2 / 0.1**2
    assert q == pytest.approx(expected)


def test_heterogeneity_test_invalid_inputs():
    with pytest.raises(ValueError):
        heterogeneity_test([0.1], [0.05])
    with pytest.raises(ValueError):
        heterogeneity_test([0.1, 0.2], [0.05, 0.0])
    with pytest.raises(ValueError):
        heterogeneity_test([0.1, 0.2], [0.05])


# --------------------------------------------------------------------- #
# apply_global_bh
# --------------------------------------------------------------------- #

def test_global_bh_is_stricter_than_per_column_bh():
    """
    Column A p = [0.02, 0.9] and column B p = [0.03, 0.04]. Corrected per
    column, three segments pass. Corrected once across all four tests
    (thresholds 0.0125, 0.025, 0.0375, 0.05), none do.
    """
    analyses = (_analysis("A", [0.02, 0.9]), _analysis("B", [0.03, 0.04]))
    corrected = apply_global_bh(analyses)

    assert all(not s.bh_significant for a in corrected for s in a.segments)
    assert all(not s.reversal_flag for a in corrected for s in a.segments)


def test_global_bh_recomputes_reversal_flags():
    analyses = (_analysis("A", [0.0001, 0.5], estimate=-0.2),)
    corrected = apply_global_bh(analyses)
    first, second = corrected[0].segments
    assert first.bh_significant and first.reversal_flag  # significant and opposite sign
    assert not second.bh_significant and not second.reversal_flag


def test_global_bh_handles_empty_input():
    assert apply_global_bh(()) == ()


# --------------------------------------------------------------------- #
# pipeline
# --------------------------------------------------------------------- #

def test_pipeline_reports_heterogeneity_and_corrects_across_columns(isolated_db):
    seed_database({"simulator": {
        "n_users": 20000, "baseline_rate": 0.10, "true_effect": 0.0,
        "covariate_correlation": 0.0, "seed": 500, "corrupted_split": None,
        "segment_heterogeneity": {"device_type": {"mobile": 0.15, "desktop": -0.08}},
    }}, database_url="sqlite:///:memory:")

    report = analyze_experiment(
        isolated_db, "exp_seed500", segment_columns=["device_type", "region"], persist=False
    )
    by_column = {a.segment_column: a for a in report.segments}

    # Configured: effects differ strongly by device, not at all by region.
    assert by_column["device_type"].heterogeneity_p_value < 1e-10
    assert by_column["region"].heterogeneity_p_value > 0.001
    assert by_column["region"].heterogeneity_q is not None
