"""
core/metrics.py — which number a metric name refers to, and how its
treatment effect is estimated.

Three metrics, all computed per USER (the randomization unit) from
core.data_access.get_inference_data()'s columns:

  - "conversion": share of users who converted (0/1 per user). Welch t-test.
  - "revenue_per_user": mean revenue per user, 0 for non-converters.
    Continuous and heavy-tailed; Welch t-test, optionally on WINSORIZED
    values (capped at a pooled quantile) to tame outliers.
  - "revenue_per_conversion": average order value = sum(revenue) /
    sum(conversions). A RATIO of two random per-user quantities, so it uses
    the delta method (core.inference.delta_method_ratio_ci), not a t-test
    on individual orders.

Mean metrics support CUPED (an additive per-user adjustment); the ratio
metric does not, so CUPED is skipped for it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.inference import InferenceResult, delta_method_ratio_ci, raw_ttest_ci

METRICS = ("conversion", "revenue_per_user", "revenue_per_conversion")
MEAN_METRICS = ("conversion", "revenue_per_user")


def validate_metric(metric: str) -> None:
    if metric not in METRICS:
        raise ValueError(f"Unknown metric {metric!r} — must be one of {METRICS}.")


def winsorization_cap(df: pd.DataFrame, metric: str, quantile: float | None) -> float | None:
    """
    Upper cap for revenue_per_user, at the given quantile of the POOLED
    (both arms) per-user revenue — pooled so the cap cannot depend on the
    treatment assignment. None when no winsorization was requested or the
    metric is not revenue_per_user (capping a 0/1 outcome is meaningless;
    the ratio metric is not winsorized here).
    """
    if quantile is None or metric != "revenue_per_user":
        return None
    if not (0.5 < quantile < 1.0):
        raise ValueError(f"winsorize_quantile must be in (0.5, 1), got {quantile}")
    return float(np.quantile(df["revenue"].to_numpy(dtype=float), quantile))


def metric_outcome(df: pd.DataFrame, metric: str, cap: float | None = None) -> np.ndarray:
    """Per-user outcome for a MEAN metric (the values CUPED adjusts)."""
    if metric == "conversion":
        return df["converted"].to_numpy(dtype=float)
    if metric == "revenue_per_user":
        values = df["revenue"].to_numpy(dtype=float)
        return np.minimum(values, cap) if cap is not None else values
    raise ValueError(f"{metric!r} is not a per-user mean metric.")


def estimate_effect(
    control_df: pd.DataFrame,
    treatment_df: pd.DataFrame,
    metric: str,
    cap: float | None = None,
    alpha: float = 0.05,
) -> InferenceResult:
    """Treatment-minus-control effect on `metric` for one pair of arms."""
    validate_metric(metric)
    if metric == "revenue_per_conversion":
        return delta_method_ratio_ci(
            control_df["revenue"].to_numpy(dtype=float),
            control_df["n_conversions"].to_numpy(dtype=float),
            treatment_df["revenue"].to_numpy(dtype=float),
            treatment_df["n_conversions"].to_numpy(dtype=float),
            alpha=alpha,
        )
    return raw_ttest_ci(
        metric_outcome(control_df, metric, cap),
        metric_outcome(treatment_df, metric, cap),
        alpha=alpha,
    )
