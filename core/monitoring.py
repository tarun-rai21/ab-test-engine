"""
core/monitoring.py — recording sequential "looks" at a live experiment.

A look computes the current p-value of every treatment arm vs control for
one metric and stores it, together with the peeking-corrected threshold
that applies at that moment, so later analyses (and the Lan-DeMets
boundary) see the real history of when the experiment was checked.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sqlalchemy import Engine

from core.data_access import (
    get_inference_data,
    get_planned_schedule,
    get_sequential_checkpoints,
    save_checkpoint,
    split_arms,
)
from core.metrics import estimate_effect, validate_metric
from core.sequential import sequential_check

NAIVE_ALPHA = 0.05


@dataclass(frozen=True)
class RecordedLook:
    treatment_arm: str
    metric_name: str
    cumulative_n: int
    p_value: float
    alpha_threshold: float  # corrected threshold in force (0.05 if no plan)
    planned_total_n: int | None
    corrected_significant: bool


def record_look(
    engine: Engine,
    experiment_id: str,
    metric_name: str = "conversion",
    users_per_variant: int | None = None,
    n_checkpoints_planned: int | None = None,
    checkpoint_n: int | None = None,
) -> tuple[RecordedLook, ...]:
    """
    Records one look: for every treatment arm, the raw p-value vs control on
    `metric_name`, stored as a sequential checkpoint for that (metric, arm).

    users_per_variant: use only the first n users of each variant — for
    replaying an EARLIER look on data that has already been collected (the
    Streamlit demo). None = all data so far (a real live look).

    cumulative_n is the experiment-wide number of users the look used, the
    same unit as the planned total (n_checkpoints_planned * checkpoint_n).
    The plan comes from the arguments, else from the experiment; with no
    plan the naive 0.05 is recorded and no correction can be computed.
    """
    validate_metric(metric_name)
    if (n_checkpoints_planned is None) != (checkpoint_n is None):
        raise ValueError("Pass both n_checkpoints_planned and checkpoint_n, or neither.")
    if n_checkpoints_planned is None:
        stored = get_planned_schedule(engine, experiment_id)
        if stored is not None:
            n_checkpoints_planned, checkpoint_n = stored
    planned_total_n = (
        n_checkpoints_planned * checkpoint_n if n_checkpoints_planned is not None else None
    )

    df = get_inference_data(engine, experiment_id, segment_columns=[])
    control_df, arms = split_arms(df)
    if users_per_variant is not None:
        if users_per_variant < 2:
            raise ValueError("users_per_variant must be at least 2")
        control_df = control_df.iloc[:users_per_variant]
        arms = {arm: arm_df.iloc[:users_per_variant] for arm, arm_df in arms.items()}
    cumulative_n = len(control_df) + sum(len(a) for a in arms.values())

    looks = []
    for arm, arm_df in arms.items():
        p_value = estimate_effect(control_df, arm_df, metric_name).p_value
        alpha_threshold = NAIVE_ALPHA
        corrected_significant = p_value < NAIVE_ALPHA
        if planned_total_n is not None:
            history = get_sequential_checkpoints(engine, experiment_id, metric_name, arm)
            this_look = {
                "cumulative_n": cumulative_n,
                "p_value_at_check": p_value,
                "alpha_threshold_at_check": NAIVE_ALPHA,
                "planned_total_n": planned_total_n,
            }
            check = sequential_check(
                [c for c in history if c["cumulative_n"] != cumulative_n] + [this_look],
                n_checkpoints_planned, checkpoint_n,
            )
            alpha_threshold = check.corrected_threshold
            corrected_significant = check.corrected_significant
        save_checkpoint(
            engine, experiment_id, cumulative_n, p_value, alpha_threshold,
            metric_name=metric_name, treatment_arm=arm, planned_total_n=planned_total_n,
        )
        looks.append(RecordedLook(
            treatment_arm=arm, metric_name=metric_name, cumulative_n=cumulative_n,
            p_value=p_value, alpha_threshold=alpha_threshold,
            planned_total_n=planned_total_n, corrected_significant=corrected_significant,
        ))
    return tuple(looks)


def max_users_per_variant(df: pd.DataFrame) -> int:
    """Size of the smallest variant — the largest replayable look."""
    control_df, arms = split_arms(df)
    return min(len(control_df), *(len(a) for a in arms.values()))
