"""
db/seed.py — generates a synthetic dataset via ExperimentSimulator and persists
it into the schema built in Phase 0.

Run (from project root):
    python -m db.seed

IDEMPOTENCY: reruns are common during iterative development — every phase from
here on will call this repeatedly with different simulator configs. Delete-then-
insert scoped to THIS experiment_id (not the whole DB) makes reruns safe without
requiring you to manually `del local.db` first, and without touching any other
experiment's data that might exist in the same DB. This is deliberately NOT an
upsert (INSERT OR REPLACE) — that's more complexity than this dev-fixture use
case needs; revisit only if a later phase requires multiple persistent
coexisting experiments.

Does NOT persist GroundTruth — no ground_truth table exists in schema.sql by
design (Section 4.2): ground truth is a development-time oracle, not queryable
data an analyst could accidentally treat as another dataset. It's printed to
console here; Phase 4's validation harness will hold it in memory directly from
the simulator object, never read back from a table.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import yaml
from sqlalchemy import text

from data_sim.simulator import ExperimentSimulator
from db.connection import get_engine, init_schema

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "default_config.yaml"

# Deletion order = child before parent — the REVERSE of insertion order.
# Deleting a parent while children still reference it would violate FK intent
# even though SQLite doesn't enforce it by default (same reasoning as the
# insertion-order comment below: get it right on principle, don't rely on the
# DB to catch what it isn't checking for).
_DELETE_ORDER = [
    "srm_checks",
    "sequential_checkpoints",
    "experiment_results",
    "events",
    "assignments",
    "variants",
    "experiments",
    # NOTE: users are intentionally NOT deleted here. A user could in principle
    # be shared across experiments in a real system; scoping deletion to
    # experiment_id alone can't safely clean users without a users.experiment_id
    # link, which doesn't exist (users aren't experiment-scoped in this schema).
    # For this synthetic single-experiment-per-seed-run workflow, stale user
    # rows from a previous seed are harmless (same user_id values get
    # reinserted identically since generation is deterministic) but ARE a
    # known imprecision — flagged rather than silently ignored.
]


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def coerce_for_sqlite(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].astype(str)
        elif pd.api.types.is_bool_dtype(df[col]):
            df[col] = df[col].apply(bool)
        elif df[col].dtype == object:
            non_null = df[col].dropna()
            if not non_null.empty and hasattr(non_null.iloc[0], "isoformat"):
                df[col] = df[col].apply(lambda v: v.isoformat() if hasattr(v, "isoformat") else v)
    return df


def _delete_existing_experiment(engine, experiment_id: str) -> None:
    """
    Removes all rows tied to experiment_id, in child-before-parent order, so a
    rerun of seed_database() with the same seed (-> same deterministic
    experiment_id) doesn't hit a primary-key collision on `experiments` or
    leave orphaned rows behind from a previous run's differently-sized dataset.
    """
    with engine.begin() as conn:
        for table in _DELETE_ORDER:
            conn.execute(
                text(f"DELETE FROM {table} WHERE experiment_id = :eid"),
                {"eid": experiment_id},
            )


def seed_database(config: dict, database_url: str | None = None) -> None:
    if database_url:
        os.environ["DATABASE_URL"] = database_url

    sim_cfg = config["simulator"]
    sim = ExperimentSimulator(
        n_users=sim_cfg["n_users"],
        baseline_rate=sim_cfg["baseline_rate"],
        true_effect=sim_cfg["true_effect"],
        covariate_correlation=sim_cfg["covariate_correlation"],
        seed=sim_cfg["seed"],
        corrupted_split=sim_cfg.get("corrupted_split"),
        segment_heterogeneity=sim_cfg.get("segment_heterogeneity"),
    )

    users_df, assignments_df, events_df, ground_truth = sim.generate()

    engine = get_engine()
    init_schema(engine)

    _delete_existing_experiment(engine, sim.experiment_id)

    treatment_pct = sim.corrupted_split if sim.corrupted_split is not None else 0.5

    experiments_df = pd.DataFrame([{
        "experiment_id": sim.experiment_id,
        "name": "Synthetic demo experiment",
        "hypothesis": "Treatment increases conversion rate",
        "start_date": "2025-01-01",
        "end_date": None,
        "status": "analyzed",
    }])

    variants_df = pd.DataFrame([
        {"variant_id": f"{sim.experiment_id}_control", "experiment_id": sim.experiment_id,
         "name": "control", "split_pct": 1 - treatment_pct},
        {"variant_id": f"{sim.experiment_id}_treatment", "experiment_id": sim.experiment_id,
         "name": "treatment", "split_pct": treatment_pct},
    ])

    users_df = coerce_for_sqlite(users_df)
    assignments_df = coerce_for_sqlite(assignments_df)
    events_df = coerce_for_sqlite(events_df)

    # Insertion order = parent before child (see _DELETE_ORDER's docstring for
    # the reverse-order deletion reasoning). NOT enforced by SQLite by default,
    # so getting this order wrong would silently insert orphaned rows rather
    # than error — correctness by construction, not by DB-enforced constraint.
    #
    # users uses if_exists="append" unconditionally (never deleted above) —
    # since user_id generation is deterministic (f"u_{i:07d}"), reinserting
    # the SAME user_ids on a rerun with the SAME n_users would violate the
    # users PRIMARY KEY. Guard against that specific case:
    with engine.begin() as conn:
        existing_user_ids = pd.read_sql(text("SELECT user_id FROM users"), conn)["user_id"].tolist()
        new_users = users_df[~users_df["user_id"].isin(existing_user_ids)]

        experiments_df.to_sql("experiments", conn, if_exists="append", index=False)
        variants_df.to_sql("variants", conn, if_exists="append", index=False)
        if not new_users.empty:
            new_users.to_sql("users", conn, if_exists="append", index=False)
        assignments_df.to_sql("assignments", conn, if_exists="append", index=False)
        events_df.to_sql("events", conn, if_exists="append", index=False)

    print(
        f"Seeded {len(users_df)} users, {len(assignments_df)} assignments, "
        f"{len(events_df)} conversion events for experiment_id={sim.experiment_id}\n"
    )
    print(ground_truth.summary())


if __name__ == "__main__":
    cfg = load_config()
    seed_database(cfg)
