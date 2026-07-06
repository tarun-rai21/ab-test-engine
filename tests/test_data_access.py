"""
tests/test_data_access.py — three-layer verification of core/data_access.py:
  1. Known-answer test against hand-inserted rows (isolates SQL logic).
  2. Cross-check against the simulator's own in-memory DataFrame (isolates
     the write-then-read round trip: dtype coercion, insertion, aggregation).
  3. Full pipeline integration test with a deliberately corrupted split.
"""

import pandas as pd
import pytest
from sqlalchemy import text

from core.data_access import get_variant_counts
from core.validity import srm_check
from data_sim.simulator import ExperimentSimulator
from db.connection import get_engine, init_schema, reset_engine
from db.seed import coerce_for_sqlite, seed_database


@pytest.fixture(autouse=True)
def isolated_db(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    reset_engine()
    engine = get_engine()
    init_schema(engine)
    yield engine
    reset_engine()


# --------------------------------------------------------------------- #
# Layer 1: known-answer test, hand-inserted rows, zero simulator involved
# --------------------------------------------------------------------- #

def test_get_variant_counts_known_answer(isolated_db):
    engine = isolated_db
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO experiments (experiment_id, name, start_date, status) "
            "VALUES ('exp_test', 'test', '2025-01-01', 'running')"
        ))
        conn.execute(text(
            "INSERT INTO variants (variant_id, experiment_id, name, split_pct) VALUES "
            "('exp_test_control', 'exp_test', 'control', 0.5), "
            "('exp_test_treatment', 'exp_test', 'treatment', 0.5)"
        ))
        # Hand-specify EXACTLY 7 control, 3 treatment — a known, arbitrary answer.
        for i in range(7):
            conn.execute(text(
                "INSERT INTO users (user_id, signup_date, device_type, region, existing_customer) "
                f"VALUES ('u_c{i}', '2025-01-01', 'mobile', 'US', 0)"
            ))
            conn.execute(text(
                f"INSERT INTO assignments (user_id, experiment_id, variant_id, assigned_at) "
                f"VALUES ('u_c{i}', 'exp_test', 'exp_test_control', '2025-01-01')"
            ))
        for i in range(3):
            conn.execute(text(
                "INSERT INTO users (user_id, signup_date, device_type, region, existing_customer) "
                f"VALUES ('u_t{i}', '2025-01-01', 'mobile', 'US', 0)"
            ))
            conn.execute(text(
                f"INSERT INTO assignments (user_id, experiment_id, variant_id, assigned_at) "
                f"VALUES ('u_t{i}', 'exp_test', 'exp_test_treatment', '2025-01-01')"
            ))

    observed, expected = get_variant_counts(engine, "exp_test")

    # ORDER BY variant_id -> 'exp_test_control' < 'exp_test_treatment' alphabetically
    assert observed == [7, 3]
    assert expected == [0.5, 0.5]


def test_get_variant_counts_zero_assignments_variant_not_dropped(isolated_db):
    """
    Directly tests the LEFT JOIN decision: a variant with ZERO assignments
    must still appear in the result with observed_n=0, not vanish.
    """
    engine = isolated_db
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO experiments (experiment_id, name, start_date, status) "
            "VALUES ('exp_zero', 'test', '2025-01-01', 'running')"
        ))
        conn.execute(text(
            "INSERT INTO variants (variant_id, experiment_id, name, split_pct) VALUES "
            "('exp_zero_control', 'exp_zero', 'control', 0.5), "
            "('exp_zero_treatment', 'exp_zero', 'treatment', 0.5)"
        ))
        conn.execute(text(
            "INSERT INTO users (user_id, signup_date, device_type, region, existing_customer) "
            "VALUES ('u1', '2025-01-01', 'mobile', 'US', 0)"
        ))
        conn.execute(text(
            "INSERT INTO assignments (user_id, experiment_id, variant_id, assigned_at) "
            "VALUES ('u1', 'exp_zero', 'exp_zero_control', '2025-01-01')"
        ))
        # NOTE: treatment variant gets ZERO assignments — deliberately.

    observed, expected = get_variant_counts(engine, "exp_zero")
    assert observed == [1, 0]  # treatment count must be 0, not missing/dropped


def test_get_variant_counts_missing_experiment_raises(isolated_db):
    with pytest.raises(ValueError):
        get_variant_counts(isolated_db, "nonexistent_experiment")


# --------------------------------------------------------------------- #
# Layer 2: cross-check SQL result against the simulator's own DataFrame,
# independent of any hand-typed expected values.
# --------------------------------------------------------------------- #

def test_get_variant_counts_matches_dataframe_groupby(isolated_db, monkeypatch):
    """
    THE key round-trip test: computes counts TWO independent ways —
    (a) pandas groupby directly on the simulator's in-memory output, before
        any DB involvement at all
    (b) SQL query after seeding that same data into SQLite

    If these disagree, the bug is in coerce_for_sqlite / insertion / the SQL
    query itself — NOT in the simulator (path (a) never touches the DB) and
    NOT in core.validity (neither path calls it yet).
    """
    sim = ExperimentSimulator(n_users=5000, baseline_rate=0.1, true_effect=0.0, seed=11)
    users_df, assignments_df, events_df, _ = sim.generate()

    # Path (a): pure pandas, zero DB involvement.
    expected_counts_from_df = (
        assignments_df["variant_id"].value_counts().sort_index().tolist()
    )

    # Seed the SAME data into the isolated in-memory DB.
    engine = isolated_db
    treatment_pct = 0.5
    experiments_df = pd.DataFrame([{
        "experiment_id": sim.experiment_id, "name": "test", "hypothesis": None,
        "start_date": "2025-01-01", "end_date": None, "status": "analyzed",
    }])
    variants_df = pd.DataFrame([
        {"variant_id": f"{sim.experiment_id}_control", "experiment_id": sim.experiment_id,
         "name": "control", "split_pct": 1 - treatment_pct},
        {"variant_id": f"{sim.experiment_id}_treatment", "experiment_id": sim.experiment_id,
         "name": "treatment", "split_pct": treatment_pct},
    ])
    users_df = coerce_for_sqlite(users_df)
    assignments_df_coerced = coerce_for_sqlite(assignments_df)

    with engine.begin() as conn:
        experiments_df.to_sql("experiments", conn, if_exists="append", index=False)
        variants_df.to_sql("variants", conn, if_exists="append", index=False)
        users_df.to_sql("users", conn, if_exists="append", index=False)
        assignments_df_coerced.to_sql("assignments", conn, if_exists="append", index=False)

    # Path (b): through the full round trip.
    observed_from_sql, _ = get_variant_counts(engine, sim.experiment_id)

    assert observed_from_sql == expected_counts_from_df, (
        f"SQL round-trip counts {observed_from_sql} do NOT match direct "
        f"DataFrame groupby {expected_counts_from_df}. Bug is in "
        f"coerce_for_sqlite, insertion, or the SQL query — not the simulator."
    )


# --------------------------------------------------------------------- #
# Layer 3: full pipeline — simulate corrupted split -> seed -> query -> SRM
# --------------------------------------------------------------------- #

def test_full_pipeline_detects_corrupted_split(isolated_db):
    """
    First true end-to-end test in this project: Phase 1 simulator ->
    Phase 0 schema/DB -> Phase 2 SQL query -> Phase 2 statistical check,
    all in one execution. Proves the SYSTEM catches broken randomization,
    not just that individual functions are independently correct.
    """
    engine = isolated_db
    config = {
        "simulator": {
            "n_users": 20000, "baseline_rate": 0.1, "true_effect": 0.0,
            "covariate_correlation": 0.0, "seed": 99, "corrupted_split": 0.45,
        }
    }
    seed_database(config, database_url="sqlite:///:memory:")
    # NOTE: seed_database() calls get_engine() internally, which — due to the
    # singleton pattern — returns the SAME engine as isolated_db's fixture,
    # since reset_engine() was called and DATABASE_URL is already set to the
    # same in-memory URL. Verify this assumption holds if this test behaves
    # unexpectedly; the singleton's interaction with :memory: URLs across
    # fixture boundaries is exactly the kind of thing worth distrusting.

    observed, expected = get_variant_counts(get_engine(), "exp_seed99")
    result = srm_check(observed, expected)

    print(f"observed={observed}, expected={expected}, chi_sq={result.chi_sq_stat:.4f}, p={result.p_value:.6f}")

    assert result.flagged, (
        f"Deliberately corrupted 45/55 split (observed={observed}) was NOT "
        f"flagged by the full pipeline. Either the corruption didn't reach "
        f"the DB, or the query/SRM wiring is broken."
    )


def test_full_pipeline_healthy_split_not_flagged(isolated_db):
    """Symmetric case: a healthy 50/50 split must NOT be flagged end-to-end."""
    config = {
        "simulator": {
            "n_users": 20000, "baseline_rate": 0.1, "true_effect": 0.0,
            "covariate_correlation": 0.0, "seed": 100, "corrupted_split": None,
        }
    }
    seed_database(config, database_url="sqlite:///:memory:")

    observed, expected = get_variant_counts(get_engine(), "exp_seed100")
    result = srm_check(observed, expected)

    assert not result.flagged, (
        f"Healthy 50/50 split (observed={observed}) was falsely flagged — "
        f"check for an off-by-one or threshold bug."
    )

