"""
tests/test_data_access.py — three-layer verification of core/data_access.py:
  1. Known-answer test against hand-inserted rows (isolates SQL logic).
  2. Cross-check against the simulator's own in-memory DataFrame (isolates
     the write-then-read round trip: dtype coercion, insertion, aggregation).
  3. Full pipeline integration test — simulate -> seed -> query -> statistics.

Phase 2 established Layers 1-3 for SRM (get_variant_counts). Phase 3 extends
the same three-layer discipline to get_inference_data/split_by_variant,
specifically targeting the event-fanout risk and the CUPED covariate wiring
that core/inference.py's own unit tests could never catch, since those were
tested only against hand-built numpy arrays with no database involved.
"""
import numpy as np
import pandas as pd
import pytest
from sqlalchemy import text

from core.data_access import get_inference_data, get_variant_counts, split_by_variant
from core.inference import cuped_adjust, raw_ttest_ci, variance_reduction_pct
from core.validity import mde_curve, required_sample_size, srm_check
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


# ===================================================================== #
# PHASE 2 TESTS — get_variant_counts (SRM data wiring)
# ===================================================================== #

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

    assert observed == [7, 3]
    assert expected == [0.5, 0.5]


def test_get_variant_counts_zero_assignments_variant_not_dropped(isolated_db):
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

    observed, expected = get_variant_counts(engine, "exp_zero")
    assert observed == [1, 0]


def test_get_variant_counts_missing_experiment_raises(isolated_db):
    with pytest.raises(ValueError):
        get_variant_counts(isolated_db, "nonexistent_experiment")


# --------------------------------------------------------------------- #
# Layer 2: cross-check SQL result against the simulator's own DataFrame
# --------------------------------------------------------------------- #

def test_get_variant_counts_matches_dataframe_groupby(isolated_db):
    sim = ExperimentSimulator(n_users=5000, baseline_rate=0.1, true_effect=0.0, seed=11)
    users_df, assignments_df, events_df, _ = sim.generate()

    expected_counts_from_df = (
        assignments_df["variant_id"].value_counts().sort_index().tolist()
    )

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

    observed_from_sql, _ = get_variant_counts(engine, sim.experiment_id)

    assert observed_from_sql == expected_counts_from_df


# --------------------------------------------------------------------- #
# Layer 3: full pipeline — simulate corrupted split -> seed -> query -> SRM
# --------------------------------------------------------------------- #

def test_full_pipeline_detects_corrupted_split(isolated_db):
    config = {
        "simulator": {
            "n_users": 20000, "baseline_rate": 0.1, "true_effect": 0.0,
            "covariate_correlation": 0.0, "seed": 99, "corrupted_split": 0.45,
        }
    }
    seed_database(config, database_url="sqlite:///:memory:")

    observed, expected = get_variant_counts(get_engine(), "exp_seed99")
    result = srm_check(observed, expected)

    assert result.flagged, (
        f"Deliberately corrupted 45/55 split (observed={observed}) was NOT "
        f"flagged by the full pipeline. Either the corruption didn't reach "
        f"the DB, or the query/SRM wiring is broken."
    )


def test_full_pipeline_healthy_split_not_flagged(isolated_db):
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


# ===================================================================== #
# PHASE 3 TESTS — get_inference_data / split_by_variant (inference wiring)
# ===================================================================== #

# --------------------------------------------------------------------- #
# Layer 1: known-answer — event fan-out defense, hand-inserted, no simulator
# --------------------------------------------------------------------- #

def test_get_inference_data_prevents_event_fanout(isolated_db):
    """
    A user with TWO conversion events must still produce exactly ONE output
    row with converted=1 — proves GROUP BY + CASE WHEN collapses correctly,
    not a doubled row/count. Exercises a scenario the current simulator never
    produces but the schema permits.
    """
    engine = isolated_db
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO experiments (experiment_id, name, start_date, status) "
            "VALUES ('exp_fanout', 'test', '2025-01-01', 'running')"
        ))
        conn.execute(text(
            "INSERT INTO variants (variant_id, experiment_id, name, split_pct) VALUES "
            "('exp_fanout_treatment', 'exp_fanout', 'treatment', 0.5)"
        ))
        conn.execute(text(
            "INSERT INTO users (user_id, signup_date, device_type, region, "
            "existing_customer, pre_period_covariate) "
            "VALUES ('u1', '2025-01-01', 'mobile', 'US', 0, 1.5)"
        ))
        conn.execute(text(
            "INSERT INTO assignments (user_id, experiment_id, variant_id, assigned_at) "
            "VALUES ('u1', 'exp_fanout', 'exp_fanout_treatment', '2025-01-01')"
        ))
        conn.execute(text(
            "INSERT INTO events (event_id, user_id, experiment_id, event_type, "
            "event_timestamp, value) VALUES "
            "('ev1', 'u1', 'exp_fanout', 'conversion', '2025-01-02', 1.0), "
            "('ev2', 'u1', 'exp_fanout', 'conversion', '2025-01-03', 1.0)"
        ))

    df = get_inference_data(engine, "exp_fanout")
    assert len(df) == 1, f"Expected 1 row (one user), got {len(df)} — fan-out occurred"
    assert df.iloc[0]["converted"] == 1


# --------------------------------------------------------------------- #
# Layer 2: cross-check against the simulator's own conversion set
# --------------------------------------------------------------------- #

def test_get_inference_data_matches_dataframe_groundtruth(isolated_db):
    sim = ExperimentSimulator(n_users=3000, baseline_rate=0.15, true_effect=0.03, seed=21)
    users_df, assignments_df, events_df, _ = sim.generate()

    expected_converted = set(events_df["user_id"])

    config = {"simulator": {
        "n_users": 3000, "baseline_rate": 0.15, "true_effect": 0.03,
        "covariate_correlation": 0.5, "seed": 21, "corrupted_split": None,
    }}
    seed_database(config, database_url="sqlite:///:memory:")

    df = get_inference_data(get_engine(), "exp_seed21")
    actual_converted = set(df[df["converted"] == 1]["user_id"])

    assert actual_converted == expected_converted


# --------------------------------------------------------------------- #
# Layer 3: full pipeline — simulate KNOWN true_effect -> seed -> query ->
# raw_ttest_ci + cuped_adjust
# --------------------------------------------------------------------- #

def test_full_pipeline_inference_end_to_end(isolated_db):
    """
    First test connecting ALL of Phase 1 (simulator), Phase 0 (schema),
    Phase 2 (data_access pattern), and Phase 3 (inference) in one execution.
    """
    config = {"simulator": {
        "n_users": 20000, "baseline_rate": 0.10, "true_effect": 0.02,
        "covariate_correlation": 0.5, "seed": 55, "corrupted_split": None,
    }}
    seed_database(config, database_url="sqlite:///:memory:")

    df = get_inference_data(get_engine(), "exp_seed55")
    control_df, treatment_df = split_by_variant(df)

    result = raw_ttest_ci(
        control_df["converted"].to_numpy(), treatment_df["converted"].to_numpy()
    )

    assert 0.0 < result.point_estimate < 0.05
    assert result.ci_lower > 0

    adj = cuped_adjust(
        treatment_df["converted"].to_numpy(), treatment_df["pre_period_covariate"].to_numpy()
    )
    assert adj.theta != 0.0

def test_full_pipeline_cuped_variance_reduction_matches_theory(isolated_db):
    """
    Closes the gap left by test_full_pipeline_inference_end_to_end (which
    only checked theta != 0.0). This test verifies CUPED's variance
    reduction on REAL seeded, SQL-round-tripped data matches the
    theoretical 100*corr(y,x)^2 prediction — the same cross-consistency
    check already proven on synthetic numpy arrays in
    test_variance_reduction_matches_correlation_squared_at_scale
    (core/inference.py tests), now proven end-to-end through the DB.

    true_effect=0 deliberately — isolates the covariate-outcome relationship
    from any treatment-effect confound, since we're testing CUPED's
    variance-reduction MACHINERY, not effect recovery (already covered by
    test_full_pipeline_inference_end_to_end).
    """
    config = {"simulator": {
        "n_users": 20000, "baseline_rate": 0.10, "true_effect": 0.0,
        "covariate_correlation": 0.5, "seed": 77, "corrupted_split": None,
    }}
    seed_database(config, database_url="sqlite:///:memory:")

    df = get_inference_data(get_engine(), "exp_seed77")

    y = df["converted"].to_numpy(dtype=float)
    x = df["pre_period_covariate"].to_numpy(dtype=float)

    adj = cuped_adjust(y, x)
    vr = variance_reduction_pct(y, adj.y_adjusted)

    measured_corr = float(np.corrcoef(y, x)[0, 1])
    theoretical_vr = 100 * measured_corr**2

    assert vr == pytest.approx(theoretical_vr, abs=3.0), (
        f"CUPED variance reduction on REAL seeded data ({vr:.2f}%) diverges "
        f"from theoretical prediction ({theoretical_vr:.2f}%) by more than "
        f"3pp — the Var(Y_cuped)=Var(Y)(1-rho^2) identity should hold "
        f"regardless of Y being binary, since it's a generic OLS-optimal-"
        f"theta property, not a normality assumption."
    )

# ===================================================================== #
# INTEGRATION GAP CLOSURE — required_sample_size() / mde_curve() against
# real seeded data. Deferred since Phase 2 as low-risk (both take plain
# floats, no DB/JOIN logic to break) — closed here now that
# core/pipeline.py's analyze_experiment() actually wires mde_curve() to a
# REALIZED baseline rate in production, so the wiring deserves its own
# direct test rather than only being exercised incidentally through the
# pipeline's own tests.
# ===================================================================== #

def test_mde_curve_matches_manual_calc_on_real_seeded_data(isolated_db):
    """
    Feeds the REALIZED control baseline rate from a real simulated + seeded
    experiment into mde_curve(), exactly as core/pipeline.py's
    analyze_experiment() does in production, then cross-checks the result
    by feeding the resulting achievable MDE back into
    required_sample_size() — since mde_curve() is required_sample_size()'s
    numerically-solved inverse (via brentq), the round trip should recover
    approximately the original n_per_variant.

    This is the specific "real data" wiring Phase 2 deferred: not that the
    math itself was ever in doubt (already proven via
    test_required_sample_size_matches_hand_derivation in test_validity.py),
    but that a REALIZED baseline_rate pulled from actual DB-seeded data —
    which will never be a round number like 0.10 — flows through cleanly.
    """
    config = {"simulator": {
        "n_users": 4000, "baseline_rate": 0.12, "true_effect": 0.0,
        "covariate_correlation": 0.0, "seed": 200, "corrupted_split": None,
    }}
    seed_database(config, database_url="sqlite:///:memory:")

    df = get_inference_data(get_engine(), "exp_seed200")
    control_df, treatment_df = split_by_variant(df)

    baseline_rate_realized = float(control_df["converted"].mean())
    n_per_variant = min(len(control_df), len(treatment_df))

    assert baseline_rate_realized != 0.12, (
        "This test needs a REALIZED rate that differs from the round "
        "configured value to actually exercise real-data wiring — if this "
        "assertion itself fails, the seed happened to land on an exact "
        "match; change the seed, don't delete the check."
    )

    achievable_mde = mde_curve(baseline_rate_realized, [n_per_variant])[0]

    back_check = required_sample_size(baseline_rate_realized, achievable_mde)
    assert back_check.required_n_per_variant == pytest.approx(n_per_variant, rel=0.02), (
        f"Round-trip mismatch: mde_curve() at n={n_per_variant} gave "
        f"mde={achievable_mde:.4f}, but required_sample_size() at that mde "
        f"requires n={back_check.required_n_per_variant} — these should be "
        f"each other's inverse within brentq's solve tolerance."
    )