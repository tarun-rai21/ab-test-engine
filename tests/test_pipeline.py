"""
tests/test_pipeline.py — end-to-end tests for core/pipeline.py, the
orchestration layer tying validity, inference, sequential, and persistence
together for a single experiment.

This file has a second job beyond testing pipeline.py itself: it is what
officially closes the gap flagged (and left open) throughout Phase 5's
documentation — core.sequential.sequential_check() had only ever been
exercised against hand-built dictionaries, never a REAL, database-persisted
sequential_checkpoints table. test_sequential_check_against_real_persisted_checkpoints
below is that missing integration test.

Same three-layer discipline established in tests/test_data_access.py:
  1. Known-answer test against hand-inserted rows (isolates pipeline wiring).
  2. Full pipeline via the simulator + seed_database (realistic data shape).
  3. The specific gap this file exists to close (real sequential_checkpoints).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import text

from core.pipeline import analyze_experiment
from data_sim.simulator import ExperimentSimulator
from db.connection import get_engine, init_schema, reset_engine
from db.seed import seed_database


@pytest.fixture(autouse=True)
def isolated_db(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    reset_engine()
    engine = get_engine()
    init_schema(engine)
    yield engine
    reset_engine()


def _insert_checkpoint(engine, experiment_id, checkpoint_id, cumulative_n, p_value, alpha=0.05):
    """Hand-inserts one real sequential_checkpoints row — the table
    get_sequential_checkpoints() reads from, and which nothing in the
    codebase wrote to or read from before this file."""
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO sequential_checkpoints
                    (checkpoint_id, experiment_id, checked_at, cumulative_n,
                     p_value_at_check, alpha_threshold_at_check)
                VALUES
                    (:cid, :eid, :checked_at, :cum_n, :pval, :alpha)
                """
            ),
            {
                "cid": checkpoint_id,
                "eid": experiment_id,
                "checked_at": datetime.now(timezone.utc).isoformat(),
                "cum_n": cumulative_n,
                "pval": p_value,
                "alpha": alpha,
            },
        )


# ===================================================================== #
# Layer 1: known-answer — hand-inserted rows, isolates pipeline wiring
# from the simulator entirely
# ===================================================================== #

def test_analyze_experiment_known_answer_computes_and_persists(isolated_db):
    """
    Hand-built 10 control / 10 treatment users, deterministic conversion
    pattern, healthy 50/50 split. Verifies the pipeline's wiring — not the
    underlying stats, which are already proven correct in isolation — by
    checking a raw effect estimate materializes AND a row lands in
    experiment_results with trusted=1 (SRM not flagged).
    """
    engine = isolated_db
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO experiments (experiment_id, name, start_date, status) "
            "VALUES ('exp_known', 'test', '2025-01-01', 'running')"
        ))
        conn.execute(text(
            "INSERT INTO variants (variant_id, experiment_id, name, split_pct) VALUES "
            "('exp_known_control', 'exp_known', 'control', 0.5), "
            "('exp_known_treatment', 'exp_known', 'treatment', 0.5)"
        ))
        for i in range(10):
            conn.execute(text(
                "INSERT INTO users (user_id, signup_date, device_type, region, "
                "existing_customer, pre_period_covariate) "
                f"VALUES ('u_c{i}', '2025-01-01', 'mobile', 'US', 0, {1.0 + i * 0.1})"
            ))
            conn.execute(text(
                f"INSERT INTO assignments (user_id, experiment_id, variant_id, assigned_at) "
                f"VALUES ('u_c{i}', 'exp_known', 'exp_known_control', '2025-01-01')"
            ))
            # Control converts on even indices only: 5/10 = 0.5 baseline rate
            if i % 2 == 0:
                conn.execute(text(
                    f"INSERT INTO events (event_id, user_id, experiment_id, event_type, "
                    f"event_timestamp, value) VALUES "
                    f"('ev_c{i}', 'u_c{i}', 'exp_known', 'conversion', '2025-01-02', 1.0)"
                ))
        for i in range(10):
            conn.execute(text(
                "INSERT INTO users (user_id, signup_date, device_type, region, "
                "existing_customer, pre_period_covariate) "
                f"VALUES ('u_t{i}', '2025-01-01', 'mobile', 'US', 0, {1.0 + i * 0.1})"
            ))
            conn.execute(text(
                f"INSERT INTO assignments (user_id, experiment_id, variant_id, assigned_at) "
                f"VALUES ('u_t{i}', 'exp_known', 'exp_known_treatment', '2025-01-01')"
            ))
            # Treatment converts on 8/10 -> higher rate than control
            if i < 8:
                conn.execute(text(
                    f"INSERT INTO events (event_id, user_id, experiment_id, event_type, "
                    f"event_timestamp, value) VALUES "
                    f"('ev_t{i}', 'u_t{i}', 'exp_known', 'conversion', '2025-01-02', 1.0)"
                ))

    report = analyze_experiment(engine, "exp_known")

    assert not report.srm.flagged
    assert report.raw_effect.point_estimate == pytest.approx(0.3, abs=1e-9)  # 0.8 - 0.5
    assert report.raw_effect.n_control == 10
    assert report.raw_effect.n_treatment == 10

    row = engine.connect().execute(
        text("SELECT trusted, method FROM experiment_results WHERE experiment_id = 'exp_known' "
             "AND method = 'raw_ttest'")
    ).fetchone()
    assert row is not None, "Expected analyze_experiment to persist a raw_ttest row"
    assert row.trusted == 1


def test_analyze_experiment_srm_flagged_still_computes_but_untrusted(isolated_db):
    """
    THE core contract carried over from persist_inference_result's own test:
    an SRM-flagged experiment must still get a computed effect estimate
    (spec 4.4 step 2 — downstream modules keep running), but the persisted
    row must be trusted=0.
    """
    engine = isolated_db
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO experiments (experiment_id, name, start_date, status) "
            "VALUES ('exp_bad_srm', 'test', '2025-01-01', 'running')"
        ))
        conn.execute(text(
            "INSERT INTO variants (variant_id, experiment_id, name, split_pct) VALUES "
            "('exp_bad_srm_control', 'exp_bad_srm', 'control', 0.5), "
            "('exp_bad_srm_treatment', 'exp_bad_srm', 'treatment', 0.5)"
        ))
        # Deliberately lopsided: 18 control, 2 treatment, against a 50/50 target.
        # pre_period_covariate is set explicitly (not left NULL) — a NULL here
        # becomes NaN once read back, which would otherwise slip past
        # cuped_adjust()'s zero-variance guard silently.
        for i in range(18):
            conn.execute(text(
                "INSERT INTO users (user_id, signup_date, device_type, region, "
                "existing_customer, pre_period_covariate) "
                f"VALUES ('u_c{i}', '2025-01-01', 'mobile', 'US', 0, {1.0 + i * 0.1})"
            ))
            conn.execute(text(
                f"INSERT INTO assignments (user_id, experiment_id, variant_id, assigned_at) "
                f"VALUES ('u_c{i}', 'exp_bad_srm', 'exp_bad_srm_control', '2025-01-01')"
            ))
            if i % 2 == 0:  # 9 of 18 control users convert
                conn.execute(text(
                    f"INSERT INTO events (event_id, user_id, experiment_id, event_type, "
                    f"event_timestamp, value) VALUES "
                    f"('ev_c{i}', 'u_c{i}', 'exp_bad_srm', 'conversion', '2025-01-02', 1.0)"
                ))
        for i in range(2):
            conn.execute(text(
                "INSERT INTO users (user_id, signup_date, device_type, region, "
                "existing_customer, pre_period_covariate) "
                f"VALUES ('u_t{i}', '2025-01-01', 'mobile', 'US', 0, {1.0 + i * 0.1})"
            ))
            conn.execute(text(
                f"INSERT INTO assignments (user_id, experiment_id, variant_id, assigned_at) "
                f"VALUES ('u_t{i}', 'exp_bad_srm', 'exp_bad_srm_treatment', '2025-01-01')"
            ))
            if i == 0:  # 1 of 2 treatment users converts -> nonzero variance
                conn.execute(text(
                    f"INSERT INTO events (event_id, user_id, experiment_id, event_type, "
                    f"event_timestamp, value) VALUES "
                    f"('ev_t{i}', 'u_t{i}', 'exp_bad_srm', 'conversion', '2025-01-02', 1.0)"
                ))

    report = analyze_experiment(engine, "exp_bad_srm")

    assert report.srm.flagged
    assert report.raw_effect is not None  # computed anyway, not skipped

    row = engine.connect().execute(
        text("SELECT trusted FROM experiment_results WHERE experiment_id = 'exp_bad_srm' "
             "AND method = 'raw_ttest'")
    ).fetchone()
    assert row.trusted == 0


# ===================================================================== #
# Layer 2: full pipeline via the simulator — realistic data shape,
# known true_effect and known covariate correlation
# ===================================================================== #

def test_analyze_experiment_full_pipeline_with_cuped(isolated_db):
    config = {"simulator": {
        "n_users": 20000, "baseline_rate": 0.10, "true_effect": 0.02,
        "covariate_correlation": 0.5, "seed": 55, "corrupted_split": None,
    }}
    seed_database(config, database_url="sqlite:///:memory:")

    report = analyze_experiment(get_engine(), "exp_seed55")

    assert not report.srm.flagged
    assert 0.0 < report.raw_effect.point_estimate < 0.05
    assert report.raw_effect.ci_lower > 0  # clearly positive, matches Phase 3's own test

    assert report.cuped_effect is not None
    assert report.cuped_effect.method == "cuped"
    assert report.cuped_variance_reduction_pct is not None
    assert report.cuped_variance_reduction_pct > 0  # correlation=0.5 -> real reduction expected

    assert report.achievable_mde is not None
    assert 0.0 < report.achievable_mde < 1.0

    # Both raw and cuped rows must be persisted, as two SEPARATE rows —
    # matches persist_inference_result's own established test
    # (test_different_methods_produce_separate_rows).
    count = get_engine().connect().execute(
        text("SELECT COUNT(*) FROM experiment_results WHERE experiment_id = 'exp_seed55'")
    ).fetchone()[0]
    assert count == 2


def test_use_cuped_false_skips_cuped_entirely(isolated_db):
    config = {"simulator": {
        "n_users": 5000, "baseline_rate": 0.10, "true_effect": 0.02,
        "covariate_correlation": 0.5, "seed": 66, "corrupted_split": None,
    }}
    seed_database(config, database_url="sqlite:///:memory:")

    report = analyze_experiment(get_engine(), "exp_seed66", use_cuped=False)

    assert report.cuped_effect is None
    assert report.cuped_variance_reduction_pct is None

    count = get_engine().connect().execute(
        text("SELECT COUNT(*) FROM experiment_results WHERE experiment_id = 'exp_seed66'")
    ).fetchone()[0]
    assert count == 1, "use_cuped=False must persist only the raw_ttest row, not a cuped row too"


def test_persist_false_computes_without_writing_to_db(isolated_db):
    config = {"simulator": {
        "n_users": 5000, "baseline_rate": 0.10, "true_effect": 0.02,
        "covariate_correlation": 0.5, "seed": 77, "corrupted_split": None,
    }}
    seed_database(config, database_url="sqlite:///:memory:")

    report = analyze_experiment(get_engine(), "exp_seed77", persist=False)

    assert report.raw_effect is not None  # still computed
    count = get_engine().connect().execute(
        text("SELECT COUNT(*) FROM experiment_results WHERE experiment_id = 'exp_seed77'")
    ).fetchone()[0]
    assert count == 0, "persist=False must not write anything to experiment_results"


# ===================================================================== #
# Layer 3: THE gap this file exists to close — sequential_check() against
# a REAL, database-persisted sequential_checkpoints table, not hand-typed
# dicts. Reproduces the exact hand-verified scenario from Phase 5's own
# unit tests (checkpoint 4 of 10, naive says significant, corrected
# disagrees) — but now the checkpoint data comes from a real INSERT/SELECT
# round trip through get_sequential_checkpoints(), never exercised before.
# ===================================================================== #

def test_sequential_check_against_real_persisted_checkpoints(isolated_db):
    engine = isolated_db
    config = {"simulator": {
        "n_users": 2000, "baseline_rate": 0.10, "true_effect": 0.0,
        "covariate_correlation": 0.0, "seed": 88, "corrupted_split": None,
    }}
    seed_database(config, database_url="sqlite:///:memory:")

    _insert_checkpoint(engine, "exp_seed88", "cp1", 200, 0.40)
    _insert_checkpoint(engine, "exp_seed88", "cp2", 400, 0.15)
    _insert_checkpoint(engine, "exp_seed88", "cp3", 600, 0.06)
    _insert_checkpoint(engine, "exp_seed88", "cp4", 800, 0.03)

    report = analyze_experiment(
        engine, "exp_seed88", n_checkpoints_planned=10, checkpoint_n=200
    )

    assert report.sequential_risk is not None, (
        "sequential_risk was None — get_sequential_checkpoints() likely "
        "isn't reading the real table, or the >1-row gate in analyze_experiment "
        "is misfiring."
    )
    assert report.sequential_risk.checkpoint_position == 4
    assert report.sequential_risk.naive_significant is True
    assert not report.sequential_risk.corrected_significant
    assert report.sequential_risk.disagreement is True


def test_sequential_check_skipped_with_zero_or_one_checkpoint(isolated_db):
    """
    Zero checkpoints (the normal, common state) and exactly one checkpoint
    (no repeated-look risk to flag yet) must both leave sequential_risk=None
    — the >1-row gate documented in core/pipeline.py's design decision 4.
    """
    engine = isolated_db
    config = {"simulator": {
        "n_users": 2000, "baseline_rate": 0.10, "true_effect": 0.0,
        "covariate_correlation": 0.0, "seed": 99, "corrupted_split": None,
    }}
    seed_database(config, database_url="sqlite:///:memory:")

    report_zero = analyze_experiment(
        engine, "exp_seed99", n_checkpoints_planned=10, checkpoint_n=200
    )
    assert report_zero.sequential_risk is None

    _insert_checkpoint(engine, "exp_seed99", "cp1", 200, 0.03)
    report_one = analyze_experiment(
        engine, "exp_seed99", n_checkpoints_planned=10, checkpoint_n=200
    )
    assert report_one.sequential_risk is None


def test_sequential_check_skipped_without_planned_params(isolated_db):
    """
    Checkpoints exist in the DB, but the caller didn't supply
    n_checkpoints_planned/checkpoint_n — must skip gracefully (None), not
    guess or crash. See core/pipeline.py's design decision 4 for why these
    can't be inferred from the schema.
    """
    engine = isolated_db
    config = {"simulator": {
        "n_users": 2000, "baseline_rate": 0.10, "true_effect": 0.0,
        "covariate_correlation": 0.0, "seed": 111, "corrupted_split": None,
    }}
    seed_database(config, database_url="sqlite:///:memory:")

    _insert_checkpoint(engine, "exp_seed111", "cp1", 200, 0.40)
    _insert_checkpoint(engine, "exp_seed111", "cp2", 400, 0.03)

    report = analyze_experiment(engine, "exp_seed111")  # no planned params
    assert report.sequential_risk is None


# ===================================================================== #
# Segments — explicitly deferred, must not silently produce something
# ===================================================================== #

def test_segments_field_is_always_none_pending_phase_6(isolated_db):
    config = {"simulator": {
        "n_users": 2000, "baseline_rate": 0.10, "true_effect": 0.0,
        "covariate_correlation": 0.0, "seed": 122, "corrupted_split": None,
    }}
    seed_database(config, database_url="sqlite:///:memory:")

    report = analyze_experiment(get_engine(), "exp_seed122")
    assert report.segments is None