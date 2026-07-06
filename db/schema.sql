-- ab-test-engine relational schema
-- SQLite-compatible, ANSI-SQL enough to port to Postgres via connection-string swap.
-- Single source of truth for table structure — connection.py executes this file
-- directly; do not redefine tables elsewhere (e.g. as SQLAlchemy ORM models that
-- can silently drift from this file).

CREATE TABLE IF NOT EXISTS users (
    user_id               TEXT PRIMARY KEY,
    signup_date           DATE NOT NULL,
    device_type           TEXT NOT NULL,              -- 'mobile' | 'desktop'
    region                TEXT NOT NULL,
    existing_customer     BOOLEAN NOT NULL,
    pre_period_covariate  REAL                          -- e.g. 30-day pre-period spend/activity
);

CREATE TABLE IF NOT EXISTS experiments (
    experiment_id  TEXT PRIMARY KEY,
    name           TEXT NOT NULL,
    hypothesis     TEXT,
    start_date     DATE NOT NULL,
    end_date       DATE,
    status         TEXT NOT NULL DEFAULT 'running'      -- 'running' | 'stopped' | 'analyzed'
);

CREATE TABLE IF NOT EXISTS variants (
    variant_id     TEXT PRIMARY KEY,
    experiment_id  TEXT NOT NULL REFERENCES experiments(experiment_id),
    name           TEXT NOT NULL,                       -- 'control' | 'treatment'
    split_pct      REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS assignments (
    user_id        TEXT NOT NULL REFERENCES users(user_id),
    experiment_id  TEXT NOT NULL REFERENCES experiments(experiment_id),
    variant_id     TEXT NOT NULL REFERENCES variants(variant_id),
    assigned_at    TIMESTAMP NOT NULL,
    PRIMARY KEY (user_id, experiment_id)
);

CREATE TABLE IF NOT EXISTS events (
    event_id         TEXT PRIMARY KEY,
    user_id          TEXT NOT NULL REFERENCES users(user_id),
    experiment_id    TEXT NOT NULL REFERENCES experiments(experiment_id),
    event_type       TEXT NOT NULL,                     -- 'conversion' | 'revenue' | ...
    event_timestamp  TIMESTAMP NOT NULL,
    value            REAL NOT NULL DEFAULT 1.0
);

CREATE TABLE IF NOT EXISTS experiment_results (
    result_id       TEXT PRIMARY KEY,
    experiment_id   TEXT NOT NULL REFERENCES experiments(experiment_id),
    metric_name     TEXT NOT NULL,
    variant_id      TEXT REFERENCES variants(variant_id),
    segment         TEXT,
    point_estimate  REAL NOT NULL,
    ci_lower        REAL NOT NULL,
    ci_upper        REAL NOT NULL,
    trusted         BOOLEAN NOT NULL DEFAULT 1,  -- FALSE if SRM was flagged at compute time
    method          TEXT NOT NULL,               -- 'raw_ttest' | 'cuped' | 'segment'
    computed_at     TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS sequential_checkpoints (
    checkpoint_id            TEXT PRIMARY KEY,
    experiment_id            TEXT NOT NULL REFERENCES experiments(experiment_id),
    checked_at               TIMESTAMP NOT NULL,
    cumulative_n             INTEGER NOT NULL,
    p_value_at_check         REAL NOT NULL,
    alpha_threshold_at_check REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS srm_checks (
    experiment_id  TEXT PRIMARY KEY REFERENCES experiments(experiment_id),
    chi_sq_stat    REAL NOT NULL,
    p_value        REAL NOT NULL,
    flagged        BOOLEAN NOT NULL,
    checked_at     TIMESTAMP NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_assignments_experiment ON assignments(experiment_id);
CREATE INDEX IF NOT EXISTS idx_events_experiment_user ON events(experiment_id, user_id);
CREATE INDEX IF NOT EXISTS idx_results_experiment ON experiment_results(experiment_id);