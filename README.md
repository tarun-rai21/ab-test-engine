# A/B Test Analysis Engine

Structural enforcement of correct experimentation methodology — SRM detection,
power analysis, CUPED variance reduction, empirical peeking-inflation
correction, and BH-corrected segment analysis that flags effect reversals —
validated against synthetic data with known ground truth, not asserted from
a single run. Exposed both as a Python library, a REST API, and an
interactive report app.

> **Status:** complete. A full audit and the fixes it led to are recorded
> in [`docs/REVIEW.md`](docs/REVIEW.md) — including a miscalibrated
> peeking correction and a cross-experiment data-mixing bug, both fixed
> and covered by regression tests.

## Quick start

```bash
git clone <repo_url>
cd ab-test-engine
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
pytest -m "not slow"
```

Runs in under 10 seconds and requires zero external services — SQLite only,
per NFR5.

**Try the interactive report app:**
```bash
streamlit run app/streamlit_app.py
```

**Try the API** (auto-generated interactive docs at `/docs`):
```bash
uvicorn api.main:app --reload
```

**Run the full validation suite** (release-gate evidence, not a per-commit
check — several minutes):
```bash
pytest validation/ -v -s -m slow --tb=short
```

## Why this exists

Naive `p < 0.05` A/B testing fails in four specific, well-documented ways:

- **Underpowered or over-run tests** — no pre-registered sample size means
  teams either stop too early or run indefinitely past the point of value.
- **Peeking / repeated significance testing** — checking a dashboard daily
  and stopping the moment `p < 0.05` appears inflates the true false-positive
  rate far above the nominal 5%, invisibly.
- **Sample ratio mismatch (SRM)** — a broken randomization pipeline can
  silently invalidate every downstream statistic.
- **Effect heterogeneity** — an aggregate result can mask
  a strong effect in one segment and the opposite effect in another.

This engine makes the relevant checks mandatory and ordered, computes what
naive workflows skip (power, SRM, CI, calibrated peeking correction,
segment-level heterogeneity), and proves its own claims against synthetic
data with a known ground-truth effect — every validation number below was
measured by actually running the code, not asserted from theory.

## Architecture

```mermaid
flowchart TB
    SIM["Data Simulator<br/>data_sim/simulator.py"]
    GT["GroundTruth<br/>data_sim/ground_truth.py"]
    DB[("SQLite / Postgres<br/>db/schema.sql")]
    SIM --> GT
    SIM -->|seed_database| DB

    DATAACCESS["Data Access Layer<br/>core/data_access.py"]
    DB --> DATAACCESS

    VALIDITY["Validity Checker<br/>core/validity.py"]
    INFERENCE["Statistical Core<br/>core/inference.py"]
    SEQUENTIAL["Sequential Engine<br/>core/sequential.py"]
    SEGMENTS["Segment Analysis<br/>core/segments.py"]

    DATAACCESS --> VALIDITY
    DATAACCESS --> INFERENCE
    DATAACCESS --> SEQUENTIAL
    DATAACCESS --> SEGMENTS

    PIPELINE["Orchestration Layer<br/>core/pipeline.py"]
    VALIDITY --> PIPELINE
    INFERENCE --> PIPELINE
    SEQUENTIAL --> PIPELINE
    SEGMENTS --> PIPELINE

    PERSIST["Persistence<br/>core/persistence.py"]
    PIPELINE --> PERSIST
    PERSIST --> DB

    VALIDATION["Validation Harness<br/>validation/*.py"]
    GT -.-> VALIDATION
    PIPELINE -.-> VALIDATION

    API["FastAPI Service<br/>api/main.py"]
    APP["Streamlit App<br/>app/streamlit_app.py"]
    PIPELINE --> API
    PIPELINE --> APP
```

The dashed box (Core Engine) is deliberate: each of its four modules is pure,
independently testable, and has zero awareness of the others except where
the spec explicitly calls for reuse (`core/segments.py` reuses
`core/inference.py`'s `raw_ttest_ci()` directly, mirroring the pooled
estimate). `core/persistence.py` is the *only* place validity and inference
results are allowed to meet, and it's structured so that meeting cannot
happen incorrectly — `trusted` is computed from the SRM result internally,
never accepted as a caller-supplied argument.

**API and Streamlit both call `core/pipeline.py` directly**, not each other
— a deliberate choice (documented in `app/streamlit_app.py`) favoring
simplicity for a local demo tool over an unnecessary HTTP round-trip.

## Project structure

```
ab-test-engine/
├── core/
│   ├── validity.py       # SRM chi-square check, power/MDE calculator
│   ├── inference.py      # Welch's t-test, CUPED adjustment, variance reduction
│   ├── sequential.py     # peeking-FPR simulation, calibrated O'Brien-Fleming boundary
│   ├── segments.py       # per-segment effects, BH correction, effect-reversal flag
│   ├── persistence.py    # the only writer of experiment_results.trusted
│   ├── data_access.py    # the only file with raw SQL
│   └── pipeline.py       # orchestration: ties the above into one report
├── data_sim/
│   ├── simulator.py      # ExperimentSimulator — synthetic data, known ground truth
│   └── ground_truth.py   # immutable record of configured-vs-realized parameters
├── db/
│   ├── schema.sql
│   ├── connection.py
│   └── seed.py           # seed_database() + seed_assignments_for_existing_experiment()
├── api/
│   ├── main.py            # FastAPI app, auto-docs at /docs
│   ├── schemas.py          # Pydantic request/response models
│   └── routers/
│       ├── experiments.py  # POST /experiments, POST /experiments/{id}/assign
│       └── analysis.py     # POST /experiments/{id}/analyze, GET /experiments/{id}/report
├── app/
│   ├── streamlit_app.py    # single-page report entrypoint
│   └── components/         # 5 panels: upload_or_generate, validity, effect, sequential, segment
├── tests/                # fast, closed-form/structural tests (pytest -m "not slow")
├── validation/           # slow, simulation-based proof (release gate, not per-commit)
├── notebooks/            # narrative walkthrough (regenerate: python generate_notebook.py)
├── config/
└── scripts/
```

## Validation evidence

Measured by actually running `validation/*.py`, not asserted — see
`validation/validation_report.md` for the full writeup.

| Claim | Target | Measured | Notes |
|---|---|---|---|
| CI coverage (n=2000, baseline=0.10) | 93%–97% | **94.4%** | Original configuration, 1000 sims. |
| CI coverage (small n=400) | 93%–97% | **94.4%** | Stress-tests Welch's df approximation at smaller n. |
| CI coverage (low baseline=0.02) | 93%–97% | **95.0%** | Stress-tests skew near the probability boundary. |
| CI coverage (high baseline=0.90) | 93%–97% | **95.4%** | Symmetric boundary stress test. |
| CUPED variance reduction @ ρ=0.3, null effect | ≈9.0% | **9.21%** | 200 sims, n=5000/arm. |
| CUPED variance reduction @ ρ=0.7, null effect | ≈49.0% | **48.67%** | Same config. |
| CUPED variance reduction, nonzero true effect | ≈25.0% | **23.87%** | Real, documented ~1pp systematic shift from effect-application mechanics — see validation file docstring. |
| CUPED variance reduction, continuous metric | ≈36.0% | **35.94%** | Proves the identity is metric-agnostic, not binary-only. |
| Naive peeking empirical FPR | ~19.3% (theory) | **19.95%** (399/2000) | 10 checkpoints × 200/arm, flat α=0.05. |
| O'Brien-Fleming corrected FPR | 5% (band 3.5%–7%) | **5.15%** (103/2000) | Calibrated constant c=2.0865; identical data (common random numbers). |
| Corrected FPR under irregular checkpoint spacing | Must beat naive | **4.80%** vs naive **20.45%** | Uneven daily-style spacing; no meaningful degradation vs. even spacing. |
| Segment recovery (deliberately heterogeneous) | Correct sign, magnitude within noise | **Mobile: +0.147** (configured +0.15); **Desktop: -0.070** (configured -0.08) | Effect-reversal flag correctly fired on the significantly opposite segment. |

The naive→corrected trigger-position shift (naive triggers front-loaded at
early checkpoints; corrected triggers essentially never at the first looks
and concentrated late) is direct evidence the correction targets the
intended failure mode — early-look noise — not just a uniform threshold
reduction.

An earlier version of the boundary used c = 1.96 instead of a calibrated
constant; its exact false-positive rate is ~6.9%, and it measured 6.6% at
500 simulations. See `docs/REVIEW.md` (issue H2) for how this was found
and fixed.

## Known limitations (stated directly, not discovered later)

- The O'Brien-Fleming boundary's constant is calibrated exactly (recursive
  numerical integration) for **equally spaced** looks. It remains
  **index-based** (checkpoint count) rather than information-based (actual
  sample-size ratio) — validated empirically under one irregular schedule
  (4.80%), but a Lan–DeMets spending function would remove the assumption.
- Segment analysis tests each segment's effect against zero (BH-corrected
  within each segment column); it has no formal interaction test for
  whether segment effects differ from each other.
- `core/pipeline.py`'s sequential-peeking check is genuinely optional: the
  schema has no column storing an experiment's *planned* checkpoint schedule
  (`n_checkpoints_planned`, `checkpoint_n`), so these must be supplied by the
  caller. This is a real schema gap, not a shortcut.
- `GET /experiments/{id}/report` recomputes the analysis rather than reading
  purely from persisted data, deviating from the original spec's literal
  wording — `experiment_results`'s schema doesn't store enough fields
  (p-value, standard error, etc.) to reconstruct a full report otherwise.
  Documented in `api/routers/analysis.py`.
- The API's `POST /experiments/{id}/assign` only supports `mode="simulate"`;
  real-data ingestion (`mode="ingest"`) is deferred to the Streamlit app's
  upload flow.
- The Streamlit app's CSV upload requires data already shaped like
  `get_inference_data()`'s output (one row per user) — it does not
  reconstruct a full multi-table schema from an arbitrary spreadsheet.
