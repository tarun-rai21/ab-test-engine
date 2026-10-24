# Project Review — A/B Test Analysis Engine

Review date: 2026-10-20. Scope: every tracked source, test, config, and doc
file plus the notebook's cells and saved outputs. Findings below were
verified by running code unless marked "code reading".

> Status: this document records the state of the project **before** the
> fixes listed in [Remediation log](#remediation-log). Each fix landed as a
> separate commit on the `review-fixes` branch.

---

## 1. Executive summary

**What it is:** an A/B-test analysis engine that enforces an ordered
methodology — SRM → power/MDE → Welch t-test → CUPED → peeking correction →
BH-corrected segment analysis — validated against a simulator with known
ground truth, exposed as a library, a FastAPI service, and a Streamlit app.

**Quality:** well above typical portfolio work. Strengths: pure, single-
responsibility core modules; `trusted` structurally uncomputable outside
persistence; allow-listed SQL identifiers; fan-out-safe aggregation; a
shared naive/corrected simulation loop; empirical validation instead of
assertion.

**Two High issues undermine headline claims:** the peeking correction is
miscalibrated by construction (~7% FPR, not 5%), and multi-experiment
databases silently mix users across experiments (which corrupts CUPED and
segment analysis — visible in the committed notebook's own output). Neither
was caught by tests: every test uses a fresh single-experiment database,
and one unit test asserted the buggy behaviour.

---

## 2. Issue register

### Confirmed bugs

#### H1 — High — Data integrity — user IDs not scoped to an experiment
- **Location:** `data_sim/simulator.py:192`, `db/seed.py` (both seed
  functions), `app/components/upload_or_generate.py` (upload user IDs).
- **Problem:** every simulated population uses `u_0000000…`, and seeding
  skips users that already exist. A second experiment in the same DB joins
  to the *first* experiment's `pre_period_covariate`, `device_type`, etc.
- **Evidence (executed):** fresh DB → CUPED variance reduction 19.3%,
  segments mobile +0.166 / desktop −0.044. Same config seeded as a second
  experiment into a shared DB → CUPED VR **0.01%**, segments +0.081 /
  **+0.080**. The committed notebook (§5 seeds `exp_seed888` into the DB
  already holding `exp_seed777`) prints `desktop: +0.0561` against a
  configured −0.08 with no Simpson's flag — contradicting its own summary.
- **Root cause:** global user-ID namespace + "insert only missing users".
  The Phase 8 in-memory-DB fix narrowed but did not remove it (regenerating
  in one Streamlit session with a different seed/`n_users` re-triggers it;
  the API uses the shared `local.db`).
- **Fix:** experiment-scoped user IDs; delete an experiment's users on
  re-seed.
- **Verification:** a test seeding two experiments into one DB and checking
  segment recovery and CUPED VR for the second.

#### H2 — High — Statistics — O'Brien-Fleming boundary uses c = 1.96
- **Location:** `core/sequential.py`, `alpha_spending_schedule`.
- **Problem:** boundary was `z_k = 1.96·√(K/k)`. True OBF uses
  `c·√(K/k)` with `c` *solved* so total crossing probability equals α.
  With c = 1.96 the final look spends the full 5% on top of earlier looks.
- **Evidence (executed):** 2M-path Brownian Monte Carlo puts this
  boundary's exact type-I error at **6.87%**. Exact recursive integration
  gives K=10 → c = **2.0865** (published tables: 2.087; K=2/5/20 also
  match). 4,000 null simulations through `simulate_peeking_fpr`: current
  **7.75%** (95% CI 6.9–8.6%), calibrated **5.70%** (5.0–6.4%). The
  reported 6.6% was systematic overshoot, not noise; 500 simulations
  (SE ≈ 1.1pp) cannot distinguish 5% from 7%. `tests/test_sequential.py`
  asserted `schedule[-1] == 0.05`, locking the bug in (correct OBF's last
  nominal α ≈ 0.037).
- **Impact:** the project's central peeking claim ("restores FPR near
  nominal") was false; the README's own 4–6.5% band was missed and the test
  band had been widened to 7.5%.
- **Fix:** solve for `c` numerically (deterministic, no new dependency).
- **Verification:** K=1 → 0.05 exactly; K=10 final threshold ≈ 0.0369;
  peeking validation re-run with a tight band and more simulations.

#### M1 — Medium — Logic — checkpoint units inconsistent
- **Location:** `core/sequential.py` (`simulate_peeking_fpr`,
  `sequential_check`), `app/components/sequential_panel.py`,
  `app/streamlit_app.py`. (Code reading.)
- **Problem:** `simulate_peeking_fpr`'s `checkpoint_n` is **per arm**; the
  app stores `cumulative_n = 2 × per_arm` (a total) and labels the input
  "Users per checkpoint"; `sequential_check` computes position as
  `round(total / checkpoint_n)` and silently clamps. Example: planned
  10 × 200, checkpoints at 500/arm and 1000/arm both map to position 10 →
  full α, no correction.
- **Fix:** one unit (total users across both arms) everywhere; flag looks
  beyond the plan instead of silently clamping.

#### M2 — Medium — Logic — trace plot uses a different schedule than the verdict
- **Location:** `app/components/sequential_panel.py` `_render_trace_plot`.
- **Problem:** plots `alpha_spending_schedule(len(logged_checkpoints))`
  while the verdict uses `n_checkpoints_planned` + position.
- **Fix:** plot the planned schedule at each checkpoint's position.

#### M3 — Medium — Robustness — CSV upload (code reading)
- **Location:** `app/components/upload_or_generate.py`.
- Second upload in a session regenerates the same user IDs → primary-key
  `IntegrityError`; `split_pct` hard-coded to 0.5 (a real 90/10 design is
  falsely SRM-flagged); `converted` not validated as 0/1 (NaN crashes
  `int()`); per-row `iterrows` inserts.

#### M4 — Medium — Documentation drift
- README's corrected-FPR target "~4%–6.5%" vs reported 6.6% vs test bound
  7.5%; `validation_report.md` has stale numbers (CUPED 9.07/48.95 vs
  README 9.21/48.67, which a re-run reproduces) and stale "limitations"
  (claims coverage tested at one config; four are tested); README status
  says "Phase 9 in progress"; README cites a missing
  `Phase_8_Documentation.md`; notebook summary contradicted by its output.

#### L1 — Low — API returns 500 for invalid input (executed)
- `covariate_correlation=-0.6` (simulator `RuntimeError`/`ValueError`),
  `corrupted_split=1.5`, duplicate variant names (PK error) → all 500.

#### L2 — Low — engine singleton ignores `database_url`
- `seed_database(..., database_url=...)` sets an env var, but
  `get_engine()` is a process singleton, so the URL is ignored if an engine
  already exists. Related: in Streamlit, a second browser session calls
  `reset_engine()` and wipes the first session's in-memory DB, contrary to
  the docstring's "never shared" claim.

#### L3 — Low — `segment_effects_realized` is the configured dict
- It copies `seg_map`; clipping to [0,1] changes the realized effect (e.g.
  desktop −0.08 on a 0.10 baseline with ρ=0.5 realizes ≈ −0.044).

#### L4 — Low — misc
- `if __name__ == "__main__":` sits mid-file in `db/seed.py`; `srm_checks`
  table is never written; `alpha_threshold_at_check` always stored as
  0.05; `.idea/` is gitignored but tracked.

### Design concerns (defensible, but know them)

- **D1 — "Simpson's paradox" flag is a sign-disagreement flag.** It fires
  on any opposite-sign point estimate, even with a CI spanning zero. Under
  randomization true Simpson's paradox essentially cannot occur (segment mix
  is balanced, so the pooled effect ≈ weighted average of segment effects).
  What is detected is qualitative effect heterogeneity; it should require
  BH significance and ideally an interaction test.
- **D2 — BH is applied per segment column**, not across all columns tested.
- **D3 — peeking check requires ≥2 checkpoints**; the first look (the most
  noise-prone) is skipped.
- **D4 — index-based schedule.** Correct only for equally spaced looks;
  Lan-DeMets α(t) = 2 − 2Φ(z_{α/2}/√t) on information fraction is the
  principled extension.
- **D5 — CUPED "unbiased for ANY θ"** holds for fixed θ; θ is estimated on
  the same data (including the treated arm), so the estimator is
  asymptotically unbiased with O(1/n) bias, and the SE ignores θ
  estimation. Negligible at these sample sizes.
- **D6 — CUPED VR at ρ=0.3** measured 9.21% vs theoretical 9.0% (2.9 SE);
  likely binary-outcome/skewed-covariate effects, not a bug.
- **D7 —** `mde_curve` root-finds a `ceil` step function (fine to ±1 user);
  report rows persisted in separate transactions; no tests for `app/`.

---

## 3. Verification status (pre-fix)

| Item | Result |
|---|---|
| All tracked files + notebook outputs | Read |
| `pytest -m "not slow"` | 113 passed (Python 3.13) |
| `ruff check .` | clean |
| `pytest validation/ -m slow` | 12 passed (7m21s). Coverage 94.4/94.4/95.0/95.4%; CUPED 9.21/48.67%; naive FPR 22.0%; corrected 6.6%; irregular 21.4→6.8%; segments +0.147/−0.070 |
| H1, H2, L1 | Reproduced with scripts |
| M1, M2, M3, L2 | Code reading; Streamlit UI not driven |
| Not verified | Postgres; Streamlit rendering; GitHub CI |

Passing tests are not proof of correctness: H2 was *encoded in* a passing
unit test, and H1 is invisible to tests that each start from a fresh DB.

---

## 4. How the project works

### Architecture
```
ExperimentSimulator ──► SQLite (users, experiments, variants, assignments, events)
                                   │  core/data_access.py (only file with SQL)
                                   ▼
           validity ── inference ── sequential ── segments   (pure, DB-free)
                                   │
                         core/pipeline.analyze_experiment()
                          │                 │
                core/persistence      FastAPI / Streamlit / notebook
             (sole writer of `trusted`)
```

### `analyze_experiment` flow
1. `get_variant_counts` (LEFT JOIN so zero-traffic variants appear) →
   `srm_check` (chi-square, flag at p < 0.001).
2. `get_inference_data` → one row per user; `COUNT(event) > 0` makes a 0/1
   conversion flag regardless of event cardinality.
3. Welch t-test on conversion → difference, Welch–Satterthwaite CI,
   p-value; persisted with `trusted = not srm.flagged`.
4. Achievable MDE by inverting the two-proportion sample-size formula at
   the observed baseline and smaller arm's n.
5. CUPED: θ = Cov(Y,X)/Var(X) pooled, Y′ = Y − θ(X − X̄), Welch again.
6. Optional peeking check: latest checkpoint p-value vs. its position's
   threshold.
7. Optional segments: Welch per segment value, BH, reversal flag.

### Key maths
- Welch SE = √(s₁²/n₁ + s₂²/n₂); on 0/1 data at large n this equals the
  unpooled two-proportion z-test.
- n = (z_{α/2} + z_β)² [p₁(1−p₁) + p₂(1−p₂)] / δ².
- CUPED: Var(Y′) = Var(Y)(1 − ρ²) at the OLS θ → VR = 100ρ².
- Peeking: corr(Z_i, Z_j) = √(n_i/n_j); a flat 1.96 at 10 looks gives
  ≈19% (z-test), 22% measured with t-tests on binary data. OBF:
  b_k = c√(K/k), flat at c√K on the cumulative-sum scale — which is why
  `c` can be found exactly by integrating a Gaussian random walk inside a
  strip.
- BH: sort p-values, largest k with p₍k₎ ≤ (k/m)q, reject ranks 1..k
  (controls FDR, not FWER).

### Design decisions
SRM at 0.001 (frequent checks); Welch over Student (unequal variances);
pooled θ; BH over Bonferroni (exploratory); OBF over Pocock (little early
α, near-full final power); SQLAlchemy Core (visible SQL); app calls the
pipeline directly (one process); GET `/report` recomputes (schema lacks
p-value/SE).

### Limitations (after fixes)
Two arms, binary metric only (no revenue/ratio metrics or delta method);
index-based look spacing; no API ingestion path; simulator cannot reach
strongly negative covariate correlations; no segment interaction test.

### Interview preparation
Why SRM at 0.001; Welch on binary data vs z-test; derive CUPED's 1 − ρ²;
why the covariate must be pre-treatment; why peeking inflates error and
why OBF's final look uses α < 0.05 (H2 is a strong story: own validation
2pp high → traced to an uncalibrated constant → fixed with exact recursive
integration); BH vs Bonferroni; why randomized experiments rarely show
true Simpson's paradox; Lan-DeMets for unequal spacing; how to add revenue
metrics (heavy tails, delta method, winsorization).

---

## 5. Final assessment (pre-fix)

- **Readiness:** research/portfolio grade pending H1/H2. Core estimators
  (SRM, Welch CI, CUPED, BH) are correct and well validated. Not
  production-ready: no auth, no real ingestion, two arms/binary only,
  shared global engine across Streamlit sessions, untested UI.
- **Evidence required to trust results:** a two-experiments-in-one-DB
  regression test; an exact-schedule unit test; a re-run peeking
  validation near 5%; a regenerated notebook where desktop is negative and
  the reversal flag fires.
- **Optional next steps:** Lan-DeMets information-based spending,
  heterogeneity interaction test, delta-method revenue metrics, a schema
  column for the planned checkpoint schedule, persisting p-value/SE so GET
  `/report` need not recompute.

---

## Remediation log

Each fix landed as a separate commit on `review-fixes`.

| Fix | Issues | What changed | Evidence |
|---|---|---|---|
| 1 | H1 | Experiment-scoped user IDs; re-seeding deletes the experiment's users; uploads prefix user IDs | Two regression tests that fail on the old code (CUPED VR 0.006% → >10%; stale user count 5000 → 3000) |
| 2 | H2 | O'Brien-Fleming constant solved by exact recursive integration (K=10 → c=2.0865, matches published tables) | Corrected FPR 6.6% → **5.15%** (2000 sims), irregular spacing 4.80%; independent Monte Carlo unit test at 5.0% ± 0.2pp |
| 3 | M1, M2, D3 | Total-users units for live checks; `beyond_plan` flag; plot uses the planned schedule; single checkpoint evaluated | New unit and pipeline tests |
| 4 | D1 | `simpsons_flag` → `reversal_flag`, gated on BH significance | Test that a noisy opposite-sign segment is not flagged |
| 5 | M3, L1, L2 | Upload validation + intended split + bulk inserts; API 422s; URL-keyed engine singleton; per-session Streamlit engine | API, upload and connection tests; headless Streamlit AppTest run with no exceptions |
| 6 | M4, L3, L4 | Realized segment effects computed after clipping; `srm_checks` persisted; checkpoint threshold stored; `.idea` untracked; README, validation report and notebook regenerated | Full slow suite re-run; notebook now shows desktop −0.089 flagged as a reversal (was +0.056, unflagged) |

### Additional issues found while fixing

- **Upload crashed without all segment columns** (found by Fix 5's tests):
  `users.device_type/region/existing_customer` were `NOT NULL`, so any CSV
  missing an "optional" segment column failed with an IntegrityError. Now
  nullable; segment analysis already drops NULL segment values.
- **Naive-peeking band floor sat above theory:** the exact null crossing
  probability for a flat 1.96 at 10 equal looks is ~19.3%, but the test
  required ≥20%; the old 22.0% was sampling noise at 500 simulations. The
  band is now centred on theory.
- **Deprecated Streamlit argument:** `st.dataframe(use_container_width=...)`
  replaced with `width="stretch"`.

### Follow-up: Lan-DeMets boundary (design concern D4)

Resolved after the review on branch `lan-demets-boundary`:
`core.sequential.lan_demets_boundary()` computes boundaries from each
look's information fraction via the same recursive-integration approach,
with an O'Brien-Fleming-type spending function. It is the default for
live checks (`spending="lan_demets"`); the index-based schedule stays
available. Matches published ldbounds values (K=5: z = 4.877, 3.357,
2.680, 2.290, 2.031) to ~0.001; independent Monte Carlo type-I error
5.0% ± 0.2pp under irregular spacing; t-test simulation FPR 5.25%
(equal) and 5.05% (irregular) over 2000 runs.

### Follow-up: remaining design limitations

Resolved on branch `close-remaining-limitations`, one commit each:

| Limitation | Resolution | Evidence |
|---|---|---|
| Planned schedule not stored | `experiments.n_checkpoints_planned` / `checkpoint_n` with in-place column migrations; pipeline falls back to the stored schedule; `PUT /experiments/{id}/schedule` | Schedule and migration tests |
| Binary metric only | `revenue_per_user` (Welch, optional winsorization) and `revenue_per_conversion` (delta method); simulator generates log-normal order values | CI coverage 95.2% / 94.8%; delta SE matches bootstrap |
| Two arms only | A/B/n via variant names; Holm-adjusted comparisons; per-arm segments; simulator allocation follows the declared split (also fixes a 90/10 design being SRM-flagged) | FWER 4.45% Holm vs 13.05% unadjusted |
| No interaction test; BH per column | Cochran's Q heterogeneity test; one BH across every segment test in a report | Q rejection 4.60% (null), 88.5% (theory 87%) |
| GET `/report` recomputes | Full report + parameters stored in `analysis_reports`; result rows store p-value, SE, df, arm sizes; GET serves the stored report or 404 | Tests that the stored report survives data changes and matches POST |

### Formerly open, since resolved

Step-down Dunnett is now the default multi-arm adjustment (Holm kept as
an option); looks are recorded per (metric, arm); the ratio metric has
CUPED via its delta-method linearization; Streamlit has headless AppTest
UI tests; the test client uses `httpx2`.

---

## Second review (2026-10-25)

Scope: every tracked file again, after the A/B/n, Dunnett, revenue-metric,
monitoring and ingestion work. The statistical core held up: no new
estimator or boundary bug was found, and the full slow suite was re-run.
The findings are at the edges: UI state, input validation, persistence
hygiene and documentation. All were reproduced with scripts before fixing,
and each fix has a regression test that fails on the old code.

| ID | Severity | Finding | Fix |
|---|---|---|---|
| R1 | Medium | Streamlit kept showing the old report after regenerating data with the same seed (same `exp_seed{seed}` id, all rows replaced): a 2,000-user report stayed on screen over 6,000-user data. | A per-session data version; a report is shown only for the data it was computed on. |
| R2 | Medium | The first look past the planned total spends the remaining alpha and CAN reject (e.g. t=1.1 → threshold 0.0489), but the panel said it "cannot produce a significant result", and the docstring said every look past t=1 gets threshold 0. | Panel distinguishes "budget already spent" (threshold 0) from "final analysis past plan"; docstring corrected; test added. |
| R3 | Low | Unticking "Enable peeking-risk check" did not disable it: the pipeline falls back to a schedule stored by an earlier run. | The app honours the checkbox when rendering. |
| R4 | Low | Ingestion stored unrecognised `existing_customer` values ("unknown", "N/A") as `True` (`bool(NaN)` is `True`). | `validate_canonical` reports them. |
| R5 | Low | A variant with `split_pct = 0` made the SRM statistic NaN, which reads as "not flagged". | API requires `split_pct > 0`; `srm_check` rejects non-positive ratios. |
| R6 | Low | `/assign` with `segment_heterogeneity` on an unknown column returned 500. | Simulator validates the column → 422. |
| R7 | Low | Re-analysis overwrote result rows by id only, so rows it did not produce (e.g. old segment rows computed on replaced data) stayed in `experiment_results`, marked trusted. | Each persisted analysis first clears that metric's rows, in the same transaction. |
| R8 | Low | Docs drift: this file's "Still open" list, the validation report's limitations (claimed revenue metrics were unvalidated), `inference.py` calling persistence "not yet built" and CUPED "unbiased for ANY theta". | Updated. |
| R9 | Gap | CUPED interval coverage (the default headline estimate) was never validated, only its variance reduction. | `test_cuped_ci_coverage_through_the_pipeline`: **94.6%** (473/500) at ρ=0.5, n=2000. |

Design points kept as they are (documented, defensible): the peeking
verdict uses the raw Welch p-value while the effect panel leads with
CUPED and a naive 95% interval; peeking control is per arm, not joint;
looks recorded before any plan existed are re-thresholded under the plan
set later; the Streamlit replay takes equal users per variant even under
unequal allocation; `GroundTruth.true_effect_realized` averages over all
arms in A/B/n simulations.
