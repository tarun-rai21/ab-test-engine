"""
ExperimentSimulator — Phase 1's central artifact.

Generates synthetic experiment data (users, assignments, events) with a KNOWN,
recorded ground-truth treatment effect — this is what makes every later
validation claim (CI coverage, CUPED variance reduction, peeking FPR)
checkable against truth instead of asserted from a single run.

Key design decisions and known limitations, all discovered empirically, not
assumed up front — see inline comments at each site for the specific evidence:

1. segment_heterogeneity effects are ADDITIVE on top of true_effect.
2. user_id / experiment_id / event_id are deterministic functions of seed —
   no uuid.uuid4() anywhere, required for NFR2 (fixed seed -> identical output).
3. Calibration solves (intercept, slope) JOINTLY via damped alternating 1D
   root-finds against a large sample from the SAME distribution family as the
   real covariate (log-normal, standardized) — not a standard-normal proxy,
   and not intercept-alone-via-logit(baseline_rate).
4. KNOWN LIMITATION: because pre_period_covariate is log-normal (right-skewed)
   even after standardization, the ACHIEVABLE covariate_correlation range is
   NOT symmetric around zero. Positive correlations align with the long right
   tail (high leverage) — verified reachable up to at least +0.5. Negative
   correlations must work against that skew (low leverage) — target_corr=-0.4
   at baseline_rate=0.10 was empirically found to be unreachable: the joint
   calibration iteration drives the intercept toward extreme saturation
   (observed as low as -14.4) without converging, and correctly raises rather
   than silently returning a miscalibrated result. If you need strong negative
   correlation, either increase baseline_rate (moves the operating point away
   from sigmoid's most saturated region) or treat it as explicitly out of
   scope and choose a smaller-magnitude target.
"""

from __future__ import annotations

from typing import Callable

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit

from data_sim.ground_truth import GroundTruth

DEVICE_TYPES = ("mobile", "desktop")
REGIONS = ("US", "EU", "APAC")
DEVICE_PROBS = (0.6, 0.4)
REGION_PROBS = (0.5, 0.3, 0.2)
EXISTING_CUSTOMER_PROB = 0.3

_CALIBRATION_N = 50_000
_CALIBRATION_SLOPE_BOUNDS = (0.0, 20.0)
_JOINT_CALIBRATION_ITERS = 25
_CALIBRATION_DAMPING = 0.6
_CALIBRATION_CONVERGENCE_TOL = 0.01  # on calibration-SAMPLE gaps, not the generated-sample check
_COVARIATE_LOGNORMAL_SIGMA = 0.75
_BRACKET_INITIAL_HALFWIDTH = 2.0
_BRACKET_MAX_EXPANSIONS = 12


class ExperimentSimulator:
    def __init__(
        self,
        n_users: int,
        baseline_rate: float,
        true_effect: float,
        covariate_correlation: float = 0.5,
        seed: int = 42,
        corrupted_split: float | None = None,
        segment_heterogeneity: dict[str, dict[str, float]] | None = None,
        experiment_id: str | None = None,
    ):
        if not (0.0 < baseline_rate < 1.0):
            raise ValueError(f"baseline_rate must be in (0,1), got {baseline_rate}")
        if not (-0.99 < covariate_correlation < 0.99):
            raise ValueError(f"covariate_correlation must be in (-0.99, 0.99), got {covariate_correlation}")
        if segment_heterogeneity and len(segment_heterogeneity) > 1:
            raise ValueError(
                "Only one segment column may carry configured heterogeneity per generate() call."
            )

        self.n_users = n_users
        self.baseline_rate = baseline_rate
        self.true_effect = true_effect
        self.covariate_correlation_target = covariate_correlation
        self.seed = seed
        self.corrupted_split = corrupted_split
        self.segment_heterogeneity = segment_heterogeneity
        self.experiment_id = experiment_id or f"exp_seed{seed}"
        self.segment_column = next(iter(segment_heterogeneity)) if segment_heterogeneity else None

    # ------------------------------------------------------------------ #

    def generate(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, GroundTruth]:
        rng = np.random.default_rng(self.seed)

        users_df = self._generate_users(rng)
        z = self._standardize(users_df["pre_period_covariate"].to_numpy())

        intercept, slope, _realized_corr_calib, _realized_mean_calib = self._calibrate()

        p_baseline = expit(intercept + slope * z)

        variant = self._assign_variant(rng)
        effect_per_user, segment_effects_realized = self._compute_effect_per_user(users_df)

        p_final = p_baseline.copy()
        is_treatment = variant == "treatment"
        p_final[is_treatment] = np.clip(p_baseline[is_treatment] + effect_per_user[is_treatment], 0.0, 1.0)

        converted = rng.binomial(1, p_final)

        realized_effect = (
            float(p_final[is_treatment].mean() - p_baseline[is_treatment].mean())
            if is_treatment.any() else 0.0
        )
        realized_corr = float(np.corrcoef(users_df["pre_period_covariate"], converted)[0, 1])
        realized_baseline_rate = float(p_baseline.mean())

        assignments_df = self._build_assignments(users_df["user_id"], variant)
        events_df = self._build_events(users_df["user_id"], converted)

        ground_truth = GroundTruth(
            n_users=self.n_users,
            baseline_rate=self.baseline_rate,
            true_effect_configured=self.true_effect,
            covariate_correlation_target=self.covariate_correlation_target,
            seed=self.seed,
            corrupted_split=self.corrupted_split,
            segment_column=self.segment_column,
            segment_effects_configured=(
                self.segment_heterogeneity[self.segment_column] if self.segment_column else None
            ),
            calibrated_intercept=intercept,
            calibrated_slope=slope,
            baseline_rate_realized=realized_baseline_rate,
            covariate_correlation_realized=realized_corr,
            true_effect_realized=realized_effect,
            segment_effects_realized=segment_effects_realized,
        )

        return users_df, assignments_df, events_df, ground_truth

    # ------------------------------------------------------------------ #
    # Generation steps
    # ------------------------------------------------------------------ #

    def _generate_users(self, rng: np.random.Generator) -> pd.DataFrame:
        user_ids = [f"u_{i:07d}" for i in range(self.n_users)]
        device_type = rng.choice(DEVICE_TYPES, size=self.n_users, p=DEVICE_PROBS)
        region = rng.choice(REGIONS, size=self.n_users, p=REGION_PROBS)
        existing_customer = rng.random(self.n_users) < EXISTING_CUSTOMER_PROB
        pre_period_covariate = rng.lognormal(mean=0.0, sigma=_COVARIATE_LOGNORMAL_SIGMA, size=self.n_users)
        signup_date = (datetime(2025, 1, 1) - timedelta(days=1)).date()

        return pd.DataFrame({
            "user_id": user_ids,
            "signup_date": [signup_date] * self.n_users,
            "device_type": device_type,
            "region": region,
            "existing_customer": existing_customer,
            "pre_period_covariate": pre_period_covariate,
        })

    @staticmethod
    def _standardize(x: np.ndarray) -> np.ndarray:
        return (x - x.mean()) / x.std()

    @staticmethod
    def _expand_bracket(
        f: Callable[[float], float],
        seed: float,
        initial_halfwidth: float = _BRACKET_INITIAL_HALFWIDTH,
        max_expansions: int = _BRACKET_MAX_EXPANSIONS,
    ) -> tuple[float, float]:
        """
        Finds (lo, hi) around `seed` such that f(lo) and f(hi) have opposite
        signs, by doubling the half-width until a sign change is found.
        A FIXED bracket width was found insufficient — the required intercept
        shift to hit a target baseline_rate scales with |slope| x (typical
        covariate magnitude), unbounded by a small constant across configs.
        """
        halfwidth = initial_halfwidth
        f_seed = f(seed)
        for _ in range(max_expansions):
            lo, hi = seed - halfwidth, seed + halfwidth
            f_lo, f_hi = f(lo), f(hi)
            if f_lo * f_seed < 0:
                return lo, seed
            if f_hi * f_seed < 0:
                return seed, hi
            if f_lo * f_hi < 0:
                return lo, hi
            halfwidth *= 2
        raise ValueError(
            f"Could not bracket a sign change around seed={seed} after "
            f"{max_expansions} expansions (final halfwidth={halfwidth}). "
            f"Target is likely infeasible given current bounds/parameters."
        )

    def _calibrate(self) -> tuple[float, float, float, float]:
        """
        Jointly solves (intercept, slope) via DAMPED alternating 1D root-finds.

        Includes a degenerate-variance guard in corr_gap: at extreme (a, b)
        combinations, sigmoid(a + b*x) saturates near 0 or 1 for the ENTIRE
        calibration sample, producing a constant y with zero variance —
        corrcoef would divide 0/0 -> NaN and crash brentq's solver. The guard
        returns a correctly-signed large gap instead, letting brentq's
        bisection treat it as "far from root" rather than crashing.

        Raises RuntimeError if the damped iteration does not converge within
        tolerance — this is NOT always a bug. For target_corr=-0.4 at
        baseline_rate=0.10, this genuinely appears to be unreachable given
        this covariate's skew (see module docstring) — the intercept was
        observed walking to -14 without settling. Treat this error as "check
        feasibility of this specific config," not "the solver is broken."
        """
        base_intercept = logit(self.baseline_rate)

        if abs(self.covariate_correlation_target) < 1e-9:
            return base_intercept, 0.0, 0.0, self.baseline_rate

        calib_rng = np.random.default_rng(self.seed + 1)
        raw = calib_rng.lognormal(mean=0.0, sigma=_COVARIATE_LOGNORMAL_SIGMA, size=_CALIBRATION_N)
        x = self._standardize(raw)
        u = calib_rng.random(_CALIBRATION_N)

        def corr_gap(a: float, b: float) -> float:
            p = expit(a + b * x)
            y = (u < p).astype(float)
            if y.std() < 1e-12:
                realized = 0.0
            else:
                realized = float(np.corrcoef(x, y)[0, 1])
            return realized - self.covariate_correlation_target

        def mean_gap(a: float, b: float) -> float:
            p = expit(a + b * x)
            return float(p.mean() - self.baseline_rate)

        slope_lo, slope_hi = _CALIBRATION_SLOPE_BOUNDS
        if self.covariate_correlation_target < 0:
            slope_lo, slope_hi = -slope_hi, -slope_lo

        intercept = base_intercept
        slope = 0.0

        for _ in range(_JOINT_CALIBRATION_ITERS):
            try:
                slope_solve = brentq(lambda b: corr_gap(intercept, b), slope_lo, slope_hi, xtol=1e-4)
            except ValueError as exc:
                raise ValueError(
                    f"Could not calibrate slope for target correlation="
                    f"{self.covariate_correlation_target} at intercept={intercept:.4f}. "
                    f"baseline_rate={self.baseline_rate} may make this unreachable within "
                    f"slope bounds {_CALIBRATION_SLOPE_BOUNDS}. This target may be outside "
                    f"the feasible region for this covariate's skew — see module docstring."
                ) from exc
            slope = slope + _CALIBRATION_DAMPING * (slope_solve - slope)

            lo, hi = self._expand_bracket(lambda a: mean_gap(a, slope), seed=intercept)
            intercept_solve = brentq(lambda a: mean_gap(a, slope), lo, hi, xtol=1e-4)
            intercept = intercept + _CALIBRATION_DAMPING * (intercept_solve - intercept)

        final_corr_gap = corr_gap(intercept, slope)
        final_mean_gap = mean_gap(intercept, slope)

        if abs(final_corr_gap) > _CALIBRATION_CONVERGENCE_TOL or abs(final_mean_gap) > _CALIBRATION_CONVERGENCE_TOL:
            raise RuntimeError(
                f"Calibration did NOT converge after {_JOINT_CALIBRATION_ITERS} damped iterations: "
                f"corr_gap={final_corr_gap:+.4f}, mean_gap={final_mean_gap:+.4f} "
                f"(tol={_CALIBRATION_CONVERGENCE_TOL}). target_corr="
                f"{self.covariate_correlation_target}, baseline_rate={self.baseline_rate}. "
                f"This combination may be outside the feasible region for this covariate's "
                f"skew — try a smaller-magnitude target before assuming a code bug."
            )

        realized_corr = final_corr_gap + self.covariate_correlation_target
        realized_mean = final_mean_gap + self.baseline_rate
        return intercept, slope, realized_corr, realized_mean

    def _assign_variant(self, rng: np.random.Generator) -> np.ndarray:
        treatment_prob = self.corrupted_split if self.corrupted_split is not None else 0.5
        return rng.choice(
            ["control", "treatment"], size=self.n_users, p=[1 - treatment_prob, treatment_prob]
        )

    def _compute_effect_per_user(
        self, users_df: pd.DataFrame
    ) -> tuple[np.ndarray, dict[str, float] | None]:
        effect = np.full(self.n_users, self.true_effect, dtype=float)
        segment_effects_realized = None
        if self.segment_column:
            seg_map = self.segment_heterogeneity[self.segment_column]
            segment_values = users_df[self.segment_column].to_numpy()
            segment_delta = np.array([seg_map.get(v, 0.0) for v in segment_values])
            effect = effect + segment_delta
            segment_effects_realized = dict(seg_map)
        return effect, segment_effects_realized

    def _build_assignments(self, user_ids: pd.Series, variant: np.ndarray) -> pd.DataFrame:
        variant_id_map = {
            "control": f"{self.experiment_id}_control",
            "treatment": f"{self.experiment_id}_treatment",
        }
        return pd.DataFrame({
            "user_id": user_ids,
            "experiment_id": self.experiment_id,
            "variant_id": [variant_id_map[v] for v in variant],
            "assigned_at": datetime(2025, 1, 1),
        })

    def _build_events(self, user_ids: pd.Series, converted: np.ndarray) -> pd.DataFrame:
        converted_mask = converted.astype(bool)
        converted_ids = user_ids[converted_mask].to_numpy()
        return pd.DataFrame({
            "event_id": [f"ev_{self.experiment_id}_{i:07d}" for i in range(len(converted_ids))],
            "user_id": converted_ids,
            "experiment_id": self.experiment_id,
            "event_type": "conversion",
            "event_timestamp": datetime(2025, 1, 2),
            "value": 1.0,
        })