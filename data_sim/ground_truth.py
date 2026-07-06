"""
GroundTruth: the single object that makes this project's correctness claims falsifiable
rather than asserted.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GroundTruth:
    n_users: int
    baseline_rate: float
    true_effect_configured: float
    covariate_correlation_target: float
    seed: int
    corrupted_split: float | None
    segment_column: str | None
    segment_effects_configured: dict[str, float] | None

    calibrated_intercept: float
    calibrated_slope: float
    baseline_rate_realized: float          # NEW — mean-matching check, catches Jensen's-gap bugs
    covariate_correlation_realized: float
    true_effect_realized: float
    segment_effects_realized: dict[str, float] | None = None

    composition: str = "additive"

    def summary(self) -> str:
        lines = [
            f"n_users={self.n_users}, baseline_rate: configured={self.baseline_rate:.4f} "
            f"realized={self.baseline_rate_realized:.4f} "
            f"(delta={self.baseline_rate_realized - self.baseline_rate:+.4f}), seed={self.seed}",
            f"true_effect: configured={self.true_effect_configured:.4f} "
            f"realized={self.true_effect_realized:.4f} "
            f"(delta={self.true_effect_realized - self.true_effect_configured:+.4f})",
            f"covariate_correlation: target={self.covariate_correlation_target:.4f} "
            f"realized={self.covariate_correlation_realized:.4f}",
        ]
        if self.corrupted_split is not None:
            lines.append(f"corrupted_split (deliberate SRM break): {self.corrupted_split}")
        if self.segment_effects_configured:
            lines.append(
                f"segment_column={self.segment_column}, "
                f"segment_effects_configured={self.segment_effects_configured}, "
                f"segment_effects_realized={self.segment_effects_realized}"
            )
        return "\n".join(lines)
