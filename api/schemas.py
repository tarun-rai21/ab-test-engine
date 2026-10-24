"""
api/schemas.py — Pydantic request/response models for the FastAPI layer.

Response schemas are built with model_config = ConfigDict(from_attributes=True)
specifically so they can be constructed directly from this project's existing
dataclasses (core.validity.SRMResult, core.inference.InferenceResult,
core.sequential.SequentialCheckResult, core.segments.SegmentAnalysisResult,
core.pipeline.AnalysisReport) via e.g. AnalysisReportSchema.model_validate(report)
— no manual field-by-field copying, and no risk of the API silently drifting
out of sync with the underlying dataclasses as they evolve.

POST /experiments/{id}/assign generates SIMULATED data; real data goes
through POST /experiments/{id}/ingest (IngestRequest), which shares
core.ingestion's column mapping and validation with the Streamlit upload.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

# ===================================================================== #
# POST /experiments
# ===================================================================== #

class VariantCreateRequest(BaseModel):
    name: str = Field(min_length=1)
    # A 0 share makes the SRM check's expected count 0 (chi-square undefined).
    split_pct: float = Field(gt=0.0, le=1.0)


class PlannedScheduleRequest(BaseModel):
    """Planned sequential-monitoring schedule; checkpoint_n is TOTAL users (both arms)."""
    n_checkpoints_planned: int = Field(ge=1)
    checkpoint_n: int = Field(ge=1)


class ExperimentCreateRequest(BaseModel):
    name: str
    hypothesis: str = ""
    variants: list[VariantCreateRequest] = Field(
        default_factory=lambda: [
            VariantCreateRequest(name="control", split_pct=0.5),
            VariantCreateRequest(name="treatment", split_pct=0.5),
        ]
    )
    schedule: PlannedScheduleRequest | None = None


class ExperimentCreateResponse(BaseModel):
    experiment_id: str
    status: str


# ===================================================================== #
# POST /experiments/{id}/assign
# ===================================================================== #

class AssignRequest(BaseModel):
    """
    mode is currently always "simulate" — see module docstring's scope note
    on why "ingest" is not implemented here.
    """
    mode: str = "simulate"
    n_users: int = Field(gt=0)
    baseline_rate: float = Field(gt=0.0, lt=1.0)
    true_effect: float = 0.0
    # Bounds mirror ExperimentSimulator's own validation, so bad input is a
    # 422 at the edge rather than a 500 from deep inside the simulator.
    covariate_correlation: float = Field(default=0.5, gt=-0.99, lt=0.99)
    seed: int = Field(default=42, ge=0)
    corrupted_split: float | None = Field(default=None, gt=0.0, lt=1.0)
    segment_heterogeneity: dict[str, dict[str, float]] | None = None
    # Order values (None = every conversion worth 1.0): log-normal with this
    # mean in control and mean * (1 + revenue_lift) in treatment.
    revenue_mean: float | None = Field(default=None, gt=0.0)
    revenue_lift: float = Field(default=0.0, gt=-1.0)
    revenue_sigma: float = Field(default=1.0, gt=0.0)
    # Per-arm true effects for A/B/n experiments, keyed by treatment variant
    # name; arms not listed get true_effect.
    arm_effects: dict[str, float] | None = None


class SRMCheckSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    flagged: bool
    p_value: float
    chi_sq_stat: float


class AssignResponse(BaseModel):
    n_assigned: int
    srm_check: SRMCheckSummary


# ===================================================================== #
# POST /experiments/{id}/analyze  and  GET /experiments/{id}/report
# ===================================================================== #

class AnalyzeRequest(BaseModel):
    metric: Literal["conversion", "revenue_per_user", "revenue_per_conversion"] = "conversion"
    # revenue_per_user only: cap per-user revenue at this pooled quantile.
    winsorize_quantile: float | None = Field(default=None, gt=0.5, lt=1.0)
    use_cuped: bool = True
    segment_columns: list[str] | None = None
    n_checkpoints_planned: int | None = Field(default=None, ge=1)
    # TOTAL users across both arms per planned look (same unit as the
    # sequential_checkpoints table's cumulative_n).
    checkpoint_n: int | None = Field(default=None, ge=1)
    # Peeking correction: "lan_demets" (information-based, any look timing)
    # or "obrien_fleming" (index-based, planned equal spacing only).
    spending: Literal["lan_demets", "obrien_fleming"] = "lan_demets"
    # A/B/n p-value adjustment across arms.
    multiple_comparison: Literal["dunnett", "holm"] = "dunnett"


class SRMResultSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    observed_counts: tuple[int, ...]
    expected_counts: tuple[float, ...]
    chi_sq_stat: float
    p_value: float
    flagged: bool


class InferenceResultSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    method: str
    point_estimate: float
    ci_lower: float
    ci_upper: float
    alpha: float
    p_value: float
    standard_error: float
    degrees_freedom: float | None
    n_control: int
    n_treatment: int


class SequentialCheckResultSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    checkpoint_position: int
    naive_p_value: float
    naive_significant: bool
    corrected_threshold: float
    corrected_significant: bool
    disagreement: bool
    beyond_plan: bool
    spending: str
    information_fraction: float | None
    treatment_arm: str | None = None
    metric_name: str | None = None
    replanned: bool = False
    latest_checkpoint_n: int


class SegmentResultSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    segment_value: str
    inference: InferenceResultSchema
    bh_significant: bool
    reversal_flag: bool


class SegmentAnalysisResultSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    segment_column: str
    pooled_point_estimate: float
    segments: tuple[SegmentResultSchema, ...]
    excluded_segments: tuple[str, ...]
    treatment_arm: str | None = None
    heterogeneity_q: float | None = None
    heterogeneity_p_value: float | None = None


class ArmComparisonSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    treatment_arm: str
    raw_effect: InferenceResultSchema
    # Reports stored before the Dunnett option used *_holm field names;
    # accept them so GET /report still serves those stored reports.
    raw_p_value_adjusted: float = Field(
        validation_alias=AliasChoices("raw_p_value_adjusted", "raw_p_value_holm")
    )
    cuped_effect: InferenceResultSchema | None
    cuped_p_value_adjusted: float | None = Field(
        default=None,
        validation_alias=AliasChoices("cuped_p_value_adjusted", "cuped_p_value_holm"),
    )


class AnalysisReportSchema(BaseModel):
    """
    Mirrors core.pipeline.AnalysisReport exactly. Constructed via
    AnalysisReportSchema.model_validate(report) directly from the dataclass
    instance analyze_experiment() returns — see module docstring.
    """
    model_config = ConfigDict(from_attributes=True)
    experiment_id: str
    metric_name: str
    srm: SRMResultSchema
    achievable_mde: float | None
    raw_effect: InferenceResultSchema
    cuped_effect: InferenceResultSchema | None
    cuped_variance_reduction_pct: float | None
    sequential_risk: SequentialCheckResultSchema | None
    segments: tuple[SegmentAnalysisResultSchema, ...] | None
    winsorization_cap: float | None = None
    comparisons: tuple[ArmComparisonSchema, ...] = ()
    # Older stored reports predate the option and were always Holm-adjusted.
    multiple_comparison: str = "holm"
    sequential_risks: tuple[SequentialCheckResultSchema, ...] = ()


# ===================================================================== #
# POST /experiments/{id}/checkpoints
# ===================================================================== #

class CheckpointRequest(BaseModel):
    metric: Literal["conversion", "revenue_per_user", "revenue_per_conversion"] = "conversion"


class RecordedLookSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    treatment_arm: str
    metric_name: str
    cumulative_n: int
    p_value: float
    alpha_threshold: float
    planned_total_n: int | None
    corrected_significant: bool


# ===================================================================== #
# POST /experiments/{id}/ingest
# ===================================================================== #

class IngestRequest(BaseModel):
    """
    Real data, one object per user. Without `columns`, rows must use the
    canonical names (variant_id ending in '_control' / containing
    '_treatment', converted, optional user_id, revenue,
    pre_period_covariate, device_type, region, existing_customer). With
    `columns` ({role: source field}) and optionally `variant_values`
    ({raw value: variant name or null to drop}), any shape works — see
    core.ingestion. Variant names must match the experiment's variants.
    """
    rows: list[dict[str, Any]] = Field(min_length=1)
    columns: dict[str, str] | None = None
    variant_values: dict[str, str | None] | None = None


class IngestResponse(BaseModel):
    n_ingested: int
    srm_check: SRMCheckSummary
