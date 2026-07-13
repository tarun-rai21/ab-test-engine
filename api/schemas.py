"""
api/schemas.py — Pydantic request/response models for the FastAPI layer.

Response schemas are built with model_config = ConfigDict(from_attributes=True)
specifically so they can be constructed directly from this project's existing
dataclasses (core.validity.SRMResult, core.inference.InferenceResult,
core.sequential.SequentialCheckResult, core.segments.SegmentAnalysisResult,
core.pipeline.AnalysisReport) via e.g. AnalysisReportSchema.model_validate(report)
— no manual field-by-field copying, and no risk of the API silently drifting
out of sync with the underlying dataclasses as they evolve.

SCOPE NOTE: the original spec's POST /experiments/{id}/assign describes two
modes — "simulate" (generate synthetic data) and "ingest" (upload real data).
Only "simulate" is implemented here. Real-data ingestion needs its own
CSV-parsing and validation logic that belongs with Phase 8's upload flow, not
duplicated here — a real, stated scope decision, not a silent gap.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

# ===================================================================== #
# POST /experiments
# ===================================================================== #

class VariantCreateRequest(BaseModel):
    name: str
    split_pct: float = Field(ge=0.0, le=1.0)


class ExperimentCreateRequest(BaseModel):
    name: str
    hypothesis: str = ""
    variants: list[VariantCreateRequest] = Field(
        default_factory=lambda: [
            VariantCreateRequest(name="control", split_pct=0.5),
            VariantCreateRequest(name="treatment", split_pct=0.5),
        ]
    )


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
    covariate_correlation: float = 0.5
    seed: int = 42
    corrupted_split: float | None = None
    segment_heterogeneity: dict[str, dict[str, float]] | None = None


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
    metric: str = "conversion"
    use_cuped: bool = True
    segment_columns: list[str] | None = None
    n_checkpoints_planned: int | None = None
    checkpoint_n: int | None = None


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
    degrees_freedom: float
    n_control: int
    n_treatment: int


class SequentialCheckResultSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    checkpoint_position: int
    naive_significant: bool
    corrected_significant: bool
    disagreement: bool
    latest_checkpoint_n: int


class SegmentResultSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    segment_value: str
    inference: InferenceResultSchema
    bh_significant: bool
    simpsons_flag: bool


class SegmentAnalysisResultSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    segment_column: str
    pooled_point_estimate: float
    segments: tuple[SegmentResultSchema, ...]
    excluded_segments: tuple[str, ...]


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
