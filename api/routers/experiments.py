"""
api/routers/experiments.py — POST /experiments and POST /experiments/{id}/assign.

Thin HTTP wrapper: all it does is call get_engine(), run existing SQL/
functions, and shape results into the Pydantic schemas from api/schemas.py.
No new business logic lives here — matches the project's established
"routers are thin, core/ holds all real logic" discipline.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException
from sqlalchemy import text

from api.schemas import (
    AssignRequest,
    AssignResponse,
    ExperimentCreateRequest,
    ExperimentCreateResponse,
    SRMCheckSummary,
)
from core.data_access import get_variant_counts
from core.validity import srm_check
from db.connection import get_engine
from db.seed import seed_assignments_for_existing_experiment

router = APIRouter(prefix="/experiments", tags=["experiments"])


def _generate_experiment_id() -> str:
    """
    Random, NOT derived from a simulator seed — deliberately different from
    ExperimentSimulator's deterministic f"exp_seed{seed}" convention. Phase
    1's "no uuid.uuid4() anywhere" rule was specifically about reproducing
    the SAME data GIVEN a seed; a live, API-created experiment has no seed
    to derive an id from in the first place, so a random id here is the
    correct, standard REST choice, not a violation of that rule's intent.
    """
    return f"exp_{uuid.uuid4().hex[:12]}"


@router.post("", response_model=ExperimentCreateResponse)
def create_experiment(request: ExperimentCreateRequest) -> ExperimentCreateResponse:
    """
    Creates an experiment row plus its variant rows. Does NOT generate any
    data yet — that's POST /experiments/{id}/assign's job, matching the
    spec's two-step create-then-assign flow.

    Validates split_pct values sum to ~1.0 BEFORE writing anything to the
    database — same "validate untrusted input before use" discipline
    already applied throughout this project (core.validity.srm_check's own
    input checks, core.data_access's segment-column allow-list).
    """
    total_split = sum(v.split_pct for v in request.variants)
    if abs(total_split - 1.0) > 1e-6:
        raise HTTPException(
            status_code=422,
            detail=f"variant split_pct values must sum to 1.0, got {total_split}",
        )

    experiment_id = _generate_experiment_id()
    engine = get_engine()

    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO experiments (experiment_id, name, hypothesis, "
                "start_date, status) VALUES (:eid, :name, :hyp, :start_date, 'running')"
            ),
            {
                "eid": experiment_id,
                "name": request.name,
                "hyp": request.hypothesis,
                "start_date": "2025-01-01",
            },
        )
        for variant in request.variants:
            conn.execute(
                text(
                    "INSERT INTO variants (variant_id, experiment_id, name, split_pct) "
                    "VALUES (:vid, :eid, :name, :split_pct)"
                ),
                {
                    "vid": f"{experiment_id}_{variant.name}",
                    "eid": experiment_id,
                    "name": variant.name,
                    "split_pct": variant.split_pct,
                },
            )

    return ExperimentCreateResponse(experiment_id=experiment_id, status="created")


@router.post("/{experiment_id}/assign", response_model=AssignResponse)
def assign(experiment_id: str, request: AssignRequest) -> AssignResponse:
    """
    Generates synthetic data for an ALREADY-CREATED experiment (see
    create_experiment above) and runs an immediate SRM check on the
    resulting assignment counts — giving the caller instant feedback on
    whether the randomization looks healthy, before any effect estimate is
    computed (matches spec 4.4's ordering: validity before inference).

    Raises 404 if experiment_id doesn't exist — checked explicitly rather
    than letting seed_assignments_for_existing_experiment's variant-naming
    convention fail confusingly deep inside the simulator.

    REAL GAP FOUND BY TESTING, fixed here: ExperimentSimulator always
    generates variant_ids using the hardcoded f"{experiment_id}_control" /
    f"{experiment_id}_treatment" suffixes (Phase 1's convention) —
    completely independent of whatever variant names create_experiment
    was actually called with. Without this check, an experiment created
    with different variant names (or only one variant) would silently
    receive assignment rows referencing variant_ids that don't exist
    anywhere in its variants table, discovered only much later as
    confusing zero-counts in get_variant_counts(). Checked explicitly and
    loudly here instead.
    """
    engine = get_engine()
    with engine.connect() as conn:
        exists = conn.execute(
            text("SELECT 1 FROM experiments WHERE experiment_id = :eid"),
            {"eid": experiment_id},
        ).fetchone()
    if not exists:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id!r} not found")

    if request.mode != "simulate":
        raise HTTPException(
            status_code=400,
            detail=(
                f"mode={request.mode!r} not supported — only 'simulate' is "
                f"implemented (see api/schemas.py's scope note)."
            ),
        )

    with engine.connect() as conn:
        variant_names = conn.execute(
            text("SELECT name FROM variants WHERE experiment_id = :eid"),
            {"eid": experiment_id},
        ).scalars().all()
    if "control" not in variant_names or "treatment" not in variant_names:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Experiment {experiment_id!r} does not have both a 'control' and "
                f"'treatment' variant (found: {variant_names}). The simulator "
                f"requires exactly these two variant names to generate data — "
                f"create the experiment with variants named 'control' and "
                f"'treatment' to use mode='simulate'."
            ),
        )

    sim_cfg = {
        "n_users": request.n_users,
        "baseline_rate": request.baseline_rate,
        "true_effect": request.true_effect,
        "covariate_correlation": request.covariate_correlation,
        "seed": request.seed,
        "corrupted_split": request.corrupted_split,
        "segment_heterogeneity": request.segment_heterogeneity,
    }
    seed_assignments_for_existing_experiment(experiment_id, sim_cfg)

    observed_counts, expected_ratios = get_variant_counts(engine, experiment_id)
    srm_result = srm_check(observed_counts, expected_ratios)

    return AssignResponse(
        n_assigned=sum(observed_counts),
        srm_check=SRMCheckSummary.model_validate(srm_result),
    )