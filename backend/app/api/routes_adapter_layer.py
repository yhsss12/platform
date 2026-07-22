from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.deps import get_current_user
from app.models.user import User
from app.schemas.adapter_layer import (
    AnalyzeCompatibilityRequest,
    CompatibilityAnalysisResponse,
    EvaluationPlanRequest,
    EvaluationPlanResponse,
    TrainingAdaptationPlanRequest,
    TrainingAdaptationPlanResponse,
    TrainingPlanRequest,
    TrainingPlanResponse,
)
from app.services.adapter_layer import adapter_service as svc

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get(
    "/datasets/{dataset_id}/compatibility",
    response_model=CompatibilityAnalysisResponse,
    tags=["adapter-layer"],
)
async def get_dataset_compatibility(
    dataset_id: str,
    _: User = Depends(get_current_user),
) -> CompatibilityAnalysisResponse:
    try:
        result = await asyncio.to_thread(svc.get_dataset_compatibility, dataset_id)
    except LookupError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    logger.info("adapter compatibility queried dataset_id=%s compatible=%s", dataset_id, result.get("compatible"))
    return CompatibilityAnalysisResponse(**result)


@router.post(
    "/compatibility/analyze",
    response_model=CompatibilityAnalysisResponse,
    tags=["adapter-layer"],
)
async def analyze_compatibility(
    payload: AnalyzeCompatibilityRequest,
    _: User = Depends(get_current_user),
) -> CompatibilityAnalysisResponse:
    analysis = await asyncio.to_thread(svc.analyze_dataset_compatibility, payload.datasetManifest)
    result = svc.compatibility_analysis_to_dict(analysis)
    return CompatibilityAnalysisResponse(**result)


@router.post(
    "/training-adaptation-plan",
    response_model=TrainingAdaptationPlanResponse,
    tags=["adapter-layer"],
)
async def create_training_adaptation_plan(
    payload: TrainingAdaptationPlanRequest,
    _: User = Depends(get_current_user),
) -> TrainingAdaptationPlanResponse:
    try:
        result = await asyncio.to_thread(
            svc.build_training_adaptation_plan,
            dataset_id=payload.datasetId,
            raw_manifest=payload.datasetManifest,
            model_type=payload.modelType,
            overrides=payload.overrides,
        )
    except LookupError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    logger.info(
        "training adaptation plan dataset_id=%s model_type=%s adaptable=%s",
        payload.datasetId,
        payload.modelType,
        result.get("validation", {}).get("adaptable"),
    )
    return TrainingAdaptationPlanResponse(**result)


@router.post(
    "/training-plan",
    response_model=TrainingPlanResponse,
    tags=["adapter-layer"],
)
async def create_training_plan(
    payload: TrainingPlanRequest,
    _: User = Depends(get_current_user),
) -> TrainingPlanResponse:
    try:
        plan = await asyncio.to_thread(
            svc.build_training_plan,
            payload.datasetManifest,
            payload.modelType,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    return TrainingPlanResponse(**plan)


@router.post(
    "/evaluation-plan",
    response_model=EvaluationPlanResponse,
    tags=["adapter-layer"],
)
async def create_evaluation_plan(
    payload: EvaluationPlanRequest,
    _: User = Depends(get_current_user),
) -> EvaluationPlanResponse:
    plan = await asyncio.to_thread(svc.build_evaluation_plan, payload.modelAssetOrTrainingPlan)
    return EvaluationPlanResponse(**plan)
