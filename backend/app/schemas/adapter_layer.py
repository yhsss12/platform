from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class ObservationSpaceResponse(BaseModel):
    type: str
    keys: list[str] = Field(default_factory=list)
    dims: dict[str, int] = Field(default_factory=dict)


class ActionSpaceResponse(BaseModel):
    type: str
    dim: Optional[int] = None
    supportsSequence: bool = False
    horizon: Optional[int] = None


class DatasetManifestResponse(BaseModel):
    datasetId: str
    datasetName: str = ""
    taskName: str = ""
    simulator: str = ""
    robotType: str = ""
    dataFormat: str = "HDF5"
    observationSpace: ObservationSpaceResponse
    actionSpace: ActionSpaceResponse
    episodeCount: int = 0
    successCount: int = 0
    horizon: Optional[int] = None
    storageUri: str = ""
    manifestVersion: str = "1.0"
    taskType: Optional[str] = None
    sourceJobId: Optional[str] = None


class ModelCompatibilityResultResponse(BaseModel):
    modelType: str
    displayName: str
    compatible: bool
    score: float = 0.0
    reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    status: str = "available"


class CompatibilityAnalysisResponse(BaseModel):
    datasetId: str
    compatible: bool
    manifestVersion: str
    recommendedModels: list[str] = Field(default_factory=list)
    blockingReasons: list[str] = Field(default_factory=list)
    results: list[ModelCompatibilityResultResponse] = Field(default_factory=list)


class AnalyzeCompatibilityRequest(BaseModel):
    datasetManifest: dict[str, Any]


class TrainingPlanRequest(BaseModel):
    datasetManifest: dict[str, Any]
    modelType: str


class TrainingPlanResponse(BaseModel):
    datasetId: str
    datasetName: str = ""
    modelType: str
    downstreamModelType: str = ""
    trainingBackend: str = ""
    dataFormat: str = "HDF5"
    epochs: int
    batchSize: int
    learningRate: float
    device: str = "cuda"
    seed: int = 1
    advancedEnabled: bool = False
    advancedConfig: dict[str, Any] = Field(default_factory=dict)
    savePolicy: dict[str, Any] = Field(default_factory=dict)
    storageUri: str = ""
    taskName: str = ""
    simulator: str = ""
    robotType: str = ""
    manifestVersion: str = "1.0"
    adapterLayerVersion: str = "1.0"
    notes: str = ""


class EvaluationPlanRequest(BaseModel):
    modelAssetOrTrainingPlan: dict[str, Any]


class EvaluationPlanResponse(BaseModel):
    evaluationMode: str
    taskTemplateId: str
    taskName: str = ""
    simulator: str = ""
    robotType: str = ""
    policyType: str = ""
    modelType: str = ""
    numEpisodes: int
    seed: int = 0
    record: bool = True
    headless: bool = True
    metrics: list[str] = Field(default_factory=list)
    datasetId: Optional[str] = None
    modelAssetId: Optional[str] = None
    checkpointPath: Optional[str] = None
    downstreamModelType: Optional[str] = None
    trainingBackend: Optional[str] = None
    adapterLayerVersion: str = "1.0"


class DatasetProfileResponse(BaseModel):
    datasetId: str = ""
    datasetName: str = ""
    taskName: str = ""
    simulator: str = "unknown"
    robotType: str = "unknown"
    episodeCount: int = 0
    successCount: int = 0
    observationType: str = "unknown"
    observationKeys: list[str] = Field(default_factory=list)
    cameraKeys: list[str] = Field(default_factory=list)
    stateDim: int = 0
    actionDim: int = 0
    actionSpace: str = "unknown"
    horizon: int = 0
    format: str = "HDF5"
    storageUri: str = ""
    hasValidationSplit: bool = False
    hasReward: bool = False
    hasSuccess: bool = False
    hasDone: bool = False
    taskType: Optional[str] = None
    warnings: list[str] = Field(default_factory=list)
    inferenceSources: list[str] = Field(default_factory=list)


class ModelAdaptationResponse(BaseModel):
    modelType: str
    displayName: str = ""
    inputConfig: dict[str, Any] = Field(default_factory=dict)
    outputConfig: dict[str, Any] = Field(default_factory=dict)
    architectureConfig: dict[str, Any] = Field(default_factory=dict)
    normalizationConfig: dict[str, Any] = Field(default_factory=dict)
    dataLoaderConfig: dict[str, Any] = Field(default_factory=dict)
    trainingConfig: dict[str, Any] = Field(default_factory=dict)
    advancedConfig: dict[str, Any] = Field(default_factory=dict)
    downstreamModelType: str = ""
    trainingBackend: str = ""


class AdaptationValidationResponse(BaseModel):
    adaptable: bool = True
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class TrainingAdaptationPlanRequest(BaseModel):
    datasetId: str
    modelType: str
    datasetManifest: Optional[dict[str, Any]] = None
    overrides: dict[str, Any] = Field(default_factory=dict)


class TrainingAdaptationPlanResponse(BaseModel):
    datasetProfile: DatasetProfileResponse
    modelAdaptation: ModelAdaptationResponse
    validation: AdaptationValidationResponse
    explanation: list[str] = Field(default_factory=list)
    configPatch: dict[str, Any] = Field(default_factory=dict)
    adapterLayerVersion: str = "2.0"
