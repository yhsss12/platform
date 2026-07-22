from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

TrainingJobStatus = Literal[
    "queued",
    "starting",
    "running",
    "completed",
    "failed",
    "backend_unavailable",
]

TrainingBackendRequest = Literal[
    "auto",
    "robomimic",
    "robomimic_bc",
    "isaac_robomimic_bc",
    "torch_bc",
    "act",
    "dt",
    "diffusion_policy",
]


class TrainingCapabilitiesResponse(BaseModel):
    foundTrainingScripts: bool
    supportedTrainingBackends: list[str]
    recommendedBackend: str
    evidence: list[str]


class RobomimicAdvancedParams(BaseModel):
    actor_hidden_dims: str = Field(default="512,512", min_length=1)
    l2_regularization: float = Field(default=0.0, ge=0)


class CreateTrainingJobRequest(BaseModel):
    datasetId: str
    datasetIds: Optional[list[str]] = None
    datasetManifestPath: Optional[str] = None
    datasetManifest: Optional[dict[str, Any]] = None
    datasetManifests: Optional[list[dict[str, Any]]] = None
    modelTypeId: Optional[str] = None
    downstreamModelType: str = "ACT"
    trainingBackend: TrainingBackendRequest = "auto"
    dataFormat: str = "HDF5"
    epochs: int = Field(default=5, ge=1)
    batchSize: int = Field(default=16, ge=1, le=512)
    learningRate: float = Field(default=0.0001, gt=0)
    device: str = "cuda"
    deviceLabel: Optional[str] = None
    trainingNodeId: Optional[str] = None
    seed: int = 1
    seedMode: Optional[Literal["random", "manual"]] = None
    advancedEnabled: bool = False
    modelParams: Optional[dict[str, Any]] = None
    pretrained: Optional[dict[str, Any]] = None
    saveFinal: bool = True
    saveBest: bool = False
    checkpointIntervalEpochs: Optional[int] = Field(default=None, ge=1)
    adaptationSnapshot: Optional[dict[str, Any]] = None
    architectureConfig: Optional[dict[str, Any]] = None
    dataLoaderConfig: Optional[dict[str, Any]] = None
    normalizationConfig: Optional[dict[str, Any]] = None
    inputConfig: Optional[dict[str, Any]] = None
    outputConfig: Optional[dict[str, Any]] = None


class CreateTrainingJobResponse(BaseModel):
    trainJobId: str
    status: TrainingJobStatus
    message: str


class TrainingJobStatusResponse(BaseModel):
    trainJobId: str
    status: TrainingJobStatus
    progress: float = 0.0
    epoch: int = 0
    totalEpochs: int = 0
    loss: Optional[float] = None
    checkpointExists: bool = False
    checkpointPath: Optional[str] = None
    modelAssetId: Optional[str] = None
    message: str = ""
    datasetId: Optional[str] = None
    datasetName: Optional[str] = None
    downstreamModelType: Optional[str] = None
    trainingBackend: Optional[str] = None
    dataFormat: Optional[str] = None
    device: Optional[str] = None
    deviceLabel: Optional[str] = None
    trainingNodeId: Optional[str] = None
    trainingNodeDisplayName: Optional[str] = None
    createdAt: Optional[str] = None


class TrainingJobLogResponse(BaseModel):
    trainJobId: str
    log: str


class TrainingJobModelResponse(BaseModel):
    trainJobId: str
    ready: bool
    modelManifest: Optional[dict[str, Any]] = None
    checkpointPath: Optional[str] = None


class TrainingJobListItem(BaseModel):
    trainJobId: str
    status: TrainingJobStatus
    datasetId: Optional[str] = None
    datasetName: Optional[str] = None
    downstreamModelType: Optional[str] = None
    trainingBackend: Optional[str] = None
    createdAt: Optional[str] = None
    updatedAt: Optional[str] = None
    checkpointExists: bool = False
    modelAssetId: Optional[str] = None
    epoch: Optional[int] = None
    totalEpochs: Optional[int] = None
    loss: Optional[float] = None
    message: Optional[str] = None
    dataFormat: Optional[str] = None
    deviceLabel: Optional[str] = None
    trainingNodeId: Optional[str] = None
    trainingNodeDisplayName: Optional[str] = None
    taskName: Optional[str] = None


class TrainingJobListResponse(BaseModel):
    jobs: list[TrainingJobListItem]
    total: int


class TrainingJobDeleteResponse(BaseModel):
    trainJobId: str
    deleted: bool = True
    deletedAt: Optional[str] = None
