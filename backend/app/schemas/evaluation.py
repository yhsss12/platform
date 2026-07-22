from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class EvaluateAsyncRequest(BaseModel):
    """统一评测启动请求。"""

    taskTemplateId: Optional[str] = None
    taskType: Optional[str] = None
    evaluationMode: str
    evaluationObject: Optional[str] = None
    numEpisodes: int = Field(default=10, ge=1, le=100)
    seed: Optional[int] = 0
    seeds: Optional[list[int]] = None
    policyType: Optional[str] = None
    checkpointId: Optional[str] = None
    checkpointPath: Optional[str] = None
    datasetId: Optional[str] = None
    modelAssetId: Optional[str] = None
    record: bool = True
    headless: bool = True
    maxCables: Optional[int] = Field(default=1, ge=1, le=5)
    horizon: Optional[int] = Field(default=None, ge=1, le=5000)
    cableThreading: Optional[dict[str, Any]] = None
    dualArmCable: Optional[dict[str, Any]] = None
    taskConfigId: Optional[str] = None
    taskName: Optional[str] = None
    name: Optional[str] = None
    evaluationTaskName: Optional[str] = None
    modelName: Optional[str] = None
    metrics: Optional[list[str]] = None
    selectedMetricIds: Optional[list[str]] = None
    config: Optional[dict[str, Any]] = None


class EvaluateAsyncResponse(BaseModel):
    evalJobId: str
    taskType: str
    taskTemplateId: Optional[str] = None
    evaluationMode: str
    status: Literal["queued", "running", "completed"] = "queued"
    runtimePath: Optional[str] = None
    resultPath: Optional[str] = None
    createdAt: Optional[str] = None
    statusUrl: Optional[str] = None
    logUrl: Optional[str] = None
    resultUrl: Optional[str] = None


class EvaluationReplayUriItem(BaseModel):
    episodeIndex: Optional[int] = None
    uri: str
    label: Optional[str] = None
    fileName: Optional[str] = None
    recordCamera: Optional[str] = None
    success: Optional[bool] = None
    sourceKind: Optional[str] = None
    evaluationMode: Optional[str] = None


class EvaluationWorkbenchBasicInfo(BaseModel):
    taskName: str
    evaluationTypeLabel: str
    evaluationObjectLabel: str
    simulationPlatform: str
    statusLabel: str
    robotType: Optional[str] = None
    modelAssetName: Optional[str] = None
    datasetName: Optional[str] = None
    associatedTaskName: Optional[str] = None
    evaluationType: Optional[str] = None
    evaluationObject: Optional[str] = None


class EvaluationJobStatusResponse(BaseModel):
    evalJobId: str
    taskType: str
    evaluationMode: str
    status: str
    phase: Optional[str] = None
    progress: Optional[float] = None
    currentEpisode: Optional[int] = None
    totalEpisodes: Optional[int] = None
    message: str = ""
    metrics: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, Any] = Field(default_factory=dict)
    updatedAt: Optional[str] = None
    requestedEpisodes: Optional[int] = None
    completedEpisodes: Optional[int] = None
    successfulEpisodes: Optional[int] = None
    failedEpisodes: Optional[int] = None
    successRate: Optional[float] = None
    recordedVideoCount: Optional[int] = None
    replayUri: Optional[str] = None
    replayUris: list[EvaluationReplayUriItem] = Field(default_factory=list)
    videoAvailable: bool = False
    videoSourceKind: Optional[str] = None
    isRepresentativeVideo: bool = False
    warning: Optional[str] = None
    taskName: Optional[str] = None
    evaluationObject: Optional[str] = None
    evaluationType: Optional[str] = None
    evaluationTypeLabel: Optional[str] = None
    simulationPlatform: Optional[str] = None
    robotType: Optional[str] = None
    modelAssetName: Optional[str] = None
    workbenchBasicInfo: Optional[EvaluationWorkbenchBasicInfo] = None


class EvaluationLogResponse(BaseModel):
    evalJobId: str
    tail: str


class EvaluationSuccessStats(BaseModel):
    successEpisodes: Optional[int] = None
    totalEpisodes: Optional[int] = None
    display: str = "-/-"
    available: bool = False
    source: Optional[str] = None
    reason: Optional[str] = None


class EvaluationJobListItem(BaseModel):
    workspaceJobId: Optional[int] = None
    evalJobId: str
    jobId: Optional[str] = None
    taskType: Optional[str] = None
    evaluationMode: Optional[str] = None
    evaluationObject: Optional[str] = None
    evaluationType: Optional[str] = None
    evaluationTypeLabel: Optional[str] = None
    status: str
    message: Optional[str] = None
    errorMessage: Optional[str] = None
    createdAt: Optional[str] = None
    updatedAt: Optional[str] = None
    startedAt: Optional[str] = None
    finishedAt: Optional[str] = None
    taskName: Optional[str] = None
    templateDisplayName: Optional[str] = None
    runner: Optional[str] = None
    runtimePath: Optional[str] = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    videoAvailable: bool = False
    requestedEpisodes: Optional[int] = None
    completedEpisodes: Optional[int] = None
    currentEpisode: Optional[int] = None
    totalEpisodes: Optional[int] = None
    progress: Optional[float] = None
    progressPercent: Optional[int] = None
    progressLabel: Optional[str] = None
    successStats: Optional[EvaluationSuccessStats] = None


class EvaluationJobListResponse(BaseModel):
    jobs: list[EvaluationJobListItem]
    total: int


class EvaluationJobDeleteResponse(BaseModel):
    success: bool = True
    evalJobId: Optional[str] = None
    deleted: bool = True
    deletedAt: Optional[str] = None
    warning: Optional[str] = None
    workspaceJobId: Optional[int] = None
    jobId: Optional[str] = None


class EvaluationPendingRecordDeleteResponse(BaseModel):
    success: bool = True
    deleted: bool = True
    workspaceJobId: int
    jobId: Optional[str] = None
    status: str = "deleted"


class EvaluationJobBatchDeleteRequest(BaseModel):
    evalJobIds: list[str] = Field(default_factory=list)
    workspaceJobIds: list[int] = Field(default_factory=list)


class EvaluationJobBatchDeleteFailedItem(BaseModel):
    evalJobId: Optional[str] = None
    workspaceJobId: Optional[int] = None
    reason: str


class EvaluationJobBatchDeleteResponse(BaseModel):
    success: bool = True
    deletedCount: int = 0
    deleted: list[str] = Field(default_factory=list)
    deletedRecordIds: list[int] = Field(default_factory=list)
    failed: list[EvaluationJobBatchDeleteFailedItem] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class DatasetEvaluateConfig(BaseModel):
    datasetId: str
    datasetName: str = ""
    metrics: list[str] = Field(default_factory=list, min_length=1)


class DatasetEvaluateRequest(BaseModel):
    evaluationType: Literal["dataset"] = "dataset"
    config: DatasetEvaluateConfig


class EvaluationReportExportRequest(BaseModel):
    format: str = "json"
    template: str = "standard"
    includeBasicInfo: bool = True
    includeConfig: bool = True
    includeMetrics: bool = True
    includeEpisodes: bool = True
    includeVideoInfo: bool = True
    includeDiagnostics: bool = True
    includeRuntimeIndex: bool = True
    includeUnavailableMetricReasons: bool = True
    force: bool = True


class EvaluationCapabilitiesResponse(BaseModel):
    taskType: str
    taskTemplateId: Optional[str] = None
    supportedModes: list[str]
    supportedPolicyTypes: list[str] = Field(default_factory=list)
    supportsCheckpoint: bool = False
    supportsPolicyEvaluation: bool = False
    supportsEpisodeStability: bool = False
    supportsTrainModelEvaluation: bool = False
    supportsVideo: bool = False
    resultArtifact: Optional[str] = None
    description: str = ""
    simulatorBackend: Optional[str] = None
    physicsBackend: Optional[str] = None
    requiresExternalRuntime: Optional[bool] = None
    defaultEnv: Optional[str] = None
    supportsDatasetGeneration: Optional[str] = None
    supportsTraining: Optional[str] = None
    supportsEvaluation: Optional[str] = None
    supportsReplay: Optional[str] = None
    scriptedExpertAvailable: Optional[bool] = None
    datasetFormats: Optional[list[str]] = None
    trainingBackends: Optional[list[str]] = None
    adapterStatus: Optional[str] = None

    model_config = {"extra": "ignore"}
