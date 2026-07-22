from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class ModelAssetResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    name: str
    sourceTrainingJobId: str
    sourceDatasetId: Optional[str] = None
    taskTemplateId: Optional[str] = None
    taskType: Optional[str] = None
    sourceTaskType: Optional[str] = None
    modelType: str = ""
    framework: str = ""
    backendType: Optional[str] = None
    trainingBackend: Optional[str] = None
    checkpointPath: str = ""
    artifactPath: Optional[str] = None
    artifactKind: Optional[str] = None
    fileName: Optional[str] = None
    fileExists: bool = False
    fileSizeBytes: int = 0
    checkpointKind: Optional[str] = None
    checkpointEpoch: Optional[int] = None
    checkpointMetricName: Optional[str] = None
    datasetDisplayName: Optional[str] = None
    manifestPath: str = ""
    version: str = "v1"
    status: str = "unknown"
    canEvaluate: bool = False
    createdAt: str = ""
    updatedAt: str = ""


class TrainingJobModelAssetItemResponse(ModelAssetResponse):
    isPlaceholder: bool = False
    canEvaluate: bool = False
    displayStatus: str = "waiting"


class ModelAssetListResponse(BaseModel):
    assets: list[ModelAssetResponse] = Field(default_factory=list)
    total: int = 0


class ModelAssetFilterOptionsResponse(BaseModel):
    modelTypes: list[str] = Field(default_factory=list)
    datasets: list[str] = Field(default_factory=list)
    sourceTasks: list[str] = Field(default_factory=list)


class TrainingJobModelAssetListResponse(BaseModel):
    assets: list[TrainingJobModelAssetItemResponse] = Field(default_factory=list)
    total: int = 0
    listMessage: Optional[str] = None


class ModelAssetDeleteResponse(BaseModel):
    deleted: bool = False
    warnings: list[str] = Field(default_factory=list)


class ModelAssetImportResponse(BaseModel):
    id: str
    name: str
    modelType: str
    taskName: Optional[str] = None
    datasetName: Optional[str] = None
    createdAt: Optional[str] = None
    validationResult: dict = Field(default_factory=dict)
    assetSource: str = "imported"


class DatasetResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    name: str
    displayName: Optional[str] = None
    taskType: Optional[str] = None
    sourceJobId: str
    sourceTaskTemplateId: Optional[str] = None
    manifestPath: str = ""
    episodeCount: int = 0
    validTrajectories: Optional[int] = None
    generationRounds: Optional[int] = None
    successfulEpisodes: Optional[int] = None
    totalEpisodes: Optional[int] = None
    storagePath: str = ""
    format: Literal["hdf5", "npz", "zarr", "manifest", "unknown", "lerobot"] = "unknown"
    datasetFormat: Optional[str] = None
    sourceFormat: Optional[str] = None
    fileFormat: Optional[str] = None
    status: str = "available"
    replayAvailable: bool = False
    replayBackend: Optional[str] = None
    taskTemplateId: Optional[str] = None
    simulatorBackend: Optional[str] = None
    datasetFile: Optional[str] = None
    taskId: Optional[str] = None
    previewVideoPath: Optional[str] = None
    videoAvailable: Optional[bool] = None
    trainable: Optional[bool] = None
    trainingBackends: Optional[list[str]] = None
    observationSchema: Optional[str] = None
    actionSchema: Optional[str] = None
    controllerSchema: Optional[str] = None
    sideChannelSchema: Optional[str] = None
    attachmentSideChannel: Optional[bool] = None
    sideChannelKeys: Optional[list[str]] = None
    trainedActionMode: Optional[str] = None
    evalExecutor: Optional[str] = None
    jointActionAvailable: Optional[bool] = None
    builtDatasetPath: Optional[str] = None
    fileSizeBytes: int = 0
    dataCount: Optional[int] = None
    dataSourceLabel: Optional[str] = None
    dataScaleLabel: Optional[str] = None
    robotType: Optional[str] = None
    directTrainable: Optional[bool] = None
    needsBuild: Optional[bool] = None
    needsMapping: Optional[bool] = None
    episodeParsed: Optional[bool] = None
    generationMode: Optional[str] = None
    policyMode: Optional[str] = None
    successEpisodes: Optional[int] = None
    successRate: Optional[float] = None
    hasEpisodeMetadata: Optional[bool] = None
    hasObjectPoses: Optional[bool] = None
    totalSteps: Optional[int] = None
    demoCount: Optional[int] = None
    validForTrainingEpisodes: Optional[int] = None
    graspSuccessEpisodes: Optional[int] = None
    liftSuccessEpisodes: Optional[int] = None
    insertionSuccessEpisodes: Optional[int] = None
    averageGraspAttempts: Optional[float] = None
    hasStageStatistics: Optional[bool] = None
    trainingFilterMode: Optional[str] = None
    defaultTrainingFilterMode: Optional[str] = None
    filteredDemoCount: Optional[int] = None
    trainingBuildReady: Optional[bool] = None
    trainingHdf5Path: Optional[str] = None
    sourceDemoOrigin: Optional[str] = None
    sourceDemoPath: Optional[str] = None
    sourceDemoHash: Optional[str] = None
    envName: Optional[str] = None
    objectPoseKeys: Optional[list[str]] = None
    hasDatagenInfo: Optional[bool] = None
    episodesRequested: Optional[int] = None
    episodesGenerated: Optional[int] = None
    datagenFailedTrials: Optional[int] = None
    datagenSuccessRate: Optional[float] = None
    successStatus: Optional[str] = None
    createdAt: str = ""
    updatedAt: str = ""


class DatasetImportUploadResponse(BaseModel):
    dataset: DatasetResponse
    datasetId: str
    status: str
    validationReport: dict = Field(default_factory=dict)


class DatasetListResponse(BaseModel):
    datasets: list[DatasetResponse] = Field(default_factory=list)
    total: int = 0


class DatasetFieldMappingRequest(BaseModel):
    action: Optional[str] = None
    qpos: Optional[str] = None
    image: Optional[str] = None
    qvel: Optional[str] = None
    done: Optional[str] = None


class DatasetEpisodeRuleRequest(BaseModel):
    type: str = "single_episode"


class BuildDatasetFromImportRequest(BaseModel):
    sourceDatasetId: str
    outputName: str
    taskType: str = "custom"
    targetFormat: str = "standard_hdf5"
    fieldMapping: Optional[DatasetFieldMappingRequest] = None
    auto: bool = True
    episodeRule: DatasetEpisodeRuleRequest = Field(default_factory=DatasetEpisodeRuleRequest)


class BuildDatasetFromImportResponse(BaseModel):
    builtDatasetId: str
    status: str
    trainable: bool = False
    directTrainable: bool = True
    dataset: Optional[DatasetResponse] = None
