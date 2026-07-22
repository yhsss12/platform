from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class CableThreadingGenerateRequest(BaseModel):
    episodes: int = Field(default=10, ge=1, le=100)
    robot: str = "Panda"
    cableModel: str = "composite_cable"
    difficulty: str = "easy"
    horizon: int = Field(default=600, ge=100, le=1000)
    seed: int = 0
    outputFormat: Literal["npz", "hdf5", "lerobot"] = "hdf5"
    saveHdf5: bool = True
    saveProcessVideo: bool = True
    lerobotTaskInstruction: Optional[str] = "thread the cable through the pole"
    lerobotRobot: Optional[str] = "Panda"
    lerobotFps: Optional[int] = 20
    taskConfigId: Optional[str] = None


class CableThreadingEvaluateRequest(BaseModel):
    episodes: int = Field(default=10, ge=1, le=100)
    robot: str = "Panda"
    cableModel: str = "composite_cable"
    difficulty: str = "easy"
    horizon: int = Field(default=600, ge=100, le=1000)
    seed: int = 0
    policy: str = "scripted"
    checkpoint: Optional[str] = None
    device: str = "cpu"
    taskConfigId: Optional[str] = None


class CableThreadingVideoRequest(BaseModel):
    episodes: int = Field(default=1, ge=1, le=100)
    robot: str = "Panda"
    cableModel: str = "composite_cable"
    difficulty: str = "easy"
    horizon: int = Field(default=600, ge=100, le=1000)
    seed: int = 0


class PathInfo(BaseModel):
    path: str
    exists: bool = False
    sizeBytes: Optional[int] = None


class CableThreadingGenerateAsyncResponse(BaseModel):
    jobId: str
    taskType: str = "cable_threading"
    status: Literal["running"] = "running"
    frameUrl: str
    statusUrl: str
    command: str


class CableThreadingEvaluateAsyncResponse(BaseModel):
    evalJobId: str
    jobId: str
    taskType: str = "cable_threading"
    status: Literal["queued", "running"] = "queued"
    statusUrl: str
    command: str


class EvaluationReplayUriItem(BaseModel):
    episodeIndex: Optional[int] = None
    uri: str
    label: Optional[str] = None
    fileName: Optional[str] = None
    recordCamera: Optional[str] = None
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


class ReplayContentTab(BaseModel):
    id: str
    label: str


class ReplayFailureRecord(BaseModel):
    episodeIndex: Optional[int] = None
    seed: Optional[int] = None
    failureReason: Optional[str] = None
    writtenToDataset: bool = False


class ReplayContentDetection(BaseModel):
    replayContentKind: Literal[
        "dataset_trajectory_replay",
        "generation_process_preview",
        "evaluation_replay",
    ]
    hasHdf5Trajectories: bool = False
    trajectoryCount: Optional[int] = None
    totalEpisodes: Optional[int] = None
    failedEpisodes: Optional[int] = None
    hasGenerationPreview: bool = False
    hasFailures: bool = False
    hasEvaluationResult: bool = False
    primarySource: Optional[str] = None
    tabs: list[ReplayContentTab] = Field(default_factory=list)
    trajectories: list[str] = Field(default_factory=list)
    failureRecords: list[ReplayFailureRecord] = Field(default_factory=list)
    debug: dict[str, Any] = Field(default_factory=dict)
    hasRgbObservation: bool = False
    rgbCameras: list[str] = Field(default_factory=list)
    trajectoryDisplayMode: Optional[Literal["rgb_frame_replay", "state_trajectory"]] = None


class CableThreadingJobStatusResponse(BaseModel):
    jobId: str
    evalJobId: Optional[str] = None
    taskType: str = "cable_threading"
    status: str
    live: dict[str, Any] = Field(default_factory=dict)
    paths: dict[str, PathInfo] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    command: str = ""
    startedAt: Optional[str] = None
    generateVideoExists: bool = False
    generateVideoSizeBytes: Optional[int] = None
    generateVideoPath: Optional[str] = None
    evalVideoExists: bool = False
    evalVideoSizeBytes: Optional[int] = None
    evalVideoPath: Optional[str] = None
    evalBrowserVideoPath: Optional[str] = None
    evalBrowserVideoExists: bool = False
    browserVideoPath: Optional[str] = None
    videoResolution: Optional[str] = None
    evalVideoStatus: Optional[str] = None
    videoUrl: Optional[str] = None
    timelineExists: bool = False
    timelinePath: Optional[str] = None
    timelineUrl: Optional[str] = None
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
    currentEpisodeIndex: Optional[int] = None
    recordCamera: Optional[str] = None
    cameraFallbackUsed: Optional[bool] = None
    taskName: Optional[str] = None
    evaluationMode: Optional[str] = None
    evaluationObject: Optional[str] = None
    evaluationType: Optional[str] = None
    evaluationTypeLabel: Optional[str] = None
    simulationPlatform: Optional[str] = None
    robotType: Optional[str] = None
    modelAssetName: Optional[str] = None
    workbenchBasicInfo: Optional[EvaluationWorkbenchBasicInfo] = None
    selectedMetricIds: Optional[list[str]] = None
    metricResults: Optional[dict[str, Any]] = None
    runMetrics: Optional[dict[str, Any]] = None
    replayContent: Optional[ReplayContentDetection] = None


class CableThreadingGenerateResponse(BaseModel):
    jobId: str
    taskType: str = "cable_threading"
    status: Literal["completed", "failed"]
    command: str
    paths: dict[str, PathInfo]
    metrics: dict[str, Any]
    stdoutTail: list[str]


class CableThreadingEvaluateResponse(BaseModel):
    jobId: str
    taskType: str = "cable_threading"
    status: Literal["completed", "failed"]
    command: str
    paths: dict[str, PathInfo]
    metrics: dict[str, Any]
    stdoutTail: list[str]


class CableThreadingVideoResponse(BaseModel):
    jobId: str
    taskType: str = "cable_threading"
    status: Literal["completed", "failed"]
    command: str
    paths: dict[str, PathInfo]
    videoExists: bool = False
    videoSizeBytes: Optional[int] = None
    stdoutTail: list[str]
