from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from app.schemas.workspace_benchmark import DatasetResponse


class IsaacLabRuntimeStatusResponse(BaseModel):
    enabled: bool = False
    configured: bool = False
    available: bool = False
    runtimeMode: str = "external_subprocess"
    isaacLabRoot: Optional[str] = None
    isaacLabSh: Optional[str] = None
    isaacLabPython: Optional[str] = None
    isaacLabVersion: Optional[str] = None
    defaultTask: str = "Isaac-Stack-Cube-Franka-IK-Rel-v0"
    gpuAvailable: bool = False
    taskRegistered: bool = False
    mimicTaskRegistered: bool = False
    outputRoot: Optional[str] = None
    defaultSeedFile: Optional[str] = None
    defaultSeedAvailable: bool = False
    stackCubeGenerationReady: bool = False
    scriptedExpertAvailable: bool = False
    scriptedExpertReady: bool = False
    scriptedExpertIssueCodes: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    stackCubeIssueCodes: list[str] = Field(default_factory=list)


class IsaacLabSmokeTestRequest(BaseModel):
    keyword: str = "Stack"


class IsaacLabSmokeTestResponse(BaseModel):
    jobId: str
    kind: str = "smoke_test"
    status: str
    runtimePath: str
    statusUrl: str
    logPaths: dict[str, str] = Field(default_factory=dict)


class IsaacLabRunJobStatusResponse(BaseModel):
    jobId: str
    kind: Optional[str] = None
    status: str
    phase: Optional[str] = None
    message: Optional[str] = None
    command: Optional[list[str]] = None
    keyword: Optional[str] = None
    taskId: Optional[str] = None
    datasetFile: Optional[str] = None
    datasetName: Optional[str] = None
    datasetAvailable: Optional[bool] = None
    datasetId: Optional[str] = None
    generationMode: Optional[str] = None
    numDemos: Optional[int] = None
    totalEpisodes: Optional[int] = None
    completedEpisodes: Optional[int] = None
    successfulEpisodes: Optional[int] = None
    currentEpisode: Optional[int] = None
    episodeCount: Optional[int] = None
    progress: Optional[int] = None
    seed: Optional[int] = None
    headless: Optional[bool] = None
    enableCameras: Optional[bool] = None
    videoRequested: Optional[bool] = None
    videoAvailable: Optional[bool] = None
    videoPath: Optional[str] = None
    videoNote: Optional[str] = None
    exitCode: Optional[int] = None
    timedOut: Optional[bool] = None
    stackEnvMatches: Optional[int] = None
    liveFrameAvailable: Optional[bool] = None
    liveFrameBlack: Optional[bool] = None
    latestFramePath: Optional[str] = None
    previewVideoAvailable: Optional[bool] = None
    visualPhase: Optional[str] = None
    visualNumEnvs: Optional[int] = None
    parallelNumEnvs: Optional[int] = None
    visualMode: Optional[str] = None
    visualEnvIndex: Optional[int] = None
    seedSource: Optional[str] = None
    artifactStatus: Optional[dict[str, bool]] = None
    startedAt: Optional[str] = None
    finishedAt: Optional[str] = None
    updatedAt: Optional[str] = None
    paths: dict[str, str] = Field(default_factory=dict)


class IsaacLabReplayDemoRequest(BaseModel):
    taskId: str = "Isaac-Stack-Cube-Franka-IK-Rel-v0"
    datasetFile: str
    headless: bool = True
    enableCameras: bool = True
    video: bool = True


class IsaacLabReplayDemoResponse(BaseModel):
    jobId: str
    kind: str = "replay_demo"
    status: str
    runtimePath: Optional[str] = None
    statusUrl: Optional[str] = None
    logPaths: dict[str, str] = Field(default_factory=dict)


class IsaacLabImportDemoRequest(BaseModel):
    datasetFile: str
    displayName: str
    taskId: str = "Isaac-Stack-Cube-Franka-IK-Rel-v0"


class IsaacLabImportDemoResponse(BaseModel):
    dataset: DatasetResponse


class IsaacLabReplayFromDatasetResponse(BaseModel):
    datasetId: str
    jobId: str
    kind: str = "replay_demo"
    status: str
    runtimePath: Optional[str] = None
    statusUrl: Optional[str] = None
    reused: bool = False


class IsaacLabDatasetPlaybackInfo(BaseModel):
    videoJobId: Optional[str] = None
    videoSource: str = "none"
    videoSourceKind: Optional[str] = None
    videoPath: Optional[str] = None
    rawVideoPath: Optional[str] = None
    browserVideoPath: Optional[str] = None
    codec: Optional[str] = None
    browserCompatible: bool = False
    transcoded: bool = False
    transcodeNote: Optional[str] = None
    playable: bool = False


class IsaacLabDatasetReplayContextResponse(BaseModel):
    dataset: DatasetResponse
    sourceJobId: Optional[str] = None
    sourceJobStatus: Optional[dict[str, object]] = None
    replayJobs: list[dict[str, object]] = Field(default_factory=list)
    replayJobId: Optional[str] = None
    replayJobStatus: Optional[str] = None
    replayInProgress: bool = False
    replayFailed: bool = False
    playback: Optional[IsaacLabDatasetPlaybackInfo] = None
    usingPreviewFallback: bool = False
    hasDatasetFile: bool = False
    videoSourceLabel: str = "视频来源：暂无"


class IsaacLabGenerateDatasetRequest(BaseModel):
    taskId: str = "Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0"
    datasetName: str
    numDemos: int = 10
    seed: int = 0
    headless: bool = True
    enableCameras: bool = True
    generationMode: str = "mimic_auto"
    seedDatasetFile: Optional[str] = None
    seedDatasetId: Optional[str] = None
    video: bool = True
    numEnvs: Optional[int] = None


class IsaacLabGenerateDatasetResponse(BaseModel):
    jobId: str
    kind: str = "generate_dataset"
    status: str
    runtimePath: Optional[str] = None
    statusUrl: Optional[str] = None
    logPaths: dict[str, str] = Field(default_factory=dict)


class IsaacLabRunJobLogResponse(BaseModel):
    jobId: str
    stream: str
    tail: str = ""
