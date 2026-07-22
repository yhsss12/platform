from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class IsaacLabFrankaStackCubeJobStatusResponse(BaseModel):
    jobId: str
    taskId: str
    status: str
    progress: Optional[float] = None
    totalEpisodes: Optional[int] = None
    completedEpisodes: Optional[int] = None
    successEpisodes: Optional[int] = None
    failedEpisodes: Optional[int] = None
    outputDir: Optional[str] = None
    datasetId: Optional[str] = None
    runtimeMode: Optional[str] = None
    message: str = ""
    videoExists: bool = False
    video_status: Optional[str] = None
    videoStatus: Optional[str] = None
    taskIdValidated: Optional[bool] = None
    validationError: Optional[str] = None
    videoPath: Optional[str] = None
    episodeId: Optional[str] = None
    episodeManifest: Optional[dict] = None
    datasetManifest: Optional[dict] = None
    metrics: dict = Field(default_factory=dict)
    statusUrl: Optional[str] = None
    videoUrl: Optional[str] = None
    logUrl: Optional[str] = None
    manifestPath: Optional[str] = None
    generationMode: Optional[str] = None
    phase: Optional[str] = None
    phaseLabel: Optional[str] = None
    phaseStartedAt: Optional[str] = None
    phaseUpdatedAt: Optional[str] = None
    phaseTimings: Optional[dict] = None
    progressMessage: Optional[str] = None
    errorSummary: Optional[str] = None
    requestedDevice: Optional[str] = None
    resolvedDevice: Optional[str] = None
    cudaVisibleDevices: Optional[str] = None
    isGpuRequested: Optional[bool] = None
    torchCudaAvailable: Optional[bool] = None
    liveFrameAvailable: Optional[bool] = None
    liveFrameBlack: Optional[bool] = None
    liveFrameExists: Optional[bool] = None
    enableCameras: Optional[bool] = None
    liveFrameUrl: Optional[str] = None
