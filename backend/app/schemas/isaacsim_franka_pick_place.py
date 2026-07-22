from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class IsaacSimFrankaPickPlaceGenerateRequest(BaseModel):
    taskId: str = Field(default="isaacsim_franka_pick_place")
    episodes: int = Field(default=1, ge=1, le=5)
    seed: int = Field(default=0)
    saveVideo: bool = Field(default=True)
    saveTrajectory: bool = Field(default=True)
    headless: bool = Field(default=True)
    taskConfigId: Optional[str] = None


class IsaacSimFrankaPickPlaceGenerateAsyncResponse(BaseModel):
    jobId: str
    taskId: str
    status: str
    message: str
    statusUrl: Optional[str] = None
    videoUrl: Optional[str] = None


class IsaacSimFrankaPickPlaceJobStatusResponse(BaseModel):
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
