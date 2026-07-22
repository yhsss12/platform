from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class DualArmCableGenerateRequest(BaseModel):
    taskType: str = "dual_arm_cable_manipulation"
    taskName: str = "线缆整理"
    maxCables: int = Field(default=1, ge=1, le=10)
    numEpisodes: Optional[int] = Field(default=None, ge=1, le=10)
    seed: int = 42
    record: bool = True
    headless: bool = True
    stretchMode: Literal["ema_jump", "fixed_distance", "fixed_force"] = "fixed_distance"
    releaseMode: Literal["three_phase", "direct_open", "slow_open"] = "three_phase"
    taskConfigId: Optional[str] = None


class DualArmCableGenerateAsyncResponse(BaseModel):
    jobId: str
    taskType: str = "dual_arm_cable_manipulation"
    status: Literal["queued", "running"] = "queued"
    frameUrl: Optional[str] = None
    statusUrl: Optional[str] = None


class DualArmCableJobStatusResponse(BaseModel):
    jobId: str
    taskType: str = "dual_arm_cable_manipulation"
    status: str
    progress: Optional[float] = None
    phase: Optional[str] = None
    maxCables: int = 1
    succeededCables: int = 0
    episodeSuccess: bool = False
    videoExists: bool = False
    liveFrameExists: bool = False
    liveFrameSeq: Optional[int] = None
    liveFrameUpdatedAt: Optional[str] = None
    liveFrameSource: Optional[str] = None
    currentStep: Optional[int] = None
    episodeIndex: Optional[int] = None
    videoPath: Optional[str] = None
    resultPath: Optional[str] = None
    runtimePath: Optional[str] = None
    logPath: Optional[str] = None
    manifestPath: Optional[str] = None
    message: str = ""
    metrics: dict[str, Any] = Field(default_factory=dict)
    frameUrl: Optional[str] = None
    videoUrl: Optional[str] = None
    logUrl: Optional[str] = None
    resultUrl: Optional[str] = None


class DualArmIlExportProbeResponse(BaseModel):
    jobId: str
    exportReady: bool
    failureReason: Optional[str] = None
    actionAvailable: bool = False
    observationAvailable: bool = False
    missingFields: list[str] = Field(default_factory=list)
    hdf5Exists: bool = False
    manifestExists: bool = False
    hdf5Path: Optional[str] = None
    manifestPath: Optional[str] = None
    trainable: bool = False
    exportReport: dict[str, Any] = Field(default_factory=dict)


class DualArmIlExportBuildResponse(BaseModel):
    jobId: str
    status: Literal["built", "already_built"]
    manifestPath: str
    hdf5Path: str
    message: str
    manifest: Optional[dict[str, Any]] = None
    exportReport: Optional[dict[str, Any]] = None
