from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class NutAssemblyPhysicsEnhancementConfig(BaseModel):
    enabled: bool = False
    method: Optional[Literal["pinn_repair"]] = None
    modelId: Optional[str] = "nut_assembly_pinn_v1"
    repairStages: Optional[list[str]] = None
    candidateSource: Optional[list[str]] = None
    maxCandidates: Optional[int] = Field(default=20, ge=1, le=100)
    maxRepairAttemptsPerCandidate: Optional[int] = Field(default=2, ge=1, le=10)
    xyErrorThreshold: Optional[float] = Field(default=0.025, gt=0)
    heightErrorThreshold: Optional[float] = Field(default=0.02, gt=0)
    validationMode: Optional[Literal["mujoco_rollout"]] = "mujoco_rollout"
    appendRepairedDemos: Optional[bool] = True


class NutAssemblyGenerateRequest(BaseModel):
    taskTemplateId: str = "nut_assembly_single_arm"
    episodes: int = Field(default=20, ge=1, le=100)
    seed: int = 0
    renderVideo: bool = True
    sourceDemoPath: Optional[str] = None
    sourceDemoSelection: Optional[Literal["official", "local", "custom", "auto"]] = None
    sourceDemoDatasetId: Optional[str] = None
    envName: str = "NutAssembly_D0"
    outputName: str = "nut_assembly_dataset"
    horizon: int = Field(default=500, ge=50, le=1000)
    taskConfigId: Optional[str] = None
    generationMode: Literal["mimicgen_datagen", "robosuite_rollout"] = "mimicgen_datagen"
    generationPath: Optional[
        Literal["expert_policy", "demo_augmentation", "expert_seed_then_augmentation"]
    ] = None
    augmentationAlgorithm: Optional[str] = None
    seedGenerationCount: Optional[int] = Field(default=None, ge=1, le=200)
    seedKeepCount: Optional[int] = Field(default=None, ge=1, le=200)
    targetCount: Optional[int] = Field(default=None, ge=1, le=500)
    autoSelectBestSeeds: Optional[bool] = None
    replayValidation: Optional[bool] = None
    expertPolicy: Optional[str] = None
    successFilter: Optional[bool] = None
    keepFailedTrajectories: Optional[bool] = None
    enablePinnRepair: Optional[bool] = None
    physicsEnhancement: Optional[NutAssemblyPhysicsEnhancementConfig] = None


class PathInfo(BaseModel):
    path: str
    exists: bool = False
    sizeBytes: Optional[int] = None


class NutAssemblyGenerateAsyncResponse(BaseModel):
    jobId: str
    taskType: str = "nut_assembly"
    status: Literal["running"] = "running"
    statusUrl: str
    resultUrl: str
    command: str


class NutAssemblyJobStatusResponse(BaseModel):
    jobId: str
    taskType: str = "nut_assembly"
    status: str
    live: dict
    paths: dict[str, PathInfo]
    metrics: dict
    command: str = ""
    startedAt: Optional[str] = None
    stage: Optional[str] = None
    progress: Optional[int] = None
    message: Optional[str] = None
    lastHeartbeatAt: Optional[str] = None
    elapsedSeconds: Optional[int] = None
    logLastModifiedAt: Optional[str] = None
    episodesRequested: Optional[int] = None
    episodesGenerated: Optional[int] = None
    datagenFailedTrials: Optional[int] = None
    datagenSuccessRate: Optional[float] = None
    traceback: Optional[str] = None
    generationMode: Optional[str] = None
    policyMode: Optional[str] = None
    sourceEnvName: Optional[str] = None
    runtimeEnvName: Optional[str] = None
    sourceDemoPath: Optional[str] = None
    sourceDemoOrigin: Optional[str] = None
    sourceDemoOriginReason: Optional[str] = None
    successRate: Optional[float] = None
    failureDistribution: Optional[dict] = None
    fallbackFrom: Optional[str] = None
    fallbackReason: Optional[str] = None
    videoUrl: Optional[str] = None
    generateVideoExists: Optional[bool] = None
    hdf5Path: Optional[str] = None
    videoPath: Optional[str] = None
    logTail: Optional[str] = None
