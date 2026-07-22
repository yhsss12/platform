from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class WorkspaceJobListQuery(BaseModel):
    jobType: Optional[str] = None
    taskType: Optional[str] = None
    status: Optional[str] = None
    source: Optional[str] = "real"
    limit: int = Field(default=50, ge=1, le=200)
    offset: int = Field(default=0, ge=0)


class WorkspaceArtifactCounts(BaseModel):
    video: int = 0
    log: int = 0
    manifest: int = 0
    metrics: int = 0
    checkpoint: int = 0
    result: int = 0
    other: int = 0


class WorkspaceJobSummary(BaseModel):
    jobId: str
    jobType: str
    taskType: str
    taskName: Optional[str] = None
    status: str
    source: str
    runner: Optional[str] = None
    createdAt: str
    updatedAt: str
    startedAt: Optional[str] = None
    finishedAt: Optional[str] = None
    runtimePath: str
    metricsSummary: dict[str, Any] = Field(default_factory=dict)
    videoAvailable: bool = False
    reportAvailable: bool = False
    artifactCounts: WorkspaceArtifactCounts = Field(default_factory=WorkspaceArtifactCounts)


class WorkspaceJobListResponse(BaseModel):
    jobs: list[WorkspaceJobSummary]
    total: int


class WorkspaceArtifactItem(BaseModel):
    id: int
    jobId: str
    artifactType: str
    name: str
    filePath: str
    urlPath: Optional[str] = None
    episodeIndex: Optional[int] = None
    createdAt: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkspaceJobDetail(WorkspaceJobSummary):
    metadata: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    errorMessage: Optional[str] = None


class WorkspaceJobArtifactsResponse(BaseModel):
    jobId: str
    artifacts: list[WorkspaceArtifactItem]


class WorkspaceReindexRequest(BaseModel):
    taskType: Optional[str] = None
    jobType: Optional[str] = None
    dryRun: bool = False
    overwrite: bool = False
    restoreDeleted: bool = False


class WorkspaceReindexResponse(BaseModel):
    scanned: int = 0
    insertedJobs: int = 0
    updatedJobs: int = 0
    insertedArtifacts: int = 0
    skipped: int = 0
    skippedDeleted: int = 0
    errors: list[str] = Field(default_factory=list)
    syncedTrainingJobs: int = 0
    syncedTrainingAssets: int = 0
    syncedEvalJobs: int = 0
    syncErrors: list[str] = Field(default_factory=list)
    scannedDatasets: int = 0
    insertedHdf5Datasets: int = 0
    updatedHdf5Datasets: int = 0
    insertedDataAssets: int = 0
    updatedDataAssets: int = 0
    skippedDatasets: int = 0


class WorkspaceJobDeleteResponse(BaseModel):
    success: bool = True
    jobId: str
    deletedJob: bool = True
    deletedArtifacts: int = 0
    deletedModelAssets: int = 0
    runtimeDeleted: bool = False
    runtimePath: str = ""
    canReindexRecover: bool = False
    reason: Optional[str] = None
    warning: Optional[str] = None
    jobType: Optional[str] = None
    taskType: Optional[str] = None
