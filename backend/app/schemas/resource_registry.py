from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class ResourceSummary(BaseModel):
    assetId: str
    assetType: str
    name: str
    version: str
    status: str
    simBackend: str
    description: str
    tags: list[str] = Field(default_factory=list)
    files: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    manifestPath: str
    lastModifiedAt: str
    taskType: Optional[str] = None
    requiredAssets: Optional[dict[str, Any]] = None
    metrics: Optional[list[str]] = None
    runner: Optional[dict[str, Any]] = None
    defaultConfig: Optional[dict[str, Any]] = None


class ResourceListResponse(BaseModel):
    resources: list[ResourceSummary]
    total: int
    source: str = "registry"
    stats: dict[str, Any] = Field(default_factory=dict)


class ResourceReindexResponse(BaseModel):
    scanned: int = 0
    valid: int = 0
    invalid: int = 0
    synced: int = 0
    created: int = 0
    updated: int = 0
    skipped: int = 0
    resourcesByType: dict[str, int] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    lastScanAt: Optional[str] = None
    source: str = "database"


class ResourceOverviewWarning(BaseModel):
    category: str
    message: str
    path: Optional[str] = None


class ResourceOverviewResponse(BaseModel):
    taskTemplates: Optional[int] = 0
    modelAssets: Optional[int] = 0
    metrics: Optional[int] = 0
    scenes: Optional[int] = 0
    robots: Optional[int] = 0
    objects: Optional[int] = 0
    policyAssets: Optional[int] = 0
    physicsProxies: Optional[int] = 0
    modelTypes: Optional[int] = 0
    craftConfig: Optional[int] = 0
    simAssets: Optional[int] = 0
    source: str = "database"
    warnings: list[ResourceOverviewWarning] = Field(default_factory=list)
    partialFailure: bool = False


class TaskConfigSummary(BaseModel):
    assetId: str
    taskType: Optional[str] = None
    name: str
    version: str
    status: str
    simBackend: str
    description: str
    requiredAssetsCount: int = 0
    metricsCount: int = 0
    runner: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    lastModifiedAt: Optional[str] = None


class TaskConfigListResponse(BaseModel):
    taskConfigs: list[TaskConfigSummary]
    total: int


class TaskConfigDetail(TaskConfigSummary):
    requiredAssets: dict[str, Any] = Field(default_factory=dict)
    metrics: list[str] = Field(default_factory=list)
    defaultConfig: dict[str, Any] = Field(default_factory=dict)
    resolvedResources: dict[str, Any] = Field(default_factory=dict)
    manifestPath: Optional[str] = None
