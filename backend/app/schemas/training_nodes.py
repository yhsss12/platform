from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

TrainingNodeStatus = Literal["available", "busy", "unreachable", "misconfigured", "placeholder"]


class TrainingNodeCheckResult(BaseModel):
    ok: Optional[bool] = None
    detail: str = ""


class TrainingNodeGpuInfo(BaseModel):
    name: Optional[str] = None
    memoryTotalMb: Optional[float] = None
    memoryUsedMb: Optional[float] = None
    memoryFreeMb: Optional[float] = None
    memoryUsedRatio: Optional[float] = None


class TrainingNodeListItem(BaseModel):
    nodeId: str
    label: str
    deviceLabel: str
    trainingNodeDisplayName: Optional[str] = None
    executionMode: str
    description: str = ""
    status: TrainingNodeStatus
    statusLabel: str
    message: str = ""
    selectable: bool = True
    host: Optional[str] = None
    sshTarget: Optional[str] = None
    workdir: Optional[str] = None
    gpuModel: Optional[str] = None
    gpuMemoryGb: Optional[float] = None
    gpu: Optional[TrainingNodeGpuInfo] = None
    checks: Optional[dict[str, TrainingNodeCheckResult]] = None


class TrainingNodeProbeResponse(BaseModel):
    node: TrainingNodeListItem


class TrainingNodesListResponse(BaseModel):
    nodes: list[TrainingNodeListItem] = Field(default_factory=list)
