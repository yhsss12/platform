from __future__ import annotations

from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field


class TaskTemplateResponse(BaseModel):
    id: str
    name: str
    description: str
    sourceType: Literal["standard_template", "real_data_reconstruction"] = "standard_template"
    taskFamily: str
    taskType: str
    simulatorType: Literal["mujoco", "isaac", "gazebo", "unknown"] = "unknown"
    supportedRobotTypes: list[str] = Field(default_factory=list)
    supportedPolicyTypes: list[str] = Field(default_factory=list)
    supportedEvaluationModes: list[str] = Field(default_factory=list)
    defaultSceneId: Optional[str] = None
    defaultMetricProfileId: Optional[str] = None
    defaultMetricIds: list[str] = Field(default_factory=list)
    registryTaskConfigId: Optional[str] = None
    status: str = "available"
    createdAt: str = ""
    updatedAt: str = ""
    physicsBackend: Optional[str] = None
    defaultEnv: Optional[str] = None
    adapterStatus: Optional[str] = None
    requiresExternalRuntime: bool = False
    simulatorBackendLabel: Optional[str] = None
    simulatorBackend: Optional[Literal["mujoco", "isaac_lab", "isaacsim"]] = None
    supportsDatasetGeneration: Union[bool, Literal["planned"]] = True
    replayAvailable: bool = False
    supportsImportedDemoReplay: bool = False
    hasExpertPolicy: bool = False
    hasEvaluationRunner: bool = False
    supportsDataGeneration: bool = False
    supportsEvaluation: bool = False
    defaultReplayCamera: Optional[str] = None


class TaskTemplateListResponse(BaseModel):
    taskTemplates: list[TaskTemplateResponse]
    total: int
