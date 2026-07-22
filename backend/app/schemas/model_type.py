from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field


ModelTypeStatus = str
BaseAlgorithm = str


class ModelTypeStructureConfig(BaseModel):
    """结构配置字段因 base_algorithm 而异，使用开放 dict。"""

    model_config = {"extra": "allow"}


class ModelTypeTrainingDefaults(BaseModel):
    default_epochs: int = Field(default=5, ge=1)
    default_batch_size: int = Field(default=16, ge=1)
    default_learning_rate: float = Field(default=0.0001, gt=0)
    default_seed_strategy: str = Field(default="random")


class ModelTypeDefinitionResponse(BaseModel):
    modelTypeId: str
    name: str
    baseAlgorithm: str
    adapterKey: str
    simulator: Optional[str] = None
    robotType: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    description: Optional[str] = None
    structureConfig: dict[str, Any] = Field(default_factory=dict)
    trainingDefaults: dict[str, Any] = Field(default_factory=dict)
    status: str
    trainingReady: bool = True
    trainingReadinessStatus: str = "ready"
    disabledReason: Optional[str] = None
    isBuiltin: bool = False
    createdAt: Optional[str] = None
    updatedAt: Optional[str] = None


class ModelTypeListResponse(BaseModel):
    modelTypes: list[ModelTypeDefinitionResponse]
    total: int


class CreateModelTypeRequest(BaseModel):
    name: str = Field(min_length=1, max_length=256)
    modelTypeId: Optional[str] = Field(default=None, max_length=128)
    baseAlgorithm: str = Field(min_length=1, max_length=64)
    simulator: Optional[str] = None
    robotType: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    description: Optional[str] = None
    structureConfig: dict[str, Any] = Field(default_factory=dict)
    trainingDefaults: dict[str, Any] = Field(default_factory=dict)
    status: str = Field(default="draft")


class UpdateModelTypeRequest(BaseModel):
    name: Optional[str] = Field(default=None, max_length=256)
    simulator: Optional[str] = None
    robotType: Optional[str] = None
    tags: Optional[list[str]] = None
    description: Optional[str] = None
    structureConfig: Optional[dict[str, Any]] = None
    trainingDefaults: Optional[dict[str, Any]] = None
    status: Optional[str] = None


class ModelTypeValidateResponse(BaseModel):
    valid: bool
    errors: list[str] = Field(default_factory=list)


class ModelTypeDeleteResponse(BaseModel):
    modelTypeId: str
    deleted: bool = True


class ModelTypeProbeRefreshResponse(BaseModel):
    accepted: bool = True
