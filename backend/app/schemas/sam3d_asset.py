from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator


class AssetSegmentRequest(BaseModel):
    prompt: Optional[str] = None
    positiveBoxes: list[list[float]] = Field(default_factory=list)
    negativeBoxes: list[list[float]] = Field(default_factory=list)
    confidenceThreshold: float = 0.05
    textOnly: bool = False


class AssetReconstructRequest(BaseModel):
    cutoutIndex: Optional[int] = None
    maskIndex: Optional[int] = None
    seed: int = 42
    prepareOnly: bool = False

    @model_validator(mode="after")
    def require_selection_index(self) -> "AssetReconstructRequest":
        if self.cutoutIndex is None and self.maskIndex is None:
            raise ValueError("cutoutIndex or maskIndex is required")
        return self


class AssetRenderMujocoRequest(BaseModel):
    xmlKind: str = "preview"
    width: int = 960
    height: int = 720


class AssetFileInfo(BaseModel):
    path: str
    sizeBytes: Optional[int] = None
    exists: bool = True


class AssetExportArtifact(BaseModel):
    format: str
    path: Optional[str] = None
    status: str = "pending"


class AssetJobCreateResponse(BaseModel):
    jobId: str
    status: str
    inputImage: Optional[str] = None
    name: Optional[str] = None


class AssetJobStatusResponse(BaseModel):
    jobId: str
    name: Optional[str] = None
    status: str
    phase: str = ""
    progress: float = 0.0
    message: Optional[str] = None
    error: Optional[str] = None
    updatedAt: Optional[str] = None
    inputImage: Optional[str] = None
    targetEngine: Optional[str] = None
    assetType: Optional[str] = None
    segmentation: Optional[dict[str, Any]] = None
    reconstruction: Optional[dict[str, Any]] = None
    mujocoExport: Optional[dict[str, Any]] = None
    mujocoVisualization: Optional[dict[str, Any]] = None
    files: list[AssetFileInfo] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)
    commandSummary: Optional[str] = None


class AssetJobListResponse(BaseModel):
    jobs: list[AssetJobStatusResponse]
    total: int


class AssetJobDeleteResponse(BaseModel):
    ok: bool = True
    jobId: str
    deleted: bool = True
