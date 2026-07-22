from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

ExperimentMethodName = Literal["proposed", "baseline_b1", "baseline_b2", "baseline_b3"]
ExperimentMethodCode = Literal["P", "B1", "B2", "B3"]
PreviewModeLock = Literal["auto", "webrtc", "mjpeg"]


class ExperimentMethodConfig(BaseModel):
    name: ExperimentMethodName
    method_code: ExperimentMethodCode
    description: str
    decoupling: bool
    dual_path_preview: bool
    recovery: bool
    preview_route: str
    relay_mode: str
    preview_mode_lock: PreviewModeLock
    browser_recovery_enabled: bool


class ExperimentMethodResponse(BaseModel):
    experiment_method: ExperimentMethodConfig
    source: str


class ExperimentMethodUpdateRequest(BaseModel):
    name: ExperimentMethodName
    decoupling: Optional[bool] = None
    dual_path_preview: Optional[bool] = None
    recovery: Optional[bool] = None
    preview_route: Optional[str] = None
    relay_mode: Optional[str] = None
    preview_mode_lock: Optional[PreviewModeLock] = None
    browser_recovery_enabled: Optional[bool] = None
    description: Optional[str] = None


class ExperimentEventRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Literal["platform", "agent", "browser"]
    event: str
    ts: Optional[str] = None
    run_id: Optional[str] = None
    scenario_id: Optional[str] = None
    task_id: Optional[str] = None
    job_id: Optional[str] = None
    device_id: Optional[str] = None
    agent_id: Optional[str] = None
    command_id: Optional[str] = None
    cmd: Optional[str] = None
    path: Optional[str] = None
    method: Optional[str] = None
    success: Optional[bool] = None
    payload: Dict[str, Any] = Field(default_factory=dict)


class ExperimentEventResponse(BaseModel):
    ok: bool = True
    event_id: str
    path: str


class ExperimentSampleStartRequest(BaseModel):
    run_id: str
    scenario_id: Optional[str] = None
    task_id: Optional[str] = None
    job_id: Optional[str] = None
    device_id: Optional[str] = None
    agent_id: Optional[str] = None
    method: Optional[str] = None
    interval_sec: float = Field(default=1.0, ge=0.2, le=60.0)
    relay_pid: Optional[int] = None


class ExperimentSampleStopRequest(BaseModel):
    run_id: str


class ExperimentSampleResponse(BaseModel):
    ok: bool = True
    run_id: str
    active: bool

