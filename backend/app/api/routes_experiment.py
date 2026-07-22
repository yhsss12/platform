from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status

from app.core.config import settings
from app.core.deps import get_current_user, get_current_user_optional, require_admin
from app.models.user import User
from app.schemas.experiment import (
    ExperimentEventRequest,
    ExperimentEventResponse,
    ExperimentMethodResponse,
    ExperimentMethodUpdateRequest,
    ExperimentSampleResponse,
    ExperimentSampleStartRequest,
    ExperimentSampleStopRequest,
)
from app.services.experiment_config import get_experiment_config_service
from app.services.experiment_logger import get_experiment_logger, log_experiment_event
from app.services.experiment_sampler import SamplerContext, get_experiment_sampler_service

router = APIRouter()


def _agent_token_authorized(token: str | None) -> bool:
    expected = str(getattr(settings, "AGENT_TUNNEL_TOKEN", "") or "").strip()
    if not expected:
        return False
    return str(token or "").strip() == expected


def _stringify_optional(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _ensure_experiment_enabled() -> None:
    if bool(getattr(settings, "EXPERIMENT_ENABLED", False)):
        return
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Experiment subsystem disabled",
    )


@router.get("/method", response_model=ExperimentMethodResponse)
async def get_experiment_method(_: User = Depends(get_current_user)) -> ExperimentMethodResponse:
    _ensure_experiment_enabled()
    return get_experiment_config_service().load()


@router.put("/method", response_model=ExperimentMethodResponse)
async def update_experiment_method(
    payload: ExperimentMethodUpdateRequest,
    current_user: User = Depends(require_admin),
) -> ExperimentMethodResponse:
    _ensure_experiment_enabled()
    out = get_experiment_config_service().save(payload)
    log_experiment_event(
        role="platform",
        event="experiment_method_updated",
        actor_user_id=_stringify_optional(getattr(current_user, "id", None)),
        actor_username=_stringify_optional(getattr(current_user, "username", None)),
        method=out.experiment_method.method_code,
        experiment_method_name=out.experiment_method.name,
        payload=out.experiment_method.model_dump(),
    )
    return out


@router.post("/event", response_model=ExperimentEventResponse)
async def ingest_experiment_event(
    payload: ExperimentEventRequest,
    request: Request,
    current_user: User | None = Depends(get_current_user_optional),
    x_agent_tunnel_token: str | None = Header(default=None),
) -> ExperimentEventResponse:
    _ensure_experiment_enabled()
    role = payload.role
    if role == "agent":
        if current_user is None and not _agent_token_authorized(x_agent_tunnel_token):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Agent event auth required")
    elif current_user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")

    extra = payload.model_dump(exclude_none=True)
    role = str(extra.pop("role"))
    event = str(extra.pop("event"))
    event_id = str(extra.pop("event_id", "") or uuid.uuid4())
    payload_fields = extra.pop("payload", {})
    if isinstance(payload_fields, dict):
        extra.update(payload_fields)
    extra["event_id"] = event_id
    extra["request_path"] = str(request.url.path)
    if current_user is not None:
        extra["actor_user_id"] = _stringify_optional(getattr(current_user, "id", None))
        extra["actor_username"] = _stringify_optional(getattr(current_user, "username", None))
    row = log_experiment_event(role=role, event=event, **extra)
    log_path = get_experiment_logger().current_log_path()
    return ExperimentEventResponse(ok=True, event_id=event_id, path=str(log_path))


@router.post("/sample/start", response_model=ExperimentSampleResponse)
async def start_platform_sampler(
    payload: ExperimentSampleStartRequest,
    _: User = Depends(require_admin),
) -> ExperimentSampleResponse:
    _ensure_experiment_enabled()
    ctx = SamplerContext(
        run_id=payload.run_id,
        scenario_id=payload.scenario_id,
        task_id=payload.task_id,
        job_id=payload.job_id,
        device_id=payload.device_id,
        agent_id=payload.agent_id,
        method=payload.method,
        interval_sec=payload.interval_sec,
        relay_pid=payload.relay_pid,
    )
    await get_experiment_sampler_service().start(ctx)
    log_experiment_event(
        role="platform",
        event="platform_resource_sampler_started",
        run_id=payload.run_id,
        scenario_id=payload.scenario_id,
        task_id=payload.task_id,
        job_id=payload.job_id,
        device_id=payload.device_id,
        agent_id=payload.agent_id,
        method=payload.method,
        interval_sec=payload.interval_sec,
        relay_pid=payload.relay_pid,
    )
    return ExperimentSampleResponse(ok=True, run_id=payload.run_id, active=True)


@router.post("/sample/stop", response_model=ExperimentSampleResponse)
async def stop_platform_sampler(
    payload: ExperimentSampleStopRequest,
    _: User = Depends(require_admin),
) -> ExperimentSampleResponse:
    _ensure_experiment_enabled()
    stopped = await get_experiment_sampler_service().stop(payload.run_id)
    log_experiment_event(
        role="platform",
        event="platform_resource_sampler_stopped",
        run_id=payload.run_id,
        active=False,
    )
    return ExperimentSampleResponse(ok=True, run_id=payload.run_id, active=not stopped)
