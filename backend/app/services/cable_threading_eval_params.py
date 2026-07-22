"""线缆穿杆评测请求参数归一化（episodes / recordVideo / horizon）。"""

from __future__ import annotations

from typing import Any, Optional

from app.schemas.evaluation import EvaluateAsyncRequest

DEFAULT_CABLE_EVAL_DISPLAY_CAMERA = "agentview"


def _pick_int(*values: Any) -> Optional[int]:
    for value in values:
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str) and value.strip().isdigit():
            return int(value.strip())
    return None


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def resolve_cable_eval_episodes(request: EvaluateAsyncRequest) -> int:
    config = _as_dict(request.config)
    cable = _as_dict(request.cableThreading)
    episodes = _pick_int(
        config.get("episodes"),
        cable.get("episodes"),
        request.numEpisodes,
    )
    return max(1, episodes if episodes is not None else 10)


def resolve_cable_eval_horizon(request: EvaluateAsyncRequest) -> int:
    config = _as_dict(request.config)
    cable = _as_dict(request.cableThreading)
    horizon = _pick_int(
        config.get("horizon"),
        cable.get("horizon"),
        request.horizon,
    )
    return max(1, horizon if horizon is not None else 600)


def resolve_cable_eval_seed(request: EvaluateAsyncRequest) -> int:
    config = _as_dict(request.config)
    cable = _as_dict(request.cableThreading)
    seed = _pick_int(config.get("seed"), cable.get("seed"), request.seed)
    return seed if seed is not None else 0


def resolve_cable_record_video(request: EvaluateAsyncRequest) -> bool:
    config = _as_dict(request.config)
    cable = _as_dict(request.cableThreading)
    if config.get("recordVideo") is not None:
        return bool(config.get("recordVideo"))
    if cable.get("recordVideo") is not None:
        return bool(cable.get("recordVideo"))
    return bool(request.record)


def _pick_str(*values: Any) -> Optional[str]:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def resolve_cable_eval_display_camera(request: EvaluateAsyncRequest) -> str:
    config = _as_dict(request.config)
    cable = _as_dict(request.cableThreading)
    camera = _pick_str(
        config.get("evalDisplayCamera"),
        config.get("eval_display_camera"),
        config.get("recordCamera"),
        cable.get("evalDisplayCamera"),
        cable.get("eval_display_camera"),
        cable.get("recordCamera"),
    )
    return camera or DEFAULT_CABLE_EVAL_DISPLAY_CAMERA


def resolve_cable_allow_camera_fallback(request: EvaluateAsyncRequest) -> bool:
    config = _as_dict(request.config)
    cable = _as_dict(request.cableThreading)
    if config.get("allowCameraFallback") is not None:
        return bool(config.get("allowCameraFallback"))
    if cable.get("allowCameraFallback") is not None:
        return bool(cable.get("allowCameraFallback"))
    return False
