from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from utils.live_status import write_live_status, write_root_status

STAGE_MESSAGES: dict[str, str] = {
    "prepare_source": "正在准备 source demo (datagen_info)...",
    "mimicgen_generate": "MimicGen 正在生成 demonstrations...",
    "pinn_repair": "PINN 轨迹修复中...",
    "pinn_validation": "MuJoCo 复核中...",
    "write_manifest": "正在写入 manifest...",
    "write_summary": "正在写入 generation_summary...",
    "robosuite_rollout": "robosuite rollout 正在采集轨迹...",
    "completed": "数据生成已完成",
    "failed": "数据生成失败",
    "stalled": "任务长时间无日志更新，可能已卡住",
}


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _elapsed_seconds(started_at: str | None) -> int | None:
    if not started_at:
        return None
    try:
        started = datetime.fromisoformat(started_at)
        return max(0, int((datetime.now().astimezone() - started).total_seconds()))
    except ValueError:
        return None


def update_job_status(job_root: Path, base: dict[str, Any], **updates: Any) -> dict[str, Any]:
    payload = {**base, **updates}
    now = _now_iso()
    payload["updatedAt"] = now
    payload["lastHeartbeatAt"] = now
    if payload.get("startedAt"):
        elapsed = _elapsed_seconds(str(payload["startedAt"]))
        if elapsed is not None:
            payload["elapsedSeconds"] = elapsed
    stage = payload.get("stage")
    if stage and not payload.get("message"):
        payload["message"] = STAGE_MESSAGES.get(str(stage), str(stage))
    write_live_status(job_root / "live" / "status.json", payload)
    write_root_status(job_root, payload)
    return payload


def heartbeat_job_status(job_root: Path, base: dict[str, Any], **updates: Any) -> dict[str, Any]:
    payload = {**base, **updates}
    now = _now_iso()
    payload["updatedAt"] = now
    payload["lastHeartbeatAt"] = now
    if payload.get("startedAt"):
        elapsed = _elapsed_seconds(str(payload["startedAt"]))
        if elapsed is not None:
            payload["elapsedSeconds"] = elapsed
    write_live_status(job_root / "live" / "status.json", payload)
    write_root_status(job_root, payload)
    return payload


def set_job_stage(
    job_root: Path,
    base: dict[str, Any],
    *,
    stage: str,
    message: str | None = None,
    progress: int | None = None,
    **extra: Any,
) -> dict[str, Any]:
    msg = message or STAGE_MESSAGES.get(stage, stage)
    updates: dict[str, Any] = {"status": "running", "stage": stage, "message": msg, **extra}
    if progress is not None:
        updates["progress"] = progress
    return update_job_status(job_root, base, **updates)


def mark_job_failed(
    job_root: Path,
    base: dict[str, Any],
    *,
    error: str,
    failure_reason: str,
    traceback_text: str | None = None,
    stage: str = "failed",
) -> dict[str, Any]:
    updates: dict[str, Any] = {
        "status": "failed",
        "stage": stage,
        "message": STAGE_MESSAGES.get("failed", "数据生成失败"),
        "error": error,
        "failureReason": failure_reason,
        "progress": base.get("progress", 0),
    }
    if traceback_text:
        updates["traceback"] = traceback_text
    return update_job_status(job_root, base, **updates)


def mark_job_completed(job_root: Path, base: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return update_job_status(
        job_root,
        base,
        status="completed",
        stage="completed",
        message=STAGE_MESSAGES.get("completed", "数据生成已完成"),
        progress=100,
        **extra,
    )


def parse_important_stats(job_root: Path) -> dict[str, Any]:
    output_dir = job_root / "datasets" / "mimicgen_output"
    candidates: list[Path] = []
    if output_dir.is_dir():
        candidates.extend(output_dir.rglob("important_stats.json"))
    for path in candidates:
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            continue
    return {}


def apply_important_stats(base: dict[str, Any], stats: dict[str, Any]) -> dict[str, Any]:
    if not stats:
        return base
    num_success = stats.get("num_success")
    num_attempts = stats.get("num_attempts")
    num_failures = stats.get("num_failures")
    if num_success is not None:
        base["episodesGenerated"] = int(num_success)
    if num_failures is not None:
        base["datagenFailedTrials"] = int(num_failures)
    elif num_attempts is not None and num_success is not None:
        base["datagenFailedTrials"] = max(int(num_attempts) - int(num_success), 0)
    episodes_requested = int(base.get("episodesRequested") or base.get("episodes") or 0)
    if episodes_requested > 0 and base.get("episodesGenerated") is not None:
        base["progress"] = min(99, int(int(base["episodesGenerated"]) / episodes_requested * 100))
    return base
