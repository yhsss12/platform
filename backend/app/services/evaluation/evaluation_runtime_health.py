"""评测任务 runtime 健康检查：进程存活、文件 stale、状态自愈。"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from app.core.platform_paths import platform_paths, resolve_runtime_reference
from app.services.evaluation.job_paths import PROJECT_ROOT

logger = logging.getLogger(__name__)

EVALUATION_RUNNING_STALE_THRESHOLD_SECONDS = 600
EVALUATION_RUNNING_MAX_AGE_SECONDS = 24 * 60 * 60

RUNNING_DECLARED_STATUSES = frozenset({"running", "evaluating", "training"})
EVALUATION_RECONCILE_STATUSES = frozenset({"running", "evaluating", "queued", "pending", "unknown"})

MAX_AGE_FAILURE_REASON = "评测任务运行时间超过上限且未检测到进程，已判定为异常终止。"
RUNTIME_MISSING_FAILURE_REASON = "评测 runtime 目录不存在或已失效，已判定为异常终止。"
STALE_RUNNING_FAILURE_REASON = "评测进程已退出或失联，状态文件长时间未更新"
TERMINAL_ACTUAL_STATUSES = frozenset({"completed", "failed", "canceled", "cancelled"})
COMPLETED_DECLARED_STATUSES = frozenset({"completed", "succeeded", "success"})
FAILED_DECLARED_STATUSES = frozenset({"failed", "error", "errored", "stale"})
PENDING_DECLARED_STATUSES = frozenset({"pending", "queued", "draft", "unknown"})

_EXCLUDED_PROCESS_KEYWORDS = (
    "uvicorn",
    "node ",
    "nodejs",
    "npm",
    "postgres",
    "grep",
    " rg ",
    "cursor",
    "tsserver",
    "bash -",
    "dump_bash_state",
    "reconcile_evaluation",
    "pytest",
)

_EVAL_RUNNER_HINTS = (
    "platform_runner",
    "run_stacked",
    "eval_worker",
    "evaluation_service",
    "cable_threading",
    "isaac_lab",
    "run.py",
    "evaluate",
    "policy_eval",
    "spawn_main",
)

_FAILURE_LOG_PATTERNS = re.compile(
    r"(Traceback \(most recent call last\)|"
    r"process exited unexpectedly|"
    r"\bKilled\b|"
    r"Segmentation fault|"
    r"Uncaught exception|"
    r"Fatal Python error)",
    re.IGNORECASE,
)

_ROOT_LOG_NAMES = ("eval.log", "run.log", "evaluation.log")

_STATUS_CANDIDATE_RELATIVE = (
    "live/status.json",
    "status.json",
    "metadata/status.json",
)

_ARTIFACT_GLOBS = (
    "logs/*.log",
    "logs/**/*.log",
    "episodes/*/logs/*.log",
    "episodes/**/logs/*.log",
    "results/*",
    "results/**/*",
    "videos/*",
    "live/frames/*",
    "live/*.jpg",
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _parse_iso_timestamp(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return None


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def resolve_evaluation_job_root(job_id: str, runtime_path: Optional[str] = None) -> Optional[Path]:
    candidate = (job_id or "").strip()
    if not candidate:
        return None

    if runtime_path:
        path = resolve_runtime_reference(runtime_path)
        if path.is_dir():
            return path.resolve()

    if candidate.startswith("ct_eval_"):
        from app.services.cable_threading_service import _job_dir

        ct_root = _job_dir(candidate)
        if ct_root.is_dir():
            return ct_root.resolve()

    if candidate.startswith("isaac_eval_"):
        isaac_root = platform_paths.runs_root / "isaac_lab" / "jobs" / candidate
        if isaac_root.is_dir():
            return isaac_root.resolve()

    eval_root = platform_paths.evaluation_jobs / candidate
    if eval_root.is_dir():
        return eval_root.resolve()

    return None


def _resolve_primary_status_path(job_root: Path) -> Optional[Path]:
    for relative in _STATUS_CANDIDATE_RELATIVE:
        path = job_root / relative
        if path.is_file():
            return path
    return None


def _collect_runtime_mtimes(job_root: Path) -> dict[str, Optional[float]]:
    status_path = _resolve_primary_status_path(job_root)
    status_mtime = status_path.stat().st_mtime if status_path and status_path.is_file() else None

    log_mtimes: list[float] = []
    result_mtimes: list[float] = []
    artifact_mtimes: list[float] = []

    if job_root.is_dir():
        for pattern in _ARTIFACT_GLOBS:
            for path in job_root.glob(pattern):
                if not path.is_file():
                    continue
                try:
                    mtime = path.stat().st_mtime
                except OSError:
                    continue
                rel = str(path.relative_to(job_root))
                if "/logs/" in rel or rel.startswith("logs/"):
                    log_mtimes.append(mtime)
                elif rel.startswith("results/"):
                    result_mtimes.append(mtime)
                else:
                    artifact_mtimes.append(mtime)

    latest_log = max(log_mtimes) if log_mtimes else None
    latest_result = max(result_mtimes) if result_mtimes else None
    latest_artifact = max(
        [value for value in (status_mtime, latest_log, latest_result, *artifact_mtimes) if value is not None],
        default=None,
    )

    return {
        "status_mtime": status_mtime,
        "latest_log_mtime": latest_log,
        "latest_result_mtime": latest_result,
        "latest_artifact_mtime": latest_artifact,
    }


def _mtime_to_iso(mtime: Optional[float]) -> Optional[str]:
    if mtime is None:
        return None
    return datetime.fromtimestamp(mtime, tz=timezone.utc).replace(microsecond=0).isoformat()


def _is_eval_runner_cmdline(cmdline: str, job_id: str, runtime_path: str) -> bool:
    lowered = cmdline.lower()
    if any(keyword in lowered for keyword in _EXCLUDED_PROCESS_KEYWORDS):
        return False
    if job_id not in cmdline:
        return False
    if runtime_path and runtime_path in cmdline:
        return True
    runtime_tail = Path(runtime_path).name if runtime_path else ""
    if runtime_tail and runtime_tail in cmdline:
        return any(hint in lowered for hint in _EVAL_RUNNER_HINTS)
    return any(hint in lowered for hint in _EVAL_RUNNER_HINTS)


def _read_declared_status(job_root: Optional[Path]) -> str:
    if job_root is None or not job_root.is_dir():
        return "unknown"
    for relative in _STATUS_CANDIDATE_RELATIVE:
        payload = _read_json(job_root / relative)
        status_value = str(payload.get("status") or "").strip().lower()
        if status_value:
            return status_value
    return "unknown"


def _scan_log_failures(job_root: Path) -> tuple[bool, Optional[str]]:
    log_paths: list[Path] = []
    logs_dir = job_root / "logs"
    if logs_dir.is_dir():
        for name in _ROOT_LOG_NAMES:
            path = logs_dir / name
            if path.is_file():
                log_paths.append(path)
    if not log_paths:
        return False, None

    newest = max(log_paths, key=lambda path: path.stat().st_mtime)
    try:
        text = newest.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False, None
    tail = text[-8000:]
    if _FAILURE_LOG_PATTERNS.search(tail):
        return True, f"日志 {newest.relative_to(job_root)} 含错误/异常信息"
    return False, None


def _detect_completed_result(job_root: Path, status_data: dict[str, Any]) -> bool:
    declared = str(status_data.get("status") or "").strip().lower()
    if declared in COMPLETED_DECLARED_STATUSES:
        return True

    for relative in (
        "results/aggregate_result.json",
        "results/eval.results.json",
        "results/result.json",
    ):
        payload = _read_json(job_root / relative)
        result_status = str(payload.get("status") or "").strip().lower()
        if result_status in COMPLETED_DECLARED_STATUSES:
            return True
        if payload.get("successRate") is not None or payload.get("finalSuccessRate") is not None:
            if result_status not in FAILED_DECLARED_STATUSES and result_status not in RUNNING_DECLARED_STATUSES:
                return True

    # A backend restart can stop the parent evaluator after its last episode
    # has already persisted a successful result but before the aggregate files
    # are written. Treat a complete episode set as recoverable completion.
    evaluation_mode = str(status_data.get("evaluationMode") or "").strip().lower()
    total_episodes = int(status_data.get("totalEpisodes") or 0)
    if evaluation_mode == "episode_stability" and total_episodes > 0:
        completed_episodes = 0
        for episode_index in range(total_episodes):
            episode_dir = job_root / "episodes" / f"episode_{episode_index:02d}"
            episode_status = _read_json(episode_dir / "status.json")
            episode_result = _read_json(episode_dir / "results" / "episode_result.json")
            if not episode_result:
                episode_result = _read_json(episode_dir / "episode" / "episode_result.json")
            if (
                str(episode_status.get("status") or "").strip().lower()
                in COMPLETED_DECLARED_STATUSES
                and bool(episode_result)
            ):
                completed_episodes += 1
        if completed_episodes == total_episodes:
            return True
    return False


def _detect_failed_result(job_root: Path, status_data: dict[str, Any]) -> tuple[bool, Optional[str]]:
    declared = str(status_data.get("status") or "").strip().lower()
    if declared in FAILED_DECLARED_STATUSES:
        reason = str(status_data.get("error") or status_data.get("message") or "runtime status=failed")
        return True, reason

    if status_data.get("error"):
        err = str(status_data.get("error"))
        if "episodes/" not in err:
            return True, err

    for relative in (
        "results/aggregate_result.json",
        "results/eval.results.json",
        "results/result.json",
    ):
        payload = _read_json(job_root / relative)
        result_status = str(payload.get("status") or "").strip().lower()
        if result_status in FAILED_DECLARED_STATUSES:
            return True, str(payload.get("error") or payload.get("message") or f"{relative} status=failed")

    log_failed, log_reason = _scan_log_failures(job_root)
    if log_failed:
        return True, log_reason
    return False, None


def find_processes_for_job(job_id: str, runtime_path: Optional[str] = None) -> list[dict[str, Any]]:
    candidate = (job_id or "").strip()
    if not candidate:
        return []

    job_root = resolve_evaluation_job_root(candidate, runtime_path)
    runtime_text = str(job_root) if job_root else (runtime_path or "")

    matches: list[dict[str, Any]] = []
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return matches

    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        cmdline_path = entry / "cmdline"
        if not cmdline_path.is_file():
            continue
        try:
            raw = cmdline_path.read_bytes().replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()
        except OSError:
            continue
        if not raw:
            continue

        if not _is_eval_runner_cmdline(raw, candidate, runtime_text):
            continue

        stat_path = entry / "stat"
        etime = ""
        try:
            stat_fields = stat_path.read_text(encoding="utf-8", errors="replace").split()
            if len(stat_fields) > 21:
                start_ticks = int(stat_fields[21])
                clk_tck = 100
                uptime_path = Path("/proc/uptime")
                if uptime_path.is_file():
                    uptime_seconds = float(uptime_path.read_text().split()[0])
                    elapsed = max(0.0, uptime_seconds - start_ticks / clk_tck)
                    hours, rem = divmod(int(elapsed), 3600)
                    minutes, seconds = divmod(rem, 60)
                    etime = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        except (OSError, ValueError, IndexError):
            etime = ""

        matches.append(
            {
                "pid": pid,
                "cmdline": raw,
                "etime": etime,
            }
        )
    return matches


def _job_age_seconds(
    now: float,
    *,
    created_at: Optional[str] = None,
    started_at: Optional[str] = None,
) -> Optional[int]:
    start_ts = _parse_iso_timestamp(started_at) or _parse_iso_timestamp(created_at)
    if start_ts is None:
        return None
    return max(0, int(now - start_ts))


def inspect_evaluation_runtime_health(
    job_id: str,
    runtime_path: Optional[str] = None,
    *,
    declared_status: Optional[str] = None,
    created_at: Optional[str] = None,
    started_at: Optional[str] = None,
) -> dict[str, Any]:
    candidate = (job_id or "").strip()
    job_root = resolve_evaluation_job_root(candidate, runtime_path)
    status_path = _resolve_primary_status_path(job_root) if job_root else None
    status_data = _read_json(status_path) if status_path else {}

    declared = str(declared_status or status_data.get("status") or "unknown").strip().lower()
    processes = find_processes_for_job(candidate, runtime_path)
    is_process_alive = bool(processes)
    matched_pids = [item["pid"] for item in processes]

    mtimes = _collect_runtime_mtimes(job_root) if job_root else {}
    now = time.time()

    status_ts = _parse_iso_timestamp(status_data.get("updatedAt"))
    if status_ts is None:
        status_ts = mtimes.get("status_mtime")

    candidate_ts = [
        status_ts,
        mtimes.get("latest_log_mtime"),
        mtimes.get("latest_result_mtime"),
        mtimes.get("latest_artifact_mtime"),
    ]
    candidate_ts = [value for value in candidate_ts if value is not None]
    last_runtime_ts = max(candidate_ts) if candidate_ts else None
    stale_seconds = int(now - last_runtime_ts) if last_runtime_ts is not None else None

    has_result = bool(mtimes.get("latest_result_mtime"))
    has_failure, failure_reason = (
        _detect_failed_result(job_root, status_data) if job_root else (False, None)
    )
    has_completed = _detect_completed_result(job_root, status_data) if job_root else False

    actual_status = declared
    reason = ""
    age_seconds = _job_age_seconds(now, created_at=created_at, started_at=started_at)

    if declared in PENDING_DECLARED_STATUSES and job_root is None and not is_process_alive:
        actual_status = declared
        reason = "任务尚未启动"
    elif has_completed and not is_process_alive:
        actual_status = "completed"
        reason = "结果文件或 runtime status 表明已完成"
    elif has_failure and not is_process_alive and declared not in FAILED_DECLARED_STATUSES:
        actual_status = "failed"
        reason = failure_reason or "runtime 或日志表明失败"
    elif is_process_alive and declared in RUNNING_DECLARED_STATUSES | PENDING_DECLARED_STATUSES:
        actual_status = "running"
        reason = "存在匹配 job/runtime 的活跃进程"
    elif declared in RUNNING_DECLARED_STATUSES and not is_process_alive:
        threshold = EVALUATION_RUNNING_STALE_THRESHOLD_SECONDS
        is_stale = stale_seconds is None or stale_seconds >= threshold
        runtime_missing = job_root is None or not job_root.is_dir()
        runtime_path_stale = bool(runtime_path) and runtime_missing
        age_seconds = _job_age_seconds(now, created_at=created_at, started_at=started_at)
        max_age_exceeded = (
            age_seconds is not None and age_seconds >= EVALUATION_RUNNING_MAX_AGE_SECONDS
        )

        if has_completed:
            actual_status = "completed"
            reason = "结果文件或 runtime status 表明已完成"
        elif is_process_alive:
            actual_status = "running"
            reason = "存在匹配 job/runtime 的活跃进程"
        elif max_age_exceeded:
            actual_status = "failed"
            reason = MAX_AGE_FAILURE_REASON
        elif runtime_path_stale or (runtime_missing and age_seconds is not None and age_seconds >= threshold):
            actual_status = "failed"
            reason = RUNTIME_MISSING_FAILURE_REASON if runtime_path_stale else STALE_RUNNING_FAILURE_REASON
        elif is_stale and not has_completed:
            actual_status = "failed"
            reason = STALE_RUNNING_FAILURE_REASON
        elif not is_stale:
            actual_status = "running"
            reason = "进程未匹配但 runtime 文件仍在近期更新"
    elif declared in COMPLETED_DECLARED_STATUSES:
        actual_status = "completed"
    elif declared in FAILED_DECLARED_STATUSES:
        actual_status = "failed"
        threshold = EVALUATION_RUNNING_STALE_THRESHOLD_SECONDS
        is_stale = stale_seconds is None or stale_seconds >= threshold
        stale_reason = STALE_RUNNING_FAILURE_REASON
        if not is_process_alive and is_stale and not has_completed:
            reason = stale_reason
        elif not is_process_alive:
            reason = failure_reason or stale_reason if is_stale else "评测进程已退出但未正常完成"
        else:
            reason = str(status_data.get("error") or status_data.get("message") or failure_reason or "")

    return {
        "jobId": candidate,
        "runtimePath": str(job_root) if job_root else runtime_path,
        "declaredStatus": declared,
        "actualStatus": actual_status,
        "isProcessAlive": is_process_alive,
        "matchedPids": matched_pids,
        "lastRuntimeUpdateAt": _mtime_to_iso(last_runtime_ts),
        "lastLogUpdateAt": _mtime_to_iso(mtimes.get("latest_log_mtime")),
        "staleSeconds": stale_seconds,
        "jobAgeSeconds": age_seconds,
        "hasResult": has_result,
        "hasFailure": has_failure or actual_status == "failed",
        "hasCompleted": has_completed,
        "reason": reason,
        "statusFile": str(status_path) if status_path else None,
        "createdAt": created_at,
        "startedAt": started_at,
    }


def _write_status_payload(status_path: Path, updates: dict[str, Any]) -> None:
    payload = _read_json(status_path)
    payload.update(updates)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def apply_evaluation_health_reconciliation(
    job_id: str,
    health: dict[str, Any],
    *,
    runtime_path: Optional[str] = None,
) -> bool:
    """将 health 判定写回 runtime status 文件；返回是否发生修正。"""
    actual = str(health.get("actualStatus") or "").strip().lower()
    declared = str(health.get("declaredStatus") or "").strip().lower()
    reason = str(health.get("reason") or "评测状态已修正")
    stale_reason = "评测进程已退出或失联，状态文件长时间未更新"
    should_update = actual in TERMINAL_ACTUAL_STATUSES and (
        actual != declared
        or (actual == "failed" and reason and reason != str(health.get("declaredReason") or ""))
    )
    if actual == declared == "failed" and stale_reason in reason:
        should_update = True
    if not should_update:
        return False

    job_root = resolve_evaluation_job_root(job_id, runtime_path or health.get("runtimePath"))
    if job_root is None:
        return False

    status_path = _resolve_primary_status_path(job_root)
    if status_path is None:
        if job_id.startswith("ct_eval_"):
            status_path = job_root / "live" / "status.json"
        else:
            status_path = job_root / "status.json"

    reason = str(health.get("reason") or "评测状态已修正")
    updates: dict[str, Any] = {
        "status": actual,
        "updatedAt": _utc_now_iso(),
    }
    if actual == "failed":
        updates["phase"] = "failed"
        updates["error"] = reason
        updates["message"] = reason
    elif actual == "completed":
        updates["phase"] = "completed"
        updates["message"] = reason or "评测已完成"
        updates["error"] = None

    _write_status_payload(status_path, updates)
    logger.info(
        "evaluation health reconcile job_id=%s declared=%s actual=%s reason=%s",
        job_id,
        declared,
        actual,
        reason,
    )
    return True


def reconcile_evaluation_runtime_health(
    job_id: str,
    runtime_path: Optional[str] = None,
    *,
    declared_status: Optional[str] = None,
    created_at: Optional[str] = None,
    started_at: Optional[str] = None,
    apply: bool = False,
) -> dict[str, Any]:
    health = inspect_evaluation_runtime_health(
        job_id,
        runtime_path,
        declared_status=declared_status,
        created_at=created_at,
        started_at=started_at,
    )
    if apply:
        health["applied"] = apply_evaluation_health_reconciliation(
            job_id,
            health,
            runtime_path=runtime_path,
        )
    else:
        health["applied"] = False
    return health


def reconcile_all_running_evaluation_jobs(*, limit: int = 200, apply: bool = True) -> list[str]:
    """对全部 running/evaluating 评测任务执行 health reconcile（不限当前页）。"""
    from app.core.database import SessionLocal
    from app.models.workspace_job import WorkspaceJob
    from app.services.training_job_sync_service import sync_eval_job_from_runtime

    reconciled: list[str] = []
    with SessionLocal() as db:
        rows = (
            db.query(WorkspaceJob)
            .filter(
                WorkspaceJob.job_type == "evaluation",
                WorkspaceJob.status != "deleted",
                WorkspaceJob.status.in_(sorted(EVALUATION_RECONCILE_STATUSES)),
            )
            .order_by(WorkspaceJob.updated_at.asc())
            .limit(limit)
            .all()
        )
        job_ids = [row.job_id for row in rows if row.job_id]

    for job_id in job_ids:
        try:
            sync_eval_job_from_runtime(job_id)
            reconciled.append(job_id)
        except Exception as exc:
            logger.warning("reconcile_all_running_evaluation_jobs failed job_id=%s: %s", job_id, exc)
    return reconciled
