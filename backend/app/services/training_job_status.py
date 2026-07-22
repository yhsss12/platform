"""训练任务状态归一化：结合 status.json、metrics、日志、进程与 checkpoint 推断完成态。"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Optional

COMPLETED_JOB_STATUSES = frozenset({"completed", "success", "succeeded", "finished", "done"})
FAILED_JOB_STATUSES = frozenset({"failed", "error", "canceled", "cancelled"})
IN_PROGRESS_JOB_STATUSES = frozenset({"running", "training", "queued", "pending", "starting"})

API_TRAINING_JOB_STATUSES = frozenset(
    {"queued", "starting", "running", "completed", "failed", "backend_unavailable"}
)


def normalize_api_training_status(raw: Optional[str]) -> str:
    """Map DB/runtime status tokens to public training API literals."""
    token = normalize_training_status_token(raw)
    if token in API_TRAINING_JOB_STATUSES:
        return token
    if token == "pending":
        return "queued"
    if token in {"canceled"}:
        return "failed"
    return "backend_unavailable"


COMPLETION_LOG_MARKERS = (
    "training completed",
    "finished training",
    "saved final model",
    "saved checkpoint:",
    "训练完成",
    "pi0 lerobot smoke completed",
    "pi0 lerobot 平台训练 smoke 已完成",
)


def normalize_training_status_token(raw: Optional[str]) -> str:
    value = (raw or "").strip().lower()
    if not value:
        return "unknown"
    aliases = {
        "success": "completed",
        "succeeded": "completed",
        "finished": "completed",
        "done": "completed",
        "training": "running",
        "queued": "pending",
        "error": "failed",
        "cancelled": "canceled",
    }
    return aliases.get(value, value)


def _progress_fraction(raw: Any) -> float:
    if raw is None:
        return 0.0
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.0
    if value > 1.0:
        return min(1.0, value / 100.0)
    return max(0.0, min(1.0, value))


def compute_training_progress_fraction(
    epoch: int,
    total_epochs: int,
    status: Optional[str],
) -> float:
    total = max(0, int(total_epochs))
    current = max(0, int(epoch))
    token = normalize_training_status_token(status)
    if token in COMPLETED_JOB_STATUSES and total > 0 and current >= total:
        return 1.0
    if total <= 0:
        return 0.0
    return min(0.99, current / total)


def _is_remote_ssh_execution(status_data: dict[str, Any], train_job_dir: Path | None = None) -> bool:
    mode = str(status_data.get("executionMode") or "").lower()
    if mode == "remote_ssh":
        return True
    if train_job_dir is not None:
        import json

        cfg_path = train_job_dir / "config" / "train_config.json"
        if cfg_path.is_file():
            try:
                train_config = json.loads(cfg_path.read_text(encoding="utf-8"))
                if str(train_config.get("executionMode") or "").lower() == "remote_ssh":
                    return True
            except (OSError, json.JSONDecodeError):
                pass
    return False


def _read_status_json(train_job_dir: Path) -> dict[str, Any]:
    import json

    path = train_job_dir / "status.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _is_pi0_terminal_completed(status_data: dict[str, Any]) -> bool:
    """pi0 LeRobot smoke 已写 completed + checkpoint 时不应被 pgrep 误匹配降级。"""
    if str(status_data.get("modelType") or "").lower() != "pi0":
        return False
    if normalize_training_status_token(str(status_data.get("status") or "")) not in COMPLETED_JOB_STATUSES:
        return False
    if not status_data.get("checkpointExists"):
        return False
    try:
        return float(status_data.get("progress") or 0) >= 1.0
    except (TypeError, ValueError):
        return False


def is_training_process_active(
    train_job_id: str,
    *,
    train_job_dir: Optional[Path] = None,
) -> bool:
    """检测 train_dp.py / 训练子进程是否仍在运行（孤儿进程场景）。"""
    import os

    job_id = (train_job_id or "").strip()
    if job_id:
        try:
            from app.services.training_service import _RUNNING_PROCS, _RUNNING_THREADS

            proc = _RUNNING_PROCS.get(job_id)
            if proc is not None and proc.poll() is None:
                return True
            thread = _RUNNING_THREADS.get(job_id)
            if thread is not None and thread.is_alive():
                return True
        except Exception:
            pass

    if train_job_dir is not None:
        status_data = _read_status_json(train_job_dir)
        if _is_remote_ssh_execution(status_data, train_job_dir):
            try:
                from app.services.training_service import _RUNNING_PROCS

                if train_job_id in _RUNNING_PROCS:
                    return True
            except Exception:
                pass
            return False

    current_pid = os.getpid()
    markers: list[str] = []
    if job_id:
        markers.append(job_id)
    if train_job_dir is not None:
        markers.append(str(train_job_dir.resolve()))
        markers.append(str((train_job_dir / "checkpoints").resolve()))
    seen: set[str] = set()
    for marker in markers:
        if not marker or marker in seen:
            continue
        seen.add(marker)
        try:
            proc = subprocess.run(
                ["pgrep", "-f", marker],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if proc.returncode == 0 and proc.stdout.strip():
            for pid_raw in proc.stdout.strip().split():
                try:
                    if int(pid_raw) != current_pid:
                        return True
                except ValueError:
                    continue
    return False


REMOTE_TRAINING_STARTUP_TIMEOUT_SEC = 1800


def training_activity_detected(
    train_job_dir: Path,
    status_data: Optional[dict[str, Any]] = None,
) -> bool:
    """训练是否已产生 epoch/loss 或有效训练日志（区分「已启动进程」与「真正开训」）。"""
    snapshot = status_data if status_data is not None else _read_status_json(train_job_dir)
    if int(snapshot.get("epoch") or 0) > 0:
        return True
    if snapshot.get("loss") is not None:
        return True

    total_epochs = int(snapshot.get("totalEpochs") or 0)
    from app.services.training_metrics import parse_training_logs

    parsed_epoch, parsed_loss = parse_training_logs(train_job_dir, total_epochs if total_epochs > 0 else 1)
    if parsed_epoch > 0 or parsed_loss is not None:
        return True

    log_text = _collect_training_logs_text(train_job_dir)
    if not log_text.strip():
        return False

    from app.services.training_service import (
        EPOCH_LOG_PATTERN,
        JSON_LOSS_LOG_PATTERN,
        LOSS_LOG_PATTERN,
        TRAIN_EPOCH_LOG_PATTERN,
    )

    for line in log_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.lower().startswith("command:"):
            continue
        if TRAIN_EPOCH_LOG_PATTERN.search(line) or EPOCH_LOG_PATTERN.search(line):
            return True
        if JSON_LOSS_LOG_PATTERN.search(line) or LOSS_LOG_PATTERN.search(line):
            return True
    return False


def resolve_remote_training_public_status(
    train_job_dir: Path,
    status_data: dict[str, Any],
    *,
    process_active: bool,
) -> str:
    """remote_ssh 任务对外状态：无训练产出前保持 starting，有产出后才 running。"""
    raw = normalize_training_status_token(str(status_data.get("status") or ""))
    if raw in FAILED_JOB_STATUSES:
        return raw
    if raw in COMPLETED_JOB_STATUSES:
        return "completed"
    if raw in {"queued", "pending"}:
        return "queued"
    if training_activity_detected(train_job_dir, status_data):
        return "running"
    if process_active or raw in {"running", "starting", "training"}:
        return "starting"
    return raw if raw in API_TRAINING_JOB_STATUSES else "starting"


def _collect_training_logs_text(train_job_dir: Path) -> str:
    from app.services.training_metrics import collect_training_log_paths

    chunks: list[str] = []
    for path in collect_training_log_paths(train_job_dir):
        try:
            chunks.append(path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
    return "\n".join(chunks)


def log_indicates_training_completed(log_text: str) -> bool:
    """仅依据训练输出正文判断完成，忽略 command 行中的 init-checkpoint 路径。"""
    body_lines: list[str] = []
    for line in (log_text or "").splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("command:"):
            continue
        body_lines.append(line)
    lower = "\n".join(body_lines).lower()
    return any(marker in lower for marker in COMPLETION_LOG_MARKERS)


def _truth_epoch_from_runtime(train_job_dir: Path, status_data: dict[str, Any]) -> int:
    from app.services.training_metrics import parse_training_logs, read_metrics_history

    history = read_metrics_history(train_job_dir)
    history_max_epoch = max(int(row.get("epoch") or 0) for row in history) if history else 0
    total_epochs = int(status_data.get("totalEpochs") or 0)
    parsed_epoch, _ = parse_training_logs(
        train_job_dir,
        total_epochs if total_epochs > 0 else max(1, int(status_data.get("epoch") or 1)),
    )
    return max(history_max_epoch, parsed_epoch, 0)


def infer_training_job_completed(
    status_data: dict[str, Any],
    *,
    train_job_dir: Path | None = None,
    normalized_metrics: dict[str, Any] | None = None,
    process_active: Optional[bool] = None,
) -> bool:
    raw = normalize_training_status_token(str(status_data.get("status") or ""))
    if raw in FAILED_JOB_STATUSES:
        return False

    metrics = normalized_metrics
    if metrics is None and train_job_dir is not None:
        from app.services.training_metrics import normalized_training_metrics

        metrics = normalized_training_metrics(train_job_dir, status_data)

    current_epoch = int((metrics or {}).get("epoch") or status_data.get("epoch") or 0)
    total_epochs = int((metrics or {}).get("totalEpochs") or status_data.get("totalEpochs") or 0)

    if train_job_dir is not None:
        truth_epoch = _truth_epoch_from_runtime(train_job_dir, status_data)
        if truth_epoch > 0:
            current_epoch = truth_epoch

    if process_active is None and train_job_dir is not None:
        process_active = is_training_process_active(
            str(status_data.get("trainJobId") or train_job_dir.name),
            train_job_dir=train_job_dir,
        )
    if process_active:
        return False

    epoch_complete = total_epochs > 0 and current_epoch >= total_epochs
    if epoch_complete:
        return True

    if train_job_dir is None:
        return False

    logs_complete = log_indicates_training_completed(_collect_training_logs_text(train_job_dir))
    return logs_complete


def resolve_canonical_training_job_status(
    train_job_id: str,
    train_job_dir: Path,
    status_data: dict[str, Any],
) -> dict[str, Any]:
    """列表 / 详情 / metrics 共用的 canonical 状态（不写盘）。"""
    from app.services.training_metrics import normalized_training_metrics

    enriched = dict(status_data)
    truth_epoch = _truth_epoch_from_runtime(train_job_dir, status_data)
    status_epoch = int(status_data.get("epoch") or 0)
    effective_epoch = truth_epoch if truth_epoch > 0 else status_epoch
    total_epochs = int(status_data.get("totalEpochs") or 0)
    process_active = is_training_process_active(train_job_id, train_job_dir=train_job_dir)
    pi0_terminal = _is_pi0_terminal_completed(enriched)
    if pi0_terminal:
        process_active = False
    raw = normalize_training_status_token(str(enriched.get("status") or ""))

    if raw in FAILED_JOB_STATUSES:
        enriched["status"] = raw
        enriched["epoch"] = effective_epoch
        enriched["totalEpochs"] = total_epochs
        enriched["progress"] = compute_training_progress_fraction(
            effective_epoch,
            total_epochs,
            raw,
        )
        return enriched

    if raw in COMPLETED_JOB_STATUSES and total_epochs > 0 and effective_epoch >= total_epochs and not process_active:
        enriched["status"] = "completed"
        enriched["epoch"] = max(effective_epoch, total_epochs)
        enriched["totalEpochs"] = total_epochs
        enriched["progress"] = 1.0
        normalized = normalized_training_metrics(train_job_dir, enriched)
        if normalized.get("loss") is not None:
            enriched["loss"] = normalized.get("loss")
        return enriched

    if process_active or (total_epochs > 0 and effective_epoch > 0 and effective_epoch < total_epochs):
        activity = training_activity_detected(train_job_dir, status_data) or effective_epoch > 0
        if activity:
            enriched["status"] = "running"
            enriched["epoch"] = effective_epoch
            enriched["totalEpochs"] = total_epochs
            enriched["progress"] = compute_training_progress_fraction(effective_epoch, total_epochs, "running")
            if process_active and not enriched.get("message"):
                backend = enriched.get("trainingBackendResolved") or enriched.get("trainingBackend") or "training"
                enriched["message"] = f"训练进行中（{backend}）"
        else:
            enriched["status"] = "starting"
            enriched["epoch"] = effective_epoch
            enriched["totalEpochs"] = total_epochs
            enriched["progress"] = 0.0
            if not enriched.get("message"):
                if _is_remote_ssh_execution(status_data, train_job_dir):
                    enriched["message"] = "远端训练已调度，等待首批训练日志"
                else:
                    enriched["message"] = "训练进程已启动，等待首批日志"
        normalized = normalized_training_metrics(train_job_dir, enriched)
        if normalized.get("loss") is not None:
            enriched["loss"] = normalized.get("loss")
        return enriched

    if raw in COMPLETED_JOB_STATUSES and total_epochs > 0 and effective_epoch < total_epochs:
        activity = training_activity_detected(train_job_dir, status_data) or effective_epoch > 0
        enriched["status"] = "running" if activity else "starting"
        enriched["epoch"] = effective_epoch
        enriched["totalEpochs"] = total_epochs
        enriched["progress"] = (
            compute_training_progress_fraction(effective_epoch, total_epochs, "running")
            if activity
            else 0.0
        )
        normalized = normalized_training_metrics(train_job_dir, enriched)
        if normalized.get("loss") is not None:
            enriched["loss"] = normalized.get("loss")
        return enriched

    normalized = normalized_training_metrics(train_job_dir, enriched)
    current_epoch = effective_epoch if effective_epoch > 0 else int(normalized.get("epoch") or enriched.get("epoch") or 0)
    enriched["epoch"] = current_epoch
    enriched["totalEpochs"] = total_epochs or int(normalized.get("totalEpochs") or 0)

    if infer_training_job_completed(
        enriched,
        train_job_dir=train_job_dir,
        normalized_metrics=normalized,
        process_active=process_active,
    ):
        enriched["status"] = "completed"
        if enriched["totalEpochs"] > 0:
            enriched["epoch"] = max(current_epoch, enriched["totalEpochs"])
        enriched["progress"] = 1.0
        if not enriched.get("message"):
            enriched["message"] = "训练已完成"
    else:
        if raw in IN_PROGRESS_JOB_STATUSES or raw == "unknown":
            activity = training_activity_detected(train_job_dir, status_data) or current_epoch > 0
            if raw in {"queued", "pending"}:
                enriched["status"] = "queued"
            elif activity:
                enriched["status"] = "running"
            elif raw in {"running", "training", "starting"}:
                enriched["status"] = "starting"
            else:
                enriched["status"] = raw
        elif raw not in COMPLETED_JOB_STATUSES:
            enriched["status"] = raw
        enriched["progress"] = compute_training_progress_fraction(
            int(enriched.get("epoch") or 0),
            int(enriched.get("totalEpochs") or 0),
            enriched.get("status"),
        )

    if normalized.get("loss") is not None:
        enriched["loss"] = normalized.get("loss")

    return enriched


def enrich_training_job_status(
    train_job_dir: Path,
    status_data: dict[str, Any],
) -> dict[str, Any]:
    train_job_id = str(status_data.get("trainJobId") or train_job_dir.name)
    return resolve_canonical_training_job_status(train_job_id, train_job_dir, status_data)


def persist_training_completion(
    train_job_id: str,
    train_job_dir: Path,
    status_data: dict[str, Any],
) -> None:
    """将推断的完成态写回 status.json，并登记 final checkpoint。"""
    from app.services.checkpoint_registry import register_checkpoint_assets
    from app.services.training_service import _read_json, _update_status, sync_workspace_job_from_runtime

    train_config = _read_json(train_job_dir / "config" / "train_config.json")
    manifest = _read_json(train_job_dir / "artifacts" / "dataset_manifest.json")
    resolved_backend = str(
        train_config.get("trainingBackend")
        or status_data.get("trainingBackendResolved")
        or status_data.get("trainingBackend")
        or "robomimic_bc"
    )
    if resolved_backend == "diffusion_policy":
        framework_label = "Diffusion Policy"
        model_type = "diffusion_policy"
    elif resolved_backend == "torch_bc":
        framework_label = "BC (PyTorch)"
        model_type = "bc"
    elif resolved_backend == "act":
        framework_label = "ACT"
        model_type = "act"
    elif resolved_backend == "pi0":
        framework_label = "pi0"
        model_type = "pi0"
    else:
        framework_label = "Robomimic BC"
        model_type = "bc"

    assets = register_checkpoint_assets(
        train_job_dir=train_job_dir,
        train_job_id=train_job_id,
        manifest=manifest,
        train_config=train_config,
        status=status_data,
        resolved_backend=resolved_backend,
        framework_label=framework_label,
        model_type=model_type,
        register_final=True,
    )
    primary = next((item for item in assets if item.get("checkpointKind") == "final"), assets[-1] if assets else None)

    patch: dict[str, Any] = {
        "status": "completed",
        "epoch": status_data.get("epoch"),
        "totalEpochs": status_data.get("totalEpochs"),
        "progress": 1.0,
        "processPid": None,
        "message": status_data.get("message") or "训练已完成",
    }
    if status_data.get("loss") is not None:
        patch["loss"] = status_data.get("loss")
    if primary is not None:
        patch["checkpointExists"] = True
        patch["checkpointPath"] = str(primary.get("checkpointPath") or "")
        patch["modelAssetId"] = primary.get("modelAssetId")

    _update_status(train_job_dir, patch)
    from app.services.training_job_sync_service import finalize_training_job_sync

    finalize_training_job_sync(train_job_id)


def enrich_and_persist_training_job_status(
    train_job_id: str,
    train_job_dir: Path,
    status_data: dict[str, Any],
) -> dict[str, Any]:
    original_status = normalize_training_status_token(str(status_data.get("status") or ""))
    enriched = enrich_training_job_status(train_job_dir, status_data)
    resolved_status = normalize_training_status_token(str(enriched.get("status") or ""))

    if resolved_status == "completed" and original_status not in COMPLETED_JOB_STATUSES:
        terminal_completed = (
            bool(status_data.get("checkpointExists"))
            and float(status_data.get("progress") or 0) >= 1.0
            and str(status_data.get("datasetFormat") or "").lower() == "lerobot"
            and str(status_data.get("modelType") or "").lower() == "pi0"
        )
        if terminal_completed or (
            not is_training_process_active(train_job_id, train_job_dir=train_job_dir)
            and not _is_remote_ssh_execution(status_data, train_job_dir)
        ):
            persist_training_completion(train_job_id, train_job_dir, enriched)
            from app.services.training_service import _read_json

            return _read_json(train_job_dir / "status.json") or enriched
        if is_training_process_active(train_job_id, train_job_dir=train_job_dir) and not _is_remote_ssh_execution(
            status_data, train_job_dir
        ):
            from app.services.training_service import _update_status

            patch = {
                "status": "running",
                "epoch": enriched.get("epoch"),
                "totalEpochs": enriched.get("totalEpochs"),
                "progress": enriched.get("progress"),
                "message": enriched.get("message") or "训练进行中",
            }
            if enriched.get("loss") is not None:
                patch["loss"] = enriched.get("loss")
            _update_status(train_job_dir, patch)
            from app.services.training_service import _read_json

            return _read_json(train_job_dir / "status.json") or enriched
        persist_training_completion(train_job_id, train_job_dir, enriched)
        from app.services.training_service import _read_json

        return _read_json(train_job_dir / "status.json") or enriched

    if resolved_status == "running" and original_status in COMPLETED_JOB_STATUSES:
        pi0_terminal = (
            str(status_data.get("modelType") or "").lower() == "pi0"
            and bool(status_data.get("checkpointExists"))
            and float(status_data.get("progress") or 0) >= 1.0
        )
        if pi0_terminal or not is_training_process_active(train_job_id, train_job_dir=train_job_dir):
            enriched["status"] = "completed"
            enriched["progress"] = 1.0
            if enriched.get("totalEpochs"):
                enriched["epoch"] = max(
                    int(enriched.get("epoch") or 0),
                    int(enriched.get("totalEpochs") or 0),
                )
            return enriched
        from app.services.training_service import _update_status

        patch: dict[str, Any] = {
            "status": "running",
            "epoch": enriched.get("epoch"),
            "totalEpochs": enriched.get("totalEpochs"),
            "progress": enriched.get("progress"),
            "message": enriched.get("message") or "训练进行中",
        }
        if enriched.get("loss") is not None:
            patch["loss"] = enriched.get("loss")
        _update_status(train_job_dir, patch)
        from app.services.training_service import _read_json

        return _read_json(train_job_dir / "status.json") or enriched

    if resolved_status == "running" and original_status in IN_PROGRESS_JOB_STATUSES:
        from app.services.training_service import _update_status

        patch = {
            "epoch": enriched.get("epoch"),
            "totalEpochs": enriched.get("totalEpochs"),
            "progress": enriched.get("progress"),
            "message": enriched.get("message"),
        }
        if enriched.get("loss") is not None:
            patch["loss"] = enriched.get("loss")
        _update_status(train_job_dir, patch)
        from app.services.training_service import _read_json

        return _read_json(train_job_dir / "status.json") or enriched

    return enriched
