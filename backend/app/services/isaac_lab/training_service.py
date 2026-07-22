"""Isaac Lab Stack Cube Robomimic BC 训练（subprocess + isaaclab.sh，不在 FastAPI 内 import isaaclab）。"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from app.core.config import settings
from app.services.isaac_lab.cli_runner import IsaacLabCliRunner, IsaacLabCliRunResult
from app.services.isaac_lab.paths import PROJECT_ROOT, resolve_isaaclab_root

logger = logging.getLogger(__name__)

ISAAC_ROBOMIMIC_TRAIN_SCRIPT = "scripts/imitation_learning/robomimic/train.py"
ISAAC_STACK_TASK_ENV = "Isaac-Stack-Cube-Franka-IK-Rel-v0"
ISAAC_STACK_DATASET_ENV = "Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0"
ISAAC_STACK_OBS_KEYS = ["eef_pos", "eef_quat", "gripper_pos", "object"]
ISAAC_STACK_ACTION_DIM = 7
ISAAC_STACK_TEMPLATE_ID = "isaac_block_stacking"
ISAAC_STACK_TASK_TYPE = "isaac_block_stacking"

MODEL_EPOCH_PATTERN = re.compile(r"model_epoch_(\d+)\.pth$", re.IGNORECASE)


def _now_label() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def probe_isaac_robomimic_training_capability() -> dict[str, Any]:
    """探测 Isaac Robomimic BC 训练是否可用。"""
    root = resolve_isaaclab_root()
    runner = IsaacLabCliRunner.from_settings()
    evidence: list[str] = []
    issues: list[str] = []

    if root is not None:
        evidence.append(str(root))
        train_script = root / ISAAC_ROBOMIMIC_TRAIN_SCRIPT
        if train_script.is_file():
            evidence.append(str(train_script))
        else:
            issues.append(f"train.py not found: {train_script}")

    if runner.is_ready():
        evidence.append(str(runner.sh_path))
    else:
        issues.append("isaaclab.sh is not configured")

    if not getattr(settings, "ISAACLAB_RUNTIME_ENABLED", False):
        issues.append("ISAACLAB_RUNTIME_ENABLED is false")

    ready = runner.is_ready() and not issues
    return {
        "ready": ready,
        "evidence": evidence,
        "issues": issues,
    }


def is_isaac_block_stacking_manifest(manifest: dict[str, Any]) -> bool:
    task_type = str(manifest.get("taskType") or "").strip()
    if task_type in {ISAAC_STACK_TASK_TYPE, "block_stacking"}:
        return True
    template_id = str(manifest.get("taskTemplateId") or "").strip()
    if template_id == ISAAC_STACK_TEMPLATE_ID:
        return True
    simulator = str(manifest.get("simulatorBackend") or manifest.get("backend") or "").strip()
    if simulator == "isaac_lab":
        return True
    source_job_id = str(manifest.get("sourceJobId") or "").strip()
    return source_job_id.startswith("isaac_gen_") or source_job_id.startswith("isaac_import_")


def build_training_experiment_name(train_job_id: str, task_name: Optional[str] = None) -> str:
    if task_name and task_name.strip():
        slug = re.sub(r"[^\w\-]+", "_", task_name.strip())[:48]
        return slug or f"isaac_stack_{train_job_id[-8:]}"
    return f"isaac_stack_{train_job_id[-12:]}"


def build_train_cli_args(
    *,
    dataset_hdf5: Path,
    train_job_id: str,
    epochs: int,
    experiment_name: str,
) -> list[str]:
    log_dir = f"platform_train/{train_job_id}"
    return [
        "--task",
        ISAAC_STACK_TASK_ENV,
        "--algo",
        "bc",
        "--dataset",
        str(dataset_hdf5.resolve()),
        "--name",
        experiment_name,
        "--log_dir",
        log_dir,
        "--epochs",
        str(max(1, int(epochs))),
    ]


def resolve_isaac_train_log_root(train_job_id: str, isaaclab_root: Path) -> Path:
    return isaaclab_root / "logs" / "platform_train" / train_job_id / ISAAC_STACK_TASK_ENV


def find_latest_model_epoch_checkpoint(search_root: Path) -> Optional[Path]:
    if not search_root.is_dir():
        return None
    candidates: list[tuple[int, Path]] = []
    for path in search_root.rglob("model_epoch_*.pth"):
        if not path.is_file() or path.stat().st_size <= 0:
            continue
        match = MODEL_EPOCH_PATTERN.search(path.name)
        epoch = int(match.group(1)) if match else -1
        candidates.append((epoch, path))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1].stat().st_mtime))
    return candidates[-1][1]


def sync_isaac_checkpoints_to_job(
    isaac_log_root: Path,
    checkpoints_dir: Path,
    *,
    save_final: bool = True,
) -> list[Path]:
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    if not isaac_log_root.is_dir():
        return copied
    for path in isaac_log_root.rglob("model_epoch_*.pth"):
        if not path.is_file() or path.stat().st_size <= 0:
            continue
        target = checkpoints_dir / path.name
        try:
            if not target.is_file() or target.stat().st_mtime < path.stat().st_mtime:
                shutil.copy2(path, target)
            copied.append(target)
        except OSError:
            continue
    latest = find_latest_model_epoch_checkpoint(isaac_log_root)
    if save_final and latest is not None:
        final_checkpoint = checkpoints_dir / "model_final.pth"
        try:
            shutil.copy2(latest, final_checkpoint)
            copied.append(final_checkpoint)
        except OSError:
            pass
    return copied


def recover_completed_isaac_training(train_job_id: str, train_job_dir: Path) -> bool:
    """Finalize an Isaac job whose API worker stopped after the trainer exited."""
    status_path = train_job_dir / "status.json"
    status_data = _read_json(status_path)
    if str(status_data.get("status") or "").lower() not in {"running", "starting"}:
        return False
    if str(status_data.get("trainingBackendResolved") or status_data.get("trainingBackend") or "") != "isaac_robomimic_bc":
        return False

    root = resolve_isaaclab_root()
    if root is None:
        return False
    isaac_log_root = resolve_isaac_train_log_root(train_job_id, root)
    source_checkpoint = find_latest_model_epoch_checkpoint(isaac_log_root)
    train_log = train_job_dir / "logs" / "train.log"
    if source_checkpoint is None or not source_checkpoint.is_file() or not train_log.is_file():
        return False
    if "finished run successfully" not in train_log.read_text(encoding="utf-8", errors="replace"):
        return False

    train_config = _read_json(train_job_dir / "config" / "train_config.json")
    manifest = _read_json(train_job_dir / "artifacts" / "dataset_manifest.json")
    total_epochs = int(status_data.get("totalEpochs") or train_config.get("epochs") or 0)
    epoch, loss = _parse_train_progress_from_logs(train_log, train_job_dir / "logs" / "stderr.log", total_epochs)
    checkpoints_dir = train_job_dir / "checkpoints"
    sync_isaac_checkpoints_to_job(isaac_log_root, checkpoints_dir, save_final=True)

    enriched_manifest = dict(manifest)
    enriched_manifest.setdefault("taskType", ISAAC_STACK_TASK_TYPE)
    enriched_manifest.setdefault("taskTemplateId", ISAAC_STACK_TEMPLATE_ID)
    enriched_manifest.setdefault("obsKeys", ISAAC_STACK_OBS_KEYS)
    enriched_manifest.setdefault("actionDim", ISAAC_STACK_ACTION_DIM)
    enriched_manifest.setdefault("simulatorBackend", "isaac_lab")

    from app.services.checkpoint_registry import register_checkpoint_assets

    assets = register_checkpoint_assets(
        train_job_dir=train_job_dir,
        train_job_id=train_job_id,
        manifest=enriched_manifest,
        train_config=train_config,
        status=status_data,
        resolved_backend="isaac_robomimic_bc",
        framework_label="Robomimic BC",
        model_type="bc",
    )
    primary = next((item for item in assets if item.get("checkpointKind") == "final"), assets[-1] if assets else None)
    if primary is None:
        return False

    status_data.update(
        {
            "status": "completed",
            "epoch": max(epoch, total_epochs),
            "totalEpochs": total_epochs,
            "progress": 1.0,
            "checkpointExists": True,
            "checkpointPath": str(primary.get("checkpointPath") or checkpoints_dir / "model_final.pth"),
            "modelAssetId": primary.get("modelAssetId"),
            "message": f"Isaac Robomimic BC 训练完成，已恢复登记 {len(assets)} 个模型资产",
            "loss": loss,
            "processPid": None,
            "updatedAt": _now_label(),
        }
    )
    _write_json(status_path, status_data)
    return True


def _parse_train_progress_from_logs(
    stdout_path: Path, stderr_path: Path, total_epochs: int
) -> tuple[int, Optional[float]]:
    """从 Isaac robomimic 训练日志解析 epoch 与 loss。"""
    epoch = 0
    loss: Optional[float] = None
    json_loss_pattern = re.compile(r'"Loss"\s*:\s*([-+0-9.eE]+)')
    train_epoch_pattern = re.compile(r"Train\s+Epoch\s+(\d+)", re.IGNORECASE)

    for path in (stdout_path, stderr_path):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            train_match = train_epoch_pattern.search(line)
            if train_match:
                try:
                    epoch = max(epoch, int(train_match.group(1)))
                except ValueError:
                    pass
            loss_match = json_loss_pattern.search(line)
            if loss_match:
                try:
                    loss = float(loss_match.group(1))
                except ValueError:
                    pass
    return min(epoch, total_epochs), loss


def _parse_train_epoch_from_logs(stdout_path: Path, stderr_path: Path, total_epochs: int) -> int:
    epoch, _ = _parse_train_progress_from_logs(stdout_path, stderr_path, total_epochs)
    return epoch


def _mirror_log_to_stdout(train_log: Path, stdout_path: Path) -> None:
    """保留 stdout.log 供旧逻辑/调试读取。"""
    try:
        if train_log.is_file():
            stdout_path.write_text(train_log.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
    except OSError:
        pass


def execute_isaac_robomimic_training(
    *,
    train_job_id: str,
    train_job_dir: Path,
    manifest: dict[str, Any],
    train_config: dict[str, Any],
    hdf5_path: Path,
    update_status,
    register_model_manifest,
    sync_workspace_job,
    register_running_proc=None,
    unregister_running_proc=None,
) -> None:
    """执行 Isaac Robomimic BC 训练并登记产物。"""
    logs_dir = train_job_dir / "logs"
    train_log = logs_dir / "train.log"
    stdout_path = logs_dir / "stdout.log"
    stderr_path = logs_dir / "stderr.log"
    checkpoints_dir = train_job_dir / "checkpoints"
    artifacts_dir = train_job_dir / "artifacts"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    capability = probe_isaac_robomimic_training_capability()
    if not capability.get("ready"):
        message = "Isaac Robomimic BC 训练未就绪：" + "; ".join(capability.get("issues") or [])
        update_status(train_job_dir, {"status": "backend_unavailable", "message": message, "progress": 0.0})
        train_log.parent.mkdir(parents=True, exist_ok=True)
        train_log.write_text(message + "\n", encoding="utf-8")
        return

    runner = IsaacLabCliRunner.from_settings()
    root = runner.root
    if root is None:
        update_status(
            train_job_dir,
            {"status": "backend_unavailable", "message": "ISAACLAB_ROOT 未配置", "progress": 0.0},
        )
        return

    total_epochs = int(train_config.get("epochs") or 2)
    experiment_name = build_training_experiment_name(
        train_job_id,
        str(train_config.get("taskName") or manifest.get("datasetName") or ""),
    )
    cli_args = build_train_cli_args(
        dataset_hdf5=hdf5_path,
        train_job_id=train_job_id,
        epochs=total_epochs,
        experiment_name=experiment_name,
    )
    command = runner.build_command(ISAAC_ROBOMIMIC_TRAIN_SCRIPT, *cli_args)

    _write_json(
        train_job_dir / "metadata" / "request.json",
        {
            "trainJobId": train_job_id,
            "backend": "isaac_robomimic_bc",
            "command": command,
            "datasetHdf5": str(hdf5_path),
            "taskEnv": ISAAC_STACK_TASK_ENV,
            "datasetEnv": ISAAC_STACK_DATASET_ENV,
            "epochs": total_epochs,
            "experimentName": experiment_name,
            "createdAt": _now_label(),
        },
    )

    update_status(
        train_job_dir,
        {
            "status": "running",
            "message": "Isaac Robomimic BC 训练进行中",
            "trainingBackendResolved": "isaac_robomimic_bc",
            "command": command,
            "totalEpochs": total_epochs,
            "epoch": 0,
            "progress": 0.0,
        },
    )

    timed_out = False
    returncode = 1
    proc: Optional[subprocess.Popen] = None

    try:
        proc = runner.popen_to_log(ISAAC_ROBOMIMIC_TRAIN_SCRIPT, *cli_args, log_path=train_log)
    except OSError as exc:
        message = f"Isaac 训练进程启动失败: {exc}"
        update_status(train_job_dir, {"status": "failed", "message": message, "progress": 0.0})
        train_log.write_text(message + "\n", encoding="utf-8")
        sync_workspace_job(train_job_id)
        return

    if register_running_proc is not None:
        register_running_proc(proc)
    update_status(train_job_dir, {"processPid": proc.pid, "startedAt": _now_label()})
    sync_workspace_job(train_job_id)

    timeout = int(getattr(settings, "ISAACLAB_TRAIN_TIMEOUT", 7200) or 7200)
    started_at = time.monotonic()
    last_sync = 0.0

    try:
        while proc.poll() is None:
            elapsed = time.monotonic() - started_at
            if elapsed > timeout:
                timed_out = True
                proc.kill()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    pass
                returncode = -1
                break

            _mirror_log_to_stdout(train_log, stdout_path)
            epoch, loss = _parse_train_progress_from_logs(train_log, stderr_path, total_epochs)
            if epoch > 0:
                append_metrics_point(train_job_dir, epoch=epoch, loss=loss)
            progress = min(0.99, epoch / total_epochs) if total_epochs > 0 else 0.0
            update_status(
                train_job_dir,
                {
                    "status": "running",
                    "epoch": epoch,
                    "totalEpochs": total_epochs,
                    "progress": progress,
                    "loss": loss,
                    "message": "Isaac Robomimic BC 训练进行中",
                },
            )
            now = time.monotonic()
            if now - last_sync >= 8.0:
                sync_workspace_job(train_job_id)
                last_sync = now
            time.sleep(2)

        if not timed_out:
            returncode = proc.returncode if proc.returncode is not None else 1
    finally:
        if unregister_running_proc is not None:
            unregister_running_proc()
        _mirror_log_to_stdout(train_log, stdout_path)

    result = IsaacLabCliRunResult(
        returncode=returncode,
        command=command,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        timed_out=timed_out,
    )

    from app.services.checkpoint_registry import parse_save_policy, register_checkpoint_assets
    from app.services.training_metrics import append_metrics_point

    save_policy = parse_save_policy(train_config)
    epoch, loss = _parse_train_progress_from_logs(train_log, stderr_path, total_epochs)
    if epoch > 0:
        append_metrics_point(train_job_dir, epoch=epoch, loss=loss)
    isaac_log_root = resolve_isaac_train_log_root(train_job_id, root)
    sync_isaac_checkpoints_to_job(
        isaac_log_root,
        checkpoints_dir,
        save_final=bool(save_policy.get("saveFinal", True)),
    )
    source_checkpoint = find_latest_model_epoch_checkpoint(isaac_log_root)

    training_manifest = {
        "trainJobId": train_job_id,
        "backend": "isaac_robomimic_bc",
        "taskEnv": ISAAC_STACK_TASK_ENV,
        "datasetEnv": ISAAC_STACK_DATASET_ENV,
        "experimentName": experiment_name,
        "isaacLogRoot": str(isaac_log_root),
        "sourceCheckpoint": str(source_checkpoint) if source_checkpoint else None,
        "returncode": result.returncode,
        "timedOut": result.timed_out,
        "finishedAt": _now_label(),
    }
    _write_json(artifacts_dir / "training_manifest.json", training_manifest)

    if result.timed_out:
        update_status(
            train_job_dir,
            {
                "status": "failed",
                "message": f"Isaac 训练超时（{timeout}s）",
                "progress": min(0.99, epoch / total_epochs) if total_epochs else 0.0,
                "epoch": epoch,
                "loss": loss,
                "checkpointExists": False,
                "processPid": None,
            },
        )
        sync_workspace_job(train_job_id)
        return

    if result.returncode != 0:
        update_status(
            train_job_dir,
            {
                "status": "failed",
                "message": f"Isaac 训练进程失败（exit={result.returncode}）",
                "progress": min(1.0, epoch / total_epochs) if total_epochs else 0.0,
                "epoch": epoch,
                "loss": loss,
                "checkpointExists": False,
                "processPid": None,
            },
        )
        sync_workspace_job(train_job_id)
        return

    if source_checkpoint is None or not source_checkpoint.is_file():
        update_status(
            train_job_dir,
            {
                "status": "failed",
                "message": "训练结束但未找到 model_epoch_*.pth，无法登记模型",
                "progress": min(1.0, epoch / total_epochs) if total_epochs else 0.0,
                "epoch": epoch,
                "loss": loss,
                "checkpointExists": False,
                "processPid": None,
            },
        )
        sync_workspace_job(train_job_id)
        return

    enriched_manifest = dict(manifest)
    enriched_manifest.setdefault("taskType", ISAAC_STACK_TASK_TYPE)
    enriched_manifest.setdefault("taskTemplateId", ISAAC_STACK_TEMPLATE_ID)
    enriched_manifest.setdefault("obsKeys", ISAAC_STACK_OBS_KEYS)
    enriched_manifest.setdefault("actionDim", ISAAC_STACK_ACTION_DIM)
    enriched_manifest.setdefault("taskEnv", ISAAC_STACK_TASK_ENV)
    enriched_manifest.setdefault("datasetEnv", ISAAC_STACK_DATASET_ENV)
    enriched_manifest.setdefault("simulatorBackend", "isaac_lab")

    metrics = {
        "epochsCompleted": epoch or total_epochs,
        "totalEpochs": total_epochs,
        "sourceCheckpointEpoch": int(MODEL_EPOCH_PATTERN.search(source_checkpoint.name).group(1))
        if MODEL_EPOCH_PATTERN.search(source_checkpoint.name)
        else None,
    }
    _write_json(artifacts_dir / "metrics.json", metrics)

    final_checkpoint = checkpoints_dir / "model_final.pth"
    if not final_checkpoint.is_file() and source_checkpoint is not None:
        try:
            shutil.copy2(source_checkpoint, final_checkpoint)
        except OSError as exc:
            update_status(
                train_job_dir,
                {
                    "status": "failed",
                    "message": f"无法复制 checkpoint: {exc}",
                    "checkpointExists": False,
                },
            )
            sync_workspace_job(train_job_id)
            return

    status_data = _read_json(train_job_dir / "status.json")
    assets = register_checkpoint_assets(
        train_job_dir=train_job_dir,
        train_job_id=train_job_id,
        manifest=enriched_manifest,
        train_config=train_config,
        status=status_data,
        resolved_backend="isaac_robomimic_bc",
        framework_label="Robomimic BC",
        model_type="bc",
    )
    primary = next((item for item in assets if item.get("checkpointKind") == "final"), assets[-1] if assets else None)
    if primary is None:
        update_status(
            train_job_dir,
            {
                "status": "failed",
                "message": "训练结束但未登记任何模型资产",
                "checkpointExists": False,
            },
        )
        sync_workspace_job(train_job_id)
        return

    update_status(
        train_job_dir,
        {
            "status": "completed",
            "epoch": max(epoch, total_epochs),
            "totalEpochs": total_epochs,
            "progress": 1.0,
            "checkpointExists": True,
            "checkpointPath": str(primary.get("checkpointPath") or final_checkpoint),
            "modelAssetId": primary.get("modelAssetId"),
            "message": f"Isaac Robomimic BC 训练完成，已登记 {len(assets)} 个模型资产",
            "loss": loss,
            "processPid": None,
        },
    )
    sync_workspace_job(train_job_id)
