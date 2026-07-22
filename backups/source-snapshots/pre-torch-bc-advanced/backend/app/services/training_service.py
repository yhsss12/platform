"""Historical snapshot retained from before the advanced torch-BC refactor."""
from __future__ import annotations

import json
import logging
import os
import re
import secrets
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from fastapi import HTTPException, status

from app.services.workspace_job_service import (
    record_workspace_job_start,
    sync_workspace_job_from_runtime,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
TRAINING_ROOT = PROJECT_ROOT / "runtime_outputs" / "training"
CABLE_WORKING_DIR = PROJECT_ROOT / "integrations" / "CableThreadingMVP"
DUAL_ARM_WORKING_DIR = PROJECT_ROOT / "integrations" / "DualArmCableManipulation"
PYTHON_BIN = Path("/home/ubuntu/miniconda3/envs/cable-threading-mvp/bin/python")
DUAL_ARM_PYTHON_BIN = Path("/home/ubuntu/miniconda3/envs/cable/bin/python")
TRAIN_BC_SCRIPT = CABLE_WORKING_DIR / "examples" / "cable_threading" / "train_bc.py"
DUAL_ARM_TRAIN_BC_SCRIPT = DUAL_ARM_WORKING_DIR / "examples" / "train_bc.py"

ALLOWED_PATH_ROOTS = [
    (PROJECT_ROOT / "runtime_outputs").resolve(),
    CABLE_WORKING_DIR.resolve(),
    DUAL_ARM_WORKING_DIR.resolve(),
]

TRAIN_JOB_ID_PATTERN = re.compile(r"^train_\d{8}_\d{6}_[a-f0-9]{4}$")
TRAINING_DEVICE_LABEL = "L20"
EPOCH_LOG_PATTERN = re.compile(r"Epoch\s+(\d+)", re.IGNORECASE)
LOSS_LOG_PATTERN = re.compile(r"Loss(?:\s+\w+)*\s*[:=]\s*([0-9.eE+-]+)")

_RUNNING_THREADS: dict[str, threading.Thread] = {}


def _now_label() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _make_train_job_id() -> str:
    suffix = secrets.token_hex(2)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"train_{ts}_{suffix}"


def _make_model_asset_id() -> str:
    suffix = secrets.token_hex(2)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"model_{ts}_{suffix}"


def probe_training_capabilities() -> dict[str, Any]:
    evidence: list[str] = []
    supported: list[str] = []

    if TRAIN_BC_SCRIPT.is_file():
        evidence.append(str(TRAIN_BC_SCRIPT))
        supported.append("robomimic_bc")

    if DUAL_ARM_TRAIN_BC_SCRIPT.is_file():
        evidence.append(str(DUAL_ARM_TRAIN_BC_SCRIPT))
        if "torch_bc" not in supported:
            supported.append("torch_bc")

    readme = CABLE_WORKING_DIR / "README.md"
    if readme.is_file():
        evidence.append(str(readme))

    setup_py = CABLE_WORKING_DIR / "setup.py"
    if setup_py.is_file():
        evidence.append(str(setup_py))

    act_candidates = list(CABLE_WORKING_DIR.rglob("train*.py"))
    act_candidates = [
        p
        for p in act_candidates
        if p.name != "train_bc.py" and "DualArmCableManipulation" not in str(p)
    ]
    if act_candidates:
        for path in act_candidates:
            evidence.append(str(path))
        supported.append("act")

    found = bool(supported)
    if "robomimic_bc" in supported:
        recommended = "robomimic"
    elif "torch_bc" in supported:
        recommended = "torch_bc"
    elif "act" in supported:
        recommended = "act"
    else:
        recommended = "unavailable"

    return {
        "foundTrainingScripts": found,
        "supportedTrainingBackends": supported,
        "recommendedBackend": recommended,
        "evidence": evidence,
    }


def _validate_train_job_id(train_job_id: str) -> str:
    candidate = (train_job_id or "").strip()
    if not TRAIN_JOB_ID_PATTERN.match(candidate):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid training job ID format")
    return candidate


def _train_job_dir(train_job_id: str) -> Path:
    return TRAINING_ROOT / "jobs" / train_job_id


def _resolve_safe_path(raw_path: str) -> Path:
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = (PROJECT_ROOT / candidate).resolve()
    else:
        candidate = candidate.resolve()

    for root in ALLOWED_PATH_ROOTS:
        if str(candidate).startswith(str(root)):
            return candidate

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail=f"Path not allowed: {raw_path}",
    )


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


def _load_manifest(
    *,
    dataset_manifest_path: Optional[str],
    dataset_manifest: Optional[dict[str, Any]],
    train_job_dir: Path,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {}

    if dataset_manifest:
        manifest = dict(dataset_manifest)
    elif dataset_manifest_path:
        manifest = _read_json(_resolve_safe_path(dataset_manifest_path))

    if not manifest:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="datasetManifest or valid datasetManifestPath is required",
        )

    snapshot_path = train_job_dir / "artifacts" / "dataset_manifest.json"
    _write_json(snapshot_path, manifest)
    return manifest


def _artifact_path(manifest: dict[str, Any], key: str) -> Optional[Path]:
    artifacts = manifest.get("artifacts") or {}
    raw = artifacts.get(key)
    if not raw:
        return None
    try:
        return _resolve_safe_path(str(raw))
    except HTTPException:
        return None


def _normalize_downstream_model_type(value: str) -> str:
    normalized = (value or "").strip().lower()
    if normalized == "robomimic":
        return "Robomimic"
    if normalized == "act":
        return "ACT"
    if normalized == "dt":
        return "DT"
    if normalized in {"diffusion policy", "diffusion_policy"}:
        return "Diffusion Policy"
    return (value or "").strip()


def _normalize_training_backend_request(value: str) -> str:
    normalized = (value or "auto").strip().lower()
    if normalized in {"robomimic", "robomimic_bc"}:
        return "robomimic_bc"
    if normalized in {"torch", "torch_bc"}:
        return "torch_bc"
    return normalized


def _manifest_task_type(manifest: dict[str, Any]) -> str:
    return str(manifest.get("taskType") or "").strip()


def _is_dual_arm_manifest(manifest: dict[str, Any]) -> bool:
    task_type = _manifest_task_type(manifest)
    if task_type == "dual_arm_cable_manipulation":
        return True
    template_id = str(manifest.get("taskTemplateId") or "").strip()
    if template_id in {"dual_arm_cable_manipulation", "task_dual_arm_cable_manipulation_v1"}:
        return True
    source_job_id = str(manifest.get("sourceJobId") or "").strip()
    return source_job_id.startswith("dac_gen_")


def _is_valid_hdf5_file(path: Optional[Path]) -> bool:
    return path is not None and path.is_file() and path.stat().st_size > 0


def _resolve_hdf5_path(manifest: dict[str, Any]) -> Optional[Path]:
    hdf5_path = _artifact_path(manifest, "hdf5")
    if _is_valid_hdf5_file(hdf5_path):
        return hdf5_path

    built_path = manifest.get("builtDatasetPath")
    if built_path:
        try:
            resolved = _resolve_safe_path(str(built_path))
            if _is_valid_hdf5_file(resolved):
                return resolved
        except HTTPException:
            pass

    npz_path = _artifact_path(manifest, "npz")
    if npz_path is not None:
        sibling = npz_path.parent / "dataset.hdf5"
        if _is_valid_hdf5_file(sibling):
            return sibling

    source_job_id = str(manifest.get("sourceJobId") or "").strip()
    if source_job_id:
        runtime_roots = ["cable_threading", "dual_arm_cable"]
        if source_job_id.startswith("dac_gen_"):
            runtime_roots = ["dual_arm_cable", "cable_threading"]
        elif source_job_id.startswith("ct_gen_"):
            runtime_roots = ["cable_threading", "dual_arm_cable"]

        for runtime_name in runtime_roots:
            guessed = (
                PROJECT_ROOT
                / "runtime_outputs"
                / runtime_name
                / "jobs"
                / source_job_id
                / "datasets"
                / "dataset.hdf5"
            )
            try:
                resolved = _resolve_safe_path(str(guessed))
                if _is_valid_hdf5_file(resolved):
                    return resolved
            except HTTPException:
                continue

    return None


def _validate_dataset_trainable(manifest: dict[str, Any]) -> tuple[bool, str]:
    successful = int(manifest.get("successfulEpisodes") or 0)
    if successful <= 0:
        return False, "数据集无成功轨迹，无法训练"

    npz_path = _artifact_path(manifest, "npz")
    hdf5_path = _artifact_path(manifest, "hdf5")
    has_npz = npz_path is not None and npz_path.is_file()
    has_hdf5 = hdf5_path is not None and hdf5_path.is_file()

    if not has_npz and not has_hdf5:
        return False, "数据集缺少 NPZ / HDF5 轨迹文件"

    return True, ""


def _resolve_training_backend(
    *,
    downstream_model_type: str,
    training_backend: str,
    has_hdf5: bool,
    capabilities: dict[str, Any],
    manifest: dict[str, Any],
) -> tuple[Optional[str], str]:
    supported = set(capabilities.get("supportedTrainingBackends") or [])
    downstream = _normalize_downstream_model_type(downstream_model_type)
    backend_req = _normalize_training_backend_request(training_backend)
    is_dual_arm = _is_dual_arm_manifest(manifest)

    if is_dual_arm:
        if backend_req == "torch_bc":
            if "torch_bc" not in supported:
                return None, "双臂 torch_bc 训练脚本未找到"
            if not has_hdf5:
                return None, "torch_bc 训练需要 HDF5 数据集"
            return "torch_bc", ""
        if backend_req in {"robomimic_bc", "robomimic"}:
            return None, "双臂线缆数据集请使用 torch_bc 后端，不支持 robomimic_bc"
        if backend_req == "act":
            return None, "当前 ACT 训练后端未接入，无法启动真实训练"
        if backend_req in {"dt", "diffusion_policy"}:
            return None, f"当前 {downstream_model_type} 训练后端未接入，无法启动真实训练"
        if "torch_bc" in supported and has_hdf5:
            return "torch_bc", ""
        if not has_hdf5:
            return None, "torch_bc 训练需要 HDF5 数据集"
        return None, "双臂 torch_bc 训练脚本未找到"

    if backend_req == "torch_bc":
        return None, "torch_bc 仅适用于双臂线缆数据集"

    if backend_req == "robomimic_bc":
        if "robomimic_bc" not in supported:
            return None, "robomimic 训练脚本未找到"
        if not has_hdf5:
            return None, "robomimic BC 训练需要 HDF5 数据集"
        return "robomimic_bc", ""

    if backend_req == "act":
        if "act" not in supported:
            return None, "当前 ACT 训练后端未接入，无法启动真实训练"
        return "act", ""

    if backend_req in {"dt", "diffusion_policy"}:
        return None, f"当前 {downstream_model_type} 训练后端未接入，无法启动真实训练"

    # auto
    if downstream == "Robomimic":
        if "robomimic_bc" in supported and has_hdf5:
            return "robomimic_bc", ""
        if not has_hdf5:
            return None, "robomimic BC 训练需要 HDF5 数据集"
        return None, "robomimic 训练脚本未找到"

    if downstream in {"ACT", "DT", "Diffusion Policy", "LeRobot", "自定义模型"}:
        return None, f"当前 {downstream} 训练后端未接入，无法启动真实训练"

    if "robomimic_bc" in supported and has_hdf5:
        return "robomimic_bc", ""

    return None, "未找到可用训练后端"


def _resolve_device(device: str) -> str:
    value = (device or "cuda").strip().lower()
    if value in {"cuda_if_available", "auto", "l20"}:
        return "cuda"
    return value if value in {"cpu", "cuda"} else "cuda"


def _append_robomimic_advanced_args(cmd: list[str], train_config: dict[str, Any]) -> list[str]:
    if not train_config.get("advancedEnabled"):
        return cmd
    model_params = train_config.get("modelParams")
    if not isinstance(model_params, dict):
        return cmd

    dim_1 = int(model_params.get("actor_hidden_dim_1") or 512)
    dim_2 = int(model_params.get("actor_hidden_dim_2") or 512)
    dims_raw = model_params.get("actor_hidden_dims")
    if isinstance(dims_raw, str) and dims_raw.strip():
        parts = [int(part.strip()) for part in dims_raw.split(",") if part.strip()]
        if len(parts) >= 2:
            dim_1, dim_2 = parts[0], parts[1]
        elif len(parts) == 1:
            dim_1 = dim_2 = parts[0]
    cmd.extend(["--actor-hidden-dims", f"{dim_1},{dim_2}"])
    cmd.extend(["--l2-regularization", str(float(model_params.get("l2_regularization") or 0.0))])

    return cmd


def _build_train_command(
    *,
    backend: str,
    hdf5_path: Path,
    out_dir: Path,
    train_config: dict[str, Any],
) -> list[str]:
    epochs = int(train_config.get("epochs") or 5)
    batch_size = int(train_config.get("batchSize") or 16)
    learning_rate = float(train_config.get("learningRate") or 1e-4)
    device = _resolve_device(str(train_config.get("device") or ""))
    seed = int(train_config.get("seed") if train_config.get("seed") is not None else 1)

    if backend == "robomimic_bc":
        if not TRAIN_BC_SCRIPT.is_file():
            raise FileNotFoundError(str(TRAIN_BC_SCRIPT))
        if not PYTHON_BIN.is_file():
            raise FileNotFoundError(str(PYTHON_BIN))
        cmd = [
            str(PYTHON_BIN),
            str(TRAIN_BC_SCRIPT),
            "--dataset",
            str(hdf5_path),
            "--out-dir",
            str(out_dir),
            "--num-epochs",
            str(epochs),
            "--batch-size",
            str(batch_size),
            "--learning-rate",
            str(learning_rate),
            "--device",
            device,
            "--seed",
            str(seed),
        ]
        return _append_robomimic_advanced_args(cmd, train_config)

    if backend == "torch_bc":
        if not DUAL_ARM_TRAIN_BC_SCRIPT.is_file():
            raise FileNotFoundError(str(DUAL_ARM_TRAIN_BC_SCRIPT))
        if not DUAL_ARM_PYTHON_BIN.is_file():
            raise FileNotFoundError(str(DUAL_ARM_PYTHON_BIN))
        return [
            str(DUAL_ARM_PYTHON_BIN),
            str(DUAL_ARM_TRAIN_BC_SCRIPT),
            "--dataset",
            str(hdf5_path),
            "--out-dir",
            str(out_dir),
            "--num-epochs",
            str(epochs),
            "--batch-size",
            str(batch_size),
            "--learning-rate",
            str(learning_rate),
            "--device",
            device,
            "--seed",
            str(seed),
        ]

    raise ValueError(f"Unsupported backend: {backend}")


def _build_env(*, backend: str | None = None) -> dict[str, str]:
    env = os.environ.copy()
    if backend != "torch_bc":
        env["PYTHONNOUSERSITE"] = "1"
    return env


def _parse_training_log(log_path: Path, total_epochs: int) -> tuple[int, Optional[float]]:
    if not log_path.is_file():
        return 0, None
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return 0, None

    epoch = 0
    loss: Optional[float] = None
    for line in text.splitlines():
        epoch_match = EPOCH_LOG_PATTERN.search(line)
        if epoch_match:
            try:
                epoch = max(epoch, int(epoch_match.group(1)))
            except ValueError:
                pass
        loss_match = LOSS_LOG_PATTERN.search(line)
        if loss_match:
            try:
                loss = float(loss_match.group(1))
            except ValueError:
                pass

    epoch = min(epoch, total_epochs)
    return epoch, loss


def _find_checkpoint(search_dir: Path) -> Optional[Path]:
    if not search_dir.exists():
        return None
    candidates = [
        p
        for p in search_dir.rglob("*")
        if p.is_file() and p.suffix in {".pth", ".pt"} and p.stat().st_size > 0
    ]
    if not candidates:
        return None
    preferred_names = ("model.pt", "model_final.pth", "model.pth")
    for name in preferred_names:
        for path in candidates:
            if path.name == name:
                return path
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _update_status(train_job_dir: Path, patch: dict[str, Any]) -> dict[str, Any]:
    status_path = train_job_dir / "status.json"
    current = _read_json(status_path)
    current.update(patch)
    current["updatedAt"] = _now_label()
    _write_json(status_path, current)
    return current


def _register_model_manifest(
    *,
    train_job_dir: Path,
    train_job_id: str,
    manifest: dict[str, Any],
    train_config: dict[str, Any],
    checkpoint_path: Path,
    resolved_backend: str,
) -> dict[str, Any]:
    model_asset_id = _make_model_asset_id()
    task_type = _manifest_task_type(manifest) or (
        "dual_arm_cable_manipulation" if _is_dual_arm_manifest(manifest) else "cable_threading"
    )
    task_template_id = str(manifest.get("taskTemplateId") or "").strip()
    if not task_template_id:
        task_template_id = (
            "dual_arm_cable_manipulation"
            if task_type == "dual_arm_cable_manipulation"
            else "task_cable_threading_v1"
        )

    backend_type = "torch_bc" if resolved_backend == "torch_bc" else resolved_backend
    model_manifest = {
        "modelAssetId": model_asset_id,
        "sourceTrainJobId": train_job_id,
        "sourceDatasetId": manifest.get("datasetId"),
        "taskType": task_type,
        "taskTemplateId": task_template_id,
        "downstreamModelType": train_config.get("downstreamModelType"),
        "trainingBackend": resolved_backend,
        "backendType": backend_type,
        "modelType": "bc",
        "actionDim": manifest.get("actionDim"),
        "observationSchema": manifest.get("observationSchema"),
        "actionSchema": manifest.get("actionSchema"),
        "checkpointPath": str(checkpoint_path),
        "trainConfigPath": str(train_job_dir / "config" / "train_config.json"),
        "trainLogPath": str(train_job_dir / "logs" / "train.log"),
        "status": "ready",
        "createdAt": _now_label(),
    }
    _write_json(train_job_dir / "artifacts" / "model_manifest.json", model_manifest)
    return model_manifest


def _execute_training_job(train_job_id: str) -> None:
    train_job_dir = _train_job_dir(train_job_id)
    train_config = _read_json(train_job_dir / "config" / "train_config.json")
    manifest = _read_json(train_job_dir / "artifacts" / "dataset_manifest.json")
    log_path = train_job_dir / "logs" / "train.log"
    checkpoints_dir = train_job_dir / "checkpoints"

    total_epochs = int(train_config.get("epochs") or 5)
    capabilities = probe_training_capabilities()

    ok, reason = _validate_dataset_trainable(manifest)
    if not ok:
        _update_status(
            train_job_dir,
            {
                "status": "failed",
                "message": reason,
                "progress": 0.0,
            },
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(reason + "\n", encoding="utf-8")
        return

    hdf5_path = _resolve_hdf5_path(manifest)
    has_hdf5 = hdf5_path is not None and hdf5_path.is_file()

    resolved_backend, backend_message = _resolve_training_backend(
        downstream_model_type=str(train_config.get("downstreamModelType") or ""),
        training_backend=str(train_config.get("trainingBackend") or "auto"),
        has_hdf5=has_hdf5,
        capabilities=capabilities,
        manifest=manifest,
    )

    if resolved_backend is None:
        _update_status(
            train_job_dir,
            {
                "status": "backend_unavailable",
                "message": backend_message,
                "progress": 0.0,
            },
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(backend_message + "\n", encoding="utf-8")
        return

    device = _resolve_device(str(train_config.get("device") or ""))
    backend_out = checkpoints_dir / resolved_backend
    backend_out.mkdir(parents=True, exist_ok=True)

    working_dir = DUAL_ARM_WORKING_DIR if resolved_backend == "torch_bc" else CABLE_WORKING_DIR

    try:
        cmd = _build_train_command(
            backend=resolved_backend,
            hdf5_path=hdf5_path,  # type: ignore[arg-type]
            out_dir=backend_out,
            train_config=train_config,
        )
    except (FileNotFoundError, ValueError) as exc:
        _update_status(
            train_job_dir,
            {"status": "backend_unavailable", "message": str(exc), "progress": 0.0},
        )
        log_path.write_text(str(exc) + "\n", encoding="utf-8")
        return

    _update_status(
        train_job_dir,
        {
            "status": "running",
            "message": f"训练进行中（{resolved_backend}）",
            "trainingBackendResolved": resolved_backend,
            "command": cmd,
        },
    )

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("command: " + " ".join(cmd) + "\n\n")
        log_file.flush()
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(working_dir),
                env=_build_env(backend=resolved_backend),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except OSError as exc:
            _update_status(
                train_job_dir,
                {"status": "failed", "message": f"训练进程启动失败: {exc}", "progress": 0.0},
            )
            return

        while proc.poll() is None:
            epoch, loss = _parse_training_log(log_path, total_epochs)
            progress = min(0.99, epoch / total_epochs) if total_epochs > 0 else 0.0
            _update_status(
                train_job_dir,
                {
                    "status": "running",
                    "epoch": epoch,
                    "totalEpochs": total_epochs,
                    "progress": progress,
                    "loss": loss,
                },
            )
            time.sleep(2)

        return_code = proc.returncode if proc.returncode is not None else 1
        epoch, loss = _parse_training_log(log_path, total_epochs)

        checkpoint = _find_checkpoint(backend_out) or _find_checkpoint(checkpoints_dir)
        final_checkpoint: Optional[Path] = None
        if checkpoint is not None:
            suffix = checkpoint.suffix or ".pt"
            final_path = checkpoints_dir / f"model_final{suffix}"
            try:
                final_path.write_bytes(checkpoint.read_bytes())
                final_checkpoint = final_path
            except OSError:
                final_checkpoint = checkpoint

        if return_code == 0 and final_checkpoint is not None and final_checkpoint.is_file():
            model_manifest = _register_model_manifest(
                train_job_dir=train_job_dir,
                train_job_id=train_job_id,
                manifest=manifest,
                train_config=train_config,
                checkpoint_path=final_checkpoint,
                resolved_backend=resolved_backend,
            )
            _update_status(
                train_job_dir,
                {
                    "status": "completed",
                    "epoch": total_epochs,
                    "totalEpochs": total_epochs,
                    "progress": 1.0,
                    "loss": loss,
                    "checkpointExists": True,
                    "checkpointPath": str(final_checkpoint),
                    "modelAssetId": model_manifest["modelAssetId"],
                    "message": "训练完成，checkpoint 已生成",
                },
            )
            sync_workspace_job_from_runtime(train_job_id)
            return

        message = (
            "训练失败，未找到有效 checkpoint"
            if return_code == 0
            else f"训练进程退出，return code={return_code}"
        )
        _update_status(
            train_job_dir,
            {
                "status": "failed",
                "epoch": epoch,
                "totalEpochs": total_epochs,
                "progress": min(1.0, epoch / total_epochs) if total_epochs else 0.0,
                "loss": loss,
                "checkpointExists": False,
                "message": message,
            },
        )
        sync_workspace_job_from_runtime(train_job_id)


def create_training_job(payload: dict[str, Any]) -> dict[str, Any]:
    capabilities = probe_training_capabilities()
    train_job_id = _make_train_job_id()
    train_job_dir = _train_job_dir(train_job_id)
    (train_job_dir / "config").mkdir(parents=True, exist_ok=True)
    (train_job_dir / "logs").mkdir(parents=True, exist_ok=True)
    (train_job_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (train_job_dir / "artifacts").mkdir(parents=True, exist_ok=True)

    manifest = _load_manifest(
        dataset_manifest_path=payload.get("datasetManifestPath"),
        dataset_manifest=payload.get("datasetManifest"),
        train_job_dir=train_job_dir,
    )

    selected_downstream = _normalize_downstream_model_type(str(payload.get("downstreamModelType") or ""))
    selected_backend = _normalize_training_backend_request(str(payload.get("trainingBackend") or "auto"))
    selected_data_format = str(payload.get("dataFormat") or "HDF5")

    hdf5_path = _resolve_hdf5_path(manifest)
    if hdf5_path is not None:
        artifacts = dict(manifest.get("artifacts") or {})
        if not artifacts.get("hdf5"):
            artifacts["hdf5"] = str(hdf5_path)
            manifest["artifacts"] = artifacts
            _write_json(train_job_dir / "artifacts" / "dataset_manifest.json", manifest)

    resolved_device = _resolve_device(str(payload.get("device") or ""))
    device_label = str(payload.get("deviceLabel") or TRAINING_DEVICE_LABEL)
    train_config = {
        "datasetId": payload.get("datasetId"),
        "datasetManifestPath": payload.get("datasetManifestPath"),
        "downstreamModelType": selected_downstream,
        "trainingBackend": selected_backend,
        "dataFormat": selected_data_format,
        "selectedDownstreamModelType": selected_downstream,
        "selectedTrainingBackend": selected_backend,
        "selectedDataFormat": selected_data_format,
        "epochs": int(payload.get("epochs") or 5),
        "batchSize": int(payload.get("batchSize") or 16),
        "learningRate": float(payload.get("learningRate") or 1e-4),
        "device": resolved_device,
        "deviceLabel": device_label,
        "seed": int(payload.get("seed") if payload.get("seed") is not None else 1),
        "seedMode": payload.get("seedMode"),
        "advancedEnabled": bool(payload.get("advancedEnabled")),
        "modelParams": payload.get("modelParams"),
        "pretrained": payload.get("pretrained"),
        "capabilities": capabilities,
        "createdAt": _now_label(),
    }
    logger.info(
        "create_training_job trainJobId=%s downstream=%s backend=%s hdf5=%s",
        train_job_id,
        selected_downstream,
        selected_backend,
        str(hdf5_path) if hdf5_path else None,
    )
    _write_json(train_job_dir / "config" / "train_config.json", train_config)

    _update_status(
        train_job_dir,
        {
            "trainJobId": train_job_id,
            "status": "queued",
            "progress": 0.0,
            "epoch": 0,
            "totalEpochs": train_config["epochs"],
            "loss": None,
            "checkpointExists": False,
            "checkpointPath": None,
            "modelAssetId": None,
            "message": "training job created",
            "datasetId": manifest.get("datasetId"),
            "datasetName": manifest.get("datasetName"),
            "downstreamModelType": train_config["downstreamModelType"],
            "trainingBackend": train_config["trainingBackend"],
            "dataFormat": train_config["dataFormat"],
            "device": train_config["device"],
            "deviceLabel": train_config["deviceLabel"],
            "createdAt": train_config["createdAt"],
        },
    )

    record_workspace_job_start(
        job_id=train_job_id,
        job_type="training",
        task_type=str(manifest.get("taskType") or ("dual_arm_cable_manipulation" if _is_dual_arm_manifest(manifest) else "unknown")),
        runtime_path=str(train_job_dir),
        runner="train_bc.py" if _is_dual_arm_manifest(manifest) else "train_bc.py",
        status="pending",
        metadata={
            "datasetId": manifest.get("datasetId"),
            "datasetName": manifest.get("datasetName"),
            "downstreamModelType": train_config["downstreamModelType"],
            "trainingBackend": train_config["trainingBackend"],
            "trainConfig": train_config,
        },
    )

    thread = threading.Thread(
        target=_execute_training_job,
        args=(train_job_id,),
        name=f"training-{train_job_id}",
        daemon=True,
    )
    _RUNNING_THREADS[train_job_id] = thread
    thread.start()

    return {
        "trainJobId": train_job_id,
        "status": "queued",
        "message": "training job created",
    }


def _status_payload(train_job_id: str) -> dict[str, Any]:
    validated = _validate_train_job_id(train_job_id)
    sync_workspace_job_from_runtime(validated)
    train_job_dir = _train_job_dir(validated)
    status_data = _read_json(train_job_dir / "status.json")
    if not status_data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Training job not found")

    log_path = train_job_dir / "logs" / "train.log"
    train_config = _read_json(train_job_dir / "config" / "train_config.json")
    total_epochs = int(status_data.get("totalEpochs") or train_config.get("epochs") or 0)

    if status_data.get("status") == "running" and log_path.is_file():
        epoch, loss = _parse_training_log(log_path, total_epochs)
        if epoch > int(status_data.get("epoch") or 0):
            status_data["epoch"] = epoch
        if loss is not None:
            status_data["loss"] = loss
        if total_epochs > 0:
            status_data["progress"] = min(0.99, epoch / total_epochs)

    status_data.setdefault("trainJobId", validated)
    status_data.setdefault("device", train_config.get("device"))
    status_data.setdefault("deviceLabel", train_config.get("deviceLabel") or TRAINING_DEVICE_LABEL)
    return status_data


def get_training_job_status(train_job_id: str) -> dict[str, Any]:
    return _status_payload(train_job_id)


def read_training_job_log(train_job_id: str, lines: int = 80) -> str:
    validated = _validate_train_job_id(train_job_id)
    log_path = _train_job_dir(validated) / "logs" / "train.log"
    if not log_path.is_file():
        return ""
    try:
        content = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(content[-lines:])
    except OSError:
        return ""


def get_training_job_model(train_job_id: str) -> dict[str, Any]:
    validated = _validate_train_job_id(train_job_id)
    train_job_dir = _train_job_dir(validated)
    status_data = _read_json(train_job_dir / "status.json")
    model_manifest_path = train_job_dir / "artifacts" / "model_manifest.json"
    model_manifest = _read_json(model_manifest_path) if model_manifest_path.is_file() else None

    ready = bool(
        status_data.get("status") == "completed"
        and status_data.get("checkpointExists")
        and model_manifest
        and model_manifest.get("status") == "ready"
    )

    return {
        "trainJobId": validated,
        "ready": ready,
        "modelManifest": model_manifest,
        "checkpointPath": status_data.get("checkpointPath"),
    }


def list_training_jobs() -> list[dict[str, Any]]:
    jobs_root = TRAINING_ROOT / "jobs"
    if not jobs_root.is_dir():
        return []

    rows: list[dict[str, Any]] = []
    for job_dir in sorted(jobs_root.iterdir(), reverse=True):
        if not job_dir.is_dir():
            continue
        status_data = _read_json(job_dir / "status.json")
        if not status_data:
            continue
        rows.append(
            {
                "trainJobId": status_data.get("trainJobId") or job_dir.name,
                "status": status_data.get("status") or "queued",
                "datasetId": status_data.get("datasetId"),
                "datasetName": status_data.get("datasetName"),
                "downstreamModelType": status_data.get("downstreamModelType"),
                "trainingBackend": status_data.get("trainingBackend"),
                "createdAt": status_data.get("createdAt"),
                "checkpointExists": bool(status_data.get("checkpointExists")),
                "modelAssetId": status_data.get("modelAssetId"),
            }
        )
    return rows
