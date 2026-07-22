from __future__ import annotations

import json
import logging
import os
import re
import secrets
import shutil
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from fastapi import HTTPException, status

from app.core.platform_paths import is_path_within, platform_paths, resolve_runtime_reference
from app.services.workspace_job_service import (
    record_workspace_job_start,
    sync_workspace_job_from_runtime,
)
from app.services.training_job_sync_service import finalize_training_job_sync
from app.services.runtime_job_lifecycle import is_job_deleted, mark_job_deleted

logger = logging.getLogger(__name__)

PROJECT_ROOT = platform_paths.project_root
RUNTIME_ROOT = platform_paths.runs_root
TRAINING_ROOT = RUNTIME_ROOT / "training"
CABLE_WORKING_DIR = PROJECT_ROOT / "integrations" / "CableThreadingMVP"
DUAL_ARM_WORKING_DIR = PROJECT_ROOT / "integrations" / "DualArmCableManipulation"
PYTHON_BIN = Path("/home/ubuntu/miniconda3/envs/cable-threading-mvp/bin/python")
# Simulation uses the dedicated ``cable`` environment, while training needs
# torch + h5py. Keep dual-arm BC on the same dependency-complete training
# environment as the other HDF5 trainers.
DUAL_ARM_PYTHON_BIN = PYTHON_BIN
TRAIN_BC_SCRIPT = CABLE_WORKING_DIR / "examples" / "cable_threading" / "train_bc.py"
TRAIN_DP_SCRIPT = CABLE_WORKING_DIR / "examples" / "cable_threading" / "train_dp.py"
TRAIN_ACT_SCRIPT = CABLE_WORKING_DIR / "examples" / "cable_threading" / "train_act.py"
DUAL_ARM_TRAIN_BC_SCRIPT = DUAL_ARM_WORKING_DIR / "examples" / "train_bc.py"

DP_IMAGE_KEYS = ("agentview_image", "robot0_eye_in_hand_image")

ALLOWED_PATH_ROOTS = [
    RUNTIME_ROOT.resolve(),
    platform_paths.models.resolve(),
    CABLE_WORKING_DIR.resolve(),
    DUAL_ARM_WORKING_DIR.resolve(),
]

TRAIN_JOB_ID_PATTERN = re.compile(r"^train_\d{8}_\d{6}_[a-f0-9]{4}$")
TRAIN_JOB_ID_REFERENCE_PATTERN = re.compile(r"^[\w.-]+$")
TRAINING_DEVICE_LABEL = "L20"
EPOCH_LOG_PATTERN = re.compile(r"Epoch\s+(\d+)", re.IGNORECASE)
TRAIN_EPOCH_LOG_PATTERN = re.compile(r"Train\s+Epoch\s+(\d+)", re.IGNORECASE)
LOSS_LOG_PATTERN = re.compile(r"Loss(?:\s+\w+)*\s*[:=]\s*([0-9.eE+-]+)")
JSON_LOSS_LOG_PATTERN = re.compile(r'"Loss"\s*:\s*([-+0-9.eE]+)')

_RUNNING_THREADS: dict[str, threading.Thread] = {}
_RUNNING_PROCS: dict[str, subprocess.Popen] = {}

_CAPABILITIES_CACHE: dict[str, Any] | None = None
_CAPABILITIES_CACHE_AT: float = 0.0
_CAPABILITIES_CACHE_TTL_SEC = 120.0
_PROBE_LOCK = threading.Lock()
_PROBE_IN_PROGRESS = False
PI0_PROBE_PENDING_MESSAGE = "正在检测 runner"


def invalidate_training_capabilities_cache() -> None:
    global _CAPABILITIES_CACHE, _CAPABILITIES_CACHE_AT, _PROBE_IN_PROGRESS
    with _PROBE_LOCK:
        _CAPABILITIES_CACHE = None
        _CAPABILITIES_CACHE_AT = 0.0
        _PROBE_IN_PROGRESS = False


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


def probe_pi0_disabled_reason(capabilities: dict[str, Any]) -> str:
    pi0_cap = capabilities.get("pi0Capability") or {}
    reason = pi0_cap.get("reason")
    if isinstance(reason, str) and reason.strip():
        return reason.strip()
    from app.services.pi0_training_runner import PI0_RUNNER_DISABLED_REASON

    return PI0_RUNNER_DISABLED_REASON


def _collect_script_supported_backends(*, evidence: list[str]) -> list[str]:
    supported: list[str] = []
    if TRAIN_BC_SCRIPT.is_file():
        evidence.append(str(TRAIN_BC_SCRIPT))
        supported.append("robomimic_bc")
    if DUAL_ARM_TRAIN_BC_SCRIPT.is_file():
        evidence.append(str(DUAL_ARM_TRAIN_BC_SCRIPT))
        if "torch_bc" not in supported:
            supported.append("torch_bc")
    if TRAIN_DP_SCRIPT.is_file():
        evidence.append(str(TRAIN_DP_SCRIPT))
        if "diffusion_policy" not in supported:
            supported.append("diffusion_policy")
    if TRAIN_ACT_SCRIPT.is_file():
        evidence.append(str(TRAIN_ACT_SCRIPT))
        if "act" not in supported:
            supported.append("act")
    readme = CABLE_WORKING_DIR / "README.md"
    if readme.is_file():
        evidence.append(str(readme))
    setup_py = CABLE_WORKING_DIR / "setup.py"
    if setup_py.is_file():
        evidence.append(str(setup_py))
    return supported


def _build_capabilities_payload(
    *,
    supported: list[str],
    evidence: list[str],
    pi0_cap: dict[str, Any],
) -> dict[str, Any]:
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
        "pi0Capability": pi0_cap,
    }


def _probe_training_capabilities_fast() -> dict[str, Any]:
    """Fast probe for model-types list: script checks only, pi0 marked pending."""
    evidence: list[str] = []
    supported = _collect_script_supported_backends(evidence=evidence)
    pi0_cap: dict[str, Any] = {
        "ready": False,
        "reason": PI0_PROBE_PENDING_MESSAGE,
        "pending": True,
        "status": "pending",
        "evidence": [],
    }
    return _build_capabilities_payload(supported=supported, evidence=evidence, pi0_cap=pi0_cap)


def _probe_training_capabilities_uncached() -> dict[str, Any]:
    evidence: list[str] = []
    supported = _collect_script_supported_backends(evidence=evidence)

    try:
        from app.services.isaac_lab.training_service import probe_isaac_robomimic_training_capability

        isaac_cap = probe_isaac_robomimic_training_capability()
        if isaac_cap.get("ready"):
            supported.append("isaac_robomimic_bc")
            evidence.extend(isaac_cap.get("evidence") or [])
    except Exception as exc:
        logger.debug("isaac robomimic training probe skipped: %s", exc)

    pi0_cap: dict[str, Any] = {"ready": False, "reason": None, "pending": False, "status": "disabled", "evidence": []}
    try:
        from app.services.pi0_lerobot_smoke_runner import probe_pi0_lerobot_platform_training_capability

        lerobot_platform_cap = probe_pi0_lerobot_platform_training_capability()
        if lerobot_platform_cap.get("ready"):
            if "pi0" not in supported:
                supported.append("pi0")
            evidence.append("pi0_lerobot_platform_smoke")
            pi0_cap = {
                **pi0_cap,
                "ready": True,
                "reason": None,
                "pending": False,
                "status": "ready",
                "lerobotPlatformReady": True,
                "platformTrainingReady": True,
                "trainingMode": "lerobot_platform_smoke",
                "openpiRequired": False,
            }
    except Exception as exc:
        logger.debug("pi0 lerobot platform probe skipped: %s", exc)

    if not pi0_cap.get("ready"):
        try:
            from app.services.pi0_training_runner import probe_pi0_training_capability

            openpi_cap = probe_pi0_training_capability()
            pi0_cap = {
                **pi0_cap,
                **openpi_cap,
                "pending": False,
                "status": "ready" if openpi_cap.get("ready") else pi0_cap.get("status") or "disabled",
            }
            if openpi_cap.get("ready"):
                if "pi0" not in supported:
                    supported.append("pi0")
                evidence.extend(openpi_cap.get("evidence") or [])
        except Exception as exc:
            logger.debug("pi0 training probe skipped: %s", exc)
            if not pi0_cap.get("ready"):
                pi0_cap = {
                    **pi0_cap,
                    "ready": False,
                    "reason": str(exc),
                    "pending": False,
                    "status": "disabled",
                    "evidence": [],
                }

    return _build_capabilities_payload(supported=supported, evidence=evidence, pi0_cap=pi0_cap)


def _store_capabilities_cache(result: dict[str, Any]) -> None:
    global _CAPABILITIES_CACHE, _CAPABILITIES_CACHE_AT
    _CAPABILITIES_CACHE = result
    _CAPABILITIES_CACHE_AT = time.time()


def _run_background_capabilities_probe() -> None:
    global _PROBE_IN_PROGRESS
    try:
        result = _probe_training_capabilities_uncached()
        _store_capabilities_cache(result)
    except Exception as exc:
        logger.warning("background training capabilities probe failed: %s", exc)
    finally:
        with _PROBE_LOCK:
            _PROBE_IN_PROGRESS = False


def schedule_training_capabilities_probe_background() -> None:
    """Kick off full probe in a daemon thread; safe to call from request handlers."""
    global _PROBE_IN_PROGRESS
    now = time.time()
    with _PROBE_LOCK:
        if _CAPABILITIES_CACHE is not None and now - _CAPABILITIES_CACHE_AT < _CAPABILITIES_CACHE_TTL_SEC:
            return
        if _PROBE_IN_PROGRESS:
            return
        _PROBE_IN_PROGRESS = True
    thread = threading.Thread(
        target=_run_background_capabilities_probe,
        name="training-capabilities-probe",
        daemon=True,
    )
    thread.start()


def get_training_capabilities_for_model_types() -> dict[str, Any]:
    """Non-blocking capabilities snapshot for model-types list API."""
    now = time.time()
    if _CAPABILITIES_CACHE is not None:
        if now - _CAPABILITIES_CACHE_AT < _CAPABILITIES_CACHE_TTL_SEC:
            return dict(_CAPABILITIES_CACHE)
        schedule_training_capabilities_probe_background()
        return dict(_CAPABILITIES_CACHE)
    schedule_training_capabilities_probe_background()
    return _probe_training_capabilities_fast()


def probe_training_capabilities(*, force_refresh: bool = False) -> dict[str, Any]:
    """Full training capability probe (may block on pi0 openpi subprocess)."""
    now = time.time()
    if (
        not force_refresh
        and _CAPABILITIES_CACHE is not None
        and now - _CAPABILITIES_CACHE_AT < _CAPABILITIES_CACHE_TTL_SEC
    ):
        return dict(_CAPABILITIES_CACHE)
    result = _probe_training_capabilities_uncached()
    _store_capabilities_cache(result)
    return dict(result)


def _validate_train_job_id(train_job_id: str) -> str:
    candidate = (train_job_id or "").strip()
    if not TRAIN_JOB_ID_PATTERN.match(candidate):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid training job ID format")
    return candidate


def _sanitize_train_job_id_reference(train_job_id: str) -> Optional[str]:
    """Sanitize a training job id for path/DB lookup (no strict format requirement)."""
    candidate = (train_job_id or "").strip()
    if not candidate:
        return None
    if "\x00" in candidate or "/" in candidate or "\\" in candidate or ".." in candidate:
        return None
    if not TRAIN_JOB_ID_REFERENCE_PATTERN.match(candidate):
        return None
    return candidate


def _sanitize_train_job_id_for_delete(train_job_id: str) -> str:
    candidate = _sanitize_train_job_id_reference(train_job_id)
    if candidate is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid training job ID")
    return candidate


def _resolve_train_job_dir(train_job_id: str) -> Optional[Path]:
    """Resolve a job directory under the configured training jobs root."""
    candidate = _sanitize_train_job_id_reference(train_job_id)
    if candidate is None:
        return None
    jobs_root = (TRAINING_ROOT / "jobs").resolve()
    try:
        target = (jobs_root / candidate).resolve()
    except (OSError, ValueError):
        return None
    if target == jobs_root:
        return None
    if not is_path_within(target, jobs_root):
        return None
    return target


def _train_job_dir(train_job_id: str) -> Path:
    resolved = _resolve_train_job_dir(train_job_id)
    if resolved is not None:
        return resolved
    validated = _validate_train_job_id(train_job_id)
    return TRAINING_ROOT / "jobs" / validated


def _resolve_safe_path(raw_path: str) -> Path:
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        if candidate.parts and candidate.parts[0] == "runs":
            candidate = resolve_runtime_reference(str(candidate))
        else:
            candidate = (PROJECT_ROOT / candidate).resolve()
    else:
        candidate = candidate.resolve()

    for root in ALLOWED_PATH_ROOTS:
        if is_path_within(candidate, root):
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
    if normalized in {"isaac_robomimic", "isaac_robomimic_bc"}:
        return "isaac_robomimic_bc"
    if normalized in {"robomimic", "robomimic_bc"}:
        return "robomimic_bc"
    if normalized in {"torch", "torch_bc"}:
        return "torch_bc"
    return normalized


def _manifest_task_type(manifest: dict[str, Any]) -> str:
    return str(manifest.get("taskType") or "").strip()


def _is_isaac_block_stacking_manifest(manifest: dict[str, Any]) -> bool:
    from app.services.isaac_lab.training_service import is_isaac_block_stacking_manifest

    return is_isaac_block_stacking_manifest(manifest)


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

    top_level_hdf5 = manifest.get("hdf5")
    if top_level_hdf5:
        try:
            resolved = _resolve_safe_path(str(top_level_hdf5))
            if _is_valid_hdf5_file(resolved):
                return resolved
        except HTTPException:
            pass

    npz_path = _artifact_path(manifest, "npz")
    if npz_path is not None:
        sibling = npz_path.parent / "dataset.hdf5"
        if _is_valid_hdf5_file(sibling):
            return sibling

    dataset_file = manifest.get("datasetFile")
    if dataset_file:
        try:
            resolved = _resolve_safe_path(str(dataset_file))
            if _is_valid_hdf5_file(resolved):
                return resolved
        except HTTPException:
            pass

    source_job_id = str(manifest.get("sourceJobId") or "").strip()
    if source_job_id.startswith("isaac_gen_") or source_job_id.startswith("isaac_import_"):
        guessed = RUNTIME_ROOT / "isaac_lab" / "jobs" / source_job_id / "artifacts" / "dataset.hdf5"
        try:
            resolved = _resolve_safe_path(str(guessed))
            if _is_valid_hdf5_file(resolved):
                return resolved
        except HTTPException:
            pass

    if source_job_id:
        runtime_roots = ["cable_threading", "dual_arm_cable"]
        if source_job_id.startswith("dac_gen_"):
            runtime_roots = ["dual_arm_cable", "cable_threading"]
        elif source_job_id.startswith("ct_gen_"):
            runtime_roots = ["cable_threading", "dual_arm_cable"]

        for runtime_name in runtime_roots:
            guessed = RUNTIME_ROOT / runtime_name / "jobs" / source_job_id / "datasets" / "dataset.hdf5"
            try:
                resolved = _resolve_safe_path(str(guessed))
                if _is_valid_hdf5_file(resolved):
                    return resolved
            except HTTPException:
                continue

    return None


def _validate_dataset_trainable(manifest: dict[str, Any]) -> tuple[bool, str]:
    if _is_isaac_block_stacking_manifest(manifest):
        episodes = int(
            manifest.get("successfulEpisodes")
            or manifest.get("episodeCount")
            or manifest.get("episodes")
            or 0
        )
        if episodes <= 0:
            return False, "数据集无 demo，无法训练"
        hdf5_path = _resolve_hdf5_path(manifest)
        if not _is_valid_hdf5_file(hdf5_path):
            return False, "数据集缺少 HDF5 文件"
        return True, ""

    successful = int(
        manifest.get("successfulEpisodes")
        or manifest.get("num_successful")
        or manifest.get("numSuccessful")
        or 0
    )
    if successful <= 0:
        return False, "数据集无成功轨迹，无法训练"

    npz_path = _artifact_path(manifest, "npz")
    hdf5_path = _resolve_hdf5_path(manifest)
    has_npz = npz_path is not None and npz_path.is_file()
    has_hdf5 = _is_valid_hdf5_file(hdf5_path)

    from app.services.pi0_lerobot_loader import resolve_lerobot_path_from_manifest, validate_lerobot_for_pi0

    lerobot_path = resolve_lerobot_path_from_manifest(manifest)
    if lerobot_path is not None:
        ok, reason = validate_lerobot_for_pi0(lerobot_path)
        if ok:
            return True, ""
        if not has_npz and not has_hdf5:
            return False, reason or "LeRobot 数据集未通过 pi0 校验"

    if not has_npz and not has_hdf5:
        return False, "数据集缺少 NPZ / HDF5 轨迹文件"

    return True, ""


def _validate_dp_hdf5(hdf5_path: Path, train_config: Optional[dict[str, Any]] = None) -> tuple[bool, str]:
    """校验 DP 数据集硬性条件：可读 demo 与 action_dim；camera keys 由适配层配置决定。"""
    try:
        import h5py
    except ImportError:
        return False, "h5py 不可用，无法校验 Diffusion Policy 数据集"

    dp_config = (train_config or {}).get("dpConfig") if isinstance((train_config or {}).get("dpConfig"), dict) else {}
    low_dim_keys = list(dp_config.get("low_dim_keys") or [])
    image_keys = list(dp_config.get("image_keys") or [])

    try:
        with h5py.File(hdf5_path, "r") as handle:
            data_group = handle.get("data")
            if data_group is None:
                return False, "HDF5 缺少 data 分组"
            demos = [key for key in data_group.keys() if str(key).startswith("demo_")]
            if not demos:
                return False, "HDF5 无 demo 数据"
            demo = data_group[demos[0]]
            action_key = str(dp_config.get("action_key") or "actions")
            if demo.get(action_key) is None:
                if action_key == "joint_actions" and demo.get("joint_actions") is not None:
                    gripper_key = dp_config.get("gripper_action_key")
                    if gripper_key and demo.get(gripper_key) is None:
                        return False, f"HDF5 缺少 gripper action key {gripper_key!r}"
                else:
                    return False, f"HDF5 缺少 action key {action_key!r}"
            elif demo.get("actions") is None and action_key == "actions":
                return False, "HDF5 缺少 actions"
            obs = demo.get("obs")
            if obs is None:
                return False, "HDF5 缺少 obs 观测"

            if image_keys:
                for key in image_keys:
                    if key not in obs:
                        return False, f"Diffusion Policy 配置需要的图像键 {key!r} 在 HDF5 中不存在"
            elif low_dim_keys:
                for key in low_dim_keys:
                    if key not in obs:
                        return False, f"Diffusion Policy 配置需要的 low_dim 键 {key!r} 在 HDF5 中不存在"
            elif not list(obs.keys()):
                return False, "HDF5 obs 分组为空，无法构建 Diffusion Policy 输入"
    except OSError as exc:
        return False, f"无法读取 HDF5: {exc}"

    return True, ""


def _validate_act_hdf5(hdf5_path: Path, train_config: dict[str, Any]) -> tuple[bool, str]:
    act_config = train_config.get("actConfig") or {}
    adaptation = (train_config.get("adaptationSnapshot") or {}).get("modelAdaptation") or {}
    input_cfg = adaptation.get("inputConfig") or {}
    image_keys = list(
        act_config.get("image_keys")
        or input_cfg.get("image_keys")
        or input_cfg.get("camera_keys")
        or input_cfg.get("camera_names")
        or []
    )
    low_dim_keys = list(act_config.get("low_dim_keys") or input_cfg.get("low_dim_keys") or [])
    if not image_keys:
        return False, "ACT 需要图像观测键，但适配配置中 image_keys 为空"

    try:
        import h5py
    except ImportError:
        return False, "h5py 不可用，无法校验 ACT HDF5"

    try:
        with h5py.File(hdf5_path, "r") as handle:
            data_group = handle.get("data")
            if data_group is None:
                return False, "HDF5 缺少 data 分组"
            demo_keys = sorted(k for k in data_group.keys() if str(k).startswith("demo_"))
            if not demo_keys:
                return False, "HDF5 data 分组内无 demo_* 轨迹"
            first_demo = data_group[demo_keys[0]]
            if first_demo.get("actions") is None:
                return False, "HDF5 demo 缺少 actions"
            obs = first_demo.get("obs")
            if obs is None:
                return False, "HDF5 demo 缺少 obs 分组"
            missing = [key for key in image_keys + low_dim_keys if key not in obs]
            if missing:
                return False, f"ACT 配置需要的 obs 键在 HDF5 中不存在: {', '.join(missing)}"
    except OSError as exc:
        return False, f"无法读取 HDF5: {exc}"

    return True, ""


def _manifest_has_pi0_ready_lerobot(manifest: dict[str, Any]) -> bool:
    from app.services.pi0_lerobot_loader import resolve_lerobot_path_from_manifest, validate_lerobot_for_pi0

    lerobot_path = resolve_lerobot_path_from_manifest(manifest)
    if lerobot_path is None:
        return False
    ok, _ = validate_lerobot_for_pi0(lerobot_path)
    return ok


def _enrich_training_manifest_lerobot(manifest: dict[str, Any]) -> dict[str, Any]:
    from app.services.pi0_lerobot_loader import resolve_lerobot_path_from_manifest

    if resolve_lerobot_path_from_manifest(manifest) is not None:
        return manifest

    source_job_id = str(manifest.get("sourceJobId") or "").strip()
    if not source_job_id.startswith("ct_gen_"):
        return manifest

    job_manifest_path = next(
        (
            root / "cable_threading" / "jobs" / source_job_id / "datasets" / "dataset.manifest.json"
            for root in (RUNTIME_ROOT,)
            if (root / "cable_threading" / "jobs" / source_job_id / "datasets" / "dataset.manifest.json").is_file()
        ),
        None,
    )
    if job_manifest_path is None:
        return manifest

    raw = _read_json(job_manifest_path)
    enriched = dict(manifest)
    for key in (
        "lerobot",
        "lerobotMetadata",
        "availableFormats",
        "datasetFormats",
        "primaryFormat",
        "format",
        "datasetFormat",
        "dataFormat",
        "pi0Ready",
        "pi0ReadyReason",
        "state_dim",
        "controller_type",
        "action_mode",
        "taskDescription",
    ):
        if key not in enriched and raw.get(key) is not None:
            enriched[key] = raw[key]

    lerobot_path = resolve_lerobot_path_from_manifest(raw)
    if lerobot_path is not None:
        artifacts = dict(enriched.get("artifacts") or {})
        artifacts.setdefault("lerobot", str(lerobot_path))
        artifacts.setdefault("lerobotPath", str(lerobot_path))
        enriched["artifacts"] = artifacts
    return enriched


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

    def _require_hdf5(backend_key: str) -> tuple[Optional[str], str]:
        if backend_key not in supported:
            return None, f"{backend_key} 训练脚本未找到"
        if not has_hdf5:
            return None, f"{backend_key} 训练需要 HDF5 数据集"
        return backend_key, ""

    # Dual-arm cable datasets use their own flattened bimanual observation
    # schema and dedicated PyTorch BC runner. The generic model selection in
    # the UI historically sends robomimic_bc here, but that runner expects the
    # single-arm cable-threading schema. Resolve the generic choice to the
    # dataset-declared backend instead of launching an incompatible process.
    if _is_dual_arm_manifest(manifest) and backend_req in {"auto", "robomimic_bc"}:
        return _require_hdf5("torch_bc")

    if _is_isaac_block_stacking_manifest(manifest) and backend_req == "robomimic_bc":
        return None, "物块堆叠数据必须使用 isaac_robomimic_bc 训练后端"

    explicit_backends = {
        "robomimic_bc": lambda: _require_hdf5("robomimic_bc"),
        "torch_bc": lambda: _require_hdf5("torch_bc"),
        "diffusion_policy": lambda: _require_hdf5("diffusion_policy"),
        "isaac_robomimic_bc": lambda: _require_hdf5("isaac_robomimic_bc"),
        "act": lambda: _require_hdf5("act"),
        "pi0": lambda: (
            (None, probe_pi0_disabled_reason(capabilities))
            if "pi0" not in supported
            else (
                ("pi0", "")
                if _manifest_has_pi0_ready_lerobot(manifest)
                else _require_hdf5("pi0")
            )
        ),
        "dt": lambda: (None, f"当前 {downstream_model_type} 训练后端未接入，无法启动真实训练"),
    }

    if backend_req in explicit_backends:
        return explicit_backends[backend_req]()

    if backend_req == "robomimic":
        if _is_isaac_block_stacking_manifest(manifest):
            return _require_hdf5("isaac_robomimic_bc")
        return _require_hdf5("robomimic_bc")

    # auto：按 downstream 类型推断
    if downstream == "Diffusion Policy":
        return _require_hdf5("diffusion_policy")

    if downstream == "ACT":
        return _require_hdf5("act")

    if downstream == "Robomimic":
        if _is_dual_arm_manifest(manifest):
            return _require_hdf5("torch_bc")
        if _is_isaac_block_stacking_manifest(manifest):
            return _require_hdf5("isaac_robomimic_bc")
        return _require_hdf5("robomimic_bc")

    if downstream in {"ACT", "DT", "LeRobot", "自定义模型"}:
        return None, f"当前 {downstream} 训练后端未接入，无法启动真实训练"

    if "robomimic_bc" in supported and has_hdf5:
        return "robomimic_bc", ""

    return None, "未找到可用训练后端"


def _resolve_device(device: str) -> str:
    value = (device or "cuda").strip().lower()
    if value in {"cuda_if_available", "auto", "l20"}:
        return "cuda"
    return value if value in {"cpu", "cuda"} else "cuda"


def _append_checkpoint_save_args(
    cmd: list[str],
    train_config: dict[str, Any],
    *,
    backend: str,
    epochs: int,
) -> list[str]:
    from app.services.checkpoint_registry import parse_save_policy, save_capabilities_for_backend

    policy = parse_save_policy(train_config)
    caps = save_capabilities_for_backend(backend)

    if backend == "robomimic_bc":
        interval = policy.get("checkpointIntervalEpochs")
        if interval and caps.get("interval"):
            cmd.extend(["--save-every-n-epochs", str(int(interval))])
        elif policy.get("saveFinal", True):
            cmd.extend(["--save-every-n-epochs", str(max(1, epochs))])
        else:
            cmd.extend(["--save-every-n-epochs", "0"])
        if policy.get("saveBest") and caps.get("best"):
            cmd.append("--save-best")
        else:
            cmd.append("--no-save-best")
        if policy.get("saveFinal", True):
            cmd.append("--save-final")
        else:
            cmd.append("--no-save-final")
    return cmd


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


def _append_torch_bc_advanced_args(cmd: list[str], train_config: dict[str, Any]) -> list[str]:
    if not train_config.get("advancedEnabled"):
        return cmd
    model_params = train_config.get("modelParams")
    if not isinstance(model_params, dict):
        return cmd

    dims_raw = model_params.get("hidden_dims")
    if isinstance(dims_raw, str) and dims_raw.strip():
        cmd.extend(["--hidden-dims", dims_raw.strip()])
    else:
        cmd.extend(["--hidden-dims", "512,512"])

    weight_decay = model_params.get("weight_decay")
    cmd.extend(["--weight-decay", str(float(weight_decay if weight_decay is not None else 0.0))])

    return cmd


def _append_init_checkpoint_args(cmd: list[str], train_config: dict[str, Any]) -> list[str]:
    pretrained = train_config.get("pretrained")
    if not isinstance(pretrained, dict):
        return cmd
    checkpoint_path = str(pretrained.get("checkpointPath") or "").strip()
    if not checkpoint_path:
        return cmd
    resolved = _resolve_safe_path(checkpoint_path)
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise FileNotFoundError(f"pretrained checkpoint not found or empty: {checkpoint_path}")
    cmd.extend(["--init-checkpoint", str(resolved)])
    return cmd


def _append_dp_advanced_args(cmd: list[str], train_config: dict[str, Any]) -> list[str]:
    if not train_config.get("advancedEnabled"):
        return cmd
    model_params = train_config.get("modelParams")
    if not isinstance(model_params, dict):
        return cmd

    int_fields = (
        ("horizon", "--horizon"),
        ("n_obs_steps", "--n-obs-steps"),
        ("n_action_steps", "--n-action-steps"),
        ("num_inference_steps", "--num-inference-steps"),
        ("num_diffusion_steps", "--num-diffusion-steps"),
        ("image_size", "--image-size"),
    )
    for key, flag in int_fields:
        value = model_params.get(key)
        if value is not None:
            cmd.extend([flag, str(int(value))])

    vision_encoder = model_params.get("vision_encoder")
    if vision_encoder is not None:
        cmd.extend(["--vision-encoder", str(vision_encoder)])

    if model_params.get("ema_decay") is not None:
        cmd.extend(["--ema-decay", str(float(model_params["ema_decay"]))])
    if model_params.get("weight_decay") is not None:
        cmd.extend(["--weight-decay", str(float(model_params["weight_decay"]))])

    use_ema = model_params.get("use_ema")
    if use_ema is not None:
        cmd.extend(["--use-ema", "true" if bool(use_ema) else "false"])

    return cmd


def _apply_pretrained_fields_to_train_config(
    train_config: dict[str, Any],
    normalized_pretrained: dict[str, Any],
) -> None:
    train_config["pretrained"] = normalized_pretrained
    init_payload = {
        "enabled": True,
        "mode": "from_checkpoint",
        "modelAssetId": normalized_pretrained.get("modelAssetId"),
        "checkpointPath": normalized_pretrained.get("checkpointPath"),
        "modelAssetName": normalized_pretrained.get("modelAssetName"),
        "sourceTrainJobId": normalized_pretrained.get("sourceTrainJobId"),
        "trainingBackend": normalized_pretrained.get("trainingBackend"),
    }
    train_config["initializationWeight"] = init_payload
    train_config["pretrainedModel"] = dict(init_payload)


def _validate_pretrained_model(
    *,
    pretrained: Any,
    resolved_backend: str,
    manifest: dict[str, Any],
    train_config: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    if not isinstance(pretrained, dict):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="pretrained payload must be an object",
        )

    model_asset_id = str(pretrained.get("modelAssetId") or "").strip()
    checkpoint_path = str(pretrained.get("checkpointPath") or "").strip()
    if not model_asset_id and not checkpoint_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="pretrained requires modelAssetId or checkpointPath",
        )

    from app.services.model_asset_checkpoint_resolver import resolve_local_checkpoint_path
    from app.services.training_backend_canonical import (
        canonicalize_training_backend,
        resolve_asset_training_backend,
        training_backends_compatible,
    )
    from app.services.workspace_model_asset_service import get_model_asset_by_id

    resolved_backend = canonicalize_training_backend(resolved_backend) or resolved_backend

    asset = get_model_asset_by_id(model_asset_id) if model_asset_id else None
    if asset:
        asset_backend = resolve_asset_training_backend(asset)
        target_backend = canonicalize_training_backend(resolved_backend)
        if asset_backend and target_backend and not training_backends_compatible(asset_backend, target_backend):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"初始化权重模型类型 {asset_backend!r} 与当前训练后端 {target_backend!r} 不一致，"
                    "请选择相同类型的模型资产。"
                ),
            )
        task_type = str(manifest.get("taskType") or "").strip()
        asset_task = str(asset.get("taskTemplateId") or "").strip()
        is_dual_arm_manifest = _is_dual_arm_manifest(manifest)
        dual_arm_templates = {"dual_arm_cable_manipulation", "task_dual_arm_cable_manipulation_v1"}
        asset_framework = resolve_asset_training_backend(asset)
        asset_is_dual_arm = asset_task in dual_arm_templates or asset_framework == "torch_bc"
        manifest_is_dual_arm = is_dual_arm_manifest or task_type == "dual_arm_cable_manipulation"
        if asset_is_dual_arm != manifest_is_dual_arm:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="pretrained model task domain does not match dataset manifest",
            )
        local_path = resolve_local_checkpoint_path(
            asset=asset,
            path_hint=checkpoint_path,
            model_asset_id=model_asset_id,
        )
        checkpoint_path = local_path or str(asset.get("checkpointPath") or checkpoint_path).strip()
    elif checkpoint_path:
        local_path = resolve_local_checkpoint_path(
            path_hint=checkpoint_path,
            model_asset_id=model_asset_id,
        )
        if local_path:
            checkpoint_path = local_path

    if not checkpoint_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="pretrained checkpointPath is required",
        )

    resolved = _resolve_safe_path(checkpoint_path)
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"pretrained checkpoint not found or empty: {checkpoint_path}",
        )

    if resolved_backend == "diffusion_policy":
        from app.services.training_dataset_compat import validate_dp_pretrained_checkpoint

        validate_dp_pretrained_checkpoint(
            checkpoint_path=resolved,
            train_config=train_config or {},
        )

    normalized = dict(pretrained)
    normalized["modelAssetId"] = model_asset_id or str(asset.get("id") if asset else "")
    normalized["checkpointPath"] = str(resolved)
    normalized["initializationMode"] = "from_checkpoint"
    normalized["trainingBackend"] = canonicalize_training_backend(resolved_backend)
    if asset and not normalized.get("modelAssetName"):
        normalized["modelAssetName"] = asset.get("displayName") or asset.get("name")
    if asset and not normalized.get("sourceTrainJobId"):
        normalized["sourceTrainJobId"] = asset.get("sourceTrainingJobId")
    return normalized


def _build_train_command(
    *,
    backend: str,
    hdf5_path: Path | list[Path],
    out_dir: Path,
    train_config: dict[str, Any],
) -> list[str]:
    epochs = int(train_config.get("epochs") or 5)
    batch_size = int(train_config.get("batchSize") or 16)
    learning_rate = float(train_config.get("learningRate") or 1e-4)
    device = _resolve_device(str(train_config.get("device") or ""))
    seed = int(train_config.get("seed") if train_config.get("seed") is not None else 1)
    hdf5_paths = hdf5_path if isinstance(hdf5_path, list) else [hdf5_path]
    primary_hdf5 = hdf5_paths[0]

    if backend == "robomimic_bc":
        if not TRAIN_BC_SCRIPT.is_file():
            raise FileNotFoundError(str(TRAIN_BC_SCRIPT))
        if not PYTHON_BIN.is_file():
            raise FileNotFoundError(str(PYTHON_BIN))
        cmd = [
            str(PYTHON_BIN),
            str(TRAIN_BC_SCRIPT),
            "--dataset",
            str(primary_hdf5),
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
        return _append_init_checkpoint_args(
            _append_checkpoint_save_args(
                _append_robomimic_advanced_args(cmd, train_config),
                train_config,
                backend=backend,
                epochs=epochs,
            ),
            train_config,
        )

    if backend == "torch_bc":
        if not DUAL_ARM_TRAIN_BC_SCRIPT.is_file():
            raise FileNotFoundError(str(DUAL_ARM_TRAIN_BC_SCRIPT))
        if not DUAL_ARM_PYTHON_BIN.is_file():
            raise FileNotFoundError(str(DUAL_ARM_PYTHON_BIN))
        cmd = [
            str(DUAL_ARM_PYTHON_BIN),
            str(DUAL_ARM_TRAIN_BC_SCRIPT),
            "--dataset",
            str(primary_hdf5),
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
        return _append_init_checkpoint_args(
            _append_torch_bc_advanced_args(cmd, train_config),
            train_config,
        )

    if backend == "diffusion_policy":
        if not TRAIN_DP_SCRIPT.is_file():
            raise FileNotFoundError(str(TRAIN_DP_SCRIPT))
        if not PYTHON_BIN.is_file():
            raise FileNotFoundError(str(PYTHON_BIN))
        dp_config_path = train_config.get("dpConfigPath")
        fallback_config = TRAIN_DP_SCRIPT.parent / "dp_configs" / "cable_threading.yaml"
        cmd = [
            str(PYTHON_BIN),
            str(TRAIN_DP_SCRIPT),
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
        if len(hdf5_paths) > 1:
            cmd.extend(["--datasets", ",".join(str(path) for path in hdf5_paths)])
        else:
            cmd.extend(["--dataset", str(primary_hdf5)])
        if dp_config_path and Path(str(dp_config_path)).is_file():
            cmd.extend(["--config", str(dp_config_path)])
        elif fallback_config.is_file():
            cmd.extend(["--config", str(fallback_config)])
        return _append_init_checkpoint_args(_append_dp_advanced_args(cmd, train_config), train_config)

    if backend == "act":
        if not TRAIN_ACT_SCRIPT.is_file():
            raise FileNotFoundError(str(TRAIN_ACT_SCRIPT))
        if not PYTHON_BIN.is_file():
            raise FileNotFoundError(str(PYTHON_BIN))
        act_config_path = train_config.get("actConfigPath")
        if not act_config_path or not Path(str(act_config_path)).is_file():
            raise FileNotFoundError("actConfigPath missing or not found")
        metrics_path = out_dir.parent.parent / "artifacts" / "metrics.jsonl"
        cmd = [
            str(PYTHON_BIN),
            str(TRAIN_ACT_SCRIPT),
            "--dataset",
            str(primary_hdf5),
            "--out-dir",
            str(out_dir),
            "--config",
            str(act_config_path),
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
            "--metrics-path",
            str(metrics_path),
        ]
        return _append_act_advanced_args(cmd, train_config)

    if backend == "pi0":
        from app.services.pi0_training_runner import build_pi0_train_command

        return build_pi0_train_command(
            hdf5_path=hdf5_path,
            out_dir=out_dir,
            train_config=train_config,
            train_job_dir=out_dir.parent.parent,
        )

    raise ValueError(f"Unsupported backend: {backend}")


def _append_act_advanced_args(cmd: list[str], train_config: dict[str, Any]) -> list[str]:
    if not train_config.get("advancedEnabled"):
        return cmd
    model_params = train_config.get("modelParams")
    if not isinstance(model_params, dict):
        return cmd

    int_fields = (
        ("chunk_size", "--chunk-size"),
        ("hidden_dim", "--hidden-dim"),
    )
    for key, flag in int_fields:
        value = model_params.get(key)
        if value is not None:
            cmd.extend([flag, str(int(value))])

    if model_params.get("kl_weight") is not None:
        cmd.extend(["--kl-weight", str(float(model_params["kl_weight"]))])

    return cmd


def _backend_framework_meta(resolved_backend: str) -> tuple[str, str]:
    if resolved_backend == "diffusion_policy":
        return "Diffusion Policy", "diffusion_policy"
    if resolved_backend == "act":
        return "ACT", "act"
    if resolved_backend == "pi0":
        return "pi0", "pi0"
    if resolved_backend == "torch_bc":
        return "BC (PyTorch)", "torch_bc"
    if resolved_backend == "isaac_robomimic_bc":
        return "Robomimic BC", "bc"
    return "Robomimic BC", "bc"


def _build_env(*, backend: str | None = None) -> dict[str, str]:
    if backend == "pi0":
        from app.services.pi0_training_runner import build_pi0_env

        return build_pi0_env(train_job_dir=TRAINING_ROOT)
    env = os.environ.copy()
    if backend != "torch_bc":
        env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONFAULTHANDLER"] = "1"
    return env


def _append_nested_training_error(train_job_dir: Path, log_file: Any) -> None:
    """Robomimic redirects the complete traceback into its own nested log."""
    candidates = [
        path
        for path in (train_job_dir / "checkpoints").rglob("log.txt")
        if path.is_file()
    ]
    if not candidates:
        return
    nested_log = max(candidates, key=lambda path: path.stat().st_mtime_ns)
    try:
        content = nested_log.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return
    marker = content.rfind("Traceback (most recent call last):")
    diagnostic = content[marker:] if marker >= 0 else content[-16_384:]
    if not diagnostic.strip():
        return
    log_file.write(f"\n\n===== nested training diagnostic: {nested_log} =====\n")
    log_file.write(diagnostic.rstrip() + "\n")
    log_file.flush()


def _resolve_training_log_path(train_job_dir: Path) -> Path:
    """优先 train.log；Isaac 训练进行中可能仅有 stdout.log。"""
    train_log = train_job_dir / "logs" / "train.log"
    stdout_log = train_job_dir / "logs" / "stdout.log"
    try:
        if train_log.is_file() and train_log.stat().st_size > 0:
            return train_log
        if stdout_log.is_file() and stdout_log.stat().st_size > 0:
            return stdout_log
    except OSError:
        pass
    return train_log


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _reconcile_remote_training_on_read(train_job_id: str, train_job_dir: Path, status_data: dict[str, Any]) -> dict[str, Any]:
    """读取状态时尝试同步 remote_ssh 任务（SSH 轮询中断后的补偿）。"""
    if str(status_data.get("executionMode") or "").lower() != "remote_ssh":
        return status_data
    token = str(status_data.get("status") or "").lower()
    if token not in {"running", "starting"}:
        return status_data
    try:
        from app.services.training_remote_runner import reconcile_remote_training_job_runtime

        reconcile_remote_training_job_runtime(train_job_id)
    except Exception as exc:
        logger.warning("remote training reconcile on read failed trainJobId=%s: %s", train_job_id, exc)
    return _read_json(train_job_dir / "status.json") or status_data


def _status_json_mtime(train_job_dir: Path) -> float:
    path = train_job_dir / "status.json"
    try:
        return path.stat().st_mtime if path.is_file() else 0.0
    except OSError:
        return 0.0


def _reconcile_stale_running_training_job(
    train_job_id: str,
    train_job_dir: Path,
    status_data: dict[str, Any],
) -> dict[str, Any]:
    """进程已退出但状态仍为 running/starting 时修正为 failed 或 starting。"""
    token = str(status_data.get("status") or "").lower()
    if token not in {"running", "starting"}:
        return status_data
    if train_job_id in _RUNNING_PROCS:
        return status_data

    if str(status_data.get("executionMode") or "").lower() == "remote_ssh":
        from app.services.training_job_status import (
            REMOTE_TRAINING_STARTUP_TIMEOUT_SEC,
            enrich_and_persist_training_job_status,
            infer_training_job_completed,
            training_activity_detected,
        )
        from app.services.training_metrics import parse_training_logs

        status_data = _reconcile_remote_training_on_read(train_job_id, train_job_dir, status_data)
        token = str(status_data.get("status") or "").lower()
        if token not in {"running", "starting"}:
            return status_data

        train_config = _read_json(train_job_dir / "config" / "train_config.json")
        total_epochs = int(status_data.get("totalEpochs") or train_config.get("epochs") or 0)
        if infer_training_job_completed(status_data, train_job_dir=train_job_dir):
            return enrich_and_persist_training_job_status(train_job_id, train_job_dir, status_data)
        epoch, loss = parse_training_logs(train_job_dir, total_epochs)
        if epoch >= total_epochs and total_epochs > 0:
            return enrich_and_persist_training_job_status(train_job_id, train_job_dir, status_data)

        if training_activity_detected(train_job_dir, status_data):
            return status_data

        stale_after = _status_json_mtime(train_job_dir)
        updated_raw = status_data.get("updatedAt")
        if isinstance(updated_raw, str) and updated_raw.strip():
            try:
                from datetime import datetime

                parsed = datetime.fromisoformat(updated_raw.replace("Z", "+00:00"))
                stale_after = max(stale_after, parsed.timestamp())
            except ValueError:
                pass

        age_sec = time.time() - stale_after if stale_after > 0 else REMOTE_TRAINING_STARTUP_TIMEOUT_SEC + 1
        if age_sec >= REMOTE_TRAINING_STARTUP_TIMEOUT_SEC:
            message = "远端训练长时间未产生日志，可能启动失败或节点不可达"
            _update_status(
                train_job_dir,
                {
                    "status": "failed",
                    "message": message,
                    "epoch": epoch,
                    "totalEpochs": total_epochs,
                    "progress": 0.0,
                    "loss": loss,
                    "processPid": None,
                },
            )
            sync_workspace_job_from_runtime(train_job_id)
            return _read_json(train_job_dir / "status.json")

        if token == "running":
            _update_status(
                train_job_dir,
                {
                    "status": "starting",
                    "message": status_data.get("message")
                    or "远端训练已调度，等待训练进程产出日志",
                    "progress": 0.0,
                },
            )
            return _read_json(train_job_dir / "status.json")

        return status_data

    if str(status_data.get("trainingBackendResolved") or status_data.get("trainingBackend") or "") == "isaac_robomimic_bc":
        from app.services.isaac_lab.training_service import recover_completed_isaac_training

        if recover_completed_isaac_training(train_job_id, train_job_dir):
            return _read_json(train_job_dir / "status.json")

    pid_raw = status_data.get("processPid")
    pid = int(pid_raw) if pid_raw is not None else 0
    if pid and _pid_alive(pid):
        return status_data

    log_path = _resolve_training_log_path(train_job_dir)
    if not log_path.is_file():
        return status_data

    try:
        log_mtime = log_path.stat().st_mtime
    except OSError:
        return status_data

    # 日志仍在刷新时暂不判定失败（无 pid 的旧任务靠 mtime 兜底）
    if not pid and (time.time() - log_mtime) < 120:
        return status_data

    train_config = _read_json(train_job_dir / "config" / "train_config.json")
    total_epochs = int(status_data.get("totalEpochs") or train_config.get("epochs") or 0)
    from app.services.training_metrics import parse_training_logs

    epoch, loss = parse_training_logs(train_job_dir, total_epochs)
    if epoch >= total_epochs and total_epochs > 0:
        from app.services.training_job_status import enrich_and_persist_training_job_status

        return enrich_and_persist_training_job_status(train_job_id, train_job_dir, status_data)

    message = "训练进程已退出但未正常完成"
    if pid and not _pid_alive(pid):
        message = f"训练进程异常退出（pid={pid}）"
    elif not pid and (time.time() - log_mtime) >= 120:
        message = "训练日志长时间无更新，进程可能已异常退出"

    _update_status(
        train_job_dir,
        {
            "status": "failed",
            "message": message,
            "epoch": epoch,
            "totalEpochs": total_epochs,
            "progress": min(1.0, epoch / total_epochs) if total_epochs else 0.0,
            "loss": loss,
            "processPid": None,
        },
    )
    sync_workspace_job_from_runtime(train_job_id)
    return _read_json(train_job_dir / "status.json")


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
        train_epoch_match = TRAIN_EPOCH_LOG_PATTERN.search(line)
        if train_epoch_match:
            try:
                epoch = max(epoch, int(train_epoch_match.group(1)))
            except ValueError:
                pass
        epoch_match = EPOCH_LOG_PATTERN.search(line)
        if epoch_match:
            try:
                epoch = max(epoch, int(epoch_match.group(1)))
            except ValueError:
                pass
        json_loss_match = JSON_LOSS_LOG_PATTERN.search(line)
        if json_loss_match:
            try:
                loss = float(json_loss_match.group(1))
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
        "isaac_block_stacking"
        if _is_isaac_block_stacking_manifest(manifest)
        else "dual_arm_cable_manipulation"
        if _is_dual_arm_manifest(manifest)
        else "cable_threading"
    )
    task_template_id = str(manifest.get("taskTemplateId") or "").strip()
    if not task_template_id:
        if task_type == "isaac_block_stacking":
            task_template_id = "isaac_block_stacking"
        elif task_type == "dual_arm_cable_manipulation":
            task_template_id = "dual_arm_cable_manipulation"
        else:
            task_template_id = "task_cable_threading_v1"

    backend_type = (
        "isaac_robomimic_bc"
        if resolved_backend == "isaac_robomimic_bc"
        else "torch_bc"
        if resolved_backend == "torch_bc"
        else resolved_backend
    )
    framework_label, model_type = _backend_framework_meta(resolved_backend)
    if resolved_backend in {"isaac_robomimic_bc", "robomimic_bc"}:
        framework_label = "Robomimic BC"
    elif resolved_backend == "torch_bc":
        framework_label = "BC (PyTorch)"
    elif resolved_backend == "diffusion_policy":
        framework_label = "Diffusion Policy"
    elif resolved_backend == "act":
        framework_label = "ACT"
    elif resolved_backend == "pi0":
        framework_label = "pi0"
    status_data = _read_json(train_job_dir / "status.json")
    dataset_display_name = str(
        status_data.get("datasetName") or manifest.get("datasetName") or ""
    ).strip() or None
    training_task_name = str(train_config.get("taskName") or status_data.get("taskName") or "").strip() or None
    created_at = _now_label()
    from app.services.model_asset_naming import build_model_asset_display_name

    display_name = build_model_asset_display_name(
        training_task_name=training_task_name,
        dataset_name=dataset_display_name,
        dataset_id=str(manifest.get("datasetId") or status_data.get("datasetId") or "") or None,
        task_template_id=task_template_id,
        task_type=task_type,
        framework=framework_label,
        model_type=model_type,
        training_backend=resolved_backend,
        created_at=created_at,
    )
    model_manifest = {
        "modelAssetId": model_asset_id,
        "name": display_name,
        "displayName": display_name,
        "trainingTaskName": training_task_name,
        "datasetDisplayName": dataset_display_name,
        "sourceTrainJobId": train_job_id,
        "sourceDatasetId": manifest.get("datasetId") or status_data.get("datasetId"),
        "taskType": task_type,
        "taskTemplateId": task_template_id,
        "downstreamModelType": train_config.get("downstreamModelType"),
        "trainingBackend": resolved_backend,
        "backendType": backend_type,
        "framework": framework_label,
        "modelType": model_type,
        "actionDim": manifest.get("actionDim"),
        "observationSchema": manifest.get("observationSchema"),
        "actionSchema": manifest.get("actionSchema"),
        "obsKeys": manifest.get("obsKeys"),
        "simulatorBackend": manifest.get("simulatorBackend"),
        "taskEnv": manifest.get("taskEnv"),
        "datasetEnv": manifest.get("datasetEnv"),
        "checkpointPath": str(checkpoint_path),
        "trainConfigPath": str(train_job_dir / "config" / "train_config.json"),
        "trainLogPath": str(train_job_dir / "logs" / "train.log"),
        "status": "ready",
        "createdAt": created_at,
    }
    dp_config = train_config.get("dpConfig") if isinstance(train_config.get("dpConfig"), dict) else {}
    if resolved_backend == "diffusion_policy":
        model_manifest["modelType"] = "diffusion_policy"
        model_manifest["actionMode"] = (
            train_config.get("actionMode")
            or train_config.get("trainedActionMode")
            or dp_config.get("trained_action_mode")
            or dp_config.get("action_mode")
        )
        model_manifest["trainedActionMode"] = model_manifest.get("actionMode")
        model_manifest["controllerType"] = (
            train_config.get("controllerType") or dp_config.get("controller_type")
        )
        model_manifest["evalExecutor"] = (
            train_config.get("evalExecutor") or dp_config.get("eval_executor")
        )
        model_manifest["actionSchema"] = (
            train_config.get("actionSchema") or dp_config.get("action_schema") or manifest.get("actionSchema")
        )
        model_manifest["observationSchema"] = (
            train_config.get("observationSchema")
            or dp_config.get("observation_schema")
            or manifest.get("observationSchema")
        )
        model_manifest["controllerSchema"] = (
            train_config.get("controllerSchema") or dp_config.get("controller_schema")
        )
        model_manifest["sideChannelSchema"] = (
            train_config.get("sideChannelSchema") or dp_config.get("side_channel_schema")
        )
        model_manifest["actionKey"] = dp_config.get("action_key")
        model_manifest["gripperActionKey"] = dp_config.get("gripper_action_key")
        if dp_config.get("action_dim") is not None:
            model_manifest["actionDim"] = dp_config.get("action_dim")
    elif resolved_backend == "pi0":
        from app.services.pi0_lerobot_loader import resolve_lerobot_path_from_manifest
        from app.services.pi0_lerobot_smoke_runner import (
            PI0_EVAL_DISABLED_REASON,
            build_pi0_lerobot_model_manifest_fields,
        )

        dataset_path = resolve_lerobot_path_from_manifest(manifest)
        if dataset_path is None:
            configured_path = str(train_config.get("datasetPath") or "").strip()
            if configured_path:
                dataset_path = Path(configured_path)
        if dataset_path is not None and dataset_path.is_dir():
            model_manifest.update(
                build_pi0_lerobot_model_manifest_fields(
                    manifest=manifest,
                    train_config=train_config,
                    dataset_path=dataset_path,
                )
            )
        model_manifest["modelType"] = "pi0"
        model_manifest["policyType"] = "pi0"
        model_manifest["framework"] = "openpi"
        model_manifest["datasetFormat"] = train_config.get("datasetFormat") or model_manifest.get("datasetFormat") or "lerobot"
        if "canEvaluate" not in model_manifest:
            from app.services.policy_schema_resolver import resolve_pi0_model_asset_eval_fields

            model_manifest.update(resolve_pi0_model_asset_eval_fields(model_manifest))
    pretrained = train_config.get("pretrained")
    if isinstance(pretrained, dict) and pretrained.get("modelAssetId"):
        model_manifest["initModelAssetId"] = pretrained.get("modelAssetId")
        if pretrained.get("sourceTrainJobId"):
            model_manifest["initSourceTrainJobId"] = pretrained.get("sourceTrainJobId")
    _write_json(train_job_dir / "artifacts" / "model_manifest.json", model_manifest)
    return model_manifest


def _execute_training_job(train_job_id: str) -> None:
    try:
        _execute_training_job_impl(train_job_id)
    finally:
        _RUNNING_THREADS.pop(train_job_id, None)


def _execute_training_job_impl(train_job_id: str) -> None:
    from app.services.training_dataset_compat import resolve_training_hdf5_paths

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

    hdf5_paths = resolve_training_hdf5_paths(manifest, train_config)
    hdf5_path = hdf5_paths[0] if hdf5_paths else _resolve_hdf5_path(manifest)
    has_hdf5 = bool(hdf5_paths) or (hdf5_path is not None and hdf5_path.is_file())

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

    if len(hdf5_paths) > 1 and resolved_backend != "diffusion_policy":
        message = "多数据集合并训练当前仅支持 Diffusion Policy"
        _update_status(train_job_dir, {"status": "failed", "message": message, "progress": 0.0})
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(message + "\n", encoding="utf-8")
        return

    if resolved_backend == "diffusion_policy":
        paths_to_validate = hdf5_paths or ([hdf5_path] if hdf5_path is not None else [])
        for path in paths_to_validate:
            ok, reason = _validate_dp_hdf5(path, train_config)
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

    if resolved_backend == "act" and hdf5_path is not None:
        ok, reason = _validate_act_hdf5(hdf5_path, train_config)
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

    if resolved_backend == "pi0":
        from app.services.pi0_lerobot_loader import resolve_lerobot_path_from_manifest

        lerobot_path = resolve_lerobot_path_from_manifest(manifest)
        if lerobot_path is not None:
            from app.services.pi0_lerobot_smoke_runner import execute_pi0_lerobot_platform_training

            execute_pi0_lerobot_platform_training(
                train_job_id=train_job_id,
                train_job_dir=train_job_dir,
                manifest=manifest,
                train_config=train_config,
                update_status=lambda patch: _update_status(train_job_dir, patch),
                register_model_manifest=_register_model_manifest,
                sync_workspace_job=sync_workspace_job_from_runtime,
                finalize_training_job_sync=finalize_training_job_sync,
            )
            return

        if hdf5_path is not None:
            from app.services.pi0_training_runner import validate_pi0_dataset

            ok, reason = validate_pi0_dataset(hdf5_path, train_config, manifest=manifest)
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

            from app.services.pi0_training_runner import prepare_pi0_job_artifacts

            try:
                prepare_pi0_job_artifacts(
                    train_job_dir=train_job_dir,
                    manifest=manifest,
                    train_config=train_config,
                    hdf5_path=hdf5_path,
                )
                train_config = _read_json(train_job_dir / "config" / "train_config.json")
            except OSError as exc:
                message = f"pi0 训练配置生成失败: {exc}"
                _update_status(train_job_dir, {"status": "failed", "message": message, "progress": 0.0})
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text(message + "\n", encoding="utf-8")
                return

    if resolved_backend == "pi0" and hdf5_path is None:
        message = "pi0 训练需要 pi0Ready LeRobot 数据集或 HDF5 数据集"
        _update_status(train_job_dir, {"status": "failed", "message": message, "progress": 0.0})
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(message + "\n", encoding="utf-8")
        return

    pretrained_raw = train_config.get("pretrained")
    if isinstance(pretrained_raw, dict) and (
        pretrained_raw.get("modelAssetId") or pretrained_raw.get("checkpointPath")
    ):
        try:
            normalized_pretrained = _validate_pretrained_model(
                pretrained=pretrained_raw,
                resolved_backend=resolved_backend,
                manifest=manifest,
                train_config=train_config,
            )
            _apply_pretrained_fields_to_train_config(train_config, normalized_pretrained)
            _write_json(train_job_dir / "config" / "train_config.json", train_config)
        except HTTPException as exc:
            message = str(exc.detail)
            _update_status(
                train_job_dir,
                {"status": "failed", "message": message, "progress": 0.0},
            )
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(message + "\n", encoding="utf-8")
            return

    if resolved_backend == "isaac_robomimic_bc":
        from app.services.isaac_lab.training_service import execute_isaac_robomimic_training

        execute_isaac_robomimic_training(
            train_job_id=train_job_id,
            train_job_dir=train_job_dir,
            manifest=manifest,
            train_config=train_config,
            hdf5_path=hdf5_path,  # type: ignore[arg-type]
            update_status=_update_status,
            register_model_manifest=_register_model_manifest,
            sync_workspace_job=sync_workspace_job_from_runtime,
            register_running_proc=lambda proc: _RUNNING_PROCS.__setitem__(train_job_id, proc),
            unregister_running_proc=lambda: _RUNNING_PROCS.pop(train_job_id, None),
        )
        return

    if str(train_config.get("executionMode") or "") == "remote_ssh":
        from app.services.training_node_service import resolve_training_node

        training_node = resolve_training_node(training_node_id=str(train_config.get("trainingNodeId") or ""))
        if training_node and training_node.execution_mode == "remote_ssh":
            from app.services.training_remote_runner import execute_remote_training_job

            execute_remote_training_job(
                train_job_id=train_job_id,
                train_job_dir=train_job_dir,
                manifest=manifest,
                train_config=train_config,
                node=training_node,
                resolved_backend=resolved_backend,
                hdf5_path=hdf5_paths if len(hdf5_paths) > 1 else hdf5_path,
                total_epochs=total_epochs,
                build_train_command=_build_train_command,
                resolve_device=_resolve_device,
                project_root=PROJECT_ROOT,
                cable_working_dir=CABLE_WORKING_DIR,
                dual_arm_working_dir=DUAL_ARM_WORKING_DIR,
                update_status=_update_status,
                register_model_manifest=_register_model_manifest,
                sync_workspace_job=sync_workspace_job_from_runtime,
                register_running_proc=lambda proc: _RUNNING_PROCS.__setitem__(train_job_id, proc),
                unregister_running_proc=lambda: _RUNNING_PROCS.pop(train_job_id, None),
                running_procs=_RUNNING_PROCS,
                finalize_training_job_sync=finalize_training_job_sync,
                find_checkpoint=_find_checkpoint,
                backend_framework_meta=_backend_framework_meta,
            )
            return

    device = _resolve_device(str(train_config.get("device") or ""))
    backend_out = checkpoints_dir / resolved_backend
    backend_out.mkdir(parents=True, exist_ok=True)

    working_dir = (
        DUAL_ARM_WORKING_DIR
        if resolved_backend == "torch_bc"
        else PROJECT_ROOT
        if resolved_backend == "pi0"
        else CABLE_WORKING_DIR
    )

    try:
        cmd = _build_train_command(
            backend=resolved_backend,
            hdf5_path=hdf5_paths if len(hdf5_paths) > 1 else hdf5_path,  # type: ignore[arg-type]
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

        _RUNNING_PROCS[train_job_id] = proc
        from app.services.checkpoint_registry import parse_save_policy, register_checkpoint_assets
        from app.services.training_metrics import append_metrics_point

        framework_label, model_type = _backend_framework_meta(resolved_backend)
        sync_counter = 0
        try:
            while proc.poll() is None:
                from app.services.training_metrics import parse_pi0_metrics_from_jsonl, parse_training_logs

                if resolved_backend == "pi0":
                    epoch, loss = parse_pi0_metrics_from_jsonl(train_job_dir, total_epochs)
                else:
                    epoch, loss = parse_training_logs(train_job_dir, total_epochs)
                if epoch > 0 and resolved_backend not in {"act", "pi0"}:
                    append_metrics_point(train_job_dir, epoch=epoch, loss=loss)
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
                if sync_counter % 4 == 0:
                    status_data = _read_json(train_job_dir / "status.json")
                    register_checkpoint_assets(
                        train_job_dir=train_job_dir,
                        train_job_id=train_job_id,
                        manifest=manifest,
                        train_config=train_config,
                        status=status_data,
                        resolved_backend=resolved_backend,
                        framework_label=framework_label,
                        model_type=model_type,
                        register_final=False,
                    )
                    sync_workspace_job_from_runtime(train_job_id)
                sync_counter += 1
                time.sleep(2)

            return_code = proc.returncode if proc.returncode is not None else 1
            log_file.flush()
            if return_code != 0:
                _append_nested_training_error(train_job_dir, log_file)
            from app.services.training_metrics import parse_pi0_metrics_from_jsonl, parse_training_logs, sync_metrics_from_logs

            status_snapshot = _read_json(train_job_dir / "status.json")
            if resolved_backend == "pi0":
                epoch, loss = parse_pi0_metrics_from_jsonl(train_job_dir, total_epochs)
            else:
                epoch, loss = parse_training_logs(train_job_dir, total_epochs)
            if epoch > 0 and resolved_backend not in {"act", "pi0"}:
                append_metrics_point(train_job_dir, epoch=epoch, loss=loss)
            sync_metrics_from_logs(train_job_dir, status_snapshot or {"totalEpochs": total_epochs, "epoch": epoch})

            checkpoint = _find_checkpoint(backend_out) or _find_checkpoint(checkpoints_dir)
            if checkpoint is not None and parse_save_policy(train_config).get("saveFinal", True):
                suffix = checkpoint.suffix or ".pt"
                final_path = checkpoints_dir / f"model_final{suffix}"
                try:
                    if not final_path.is_file():
                        final_path.write_bytes(checkpoint.read_bytes())
                except OSError:
                    pass

            status_data = _read_json(train_job_dir / "status.json")
            completion_status = {
                **status_data,
                "status": "completed" if return_code == 0 else status_data.get("status"),
                "epoch": max(epoch, total_epochs) if return_code == 0 else epoch,
                "totalEpochs": total_epochs,
            }
            assets = register_checkpoint_assets(
                train_job_dir=train_job_dir,
                train_job_id=train_job_id,
                manifest=manifest,
                train_config=train_config,
                status=completion_status,
                resolved_backend=resolved_backend,
                framework_label=framework_label,
                model_type=model_type,
                register_final=(return_code == 0),
            )
            primary = next((item for item in assets if item.get("checkpointKind") == "final"), assets[-1] if assets else None)

            if return_code == 0 and primary is not None:
                _update_status(
                    train_job_dir,
                    {
                        "status": "completed",
                        "epoch": max(epoch, total_epochs),
                        "totalEpochs": total_epochs,
                        "progress": 1.0,
                        "loss": loss,
                        "checkpointExists": True,
                        "checkpointPath": str(primary.get("checkpointPath") or ""),
                        "modelAssetId": primary.get("modelAssetId"),
                        "message": f"训练完成，已登记 {len(assets)} 个模型资产",
                    },
                )
                finalize_training_job_sync(train_job_id)
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
            finalize_training_job_sync(train_job_id)
        finally:
            _RUNNING_PROCS.pop(train_job_id, None)


def stop_training_job_if_active(train_job_id: str) -> None:
    """Terminate an in-flight training subprocess so runtime files can be removed."""
    train_job_dir = _resolve_train_job_dir(train_job_id)
    if train_job_dir is None:
        return

    proc = _RUNNING_PROCS.pop(train_job_id, None)
    if proc is None:
        sanitized = _sanitize_train_job_id_reference(train_job_id)
        if sanitized is not None:
            proc = _RUNNING_PROCS.pop(sanitized, None)
    if proc is not None and proc.poll() is None:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    status_path = train_job_dir / "status.json"
    if status_path.is_file():
        status_data = _read_json(status_path) or {}
        current = str(status_data.get("status") or "").lower()
        if current in {"queued", "pending", "running"}:
            _update_status(
                train_job_dir,
                {
                    "status": "canceled",
                    "message": "训练任务已取消",
                },
            )
            sync_workspace_job_from_runtime(train_job_id)


def _resolve_training_manifest_from_payload(
    payload: dict[str, Any],
    train_job_dir: Path,
) -> tuple[dict[str, Any], list[Path]]:
    manifests_raw = payload.get("datasetManifests")
    if isinstance(manifests_raw, list) and len(manifests_raw) >= 1:
        manifests = [dict(item) for item in manifests_raw if isinstance(item, dict)]
        if len(manifests) > 1:
            from app.services.training_dataset_compat import merge_training_manifests

            merged, hdf5_paths, _ = merge_training_manifests(manifests)
            _write_json(train_job_dir / "artifacts" / "dataset_manifest.json", merged)
            return merged, hdf5_paths
        if len(manifests) == 1:
            manifest = _load_manifest(
                dataset_manifest_path=payload.get("datasetManifestPath"),
                dataset_manifest=manifests[0],
                train_job_dir=train_job_dir,
            )
            return manifest, []

    manifest = _load_manifest(
        dataset_manifest_path=payload.get("datasetManifestPath"),
        dataset_manifest=payload.get("datasetManifest"),
        train_job_dir=train_job_dir,
    )
    return manifest, []


def _validate_nut_assembly_training_dataset(
    manifest: dict[str, Any], hdf5_path: Optional[Path]
) -> None:
    """Reject generated NutAssembly datasets that contain no trainable demo."""
    if str(manifest.get("taskType") or "").strip() != "nut_assembly" or hdf5_path is None:
        return

    try:
        import h5py

        with h5py.File(hdf5_path, "r") as dataset:
            mask = dataset.get("mask/train")
            if mask is not None and len(mask) > 0:
                return
            demos = dataset.get("data")
            valid_count = 0
            if demos is not None:
                valid_count = sum(
                    1
                    for demo in demos.values()
                    if bool(
                        demo.attrs.get(
                            "valid_for_training",
                            demo.attrs.get("success", demo.attrs.get("success_flag", False)),
                        )
                    )
                )
            if valid_count > 0:
                return
    except (OSError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"螺母装配训练数据无法读取: {exc}",
        ) from exc

    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail="当前螺母装配数据集中没有成功且可用于训练的轨迹，请重新生成数据后再训练。",
    )


def create_training_job(payload: dict[str, Any]) -> dict[str, Any]:
    from app.services.adapter_layer.model_type_training_config import resolve_training_payload_from_model_type

    try:
        payload = resolve_training_payload_from_model_type(payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    capabilities = probe_training_capabilities()
    train_job_id = _make_train_job_id()
    train_job_dir = _train_job_dir(train_job_id)
    (train_job_dir / "config").mkdir(parents=True, exist_ok=True)
    (train_job_dir / "logs").mkdir(parents=True, exist_ok=True)
    (train_job_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (train_job_dir / "artifacts").mkdir(parents=True, exist_ok=True)

    manifest, merged_hdf5_paths = _resolve_training_manifest_from_payload(payload, train_job_dir)
    manifest = _enrich_training_manifest_lerobot(manifest)
    _write_json(train_job_dir / "artifacts" / "dataset_manifest.json", manifest)

    selected_downstream = _normalize_downstream_model_type(str(payload.get("downstreamModelType") or ""))
    selected_backend = _normalize_training_backend_request(str(payload.get("trainingBackend") or "auto"))
    selected_data_format = str(payload.get("dataFormat") or "HDF5")

    hdf5_path = _resolve_hdf5_path(manifest)
    try:
        _validate_nut_assembly_training_dataset(manifest, hdf5_path)
    except HTTPException:
        shutil.rmtree(train_job_dir, ignore_errors=True)
        raise
    if merged_hdf5_paths:
        artifacts = dict(manifest.get("artifacts") or {})
        artifacts["hdf5"] = str(merged_hdf5_paths[0])
        artifacts["hdf5Paths"] = [str(path) for path in merged_hdf5_paths]
        manifest["artifacts"] = artifacts
        _write_json(train_job_dir / "artifacts" / "dataset_manifest.json", manifest)
    elif hdf5_path is not None:
        artifacts = dict(manifest.get("artifacts") or {})
        if not artifacts.get("hdf5"):
            artifacts["hdf5"] = str(hdf5_path)
            manifest["artifacts"] = artifacts
            _write_json(train_job_dir / "artifacts" / "dataset_manifest.json", manifest)

    from app.services.adapter_layer.training_adaptation_integration import (
        apply_training_adaptation,
        write_adaptation_artifact,
    )

    adaptation_fields, adaptation_snapshot = apply_training_adaptation(
        manifest=manifest,
        payload=payload,
        train_job_dir=train_job_dir,
    )
    write_adaptation_artifact(train_job_dir, adaptation_snapshot)

    validation = adaptation_snapshot.get("validation") or {}
    if validation.get("adaptable") is False:
        errors = validation.get("errors") or []
        if train_job_dir.is_dir():
            shutil.rmtree(train_job_dir, ignore_errors=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=errors[0] if errors else "数据集与所选模型类型不兼容",
        )

    selected_downstream = _normalize_downstream_model_type(
        str(adaptation_fields.get("downstreamModelType") or payload.get("downstreamModelType") or "")
    )
    selected_backend = _normalize_training_backend_request(
        str(adaptation_fields.get("trainingBackend") or payload.get("trainingBackend") or "auto")
    )

    supported_backends = set(capabilities.get("supportedTrainingBackends") or [])
    if selected_backend == "pi0" and "pi0" not in supported_backends:
        if train_job_dir.is_dir():
            shutil.rmtree(train_job_dir, ignore_errors=True)

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=probe_pi0_disabled_reason(capabilities),
        )

    resolved_device = _resolve_device(str(payload.get("device") or ""))
    device_label = str(payload.get("deviceLabel") or TRAINING_DEVICE_LABEL)
    training_node_id = str(payload.get("trainingNodeId") or "").strip()
    task_name = str(payload.get("taskName") or "").strip()

    from app.services.training_node_service import (
        enrich_training_node_display_fields,
        resolve_training_node,
        validate_remote_node_for_job,
    )

    training_node = resolve_training_node(training_node_id=training_node_id) if training_node_id else None
    if training_node_id and training_node is None:
        if train_job_dir.is_dir():
            shutil.rmtree(train_job_dir, ignore_errors=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"未知训练节点: {training_node_id}",
        )
    if training_node and training_node.execution_mode == "remote_ssh":
        try:
            validate_remote_node_for_job(training_node, allow_busy=True)
        except ValueError as exc:
            if train_job_dir.is_dir():
                shutil.rmtree(train_job_dir, ignore_errors=True)
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        device_label = training_node.device_label
    elif training_node and training_node.execution_mode == "local":
        device_label = training_node.device_label

    train_config = {
        "datasetId": manifest.get("datasetId") or payload.get("datasetId"),
        "datasetIds": manifest.get("datasetIds")
        or payload.get("datasetIds")
        or ([manifest.get("datasetId")] if manifest.get("datasetId") else None),
        "datasetName": manifest.get("datasetName"),
        "datasetManifestPath": payload.get("datasetManifestPath"),
        "modelTypeId": payload.get("modelTypeId"),
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
        "trainingNodeId": training_node.node_id if training_node else None,
        "executionMode": training_node.execution_mode if training_node else "local",
        "seed": int(payload.get("seed") if payload.get("seed") is not None else 1),
        "seedMode": payload.get("seedMode"),
        "advancedEnabled": bool(adaptation_fields.get("advancedEnabled") or payload.get("advancedEnabled")),
        "modelParams": adaptation_fields.get("modelParams") or payload.get("modelParams"),
        "pretrained": payload.get("pretrained"),
        "saveFinal": bool(payload.get("saveFinal", adaptation_fields.get("saveFinal", True))),
        "saveBest": bool(payload.get("saveBest", adaptation_fields.get("saveBest", False))),
        "checkpointIntervalEpochs": payload.get("checkpointIntervalEpochs")
        if payload.get("checkpointIntervalEpochs") is not None
        else adaptation_fields.get("checkpointIntervalEpochs"),
        "capabilities": capabilities,
        "createdAt": _now_label(),
        "adaptationSnapshot": adaptation_snapshot,
    }
    for optional_key in (
        "architectureConfig",
        "dataLoaderConfig",
        "normalizationConfig",
        "inputConfig",
        "outputConfig",
        "dpConfigPath",
        "dpConfig",
        "actConfigPath",
        "actConfig",
        "pi0ConfigPath",
        "pi0Config",
        "openpiPlatformConfigPath",
    ):
        if adaptation_fields.get(optional_key) is not None:
            train_config[optional_key] = adaptation_fields[optional_key]
        elif payload.get(optional_key) is not None:
            train_config[optional_key] = payload.get(optional_key)
    if task_name:
        train_config["taskName"] = task_name
    if merged_hdf5_paths:
        train_config["datasetHdf5Paths"] = [str(path) for path in merged_hdf5_paths]
    elif manifest.get("artifacts", {}).get("hdf5Paths"):
        train_config["datasetHdf5Paths"] = list(manifest["artifacts"]["hdf5Paths"])

    defn = payload.get("_modelTypeDefinition")
    if isinstance(defn, dict):
        train_config["_modelTypeDefinition"] = defn
    from app.services.model_type_traceability import build_model_type_traceability_fields

    train_config.update(build_model_type_traceability_fields(train_config, model_type_definition=defn if isinstance(defn, dict) else None))

    from app.services.pi0_lerobot_loader import resolve_lerobot_path_from_manifest, validate_lerobot_for_pi0

    native_lerobot_path = resolve_lerobot_path_from_manifest(manifest)
    if selected_backend == "pi0" and native_lerobot_path is not None:
        ok, reason = validate_lerobot_for_pi0(native_lerobot_path)
        if not ok:
            if train_job_dir.is_dir():
                shutil.rmtree(train_job_dir, ignore_errors=True)
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=reason)
        artifacts = dict(manifest.get("artifacts") or {})
        artifacts["lerobot"] = str(native_lerobot_path)
        artifacts["lerobotPath"] = str(native_lerobot_path)
        manifest["artifacts"] = artifacts
        manifest["dataFormat"] = "lerobot"
        manifest["primaryFormat"] = "lerobot"
        manifest["availableFormats"] = manifest.get("availableFormats") or ["lerobot"]
        _write_json(train_job_dir / "artifacts" / "dataset_manifest.json", manifest)

        from app.services.pi0_lerobot_loader import inspect_lerobot_dataset

        spec = inspect_lerobot_dataset(native_lerobot_path)
        train_config["datasetFormat"] = "lerobot"
        train_config["dataFormat"] = "LeRobot"
        train_config["selectedDataFormat"] = "LeRobot"
        train_config["datasetPath"] = str(native_lerobot_path)
        train_config["maxSteps"] = int(payload.get("maxSteps") or payload.get("smokeSteps") or 10)
        train_config["smokeSteps"] = train_config["maxSteps"]
        train_config["taskInstruction"] = str(
            payload.get("taskInstruction") or spec.task_instruction or manifest.get("taskDescription") or ""
        ).strip()
        train_config["policyType"] = "pi0"
        train_config["robot"] = spec.robot
        train_config["stateDim"] = spec.state_dim
        train_config["actionDim"] = spec.action_dim
        train_config["controllerType"] = spec.controller_type
        train_config["actionMode"] = spec.action_mode
        train_config["actionRepresentation"] = spec.action_representation
    elif selected_backend == "pi0" and hdf5_path is None:
        if train_job_dir.is_dir():
            shutil.rmtree(train_job_dir, ignore_errors=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="pi0 训练需要 pi0Ready LeRobot 数据集或 HDF5 数据集",
        )
    elif selected_backend == "pi0" and hdf5_path is not None:
        adaptation = adaptation_snapshot.get("modelAdaptation") or {}
        input_cfg = adaptation.get("inputConfig") or {}
        camera_keys = list(input_cfg.get("camera_keys") or [])
        low_dim_keys = list(input_cfg.get("low_dim_keys") or [])
        from app.services.pi0_hdf5_converter import convert_hdf5_to_lerobot_index
        from app.services.pi0_training_runner import resolve_task_prompt

        lerobot_root = convert_hdf5_to_lerobot_index(
            hdf5_path=hdf5_path,
            output_dir=train_job_dir / "artifacts",
            manifest=manifest,
            camera_keys=camera_keys,
            low_dim_keys=low_dim_keys,
            task_prompt=resolve_task_prompt(manifest),
        )
        artifacts = dict(manifest.get("artifacts") or {})
        artifacts["lerobot"] = str(lerobot_root)
        artifacts["lerobotPath"] = str(lerobot_root)
        manifest["artifacts"] = artifacts
        manifest["dataFormat"] = manifest.get("dataFormat") or "platform_lerobot_export_v1"
        _write_json(train_job_dir / "artifacts" / "dataset_manifest.json", manifest)

        from app.services.pi0_training_runner import validate_pi0_dataset

        ok, reason = validate_pi0_dataset(hdf5_path, train_config, manifest=manifest)
        if not ok:
            if train_job_dir.is_dir():
                shutil.rmtree(train_job_dir, ignore_errors=True)
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=reason)

    pretrained_raw = train_config.get("pretrained")
    if isinstance(pretrained_raw, dict) and (
        pretrained_raw.get("modelAssetId") or pretrained_raw.get("checkpointPath")
    ):
        normalized_pretrained = _validate_pretrained_model(
            pretrained=pretrained_raw,
            resolved_backend=selected_backend,
            manifest=manifest,
            train_config=train_config,
        )
        _apply_pretrained_fields_to_train_config(train_config, normalized_pretrained)

    train_config = enrich_training_node_display_fields(train_config)
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
        enrich_training_node_display_fields(
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
                "trainingNodeId": train_config.get("trainingNodeId"),
                "trainingNodeDisplayName": train_config.get("trainingNodeDisplayName"),
                "executionMode": train_config.get("executionMode"),
                "createdAt": train_config["createdAt"],
            },
            train_config=train_config,
        ),
    )

    isaac_manifest = _is_isaac_block_stacking_manifest(manifest)
    record_workspace_job_start(
        job_id=train_job_id,
        job_type="training",
        task_type=str(
            manifest.get("taskType")
            or ("isaac_block_stacking" if isaac_manifest else "dual_arm_cable_manipulation" if _is_dual_arm_manifest(manifest) else "unknown")
        ),
        runtime_path=str(train_job_dir),
        runner=(
            "pi0_lerobot_smoke_runner.py"
            if selected_backend == "pi0" and native_lerobot_path is not None
            else "run_openpi_train.py"
            if selected_backend == "pi0"
            else "train_dp.py"
            if selected_backend == "diffusion_policy"
            else "train_act.py"
            if selected_backend == "act"
            else "isaac_robomimic/train.py"
            if selected_backend == "isaac_robomimic_bc"
            else "train_bc.py"
        ),
        status="pending",
        task_name=task_name or manifest.get("datasetName") or None,
        metadata={
            "datasetId": manifest.get("datasetId"),
            "datasetName": manifest.get("datasetName"),
            "modelTypeId": train_config.get("modelTypeId"),
            "downstreamModelType": train_config["downstreamModelType"],
            "trainingBackend": train_config["trainingBackend"],
            "taskName": task_name or None,
            "trainConfig": train_config,
            "adaptationSnapshot": train_config.get("adaptationSnapshot"),
            "trainingNodeId": train_config.get("trainingNodeId"),
            "trainingNodeDisplayName": train_config.get("trainingNodeDisplayName"),
            "executionMode": train_config.get("executionMode"),
        },
    )

    from app.services.training_job_sync_service import sync_training_job_from_runtime

    sync_training_job_from_runtime(train_job_id)

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


def _status_payload(train_job_id: str, *, sync_db: bool = False) -> dict[str, Any]:
    validated = _validate_train_job_id(train_job_id)
    from app.services.training_job_sync_service import get_training_job_summary_from_db, sync_training_job_from_runtime
    from app.services.training_job_status import enrich_and_persist_training_job_status, enrich_training_job_status, normalize_api_training_status
    from app.services.training_metrics import normalized_training_metrics
    from app.services.training_node_service import enrich_training_node_display_fields

    train_job_dir = _train_job_dir(validated)
    status_data = _read_json(train_job_dir / "status.json") if train_job_dir.is_dir() else {}

    if train_job_dir.is_dir() and status_data:
        status_data = _reconcile_remote_training_on_read(validated, train_job_dir, status_data)
        status_data = _reconcile_stale_running_training_job(validated, train_job_dir, status_data)
        status_data = enrich_and_persist_training_job_status(validated, train_job_dir, status_data)

        enriched = enrich_training_job_status(train_job_dir, status_data)
        normalized = normalized_training_metrics(train_job_dir, enriched)
        payload = dict(enriched)
        payload.setdefault("trainJobId", validated)
        payload["epoch"] = normalized.get("epoch", payload.get("epoch"))
        payload["totalEpochs"] = normalized.get("totalEpochs", payload.get("totalEpochs"))
        payload["loss"] = normalized.get("loss", payload.get("loss"))
        payload["progress"] = normalized.get("progress", payload.get("progress"))
        payload["lossHistory"] = normalized.get("lossSeries", [])
        payload["status"] = normalize_api_training_status(str(payload.get("status") or ""))
        train_config = _read_json(train_job_dir / "config" / "train_config.json")
        payload = enrich_training_node_display_fields(payload, train_config=train_config)
        payload.setdefault("device", train_config.get("device"))
        payload.setdefault("trainingNodeId", train_config.get("trainingNodeId"))
        if train_config.get("taskName"):
            payload.setdefault("taskName", train_config.get("taskName"))
        if sync_db:
            try:
                sync_training_job_from_runtime(validated)
            except Exception as exc:
                logger.warning("training status sync_db failed trainJobId=%s: %s", validated, exc)
        return payload

    if train_job_dir.is_dir() and sync_db:
        sync_training_job_from_runtime(validated)

    db_summary = get_training_job_summary_from_db(validated)
    if db_summary and str(db_summary.get("status") or "").lower() in {
        "completed",
        "failed",
        "canceled",
        "queued",
        "starting",
        "running",
        "backend_unavailable",
    }:
        payload = dict(db_summary)
        payload.setdefault("trainJobId", validated)
        payload["lossHistory"] = db_summary.get("lossHistory") or []
        from app.services.training_job_status import normalize_api_training_status

        payload["status"] = normalize_api_training_status(str(payload.get("status") or ""))
        return payload

    status_data = _read_json(train_job_dir / "status.json")
    if not status_data and db_summary:
        payload = dict(db_summary)
        payload.setdefault("trainJobId", validated)
        payload["lossHistory"] = db_summary.get("lossHistory") or []
        return payload
    if not status_data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Training job not found")

    from app.services.training_metrics import normalized_training_metrics

    normalized = normalized_training_metrics(train_job_dir, status_data)
    if int(normalized.get("epoch") or 0) > int(status_data.get("epoch") or 0):
        status_data["epoch"] = normalized.get("epoch")
    if normalized.get("loss") is not None:
        status_data["loss"] = normalized.get("loss")
    if normalized.get("progress") is not None:
        status_data["progress"] = normalized.get("progress")
    status_data["lossHistory"] = normalized.get("lossSeries", [])

    status_data = _reconcile_stale_running_training_job(validated, train_job_dir, status_data)
    from app.services.training_job_status import enrich_and_persist_training_job_status

    status_data = enrich_and_persist_training_job_status(validated, train_job_dir, status_data)
    train_config = _read_json(train_job_dir / "config" / "train_config.json")

    status_data.setdefault("trainJobId", validated)
    status_data.setdefault("device", train_config.get("device"))
    status_data.setdefault("deviceLabel", train_config.get("deviceLabel") or TRAINING_DEVICE_LABEL)
    if train_config.get("taskName"):
        status_data.setdefault("taskName", train_config.get("taskName"))
    from app.services.training_job_status import normalize_api_training_status

    status_data["status"] = normalize_api_training_status(str(status_data.get("status") or ""))
    return status_data


def get_training_job_status(train_job_id: str) -> dict[str, Any]:
    return _status_payload(train_job_id)


def read_training_job_log(train_job_id: str, lines: int | None = None) -> str:
    from app.services.training_metrics import resolve_training_log_for_display, sanitize_training_log_for_display
    from app.services.workspace_runtime_paths import resolve_training_job_root

    candidate = (train_job_id or "").strip()
    train_job_dir = resolve_training_job_root(candidate)
    if train_job_dir is None:
        return ""
    log_path = resolve_training_log_for_display(train_job_dir)
    if not log_path.is_file():
        pipeline_log = train_job_dir.parent / "train.log"
        if pipeline_log.is_file():
            log_path = pipeline_log
        else:
            return ""
    try:
        content = sanitize_training_log_for_display(
            log_path.read_text(encoding="utf-8", errors="replace")
        ).splitlines()
        if lines is not None and lines > 0:
            return "\n".join(content[-lines:])
        return "\n".join(content)
    except OSError:
        return ""


def get_training_job_model(train_job_id: str) -> dict[str, Any]:
    from app.services.model_asset_db_service import get_model_asset_from_db
    from app.services.workspace_runtime_paths import resolve_training_job_root

    candidate = (train_job_id or "").strip()
    train_job_dir = resolve_training_job_root(candidate)
    status_data = _read_json(train_job_dir / "status.json") if train_job_dir else {}
    model_manifest_path = train_job_dir / "artifacts" / "model_manifest.json" if train_job_dir else Path()
    model_manifest = _read_json(model_manifest_path) if model_manifest_path.is_file() else None

    model_asset_id = str(status_data.get("modelAssetId") or "").strip()
    if not model_asset_id:
        try:
            from app.core.database import SessionLocal
            from app.models.workspace_index import ModelAsset

            with SessionLocal() as db:
                row = (
                    db.query(ModelAsset)
                    .filter(ModelAsset.train_job_id == candidate, ModelAsset.status != "deleted")
                    .order_by(ModelAsset.updated_at.desc())
                    .first()
                )
                if row is not None:
                    model_asset_id = row.model_asset_id
                    if not model_manifest and isinstance(row.manifest_json, dict):
                        model_manifest = dict(row.manifest_json)
        except Exception:
            pass

    db_asset = get_model_asset_from_db(model_asset_id) if model_asset_id else None
    checkpoint_path = status_data.get("checkpointPath") or (db_asset or {}).get("checkpointPath")
    ready = bool(
        (status_data.get("status") == "completed" or status_data.get("checkpointExists"))
        and (checkpoint_path or db_asset)
        and (
            not model_manifest
            or model_manifest.get("status") in {None, "ready", "available"}
            or db_asset
        )
    )

    return {
        "trainJobId": candidate,
        "ready": ready,
        "modelManifest": model_manifest or db_asset,
        "checkpointPath": checkpoint_path,
        "modelAssetId": model_asset_id or None,
    }


def list_training_jobs() -> list[dict[str, Any]]:
    from app.services.training_job_sync_service import (
        list_training_jobs_from_db,
        reindex_runtime_jobs,
        sync_training_job_from_runtime,
    )

    rows = list_training_jobs_from_db(sync_stale=True)
    if rows:
        return rows

    jobs_root = TRAINING_ROOT / "jobs"
    if not jobs_root.is_dir():
        return []

    from app.services.training_metrics import normalized_training_metrics

    for job_dir in sorted(jobs_root.iterdir(), reverse=True):
        if job_dir.is_dir() and TRAIN_JOB_ID_PATTERN.match(job_dir.name):
            sync_training_job_from_runtime(job_dir.name)

    rows = list_training_jobs_from_db(sync_stale=False)
    if rows:
        return rows

    reindex_runtime_jobs(job_type="training", dry_run=False)
    rows = list_training_jobs_from_db(sync_stale=False)
    if rows:
        return rows

    legacy_rows: list[dict[str, Any]] = []
    for job_dir in sorted(jobs_root.iterdir(), reverse=True):
        if not job_dir.is_dir():
            continue
        status_data = _read_json(job_dir / "status.json")
        if not status_data or is_job_deleted(status_data):
            continue

        from app.services.training_job_status import enrich_and_persist_training_job_status

        train_job_id = str(status_data.get("trainJobId") or job_dir.name)
        status_data = enrich_and_persist_training_job_status(train_job_id, job_dir, status_data)
        normalized = normalized_training_metrics(job_dir, status_data)
        epoch = int(normalized.get("epoch") or status_data.get("epoch") or 0)
        total_epochs = int(status_data.get("totalEpochs") or 0)
        loss = normalized.get("loss", status_data.get("loss"))
        progress = normalized.get("progress", status_data.get("progress"))

        legacy_rows.append(
            {
                "trainJobId": status_data.get("trainJobId") or job_dir.name,
                "status": status_data.get("status") or "queued",
                "datasetId": status_data.get("datasetId"),
                "datasetName": status_data.get("datasetName"),
                "downstreamModelType": status_data.get("downstreamModelType"),
                "trainingBackend": status_data.get("trainingBackend"),
                "createdAt": status_data.get("createdAt"),
                "updatedAt": status_data.get("updatedAt"),
                "checkpointExists": bool(status_data.get("checkpointExists")),
                "modelAssetId": status_data.get("modelAssetId"),
                "epoch": epoch,
                "totalEpochs": total_epochs,
                "loss": loss,
                "progress": progress,
                "lossHistory": normalized.get("lossSeries", []),
                "message": status_data.get("message"),
                "dataFormat": status_data.get("dataFormat"),
                "deviceLabel": status_data.get("deviceLabel"),
                "taskName": status_data.get("taskName"),
            }
        )
    return legacy_rows


def hard_delete_training_job_disk_only(train_job_id: str) -> bool:
    """Hard-delete a training job directory when no workspace_jobs row exists.

    Returns True if a runtime directory was removed.
    Path safety is enforced by _resolve_train_job_dir (must be under training/jobs).
    """
    sanitized = _sanitize_train_job_id_for_delete(train_job_id)
    train_job_dir = _resolve_train_job_dir(sanitized)
    if train_job_dir is None or not train_job_dir.is_dir():
        return False
    stop_training_job_if_active(sanitized)
    shutil.rmtree(train_job_dir)
    return True


def delete_training_job(train_job_id: str) -> dict[str, Any]:
    """Hard-delete training job: DB rows, artifacts, model assets, and runtime directory.

    Prefer the async API path (DELETE /workspace/training/jobs/{id}) which reuses
    delete_workspace_job_async. This sync helper remains for scripts/tests.
    """
    from datetime import timezone

    from app.core.database import SessionLocal
    from app.models.workspace_index import ModelAsset, TrainingMetricSummary
    from app.models.workspace_job import WorkspaceArtifact, WorkspaceJob
    from app.services.workspace_job_service import (
        RuntimeDeleteFailedError,
        _delete_runtime_job_directory,
    )
    from app.services.workspace_model_asset_list_cache import invalidate_model_asset_list_cache

    sanitized = _sanitize_train_job_id_for_delete(train_job_id)
    deleted_at = datetime.now(timezone.utc).isoformat()
    stop_training_job_if_active(sanitized)

    try:
        with SessionLocal() as db:
            row = db.query(WorkspaceJob).filter(WorkspaceJob.job_id == sanitized).one_or_none()
            if row is not None:
                runtime_path = row.runtime_path or ""
                try:
                    runtime_deleted, reason = _delete_runtime_job_directory(sanitized, runtime_path)
                except RuntimeDeleteFailedError as exc:
                    logger.warning(
                        "delete_training_job runtime failed, continue db delete: job_id=%s error=%s",
                        sanitized,
                        exc.message,
                    )
                    runtime_deleted, reason = False, exc.message

                deleted_model_assets = (
                    db.query(ModelAsset)
                    .filter(ModelAsset.train_job_id == sanitized)
                    .delete(synchronize_session=False)
                )
                db.query(TrainingMetricSummary).filter(
                    TrainingMetricSummary.job_id == sanitized
                ).delete(synchronize_session=False)
                deleted_artifacts = (
                    db.query(WorkspaceArtifact)
                    .filter(WorkspaceArtifact.job_id == sanitized)
                    .delete(synchronize_session=False)
                )
                db.delete(row)
                db.commit()
                invalidate_model_asset_list_cache()
                result: dict[str, Any] = {
                    "trainJobId": sanitized,
                    "deleted": True,
                    "deletedAt": deleted_at,
                    "runtimeDeleted": runtime_deleted,
                    "deletedArtifacts": int(deleted_artifacts or 0),
                    "deletedModelAssets": int(deleted_model_assets or 0),
                }
                if reason:
                    result["reason"] = reason
                return result
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("delete_training_job db hard-delete failed job_id=%s: %s", sanitized, exc)

    if hard_delete_training_job_disk_only(sanitized):
        invalidate_model_asset_list_cache()
        return {
            "trainJobId": sanitized,
            "deleted": True,
            "deletedAt": deleted_at,
            "runtimeDeleted": True,
        }

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="training job not found",
    )
