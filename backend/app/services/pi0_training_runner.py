"""openpi / pi0 training runner: probe, dataset adapter, config generation, command builder."""

from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
PI0_RUNNER_DIR = PROJECT_ROOT / "backend" / "integrations" / "pi0_runner"
PI0_RUNNER_SCRIPT = PI0_RUNNER_DIR / "run_openpi_train.py"
PI0_PROBE_SCRIPT = PI0_RUNNER_DIR / "probe_openpi_env.py"

PI0_RUNNER_DISABLED_REASON = "openpi 环境未配置，无法训练 pi0（请设置 PI0_RUNNER_ENABLED、OPENPI_ROOT、OPENPI_PYTHON）"
PI0_HDF5_NOT_SUPPORTED_MESSAGE = (
    "pi0 当前需要 LeRobot/HF 格式数据，平台 HDF5 转换器尚未接入。"
)
FAKE_DATA_OPENPI_CONFIGS = frozenset({"debug", "debug_pi05", "debug_restore", "pi0_mock"})
DEBUG_PI05_SMOKE_CONFIG = "debug_pi05"
DEBUG_PI05_SMOKE_LOG_MESSAGE = "debug_pi05 smoke mode: skip platform dataset conversion"
LEROBOT_OPENPI_CONFIG_MARKERS = ("libero", "lerobot", "aloha", "droid")


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def get_pi0_env() -> dict[str, str]:
    return {
        "enabled": os.environ.get("PI0_RUNNER_ENABLED", "").strip(),
        "openpi_root": os.environ.get("OPENPI_ROOT", "").strip(),
        "openpi_python": os.environ.get("OPENPI_PYTHON", "").strip(),
        "openpi_base_config": os.environ.get("OPENPI_BASE_CONFIG", "").strip(),
        "openpi_train_script": os.environ.get("OPENPI_TRAIN_SCRIPT", "").strip(),
    }


def _resolve_openpi_python_command() -> Optional[list[str]]:
    raw = os.environ.get("OPENPI_PYTHON", "").strip()
    if not raw:
        return None
    parts = shlex.split(raw)
    if not parts:
        return None
    executable = Path(parts[0]).expanduser()
    if executable.is_file() and os.access(executable, os.X_OK):
        return [str(executable.resolve()), *parts[1:]]
    return parts


def _find_openpi_train_script(root: Path) -> Optional[Path]:
    candidates: list[Path] = [
        root / "scripts" / "train.py",
        root / "src" / "openpi" / "training" / "train.py",
    ]
    override = os.environ.get("OPENPI_TRAIN_SCRIPT", "").strip()
    if override:
        override_path = Path(override).expanduser()
        if override_path.is_file():
            try:
                override_path.resolve().relative_to(root.resolve())
            except ValueError:
                pass
            else:
                candidates.insert(0, override_path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def probe_pi0_training_capability(*, timeout_sec: float = 30.0) -> dict[str, Any]:
    """Probe openpi runtime. Returns {ready, reason, evidence}."""
    evidence: list[str] = []

    if not _env_truthy("PI0_RUNNER_ENABLED"):
        return {
            "ready": False,
            "reason": "PI0_RUNNER_ENABLED 未启用",
            "evidence": evidence,
        }

    if not PI0_RUNNER_SCRIPT.is_file():
        return {
            "ready": False,
            "reason": f"平台 pi0 runner 脚本缺失: {PI0_RUNNER_SCRIPT}",
            "evidence": evidence,
        }
    evidence.append(str(PI0_RUNNER_SCRIPT))

    openpi_root = os.environ.get("OPENPI_ROOT", "").strip()
    if not openpi_root:
        return {"ready": False, "reason": PI0_RUNNER_DISABLED_REASON, "evidence": evidence}

    root_path = Path(openpi_root).expanduser()
    if not root_path.is_dir():
        return {
            "ready": False,
            "reason": f"OPENPI_ROOT 不存在: {openpi_root}",
            "evidence": evidence,
        }
    evidence.append(str(root_path.resolve()))

    python_cmd = _resolve_openpi_python_command()
    if not python_cmd:
        return {"ready": False, "reason": PI0_RUNNER_DISABLED_REASON, "evidence": evidence}

    if not PI0_PROBE_SCRIPT.is_file():
        return {
            "ready": False,
            "reason": f"pi0 probe 脚本缺失: {PI0_PROBE_SCRIPT}",
            "evidence": evidence,
        }

    env = os.environ.copy()
    env["OPENPI_ROOT"] = str(root_path.resolve())
    cmd = [*python_cmd, str(PI0_PROBE_SCRIPT)]
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(root_path),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("pi0 openpi probe failed: %s", exc)
        return {
            "ready": False,
            "reason": PI0_RUNNER_DISABLED_REASON,
            "evidence": evidence,
        }

    stdout = (completed.stdout or "").strip()
    stderr = (completed.stderr or "").strip()
    if completed.returncode == 0 and "OPENPI_PROBE_OK" in stdout:
        evidence.append(stdout.splitlines()[-1] if stdout else "OPENPI_PROBE_OK")
        train_script = _find_openpi_train_script(root_path)
        if train_script:
            evidence.append(str(train_script))
        return {"ready": True, "reason": None, "evidence": evidence}

    detail = stderr or stdout or f"exit={completed.returncode}"
    logger.info("pi0 probe not ready: %s", detail)
    return {
        "ready": False,
        "reason": PI0_RUNNER_DISABLED_REASON,
        "evidence": evidence + ([detail] if detail else []),
    }


def is_debug_pi05_smoke_config(base_config: str | None = None) -> bool:
    resolved = (base_config or os.environ.get("OPENPI_BASE_CONFIG", "")).strip()
    return resolved == DEBUG_PI05_SMOKE_CONFIG


def openpi_base_config_requires_lerobot(base_config: str) -> bool:
    if not base_config or base_config in FAKE_DATA_OPENPI_CONFIGS:
        return False
    lowered = base_config.lower()
    return any(marker in lowered for marker in LEROBOT_OPENPI_CONFIG_MARKERS)


def manifest_has_lerobot_dataset(manifest: dict[str, Any]) -> bool:
    from app.services.pi0_lerobot_loader import is_platform_lerobot_v3_dataset, resolve_lerobot_path_from_manifest

    resolved = resolve_lerobot_path_from_manifest(manifest)
    if resolved is not None and is_platform_lerobot_v3_dataset(resolved):
        return True

    artifacts = manifest.get("artifacts") or {}
    lerobot_path = artifacts.get("lerobot") or artifacts.get("lerobotPath")
    if lerobot_path:
        root = Path(str(lerobot_path)).expanduser()
        if is_platform_lerobot_v3_dataset(root):
            return True
        if (root / "meta" / "info.json").is_file() or (root / "dataset_index.json").is_file():
            data_dir = root / "data"
            if data_dir.is_dir() and (any(data_dir.rglob("episode_*")) or any(data_dir.rglob("*.parquet"))):
                return True
    dataset = manifest.get("dataset")
    if isinstance(dataset, dict) and (dataset.get("lerobotPath") or dataset.get("repo_id")):
        return True
    fmt = str(manifest.get("dataFormat") or manifest.get("format") or manifest.get("primaryFormat") or "").lower()
    if fmt == "lerobot":
        return resolved is not None or bool(lerobot_path)
    return "lerobot" in fmt or fmt in {"hf", "huggingface", "lerobot_index", "platform_lerobot_export_v1"}


def validate_openpi_dataset_readiness(
    *,
    base_config: str,
    manifest: dict[str, Any],
    hdf5_path: Optional[Path] = None,
) -> tuple[bool, str]:
    """Ensure openpi base config can consume the platform dataset format."""
    if not openpi_base_config_requires_lerobot(base_config):
        return True, ""
    if manifest_has_lerobot_dataset(manifest):
        return True, ""
    if hdf5_path is not None and hdf5_path.is_file():
        return False, (
            f"{PI0_HDF5_NOT_SUPPORTED_MESSAGE} "
            "请先完成 HDF5→LeRobot 转换，或选择 debug_pi05 / pi0_mock 等 smoke 配置。"
        )
    return False, PI0_HDF5_NOT_SUPPORTED_MESSAGE


def resolve_task_prompt(manifest: dict[str, Any]) -> str:
    for key in ("taskDescription", "taskPrompt", "languageInstruction", "instruction"):
        value = manifest.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    task_type = str(manifest.get("taskType") or manifest.get("taskName") or "").strip()
    if task_type:
        return task_type.replace("_", " ")
    dataset_name = str(manifest.get("datasetName") or "").strip()
    return dataset_name or "complete the task"


def validate_pi0_dataset(
    hdf5_path: Path,
    train_config: dict[str, Any],
    *,
    manifest: Optional[dict[str, Any]] = None,
) -> tuple[bool, str]:
    pi0_config = train_config.get("pi0Config") or {}
    adaptation = (train_config.get("adaptationSnapshot") or {}).get("modelAdaptation") or {}
    input_cfg = adaptation.get("inputConfig") or {}
    architecture = adaptation.get("architectureConfig") or pi0_config.get("structure") or {}

    camera_keys = list(
        pi0_config.get("camera_keys")
        or input_cfg.get("camera_keys")
        or input_cfg.get("camera_names")
        or []
    )
    low_dim_keys = list(pi0_config.get("low_dim_keys") or input_cfg.get("low_dim_keys") or [])
    language_conditioning = bool(
        architecture.get("language_conditioning", pi0_config.get("language_conditioning", True))
    )

    if not camera_keys:
        return False, "pi0 需要图像观测键，但适配配置中 camera_keys 为空"

    if language_conditioning and not resolve_task_prompt(manifest or {}):
        return False, "pi0 启用 language_conditioning 但数据集缺少 task 描述 / taskType"

    try:
        import h5py
    except ImportError:
        return False, "h5py 不可用，无法校验 pi0 HDF5"

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
            missing = [key for key in camera_keys + low_dim_keys if key not in obs]
            if missing:
                return False, f"pi0 配置需要的 obs 键在 HDF5 中不存在: {', '.join(missing)}"
            for key in camera_keys:
                shape = obs[key].shape
                if len(shape) < 3:
                    return False, f"pi0 图像键 {key} 维度异常: shape={shape}"
    except OSError as exc:
        return False, f"无法读取 HDF5: {exc}"

    base_config = os.environ.get("OPENPI_BASE_CONFIG", "").strip()
    ok_lerobot, lerobot_reason = validate_openpi_dataset_readiness(
        base_config=base_config,
        manifest=manifest or {},
        hdf5_path=hdf5_path,
    )
    if not ok_lerobot:
        return False, lerobot_reason
    if is_debug_pi05_smoke_config(base_config):
        logger.info(DEBUG_PI05_SMOKE_LOG_MESSAGE)

    return True, ""


def build_pi0_dataset_index(
    *,
    train_job_dir: Path,
    hdf5_path: Path,
    manifest: dict[str, Any],
    pi0_config: dict[str, Any],
) -> Path:
    import h5py

    camera_keys = list(pi0_config.get("camera_keys") or [])
    low_dim_keys = list(pi0_config.get("low_dim_keys") or [])
    episodes: list[dict[str, Any]] = []

    with h5py.File(hdf5_path, "r") as handle:
        data_group = handle["data"]
        for demo_key in sorted(k for k in data_group.keys() if str(k).startswith("demo_")):
            demo = data_group[demo_key]
            actions = demo["actions"]
            obs = demo["obs"]
            episodes.append(
                {
                    "episodeId": str(demo_key),
                    "length": int(actions.shape[0]),
                    "actionDim": int(actions.shape[-1]) if actions.ndim > 1 else 1,
                    "cameraKeys": camera_keys,
                    "lowDimKeys": low_dim_keys,
                    "obsKeys": sorted(obs.keys()),
                }
            )

    payload = {
        "format": "platform_hdf5_v1",
        "datasetId": manifest.get("datasetId"),
        "datasetName": manifest.get("datasetName"),
        "hdf5Path": str(hdf5_path),
        "taskPrompt": resolve_task_prompt(manifest),
        "episodes": episodes,
        "cameraKeys": camera_keys,
        "lowDimKeys": low_dim_keys,
    }
    out_path = train_job_dir / "artifacts" / "pi0_dataset_index.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def build_pi0_config_dict(
    *,
    train_job_dir: Path,
    manifest: dict[str, Any],
    train_config: dict[str, Any],
    model_adaptation: dict[str, Any],
    dataset_index_path: Path,
    hdf5_path: Path,
) -> dict[str, Any]:
    input_cfg = model_adaptation.get("inputConfig") or {}
    output_cfg = model_adaptation.get("outputConfig") or {}
    architecture = model_adaptation.get("architectureConfig") or {}
    advanced = model_adaptation.get("advancedConfig") or {}
    structure = {**architecture, **advanced}

    epochs = int(train_config.get("epochs") or 5)
    batch_size = int(train_config.get("batchSize") or 8)
    learning_rate = float(train_config.get("learningRate") or 1e-4)
    seed = int(train_config.get("seed") if train_config.get("seed") is not None else 1)
    steps_per_epoch = max(1, int(structure.get("steps_per_epoch") or 4))
    num_train_steps = int(structure.get("num_train_steps") or epochs * steps_per_epoch)
    openpi_base_config = os.environ.get("OPENPI_BASE_CONFIG", "").strip() or None
    if is_debug_pi05_smoke_config(openpi_base_config):
        num_train_steps = min(num_train_steps, 4)
        batch_size = min(batch_size, 2)

    backend_out = train_job_dir / "checkpoints" / "pi0"
    platform_final_checkpoint = backend_out / "checkpoints" / "model_final.pt"
    metrics_path = train_job_dir / "artifacts" / "metrics.jsonl"
    exp_name = train_job_dir.name

    pi0_config = {
        "camera_keys": list(input_cfg.get("camera_keys") or []),
        "low_dim_keys": list(input_cfg.get("low_dim_keys") or []),
        "language_conditioning": bool(structure.get("language_conditioning", True)),
        "structure": {
            "context_window": int(structure.get("context_window") or 256),
            "action_horizon": int(structure.get("action_horizon") or output_cfg.get("action_horizon") or 16),
            "vision_encoder": str(structure.get("vision_encoder") or "siglip"),
            "language_conditioning": bool(structure.get("language_conditioning", True)),
            "action_head": str(structure.get("action_head") or "flow_matching"),
            "tokenizer_or_processor": str(structure.get("tokenizer_or_processor") or "default"),
        },
    }

    return {
        "platform": {
            "trainJobId": train_job_dir.name,
            "backend": "pi0",
            "modelTypeId": train_config.get("modelTypeId"),
        },
        "openpi_base_config": openpi_base_config,
        "smoke_mode": {
            "enabled": is_debug_pi05_smoke_config(openpi_base_config),
            "message": DEBUG_PI05_SMOKE_LOG_MESSAGE if is_debug_pi05_smoke_config(openpi_base_config) else None,
        },
        "openpi": {
            "exp_name": exp_name,
            "checkpoint_base_dir": str(backend_out),
            "platform_final_checkpoint": str(platform_final_checkpoint),
        },
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "seed": seed,
            "steps_per_epoch": steps_per_epoch,
            "num_train_steps": num_train_steps,
        },
        "structure": pi0_config["structure"],
        "dataset": {
            "hdf5_path": str(hdf5_path),
            "dataset_index_path": str(dataset_index_path),
            "lerobot_path": str(
                (manifest.get("artifacts") or {}).get("lerobotPath")
                or (manifest.get("artifacts") or {}).get("lerobot")
                or ""
            ),
            "task_prompt": resolve_task_prompt(manifest),
            "camera_keys": pi0_config["camera_keys"],
            "low_dim_keys": pi0_config["low_dim_keys"],
            "action_dim": int(output_cfg.get("action_dim") or 0),
            "action_horizon": int(output_cfg.get("action_horizon") or pi0_config["structure"]["action_horizon"]),
        },
        "manifest": {
            "datasetId": manifest.get("datasetId"),
            "dataFormat": manifest.get("dataFormat"),
            "format": manifest.get("format"),
            "artifacts": manifest.get("artifacts") or {},
        },
        "paths": {
            "output_dir": str(backend_out),
            "checkpoint_base_dir": str(backend_out),
            "platform_final_checkpoint": str(platform_final_checkpoint),
            "metrics_path": str(metrics_path),
            "train_log_path": str(train_job_dir / "logs" / "train.log"),
        },
        "save_policy": {
            "saveFinal": bool(train_config.get("saveFinal", True)),
            "saveBest": bool(train_config.get("saveBest", False)),
            "checkpointIntervalEpochs": train_config.get("checkpointIntervalEpochs"),
        },
    }


def write_pi0_config_yaml(train_job_dir: Path, config: dict[str, Any]) -> Path:
    path = train_job_dir / "config" / "openpi_platform_config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml  # type: ignore

        path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    except ImportError:
        path = train_job_dir / "config" / "openpi_platform_config.json"
        path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def build_pi0_train_command(
    *,
    hdf5_path: Path,
    out_dir: Path,
    train_config: dict[str, Any],
    train_job_dir: Path,
) -> list[str]:
    if not PI0_RUNNER_SCRIPT.is_file():
        raise FileNotFoundError(str(PI0_RUNNER_SCRIPT))

    platform_config = train_config.get("pi0ConfigPath") or train_config.get("openpiPlatformConfigPath")
    if not platform_config or not Path(str(platform_config)).is_file():
        raise FileNotFoundError("pi0ConfigPath missing or not found")

    python_cmd = _resolve_openpi_python_command()
    if not python_cmd:
        raise FileNotFoundError("OPENPI_PYTHON is not configured or not executable")

    metrics_path = train_job_dir / "artifacts" / "metrics.jsonl"
    return [
        *python_cmd,
        str(PI0_RUNNER_SCRIPT),
        "--platform-config",
        str(platform_config),
        "--dataset",
        str(hdf5_path),
        "--out-dir",
        str(out_dir),
        "--metrics-path",
        str(metrics_path),
    ]


def build_pi0_env(*, train_job_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env["PI0_TRAIN_MODE"] = "openpi"
    openpi_root = os.environ.get("OPENPI_ROOT", "").strip()
    if openpi_root:
        env["OPENPI_ROOT"] = str(Path(openpi_root).expanduser().resolve())
    return env


def prepare_pi0_job_artifacts(
    *,
    train_job_dir: Path,
    manifest: dict[str, Any],
    train_config: dict[str, Any],
    hdf5_path: Path | None = None,
) -> Path:
    """Build openpi platform config; use native LeRobot when available, else HDF5 converter."""
    adaptation = (train_config.get("adaptationSnapshot") or {}).get("modelAdaptation") or {}
    input_cfg = adaptation.get("inputConfig") or {}
    pi0_config = dict(train_config.get("pi0Config") or {})
    pi0_config.setdefault("camera_keys", list(input_cfg.get("camera_keys") or []))
    pi0_config.setdefault("low_dim_keys", list(input_cfg.get("low_dim_keys") or []))

    from app.services.pi0_lerobot_loader import resolve_lerobot_path_from_manifest, validate_lerobot_for_pi0

    native_lerobot = resolve_lerobot_path_from_manifest(manifest)
    lerobot_root: Path | None = None
    if native_lerobot is not None:
        ok, reason = validate_lerobot_for_pi0(native_lerobot)
        if not ok:
            raise ValueError(f"LeRobot 数据集不满足 pi0 要求: {reason}")
        lerobot_root = native_lerobot
        logger.info("pi0 prepare: using native LeRobot dataset at %s (skip HDF5 converter)", lerobot_root)
    elif hdf5_path is not None and hdf5_path.is_file():
        from app.services.pi0_hdf5_converter import convert_hdf5_to_lerobot_dataset

        lerobot_root = convert_hdf5_to_lerobot_dataset(
            hdf5_path=hdf5_path,
            output_dir=train_job_dir / "artifacts",
            manifest=manifest,
            camera_keys=list(pi0_config.get("camera_keys") or []),
            low_dim_keys=list(pi0_config.get("low_dim_keys") or []),
            task_prompt=resolve_task_prompt(manifest),
        )
        logger.info("pi0 prepare: converted HDF5 to LeRobot at %s", lerobot_root)
    else:
        raise ValueError("pi0 训练需要 LeRobot 数据集或 HDF5 源数据")

    artifacts = dict(manifest.get("artifacts") or {})
    artifacts["lerobot"] = str(lerobot_root)
    artifacts["lerobotPath"] = str(lerobot_root)
    manifest["artifacts"] = artifacts
    if native_lerobot is not None:
        manifest["dataFormat"] = "lerobot"
        manifest["primaryFormat"] = "lerobot"
        manifest["availableFormats"] = ["lerobot"]
    else:
        manifest["dataFormat"] = manifest.get("dataFormat") or "platform_lerobot_export_v1"
    (train_job_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    (train_job_dir / "artifacts" / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    dataset_index_path = train_job_dir / "artifacts" / "pi0_lerobot_dataset_index.json"
    if hdf5_path is not None and hdf5_path.is_file() and native_lerobot is None:
        dataset_index_path = build_pi0_dataset_index(
            train_job_dir=train_job_dir,
            hdf5_path=hdf5_path,
            manifest=manifest,
            pi0_config=pi0_config,
        )
    else:
        from app.services.pi0_lerobot_loader import build_smoke_schema_record, inspect_lerobot_dataset

        spec = inspect_lerobot_dataset(lerobot_root)
        dataset_index_path.parent.mkdir(parents=True, exist_ok=True)
        dataset_index_path.write_text(
            json.dumps(
                {
                    "format": "platform_lerobot_v3",
                    "lerobotPath": str(lerobot_root),
                    **build_smoke_schema_record(spec, dataset_path=lerobot_root),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    config_dict = build_pi0_config_dict(
        train_job_dir=train_job_dir,
        manifest=manifest,
        train_config=train_config,
        model_adaptation=adaptation,
        dataset_index_path=dataset_index_path,
        hdf5_path=hdf5_path or Path(""),
    )
    config_dict.setdefault("dataset", {})
    config_dict["dataset"]["lerobot_path"] = str(lerobot_root)
    config_dict["dataset"]["dataset_format"] = "lerobot" if native_lerobot else "platform_lerobot_export_v1"
    config_dict["dataset"]["hdf5_path"] = str(hdf5_path) if hdf5_path else ""
    config_path = write_pi0_config_yaml(train_job_dir, config_dict)
    train_config["pi0ConfigPath"] = str(config_path)
    train_config["openpiPlatformConfigPath"] = str(config_path)
    train_config["pi0Config"] = {
        **pi0_config,
        "structure": config_dict.get("structure") or {},
        "dataset_index_path": str(dataset_index_path),
    }
    config_file = train_job_dir / "config" / "train_config.json"
    config_file.write_text(json.dumps(train_config, ensure_ascii=False, indent=2), encoding="utf-8")
    return config_path
