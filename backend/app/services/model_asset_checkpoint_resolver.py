"""Resolve model asset / checkpoint hints to local on-disk checkpoint paths."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from app.core.platform_paths import platform_paths, resolve_runtime_reference
from app.services.model_asset_validation import (
    PROJECT_ROOT,
    check_checkpoint_file,
    normalize_backend_type,
    resolve_checkpoint_path_on_disk,
    resolve_model_asset_backend_type,
)

TRAINING_JOBS_ROOT = platform_paths.training_jobs

_STANDARD_CHECKPOINT_REL_PATHS = (
    "checkpoints/diffusion_policy/checkpoints/model_final.pt",
    "checkpoints/diffusion_policy/checkpoints/model.pth",
    "checkpoints/model_final.pt",
    "checkpoints/model.pth",
    "checkpoints/model_best.pt",
    "checkpoints/robomimic/model.pth",
    "checkpoints/act/checkpoints/model_final.pt",
)


def _read_json(path: Path) -> dict[str, Any]:
    from app.services.safe_file_io import safe_read_json

    return safe_read_json(path) or {}


def _is_remote_uri(text: str) -> bool:
    lowered = text.lower()
    return lowered.startswith("minio://") or lowered.startswith("s3://")


def _path_candidates(text: str) -> list[Path]:
    raw = str(text or "").strip()
    if not raw or _is_remote_uri(raw):
        return []
    if raw.startswith("file://"):
        raw = raw[len("file://") :]
    path = Path(raw)
    if path.is_absolute():
        return [path]
    resolved_runtime = resolve_runtime_reference(raw)
    return list(dict.fromkeys((resolved_runtime, PROJECT_ROOT / path, path)))


def _infer_train_job_id(
    *,
    asset: Optional[dict[str, Any]] = None,
    path_hint: Optional[str] = None,
) -> str:
    if asset:
        for key in ("sourceTrainingJobId", "sourceTrainJobId"):
            value = str(asset.get(key) or "").strip()
            if value:
                return value
    hint = str(path_hint or "")
    marker = "/training/jobs/"
    if marker in hint:
        tail = hint.split(marker, 1)[1]
        job_id = tail.split("/", 1)[0].strip()
        if job_id.startswith("train_"):
            return job_id
    for part in Path(hint).parts:
        if str(part).startswith("train_"):
            return str(part)
    return ""


def _registry_asset_paths(train_job_dir: Path, model_asset_id: str) -> list[Path]:
    registry = _read_json(train_job_dir / "artifacts" / "model_assets_registry.json")
    assets = registry.get("assets") if isinstance(registry.get("assets"), list) else []
    paths: list[Path] = []
    for entry in assets:
        if not isinstance(entry, dict):
            continue
        if model_asset_id and str(entry.get("modelAssetId") or entry.get("id") or "") != model_asset_id:
            continue
        for key in ("checkpointPath", "localCachePath", "artifactPath"):
            paths.extend(_path_candidates(str(entry.get(key) or "")))
    return paths


def _manifest_paths(train_job_dir: Path, model_asset_id: str) -> list[Path]:
    manifest = _read_json(train_job_dir / "artifacts" / "model_manifest.json")
    if not manifest:
        return []
    if model_asset_id:
        manifest_id = str(manifest.get("modelAssetId") or manifest.get("id") or "")
        if manifest_id and manifest_id != model_asset_id:
            return []
    paths: list[Path] = []
    for key in ("checkpointPath", "localCachePath", "artifactPath"):
        paths.extend(_path_candidates(str(manifest.get(key) or "")))
    return paths


def _train_config_checkpoint_paths(train_job_dir: Path) -> list[Path]:
    train_config = _read_json(train_job_dir / "config" / "train_config.json")
    paths: list[Path] = []
    for key in ("checkpointPath", "finalCheckpointPath"):
        paths.extend(_path_candidates(str(train_config.get(key) or "")))
    dp_config = train_config.get("dpConfig") if isinstance(train_config.get("dpConfig"), dict) else {}
    for key in ("checkpointPath", "finalCheckpointPath"):
        paths.extend(_path_candidates(str(dp_config.get(key) or "")))
    return paths


def _standard_job_checkpoint_paths(train_job_dir: Path) -> list[Path]:
    return [train_job_dir / rel for rel in _STANDARD_CHECKPOINT_REL_PATHS]


def iter_local_checkpoint_candidates(
    *,
    asset: Optional[dict[str, Any]] = None,
    path_hint: Optional[str] = None,
    model_asset_id: Optional[str] = None,
    train_job_id: Optional[str] = None,
) -> list[Path]:
    seen: set[str] = set()
    ordered: list[Path] = []

    def _add(path: Path) -> None:
        try:
            key = str(path.resolve())
        except OSError:
            key = str(path)
        if key in seen:
            return
        seen.add(key)
        ordered.append(path)

    asset = dict(asset or {})
    asset_id = str(model_asset_id or asset.get("id") or asset.get("modelAssetId") or "").strip()
    job_id = str(train_job_id or _infer_train_job_id(asset=asset, path_hint=path_hint)).strip()

    for key in (
        "checkpointPath",
        "artifactPath",
        "localCachePath",
        "manifestPath",
        "storageUri",
    ):
        for candidate in _path_candidates(str(asset.get(key) or "")):
            _add(candidate)

    manifest_json = asset.get("manifestJson") if isinstance(asset.get("manifestJson"), dict) else {}
    for key in ("checkpointPath", "localCachePath", "artifactPath"):
        for candidate in _path_candidates(str(manifest_json.get(key) or "")):
            _add(candidate)

    for candidate in _path_candidates(str(path_hint or "")):
        _add(candidate)

    if job_id:
        train_job_dir = next(
            (
                root / job_id
                for root in (TRAINING_JOBS_ROOT,)
                if (root / job_id).is_dir()
            ),
            None,
        )
        if train_job_dir is not None:
            for candidate in (
                *_manifest_paths(train_job_dir, asset_id),
                *_registry_asset_paths(train_job_dir, asset_id),
                *_train_config_checkpoint_paths(train_job_dir),
                *_standard_job_checkpoint_paths(train_job_dir),
            ):
                _add(candidate)

    return ordered


def resolve_local_checkpoint_path(
    *,
    asset: Optional[dict[str, Any]] = None,
    path_hint: Optional[str] = None,
    model_asset_id: Optional[str] = None,
    train_job_id: Optional[str] = None,
) -> Optional[str]:
    """Return the first existing local checkpoint path, or None."""
    for candidate in iter_local_checkpoint_candidates(
        asset=asset,
        path_hint=path_hint,
        model_asset_id=model_asset_id,
        train_job_id=train_job_id,
    ):
        exists, _size = check_checkpoint_file(str(candidate))
        if exists:
            try:
                return str(candidate.resolve())
            except OSError:
                return str(candidate)
    return None


def _load_checkpoint_payload(checkpoint_path: Path) -> dict[str, Any]:
    if not checkpoint_path.is_file():
        return {}
    suffix = checkpoint_path.suffix.lower()
    if suffix == ".json" or checkpoint_path.name.lower() in {
        "model_manifest.json",
        "train_config.json",
    }:
        from app.services.safe_file_io import safe_read_json

        return safe_read_json(checkpoint_path) or {}

    if suffix in {".pt", ".pth", ".ckpt", ""}:
        try:
            raw = checkpoint_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raw = ""
        except OSError:
            raw = ""
        if raw.lstrip().startswith("{"):
            try:
                payload = json.loads(raw)
                if isinstance(payload, dict):
                    return payload
            except json.JSONDecodeError:
                pass

    try:
        import torch
    except ImportError:
        return {}
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def infer_trained_policy_type(
    *,
    model_asset: Optional[dict[str, Any]] = None,
    checkpoint_path: Optional[str] = None,
) -> str:
    """Infer cable-threading trained policy adapter from asset metadata and checkpoint."""
    asset = dict(model_asset or {})
    backend = resolve_model_asset_backend_type(asset)
    if backend == "diffusion_policy":
        return "diffusion_policy"
    if backend in {"robomimic_bc", "robomimic"}:
        return "robomimic"
    if backend == "act":
        return "act"
    if backend == "pi0":
        return "pi0"

    model_type_id = str(asset.get("modelTypeId") or "").strip().lower()
    model_type = str(asset.get("modelType") or "").strip().lower()
    framework = str(asset.get("framework") or "").strip().lower()
    training_backend = str(asset.get("trainingBackend") or asset.get("backendType") or "").strip().lower()
    base_algorithm = str(asset.get("baseAlgorithm") or "").strip().lower()

    if model_type == "pi0" or training_backend == "pi0" or base_algorithm == "pi0" or model_type_id == "pi0":
        return "pi0"
    if model_type == "act" or training_backend == "act" or base_algorithm == "act" or model_type_id == "act":
        return "act"
    if model_type == "diffusion_policy" or "diffusion" in framework or training_backend == "diffusion_policy":
        return "diffusion_policy"

    resolved = resolve_local_checkpoint_path(asset=asset, path_hint=checkpoint_path)
    if not resolved:
        return "robomimic"

    payload = _load_checkpoint_payload(Path(resolved))
    train_config = payload.get("train_config") if isinstance(payload.get("train_config"), dict) else {}
    payload_backend = normalize_backend_type(
        str(payload.get("backend") or train_config.get("backend") or train_config.get("trainingBackend") or "")
    )
    if payload_backend == "diffusion_policy" or payload.get("action_key"):
        if payload.get("algo_name"):
            return "robomimic"
        if payload_backend == "diffusion_policy":
            return "diffusion_policy"
        action_key = str(payload.get("action_key") or train_config.get("action_key") or "")
        if action_key in {"joint_actions", "actions"} and train_config.get("eval_executor"):
            return "diffusion_policy"
        if train_config.get("low_dim_keys") or train_config.get("low_dim_dim"):
            return "diffusion_policy"
    if payload.get("algo_name"):
        return "robomimic"
    if payload_backend == "act":
        return "act"
    return "robomimic"


def resolve_eval_checkpoint_path(
    *,
    asset: Optional[dict[str, Any]] = None,
    path_hint: Optional[str] = None,
    model_asset_id: Optional[str] = None,
) -> tuple[Optional[str], bool]:
    """Resolve a checkpoint path suitable for evaluation launch."""
    local_path = resolve_local_checkpoint_path(
        asset=asset,
        path_hint=path_hint,
        model_asset_id=model_asset_id,
    )
    if local_path:
        return local_path, True

    direct = str(path_hint or "").strip()
    if direct:
        on_disk = resolve_checkpoint_path_on_disk(direct)
        if on_disk is not None:
            exists, _size = check_checkpoint_file(str(on_disk))
            if exists:
                return str(on_disk.resolve()), True
        if not _is_remote_uri(direct):
            return direct, False

    asset_path = str((asset or {}).get("checkpointPath") or (asset or {}).get("artifactPath") or "").strip()
    if asset_path and not _is_remote_uri(asset_path):
        on_disk = resolve_checkpoint_path_on_disk(asset_path)
        if on_disk is not None:
            exists, _size = check_checkpoint_file(str(on_disk))
            if exists:
                return str(on_disk.resolve()), True
        return asset_path, False

    return None, False
