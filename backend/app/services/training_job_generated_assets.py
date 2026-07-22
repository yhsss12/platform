"""训练任务详情页「生成的模型资产」过滤：排除 init/pretrained，按任务状态门控展示。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Set

from app.services.model_asset_validation import resolve_checkpoint_path_on_disk


def _normalize_path_key(path_str: str) -> str:
    text = str(path_str or "").strip()
    if not text:
        return ""
    if text.startswith("file://"):
        text = text[len("file://") :]
    path = Path(text)
    if not path.is_absolute():
        from app.core.platform_paths import resolve_runtime_reference
        from app.services.model_asset_validation import PROJECT_ROOT

        if path.parts and path.parts[0] == "runs":
            path = resolve_runtime_reference(text)
        else:
            path = PROJECT_ROOT / path
    try:
        return str(path.resolve())
    except OSError:
        return str(path)


def collect_init_checkpoint_paths(
    train_config: dict[str, Any],
    *,
    train_job_dir: Optional[Path] = None,
) -> Set[str]:
    paths: set[str] = set()
    for key in ("pretrained", "pretrainedModel"):
        block = train_config.get(key)
        if isinstance(block, dict):
            cp = str(block.get("checkpointPath") or "").strip()
            if cp:
                paths.add(_normalize_path_key(cp))

    if train_job_dir is not None and train_job_dir.is_dir():
        for rel in (
            "checkpoints/diffusion_policy/config/train_config.json",
            "checkpoints/act/config/train_config.json",
        ):
            cfg_path = train_job_dir / rel
            if not cfg_path.is_file():
                continue
            try:
                payload = json.loads(cfg_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict):
                init_cp = str(payload.get("init_checkpoint") or "").strip()
                if init_cp:
                    paths.add(_normalize_path_key(init_cp))
    return paths


def collect_init_model_asset_ids(train_config: dict[str, Any]) -> Set[str]:
    ids: set[str] = set()
    for key in ("pretrained", "pretrainedModel"):
        block = train_config.get(key)
        if isinstance(block, dict):
            asset_id = str(block.get("modelAssetId") or "").strip()
            if asset_id:
                ids.add(asset_id)
    return ids


def is_asset_path_under_job_dir(path_str: str, train_job_dir: Path) -> bool:
    path = resolve_checkpoint_path_on_disk(path_str)
    if path is None or not train_job_dir.is_dir():
        return False
    try:
        return path.resolve().is_relative_to(train_job_dir.resolve())
    except (OSError, ValueError):
        return False


def is_asset_owned_by_training_job(
    asset: dict[str, Any],
    train_job_id: str,
    train_job_dir: Path,
) -> bool:
    source = str(
        asset.get("sourceTrainingJobId")
        or asset.get("sourceTrainJobId")
        or asset.get("trainJobId")
        or ""
    ).strip()
    if source and source != train_job_id:
        return False

    asset_source = str(asset.get("assetSource") or "").lower()
    if asset_source == "imported":
        return False

    checkpoint_path = str(asset.get("checkpointPath") or "").strip()
    if checkpoint_path and train_job_dir.is_dir():
        return is_asset_path_under_job_dir(checkpoint_path, train_job_dir)

    if not checkpoint_path:
        return bool(asset.get("isPlaceholder"))

    return True


def is_init_checkpoint_asset(
    asset: dict[str, Any],
    *,
    init_paths: Set[str],
    init_asset_ids: Set[str],
) -> bool:
    checkpoint_path = _normalize_path_key(str(asset.get("checkpointPath") or ""))
    if checkpoint_path and checkpoint_path in init_paths:
        return True
    asset_id = str(asset.get("id") or asset.get("modelAssetId") or "").strip()
    if asset_id and asset_id in init_asset_ids:
        return True
    return False


def is_training_job_detail_in_progress(
    status: dict[str, Any],
    *,
    train_job_dir: Optional[Path] = None,
) -> bool:
    from app.services.checkpoint_registry import is_training_job_complete, is_training_job_in_progress

    if is_training_job_complete(status, train_job_dir=train_job_dir):
        return False
    return is_training_job_in_progress(status)


def filter_training_job_detail_model_assets(
    assets: list[dict[str, Any]],
    *,
    train_job_id: str,
    train_job_dir: Path,
    status: dict[str, Any],
    train_config: dict[str, Any],
) -> list[dict[str, Any]]:
    """训练详情页资产列表：训练中仅占位；完成后仅当前任务目录内 ready 资产。"""
    job_in_progress = is_training_job_detail_in_progress(
        status,
        train_job_dir=train_job_dir if train_job_dir.is_dir() else None,
    )
    init_paths = collect_init_checkpoint_paths(
        train_config,
        train_job_dir=train_job_dir if train_job_dir.is_dir() else None,
    )
    init_asset_ids = collect_init_model_asset_ids(train_config)

    filtered: list[dict[str, Any]] = []
    for raw in assets:
        asset = dict(raw)
        if asset.get("isPlaceholder"):
            if job_in_progress:
                asset["displayStatus"] = "waiting"
                asset["canEvaluate"] = False
            filtered.append(asset)
            continue

        if is_init_checkpoint_asset(asset, init_paths=init_paths, init_asset_ids=init_asset_ids):
            continue
        if train_job_dir.is_dir() and not is_asset_owned_by_training_job(asset, train_job_id, train_job_dir):
            continue
        if job_in_progress:
            continue

        asset["canEvaluate"] = bool(asset.get("canEvaluate")) and asset.get("displayStatus") == "ready"
        filtered.append(asset)

    return filtered


def resolve_training_job_detail_empty_message(
    *,
    status: dict[str, Any],
    train_job_dir: Optional[Path] = None,
) -> Optional[str]:
    if is_training_job_detail_in_progress(status, train_job_dir=train_job_dir):
        return "模型资产将在当前训练任务完成后生成。"
    return None
