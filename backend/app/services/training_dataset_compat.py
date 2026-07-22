"""Training dataset merge compatibility and pretrained checkpoint validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from fastapi import HTTPException, status

from app.services.adapter_layer.hdf5_inspector import inspect_hdf5

JOINT_ACTION_MODES = frozenset({"joint_delta", "joint_delta_derived", "joint_position"})
JOINT_ACTION_KEYS = frozenset({"joint_actions"})
EEF_ACTION_KEYS = frozenset({"actions"})


def _training_paths():
    from app.services import training_service as ts

    return ts


def _norm_keys(keys: Any) -> tuple[str, ...]:
    if not isinstance(keys, list):
        return tuple()
    return tuple(sorted(str(k) for k in keys if str(k).strip()))


def _image_size_from_inspection(inspection: Any) -> Optional[int]:
    shape = getattr(inspection, "image_shape", None) or {}
    if isinstance(shape, dict):
        for key in ("height", "width", "h", "w"):
            value = shape.get(key)
            if value is not None and int(value) > 0:
                return int(value)
    return None


def extract_dataset_structure_signature(manifest: dict[str, Any], hdf5_path: Optional[Path] = None) -> dict[str, Any]:
    """Build a comparable structure signature for merge / pretrained checks."""
    ts = _training_paths()
    path = hdf5_path or ts._resolve_hdf5_path(manifest)
    inspection = inspect_hdf5(path) if path is not None else None

    camera_keys = _norm_keys(manifest.get("cameraKeys") or manifest.get("imageKeys"))
    low_dim_keys = _norm_keys(manifest.get("observationKeys"))
    action_dim = manifest.get("actionDim")

    if inspection is not None and inspection.source == "hdf5":
        if inspection.camera_keys:
            camera_keys = tuple(sorted(str(k) for k in inspection.camera_keys))
        if inspection.state_keys:
            low_dim_keys = tuple(sorted(str(k) for k in inspection.state_keys if str(k) not in camera_keys))
        if inspection.action_dim is not None:
            action_dim = int(inspection.action_dim)

    obs_keys = manifest.get("obsKeys") or manifest.get("observationKeys") or []
    if not camera_keys and isinstance(obs_keys, list):
        camera_keys = tuple(
            sorted(str(k) for k in obs_keys if str(k).endswith("_image") or "image" in str(k).lower())
        )
    if not low_dim_keys and isinstance(obs_keys, list):
        low_dim_keys = tuple(sorted(str(k) for k in obs_keys if k not in camera_keys))

    image_size = manifest.get("imageSize") or manifest.get("image_size")
    if image_size is None and inspection is not None:
        image_size = _image_size_from_inspection(inspection)

    return {
        "taskType": str(manifest.get("taskType") or "").strip(),
        "taskTemplateId": str(manifest.get("taskTemplateId") or "").strip(),
        "taskName": str(manifest.get("taskName") or manifest.get("displayName") or "").strip(),
        "robotType": str(manifest.get("robotType") or manifest.get("robot") or "").strip(),
        "simulatorBackend": str(
            manifest.get("simulatorBackend") or manifest.get("backend") or manifest.get("simBackend") or ""
        ).strip(),
        "imageKeys": camera_keys,
        "lowDimKeys": low_dim_keys,
        "actionDim": int(action_dim) if action_dim is not None else None,
        "imageSize": int(image_size) if image_size is not None else None,
    }


def _signature_mismatch_reason(base: dict[str, Any], other: dict[str, Any]) -> str:
    checks = [
        ("taskType", base.get("taskType"), other.get("taskType")),
        ("robotType", base.get("robotType"), other.get("robotType")),
        ("simulatorBackend", base.get("simulatorBackend"), other.get("simulatorBackend")),
        ("imageKeys", base.get("imageKeys"), other.get("imageKeys")),
        ("lowDimKeys", base.get("lowDimKeys"), other.get("lowDimKeys")),
        ("actionDim", base.get("actionDim"), other.get("actionDim")),
        ("imageSize", base.get("imageSize"), other.get("imageSize")),
    ]
    for field, left, right in checks:
        if left and right and left != right:
            return f"数据集 {field} 不一致（{left!r} vs {right!r}）"
    return "数据集 observation 结构不一致"


def merge_training_manifests(manifests: list[dict[str, Any]]) -> tuple[dict[str, Any], list[Path], list[dict[str, Any]]]:
    """Validate homogeneity and return merged manifest + HDF5 paths."""
    ts = _training_paths()
    if not manifests:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="至少选择一个数据集")

    resolved: list[tuple[dict[str, Any], Optional[Path], dict[str, Any]]] = []
    for manifest in manifests:
        hdf5_path = ts._resolve_hdf5_path(manifest)
        if hdf5_path is None or not ts._is_valid_hdf5_file(hdf5_path):
            dataset_id = manifest.get("datasetId") or "unknown"
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"数据集 {dataset_id} 缺少可用 HDF5，无法合并训练",
            )
        signature = extract_dataset_structure_signature(manifest, hdf5_path)
        resolved.append((manifest, hdf5_path, signature))

    base_manifest, _, base_signature = resolved[0]
    for manifest, _, signature in resolved[1:]:
        if signature != base_signature:
            reason = _signature_mismatch_reason(base_signature, signature)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"数据集 observation 结构不一致，无法合并训练：{reason}",
            )

    dataset_ids = [str(m.get("datasetId") or "").strip() for m, _, _ in resolved if m.get("datasetId")]
    dataset_names = [str(m.get("datasetName") or m.get("displayName") or "").strip() for m, _, _ in resolved]
    sample_counts = [int(m.get("sampleCount") or m.get("validTrajectories") or m.get("episodeCount") or 0) for m, _, _ in resolved]

    merged = dict(base_manifest)
    merged["datasetId"] = dataset_ids[0] if len(dataset_ids) == 1 else "+".join(dataset_ids)
    merged["datasetIds"] = dataset_ids
    merged["datasetNames"] = [name for name in dataset_names if name]
    merged["datasetName"] = (
        dataset_names[0]
        if len(dataset_names) == 1
        else f"{len(dataset_ids)} 个数据集合并"
    )
    merged["sampleCount"] = sum(sample_counts)
    merged["validTrajectories"] = sum(sample_counts)
    merged["mergedDatasetCount"] = len(resolved)

    hdf5_paths = [path for _, path, _ in resolved if path is not None]
    artifacts = dict(merged.get("artifacts") or {})
    artifacts["hdf5"] = str(hdf5_paths[0])
    artifacts["hdf5Paths"] = [str(path) for path in hdf5_paths]
    merged["artifacts"] = artifacts
    merged["structureSignature"] = base_signature
    return merged, hdf5_paths, [sig for _, _, sig in resolved]


def resolve_training_hdf5_paths(manifest: dict[str, Any], train_config: Optional[dict[str, Any]] = None) -> list[Path]:
    ts = _training_paths()
    paths: list[Path] = []
    config = train_config or {}
    for raw in list(config.get("datasetHdf5Paths") or []):
        try:
            resolved = ts._resolve_safe_path(str(raw))
        except HTTPException:
            continue
        if ts._is_valid_hdf5_file(resolved):
            paths.append(resolved)

    if paths:
        return paths

    artifacts = manifest.get("artifacts") or {}
    for raw in list(artifacts.get("hdf5Paths") or []):
        try:
            resolved = ts._resolve_safe_path(str(raw))
        except HTTPException:
            continue
        if ts._is_valid_hdf5_file(resolved):
            paths.append(resolved)
    if paths:
        return paths

    single = ts._resolve_hdf5_path(manifest)
    if single is not None and ts._is_valid_hdf5_file(single):
        return [single]
    return []


def extract_dp_action_schema(cfg: dict[str, Any]) -> dict[str, Any]:
    action_key = str(cfg.get("action_key") or "").strip()
    eval_executor = str(cfg.get("eval_executor") or "").strip().lower()
    trained_mode = str(cfg.get("trained_action_mode") or cfg.get("action_mode") or "").strip().lower()
    controller_type = str(cfg.get("controller_type") or "").strip().upper()

    is_joint = (
        eval_executor == "joint_position"
        or action_key in JOINT_ACTION_KEYS
        or trained_mode in JOINT_ACTION_MODES
        or controller_type == "JOINT_POSITION"
    )
    family = "joint" if is_joint else "eef"
    return {
        "family": family,
        "action_key": action_key or ("joint_actions" if is_joint else "actions"),
        "action_dim": int(cfg["action_dim"]) if cfg.get("action_dim") is not None else None,
        "eval_executor": eval_executor or ("joint_position" if is_joint else "osc_pose"),
        "controller_type": controller_type or ("JOINT_POSITION" if is_joint else "OSC_POSE"),
        "trained_action_mode": trained_mode,
        "image_keys": _norm_keys(cfg.get("image_keys")),
        "low_dim_keys": _norm_keys(cfg.get("low_dim_keys")),
        "low_dim_dim": int(cfg["low_dim_dim"]) if cfg.get("low_dim_dim") is not None else None,
    }


def _resolved_low_dim_dim(cfg: dict[str, Any]) -> Optional[int]:
    explicit = cfg.get("low_dim_dim")
    if explicit is not None and int(explicit) > 0:
        return int(explicit)
    keys = list(cfg.get("low_dim_keys") or [])
    if keys:
        dim_map = {"robot0_joint_pos": 7, "robot0_gripper_qpos": 2}
        if all(key in dim_map for key in keys):
            return sum(dim_map[key] for key in keys)
    return None


def _validate_dp_schema_family_compatibility(ckpt_cfg: dict[str, Any], dp_cfg: dict[str, Any]) -> None:
    ckpt_schema = extract_dp_action_schema(ckpt_cfg)
    target_schema = extract_dp_action_schema(dp_cfg)
    if ckpt_schema["family"] == target_schema["family"]:
        return
    if ckpt_schema["family"] == "eef" and target_schema["family"] == "joint":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "该 checkpoint 为 EEF/OSC Diffusion Policy，当前训练任务为 Joint-Space Diffusion Policy，"
                "action schema 不一致，不能作为初始化权重。"
            ),
        )
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=(
            "该 checkpoint 为 Joint-Space Diffusion Policy，当前训练任务为 EEF/OSC Diffusion Policy，"
            "action schema 不一致，不能作为初始化权重。"
        ),
    )


def validate_dp_pretrained_checkpoint(
    *,
    checkpoint_path: Path,
    train_config: dict[str, Any],
) -> None:
    import torch

    from app.services.dp_init_weight_compat import (
        assert_dp_init_weights_compatible,
        extract_dp_init_schema_from_cfg,
    )

    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="预训练 checkpoint 格式无效")

    if "state_dict" not in payload or "normalizer" not in payload:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="预训练 checkpoint 缺少 state_dict 或 normalizer，无法初始化训练",
        )

    ckpt_cfg = payload.get("train_config") or {}
    if not isinstance(ckpt_cfg, dict):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="预训练 checkpoint 缺少 train_config")

    dp_cfg = train_config.get("dpConfig") or {}
    if not isinstance(dp_cfg, dict):
        dp_cfg = {}

    source_schema = extract_dp_init_schema_from_cfg(ckpt_cfg)
    target_schema = extract_dp_init_schema_from_cfg(dp_cfg)
    assert_dp_init_weights_compatible(source_schema, target_schema)

    checks: list[tuple[str, Any, Any]] = [
        ("vision_encoder", ckpt_cfg.get("vision_encoder"), dp_cfg.get("vision_encoder")),
    ]
    for field, left, right in checks:
        if left is None or right is None:
            continue
        if left != right:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"模型与当前任务结构不匹配，无法作为初始化权重。（{field} 不一致）",
            )

    shape_meta = payload.get("shape_meta") or {}
    if isinstance(shape_meta, dict):
        meta_action = shape_meta.get("action_dim")
        target_action = dp_cfg.get("action_dim")
        if meta_action is not None and target_action is not None and int(meta_action) != int(target_action):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="模型与当前任务结构不匹配：action_dim 与 checkpoint shape_meta 不一致",
            )
