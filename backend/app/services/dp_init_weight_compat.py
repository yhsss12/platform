"""Diffusion Policy initialization-weight schema extraction and compatibility."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from fastapi import HTTPException, status

JOINT_ACTION_KEYS = frozenset({"joint_actions"})
EEF_ACTION_KEYS = frozenset({"actions"})
JOINT_ACTION_MODES = frozenset({"joint_delta", "joint_delta_derived"})
JOINT_LOW_DIM_KEYS = frozenset({"robot0_joint_pos", "robot0_joint_pos_rel"})
EEF_LOW_DIM_KEYS = frozenset({"robot0_eef_pos", "robot0_eef_quat"})

INCOMPATIBLE_ACTION_SPACE_HINT = "不兼容：动作空间不同"


def _norm_keys(keys: Any) -> tuple[str, ...]:
    if not isinstance(keys, (list, tuple)):
        return tuple()
    return tuple(sorted({str(key).strip() for key in keys if str(key).strip()}))


def _norm_token(value: Any) -> str:
    return str(value or "").strip().lower()


def _norm_controller(value: Any) -> str:
    return str(value or "").strip().upper()


def extract_dp_init_schema_from_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    from app.services.training_dataset_compat import _resolved_low_dim_dim, extract_dp_action_schema

    schema = extract_dp_action_schema(cfg)
    return {
        "family": schema["family"],
        "actionKey": schema["action_key"],
        "gripperActionKey": str(cfg.get("gripper_action_key") or "").strip() or None,
        "actionDim": schema.get("action_dim"),
        "trainedActionMode": schema.get("trained_action_mode")
        or str(cfg.get("trained_action_mode") or cfg.get("action_mode") or "").strip()
        or None,
        "evalExecutor": schema.get("eval_executor"),
        "controllerType": schema.get("controller_type"),
        "imageKeys": list(schema.get("image_keys") or ()),
        "lowDimKeys": list(schema.get("low_dim_keys") or ()),
        "lowDimDim": schema.get("low_dim_dim") or _resolved_low_dim_dim(cfg),
        "imageSize": cfg.get("image_size"),
    }


def extract_dp_schema_fields_from_checkpoint(checkpoint_path: Path | str) -> dict[str, Any]:
    from app.services.dp_schema_resolver import _load_checkpoint_train_config

    train_config = _load_checkpoint_train_config(Path(checkpoint_path))
    if not train_config:
        return {}
    schema = extract_dp_init_schema_from_cfg(train_config)
    fields: dict[str, Any] = {
        "actionKey": schema.get("actionKey"),
        "gripperActionKey": schema.get("gripperActionKey"),
        "actionDim": schema.get("actionDim"),
        "trainedActionMode": schema.get("trainedActionMode"),
        "actionMode": schema.get("trainedActionMode"),
        "evalExecutor": schema.get("evalExecutor"),
        "controllerType": schema.get("controllerType"),
    }
    structure_config = {
        "input": {
            "image_keys": schema.get("imageKeys") or [],
            "low_dim_keys": schema.get("lowDimKeys") or [],
            "image_size": schema.get("imageSize"),
        },
        "output": {
            "action_dim": schema.get("actionDim"),
            "action_key": schema.get("actionKey"),
            "gripper_action_key": schema.get("gripperActionKey"),
        },
    }
    fields["structureConfig"] = structure_config
    resolved = {key: value for key, value in fields.items() if value not in (None, "", [], {})}
    return resolved


def extract_dp_init_schema_from_asset(asset: dict[str, Any]) -> dict[str, Any]:
    checkpoint_path = str(asset.get("checkpointPath") or asset.get("artifactPath") or "").strip()
    if checkpoint_path:
        from app.services.model_asset_checkpoint_resolver import resolve_local_checkpoint_path

        local_path = resolve_local_checkpoint_path(
            asset=asset,
            path_hint=checkpoint_path,
            model_asset_id=str(asset.get("id") or asset.get("modelAssetId") or ""),
        )
        if local_path:
            checkpoint_fields = extract_dp_schema_fields_from_checkpoint(local_path)
            ckpt_cfg = _checkpoint_fields_to_cfg(checkpoint_fields)
            if ckpt_cfg:
                return extract_dp_init_schema_from_cfg(ckpt_cfg)

    cfg: dict[str, Any] = {}
    for key, target in (
        ("action_key", "actionKey"),
        ("gripper_action_key", "gripperActionKey"),
        ("action_dim", "actionDim"),
        ("trained_action_mode", "trainedActionMode"),
        ("action_mode", "actionMode"),
        ("eval_executor", "evalExecutor"),
        ("controller_type", "controllerType"),
        ("image_size", "imageSize"),
    ):
        camel = asset.get(target)
        if camel not in (None, ""):
            cfg[key] = camel

    structure = asset.get("structureConfig") if isinstance(asset.get("structureConfig"), dict) else {}
    input_cfg = structure.get("input") if isinstance(structure.get("input"), dict) else {}
    output_cfg = structure.get("output") if isinstance(structure.get("output"), dict) else {}
    if input_cfg.get("image_keys"):
        cfg["image_keys"] = input_cfg.get("image_keys")
    if input_cfg.get("low_dim_keys"):
        cfg["low_dim_keys"] = input_cfg.get("low_dim_keys")
    if output_cfg.get("action_dim") is not None:
        cfg["action_dim"] = output_cfg.get("action_dim")
    if output_cfg.get("action_key"):
        cfg["action_key"] = output_cfg.get("action_key")
    if output_cfg.get("gripper_action_key"):
        cfg["gripper_action_key"] = output_cfg.get("gripper_action_key")
    if input_cfg.get("image_size") is not None:
        cfg["image_size"] = input_cfg.get("image_size")

    if cfg:
        return extract_dp_init_schema_from_cfg(cfg)

    return {
        "family": "",
        "actionKey": "",
        "gripperActionKey": None,
        "actionDim": None,
        "trainedActionMode": None,
        "evalExecutor": "",
        "controllerType": "",
        "imageKeys": [],
        "lowDimKeys": [],
        "lowDimDim": None,
        "imageSize": None,
    }


def _checkpoint_fields_to_cfg(fields: dict[str, Any]) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    mapping = {
        "actionKey": "action_key",
        "gripperActionKey": "gripper_action_key",
        "actionDim": "action_dim",
        "trainedActionMode": "trained_action_mode",
        "actionMode": "action_mode",
        "evalExecutor": "eval_executor",
        "controllerType": "controller_type",
    }
    for src, dst in mapping.items():
        if fields.get(src) not in (None, ""):
            cfg[dst] = fields[src]
    structure = fields.get("structureConfig")
    if isinstance(structure, dict):
        input_cfg = structure.get("input") if isinstance(structure.get("input"), dict) else {}
        if input_cfg.get("image_keys"):
            cfg["image_keys"] = input_cfg.get("image_keys")
        if input_cfg.get("low_dim_keys"):
            cfg["low_dim_keys"] = input_cfg.get("low_dim_keys")
        if input_cfg.get("image_size") is not None:
            cfg["image_size"] = input_cfg.get("image_size")
    return cfg


def resolve_dp_init_target_from_train_config(train_config: dict[str, Any]) -> dict[str, Any]:
    dp_cfg = train_config.get("dpConfig") if isinstance(train_config.get("dpConfig"), dict) else {}
    if dp_cfg:
        return extract_dp_init_schema_from_cfg(dp_cfg)
    return extract_dp_init_schema_from_cfg(train_config)


def _is_joint_legacy_actions_alias(source: dict[str, Any], target: dict[str, Any]) -> bool:
    """Standalone joint-DP checkpoints may store joint deltas under HDF5 key ``actions``."""
    if str(source.get("family") or "") != "joint" or str(target.get("family") or "") != "joint":
        return False
    source_key = str(source.get("actionKey") or "").strip()
    target_key = str(target.get("actionKey") or "").strip()
    if source_key != "actions" or target_key != "joint_actions":
        return False
    trained_mode = _norm_token(source.get("trainedActionMode"))
    controller_type = _norm_controller(source.get("controllerType"))
    return trained_mode in JOINT_ACTION_MODES or controller_type == "JOINT_POSITION"


def dp_init_weights_compatible(
    source: dict[str, Any],
    target: dict[str, Any],
) -> tuple[bool, Optional[str]]:
    source_key = str(source.get("actionKey") or "").strip()
    target_key = str(target.get("actionKey") or "").strip()
    source_family = str(source.get("family") or "").strip()
    target_family = str(target.get("family") or "").strip()
    joint_legacy_alias = _is_joint_legacy_actions_alias(source, target)

    if source_family and target_family and source_family != target_family:
        if source_family == "eef" and target_family == "joint":
            return False, (
                "该 checkpoint 为 EEF/OSC Diffusion Policy，当前训练任务为 Joint-Space Diffusion Policy，"
                "action schema 不一致，不能作为初始化权重。"
            )
        return False, (
            "该 checkpoint 为 Joint-Space Diffusion Policy，当前训练任务为 EEF/OSC Diffusion Policy，"
            "action schema 不一致，不能作为初始化权重。"
        )

    if target_key and source_key and source_key != target_key and not joint_legacy_alias:
        if target_key == "joint_actions" and source_key == "actions" and source_family == "eef":
            return False, (
                "该 checkpoint 为 EEF/OSC Diffusion Policy，当前训练任务为 Joint-Space Diffusion Policy，"
                "action schema 不一致，不能作为初始化权重。"
            )
        return False, f"{INCOMPATIBLE_ACTION_SPACE_HINT}（{source_key} ≠ {target_key}）"

    source_executor = _norm_token(source.get("evalExecutor"))
    target_executor = _norm_token(target.get("evalExecutor"))
    if source_executor and target_executor and source_executor != target_executor:
        return False, (
            f"模型 eval_executor={source_executor!r} 与当前任务 {target_executor!r} 不一致，"
            "不能作为初始化权重。"
        )

    source_controller = _norm_controller(source.get("controllerType"))
    target_controller = _norm_controller(target.get("controllerType"))
    if source_controller and target_controller and source_controller != target_controller:
        return False, (
            f"模型 controller_type={source_controller!r} 与当前任务 {target_controller!r} 不一致，"
            "不能作为初始化权重。"
        )

    source_dim = source.get("actionDim")
    target_dim = target.get("actionDim")
    if source_dim is not None and target_dim is not None and int(source_dim) != int(target_dim):
        return False, "模型与当前任务结构不匹配，无法作为初始化权重。（action_dim 不一致）"

    checks = (
        ("imageKeys", _norm_keys(source.get("imageKeys")), _norm_keys(target.get("imageKeys"))),
        ("lowDimKeys", _norm_keys(source.get("lowDimKeys")), _norm_keys(target.get("lowDimKeys"))),
    )
    for field, left, right in checks:
        if left and right and left != right:
            return False, f"模型与当前任务结构不匹配，无法作为初始化权重。（{field} 不一致）"

    if (
        source.get("lowDimDim") is not None
        and target.get("lowDimDim") is not None
        and int(source["lowDimDim"]) != int(target["lowDimDim"])
    ):
        return False, "模型与当前任务结构不匹配，无法作为初始化权重。（low_dim_dim 不一致）"

    if (
        source.get("imageSize") is not None
        and target.get("imageSize") is not None
        and int(source["imageSize"]) != int(target["imageSize"])
    ):
        return False, "模型与当前任务结构不匹配，无法作为初始化权重。（image_size 不一致）"

    return True, None


def assert_dp_init_weights_compatible(source: dict[str, Any], target: dict[str, Any]) -> None:
    ok, reason = dp_init_weights_compatible(source, target)
    if not ok:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=reason or INCOMPATIBLE_ACTION_SPACE_HINT)


def enrich_asset_dp_init_schema(asset: dict[str, Any]) -> dict[str, Any]:
    schema = extract_dp_init_schema_from_asset(asset)
    row = dict(asset)
    for key, value in (
        ("actionKey", schema.get("actionKey")),
        ("gripperActionKey", schema.get("gripperActionKey")),
        ("actionDim", schema.get("actionDim")),
        ("trainedActionMode", schema.get("trainedActionMode")),
        ("evalExecutor", schema.get("evalExecutor")),
        ("controllerType", schema.get("controllerType")),
    ):
        if value not in (None, "", []):
            row[key] = value
    if schema.get("imageKeys") or schema.get("lowDimKeys"):
        structure = dict(row.get("structureConfig") or {}) if isinstance(row.get("structureConfig"), dict) else {}
        input_cfg = dict(structure.get("input") or {}) if isinstance(structure.get("input"), dict) else {}
        if schema.get("imageKeys"):
            input_cfg["image_keys"] = schema.get("imageKeys")
        if schema.get("lowDimKeys"):
            input_cfg["low_dim_keys"] = schema.get("lowDimKeys")
        if schema.get("imageSize") is not None:
            input_cfg["image_size"] = schema.get("imageSize")
        structure["input"] = input_cfg
        output_cfg = dict(structure.get("output") or {}) if isinstance(structure.get("output"), dict) else {}
        if schema.get("actionDim") is not None:
            output_cfg["action_dim"] = schema.get("actionDim")
        if schema.get("actionKey"):
            output_cfg["action_key"] = schema.get("actionKey")
        if schema.get("gripperActionKey"):
            output_cfg["gripper_action_key"] = schema.get("gripperActionKey")
        structure["output"] = output_cfg
        row["structureConfig"] = structure
    row["dpInitSchema"] = schema
    return row
