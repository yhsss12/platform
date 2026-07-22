"""Resolve ACT / pi0 training and eval schemas (shared with DP joint-space conventions)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from app.services.dp_schema_resolver import (
    DEFAULT_JOINT_POLICY_SCHEMA,
    DpEvalExecutorSpec,
    DpTrainingSchemaSpec,
    JOINT_ACTION_MODES,
    resolve_dp_eval_executor,
    resolve_dp_training_schema,
)

PI0_JOINT_SPACE_ENABLED = False
PI0_JOINT_SPACE_DISABLED_REASON = (
    "pi0 joint-space 尚未接入：平台 HDF5→LeRobot 转换与 openpi 训练链路未完成"
)
PI0_PLATFORM_EVAL_NOT_ENABLED_REASON = "pi0 platform evaluation not enabled"
PI0_EVAL_ADAPTER_READY_MARKER = (
    Path(__file__).resolve().parents[3]
    / "runs/evaluation/pi0_smoke/adapter_ready.json"
)
PI0_PLATFORM_EVAL_READY_MARKER = (
    Path(__file__).resolve().parents[3]
    / "runs/evaluation/pi0_smoke/platform_eval_ready.json"
)
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def resolve_act_training_schema(
    raw_manifest: dict[str, Any],
    *,
    hdf5_path: Path | str | None = None,
    profile: dict[str, Any] | None = None,
) -> DpTrainingSchemaSpec:
    """Resolve ACT training schema; prefers joint-space when manifest/HDF5 supports it."""
    spec = resolve_dp_training_schema(raw_manifest, hdf5_path=hdf5_path, profile=profile)
    policy_schemas = raw_manifest.get("policySchemas") if isinstance(raw_manifest.get("policySchemas"), dict) else {}
    joint_schema = policy_schemas.get(DEFAULT_JOINT_POLICY_SCHEMA) or {}
    joint_out = joint_schema.get("output") if isinstance(joint_schema.get("output"), dict) else {}
    if joint_out.get("action_key") == "actions" and int(joint_out.get("action_dim") or spec.action_dim) == 8:
        return DpTrainingSchemaSpec(
            policy_schema_id=DEFAULT_JOINT_POLICY_SCHEMA,
            action_key="actions",
            gripper_action_key=None,
            action_dim=8,
            action_mode=str(joint_out.get("action_mode") or spec.action_mode or "joint_delta_derived"),
            controller_type="JOINT_POSITION",
            eval_executor="joint_position",
            trained_action_mode=str(
                raw_manifest.get("trainedActionMode") or joint_out.get("action_mode") or "joint_delta_derived"
            ),
            low_dim_keys=list(spec.low_dim_keys or ["robot0_joint_pos", "robot0_gripper_qpos"]),
            image_keys=list(spec.image_keys or []),
            observation_schema=spec.observation_schema,
            action_schema=spec.action_schema,
            controller_schema=spec.controller_schema,
            side_channel_schema=spec.side_channel_schema,
        )
    return spec


def to_act_config_fields(
    schema: DpTrainingSchemaSpec,
    *,
    low_dim_dim: int | None = None,
) -> dict[str, Any]:
    return {
        "action_key": schema.action_key,
        "gripper_action_key": schema.gripper_action_key,
        "action_dim": schema.action_dim,
        "action_mode": schema.action_mode,
        "controller_type": schema.controller_type,
        "eval_executor": schema.eval_executor,
        "trained_action_mode": schema.trained_action_mode,
        "image_keys": list(schema.image_keys),
        "low_dim_keys": list(schema.low_dim_keys),
        "low_dim_dim": low_dim_dim,
        "observation_schema": schema.observation_schema,
        "action_schema": schema.action_schema,
        "controller_schema": schema.controller_schema,
        "side_channel_schema": schema.side_channel_schema,
        "preferred_policy_schema_id": schema.policy_schema_id,
    }


def _load_act_checkpoint_meta(checkpoint_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    from app.services.model_asset_checkpoint_resolver import _load_checkpoint_payload

    payload = _load_checkpoint_payload(checkpoint_path)
    if not payload:
        return {}, {}
    shape_meta = dict(payload.get("shape_meta") or {})
    train_config = payload.get("train_config")
    if not isinstance(train_config, dict):
        train_config = payload.get("config")
    return shape_meta, dict(train_config) if isinstance(train_config, dict) else {}


def resolve_act_eval_executor(
    *,
    policy: str,
    model_asset: dict[str, Any] | None = None,
    checkpoint_path: str | Path | None = None,
) -> DpEvalExecutorSpec:
    if policy in {"scripted", "random"}:
        return resolve_dp_eval_executor(policy=policy)
    if policy != "act":
        return resolve_dp_eval_executor(policy=policy, model_asset=model_asset, checkpoint_path=checkpoint_path)

    asset = dict(model_asset or {})
    shape_meta, train_config = (
        _load_act_checkpoint_meta(Path(checkpoint_path).expanduser())
        if checkpoint_path
        else ({}, {})
    )

    eval_executor = str(
        asset.get("evalExecutor")
        or shape_meta.get("eval_executor")
        or train_config.get("eval_executor")
        or ""
    ).strip()
    action_mode = str(
        asset.get("trainedActionMode")
        or asset.get("actionMode")
        or shape_meta.get("trained_action_mode")
        or shape_meta.get("action_mode")
        or train_config.get("trained_action_mode")
        or train_config.get("action_mode")
        or ""
    ).strip()
    controller_type = str(
        asset.get("controllerType")
        or (asset.get("controllerSchema") or {}).get("controllerType")
        if isinstance(asset.get("controllerSchema"), dict)
        else asset.get("controllerSchema")
        or shape_meta.get("controller_type")
        or train_config.get("controller_type")
        or ""
    ).strip()
    action_dim = int(
        asset.get("actionDim")
        or shape_meta.get("action_dim")
        or train_config.get("action_dim")
        or 0
    )
    low_dim_keys = list(
        asset.get("lowDimKeys")
        or shape_meta.get("low_dim_keys")
        or train_config.get("low_dim_keys")
        or []
    )

    if eval_executor == "osc_pose" and controller_type == "JOINT_POSITION":
        raise ValueError("evalExecutor osc_pose inconsistent with controller_type JOINT_POSITION")

    if action_dim == 7 and (
        eval_executor == "joint_position"
        or controller_type == "JOINT_POSITION"
        or action_mode in JOINT_ACTION_MODES
    ):
        raise ValueError("7D OSC actions cannot use joint_position executor")

    if (
        eval_executor == "joint_position"
        or controller_type == "JOINT_POSITION"
        or action_mode in JOINT_ACTION_MODES
        or (
            action_dim == 8
            and low_dim_keys == ["robot0_joint_pos", "robot0_gripper_qpos"]
        )
    ):
        return DpEvalExecutorSpec(
            eval_executor="joint_position",
            controller_type="JOINT_POSITION",
            action_mode=action_mode or "joint_delta_derived",
            policy_type="act",
            side_channel_mode="policy",
            source="act_model_schema",
        )

    return DpEvalExecutorSpec(
        eval_executor="osc_pose",
        controller_type=controller_type or "OSC_POSE",
        action_mode=action_mode or "osc_pose_delta_eef",
        policy_type="act",
        side_channel_mode="policy",
        source="act_legacy_default",
    )


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def is_pi0_joint_space_eval_asset(
    model_asset: dict[str, Any] | None = None,
    *,
    checkpoint_path: str | Path | None = None,
) -> bool:
    """True when pi0 LeRobot joint-space eval schema is present (Panda / 9D state / 8D action)."""
    asset = dict(model_asset or {})
    fields: dict[str, Any] = {}
    if checkpoint_path:
        fields = extract_pi0_schema_fields_from_checkpoint(checkpoint_path)
    model_type = str(
        asset.get("modelType") or asset.get("policyType") or fields.get("modelType") or ""
    ).lower()
    if model_type not in {"pi0", "openpi"}:
        return False
    dataset_format = str(
        asset.get("datasetFormat") or fields.get("datasetFormat") or ""
    ).lower()
    if dataset_format and dataset_format not in {"lerobot"}:
        return False
    state_dim = _coerce_int(asset.get("stateDim") or fields.get("stateDim"))
    action_dim = _coerce_int(asset.get("actionDim") or fields.get("actionDim"))
    robot = str(asset.get("robot") or fields.get("robot") or "").strip()
    controller_type = str(
        asset.get("controllerType") or fields.get("controllerType") or ""
    ).strip()
    action_mode = str(
        asset.get("trainedActionMode") or asset.get("actionMode") or fields.get("actionMode") or ""
    ).strip()
    low_dim_keys = _norm_low_dim_keys(asset.get("lowDimKeys") or fields.get("lowDimKeys") or [])
    if robot and robot != "Panda":
        return False
    if state_dim is not None and state_dim != 9:
        return False
    if action_dim is not None and action_dim != 8:
        return False
    if controller_type and controller_type != "JOINT_POSITION":
        return False
    if action_mode and action_mode not in JOINT_ACTION_MODES:
        return False
    if low_dim_keys and set(low_dim_keys) != {"robot0_joint_pos", "robot0_gripper_qpos"}:
        return False
    return state_dim == 9 and action_dim == 8 and controller_type == "JOINT_POSITION"


def pi0_eval_adapter_ready() -> bool:
    """True after Phase F standalone joint-position rollout smoke succeeds."""
    marker = PI0_EVAL_ADAPTER_READY_MARKER
    if not marker.is_file():
        return False
    try:
        import json

        payload = json.loads(marker.read_text(encoding="utf-8"))
        return bool(payload.get("eval_adapter_ready"))
    except (OSError, json.JSONDecodeError):
        return False


def pi0_platform_eval_ready() -> bool:
    """True after Phase G platform evaluation job smoke succeeds."""
    marker = PI0_PLATFORM_EVAL_READY_MARKER
    if not marker.is_file():
        return False
    try:
        import json

        payload = json.loads(marker.read_text(encoding="utf-8"))
        return bool(payload.get("platform_eval_ready"))
    except (OSError, json.JSONDecodeError):
        return False


def mark_pi0_platform_eval_ready(*, eval_job_id: str, model_asset_id: str) -> None:
    """Persist platform eval readiness and enable pi0 joint-space for production."""
    global PI0_JOINT_SPACE_ENABLED
    marker_dir = PI0_PLATFORM_EVAL_READY_MARKER.parent
    marker_dir.mkdir(parents=True, exist_ok=True)
    import json

    PI0_PLATFORM_EVAL_READY_MARKER.write_text(
        json.dumps(
            {
                "platform_eval_ready": True,
                "pi0_joint_space_enabled": True,
                "evalJobId": eval_job_id,
                "modelAssetId": model_asset_id,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    PI0_JOINT_SPACE_ENABLED = True


def is_pi0_joint_space_enabled() -> bool:
    """True when platform eval smoke succeeded (persisted marker or in-process flag)."""
    global PI0_JOINT_SPACE_ENABLED
    if PI0_JOINT_SPACE_ENABLED:
        return True
    if pi0_platform_eval_ready():
        PI0_JOINT_SPACE_ENABLED = True
        return True
    return False


def resolve_pi0_eval_disabled_reason(*, eval_adapter_ready: bool | None = None) -> str | None:
    adapter_ready = pi0_eval_adapter_ready() if eval_adapter_ready is None else bool(eval_adapter_ready)
    if not adapter_ready:
        from app.services.pi0_lerobot_smoke_runner import PI0_EVAL_DISABLED_REASON

        return PI0_EVAL_DISABLED_REASON
    if not pi0_platform_eval_ready():
        return PI0_PLATFORM_EVAL_NOT_ENABLED_REASON
    return None


def explain_pi0_model_asset_eval_blocker(
    model_asset: dict[str, Any] | None = None,
    *,
    checkpoint_path: str | Path | None = None,
) -> str | None:
    asset = dict(model_asset or {})
    ckpt = checkpoint_path or asset.get("checkpointPath") or asset.get("artifactPath")
    if not is_pi0_joint_space_eval_asset(asset, checkpoint_path=ckpt):
        return "pi0 eval adapter not ready"
    return resolve_pi0_eval_disabled_reason()


def resolve_pi0_model_asset_eval_fields(
    model_asset: dict[str, Any] | None = None,
    *,
    checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    asset = dict(model_asset or {})
    ckpt = checkpoint_path or asset.get("checkpointPath") or asset.get("artifactPath")
    blocker = explain_pi0_model_asset_eval_blocker(asset, checkpoint_path=ckpt)
    can_evaluate = blocker is None
    joint_ready = is_pi0_joint_space_eval_asset(asset, checkpoint_path=ckpt)
    return {
        "canEvaluate": can_evaluate,
        "evalDisabledReason": blocker,
        "evalExecutor": "joint_position" if joint_ready else None,
    }


def pi0_eval_creation_allowed(
    model_asset: dict[str, Any] | None = None,
    *,
    checkpoint_path: str | Path | None = None,
) -> tuple[bool, str | None]:
    """Platform eval job creation: allowed once Phase F adapter is ready."""
    asset = dict(model_asset or {})
    if not is_pi0_joint_space_eval_asset(asset, checkpoint_path=checkpoint_path):
        return False, "pi0 eval adapter not ready"
    if not pi0_eval_adapter_ready():
        from app.services.pi0_lerobot_smoke_runner import PI0_EVAL_DISABLED_REASON

        return False, PI0_EVAL_DISABLED_REASON
    return True, None


def resolve_pi0_eval_job_paths(
    model_asset: dict[str, Any] | None = None,
    *,
    checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    asset = dict(model_asset or {})
    train_config_path = _resolve_pi0_train_config_path(asset)
    runtime = resolve_pi0_eval_runtime(
        policy="pi0",
        model_asset=asset,
        checkpoint_path=checkpoint_path,
        train_config_path=train_config_path,
    )
    return {
        "trainConfigPath": train_config_path,
        "taskInstruction": runtime.get("taskInstruction"),
        "sourceTrainJobId": str(
            asset.get("sourceTrainingJobId") or asset.get("sourceTrainJobId") or ""
        ).strip()
        or None,
        "modelType": "pi0",
        "policyRuntime": "pi0",
        **runtime,
    }


def _resolve_pi0_train_config_path(model_asset: dict[str, Any]) -> str | None:
    explicit = str(model_asset.get("trainConfigPath") or model_asset.get("train_config_path") or "").strip()
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_absolute():
            if path.parts and path.parts[0] == "runs":
                from app.core.platform_paths import resolve_runtime_reference

                path = resolve_runtime_reference(explicit)
            else:
                path = PROJECT_ROOT / path
        if path.is_file():
            return str(path.resolve())
    source_train_job_id = str(
        model_asset.get("sourceTrainingJobId") or model_asset.get("sourceTrainJobId") or ""
    ).strip()
    if not source_train_job_id:
        return None
    from app.core.platform_paths import platform_paths

    for root in dict.fromkeys(
        (platform_paths.training_jobs,)
    ):
        candidate = root / source_train_job_id / "config" / "train_config.json"
        if candidate.is_file():
            return str(candidate.resolve())
    return None


def resolve_pi0_eval_executor(
    *,
    policy: str,
    model_asset: dict[str, Any] | None = None,
    checkpoint_path: str | Path | None = None,
) -> DpEvalExecutorSpec:
    if policy in {"scripted", "random"}:
        return resolve_dp_eval_executor(policy=policy)
    if policy != "pi0":
        return resolve_dp_eval_executor(policy=policy, model_asset=model_asset, checkpoint_path=checkpoint_path)

    asset = dict(model_asset or {})
    if is_pi0_joint_space_eval_asset(asset, checkpoint_path=checkpoint_path):
        action_mode = str(
            asset.get("trainedActionMode")
            or asset.get("actionMode")
            or extract_pi0_schema_fields_from_checkpoint(checkpoint_path or "").get("actionMode")
            or "joint_delta_derived"
        )
        return DpEvalExecutorSpec(
            eval_executor="joint_position",
            controller_type="JOINT_POSITION",
            action_mode=action_mode,
            policy_type="pi0",
            side_channel_mode="policy",
            source="pi0_lerobot_joint_schema",
        )

    if not PI0_JOINT_SPACE_ENABLED:
        return DpEvalExecutorSpec(
            eval_executor="osc_pose",
            controller_type="OSC_POSE",
            action_mode="legacy",
            policy_type="pi0",
            side_channel_mode="policy",
            source="pi0_joint_space_not_enabled",
        )

    eval_executor = str(asset.get("evalExecutor") or "").strip()
    if eval_executor == "joint_position":
        return DpEvalExecutorSpec(
            eval_executor="joint_position",
            controller_type="JOINT_POSITION",
            action_mode=str(asset.get("trainedActionMode") or "joint_delta_derived"),
            policy_type="pi0",
            side_channel_mode="policy",
            source="pi0_model_schema",
        )
    return resolve_dp_eval_executor(policy=policy, model_asset=model_asset, checkpoint_path=checkpoint_path)


def resolve_pi0_eval_runtime(
    *,
    policy: str,
    model_asset: dict[str, Any] | None = None,
    checkpoint_path: str | Path | None = None,
    train_config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Backend resolver payload for pi0 eval rollout (smoke / platform dev path)."""
    spec = resolve_pi0_eval_executor(
        policy=policy,
        model_asset=model_asset,
        checkpoint_path=checkpoint_path,
    )
    asset = dict(model_asset or {})
    fields = extract_pi0_schema_fields_from_checkpoint(checkpoint_path or "") if checkpoint_path else {}
    train_config: dict[str, Any] = {}
    if train_config_path:
        cfg_path = Path(train_config_path).expanduser()
        if cfg_path.is_file():
            try:
                import json

                train_config = json.loads(cfg_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                train_config = {}
    robot, _warnings = resolve_eval_robot_for_policy(
        policy="pi0",
        model_asset=asset,
        checkpoint_path=checkpoint_path,
        eval_executor=spec.eval_executor,
        controller_type=spec.controller_type,
        action_mode=spec.action_mode,
    )
    if spec.eval_executor == "joint_position" and robot != "Panda":
        raise ValueError(f"pi0 joint-space eval requires Panda robot, got {robot or 'unknown'}")
    task_instruction = str(
        asset.get("taskInstruction")
        or train_config.get("taskInstruction")
        or train_config.get("task_instruction")
        or fields.get("taskInstruction")
        or ""
    ).strip()
    if spec.eval_executor == "joint_position" and not task_instruction:
        raise ValueError("pi0 joint-space eval requires taskInstruction")
    return {
        "policyType": "pi0",
        "modelType": "pi0",
        "policyRuntime": "pi0",
        "evalExecutor": spec.eval_executor,
        "controllerType": spec.controller_type,
        "actionMode": spec.action_mode,
        "sideChannelMode": spec.side_channel_mode,
        "robot": robot,
        "stateDim": _coerce_int(asset.get("stateDim") or fields.get("stateDim") or train_config.get("stateDim")),
        "actionDim": _coerce_int(asset.get("actionDim") or fields.get("actionDim") or train_config.get("actionDim")),
        "taskInstruction": task_instruction,
        "source": spec.source,
    }


def pi0_joint_space_capability() -> dict[str, Any]:
    return {
        "jointSpaceEnabled": PI0_JOINT_SPACE_ENABLED,
        "reason": PI0_JOINT_SPACE_DISABLED_REASON if not PI0_JOINT_SPACE_ENABLED else None,
        "policySchemaId": DEFAULT_JOINT_POLICY_SCHEMA,
    }


def assess_pi0_lerobot_data_format_readiness(manifest: dict[str, Any]) -> dict[str, Any]:
    """Assess whether a cable_threading manifest / data asset is pi0 joint-space data-format ready."""
    available = manifest.get("availableFormats") or manifest.get("datasetFormats") or []
    if isinstance(available, str):
        available = [available]
    available_norm = {str(item).strip().lower() for item in available if str(item).strip()}
    primary = str(manifest.get("primaryFormat") or manifest.get("datasetFormat") or manifest.get("format") or "").lower()

    lerobot_block = manifest.get("lerobot")
    if not isinstance(lerobot_block, dict):
        lerobot_meta = manifest.get("lerobotMetadata")
        if isinstance(lerobot_meta, dict):
            lerobot_block = lerobot_meta

    if "lerobot" not in available_norm and primary != "lerobot" and not lerobot_block:
        return {
            "data_format_ready": False,
            "reason": "dataset does not include LeRobot export",
        }

    status = str((lerobot_block or {}).get("status") or "").lower()
    if status and status != "ready":
        return {
            "data_format_ready": False,
            "reason": f"LeRobot export status is {status}",
        }

    pi0_ready = bool((lerobot_block or {}).get("pi0Ready", manifest.get("pi0Ready")))
    pi0_reason = str((lerobot_block or {}).get("pi0ReadyReason") or manifest.get("pi0ReadyReason") or "")
    state_dim = (lerobot_block or {}).get("stateDim", manifest.get("state_dim"))
    action_dim = (lerobot_block or {}).get("actionDim", manifest.get("action_dim"))
    robot = str((lerobot_block or {}).get("robot") or manifest.get("robot") or "")
    task_instruction = (
        (lerobot_block or {}).get("taskInstruction")
        or manifest.get("taskDescription")
        or manifest.get("task_instruction")
        or ""
    )

    if not pi0_ready:
        return {
            "data_format_ready": False,
            "reason": pi0_reason or "LeRobot export is not pi0 joint-space ready",
            "lerobotReady": True,
            "pi0Ready": False,
        }

    missing: list[str] = []
    if robot != "Panda":
        missing.append(f"robot={robot or 'unknown'}")
    if state_dim != 9:
        missing.append(f"stateDim={state_dim}")
    if action_dim != 8:
        missing.append(f"actionDim={action_dim}")
    if not str(task_instruction).strip():
        missing.append("taskInstruction missing")
    if missing:
        return {
            "data_format_ready": False,
            "reason": "; ".join(missing),
            "lerobotReady": True,
            "pi0Ready": False,
        }

    return {
        "data_format_ready": True,
        "reason": None,
        "lerobotReady": True,
        "pi0Ready": True,
        "robot": robot,
        "stateDim": state_dim,
        "actionDim": action_dim,
        "taskInstruction": str(task_instruction).strip(),
    }


JOINT_LOW_DIM_KEYS = frozenset({"robot0_joint_pos", "robot0_gripper_qpos", "robot0_joint_pos_rel"})


def _norm_low_dim_keys(keys: Any) -> list[str]:
    if not isinstance(keys, list):
        return []
    return [str(item).strip() for item in keys if str(item).strip()]


def is_joint_space_policy_schema(
    *,
    eval_executor: str | None = None,
    controller_type: str | None = None,
    action_mode: str | None = None,
    action_dim: int | None = None,
    low_dim_keys: list[str] | None = None,
    preferred_policy_schema_id: str | None = None,
) -> bool:
    executor = str(eval_executor or "").strip()
    controller = str(controller_type or "").strip()
    mode = str(action_mode or "").strip()
    keys = set(_norm_low_dim_keys(low_dim_keys or []))
    schema_id = str(preferred_policy_schema_id or "").strip()
    if schema_id == DEFAULT_JOINT_POLICY_SCHEMA:
        return True
    if executor == "joint_position" or controller == "JOINT_POSITION":
        return True
    if mode in JOINT_ACTION_MODES:
        return True
    if action_dim == 8 and keys.intersection(JOINT_LOW_DIM_KEYS):
        return True
    return False


def extract_act_schema_fields_from_checkpoint(checkpoint_path: Path | str) -> dict[str, Any]:
    shape_meta, train_config = _load_act_checkpoint_meta(Path(checkpoint_path).expanduser())
    if not shape_meta and not train_config:
        return {}

    image_keys = list(
        shape_meta.get("image_keys")
        or train_config.get("image_keys")
        or []
    )
    low_dim_keys = _norm_low_dim_keys(
        shape_meta.get("low_dim_keys") or train_config.get("low_dim_keys") or []
    )
    action_dim_raw = shape_meta.get("action_dim") or train_config.get("action_dim")
    try:
        action_dim = int(action_dim_raw) if action_dim_raw is not None else None
    except (TypeError, ValueError):
        action_dim = None

    low_dim_dim_raw = (
        shape_meta.get("low_dim_dim")
        or shape_meta.get("state_dim")
        or train_config.get("low_dim_dim")
        or train_config.get("state_dim")
    )
    try:
        low_dim_dim = int(low_dim_dim_raw) if low_dim_dim_raw is not None else None
    except (TypeError, ValueError):
        low_dim_dim = None

    eval_executor = str(
        shape_meta.get("eval_executor") or train_config.get("eval_executor") or ""
    ).strip()
    controller_type = str(
        shape_meta.get("controller_type") or train_config.get("controller_type") or ""
    ).strip()
    action_mode = str(
        shape_meta.get("trained_action_mode")
        or shape_meta.get("action_mode")
        or train_config.get("trained_action_mode")
        or train_config.get("action_mode")
        or ""
    ).strip()
    action_key = str(shape_meta.get("action_key") or train_config.get("action_key") or "actions").strip()
    preferred_policy_schema_id = str(
        shape_meta.get("preferred_policy_schema_id")
        or train_config.get("preferred_policy_schema_id")
        or train_config.get("policy_schema_id")
        or ""
    ).strip()
    robot = str(
        shape_meta.get("robot")
        or shape_meta.get("robot_type")
        or train_config.get("robot")
        or train_config.get("robot_type")
        or ""
    ).strip()

    if is_joint_space_policy_schema(
        eval_executor=eval_executor,
        controller_type=controller_type,
        action_mode=action_mode,
        action_dim=action_dim,
        low_dim_keys=low_dim_keys,
        preferred_policy_schema_id=preferred_policy_schema_id,
    ):
        eval_executor = eval_executor or "joint_position"
        controller_type = controller_type or "JOINT_POSITION"
        action_mode = action_mode or "joint_delta_derived"

    fields: dict[str, Any] = {
        "actionKey": action_key or None,
        "actionDim": action_dim,
        "lowDimDim": low_dim_dim,
        "lowDimKeys": low_dim_keys or None,
        "imageKeys": image_keys or None,
        "trainedActionMode": action_mode or None,
        "actionMode": action_mode or None,
        "evalExecutor": eval_executor or None,
        "controllerType": controller_type or None,
        "preferredPolicySchemaId": preferred_policy_schema_id or None,
        "robotType": robot or None,
    }
    if image_keys or low_dim_keys or action_dim is not None:
        fields["structureConfig"] = {
            "input": {
                "image_keys": image_keys,
                "low_dim_keys": low_dim_keys,
            },
            "output": {
                "action_dim": action_dim,
                "action_key": action_key,
            },
        }
    return {key: value for key, value in fields.items() if value not in (None, "", [], {})}


def extract_pi0_schema_fields_from_checkpoint(checkpoint_path: Path | str) -> dict[str, Any]:
    from app.services.model_asset_checkpoint_resolver import _load_checkpoint_payload

    payload = _load_checkpoint_payload(Path(checkpoint_path).expanduser())
    if not payload:
        return {}
    backend = str(payload.get("backend") or payload.get("modelType") or "").strip().lower()
    if payload.get("format") != "pi0_lerobot_smoke_v1" and backend not in {"pi0", "openpi"}:
        return {}
    image_keys = payload.get("image_keys") or payload.get("imageKeys") or []
    low_dim_keys = payload.get("low_dim_keys") or payload.get("lowDimKeys") or []
    fields: dict[str, Any] = {
        "modelType": "pi0",
        "policyType": "pi0",
        "framework": backend or "pi0",
        "trainingBackend": backend or "pi0",
        "backendType": backend or "pi0",
        "datasetFormat": payload.get("datasetFormat") or payload.get("dataset_format") or "lerobot",
        "stateDim": payload.get("state_dim") or payload.get("stateDim"),
        "actionDim": payload.get("action_dim") or payload.get("actionDim"),
        "robot": payload.get("robot"),
        "controllerType": payload.get("controller_type") or payload.get("controllerType"),
        "actionMode": payload.get("action_mode") or payload.get("actionMode"),
        "trainedActionMode": payload.get("action_mode") or payload.get("actionMode"),
        "actionRepresentation": payload.get("action_representation") or payload.get("actionRepresentation"),
        "taskInstruction": payload.get("task_instruction") or payload.get("taskInstruction"),
        "imageKeys": list(image_keys) if isinstance(image_keys, list) else [],
        "lowDimKeys": list(low_dim_keys) if isinstance(low_dim_keys, list) else [],
        "evalExecutor": "joint_position",
    }
    return {key: value for key, value in fields.items() if value not in (None, "", [], {})}


def extract_model_schema_fields_from_checkpoint(checkpoint_path: Path | str) -> dict[str, Any]:
    from app.services.dp_init_weight_compat import extract_dp_schema_fields_from_checkpoint

    path = Path(checkpoint_path).expanduser()
    fields = extract_pi0_schema_fields_from_checkpoint(path)
    if fields:
        return fields
    fields = extract_dp_schema_fields_from_checkpoint(path)
    if fields:
        return fields
    return extract_act_schema_fields_from_checkpoint(path)


def resolve_eval_robot_for_policy(
    *,
    policy: str,
    model_asset: dict[str, Any] | None = None,
    checkpoint_path: str | Path | None = None,
    eval_executor: str | None = None,
    controller_type: str | None = None,
    action_mode: str | None = None,
) -> tuple[str, list[str]]:
    """Resolve simulation robot for cable_threading eval (Panda for joint-space ACT/DP)."""
    asset = dict(model_asset or {})
    warnings: list[str] = []

    for key in ("robot", "robotType", "robot_type"):
        value = str(asset.get(key) or "").strip()
        if value:
            return value, warnings

    manifest = asset.get("manifest") if isinstance(asset.get("manifest"), dict) else {}
    for key in ("robot", "robotType", "robot_type"):
        value = str(manifest.get(key) or "").strip()
        if value:
            return value, warnings

    shape_meta, train_config = ({}, {})
    if checkpoint_path:
        shape_meta, train_config = _load_act_checkpoint_meta(Path(checkpoint_path).expanduser())

    for source in (shape_meta, train_config):
        for key in ("robot", "robot_type", "robotType"):
            value = str(source.get(key) or "").strip()
            if value:
                return value, warnings

    low_dim_keys = _norm_low_dim_keys(
        asset.get("lowDimKeys")
        or shape_meta.get("low_dim_keys")
        or train_config.get("low_dim_keys")
        or []
    )
    action_dim_raw = asset.get("actionDim") or shape_meta.get("action_dim") or train_config.get("action_dim")
    try:
        action_dim = int(action_dim_raw) if action_dim_raw is not None else None
    except (TypeError, ValueError):
        action_dim = None

    preferred_policy_schema_id = str(
        asset.get("preferredPolicySchemaId")
        or shape_meta.get("preferred_policy_schema_id")
        or train_config.get("preferred_policy_schema_id")
        or ""
    ).strip()

    resolved_executor = str(
        eval_executor or asset.get("evalExecutor") or shape_meta.get("eval_executor") or train_config.get("eval_executor") or ""
    ).strip()
    resolved_controller = str(
        controller_type
        or asset.get("controllerType")
        or shape_meta.get("controller_type")
        or train_config.get("controller_type")
        or ""
    ).strip()
    resolved_action_mode = str(
        action_mode
        or asset.get("trainedActionMode")
        or asset.get("actionMode")
        or shape_meta.get("action_mode")
        or train_config.get("action_mode")
        or ""
    ).strip()

    if is_joint_space_policy_schema(
        eval_executor=resolved_executor,
        controller_type=resolved_controller,
        action_mode=resolved_action_mode,
        action_dim=action_dim,
        low_dim_keys=low_dim_keys,
        preferred_policy_schema_id=preferred_policy_schema_id,
    ):
        if action_dim == 8 and set(low_dim_keys) >= {"robot0_joint_pos", "robot0_gripper_qpos"}:
            warnings.append("robot inferred as Panda from joint-space checkpoint schema")
            return "Panda", warnings
        if action_dim == 8:
            warnings.append("robot inferred as Panda from action_dim=8 joint-space schema")
            return "Panda", warnings

    if policy in {"act", "diffusion_policy", "pi0"} and resolved_executor == "joint_position":
        warnings.append("robot inferred as Panda from joint_position eval executor")
        return "Panda", warnings

    if policy == "pi0" and is_pi0_joint_space_eval_asset(asset, checkpoint_path=checkpoint_path):
        warnings.append("robot inferred as Panda from pi0 LeRobot joint-space schema")
        return "Panda", warnings

    return "UR5e", warnings
