"""Resolve Diffusion Policy training/eval schemas from HDF5, manifest, and model assets."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

JOINT_ACTION_MODES = frozenset({"joint_delta", "joint_delta_derived"})
DEFAULT_JOINT_POLICY_SCHEMA = "joint_state_obs_joint_action"
LEGACY_EEF_POLICY_SCHEMA = "eef_pose_obs_eef_action"


@dataclass
class DpTrainingSchemaSpec:
    policy_schema_id: str
    action_key: str
    gripper_action_key: str | None
    action_dim: int
    action_mode: str
    controller_type: str
    eval_executor: str
    trained_action_mode: str
    low_dim_keys: list[str] = field(default_factory=list)
    image_keys: list[str] = field(default_factory=list)
    observation_schema: str | None = None
    action_schema: str | None = None
    controller_schema: str | None = None
    side_channel_schema: str | None = None

    def to_dp_config_fields(self) -> dict[str, Any]:
        return {
            "action_key": self.action_key,
            "gripper_action_key": self.gripper_action_key,
            "action_dim": self.action_dim,
            "action_mode": self.action_mode,
            "controller_type": self.controller_type,
            "eval_executor": self.eval_executor,
            "trained_action_mode": self.trained_action_mode,
            "low_dim_keys": list(self.low_dim_keys),
            "image_keys": list(self.image_keys),
            "observation_schema": self.observation_schema,
            "action_schema": self.action_schema,
            "controller_schema": self.controller_schema,
            "side_channel_schema": self.side_channel_schema,
            "preferred_policy_schema_id": self.policy_schema_id,
        }


@dataclass(frozen=True)
class DpEvalExecutorSpec:
    eval_executor: str
    controller_type: str
    action_mode: str
    policy_type: str = "diffusion_policy"
    side_channel_mode: str = "policy"
    source: str = "default"

    def uses_joint_executor(self) -> bool:
        return self.eval_executor == "joint_position"


def _read_hdf5_attrs(hdf5_path: Path) -> dict[str, Any]:
    try:
        import h5py
    except ImportError:
        return {}
    if not hdf5_path.is_file():
        return {}
    try:
        with h5py.File(hdf5_path, "r") as handle:
            data = handle.get("data")
            if data is None:
                return {}
            attrs = dict(data.attrs)
            parsed: dict[str, Any] = {}
            for key, value in attrs.items():
                if isinstance(value, bytes):
                    value = value.decode("utf-8")
                if isinstance(value, str) and value.startswith(("{", "[")):
                    try:
                        parsed[key] = json.loads(value)
                        continue
                    except json.JSONDecodeError:
                        pass
                parsed[key] = value
            demos = [k for k in data.keys() if str(k).startswith("demo_")]
            if demos:
                demo0 = data[demos[0]]
                parsed["available_action_keys"] = parsed.get("available_action_keys") or []
                if demo0.get("joint_actions") is not None and "joint_actions" not in parsed["available_action_keys"]:
                    parsed.setdefault("joint_action_available", True)
            return parsed
    except OSError:
        return {}


def _schema_from_manifest_dict(raw: dict[str, Any]) -> DpTrainingSchemaSpec | None:
    action_schema_obj = raw.get("actionSchema")
    if isinstance(action_schema_obj, dict):
        action_mode = str(action_schema_obj.get("actionMode") or "")
        if action_mode in JOINT_ACTION_MODES or action_schema_obj.get("actionKey") == "joint_actions":
            gripper_key = action_schema_obj.get("gripperActionKey")
            action_dim = int(action_schema_obj.get("totalActionDim") or action_schema_obj.get("actionDim") or 8)
            if gripper_key and action_dim <= 7:
                action_dim = int(action_schema_obj.get("actionDim") or 7) + int(
                    action_schema_obj.get("gripperActionDim") or 1
                )
            obs_obj = raw.get("observationSchema") if isinstance(raw.get("observationSchema"), dict) else {}
            ctrl_obj = raw.get("controllerSchema") if isinstance(raw.get("controllerSchema"), dict) else {}
            return DpTrainingSchemaSpec(
                policy_schema_id=str(raw.get("preferredPolicySchemaId") or DEFAULT_JOINT_POLICY_SCHEMA),
                action_key=str(action_schema_obj.get("actionKey") or "joint_actions"),
                gripper_action_key=str(gripper_key) if gripper_key else "gripper_actions",
                action_dim=action_dim,
                action_mode="joint_delta",
                controller_type=str(ctrl_obj.get("controllerType") or "JOINT_POSITION"),
                eval_executor=str(ctrl_obj.get("evalExecutor") or raw.get("evalExecutor") or "joint_position"),
                trained_action_mode=str(raw.get("trainedActionMode") or "joint_delta"),
                low_dim_keys=list(obs_obj.get("lowDimKeys") or ["robot0_joint_pos", "robot0_gripper_qpos"]),
                image_keys=list(obs_obj.get("imageKeys") or raw.get("imageKeys") or []),
                observation_schema=obs_obj.get("id") or raw.get("observationSchema"),
                action_schema=action_schema_obj.get("id"),
                controller_schema=ctrl_obj.get("id") or raw.get("controllerSchema"),
                side_channel_schema=(raw.get("sideChannelSchema") or {}).get("id")
                if isinstance(raw.get("sideChannelSchema"), dict)
                else raw.get("sideChannelSchema"),
            )
    action_schema_id = raw.get("actionSchema")
    if isinstance(action_schema_id, str) and "joint" in action_schema_id:
        return _joint_spec_from_profile(raw)
    return None


def _joint_spec_from_profile(raw: dict[str, Any]) -> DpTrainingSchemaSpec:
    policy_schemas = raw.get("policySchemas") if isinstance(raw.get("policySchemas"), dict) else {}
    joint_schema = policy_schemas.get(DEFAULT_JOINT_POLICY_SCHEMA) or {}
    joint_out = joint_schema.get("output") if isinstance(joint_schema.get("output"), dict) else {}
    joint_in = joint_schema.get("input") if isinstance(joint_schema.get("input"), dict) else {}
    gripper_key = joint_out.get("gripper_action_key") or "gripper_actions"
    arm_dim = int(joint_out.get("action_dim") or 7)
    grip_dim = int(joint_out.get("gripper_action_dim") or 1)
    return DpTrainingSchemaSpec(
        policy_schema_id=DEFAULT_JOINT_POLICY_SCHEMA,
        action_key=str(joint_out.get("action_key") or "joint_actions"),
        gripper_action_key=str(gripper_key),
        action_dim=arm_dim + grip_dim,
        action_mode="joint_delta",
        controller_type="JOINT_POSITION",
        eval_executor="joint_position",
        trained_action_mode="joint_delta",
        low_dim_keys=list(joint_in.get("low_dim_keys") or raw.get("lowDimKeys") or ["robot0_joint_pos", "robot0_gripper_qpos"]),
        image_keys=list(joint_in.get("image_keys") or raw.get("imageKeys") or []),
        observation_schema=raw.get("observationSchema") if isinstance(raw.get("observationSchema"), str) else None,
        action_schema=raw.get("actionSchema") if isinstance(raw.get("actionSchema"), str) else None,
        controller_schema=raw.get("controllerSchema") if isinstance(raw.get("controllerSchema"), str) else None,
        side_channel_schema=raw.get("sideChannelSchema") if isinstance(raw.get("sideChannelSchema"), str) else None,
    )


def _legacy_eef_spec(raw: dict[str, Any]) -> DpTrainingSchemaSpec:
    policy_schemas = raw.get("policySchemas") if isinstance(raw.get("policySchemas"), dict) else {}
    eef_schema = policy_schemas.get(LEGACY_EEF_POLICY_SCHEMA) or {}
    eef_out = eef_schema.get("output") if isinstance(eef_schema.get("output"), dict) else {}
    eef_in = eef_schema.get("input") if isinstance(eef_schema.get("input"), dict) else {}
    return DpTrainingSchemaSpec(
        policy_schema_id=LEGACY_EEF_POLICY_SCHEMA,
        action_key=str(eef_out.get("action_key") or "actions"),
        gripper_action_key=None,
        action_dim=int(eef_out.get("action_dim") or raw.get("actionDim") or 7),
        action_mode="osc_pose_delta_eef",
        controller_type=str(raw.get("controller_type") or "OSC_POSE"),
        eval_executor="osc_pose",
        trained_action_mode="osc_pose_delta_eef",
        low_dim_keys=list(
            eef_in.get("low_dim_keys")
            or raw.get("lowDimKeys")
            or ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
        ),
        image_keys=list(eef_in.get("image_keys") or raw.get("imageKeys") or []),
        observation_schema=raw.get("observationSchema") if isinstance(raw.get("observationSchema"), str) else None,
        action_schema=raw.get("actionSchema") if isinstance(raw.get("actionSchema"), str) else None,
        controller_schema=raw.get("controllerSchema") if isinstance(raw.get("controllerSchema"), str) else None,
        side_channel_schema=raw.get("sideChannelSchema") if isinstance(raw.get("sideChannelSchema"), str) else None,
    )


def resolve_dp_training_schema(
    raw_manifest: dict[str, Any],
    *,
    hdf5_path: Path | str | None = None,
    profile: dict[str, Any] | None = None,
) -> DpTrainingSchemaSpec:
    """Pick DP training schema; prefer joint-space when HDF5/manifest supports it."""
    raw = dict(raw_manifest or {})
    if profile:
        raw.setdefault("imageKeys", profile.get("imageKeys") or profile.get("cameraKeys"))
        raw.setdefault("lowDimKeys", profile.get("lowDimKeys"))
        raw.setdefault("actionDim", profile.get("actionDim"))

    explicit = _schema_from_manifest_dict(raw)
    if explicit is not None:
        return explicit

    hdf5_attrs: dict[str, Any] = {}
    if hdf5_path:
        hdf5_attrs = _read_hdf5_attrs(Path(hdf5_path))

    joint_available = bool(
        raw.get("joint_action_available")
        or hdf5_attrs.get("joint_action_available")
        or "joint_actions" in (raw.get("availableActionKeys") or [])
        or "joint_actions" in (hdf5_attrs.get("available_action_keys") or [])
    )
    preferred = str(raw.get("preferredPolicySchemaId") or hdf5_attrs.get("preferred_policy_schema_id") or "")
    if joint_available or preferred == DEFAULT_JOINT_POLICY_SCHEMA:
        merged = {**raw, **{k: v for k, v in hdf5_attrs.items() if v is not None}}
        return _joint_spec_from_profile(merged)

    return _legacy_eef_spec({**raw, **{k: v for k, v in hdf5_attrs.items() if v is not None}})


def _load_checkpoint_train_config(checkpoint_path: Path) -> dict[str, Any]:
    from app.services.model_asset_checkpoint_resolver import _load_checkpoint_payload

    payload = _load_checkpoint_payload(checkpoint_path)
    if not payload:
        return {}
    return payload.get("train_config") if isinstance(payload.get("train_config"), dict) else {}


def resolve_dp_eval_executor(
    *,
    policy: str,
    model_asset: dict[str, Any] | None = None,
    checkpoint_path: str | Path | None = None,
) -> DpEvalExecutorSpec:
    """Resolve eval executor. Expert / non-DP policies always use legacy OSC pose env."""
    if policy in {"scripted", "random"}:
        return DpEvalExecutorSpec(
            eval_executor="osc_pose",
            controller_type="OSC_POSE",
            action_mode="expert",
            policy_type="expert",
            side_channel_mode="policy",
            source="expert_default",
        )
    if policy != "diffusion_policy":
        return DpEvalExecutorSpec(
            eval_executor="osc_pose",
            controller_type="OSC_POSE",
            action_mode="legacy",
            policy_type=policy,
            side_channel_mode="policy",
            source="legacy_non_dp",
        )

    asset = dict(model_asset or {})
    train_config = _load_checkpoint_train_config(Path(checkpoint_path).expanduser()) if checkpoint_path else {}

    eval_executor = str(asset.get("evalExecutor") or train_config.get("eval_executor") or "").strip()
    action_mode = str(
        asset.get("trainedActionMode")
        or asset.get("actionMode")
        or train_config.get("trained_action_mode")
        or train_config.get("action_mode")
        or ""
    ).strip()
    controller_type = str(
        asset.get("controllerType")
        or (asset.get("controllerSchema") or {}).get("controllerType")
        if isinstance(asset.get("controllerSchema"), dict)
        else asset.get("controllerSchema")
        or train_config.get("controller_type")
        or "OSC_POSE"
    ).strip()

    action_schema = asset.get("actionSchema")
    if isinstance(action_schema, str) and "joint" in action_schema:
        eval_executor = eval_executor or "joint_position"
        action_mode = action_mode or "joint_delta"
        controller_type = "JOINT_POSITION"

    if action_mode in JOINT_ACTION_MODES or eval_executor == "joint_position":
        return DpEvalExecutorSpec(
            eval_executor="joint_position",
            controller_type="JOINT_POSITION",
            action_mode=action_mode or "joint_delta",
            policy_type="diffusion_policy",
            side_channel_mode="policy",
            source="model_schema",
        )

    return DpEvalExecutorSpec(
        eval_executor="osc_pose",
        controller_type=controller_type or "OSC_POSE",
        action_mode=action_mode or "osc_pose_delta_eef",
        policy_type="diffusion_policy",
        side_channel_mode="policy",
        source="legacy_default",
    )
