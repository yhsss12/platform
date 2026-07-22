"""Platform-level HDF5 / manifest schema descriptors for cable_threading datasets."""
from __future__ import annotations

from typing import Any

OBS_SCHEMA_JOINT = "cable_threading_joint_obs_v1"
OBS_SCHEMA_EEF = "cable_threading_eef_obs_v1"
ACTION_SCHEMA_JOINT = "cable_threading_joint_delta_v1"
ACTION_SCHEMA_EEF = "cable_threading_osc_pose_v1"
CONTROLLER_SCHEMA_JOINT = "panda_joint_position_v1"
CONTROLLER_SCHEMA_EEF = "panda_osc_pose_v1"
SIDE_CHANNEL_SCHEMA = "cable_threading_attachment_side_channel_v1"
SUCCESS_METRIC_SCHEMA = "cable_threading_success_v1"
DEFAULT_DP_POLICY_SCHEMA_ID = "joint_state_obs_joint_action"
LEGACY_DP_POLICY_SCHEMA_ID = "eef_pose_obs_eef_action"

JOINT_ACTION_MODES = frozenset({"joint_delta", "joint_delta_derived"})


def build_observation_schema(
    *,
    schema_id: str,
    image_keys: list[str],
    low_dim_keys: list[str],
    task_obs_keys: list[str] | None = None,
    policy_input: bool = True,
) -> dict[str, Any]:
    return {
        "id": schema_id,
        "version": 1,
        "imageKeys": list(image_keys),
        "lowDimKeys": list(low_dim_keys),
        "taskObsKeys": list(task_obs_keys or []),
        "policyInput": bool(policy_input),
    }


def build_action_schema(
    *,
    schema_id: str,
    action_key: str,
    action_mode: str,
    action_dim: int,
    gripper_action_key: str | None = None,
    gripper_action_dim: int = 0,
) -> dict[str, Any]:
    total_dim = int(action_dim) + (int(gripper_action_dim) if gripper_action_key else 0)
    return {
        "id": schema_id,
        "version": 1,
        "actionKey": action_key,
        "gripperActionKey": gripper_action_key,
        "actionMode": action_mode,
        "actionDim": int(action_dim),
        "gripperActionDim": int(gripper_action_dim),
        "totalActionDim": total_dim,
    }


def build_controller_schema(
    *,
    schema_id: str,
    controller_type: str,
    eval_executor: str,
    robot_type: str = "Panda",
) -> dict[str, Any]:
    return {
        "id": schema_id,
        "version": 1,
        "controllerType": controller_type,
        "evalExecutor": eval_executor,
        "robotType": robot_type,
    }


def build_side_channel_schema(
    *,
    schema_id: str,
    side_channel_keys: list[str],
    attachment_field: str = "attachment_enabled",
    policy_input: bool = False,
    eval_control_mode: str = "eval_controller",
) -> dict[str, Any]:
    return {
        "id": schema_id,
        "version": 1,
        "sideChannelKeys": list(side_channel_keys),
        "attachmentField": attachment_field,
        "policyInput": bool(policy_input),
        "evalControlMode": eval_control_mode,
    }


def build_success_metric_schema(*, schema_id: str = SUCCESS_METRIC_SCHEMA) -> dict[str, Any]:
    return {
        "id": schema_id,
        "version": 1,
        "finalSuccessKey": "final_success",
        "everSuccessKey": "ever_success",
        "threadCompletionKey": "thread_completion_max",
    }


def _joint_policy_bundle(save_info: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    image_keys = list(save_info.get("image_keys") or [])
    low_dim_keys = ["robot0_joint_pos", "robot0_gripper_qpos"]
    side_keys = list(save_info.get("side_channel_keys") or ["attachment_enabled"])
    robot = str(metadata.get("robot") or "Panda")
    return {
        "observationSchema": build_observation_schema(
            schema_id=OBS_SCHEMA_JOINT,
            image_keys=image_keys,
            low_dim_keys=low_dim_keys,
            task_obs_keys=list(save_info.get("task_obs_keys") or []),
        ),
        "actionSchema": build_action_schema(
            schema_id=ACTION_SCHEMA_JOINT,
            action_key="joint_actions",
            gripper_action_key="gripper_actions",
            action_mode="joint_delta",
            action_dim=7,
            gripper_action_dim=1,
        ),
        "controllerSchema": build_controller_schema(
            schema_id=CONTROLLER_SCHEMA_JOINT,
            controller_type="JOINT_POSITION",
            eval_executor="joint_position",
            robot_type=robot,
        ),
        "sideChannelSchema": build_side_channel_schema(
            schema_id=SIDE_CHANNEL_SCHEMA,
            side_channel_keys=side_keys,
            attachment_field=str(save_info.get("attachment_field") or "attachment_enabled"),
            policy_input=False,
            eval_control_mode=str(save_info.get("attachment_control_mode") or "eval_controller"),
        ),
        "successMetricSchema": build_success_metric_schema(),
        "preferredPolicySchemaId": DEFAULT_DP_POLICY_SCHEMA_ID,
        "trainedActionMode": "joint_delta",
        "evalExecutor": "joint_position",
    }


def _eef_policy_bundle(save_info: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    image_keys = list(save_info.get("image_keys") or [])
    low_dim_keys = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
    side_keys = list(save_info.get("side_channel_keys") or ["attachment_enabled"])
    robot = str(metadata.get("robot") or "Panda")
    return {
        "observationSchema": build_observation_schema(
            schema_id=OBS_SCHEMA_EEF,
            image_keys=image_keys,
            low_dim_keys=low_dim_keys,
            task_obs_keys=list(save_info.get("task_obs_keys") or []),
        ),
        "actionSchema": build_action_schema(
            schema_id=ACTION_SCHEMA_EEF,
            action_key="actions",
            action_mode="osc_pose_delta_eef",
            action_dim=int(save_info.get("action_dim") or 7),
        ),
        "controllerSchema": build_controller_schema(
            schema_id=CONTROLLER_SCHEMA_EEF,
            controller_type="OSC_POSE",
            eval_executor="osc_pose",
            robot_type=robot,
        ),
        "sideChannelSchema": build_side_channel_schema(
            schema_id=SIDE_CHANNEL_SCHEMA,
            side_channel_keys=side_keys,
            attachment_field=str(save_info.get("attachment_field") or "attachment_enabled"),
            policy_input=False,
            eval_control_mode=str(save_info.get("attachment_control_mode") or "eval_controller"),
        ),
        "successMetricSchema": build_success_metric_schema(),
        "preferredPolicySchemaId": LEGACY_DP_POLICY_SCHEMA_ID,
        "trainedActionMode": "osc_pose_delta_eef",
        "evalExecutor": "osc_pose",
    }


def build_platform_schema_bundle(
    save_info: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build platform schema bundle for manifest + HDF5 attrs."""
    metadata = metadata or {}
    joint_available = bool(save_info.get("joint_action_available"))
    bundle = _joint_policy_bundle(save_info, metadata) if joint_available else _eef_policy_bundle(save_info, metadata)
    task_template_id = str(metadata.get("taskTemplateId") or "cable_threading_single_arm")
    task_type = str(metadata.get("taskType") or "cable_threading")
    return {
        **bundle,
        "taskTemplateId": task_template_id,
        "taskType": task_type,
        "simulator": str(metadata.get("simulator") or metadata.get("simulatorBackend") or "mujoco"),
        "robotType": str(metadata.get("robot") or metadata.get("robotType") or "Panda"),
        "envArgs": {
            "cable_model": metadata.get("cable_model"),
            "difficulty": metadata.get("difficulty"),
            "grasp_mode": metadata.get("grasp_mode"),
            "horizon": metadata.get("horizon"),
            "control_freq": metadata.get("control_freq"),
        },
    }


def flatten_schema_ids(bundle: dict[str, Any]) -> dict[str, Any]:
    """Top-level manifest fields for quick indexing."""
    obs = bundle.get("observationSchema") or {}
    act = bundle.get("actionSchema") or {}
    ctrl = bundle.get("controllerSchema") or {}
    side = bundle.get("sideChannelSchema") or {}
    return {
        "observationSchema": obs.get("id"),
        "actionSchema": act.get("id"),
        "controllerSchema": ctrl.get("id"),
        "sideChannelSchema": side.get("id"),
        "successMetricSchema": (bundle.get("successMetricSchema") or {}).get("id"),
        "preferredPolicySchemaId": bundle.get("preferredPolicySchemaId"),
        "trainedActionMode": bundle.get("trainedActionMode"),
        "evalExecutor": bundle.get("evalExecutor"),
    }
