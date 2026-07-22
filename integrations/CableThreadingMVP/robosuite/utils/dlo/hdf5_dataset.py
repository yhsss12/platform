"""
hdf5_dataset.py -- HDF5 数据集保存工具

提供统一的 HDF5 数据集保存功能，遵循 robomimic 兼容格式。
支持低维观测、任务观测和图像观测；训练时可从完整 obs 中按配置选取子集。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np

logger = logging.getLogger(__name__)

# 图像观测（默认）
HDF5_IMAGE_KEYS = ("agentview_image", "robot0_eye_in_hand_image")

# 机器人 proprio + eef（默认低维）
HDF5_LOW_DIM_KEYS = (
    "robot0_joint_pos",
    "robot0_joint_vel",
    "robot0_gripper_qpos",
    "robot0_gripper_qvel",
    "robot0_eef_pos",
    "robot0_eef_quat",
)

# 旧版仅 eef pose（向后兼容引用）
HDF5_LEGACY_LOW_DIM_KEYS = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")

# cable_threading 任务观测（默认）
HDF5_TASK_OBS_KEYS = (
    "attachment_state",
    "cable_end_pos",
    "pole_points",
    "endpoint_goal_pos",
    "cable_points",
    "physical_grasp_state",
    "object-state",
)

PREFERRED_POLICY_SCHEMAS: dict[str, dict[str, Any]] = {
    "joint_state_dp": {
        "image_keys": ["agentview_image", "robot0_eye_in_hand_image"],
        "low_dim_keys": ["robot0_joint_pos", "robot0_gripper_qpos"],
        "low_dim_dim": 9,
        "description": "video + joint angles + gripper",
    },
    "eef_pose_dp": {
        "image_keys": ["agentview_image", "robot0_eye_in_hand_image"],
        "low_dim_keys": ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
        "low_dim_dim": 9,
        "description": "video + end-effector pose + gripper",
    },
}

# 输入/输出分离的 policy schema（manifest / HDF5 attrs）
POLICY_SCHEMAS: dict[str, dict[str, Any]] = {
    "joint_state_obs_eef_action": {
        "input": {
            "image_keys": ["agentview_image", "robot0_eye_in_hand_image"],
            "low_dim_keys": ["robot0_joint_pos", "robot0_gripper_qpos"],
        },
        "output": {
            "action_key": "actions",
            "action_mode": "osc_pose_delta_eef",
            "action_dim": 7,
        },
        "note": "输入为关节状态 + 视频；输出仍为当前 OSC_POSE 末端 delta + gripper 原始 actions。",
    },
    "joint_state_obs_joint_action": {
        "input": {
            "image_keys": ["agentview_image", "robot0_eye_in_hand_image"],
            "low_dim_keys": ["robot0_joint_pos", "robot0_gripper_qpos"],
        },
        "output": {
            "action_key": "joint_actions",
            "gripper_action_key": "gripper_actions",
            "action_mode": "joint_delta_derived",
            "action_dim": 7,
            "gripper_action_dim": 1,
        },
        "note": "joint_actions 由 robot0_joint_pos 序列差分导出，非控制器原生输出；评测需 joint controller。",
    },
    "eef_pose_obs_eef_action": {
        "input": {
            "image_keys": ["agentview_image", "robot0_eye_in_hand_image"],
            "low_dim_keys": ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
        },
        "output": {
            "action_key": "actions",
            "action_mode": "osc_pose_delta_eef",
            "action_dim": 7,
        },
        "note": "旧版 eef pose 输入 + 原始 OSC_POSE actions。",
    },
}

CURRENT_ACTION_MODE = "osc_pose_delta_eef"
CONTROLLER_TYPE = "OSC_POSE"
JOINT_ACTION_MODE_DERIVED = "joint_delta_derived"


def extract_obs_key(obs_dict, key):
    """从 obs dict 中安全提取指定键。"""
    if obs_dict is None:
        return None
    value = obs_dict.get(key, None)
    if value is None:
        return None
    return np.asarray(value)


def _is_serializable_obs_array(value: Any) -> bool:
    if value is None:
        return False
    try:
        arr = np.asarray(value)
    except (TypeError, ValueError):
        return False
    if arr.dtype == object:
        return False
    if np.issubdtype(arr.dtype, np.number) or arr.dtype == np.bool_:
        return True
    if arr.dtype == np.uint8 and arr.ndim >= 3:
        return True
    return False


def _is_image_obs(value: np.ndarray, key: str) -> bool:
    if value.dtype == np.uint8 and value.ndim >= 3:
        return True
    return key.endswith("_image") or "image" in key.lower()


def _discover_obs_keys(raw_obs_list: list[dict[str, Any]]) -> list[str]:
    keys: set[str] = set()
    for obs in raw_obs_list:
        if not obs:
            continue
        for key, value in obs.items():
            if _is_serializable_obs_array(value):
                keys.add(str(key))
    return sorted(keys)


def _resolve_keys_to_save(
    *,
    obs_keys: list[str] | tuple[str, ...] | None,
    low_dim_keys: list[str] | tuple[str, ...] | None,
    image_keys: list[str] | tuple[str, ...] | None,
    task_obs_keys: list[str] | tuple[str, ...] | None,
    save_all_obs: bool,
    raw_obs_list: list[dict[str, Any]],
) -> tuple[list[str], list[str], list[str], list[str]]:
    if save_all_obs:
        discovered = _discover_obs_keys(raw_obs_list)
        image: list[str] = []
        low_dim: list[str] = []
        for key in discovered:
            sample = None
            for obs in raw_obs_list:
                if key in obs:
                    sample = np.asarray(obs[key])
                    break
            if sample is None:
                continue
            if _is_image_obs(sample, key):
                image.append(key)
            else:
                low_dim.append(key)
        return discovered, image, low_dim, []

    resolved_image = list(image_keys if image_keys is not None else HDF5_IMAGE_KEYS)
    resolved_low_dim = list(low_dim_keys if low_dim_keys is not None else HDF5_LOW_DIM_KEYS)
    resolved_task = list(task_obs_keys if task_obs_keys is not None else HDF5_TASK_OBS_KEYS)

    if obs_keys is not None:
        ordered = list(obs_keys)
        return ordered, resolved_image, resolved_low_dim, resolved_task

    combined: list[str] = []
    seen: set[str] = set()
    for key in (*resolved_image, *resolved_low_dim, *resolved_task):
        if key not in seen:
            combined.append(key)
            seen.add(key)
    return combined, resolved_image, resolved_low_dim, resolved_task


def _stack_obs_timesteps(values: list[np.ndarray | None]) -> np.ndarray | None:
    if not values or values[0] is None:
        return None
    cleaned = [np.asarray(v) for v in values if v is not None]
    if len(cleaned) != len(values):
        return None
    first = cleaned[0]
    if first.ndim == 0:
        stacked = np.asarray([float(v) for v in cleaned], dtype=np.float32)
        return stacked.reshape(-1, 1)
    return np.stack(cleaned, axis=0)


def _write_obs_dataset(obs_grp, key: str, stacked: np.ndarray) -> tuple[tuple[int, ...], str]:
    if _is_image_obs(stacked, key):
        if stacked.dtype != np.uint8:
            stacked = stacked.astype(np.uint8)
        obs_grp.create_dataset(
            key,
            data=stacked,
            compression="gzip",
            compression_opts=4,
            chunks=(1, *stacked.shape[1:]),
        )
        return tuple(stacked.shape), str(stacked.dtype)
    numeric = stacked.astype(np.float32)
    obs_grp.create_dataset(key, data=numeric)
    return tuple(numeric.shape), str(numeric.dtype)


def split_episode_names(names, val_ratio=0.1, test_ratio=0.1):
    """将 episode 名称列表划分为 train/valid/test。"""
    num_eps = len(names)
    num_test = int(round(num_eps * test_ratio))
    num_val = int(round(num_eps * val_ratio))
    if num_test + num_val >= num_eps and num_eps > 1:
        overflow = num_test + num_val - num_eps + 1
        num_test = max(0, num_test - overflow)
    num_train = max(1, num_eps - num_val - num_test) if num_eps else 0

    train = names[:num_train]
    val = names[num_train : num_train + num_val]
    test = names[num_train + num_val :]
    return train, val, test


def derive_joint_delta_actions(
    raw_obs_list: list[dict[str, Any]],
    *,
    joint_key: str = "robot0_joint_pos",
) -> np.ndarray | None:
    """从 post-step joint_pos 序列构造派生关节 delta 标签 (T, dof)。

    raw_obs_list[t] 为执行 action[t] 后的观测，因此
    joint_delta[t] = joint_pos[t+1] - joint_pos[t]（最后一帧复制前一帧 delta）。
    """
    positions = []
    for obs in raw_obs_list:
        value = extract_obs_key(obs, joint_key)
        if value is None:
            return None
        positions.append(np.asarray(value, dtype=np.float32).reshape(-1))
    if len(positions) < 1:
        return None
    stacked = np.stack(positions, axis=0)
    if stacked.shape[0] == 1:
        return np.zeros((1, stacked.shape[1]), dtype=np.float32)
    deltas = np.diff(stacked, axis=0)
    deltas = np.vstack([deltas, deltas[-1:]])
    return deltas.astype(np.float32)


def derive_gripper_actions(actions: np.ndarray) -> np.ndarray:
    """从原始 env action 向量提取夹爪命令 (T, 1)。"""
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim == 1:
        return actions.reshape(1, 1)
    return actions[:, -1:].astype(np.float32)


def build_hdf5_manifest_fields(save_info: dict[str, Any]) -> dict[str, Any]:
    """从 save_dataset_hdf5 返回值构建 manifest 观测 + 动作 schema 字段。"""
    from robosuite.utils.dlo.hdf5_platform_schema import (
        build_platform_schema_bundle,
        flatten_schema_ids,
    )

    obs_fields = build_hdf5_manifest_obs_fields(save_info)
    available_actions = list(save_info.get("available_action_keys") or ["actions"])
    attachment_fields = {
        "attachment_side_channel": bool(save_info.get("attachment_side_channel")),
        "attachmentSideChannel": bool(save_info.get("attachment_side_channel")),
        "attachment_field": save_info.get("attachment_field", "attachment_enabled"),
        "attachmentField": save_info.get("attachment_field", "attachment_enabled"),
        "side_channel_keys": list(save_info.get("side_channel_keys") or []),
        "sideChannelKeys": list(save_info.get("side_channel_keys") or []),
        "attachment_policy": save_info.get("attachment_policy"),
        "attachmentPolicy": save_info.get("attachment_policy"),
        "attachmentInputMode": save_info.get("attachment_input_mode", "not_used_by_policy"),
        "attachmentControlMode": save_info.get("attachment_control_mode", "eval_controller"),
        "includeAttachmentObs": bool(save_info.get("include_attachment_obs", False)),
    }
    metadata = dict(save_info.get("platform_metadata") or {})
    schema_bundle = build_platform_schema_bundle(save_info, metadata)
    schema_ids = flatten_schema_ids(schema_bundle)
    return {
        **obs_fields,
        **attachment_fields,
        **schema_ids,
        "observationSchemaDetail": schema_bundle.get("observationSchema"),
        "actionSchemaDetail": schema_bundle.get("actionSchema"),
        "controllerSchemaDetail": schema_bundle.get("controllerSchema"),
        "sideChannelSchemaDetail": schema_bundle.get("sideChannelSchema"),
        "successMetricSchemaDetail": schema_bundle.get("successMetricSchema"),
        "taskTemplateId": schema_bundle.get("taskTemplateId"),
        "taskType": schema_bundle.get("taskType"),
        "simulator": schema_bundle.get("simulator"),
        "robotType": schema_bundle.get("robotType"),
        "envArgs": schema_bundle.get("envArgs"),
        "preferredPolicySchemaId": schema_bundle.get("preferredPolicySchemaId"),
        "availableActionKeys": available_actions,
        "policySchemas": dict(save_info.get("policy_schemas") or POLICY_SCHEMAS),
        "action_dim": int(save_info.get("action_dim") or 7),
        "current_action_mode": str(save_info.get("current_action_mode") or CURRENT_ACTION_MODE),
        "controller_type": str(save_info.get("controller_type") or CONTROLLER_TYPE),
        "joint_action_available": bool(save_info.get("joint_action_available")),
        "joint_action_mode": save_info.get("joint_action_mode"),
        "gripper_action_available": bool(save_info.get("gripper_action_available")),
        "derived_action_note": save_info.get("derived_action_note"),
    }


def build_hdf5_manifest_obs_fields(save_info: dict[str, Any]) -> dict[str, Any]:
    """从 save_dataset_hdf5 返回值构建 manifest 观测 schema 字段。"""
    available = list(save_info.get("available_obs_keys") or [])
    image_keys = list(save_info.get("image_keys") or [])
    low_dim_keys = list(save_info.get("low_dim_keys") or [])
    task_obs_keys = list(save_info.get("task_obs_keys") or [])
    all_low_dim = sorted(
        set(low_dim_keys + task_obs_keys),
        key=lambda k: (k not in low_dim_keys, k),
    )
    return {
        "availableObservationKeys": available,
        "imageKeys": image_keys,
        "lowDimKeys": all_low_dim,
        "preferredPolicySchemas": dict(save_info.get("preferred_policy_schemas") or PREFERRED_POLICY_SCHEMAS),
        "robot_state_available": bool(save_info.get("robot_state_available")),
        "task_state_available": bool(save_info.get("task_state_available")),
        "missingObservationKeys": list(save_info.get("missing_obs_keys") or []),
    }


def validate_hdf5_trajectory_actions(
    hdf5_path,
    source_trajectories: list[list[dict[str, Any]]],
    *,
    atol: float = 1e-5,
) -> dict[str, Any]:
    """校验 HDF5 中 actions / attachment_enabled 与采集轨迹一致。"""
    path = Path(hdf5_path).expanduser()
    issues: list[str] = []
    max_action_diff = 0.0
    if not path.is_file():
        return {"ok": False, "issues": [f"HDF5 missing: {path}"], "max_action_diff": None}

    with h5py.File(path, "r") as handle:
        data = handle["data"]
        for ep_idx, traj in enumerate(source_trajectories):
            demo = f"demo_{ep_idx}"
            if demo not in data:
                issues.append(f"{demo} missing in HDF5")
                continue
            grp = data[demo]
            h5_actions = np.asarray(grp["actions"], dtype=np.float32)
            src_actions = np.stack([np.asarray(step["action"], dtype=np.float32) for step in traj], axis=0)
            if h5_actions.shape != src_actions.shape:
                issues.append(
                    f"{demo}: action shape {h5_actions.shape} != source {src_actions.shape}"
                )
            else:
                diff = float(np.max(np.abs(h5_actions - src_actions)))
                max_action_diff = max(max_action_diff, diff)
                if diff > atol:
                    issues.append(f"{demo}: max action diff {diff} > {atol}")
            if "attachment_enabled" in grp:
                h5_attach = np.asarray(grp["attachment_enabled"], dtype=bool)
                src_attach = np.asarray(
                    [bool(step.get("attachment_enabled", False)) for step in traj],
                    dtype=bool,
                )
                if h5_attach.shape[0] != src_attach.shape[0]:
                    issues.append(f"{demo}: attachment length mismatch")
                elif not np.array_equal(h5_attach, src_attach):
                    issues.append(f"{demo}: attachment_enabled mismatch")

    return {
        "ok": not issues,
        "issues": issues,
        "max_action_diff": max_action_diff if source_trajectories else 0.0,
    }


def save_dataset_hdf5(
    path,
    trajectories,
    *,
    obs_keys=None,
    low_dim_keys=None,
    image_keys=None,
    task_obs_keys=None,
    save_all_obs: bool = False,
    metadata=None,
    episode_metadata=None,
    val_ratio=0.1,
    test_ratio=0.1,
) -> dict[str, Any]:
    """将专家轨迹保存为 HDF5 数据集（robomimic 兼容格式）。

    每个 transition 必须包含 "raw_obs" 键（dict），用于提取各观测分量。
    返回保存摘要（available_obs_keys、missing_obs_keys、shape 等）供 manifest 使用。
    """
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)

    has_raw_obs = any("raw_obs" in step for traj in trajectories for step in traj[:1])
    all_raw_obs: list[dict[str, Any]] = []
    if has_raw_obs:
        for traj in trajectories:
            for step in traj:
                obs = step.get("raw_obs")
                if obs:
                    all_raw_obs.append(obs)

    keys_to_save, resolved_image, resolved_low_dim, resolved_task = _resolve_keys_to_save(
        obs_keys=obs_keys,
        low_dim_keys=low_dim_keys,
        image_keys=image_keys,
        task_obs_keys=task_obs_keys,
        save_all_obs=save_all_obs,
        raw_obs_list=all_raw_obs,
    )

    saved_keys: list[str] = []
    missing_keys: list[str] = []
    saved_shapes: dict[str, tuple[int, ...]] = {}
    available_action_keys: list[str] = ["actions"]
    joint_action_available = False
    gripper_action_available = False
    joint_action_mode: str | None = None
    derived_action_note: str | None = None

    with h5py.File(path, "w") as f:
        data_grp = f.create_group("data")
        env_name = (metadata or {}).get("env_name", "unknown")
        data_grp.attrs["env"] = env_name
        data_grp.attrs["env_args"] = json.dumps(metadata or {}, ensure_ascii=False)
        total_steps = sum(len(traj) for traj in trajectories)
        data_grp.attrs["total"] = total_steps
        data_grp.attrs["success_semantics"] = (metadata or {}).get("success_semantics", "")
        data_grp.attrs["obs_keys"] = json.dumps(keys_to_save, ensure_ascii=False)
        data_grp.attrs["image_keys"] = json.dumps(resolved_image, ensure_ascii=False)
        data_grp.attrs["low_dim_keys"] = json.dumps(resolved_low_dim, ensure_ascii=False)
        data_grp.attrs["task_obs_keys"] = json.dumps(resolved_task, ensure_ascii=False)
        data_grp.attrs["current_action_mode"] = str((metadata or {}).get("current_action_mode", CURRENT_ACTION_MODE))
        data_grp.attrs["controller_type"] = str((metadata or {}).get("controller_type", CONTROLLER_TYPE))
        if metadata:
            if metadata.get("grasp_mode"):
                data_grp.attrs["grasp_mode"] = str(metadata["grasp_mode"])
            if metadata.get("attachment_side_channel"):
                data_grp.attrs["attachment_side_channel"] = True
                field = str(metadata.get("attachment_field", "attachment_enabled"))
                data_grp.attrs["attachment_field"] = field
                data_grp.attrs["side_channel_keys"] = json.dumps([field], ensure_ascii=False)
                policy = metadata.get("attachment_policy", "recorded_or_controller")
                data_grp.attrs["attachment_policy"] = str(policy)

        demo_names = []
        for episode_idx, traj in enumerate(trajectories):
            demo_name = f"demo_{episode_idx}"
            demo_names.append(demo_name)
            demo_grp = data_grp.create_group(demo_name)
            demo_grp.attrs["num_samples"] = len(traj)
            if episode_metadata and episode_idx < len(episode_metadata):
                ep_meta = episode_metadata[episode_idx]
                demo_grp.attrs["benchmark_episode_metadata"] = json.dumps(ep_meta, ensure_ascii=False)
                if ep_meta.get("seed") is not None:
                    demo_grp.attrs["seed"] = int(ep_meta["seed"])
                if ep_meta.get("episode") is not None:
                    demo_grp.attrs["episode_index"] = int(ep_meta["episode"])

            actions = np.stack([np.asarray(step["action"], dtype=np.float32) for step in traj], axis=0)
            rewards = np.asarray([float(step["reward"]) for step in traj], dtype=np.float32)
            dones = np.asarray([bool(step["done"]) for step in traj], dtype=np.int32)
            demo_grp.create_dataset("actions", data=actions)
            demo_grp.create_dataset("rewards", data=rewards)
            demo_grp.create_dataset("dones", data=dones)

            gripper_actions = derive_gripper_actions(actions)
            demo_grp.create_dataset("gripper_actions", data=gripper_actions)
            if "gripper_actions" not in available_action_keys:
                available_action_keys.append("gripper_actions")
            gripper_action_available = True

            if has_raw_obs:
                raw_obs_list = [step.get("raw_obs", {}) for step in traj]
                joint_actions = derive_joint_delta_actions(raw_obs_list)
                if joint_actions is not None and joint_actions.shape[0] == len(traj):
                    demo_grp.create_dataset("joint_actions", data=joint_actions)
                    if "joint_actions" not in available_action_keys:
                        available_action_keys.append("joint_actions")
                    joint_action_available = True
                    joint_action_mode = JOINT_ACTION_MODE_DERIVED
                    derived_action_note = (
                        "joint_actions = diff(robot0_joint_pos) from post-step observations; "
                        "not native controller output. Eval requires joint-space controller."
                    )
                    if episode_idx == 0:
                        logger.info(
                            "HDF5 joint_actions: shape=%s dtype=float32 (derived)",
                            joint_actions.shape,
                        )

            attach_vals = [bool(step.get("attachment_enabled", False)) for step in traj]
            if attach_vals or (metadata or {}).get("attachment_side_channel"):
                demo_grp.create_dataset(
                    "attachment_enabled",
                    data=np.asarray(attach_vals, dtype=np.bool_),
                )

            obs_grp = demo_grp.create_group("obs")

            if has_raw_obs:
                raw_obs_list = [step.get("raw_obs", {}) for step in traj]
                for key in keys_to_save:
                    values = [extract_obs_key(obs, key) for obs in raw_obs_list]
                    if values[0] is None:
                        if key not in missing_keys:
                            missing_keys.append(key)
                            logger.warning("HDF5 export: missing obs key %r in demo %s", key, demo_name)
                        continue
                    stacked = _stack_obs_timesteps(values)
                    if stacked is None:
                        if key not in missing_keys:
                            missing_keys.append(key)
                            logger.warning("HDF5 export: incomplete timesteps for obs key %r in demo %s", key, demo_name)
                        continue
                    if stacked.shape[0] != len(traj):
                        logger.warning(
                            "HDF5 export: length mismatch for %r in %s: obs=%s actions=%s",
                            key,
                            demo_name,
                            stacked.shape[0],
                            len(traj),
                        )
                        continue
                    shape, dtype = _write_obs_dataset(obs_grp, key, stacked)
                    if key not in saved_keys:
                        saved_keys.append(key)
                    saved_shapes[key] = shape
                    if episode_idx == 0:
                        logger.info("HDF5 obs %s: shape=%s dtype=%s", key, shape, dtype)

        data_grp.attrs["available_obs_keys"] = json.dumps(saved_keys, ensure_ascii=False)
        data_grp.attrs["missing_obs_keys"] = json.dumps(missing_keys, ensure_ascii=False)
        data_grp.attrs["available_action_keys"] = json.dumps(available_action_keys, ensure_ascii=False)
        sample_action_dim = (
            int(np.asarray(trajectories[0][0]["action"]).shape[0]) if trajectories and trajectories[0] else 7
        )
        data_grp.attrs["action_dim"] = sample_action_dim
        data_grp.attrs["joint_action_available"] = bool(joint_action_available)
        data_grp.attrs["gripper_action_available"] = bool(gripper_action_available)
        if joint_action_mode:
            data_grp.attrs["joint_action_mode"] = joint_action_mode
        if derived_action_note:
            data_grp.attrs["derived_action_note"] = derived_action_note
        data_grp.attrs["preferred_policy_schemas"] = json.dumps(
            PREFERRED_POLICY_SCHEMAS, ensure_ascii=False
        )
        data_grp.attrs["policy_schemas"] = json.dumps(POLICY_SCHEMAS, ensure_ascii=False)
        platform_metadata = dict((metadata or {}))
        platform_metadata.setdefault("taskTemplateId", "cable_threading_single_arm")
        platform_metadata.setdefault("taskType", "cable_threading")
        platform_metadata.setdefault("simulatorBackend", "mujoco")
        from robosuite.utils.dlo.hdf5_platform_schema import (
            build_platform_schema_bundle,
        )

        pre_save_info = {
            "available_obs_keys": saved_keys,
            "image_keys": [k for k in resolved_image if k in saved_keys],
            "low_dim_keys": [k for k in resolved_low_dim if k in saved_keys],
            "task_obs_keys": [k for k in resolved_task if k in saved_keys],
            "available_action_keys": available_action_keys,
            "action_dim": sample_action_dim,
            "current_action_mode": (metadata or {}).get("current_action_mode", CURRENT_ACTION_MODE),
            "controller_type": (metadata or {}).get("controller_type", CONTROLLER_TYPE),
            "joint_action_available": bool(joint_action_available),
            "joint_action_mode": joint_action_mode,
            "gripper_action_available": bool(gripper_action_available),
            "attachment_side_channel": bool((metadata or {}).get("attachment_side_channel")),
            "attachment_field": (metadata or {}).get("attachment_field", "attachment_enabled"),
            "side_channel_keys": list((metadata or {}).get("side_channel_keys") or ["attachment_enabled"]),
            "attachment_control_mode": (metadata or {}).get("attachment_control_mode", "eval_controller"),
        }
        schema_bundle = build_platform_schema_bundle(pre_save_info, platform_metadata)
        data_grp.attrs["observation_schema"] = json.dumps(schema_bundle["observationSchema"], ensure_ascii=False)
        data_grp.attrs["action_schema"] = json.dumps(schema_bundle["actionSchema"], ensure_ascii=False)
        data_grp.attrs["controller_schema"] = json.dumps(schema_bundle["controllerSchema"], ensure_ascii=False)
        data_grp.attrs["side_channel_schema"] = json.dumps(schema_bundle["sideChannelSchema"], ensure_ascii=False)
        data_grp.attrs["success_metric_schema"] = json.dumps(schema_bundle["successMetricSchema"], ensure_ascii=False)
        data_grp.attrs["task_template_id"] = schema_bundle.get("taskTemplateId", "")
        data_grp.attrs["task_type"] = schema_bundle.get("taskType", "")
        data_grp.attrs["preferred_policy_schema_id"] = schema_bundle.get("preferredPolicySchemaId", "")
        data_grp.attrs["trained_action_mode"] = schema_bundle.get("trainedActionMode", "")
        data_grp.attrs["eval_executor"] = schema_bundle.get("evalExecutor", "")

        mask_grp = f.create_group("mask")
        train_names, val_names, test_names = split_episode_names(
            demo_names, val_ratio=val_ratio, test_ratio=test_ratio,
        )
        mask_grp.create_dataset("train", data=np.asarray(train_names, dtype="S"))
        if val_names:
            mask_grp.create_dataset("valid", data=np.asarray(val_names, dtype="S"))
        if test_names:
            mask_grp.create_dataset("test", data=np.asarray(test_names, dtype="S"))

    robot_state_available = "robot0_joint_pos" in saved_keys
    task_state_available = any(
        k in saved_keys
        for k in (
            "attachment_state",
            "cable_end_pos",
            "pole_points",
            "endpoint_goal_pos",
            "cable_points",
            "physical_grasp_state",
            "object-state",
        )
    )

    save_info = {
        "available_obs_keys": saved_keys,
        "image_keys": [k for k in resolved_image if k in saved_keys],
        "low_dim_keys": [k for k in resolved_low_dim if k in saved_keys],
        "task_obs_keys": [k for k in resolved_task if k in saved_keys],
        "missing_obs_keys": missing_keys,
        "saved_obs_shapes": saved_shapes,
        "preferred_policy_schemas": PREFERRED_POLICY_SCHEMAS,
        "policy_schemas": POLICY_SCHEMAS,
        "robot_state_available": robot_state_available,
        "task_state_available": task_state_available,
        "available_action_keys": available_action_keys,
        "action_dim": int(
            np.asarray(trajectories[0][0]["action"]).shape[0] if trajectories and trajectories[0] else 7
        ),
        "current_action_mode": (metadata or {}).get("current_action_mode", CURRENT_ACTION_MODE),
        "controller_type": (metadata or {}).get("controller_type", CONTROLLER_TYPE),
        "joint_action_available": joint_action_available,
        "joint_action_mode": joint_action_mode,
        "gripper_action_available": gripper_action_available,
        "derived_action_note": derived_action_note,
        "attachment_side_channel": bool((metadata or {}).get("attachment_side_channel")),
        "attachment_field": (metadata or {}).get("attachment_field", "attachment_enabled"),
        "side_channel_keys": list((metadata or {}).get("side_channel_keys") or ["attachment_enabled"]),
        "attachment_policy": (metadata or {}).get("attachment_policy"),
        "attachment_input_mode": (metadata or {}).get("attachment_input_mode", "not_used_by_policy"),
        "attachment_control_mode": (metadata or {}).get("attachment_control_mode", "eval_controller"),
        "include_attachment_obs": bool((metadata or {}).get("include_attachment_obs", False)),
        "platform_metadata": dict(metadata or {}),
    }

    print(f"saved_hdf5: {path}")
    print(f"episodes: {len(trajectories)}")
    print(f"total_steps: {total_steps}")
    print(f"train_split: {len(train_names)}")
    print(f"valid_split: {len(val_names)}")
    print(f"test_split: {len(test_names)}")
    if has_raw_obs:
        print(f"available_obs_keys: {saved_keys}")
        print(f"available_action_keys: {available_action_keys}")
        print(f"image_keys: {save_info['image_keys']}")
        print(f"low_dim_keys: {save_info['low_dim_keys']}")
        print(f"task_obs_keys: {save_info['task_obs_keys']}")
        if missing_keys:
            print(f"missing_obs_keys: {missing_keys}")
    else:
        print("warning: no raw_obs found, only actions/rewards/dones saved")

    return save_info
