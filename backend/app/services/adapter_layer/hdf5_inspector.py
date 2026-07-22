from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


ACTION_OBS_EXCLUDE = frozenset({"actions", "action"})
IMAGE_KEY_HINTS = ("image", "rgb", "camera", "agentview", "eye_in_hand")
LOW_DIM_KEY_HINTS = ("qpos", "eef", "gripper", "state", "pos", "quat", "vel")


@dataclass
class Hdf5InspectionResult:
    observation_keys: list[str] = field(default_factory=list)
    camera_keys: list[str] = field(default_factory=list)
    state_keys: list[str] = field(default_factory=list)
    action_dim: Optional[int] = None
    state_dim: Optional[int] = None
    episode_count: int = 0
    horizon: Optional[int] = None
    action_space: Optional[str] = None
    image_shape: Optional[dict[str, int]] = None
    has_reward: bool = False
    has_done: bool = False
    has_success: bool = False
    has_validation_split: bool = False
    attachment_side_channel: bool = False
    attachment_field: str = "attachment_enabled"
    side_channel_keys: list[str] = field(default_factory=list)
    attachment_policy: Optional[str] = None
    attachment_input_mode: str = "not_used_by_policy"
    attachment_control_mode: str = "eval_controller"
    controller_type: str = "OSC_POSE"
    joint_action_available: bool = False
    available_action_keys: list[str] = field(default_factory=list)
    preferred_policy_schema_id: str = ""
    trained_action_mode: str = ""
    eval_executor: str = ""
    observation_schema_id: str = ""
    action_schema_id: str = ""
    warnings: list[str] = field(default_factory=list)
    source: str = "none"


def _is_image_key(key: str, shape: tuple[Any, ...] | None = None) -> bool:
    lower = key.lower()
    if any(hint in lower for hint in IMAGE_KEY_HINTS):
        return True
    if shape and len(shape) >= 4 and int(shape[-1]) in {1, 3, 4}:
        return True
    return False


def _infer_action_space(action_dim: Optional[int], action_keys: list[str]) -> Optional[str]:
    if action_keys:
        joined = " ".join(action_keys).lower()
        if "delta" in joined or "pose" in joined:
            return "delta_pose"
        if "joint" in joined or "qpos" in joined:
            return "joint_position"
    if action_dim == 7:
        return "delta_pose"
    if action_dim == 14:
        return "joint_position"
    return None


def _obs_key_feature_dim(shape: tuple[Any, ...] | None) -> int:
    if not shape:
        return 0
    if len(shape) >= 2:
        return int(shape[-1]) if shape[-1] else 1
    if len(shape) == 1:
        return 1
    return 0


def sum_low_dim_key_dims(hdf5_path: Path | str, low_dim_keys: list[str]) -> Optional[int]:
    """Sum feature dimensions for the selected low-dim observation keys."""
    keys = [str(key).strip() for key in low_dim_keys if str(key).strip()]
    if not keys:
        return None

    path = Path(hdf5_path)
    if not path.is_file():
        return None

    try:
        import h5py
    except ImportError:
        return None

    try:
        with h5py.File(path, "r") as handle:
            data_group = handle.get("data")
            if data_group is None:
                return None
            demo_keys = sorted(k for k in data_group.keys() if str(k).startswith("demo_"))
            if not demo_keys:
                return None
            obs_group = data_group[demo_keys[0]].get("obs")
            if obs_group is None:
                return None
            total = 0
            for key in keys:
                if key not in obs_group:
                    return None
                ds = obs_group[key]
                dim = _obs_key_feature_dim(getattr(ds, "shape", None))
                if dim <= 0:
                    return None
                total += dim
            return total if total > 0 else None
    except OSError:
        return None


def inspect_hdf5(hdf5_path: Path | str) -> Hdf5InspectionResult:
    """从 HDF5 结构推断观测/动作元信息；无法读取时返回 warnings。"""
    path = Path(hdf5_path)
    result = Hdf5InspectionResult()

    if not path.is_file():
        result.warnings.append(f"HDF5 文件不存在: {path}")
        return result

    try:
        import h5py
    except ImportError:
        result.warnings.append("h5py 不可用，无法从 HDF5 推断结构")
        return result

    try:
        with h5py.File(path, "r") as handle:
            result.source = "hdf5"
            data_group = handle.get("data")
            if data_group is None:
                result.warnings.append("HDF5 缺少 data 分组，无法推断 demo 结构")
                return result

            demo_keys = sorted(k for k in data_group.keys() if str(k).startswith("demo_"))
            result.episode_count = len(demo_keys)
            if not demo_keys:
                result.warnings.append("HDF5 data 分组内无 demo_* 轨迹")
                return result

            first_demo = data_group[demo_keys[0]]
            obs_group = first_demo.get("obs")
            if obs_group is not None:
                state_total = 0
                for key in obs_group.keys():
                    key_str = str(key)
                    if key_str in ACTION_OBS_EXCLUDE:
                        continue
                    ds = obs_group[key]
                    shape = getattr(ds, "shape", None)
                    result.observation_keys.append(key_str)
                    if _is_image_key(key_str, shape):
                        result.camera_keys.append(key_str)
                        if result.image_shape is None and shape and len(shape) >= 4:
                            result.image_shape = {
                                "height": int(shape[-3]),
                                "width": int(shape[-2]),
                                "channels": int(shape[-1]),
                            }
                    else:
                        result.state_keys.append(key_str)
                        if shape and len(shape) >= 2:
                            state_total += int(shape[-1]) if shape[-1] else 1
                        elif shape and len(shape) == 1:
                            state_total += 1
                if state_total:
                    result.state_dim = state_total

            action_keys: list[str] = []
            actions = first_demo.get("actions")
            if actions is not None:
                shape = getattr(actions, "shape", None)
                if shape and len(shape) >= 2:
                    result.action_dim = int(shape[-1])
                    result.horizon = int(shape[0])
                elif shape and len(shape) == 1:
                    result.action_dim = int(shape[0])
                action_keys.append("actions")

            if first_demo.get("joint_actions") is not None:
                action_keys.append("joint_actions")
                result.joint_action_available = True
                joint_shape = getattr(first_demo.get("joint_actions"), "shape", None)
                if joint_shape and len(joint_shape) >= 2 and result.preferred_policy_schema_id == "":
                    grip_dim = 1 if first_demo.get("gripper_actions") is not None else 0
                    result.action_dim = int(joint_shape[-1]) + grip_dim

            for alt in ("action", "action_dict/abs_pos"):
                if first_demo.get(alt) is not None:
                    action_keys.append(str(alt))

            rewards = first_demo.get("rewards")
            dones = first_demo.get("dones")
            result.has_reward = rewards is not None
            result.has_done = dones is not None

            attrs = dict(first_demo.attrs)
            result.has_success = any(k in attrs for k in ("success", "is_success", "successful"))

            max_horizon = result.horizon or 0
            for demo_key in demo_keys[1:]:
                demo = data_group[demo_key]
                demo_actions = demo.get("actions")
                if demo_actions is not None and getattr(demo_actions, "shape", None):
                    max_horizon = max(max_horizon, int(demo_actions.shape[0]))
            if max_horizon:
                result.horizon = max_horizon

            mask_group = handle.get("mask")
            if mask_group is not None:
                for split_name in ("valid", "val", "validation", "train"):
                    if split_name in mask_group:
                        result.has_validation_split = True
                        break

            data_attrs = dict(data_group.attrs)
            attach_flag = data_attrs.get("attachment_side_channel")
            result.attachment_side_channel = attach_flag in (True, 1, "true", "True")
            if data_attrs.get("attachment_field"):
                result.attachment_field = str(data_attrs.get("attachment_field"))
            side_raw = data_attrs.get("side_channel_keys")
            if side_raw is not None:
                try:
                    import json

                    if isinstance(side_raw, bytes):
                        side_raw = side_raw.decode("utf-8")
                    parsed = json.loads(side_raw) if isinstance(side_raw, str) else list(side_raw)
                    result.side_channel_keys = [str(k) for k in parsed]
                except (TypeError, ValueError, json.JSONDecodeError):
                    result.warnings.append("无法解析 HDF5 side_channel_keys")
            if data_attrs.get("attachment_policy"):
                result.attachment_policy = str(data_attrs.get("attachment_policy"))
            if data_attrs.get("attachment_input_mode"):
                result.attachment_input_mode = str(data_attrs.get("attachment_input_mode"))
            if data_attrs.get("attachment_control_mode"):
                result.attachment_control_mode = str(data_attrs.get("attachment_control_mode"))
            if data_attrs.get("controller_type"):
                result.controller_type = str(data_attrs.get("controller_type"))
            if data_attrs.get("joint_action_available") in (True, 1, "true", "True"):
                result.joint_action_available = True
            if data_attrs.get("preferred_policy_schema_id"):
                result.preferred_policy_schema_id = str(data_attrs.get("preferred_policy_schema_id"))
            if data_attrs.get("trained_action_mode"):
                result.trained_action_mode = str(data_attrs.get("trained_action_mode"))
            if data_attrs.get("eval_executor"):
                result.eval_executor = str(data_attrs.get("eval_executor"))
            action_schema_raw = data_attrs.get("action_schema")
            if isinstance(action_schema_raw, bytes):
                action_schema_raw = action_schema_raw.decode("utf-8")
            if isinstance(action_schema_raw, str) and action_schema_raw.startswith("{"):
                try:
                    import json as _json

                    parsed_action = _json.loads(action_schema_raw)
                    if isinstance(parsed_action, dict) and parsed_action.get("id"):
                        result.action_schema_id = str(parsed_action["id"])
                except ValueError:
                    pass
            available_raw = data_attrs.get("available_action_keys")
            if available_raw is not None:
                try:
                    import json as _json

                    if isinstance(available_raw, bytes):
                        available_raw = available_raw.decode("utf-8")
                    parsed_keys = _json.loads(available_raw) if isinstance(available_raw, str) else list(available_raw)
                    result.available_action_keys = [str(k) for k in parsed_keys]
                except (TypeError, ValueError, _json.JSONDecodeError):
                    pass
            if not result.attachment_side_channel and first_demo.get("attachment_enabled") is not None:
                result.attachment_side_channel = True
                if "attachment_enabled" not in result.side_channel_keys:
                    result.side_channel_keys.append("attachment_enabled")

            if not result.observation_keys:
                result.warnings.append("HDF5 demo 中未找到 obs 观测键")
            if result.action_dim is None:
                result.warnings.append("HDF5 demo 中未找到 actions 或无法推断 action_dim")

            result.action_space = _infer_action_space(result.action_dim, action_keys)

    except OSError as exc:
        result.warnings.append(f"无法读取 HDF5: {exc}")

    return result
