"""RoboSuite / MuJoCo 环境加载与 demo 状态复位（V2-B rollout 基础设施）。"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np

# ---------------------------------------------------------------------------
# 依赖路径：使用 integrations/CableThreadingMVP 内 vendored robosuite
# ---------------------------------------------------------------------------

_EXPERIMENT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _EXPERIMENT_DIR.parents[2]
_CABLE_MVP = _REPO_ROOT / "integrations" / "CableThreadingMVP"

ENV_NAME_MAP = {
    "Square_D0": "NutAssemblySquare",
    "NutAssemblySquare": "NutAssemblySquare",
}

_MUJOCO_PATCHED = False
_ROBOSUITE_IMPORT_ERROR: str | None = None


def _ensure_robosuite_path() -> None:
    cable_str = str(_CABLE_MVP)
    if cable_str not in sys.path:
        sys.path.insert(0, cable_str)


def _apply_mujoco310_controller_patch() -> None:
    """MuJoCo 3.x 下 mj_fullM 签名变更；仅在实验脚本内 monkeypatch，不改 integrations 源码。"""
    global _MUJOCO_PATCHED
    if _MUJOCO_PATCHED:
        return
    import mujoco
    import robosuite.controllers.parts.controller as controller_mod

    def _patched_update(self, force: bool = False) -> None:
        if self.new_update or force:
            if not (self.lite_physics and not force):
                self.sim.forward()
            if self.ref_name is not None:
                self.update_reference_data()
            self.joint_pos = np.array(self.sim.data.qpos[self.qpos_index])
            self.joint_vel = np.array(self.sim.data.qvel[self.qvel_index])
            mass_matrix = np.ndarray(
                shape=(self.sim.model.nv, self.sim.model.nv),
                dtype=np.float64,
                order="C",
            )
            mujoco.mj_fullM(self.sim.model._model, self.sim.data._data, mass_matrix)
            mass_matrix = np.reshape(mass_matrix, (len(self.sim.data.qvel), len(self.sim.data.qvel)))
            self.mass_matrix = mass_matrix[self.qvel_index, :][:, self.qvel_index]
            self.new_update = False

    controller_mod.Controller.update = _patched_update
    _MUJOCO_PATCHED = True


def check_environment() -> dict[str, Any]:
    """探测 RoboSuite / MuJoCo / 渲染后端是否可用，返回阻塞原因列表。"""
    blockers: list[str] = []
    notes: list[str] = []

    if not _CABLE_MVP.is_dir():
        blockers.append(f"robosuite vendored path missing: {_CABLE_MVP}")

    try:
        import mujoco  # noqa: F401

        notes.append(f"mujoco={mujoco.__version__}")
    except Exception as exc:  # pragma: no cover
        blockers.append(f"mujoco import failed: {exc}")

    try:
        _ensure_robosuite_path()
        _apply_mujoco310_controller_patch()
        import robosuite  # noqa: F401

        notes.append(f"robosuite from {_CABLE_MVP}")
    except Exception as exc:
        blockers.append(f"robosuite import failed: {exc}")

    try:
        import imageio  # noqa: F401

        import imageio_ffmpeg  # noqa: F401

        notes.append("imageio+ffmpeg available for mp4")
    except Exception as exc:
        blockers.append(f"mp4 backend missing (pip install imageio imageio-ffmpeg): {exc}")

    if os.environ.get("MUJOCO_GL") is None:
        notes.append("MUJOCO_GL not set; offscreen render may need MUJOCO_GL=egl")

    return {
        "available": len(blockers) == 0,
        "blockers": blockers,
        "notes": notes,
        "robosuite_path": str(_CABLE_MVP),
    }


def import_robosuite():
    """延迟导入 robosuite（含 MuJoCo patch）。"""
    global _ROBOSUITE_IMPORT_ERROR
    _ensure_robosuite_path()
    try:
        _apply_mujoco310_controller_patch()
        import robosuite
        from robosuite.controllers.composite.composite_controller_factory import (
            refactor_composite_controller_config,
        )

        return robosuite, refactor_composite_controller_config
    except Exception as exc:
        _ROBOSUITE_IMPORT_ERROR = str(exc)
        raise


@dataclass
class DemoRolloutData:
    demo_key: str
    source_file: str
    env_args: dict[str, Any]
    model_xml: str | None
    states: np.ndarray
    actions: np.ndarray
    label: str = "unknown"


@dataclass
class EnvBuildResult:
    env: Any
    env_kwargs: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    xml_reset_attempted: bool = False
    xml_reset_ok: bool = False


def read_env_metadata(hdf5_path: str) -> dict[str, Any]:
    with h5py.File(hdf5_path, "r") as handle:
        if "env_args" in handle["data"].attrs:
            return json.loads(handle["data"].attrs["env_args"])
        if "env_info" in handle["data"].attrs:
            return json.loads(handle["data"].attrs["env_info"])
        raise KeyError(f"no env_args/env_info in {hdf5_path}")


def load_demo_rollout_data(
    hdf5_path: str,
    demo_key: str,
    label: str,
) -> DemoRolloutData:
    env_args = read_env_metadata(hdf5_path)
    with h5py.File(hdf5_path, "r") as handle:
        demo_grp = handle[f"data/{demo_key}"]
        states = demo_grp["states"][:].astype(float)
        actions = demo_grp["actions"][:].astype(float)
        model_xml = demo_grp.attrs.get("model_file")
        if model_xml is not None and isinstance(model_xml, bytes):
            model_xml = model_xml.decode("utf-8")
    return DemoRolloutData(
        demo_key=demo_key,
        source_file=hdf5_path,
        env_args=env_args,
        model_xml=model_xml,
        states=states,
        actions=actions,
        label=label,
    )


def build_env_kwargs(
    env_args: dict[str, Any],
    *,
    for_video: bool = False,
    camera_name: str = "agentview",
    camera_size: int = 256,
) -> dict[str, Any]:
    env_kwargs = dict(env_args.get("env_kwargs", env_args))
    raw_name = env_args.get("env_name", env_kwargs.pop("env_name", "Square_D0"))
    env_kwargs["env_name"] = ENV_NAME_MAP.get(raw_name, raw_name)
    env_kwargs["ignore_done"] = True
    env_kwargs["has_renderer"] = False
    env_kwargs["has_offscreen_renderer"] = for_video
    env_kwargs["use_camera_obs"] = for_video
    if for_video:
        env_kwargs["camera_names"] = camera_name
        env_kwargs["camera_heights"] = camera_size
        env_kwargs["camera_widths"] = camera_size
    else:
        env_kwargs["use_camera_obs"] = False
    return env_kwargs


def postprocess_demo_model_xml(xml_str: str) -> str:
    """修复旧 demo XML 中 mounts/.stl 与当前 robosuite assets 不一致的路径。"""
    xml_str = xml_str.replace("models/assets/mounts/meshes", "models/assets/bases/meshes")
    xml_str = xml_str.replace("pedestal.stl", "pedestal.obj")
    return xml_str


def create_env_from_metadata(
    env_args: dict[str, Any],
    *,
    for_video: bool = False,
    camera_name: str = "agentview",
    camera_size: int = 256,
) -> EnvBuildResult:
    robosuite, refactor_composite_controller_config = import_robosuite()
    env_kwargs = build_env_kwargs(
        env_args,
        for_video=for_video,
        camera_name=camera_name,
        camera_size=camera_size,
    )
    warnings: list[str] = []
    controller_cfg = env_kwargs.get("controller_configs")
    robots = env_kwargs.get("robots", ["Panda"])
    robot = robots[0] if isinstance(robots, list) else robots
    if isinstance(controller_cfg, dict) and controller_cfg.get("type") == "OSC_POSE":
        env_kwargs["controller_configs"] = refactor_composite_controller_config(
            controller_cfg,
            robot,
            ["right"],
        )
        warnings.append("refactored legacy OSC_POSE to composite controller for MuJoCo 3.x compat")
    env = robosuite.make(**env_kwargs)
    return EnvBuildResult(env=env, env_kwargs=env_kwargs, warnings=warnings)


def try_reset_from_demo_xml(
    env: Any,
    model_xml: str | None,
    *,
    enabled: bool = False,
) -> tuple[bool, list[str]]:
    """
    尝试用 demo XML 复位场景。

    默认关闭：旧 HDF5 的 model_file 与当前 robosuite composite controller 模型不兼容
    （缺少 robot0_right_center site）。本阶段使用 states[0] 向量复位即可。
    """
    notes: list[str] = []
    if not enabled:
        notes.append("demo XML replay disabled (legacy XML incompatible); using state vector reset")
        env.reset()
        return False, notes
    if not model_xml:
        notes.append("demo has no model_file attr; using default env geometry + state vector")
        env.reset()
        return False, notes
    try:
        env.reset()
        xml = env.edit_model_xml(model_xml)
        xml = postprocess_demo_model_xml(xml)
        env.reset_from_xml_string(xml)
        notes.append("reset_from_xml_string succeeded")
        return True, notes
    except Exception as exc:
        notes.append(f"reset_from_xml_string failed: {exc}")
        notes.append("recreating env is required after failed xml reload; caller should rebuild env")
        raise RuntimeError(f"demo XML reset failed: {exc}") from exc


def reset_env_to_demo_state(
    env: Any,
    initial_state: np.ndarray,
    *,
    model_xml: str | None = None,
) -> dict[str, Any]:
    """将环境 reset 到 demo 初始 MuJoCo state（不修改 object_poses HDF5）。"""
    info: dict[str, Any] = {"state_dim": int(len(initial_state))}
    xml_ok, xml_notes = try_reset_from_demo_xml(env, model_xml)
    info["xml_reset_ok"] = xml_ok
    info["xml_notes"] = xml_notes

    sim_state_dim = len(env.sim.get_state().flatten())
    info["sim_state_dim"] = sim_state_dim
    if sim_state_dim != len(initial_state):
        raise ValueError(
            f"state dimension mismatch: demo={len(initial_state)} sim={sim_state_dim}"
        )

    env.sim.reset()
    env.sim.set_state_from_flattened(initial_state)
    env.sim.forward()
    info["reset_method"] = "set_state_from_flattened(states[0])"
    return info


def get_sim_eef_pose4(env: Any, arm: str = "right") -> np.ndarray:
    robot = env.robots[0]
    site_id = robot.eef_site_id[arm]
    pos = env.sim.data.site_xpos[site_id].copy()
    rot = env.sim.data.site_xmat[site_id].reshape(3, 3).copy()
    mat = np.eye(4, dtype=float)
    mat[:3, :3] = rot
    mat[:3, 3] = pos
    return mat


def _body_rotmat(env: Any, body_id: int) -> np.ndarray:
    return env.sim.data.body_xmat[body_id].reshape(3, 3).copy()


def extract_sim_features(env: Any) -> dict[str, float]:
    """从当前 sim 读取 nut-peg 几何残差（不读取 HDF5 object_poses）。"""
    from extract_features import square_yaw_error

    nut = env.nuts[env.nut_id]
    nut_name = nut.name
    nut_pos = env.sim.data.body_xpos[env.obj_body_id[nut_name]].copy()
    peg_pos = env.sim.data.body_xpos[env.peg1_body_id].copy()
    nut_rot = _body_rotmat(env, env.obj_body_id[nut_name])
    peg_rot = _body_rotmat(env, env.peg1_body_id)

    xy = float(np.linalg.norm(nut_pos[:2] - peg_pos[:2]))
    z_diff = float(nut_pos[2] - peg_pos[2])
    yaw_err = float(square_yaw_error(nut_rot[None], peg_rot[None])[0])
    return {
        "final_nut_peg_xy": xy,
        "final_z_diff": z_diff,
        "min_yaw_error": yaw_err,
        "final_yaw_error": yaw_err,
    }


def rollout_metrics_to_features_dict(
    demo_key: str,
    label: str,
    source_file: str,
    metrics: dict[str, float],
    action_acceleration_max: float,
    length: int,
) -> dict[str, Any]:
    return {
        "demo_key": demo_key,
        "label": label,
        "source_file": source_file,
        "length": length,
        "final_nut_peg_xy_distance": metrics["final_nut_peg_xy"],
        "min_nut_peg_xy_distance": metrics.get("min_nut_peg_xy", metrics["final_nut_peg_xy"]),
        "final_nut_peg_z_difference": metrics["final_z_diff"],
        "min_nut_peg_yaw_error": metrics["min_yaw_error"],
        "final_nut_peg_yaw_error": metrics.get("final_yaw_error", metrics["min_yaw_error"]),
        "action_acceleration_mean": 0.0,
        "action_acceleration_max": action_acceleration_max,
        "grasp_signal_index": None,
    }


def capture_camera_frame(env: Any, camera_name: str = "agentview") -> np.ndarray:
    obs = env._get_observations()
    frame = obs[f"{camera_name}_image"]
    return np.asarray(frame)


def write_mp4(frames: list[np.ndarray], path: str | Path, fps: int = 20) -> None:
    import imageio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(path), fps=fps)
    for frame in frames:
        arr = np.asarray(frame)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        writer.append_data(arr)
    writer.close()
