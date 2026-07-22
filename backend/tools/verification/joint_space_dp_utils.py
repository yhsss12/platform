"""Joint-space Diffusion Policy backend test utilities (standalone, no platform DB/UI)."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np

logger = logging.getLogger(__name__)

CABLE_MVP_ROOT = Path(__file__).resolve().parents[3] / "integrations" / "CableThreadingMVP"
JOINT_PART_CONFIG = (
    CABLE_MVP_ROOT / "robosuite" / "controllers" / "config" / "default" / "parts" / "joint_position.json"
)
PANDA_CONTROLLER_CONFIG = (
    CABLE_MVP_ROOT / "robosuite" / "controllers" / "config" / "robots" / "default_panda.json"
)

JOINT_OBS_KEYS = (
    "agentview_image",
    "robot0_eye_in_hand_image",
    "robot0_joint_pos",
    "robot0_joint_vel",
    "robot0_gripper_qpos",
    "robot0_gripper_qvel",
    "robot0_eef_pos",
    "robot0_eef_quat",
    "attachment_state",
    "cable_end_pos",
    "pole_points",
    "endpoint_goal_pos",
    "cable_points",
    "physical_grasp_state",
    "object-state",
)


def _ensure_cable_mvp_path() -> None:
    import sys

    root = str(CABLE_MVP_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def build_joint_position_controller_config(*, robot: str = "Panda") -> dict[str, Any]:
    _ensure_cable_mvp_path()
    from robosuite.controllers import load_composite_controller_config

    cfg = load_composite_controller_config(robot=robot)
    joint_part = json.loads(JOINT_PART_CONFIG.read_text(encoding="utf-8"))
    joint_part["input_type"] = "delta"
    body_parts = cfg.get("body_parts") or {}
    arm_key = "right" if "right" in body_parts else "arms"
    if arm_key == "arms":
        cfg["body_parts"]["arms"]["right"] = {**joint_part, "gripper": {"type": "GRIP"}}
    else:
        cfg["body_parts"]["right"] = {**joint_part, "gripper": {"type": "GRIP"}}
    return cfg


def make_joint_position_env(
    *,
    robot: str = "Panda",
    cable_model: str = "composite_cable",
    grasp_mode: str = "attachment",
    difficulty: str = "easy",
    horizon: int = 600,
    seed: int | None = None,
    use_camera_obs: bool = True,
    has_offscreen_renderer: bool | None = None,
):
    _ensure_cable_mvp_path()
    from examples.cable_threading.utils import make_env

    if has_offscreen_renderer is None:
        has_offscreen_renderer = use_camera_obs
    controller_configs = build_joint_position_controller_config(robot=robot)
    return make_env(
        robot=robot,
        cable_model=cable_model,
        grasp_mode=grasp_mode,
        difficulty=difficulty,
        horizon=horizon,
        seed=seed,
        use_camera_obs=use_camera_obs,
        has_offscreen_renderer=has_offscreen_renderer,
        camera_names=["agentview", "robot0_eye_in_hand"] if use_camera_obs else None,
        controller_configs=controller_configs,
    )


def _get_arm_joint_controller(env):
    robot = env.robots[0]
    arm = robot.arms[0]
    return robot.part_controllers[arm]


def _get_gripper_controller(env):
    robot = env.robots[0]
    arm = robot.arms[0]
    grip_name = robot.get_gripper_name(arm)
    return robot.part_controllers[grip_name]


def inverse_scale_action(controller, scaled_delta: np.ndarray) -> np.ndarray:
    """Map physical delta back to normalized controller input in [-1, 1]."""
    scaled_delta = np.asarray(scaled_delta, dtype=np.float64)
    if controller.action_scale is None:
        controller.scale_action(np.zeros(scaled_delta.shape, dtype=np.float64))
    normalized = (
        (scaled_delta - controller.action_output_transform) / controller.action_scale
        + controller.action_input_transform
    )
    return np.clip(normalized, controller.input_min, controller.input_max).astype(np.float32)


def inspect_joint_position_controller(*, robot: str = "Panda") -> dict[str, Any]:
    env = make_joint_position_env(robot=robot, use_camera_obs=False, has_offscreen_renderer=False, seed=0)
    try:
        low, high = env.action_spec
        joint_ctrl = _get_arm_joint_controller(env)
        grip_ctrl = _get_gripper_controller(env)
        arm_dim = int(joint_ctrl.control_dim)
        grip_dim = int(grip_ctrl.control_dim)
        info = {
            "controller_config_path": str(PANDA_CONTROLLER_CONFIG),
            "joint_part_config_path": str(JOINT_PART_CONFIG),
            "controller_type": str(getattr(joint_ctrl, "name", type(joint_ctrl).__name__)),
            "input_type": str(joint_ctrl.input_type),
            "action_dim": int(low.shape[0]),
            "arm_action_dim": arm_dim,
            "gripper_action_dim": grip_dim,
            "action_low": low.tolist(),
            "action_high": high.tolist(),
            "joint_input_min": np.asarray(joint_ctrl.input_min).tolist(),
            "joint_input_max": np.asarray(joint_ctrl.input_max).tolist(),
            "joint_output_min": np.asarray(joint_ctrl.output_min).tolist(),
            "joint_output_max": np.asarray(joint_ctrl.output_max).tolist(),
            "joint_action_semantics": (
                "normalized joint delta in [-1,1]; scaled by controller to rad delta "
                f"in [{float(np.min(joint_ctrl.output_min))}, {float(np.max(joint_ctrl.output_max))}] per step"
            ),
            "gripper_action_semantics": (
                "normalized gripper command in [-1,1] (GRIP controller); "
                f"{grip_dim}D broadcast, last action dim controls gripper"
            ),
            "impedance_mode": str(joint_ctrl.impedance_mode),
        }
        return info
    finally:
        env.close()


def compute_action_diagnostics(actions: np.ndarray, *, saturation_threshold: float = 0.95) -> dict[str, float]:
    """Offline action distribution diagnostics (normalized controller space)."""
    arr = np.asarray(actions, dtype=np.float32)
    if arr.size == 0:
        return {
            "action_saturation_ratio": 0.0,
            "joint_action_abs_mean": 0.0,
            "joint_action_abs_max": 0.0,
            "gripper_action_abs_mean": 0.0,
        }
    joint = arr[:, :7] if arr.shape[-1] >= 7 else arr
    grip = arr[:, 7:8] if arr.shape[-1] >= 8 else np.zeros((arr.shape[0], 1), dtype=np.float32)
    sat_mask = np.abs(arr) >= saturation_threshold
    return {
        "action_saturation_ratio": float(np.mean(sat_mask)),
        "joint_action_abs_mean": float(np.mean(np.abs(joint))),
        "joint_action_abs_max": float(np.max(np.abs(joint))),
        "gripper_action_abs_mean": float(np.mean(np.abs(grip))),
    }


def _joint_policy_schemas() -> dict[str, Any]:
    schemas = {
        "joint_state_obs_joint_action": {
            "input": {
                "image_keys": ["agentview_image", "robot0_eye_in_hand_image"],
                "low_dim_keys": ["robot0_joint_pos", "robot0_gripper_qpos"],
            },
            "output": {
                "action_key": "actions",
                "joint_action_key": "joint_actions",
                "gripper_action_key": "gripper_actions",
                "action_mode": "joint_delta_derived",
                "controller_type": "JOINT_POSITION",
                "action_dim": 8,
                "arm_action_dim": 7,
                "gripper_action_dim": 1,
            },
            "note": "8D JOINT_POSITION delta actions derived from OSC replay joint_pos; not native expert joint commands.",
        }
    }
    return schemas


def _write_joint_dataset_manifest(output_hdf5: Path, save_info: dict[str, Any]) -> Path:
    _ensure_cable_mvp_path()
    from robosuite.utils.dlo.hdf5_dataset import build_hdf5_manifest_fields

    manifest_path = output_hdf5.with_suffix(".manifest.json")
    manifest = {
        "datasetPath": str(output_hdf5),
        "taskName": "cable_threading",
        "taskType": "cable_threading",
        "simulator": "robosuite",
        "robotType": "Panda",
        "graspMode": "attachment",
        **build_hdf5_manifest_fields(save_info),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest_path


def derive_joint_space_actions(
    joint_positions: np.ndarray,
    gripper_commands: np.ndarray,
    *,
    joint_ctrl,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Derive 8D JOINT_POSITION actions from post-step joint_pos sequence."""
    q = np.asarray(joint_positions, dtype=np.float32)
    grip = np.asarray(gripper_commands, dtype=np.float32).reshape(-1)
    if q.ndim != 2 or q.shape[1] != joint_ctrl.control_dim:
        raise ValueError(f"joint_positions shape {q.shape} incompatible with arm dof {joint_ctrl.control_dim}")
    if len(grip) != q.shape[0]:
        raise ValueError(f"gripper length {len(grip)} != steps {q.shape[0]}")

    deltas = np.diff(q, axis=0)
    deltas = np.vstack([deltas, deltas[-1:]])
    joint_actions = np.stack(
        [inverse_scale_action(joint_ctrl, deltas[t]) for t in range(q.shape[0])],
        axis=0,
    ).astype(np.float32)
    gripper_actions = grip.reshape(-1, 1).astype(np.float32)
    actions = np.concatenate([joint_actions, gripper_actions], axis=-1).astype(np.float32)

    clipped = np.clip(actions, action_low, action_high)
    clip_mask = clipped != actions
    clip_ratio = float(np.mean(clip_mask))
    actions = clipped.astype(np.float32)

    stats = {
        "clip_ratio": clip_ratio,
        "action_dim": float(actions.shape[-1]),
        "joint_action_dim": float(joint_actions.shape[-1]),
        "gripper_action_dim": float(gripper_actions.shape[-1]),
    }
    return actions, joint_actions, gripper_actions, stats


def replay_osc_collect_full_obs(
    env,
    source_demo: dict[str, Any],
    *,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Replay OSC HDF5 demo; collect post-step raw obs + OSC gripper."""
    _ensure_cable_mvp_path()
    from examples.cable_threading.attachment_controller import apply_attachment_flag
    from examples.cable_threading.hdf5_replay import reset_env_for_replay
    from examples.cable_threading.utils import clip_action

    osc_actions = np.asarray(source_demo["actions"], dtype=np.float32)
    attach_arr = np.asarray(source_demo.get("attachment_enabled", []), dtype=bool)
    if len(attach_arr) != len(osc_actions):
        attach_flags = [False] * len(osc_actions)
    else:
        attach_flags = [bool(x) for x in attach_arr]

    reset_env_for_replay(env)
    steps: list[dict[str, Any]] = []
    prev_attach = False
    clip_count = 0
    for t, (osc_action, want_attach) in enumerate(zip(osc_actions, attach_flags)):
        apply_attachment_flag(env, want_attach, prev_attach)
        prev_attach = want_attach
        clipped = clip_action(env, osc_action)
        if not np.allclose(clipped, osc_action):
            clip_count += 1
        obs, reward, done, info = env.step(clipped)
        steps.append(
            {
                "raw_obs": dict(obs),
                "osc_action": clipped.copy(),
                "reward": float(reward),
                "done": bool(done),
                "info": dict(info),
                "attachment_enabled": bool(want_attach),
            }
        )
    meta = {"steps": len(steps), "osc_action_clip_steps": clip_count}
    return steps, meta


def build_joint_space_hdf5(
    source_hdf5: Path,
    output_hdf5: Path,
    *,
    max_demos: int | None = 10,
    robot: str = "Panda",
    log_path: Path | None = None,
    fail_on_demo_errors: bool = False,
) -> dict[str, Any]:
    _ensure_cable_mvp_path()
    source_hdf5 = source_hdf5.expanduser().resolve()
    output_hdf5 = output_hdf5.expanduser().resolve()
    output_hdf5.parent.mkdir(parents=True, exist_ok=True)

    osc_env = make_env_for_osc_replay(robot=robot)
    joint_env = make_joint_position_env(robot=robot, use_camera_obs=False, has_offscreen_renderer=False, seed=0)
    joint_ctrl = _get_arm_joint_controller(joint_env)
    j_low, j_high = joint_env.action_spec

    report: dict[str, Any] = {
        "source_hdf5": str(source_hdf5),
        "output_hdf5": str(output_hdf5),
        "demos_requested": 0,
        "demos_written": 0,
        "demos_failed": 0,
        "failed_demos": [],
        "clip_stats": [],
        "errors": [],
    }
    log_lines: list[str] = []

    def _log(msg: str) -> None:
        log_lines.append(msg)
        logger.info(msg)

    try:
        with h5py.File(source_hdf5, "r") as src:
            demo_names = sorted(k for k in src["data"].keys() if str(k).startswith("demo_"))
            report["demos_requested"] = len(demo_names)
            if max_demos is not None and max_demos > 0:
                demo_names = demo_names[: max_demos]

            with h5py.File(output_hdf5, "w") as out:
                data_grp = out.create_group("data")
                data_grp.attrs["env"] = "CableThreading"
                data_grp.attrs["grasp_mode"] = "attachment"
                data_grp.attrs["attachment_side_channel"] = True
                data_grp.attrs["controller_type"] = "JOINT_POSITION"
                data_grp.attrs["action_mode"] = "joint_delta_derived"
                data_grp.attrs["action_dim"] = 8
                data_grp.attrs["obs_keys"] = json.dumps(list(JOINT_OBS_KEYS))
                data_grp.attrs["available_action_keys"] = json.dumps(
                    ["actions", "joint_actions", "gripper_actions"]
                )

                written_names: list[str] = []

                for demo_name in demo_names:
                    grp = src["data"][demo_name]
                    osc_actions = np.asarray(grp["actions"], dtype=np.float32)
                    attach = (
                        np.asarray(grp["attachment_enabled"], dtype=bool)
                        if "attachment_enabled" in grp
                        else np.zeros(len(osc_actions), dtype=bool)
                    )
                    meta_raw = grp.attrs.get("benchmark_episode_metadata", "{}")
                    if isinstance(meta_raw, bytes):
                        meta_raw = meta_raw.decode("utf-8")
                    ep_meta = json.loads(meta_raw) if meta_raw else {}
                    seed = int(ep_meta.get("seed", 0))

                    source_demo = {
                        "actions": osc_actions,
                        "attachment_enabled": attach,
                    }
                    try:
                        steps, replay_meta = replay_osc_collect_full_obs(osc_env, source_demo, seed=seed)
                    except Exception as exc:
                        err_msg = f"{demo_name}: replay failed: {exc}"
                        report["errors"].append(err_msg)
                        report["failed_demos"].append({"demo": demo_name, "reason": str(exc)})
                        report["demos_failed"] += 1
                        _log(err_msg)
                        if fail_on_demo_errors:
                            raise
                        continue

                    joint_pos = np.stack(
                        [np.asarray(s["raw_obs"]["robot0_joint_pos"], dtype=np.float32) for s in steps],
                        axis=0,
                    )
                    gripper_cmds = np.asarray([s["osc_action"][-1] for s in steps], dtype=np.float32)
                    actions, joint_actions, gripper_actions, clip_stats = derive_joint_space_actions(
                        joint_pos,
                        gripper_cmds,
                        joint_ctrl=joint_ctrl,
                        action_low=j_low,
                        action_high=j_high,
                    )
                    report["clip_stats"].append({"demo": demo_name, **clip_stats})

                    demo_out = data_grp.create_group(demo_name)
                    demo_out.attrs["num_samples"] = len(steps)
                    demo_out.attrs["benchmark_episode_metadata"] = json.dumps(ep_meta, ensure_ascii=False)
                    demo_out.create_dataset("actions", data=actions)
                    demo_out.create_dataset("joint_actions", data=joint_actions)
                    demo_out.create_dataset("gripper_actions", data=gripper_actions)
                    demo_out.create_dataset(
                        "attachment_enabled",
                        data=np.asarray([s["attachment_enabled"] for s in steps], dtype=np.bool_),
                    )
                    obs_grp = demo_out.create_group("obs")
                    for key in JOINT_OBS_KEYS:
                        values = []
                        for s in steps:
                            raw = s["raw_obs"]
                            if key not in raw:
                                values = []
                                break
                            values.append(np.asarray(raw[key]))
                        if not values:
                            continue
                        stacked = np.stack(values, axis=0)
                        if stacked.dtype == np.uint8:
                            obs_grp.create_dataset(key, data=stacked, compression="gzip", chunks=(1, *stacked.shape[1:]))
                        else:
                            obs_grp.create_dataset(key, data=stacked.astype(np.float32))

                    report["demos_written"] += 1
                    written_names.append(demo_name)
                    _log(f"{demo_name}: steps={len(steps)} clip_ratio={clip_stats['clip_ratio']:.4f}")

                mask_grp = out.create_group("mask")
                mask_grp.create_dataset("train", data=np.asarray(written_names, dtype="S"))

                all_actions = []
                for demo_name in written_names:
                    all_actions.append(np.asarray(out["data"][demo_name]["actions"], dtype=np.float32))
                action_diag = compute_action_diagnostics(np.concatenate(all_actions, axis=0) if all_actions else np.zeros((0, 8)))
                report["action_diagnostics"] = action_diag

                save_info = {
                    "attachment_side_channel": True,
                    "attachment_field": "attachment_enabled",
                    "side_channel_keys": ["attachment_enabled"],
                    "available_obs_keys": list(JOINT_OBS_KEYS),
                    "image_keys": ["agentview_image", "robot0_eye_in_hand_image"],
                    "low_dim_keys": ["robot0_joint_pos", "robot0_gripper_qpos"],
                    "task_obs_keys": [
                        k
                        for k in JOINT_OBS_KEYS
                        if k
                        not in {
                            "agentview_image",
                            "robot0_eye_in_hand_image",
                            "robot0_joint_pos",
                            "robot0_joint_vel",
                            "robot0_gripper_qpos",
                            "robot0_gripper_qvel",
                            "robot0_eef_pos",
                            "robot0_eef_quat",
                        }
                    ],
                    "available_action_keys": ["actions", "joint_actions", "gripper_actions"],
                    "policy_schemas": _joint_policy_schemas(),
                    "preferred_policy_schemas": {
                        "joint_state_dp": _joint_policy_schemas()["joint_state_obs_joint_action"]["input"]
                        | {"low_dim_dim": 9},
                    },
                    "action_dim": 8,
                    "current_action_mode": "joint_delta_derived",
                    "controller_type": "JOINT_POSITION",
                    "joint_action_available": True,
                    "joint_action_mode": "joint_delta_derived",
                    "gripper_action_available": True,
                    "derived_action_note": "actions derived from OSC replay joint_pos deltas + OSC gripper dim",
                    "robot_state_available": True,
                    "task_state_available": True,
                    "include_attachment_obs": False,
                    "num_demos": report["demos_written"],
                }
                manifest_path = _write_joint_dataset_manifest(output_hdf5, save_info)
                report["manifest_path"] = str(manifest_path)

        if report["demos_written"] == 0:
            report["ok"] = False
        else:
            clip_ratios = [row["clip_ratio"] for row in report["clip_stats"]]
            report["mean_clip_ratio"] = float(np.mean(clip_ratios))
            report["max_clip_ratio"] = float(np.max(clip_ratios))
            report["joint_action_clip_ratio"] = report["mean_clip_ratio"]
            report["ok"] = report["demos_failed"] == 0 or report["demos_written"] >= max(1, report["demos_requested"] // 2)
    finally:
        osc_env.close()
        joint_env.close()

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    return report


def make_env_for_osc_replay(*, robot: str = "Panda"):
    _ensure_cable_mvp_path()
    from examples.cable_threading.utils import make_env

    return make_env(
        robot=robot,
        cable_model="composite_cable",
        grasp_mode="attachment",
        difficulty="easy",
        horizon=600,
        seed=0,
        use_camera_obs=True,
        has_offscreen_renderer=True,
        camera_names=["agentview", "robot0_eye_in_hand"],
    )


def _replay_single_demo(
    joint_hdf5: Path,
    demo_name: str,
    handle: h5py.File,
) -> dict[str, Any]:
    _ensure_cable_mvp_path()
    from examples.cable_threading.attachment_controller import apply_attachment_flag
    from examples.cable_threading.hdf5_replay import reset_env_for_replay
    from examples.cable_threading.utils import clip_action

    grp = handle["data"][demo_name]
    actions = np.asarray(grp["actions"], dtype=np.float32)
    stored_joint = (
        np.asarray(grp["obs"]["robot0_joint_pos"], dtype=np.float32)
        if "obs" in grp and "robot0_joint_pos" in grp["obs"]
        else None
    )
    attach = np.asarray(grp["attachment_enabled"], dtype=bool) if "attachment_enabled" in grp else None
    meta_raw = grp.attrs.get("benchmark_episode_metadata", "{}")
    if isinstance(meta_raw, bytes):
        meta_raw = meta_raw.decode("utf-8")
    ep_meta = json.loads(meta_raw) if meta_raw else {}
    seed = int(ep_meta.get("seed", 0))

    env = make_joint_position_env(seed=seed, use_camera_obs=False, has_offscreen_renderer=False)
    row: dict[str, Any] = {"demo": demo_name, "seed": seed, "steps": len(actions)}
    try:
        low, high = env.action_spec
        if actions.shape[-1] != low.shape[0]:
            raise ValueError(f"action_dim mismatch: hdf5 {actions.shape[-1]} env {low.shape[0]}")
        reset_env_for_replay(env)
        info_rows = []
        replay_joint = []
        prev_attach = False
        attach_transitions = 0
        detach_transitions = 0
        for t in range(len(actions)):
            if attach is not None:
                want = bool(attach[t])
                if want != prev_attach:
                    if want:
                        attach_transitions += 1
                    else:
                        detach_transitions += 1
                apply_attachment_flag(env, want, prev_attach)
                prev_attach = want
            act = clip_action(env, actions[t])
            obs, _, _, info = env.step(act)
            info_rows.append(dict(info))
            replay_joint.append(np.asarray(obs["robot0_joint_pos"], dtype=np.float32).reshape(-1))

        last = info_rows[-1] if info_rows else {}
        joint_l2_mean = None
        joint_l2_max = None
        if stored_joint is not None and replay_joint:
            replay_arr = np.stack(replay_joint, axis=0)
            n = min(len(replay_arr), len(stored_joint))
            diff = replay_arr[:n] - stored_joint[:n]
            l2 = np.linalg.norm(diff, axis=-1)
            joint_l2_mean = float(np.mean(l2))
            joint_l2_max = float(np.max(l2))

        row.update(
            {
                "completed": True,
                "final_success": bool(last.get("final_success", False)),
                "ever_success": bool(last.get("ever_success", False)),
                "endpoint_goal_error_final": float(
                    last.get("endpoint_goal_error_final", last.get("endpoint_goal_error", 1.0))
                ),
                "thread_completion_max": float(
                    last.get("thread_completion_max", last.get("thread_completion", 0.0))
                ),
                "joint_pos_replay_l2_mean": joint_l2_mean,
                "joint_pos_replay_l2_max": joint_l2_max,
                "attach_transitions": attach_transitions,
                "detach_transitions": detach_transitions,
                "action_saturation_ratio": compute_action_diagnostics(actions)["action_saturation_ratio"],
            }
        )
    except Exception as exc:
        row["completed"] = False
        row["error"] = str(exc)
    finally:
        env.close()
    return row


def _select_sanity_demos(handle: h5py.File, *, first_n: int, random_n: int, success_n: int, seed: int) -> list[str]:
    all_demos = sorted(k for k in handle["data"].keys() if str(k).startswith("demo_"))
    selected: list[str] = []
    selected.extend(all_demos[:first_n])

    rng = np.random.default_rng(seed)
    pool = [d for d in all_demos if d not in selected]
    if pool and random_n > 0:
        pick = rng.choice(pool, size=min(random_n, len(pool)), replace=False)
        selected.extend(str(x) for x in pick)

    success_candidates = []
    for demo in all_demos:
        meta_raw = handle["data"][demo].attrs.get("benchmark_episode_metadata", "{}")
        if isinstance(meta_raw, bytes):
            meta_raw = meta_raw.decode("utf-8")
        meta = json.loads(meta_raw) if meta_raw else {}
        if bool(meta.get("final_success", False)) or bool(meta.get("success", False)):
            success_candidates.append(demo)
    for demo in success_candidates[:success_n]:
        if demo not in selected:
            selected.append(demo)

    return sorted(set(selected), key=lambda d: all_demos.index(d))


def replay_joint_sanity(
    joint_hdf5: Path,
    *,
    max_demos: int = 5,
    min_successful_replays: int = 3,
    sample_first: int | None = None,
    sample_random: int | None = None,
    sample_success: int | None = None,
    random_seed: int = 42,
) -> dict[str, Any]:
    joint_hdf5 = joint_hdf5.expanduser().resolve()
    results: dict[str, Any] = {
        "hdf5": str(joint_hdf5),
        "demos_tested": 0,
        "demos_completed": 0,
        "demos": [],
        "ok": False,
    }

    with h5py.File(joint_hdf5, "r") as handle:
        if sample_first is not None or sample_random is not None or sample_success is not None:
            demo_names = _select_sanity_demos(
                handle,
                first_n=sample_first or 0,
                random_n=sample_random or 0,
                success_n=sample_success or 0,
                seed=random_seed,
            )
            results["sample_strategy"] = {
                "first_n": sample_first,
                "random_n": sample_random,
                "success_n": sample_success,
                "selected_count": len(demo_names),
            }
        else:
            demo_names = sorted(k for k in handle["data"].keys() if str(k).startswith("demo_"))[:max_demos]

        for demo_name in demo_names:
            row = _replay_single_demo(joint_hdf5, demo_name, handle)
            results["demos"].append(row)
            results["demos_tested"] += 1
            if row.get("completed"):
                results["demos_completed"] += 1

    completed_rows = [r for r in results["demos"] if r.get("completed")]
    if completed_rows:
        results["final_success_rate"] = float(np.mean([1.0 if r.get("final_success") else 0.0 for r in completed_rows]))
        results["ever_success_rate"] = float(np.mean([1.0 if r.get("ever_success") else 0.0 for r in completed_rows]))
        l2_means = [r["joint_pos_replay_l2_mean"] for r in completed_rows if r.get("joint_pos_replay_l2_mean") is not None]
        if l2_means:
            results["mean_joint_pos_replay_l2"] = float(np.mean(l2_means))
            results["max_joint_pos_replay_l2"] = float(np.max(l2_means))
        results["mean_endpoint_goal_error_final"] = float(
            np.mean([r.get("endpoint_goal_error_final", 1.0) for r in completed_rows])
        )
        results["mean_attach_transitions"] = float(np.mean([r.get("attach_transitions", 0) for r in completed_rows]))
        results["mean_detach_transitions"] = float(np.mean([r.get("detach_transitions", 0) for r in completed_rows]))
        action_dim_errors = [r for r in results["demos"] if "action_dim mismatch" in str(r.get("error", ""))]
        results["action_dim_errors"] = len(action_dim_errors)

    results["ok"] = results["demos_completed"] >= min_successful_replays
    return results


def validate_eval_controller_match(checkpoint_path: Path, env) -> None:
    import torch
    from examples.cable_threading.dp_lab.config import DpLabConfig

    payload = torch.load(checkpoint_path.expanduser(), map_location="cpu")
    train_config = payload.get("train_config") or {}
    fields = DpLabConfig.__dataclass_fields__
    cfg = DpLabConfig(**{k: train_config[k] for k in fields if k in train_config})
    expected = str(cfg.controller_type or "OSC_POSE").upper()
    if expected != "JOINT_POSITION":
        raise ValueError(f"checkpoint controller_type={expected}, expected JOINT_POSITION")
    joint_ctrl = _get_arm_joint_controller(env)
    actual = str(getattr(joint_ctrl, "name", "")).upper()
    if "JOINT" not in actual:
        raise ValueError(f"env controller={actual}, expected JOINT_POSITION family")
    low, _ = env.action_spec
    if int(low.shape[0]) != int(cfg.action_dim):
        raise ValueError(f"env action_dim={low.shape[0]} != checkpoint action_dim={cfg.action_dim}")


def run_joint_eval_smoke(
    checkpoint_path: Path,
    *,
    episodes: int = 2,
    seed: int = 0,
    device: str = "cuda",
    horizon: int = 1200,
    out_dir: Path,
    live_video_out: Path | str | None = None,
    live_frame_dir: Path | str | None = None,
    live_save_frames: bool = True,
    live_frame_every: int = 5,
    live_camera: str = "agentview",
    live_video_fps: int = 20,
) -> dict[str, Any]:
    _ensure_cable_mvp_path()
    import torch
    from examples.cable_threading.attachment_controller import build_attachment_controller
    from examples.cable_threading.dp_lab.policy_runtime import DiffusionPolicyAdapter
    from examples.cable_threading.utils import (
        clip_action,
        maybe_capture_and_persist_live_frame,
        summarize_episode,
        clear_live_saved_frames,
        synthesize_live_video,
        write_live_status,
        _ensure_episode_live_frames,
        _warmup_live_after_reset,
    )

    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir = out_dir / "results"
    videos_dir = out_dir / "videos"
    results_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    if live_frame_dir is None:
        live_frame_dir = videos_dir / "live"
    live_frame_dir = Path(live_frame_dir).expanduser().resolve()
    live_frame_dir.mkdir(parents=True, exist_ok=True)
    if live_video_out is None:
        live_video_out = videos_dir / "eval.mp4"
    live_video_out = Path(live_video_out).expanduser().resolve()

    adapter = DiffusionPolicyAdapter(checkpoint_path, device=device)
    cfg = adapter.cfg
    if str(cfg.controller_type).upper() != "JOINT_POSITION":
        raise ValueError(f"refusing eval: checkpoint controller_type={cfg.controller_type}")
    forbidden = {"robot0_eef_pos", "robot0_eef_quat"}
    if forbidden.intersection(set(cfg.low_dim_keys)):
        raise ValueError(f"joint eval low_dim_keys must not include eef keys: {cfg.low_dim_keys}")

    live_config: dict[str, Any] = {
        "status": "running",
        "jobType": "evaluate",
        "frame_dir": str(live_frame_dir),
        "frames_dir": str(live_frame_dir / "frames") if live_save_frames else None,
        "save_frames": bool(live_save_frames),
        "video_out": str(live_video_out),
        "video_fps": int(live_video_fps),
        "eval_video_status": "pending",
        "eval_video_exists": False,
        "eval_video_size_bytes": 0,
        "saved_frame_count": 0,
        "frame_every": int(live_frame_every),
        "camera": live_camera,
        "display_camera": live_camera,
        "record_camera": live_camera,
        "episode": 0,
        "episodes": episodes,
        "horizon": int(horizon),
        "step": 0,
        "frame_count": 0,
        "has_valid_frame": False,
        "live_warmup_steps": 10,
        "live_required_consecutive_valid": 3,
        "multi_episode_video": episodes > 1,
        "global_saved_frame_count": 0,
    }
    if live_save_frames:
        (live_frame_dir / "frames").mkdir(parents=True, exist_ok=True)
    if episodes > 1:
        clear_live_saved_frames(live_config)
        live_config["global_saved_frame_count"] = 0
        live_config["saved_frame_count"] = 0

    rows = []
    all_actions: list[np.ndarray] = []
    for episode in range(episodes):
        live_config["episode"] = episode
        live_config["episode_frame_start"] = int(live_config.get("global_saved_frame_count", 0))
        live_config["step"] = 0
        live_config["frame_count"] = 0
        if episodes <= 1:
            live_config["saved_frame_count"] = 0
            live_config["global_saved_frame_count"] = 0
        live_config["has_valid_frame"] = episode > 0 and bool(
            live_config.get("global_saved_frame_count", 0)
        )
        live_config["_consecutive_valid"] = (
            int(live_config.get("live_required_consecutive_valid", 3))
            if live_config["has_valid_frame"]
            else 0
        )

        env = make_joint_position_env(
            seed=seed + episode,
            use_camera_obs=True,
            has_offscreen_renderer=True,
            horizon=int(live_config["horizon"]),
        )
        validate_eval_controller_match(checkpoint_path, env)
        attach_ctrl = build_attachment_controller(env, replay_mode="policy")
        attach_ctrl.reset()
        adapter.reset()
        obs = env.reset()
        obs = _warmup_live_after_reset(env, obs, live_config)
        info_rows = []
        total_reward = 0.0
        done = False
        episode_actions: list[np.ndarray] = []
        while not done and len(info_rows) < env.horizon:
            action = adapter.act(obs)
            action = clip_action(env, action)
            if action.shape[0] != env.action_spec[0].shape[0]:
                raise ValueError(f"action_dim mismatch at step {len(info_rows)}")
            episode_actions.append(action.copy())
            attach_ctrl.pre_step(action, info=info_rows[-1] if info_rows else None)
            obs, reward, done, info = env.step(action)
            total_reward += float(reward)
            info_rows.append(dict(info))
            live_config["step"] = len(info_rows)
            maybe_capture_and_persist_live_frame(env, obs, live_config)
            write_live_status(live_config)
        _ensure_episode_live_frames(env, obs, live_config)
        if episode_actions:
            all_actions.append(np.stack(episode_actions, axis=0))

        summary = summarize_episode(
            info_rows,
            env,
            total_reward,
            policy_name="diffusion_policy_joint",
            episode_index=episode,
            seed=seed + episode,
        )
        if hasattr(attach_ctrl, "attachment_stats"):
            summary.update(attach_ctrl.attachment_stats())
        if episode_actions:
            act_diag = compute_action_diagnostics(np.stack(episode_actions, axis=0))
            summary.update(
                {
                    "joint_action_abs_mean": act_diag["joint_action_abs_mean"],
                    "joint_action_abs_max": act_diag["joint_action_abs_max"],
                    "action_saturation_ratio": act_diag["action_saturation_ratio"],
                }
            )
        rows.append(summary)
        env.close()

    synthesize_live_video(live_config)
    video_path = live_config.get("videoPath") or str(live_video_out)
    video_exists = Path(video_path).is_file() if video_path else False

    import csv

    csv_path = results_dir / "eval.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=sorted({k for row in rows for k in row.keys()}))
            writer.writeheader()
            writer.writerows(rows)

    success = sum(1 for r in rows if r.get("final_success"))
    ever_success = sum(1 for r in rows if r.get("ever_success"))
    action_diag = compute_action_diagnostics(
        np.concatenate(all_actions, axis=0) if all_actions else np.zeros((0, cfg.action_dim))
    )
    aggregate = {
        "task_name": "cable_threading_joint_space",
        "requested_episodes": episodes,
        "total_episodes": len(rows),
        "success_episodes": success,
        "final_success_rate": success / max(len(rows), 1),
        "ever_success_episodes": ever_success,
        "ever_success_rate": ever_success / max(len(rows), 1),
        "attachment_mode": "policy",
        "controller_type": "JOINT_POSITION",
        "action_dim": cfg.action_dim,
        "eval_horizon": int(horizon),
        "low_dim_keys": list(cfg.low_dim_keys),
        "image_keys": list(cfg.image_keys),
        "mean_thread_completion_max": float(
            np.mean([r.get("thread_completion_max", r.get("max_thread_completion", 0.0)) for r in rows])
        )
        if rows
        else 0.0,
        "mean_endpoint_goal_error_final": float(
            np.mean([r.get("endpoint_goal_error_final", r.get("endpoint_goal_error", 1.0)) for r in rows])
        )
        if rows
        else 1.0,
        "mean_attach_count": float(np.mean([r.get("attach_count", 0) for r in rows])) if rows else 0.0,
        "mean_detach_count": float(np.mean([r.get("detach_count", 0) for r in rows])) if rows else 0.0,
        "mean_attachment_enabled_ratio": float(
            np.mean([r.get("attachment_enabled_ratio", 0.0) for r in rows])
        )
        if rows
        else 0.0,
        "joint_action_abs_mean": action_diag["joint_action_abs_mean"],
        "joint_action_abs_max": action_diag["joint_action_abs_max"],
        "action_saturation_ratio": action_diag["action_saturation_ratio"],
        "eval_video_path": video_path if video_exists else None,
        "eval_video_exists": video_exists,
        "recorded_frame_count": int(live_config.get("global_saved_frame_count", live_config.get("saved_frame_count", 0))),
        "video_fps": int(live_video_fps),
    }
    (results_dir / "aggregate_result.json").write_text(
        json.dumps(aggregate, indent=2, default=str),
        encoding="utf-8",
    )
    return {
        "ok": True,
        "csv_path": str(csv_path),
        "aggregate_path": str(results_dir / "aggregate_result.json"),
        "video_path": video_path if video_exists else None,
        "video_exists": video_exists,
        "rows": rows,
        "aggregate": aggregate,
        "controller_type": "JOINT_POSITION",
        "action_dim": cfg.action_dim,
    }
