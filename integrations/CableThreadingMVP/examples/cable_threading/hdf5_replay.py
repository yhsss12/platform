"""HDF5 / trajectory replay helpers with attachment side-channel."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .attachment_controller import apply_attachment_flag
from .utils import clip_action, make_env


def trajectory_attachment_flags(trajectory: list[dict[str, Any]]) -> list[bool]:
    flags = []
    for step in trajectory:
        if "attachment_enabled" in step:
            flags.append(bool(step["attachment_enabled"]))
        else:
            flags.append(flags[-1] if flags else False)
    return flags


def reset_env_for_replay(env) -> None:
    """Match expert rollout reset semantics (seed via make_env only)."""
    env.reset()
    if str(getattr(env, "grasp_mode", "attachment")) == "attachment":
        env.set_attachment_enabled(False)


def replay_trajectory_collect_obs(
    env,
    trajectory: list[dict[str, Any]],
    *,
    seed: int,
    image_keys: tuple[str, ...] = ("agentview_image", "robot0_eye_in_hand_image"),
) -> list[dict[str, Any]]:
    """Replay recorded actions; return steps with raw_obs for HDF5 export."""
    reset_env_for_replay(env)
    attach_flags = trajectory_attachment_flags(trajectory)
    raw_traj: list[dict[str, Any]] = []
    prev_attach = False
    for step, want_attach in zip(trajectory, attach_flags):
        apply_attachment_flag(env, want_attach, prev_attach)
        prev_attach = want_attach
        action = np.asarray(step["action"], dtype=np.float32)
        obs, reward, done, info = env.step(clip_action(env, action))
        entry = {
            "raw_obs": dict(obs),
            "action": action.copy(),
            "reward": float(step.get("reward", reward)),
            "done": bool(step.get("done", done)),
            "attachment_enabled": bool(want_attach),
        }
        raw_traj.append(entry)
    return raw_traj


def replay_hdf5_demo(
    hdf5_path: str | Path,
    demo: str,
    *,
    seed: int | None = None,
    grasp_mode: str = "attachment",
    robot: str = "Panda",
    cable_model: str = "composite_cable",
    difficulty: str = "easy",
    horizon: int = 600,
    use_recorded_attachment: bool = True,
    max_steps: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Replay one HDF5 demo; returns final info metrics and per-step rows."""
    path = Path(hdf5_path).expanduser()
    with h5py.File(path, "r") as f:
        grp = f["data"][demo]
        actions = np.asarray(grp["actions"], dtype=np.float32)
        if "attachment_enabled" in grp:
            attach = np.asarray(grp["attachment_enabled"], dtype=bool)
        else:
            attach = np.zeros(len(actions), dtype=bool)
        meta_raw = grp.attrs.get("benchmark_episode_metadata", "{}")
        if isinstance(meta_raw, bytes):
            meta_raw = meta_raw.decode("utf-8")
        meta = json.loads(meta_raw) if meta_raw else {}
        env_args_raw = f["data"].attrs.get("env_args", "{}")
        if isinstance(env_args_raw, bytes):
            env_args_raw = env_args_raw.decode("utf-8")
        env_args = json.loads(env_args_raw) if env_args_raw else {}

    if seed is None:
        seed = int(meta.get("seed", env_args.get("seed", 0)))
    robot = str(env_args.get("robot", robot))
    cable_model = str(env_args.get("cable_model", cable_model))
    difficulty = str(env_args.get("difficulty", difficulty))
    grasp_mode = str(env_args.get("grasp_mode", grasp_mode))

    n_steps = len(actions) if max_steps is None else min(len(actions), max_steps)
    env = make_env(
        robot=robot,
        cable_model=cable_model,
        grasp_mode=grasp_mode,
        difficulty=difficulty,
        horizon=horizon,
        seed=seed,
        has_offscreen_renderer=False,
        use_camera_obs=False,
    )
    reset_env_for_replay(env)
    rows = []
    prev_attach = False
    last_info: dict[str, Any] = {}
    for t in range(n_steps):
        want = bool(attach[t]) if use_recorded_attachment else False
        apply_attachment_flag(env, want, prev_attach)
        prev_attach = want
        _, _, _, info = env.step(clip_action(env, actions[t]))
        last_info = dict(info)
        rows.append(dict(info))
    env.close()
    return last_info, rows


def load_npz_trajectories(npz_path: str | Path) -> tuple[list[list[dict[str, Any]]], list[dict[str, Any]], dict[str, Any]]:
    """Reconstruct trajectory list from platform NPZ (for HDF5 rebuild)."""
    from .utils import _expert_phases_for_env, _phase_cfg_for_env, make_env

    data = np.load(npz_path, allow_pickle=True)
    lengths = np.asarray(data["episode_lengths"], dtype=np.int32)
    actions = np.asarray(data["actions"], dtype=np.float32)
    rewards = np.asarray(data["rewards"], dtype=np.float32)
    dones = np.asarray(data["dones"], dtype=bool)
    phases = np.asarray(data["phases"], dtype=object)

    meta_raw = data["metadata"]
    if hasattr(meta_raw, "item"):
        meta_raw = meta_raw.item()
    metadata = json.loads(meta_raw) if isinstance(meta_raw, str) else dict(meta_raw)

    ep_meta = []
    for item in np.asarray(data["episode_metadata"], dtype=object):
        ep_meta.append(json.loads(item) if isinstance(item, str) else dict(item))

    env = make_env(
        robot=str(metadata.get("robot", "Panda")),
        cable_model=str(metadata.get("cable_model", "composite_cable")),
        grasp_mode=str(metadata.get("grasp_mode", "attachment")),
        difficulty=str(metadata.get("difficulty", "easy")),
        horizon=int(metadata.get("horizon", 600)),
        seed=int(metadata.get("seed", 0)),
        has_offscreen_renderer=False,
        use_camera_obs=False,
    )
    phase_cfg_by_name = {
        cfg["name"]: _phase_cfg_for_env(env, cfg) for cfg in _expert_phases_for_env(env)
    }
    env.close()

    def _attachment_flags(phase_seq: list[str]) -> list[bool]:
        attached = False
        phase_counters: dict[str, int] = {}
        flags: list[bool] = []
        for ph in phase_seq:
            ls = phase_counters.get(ph, 0)
            cfg = phase_cfg_by_name.get(ph)
            if cfg and ls == cfg.get("attach_on_step", -1):
                attached = True
            if cfg and ls == cfg.get("detach_on_step", -1):
                attached = False
            phase_counters[ph] = ls + 1
            flags.append(bool(attached))
        return flags

    trajectories: list[list[dict[str, Any]]] = []
    offset = 0
    for length in lengths:
        phase_seq = [str(phases[offset + i]) for i in range(int(length))]
        attach_flags = _attachment_flags(phase_seq)
        traj = []
        for i in range(int(length)):
            idx = offset + i
            traj.append(
                {
                    "action": actions[idx].copy(),
                    "reward": float(rewards[idx]),
                    "done": bool(dones[idx]),
                    "phase": phase_seq[i],
                    "attachment_enabled": attach_flags[i],
                }
            )
        trajectories.append(traj)
        offset += int(length)
    return trajectories, ep_meta, metadata
