from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def save_rollout_hdf5(
    output_path: Path,
    episodes: list[dict[str, Any]],
    *,
    env_name: str,
    generation_mode: str = "robosuite_rollout",
    policy_mode: str = "partial_scripted",
    env_type: int = 1,
) -> dict[str, Any]:
    """Write robomimic-compatible HDF5 from rollout episodes with episode metadata attrs."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_steps = 0
    success_episodes = 0
    valid_for_training = 0
    has_episode_metadata = False
    has_object_poses = False

    with h5py.File(output_path, "w") as f:
        data_grp = f.create_group("data")
        data_grp.attrs["total"] = len(episodes)
        env_args = json.dumps(
            {
                "env_name": env_name,
                "type": env_type,
                "env_kwargs": {"has_renderer": False},
                "generation_mode": generation_mode,
                "policy_mode": policy_mode,
            }
        )
        data_grp.attrs["env_args"] = env_args

        if episodes:
            first_obs = episodes[0].get("obs", {})
            data_grp.attrs["obs_keys"] = json.dumps(sorted(str(key) for key in first_obs.keys()))

        train_demo_names: list[str] = []

        for idx, ep in enumerate(episodes):
            demo_name = f"demo_{idx}"
            demo_grp = data_grp.create_group(demo_name)
            actions = np.asarray(ep["actions"], dtype=np.float32)
            if actions.size == 0:
                actions = np.zeros((0, 7), dtype=np.float32)
            demo_grp.create_dataset("actions", data=actions, compression="gzip")
            total_steps += int(actions.shape[0])

            obs_grp = demo_grp.create_group("obs")
            for key, values in ep.get("obs", {}).items():
                obs_grp.create_dataset(key, data=np.asarray(values), compression="gzip")

            if ep.get("states") is not None:
                demo_grp.create_dataset("states", data=np.asarray(ep["states"]), compression="gzip")

            if ep.get("datagen_info"):
                dg_grp = demo_grp.create_group("datagen_info")
                for key, values in ep["datagen_info"].items():
                    if key == "object_poses" and isinstance(values, dict) and values:
                        obj_grp = dg_grp.create_group("object_poses")
                        for obj_key, obj_values in values.items():
                            obj_grp.create_dataset(obj_key, data=np.asarray(obj_values), compression="gzip")
                        has_object_poses = True
                    elif key == "object_poses":
                        continue
                    else:
                        dg_grp.create_dataset(key, data=np.asarray(values), compression="gzip")

            metadata = ep.get("metadata") or {}
            if not metadata:
                metadata = {
                    "success_flag": bool(ep.get("success", False)),
                    "failure_type": "success" if ep.get("success") else "unknown_failure",
                    "episode_steps": int(actions.shape[0]),
                    "env_name": env_name,
                    "generation_mode": generation_mode,
                    "policy_mode": policy_mode,
                    "seed": 0,
                }

            demo_grp.attrs["num_samples"] = int(actions.shape[0])
            demo_grp.attrs["success"] = bool(metadata.get("success_flag", ep.get("success", False)))
            demo_grp.attrs["success_flag"] = bool(metadata.get("success_flag", False))
            demo_grp.attrs["failure_type"] = str(metadata.get("failure_type", "unknown_failure"))
            demo_grp.attrs["episode_steps"] = int(metadata.get("episode_steps", actions.shape[0]))
            demo_grp.attrs["env_name"] = str(metadata.get("env_name", env_name))
            demo_grp.attrs["generation_mode"] = str(metadata.get("generation_mode", generation_mode))
            demo_grp.attrs["policy_mode"] = str(metadata.get("policy_mode", policy_mode))
            demo_grp.attrs["seed"] = int(metadata.get("seed", 0))
            demo_grp.attrs["benchmark_episode_metadata"] = json.dumps(metadata, ensure_ascii=False)
            if metadata.get("valid_for_training") is not None:
                demo_grp.attrs["valid_for_training"] = bool(metadata.get("valid_for_training"))
            if metadata.get("grasp_attempts") is not None:
                demo_grp.attrs["grasp_attempts"] = int(metadata.get("grasp_attempts"))
            has_episode_metadata = True

            if metadata.get("success_flag"):
                success_episodes += 1
            if metadata.get("valid_for_training"):
                valid_for_training += 1
                train_demo_names.append(demo_name)

        if train_demo_names:
            mask_grp = f.create_group("mask")
            mask_grp.create_dataset("train", data=np.asarray(train_demo_names, dtype="S"))

    return {
        "demoCount": len(episodes),
        "totalSteps": total_steps,
        "successEpisodes": success_episodes,
        "validForTrainingEpisodes": valid_for_training,
        "path": str(output_path),
        "hasEpisodeMetadata": has_episode_metadata,
        "hasObjectPoses": has_object_poses,
    }
