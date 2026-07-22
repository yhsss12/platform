from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def _demo_keys(data_group: h5py.Group) -> list[str]:
    return sorted(k for k in data_group.keys() if k.startswith("demo_"))


def copy_hdf5_with_demo_tags(
    source: Path,
    destination: Path,
    *,
    demo_source: str,
    generation_mode: str,
    policy_mode: str,
    enhancement_mode: str | None = None,
) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    count = 0
    with h5py.File(destination, "a") as f:
        data = f.get("data")
        if data is None:
            return 0
        for demo_key in _demo_keys(data):
            grp = data[demo_key]
            grp.attrs["demo_source"] = demo_source
            grp.attrs["generationMode"] = generation_mode
            grp.attrs["policyMode"] = policy_mode
            if enhancement_mode:
                grp.attrs["enhancementMode"] = enhancement_mode
            grp.attrs["valid_for_training"] = True
            count += 1
        data.attrs["total"] = count
    return count


def append_repaired_demo(
    hdf5_path: Path,
    episode: dict[str, Any],
    *,
    metadata: dict[str, Any],
    env_name: str,
) -> str:
    hdf5_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if hdf5_path.is_file() else "w"
    with h5py.File(hdf5_path, mode) as f:
        if "data" not in f:
            data_grp = f.create_group("data")
            env_args = json.dumps(
                {
                    "env_name": env_name,
                    "type": 1,
                    "env_kwargs": {"has_renderer": False},
                }
            )
            data_grp.attrs["env_args"] = env_args
        else:
            data_grp = f["data"]

        existing = _demo_keys(data_grp)
        next_idx = len(existing)
        demo_name = f"demo_{next_idx}"
        demo_grp = data_grp.create_group(demo_name)

        actions = np.asarray(episode["actions"], dtype=np.float32)
        demo_grp.create_dataset("actions", data=actions, compression="gzip")
        obs_grp = demo_grp.create_group("obs")
        for key, values in (episode.get("obs") or {}).items():
            obs_grp.create_dataset(key, data=np.asarray(values), compression="gzip")
        if episode.get("states") is not None:
            demo_grp.create_dataset("states", data=np.asarray(episode["states"]), compression="gzip")

        demo_grp.attrs["num_samples"] = int(actions.shape[0])
        demo_grp.attrs["demo_source"] = "pinn_repaired"
        demo_grp.attrs["generationMode"] = metadata.get("generationMode", "mimicgen_datagen")
        demo_grp.attrs["enhancementMode"] = metadata.get("enhancementMode", "pinn_repair")
        demo_grp.attrs["pinn_model_id"] = metadata.get("pinn_model_id", "nut_assembly_pinn_v1")
        if metadata.get("pinnBackend"):
            demo_grp.attrs["pinnBackend"] = metadata.get("pinnBackend")
        demo_grp.attrs["repair_parent"] = metadata.get("repair_parent", demo_name)
        demo_grp.attrs["validation_success"] = bool(metadata.get("validation_success", False))
        demo_grp.attrs["repair_success"] = bool(metadata.get("repair_success", False))
        demo_grp.attrs["valid_for_training"] = bool(metadata.get("valid_for_training", False))
        demo_grp.attrs["success"] = bool(metadata.get("validation_success", False))
        demo_grp.attrs["success_flag"] = bool(metadata.get("validation_success", False))
        if metadata.get("final_xy_error") is not None:
            demo_grp.attrs["final_xy_error"] = float(metadata["final_xy_error"])
        if metadata.get("final_height_error") is not None:
            demo_grp.attrs["final_height_error"] = float(metadata["final_height_error"])
        demo_grp.attrs["benchmark_episode_metadata"] = json.dumps(metadata, ensure_ascii=False)
        data_grp.attrs["total"] = len(_demo_keys(data_grp))
        return demo_name


def score_demo_for_repair(demo_grp: h5py.Group) -> float:
    score = 0.0
    failure_type = str(demo_grp.attrs.get("failure_type", "") or "")
    if failure_type in {"alignment_failed", "insertion_failed"}:
        score += 100.0
    elif failure_type not in {"", "success", "unknown_failure"}:
        score += 40.0

    meta_raw = demo_grp.attrs.get("benchmark_episode_metadata")
    if meta_raw:
        try:
            meta = json.loads(meta_raw) if isinstance(meta_raw, str) else dict(meta_raw)
            xy = meta.get("final_xy_error")
            if xy is not None:
                score += float(xy) * 1000.0
            if not meta.get("alignment_success", True):
                score += 50.0
            if not meta.get("insertion_success", True):
                score += 50.0
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    return score


def select_repair_candidates(hdf5_path: Path, *, max_candidates: int) -> list[dict[str, Any]]:
    if not hdf5_path.is_file():
        return []
    ranked: list[tuple[float, str]] = []
    with h5py.File(hdf5_path, "r") as f:
        data = f.get("data")
        if data is None:
            return []
        for demo_key in _demo_keys(data):
            score = score_demo_for_repair(data[demo_key])
            if score > 0:
                ranked.append((score, demo_key))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [{"demoKey": key, "score": score} for score, key in ranked[:max_candidates]]
