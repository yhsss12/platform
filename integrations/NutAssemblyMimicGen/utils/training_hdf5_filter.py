from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np

FilterMode = Literal["all", "all_generated_demos", "success_only", "valid_for_training_only"]

DEFAULT_FILTER_MODE: FilterMode = "valid_for_training_only"
NO_TRAINABLE_MESSAGE = "当前数据集中暂无可训练成功轨迹，请先生成更多数据或优化策略。"


def normalize_filter_mode(mode: str) -> FilterMode:
    normalized = (mode or "").strip()
    if normalized == "all_generated_demos":
        return "all_generated_demos"
    if normalized in {"all", "success_only", "valid_for_training_only"}:
        return normalized  # type: ignore[return-value]
    return DEFAULT_FILTER_MODE


def _demo_passes_filter(attrs: dict[str, Any], mode: FilterMode) -> bool:
    if mode in {"all", "all_generated_demos"}:
        return True
    if mode == "success_only":
        return bool(attrs.get("success_flag") or attrs.get("success"))
    return bool(attrs.get("valid_for_training"))


def _attrs_from_demo(demo_grp: h5py.Group) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    for key in demo_grp.attrs.keys():
        val = demo_grp.attrs[key]
        if isinstance(val, (bytes, bytearray)):
            val = val.decode("utf-8", errors="replace")
        if key == "benchmark_episode_metadata" and isinstance(val, str):
            try:
                attrs["benchmark_episode_metadata"] = json.loads(val)
            except json.JSONDecodeError:
                attrs[key] = val
        else:
            attrs[key] = val
    return attrs


def _copy_group(src: h5py.Group, dst: h5py.Group) -> None:
    for key, val in src.attrs.items():
        dst.attrs[key] = val
    for name, item in src.items():
        if isinstance(item, h5py.Dataset):
            dst.create_dataset(name, data=item[()], compression="gzip")
        else:
            _copy_group(item, dst.create_group(name))


def probe_training_filter(
    source_hdf5: Path,
    *,
    filter_mode: FilterMode = DEFAULT_FILTER_MODE,
) -> dict[str, Any]:
    if not source_hdf5.is_file():
        return {"ok": False, "error": f"source hdf5 not found: {source_hdf5}"}

    total = 0
    success = 0
    valid_for_training = 0
    selected_keys: list[str] = []

    with h5py.File(source_hdf5, "r") as f:
        data_grp = f.get("data")
        if data_grp is None:
            return {"ok": False, "error": "missing data group"}
        for demo_key in sorted(data_grp.keys()):
            if not demo_key.startswith("demo_"):
                continue
            total += 1
            demo = data_grp[demo_key]
            attrs = _attrs_from_demo(demo)
            if attrs.get("success_flag") or attrs.get("success"):
                success += 1
            if attrs.get("valid_for_training"):
                valid_for_training += 1
            if _demo_passes_filter(attrs, filter_mode):
                selected_keys.append(demo_key)

    return {
        "ok": True,
        "filterMode": filter_mode,
        "totalDemos": total,
        "successDemos": success,
        "validForTrainingDemos": valid_for_training,
        "selectedDemos": len(selected_keys),
        "filteredDemos": total - len(selected_keys),
        "selectedKeys": selected_keys,
        "canBuild": len(selected_keys) > 0,
    }


def _infer_training_obs_keys(src_data: h5py.Group, selected_keys: list[str]) -> list[str]:
    preferred = ["robot0_eef_pos", "robot0_gripper_qpos", "object"]
    if not selected_keys:
        return preferred
    demo = src_data[selected_keys[0]]
    obs = demo.get("obs")
    if obs is None:
        return preferred
    available = set(obs.keys())
    chosen = [key for key in preferred if key in available]
    if chosen:
        return chosen
    return sorted(key for key in available if isinstance(obs[key], h5py.Dataset))


def build_filtered_training_hdf5(
    *,
    source_hdf5: Path,
    output_hdf5: Path,
    filter_mode: FilterMode = DEFAULT_FILTER_MODE,
    task_template_id: str = "nut_assembly_single_arm",
    task_type: str = "nut_assembly",
    source_job_id: str | None = None,
) -> dict[str, Any]:
    probe = probe_training_filter(source_hdf5, filter_mode=filter_mode)
    if not probe.get("ok"):
        return probe
    if not probe.get("canBuild"):
        return {
            **probe,
            "ok": False,
            "error": NO_TRAINABLE_MESSAGE,
            "status": "no_trainable_demos",
        }

    output_hdf5.parent.mkdir(parents=True, exist_ok=True)
    selected_keys: list[str] = probe["selectedKeys"]

    with h5py.File(source_hdf5, "r") as src, h5py.File(output_hdf5, "w") as dst:
        src_data = src["data"]
        dst_data = dst.create_group("data")
        for attr_key, attr_val in src_data.attrs.items():
            dst_data.attrs[attr_key] = attr_val
        env_args_raw = dst_data.attrs.get("env_args")
        if isinstance(env_args_raw, (str, bytes)):
            try:
                env_args = json.loads(env_args_raw if isinstance(env_args_raw, str) else env_args_raw.decode())
                env_args["filter_mode"] = filter_mode
                env_args["task_template_id"] = task_template_id
                env_args["task_type"] = task_type
                dst_data.attrs["env_args"] = json.dumps(env_args)
            except json.JSONDecodeError:
                pass
        dst_data.attrs["filter_mode"] = filter_mode
        dst_data.attrs["task_template_id"] = task_template_id
        dst_data.attrs["task_type"] = task_type
        dst_data.attrs["source_job_id"] = source_job_id or ""
        dst_data.attrs["total"] = len(selected_keys)
        dst_data.attrs["obs_keys"] = json.dumps(_infer_training_obs_keys(src_data, selected_keys))

        for out_idx, src_key in enumerate(selected_keys):
            out_key = f"demo_{out_idx}"
            _copy_group(src_data[src_key], dst_data.create_group(out_key))

        demo_names = np.array([f"demo_{idx}".encode("utf-8") for idx in range(len(selected_keys))])
        mask_grp = dst.create_group("mask")
        mask_grp.create_dataset("train", data=demo_names)

    return {
        **probe,
        "ok": True,
        "status": "built",
        "outputPath": str(output_hdf5),
        "builtDemoCount": len(selected_keys),
        "taskTemplateId": task_template_id,
        "taskType": task_type,
        "filterMode": filter_mode,
    }
