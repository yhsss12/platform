"""将 repair 成功的 rollout 写入 MimicGen 风格 HDF5（不修改 object_poses）。"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np


def _copy_static_datagen_fields(src_di: h5py.Group, dest_di: h5py.Group, *, length: int) -> None:
    """复制 target_pose / subtask_term_signals / object_poses（object_poses 原样，不修改）。"""
    if "target_pose" in src_di:
        dest_di.create_dataset("target_pose", data=src_di["target_pose"][:length])

    if "subtask_term_signals" in src_di:
        sig_grp = dest_di.create_group("subtask_term_signals")
        for key in src_di["subtask_term_signals"].keys():
            sig_grp.create_dataset(key, data=src_di["subtask_term_signals"][key][:length])

    if "object_poses" in src_di:
        obj_src = src_di["object_poses"]
        obj_dst = dest_di.create_group("object_poses")
        for key in obj_src.keys():
            obj_dst.create_dataset(key, data=obj_src[key][:])


def init_repaired_dataset(output_hdf5: Path, source_hdf5: Path) -> None:
    output_hdf5.parent.mkdir(parents=True, exist_ok=True)
    if output_hdf5.exists():
        return
    with h5py.File(source_hdf5, "r") as src, h5py.File(output_hdf5, "w") as out:
        data = out.create_group("data")
        for attr_key in src["data"].attrs.keys():
            data.attrs[attr_key] = src["data"].attrs[attr_key]


def append_successful_repair_demo(
    *,
    output_hdf5: Path,
    source_hdf5: Path,
    source_demo_key: str,
    repaired_demo_key: str,
    rollout: dict,
    meta: dict,
) -> None:
    if not rollout.get("success_flag"):
        raise ValueError("only success_flag=true rollouts may be written to repaired_dataset.hdf5")

    actions = np.asarray(rollout["recorded_actions"], dtype=np.float64)
    states = np.asarray(rollout["recorded_states"], dtype=np.float64)
    eef_pose = np.asarray(rollout["recorded_eef_pose"], dtype=np.float64)
    gripper = np.asarray(rollout["recorded_gripper_action"], dtype=np.float64)
    length = actions.shape[0]

    write_mode = "a" if output_hdf5.exists() else "w"
    with h5py.File(source_hdf5, "r") as src_handle, h5py.File(output_hdf5, write_mode) as out_handle:
        if "data" not in out_handle:
            data_grp = out_handle.create_group("data")
            for attr_key in src_handle["data"].attrs.keys():
                data_grp.attrs[attr_key] = src_handle["data"].attrs[attr_key]
        else:
            data_grp = out_handle["data"]

        if repaired_demo_key in data_grp:
            del data_grp[repaired_demo_key]

        src_demo = src_handle[f"data/{source_demo_key}"]
        dst_demo = data_grp.create_group(repaired_demo_key)
        if "model_file" in src_demo.attrs:
            dst_demo.attrs["model_file"] = src_demo.attrs["model_file"]

        dst_demo.create_dataset("actions", data=actions)
        dst_demo.create_dataset("states", data=states)

        src_di = src_demo["datagen_info"]
        di = dst_demo.create_group("datagen_info")
        di.create_dataset("eef_pose", data=eef_pose[:length])
        di.create_dataset("gripper_action", data=gripper[:length])
        _copy_static_datagen_fields(src_di, di, length=length)

        if "src_demo_inds" in src_demo:
            dst_demo.create_dataset("src_demo_inds", data=src_demo["src_demo_inds"][:length])
        if "src_demo_labels" in src_demo:
            dst_demo.create_dataset("src_demo_labels", data=src_demo["src_demo_labels"][:length])

        dst_demo.attrs["repair_meta_json"] = json.dumps(meta)
