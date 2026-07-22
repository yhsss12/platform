from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py


def _demo_keys(data_group: h5py.Group) -> list[str]:
    return sorted(k for k in data_group.keys() if k.startswith("demo_"))


def inspect_hdf5_dataset(path: Path) -> dict[str, Any]:
    """Inspect robomimic HDF5 for demo count, datagen_info, object_poses."""
    result: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
        "demoCount": 0,
        "totalSteps": 0,
        "hasDatagenInfo": False,
        "hasObjectPoses": False,
        "hasSubtaskTermSignals": False,
        "objectPoseKeys": [],
        "subtaskTermSignalKeys": [],
        "hasEpisodeMetadata": False,
        "validEpisodes": None,
        "successEpisodes": None,
    }
    if not path.is_file():
        return result

    with h5py.File(path, "r") as f:
        if "data" not in f:
            return result
        data = f["data"]
        demos = _demo_keys(data)
        result["demoCount"] = len(demos)
        steps = 0
        object_pose_keys: set[str] = set()
        subtask_keys: set[str] = set()
        for demo in demos:
            grp = data[demo]
            if "actions" in grp:
                steps += int(grp["actions"].shape[0])
            if "datagen_info" in grp:
                result["hasDatagenInfo"] = True
                dg = grp["datagen_info"]
                if "object_poses" in dg:
                    result["hasObjectPoses"] = True
                    for k in dg["object_poses"].keys():
                        object_pose_keys.add(str(k))
                if "subtask_term_signals" in dg:
                    result["hasSubtaskTermSignals"] = True
                    for k in dg["subtask_term_signals"].keys():
                        subtask_keys.add(str(k))
            attrs = dict(grp.attrs)
            if attrs.get("successful") is not None or attrs.get("success") is not None:
                result["hasEpisodeMetadata"] = True

        result["totalSteps"] = steps
        result["objectPoseKeys"] = sorted(object_pose_keys)
        result["subtaskTermSignalKeys"] = sorted(subtask_keys)

    return result


def source_demo_already_prepared(path: Path) -> bool:
    info = inspect_hdf5_dataset(path)
    if not (info.get("hasDatagenInfo") and info.get("hasObjectPoses")):
        return False
    try:
        import h5py

        with h5py.File(path, "r") as f:
            data = f.get("data")
            if data is None:
                return False
            for demo in _demo_keys(data):
                dg = data[demo].get("datagen_info")
                if dg is None:
                    return False
                if "env_interface_name" not in dg.attrs:
                    return False
        return True
    except Exception:
        return False
