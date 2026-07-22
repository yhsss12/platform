"""Read platform schema metadata from HDF5 attrs and manifests."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.services.adapter_layer.hdf5_inspector import inspect_hdf5


def _parse_json_attr(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("{") or text.startswith("["):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return text
        return text
    return value


def read_hdf5_data_attrs(hdf5_path: Path | str) -> dict[str, Any]:
    path = Path(hdf5_path).expanduser().resolve()
    if not path.is_file():
        return {}
    try:
        import h5py
    except ImportError:
        return {}
    try:
        with h5py.File(path, "r") as handle:
            data = handle.get("data")
            if data is None:
                return {}
            attrs = {str(k): _parse_json_attr(v) for k, v in dict(data.attrs).items()}
            demo_keys = sorted(k for k in data.keys() if str(k).startswith("demo_"))
            attrs["episode_count"] = len(demo_keys)
            if demo_keys:
                demo0 = data[demo_keys[0]]
                if demo0.get("joint_actions") is not None:
                    attrs["joint_action_available"] = True
                if demo0.get("gripper_actions") is not None:
                    attrs["gripper_action_available"] = True
                if demo0.get("attachment_enabled") is not None:
                    attrs["attachment_side_channel"] = True
            return attrs
    except OSError:
        return {}


def build_dataset_row_from_hdf5(
    hdf5_path: Path,
    *,
    manifest: dict[str, Any] | None = None,
    source_job_id: str | None = None,
) -> dict[str, Any]:
    """Build workspace dataset index fields from HDF5 + optional manifest."""
    manifest = dict(manifest or {})
    attrs = read_hdf5_data_attrs(hdf5_path)
    inspection = inspect_hdf5(hdf5_path)

    joint_available = bool(
        attrs.get("joint_action_available")
        or inspection.joint_action_available
        or "joint_actions" in (attrs.get("available_action_keys") or [])
    )

    action_schema_obj = attrs.get("action_schema")
    if isinstance(action_schema_obj, dict):
        action_mode = str(action_schema_obj.get("actionMode") or "")
        action_schema_id = action_schema_obj.get("id")
    else:
        action_mode = str(attrs.get("trained_action_mode") or manifest.get("trainedActionMode") or "")
        action_schema_id = manifest.get("actionSchema")

    if joint_available and not action_mode:
        action_mode = "joint_delta"

    eval_executor = str(
        attrs.get("eval_executor")
        or manifest.get("evalExecutor")
        or ("joint_position" if joint_available else "osc_pose")
    )

    observation_schema_obj = attrs.get("observation_schema")
    controller_schema_obj = attrs.get("controller_schema")
    side_channel_schema_obj = attrs.get("side_channel_schema")

    episode_count = int(
        inspection.episode_count
        or attrs.get("episode_count")
        or manifest.get("num_successful")
        or manifest.get("episodeCount")
        or 0
    )

    job_id = source_job_id or manifest.get("sourceJobId") or manifest.get("jobId")
    dataset_id = str(manifest.get("datasetId") or (f"ds_{job_id}" if job_id else ""))

    row: dict[str, Any] = {
        "id": dataset_id or f"ds_{hdf5_path.parent.parent.name}",
        "datasetId": dataset_id,
        "sourceJobId": job_id,
        "format": "hdf5",
        "datasetFormat": "hdf5",
        "episodeCount": episode_count,
        "actionDim": inspection.action_dim or manifest.get("action_dim") or (8 if joint_available else 7),
        "taskTemplateId": str(
            attrs.get("task_template_id")
            or manifest.get("taskTemplateId")
            or "cable_threading_single_arm"
        ),
        "taskType": str(attrs.get("task_type") or manifest.get("taskType") or "cable_threading"),
        "robotType": str(manifest.get("robot") or manifest.get("robotType") or "Panda"),
        "simulatorBackend": str(manifest.get("simulatorBackend") or "mujoco"),
        "jointActionAvailable": joint_available,
        "trainedActionMode": action_mode or ("joint_delta" if joint_available else "osc_pose_delta_eef"),
        "evalExecutor": eval_executor,
        "attachmentSideChannel": bool(
            attrs.get("attachment_side_channel") or manifest.get("attachmentSideChannel")
        ),
        "hdf5Path": str(hdf5_path),
        "manifestPath": str(manifest.get("manifestPath") or ""),
    }

    if isinstance(action_schema_obj, dict):
        row["actionSchema"] = action_schema_obj.get("id")
        row["actionSchemaDetail"] = action_schema_obj
    elif action_schema_id:
        row["actionSchema"] = action_schema_id

    if isinstance(observation_schema_obj, dict):
        row["observationSchema"] = observation_schema_obj.get("id")
        row["observationSchemaDetail"] = observation_schema_obj
    elif manifest.get("observationSchema"):
        row["observationSchema"] = manifest.get("observationSchema")

    if isinstance(controller_schema_obj, dict):
        row["controllerSchema"] = controller_schema_obj.get("id")
        row["controllerSchemaDetail"] = controller_schema_obj
    elif manifest.get("controllerSchema"):
        row["controllerSchema"] = manifest.get("controllerSchema")

    if isinstance(side_channel_schema_obj, dict):
        row["sideChannelSchema"] = side_channel_schema_obj.get("id")

    if attrs.get("available_action_keys"):
        row["availableActionKeys"] = list(attrs.get("available_action_keys") or [])
    if attrs.get("policy_schemas"):
        row["policySchemas"] = attrs.get("policy_schemas")
    if attrs.get("side_channel_keys"):
        row["sideChannelKeys"] = list(attrs.get("side_channel_keys") or [])

    return row
