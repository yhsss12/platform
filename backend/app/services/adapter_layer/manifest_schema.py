from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

MANIFEST_VERSION = "1.0"


class ObservationSpaceSpec(BaseModel):
    """观测空间描述。"""

    type: str = "low_dim"
    keys: list[str] = Field(default_factory=list)
    dims: dict[str, int] = Field(default_factory=dict)


class ActionSpaceSpec(BaseModel):
    """动作空间描述。"""

    type: str = "continuous"
    dim: Optional[int] = None
    supportsSequence: bool = False
    horizon: Optional[int] = None


class DatasetManifest(BaseModel):
    """标准 Dataset Manifest schema。"""

    datasetId: str
    datasetName: str = ""
    taskName: str = ""
    simulator: str = ""
    robotType: str = ""
    dataFormat: str = "HDF5"
    observationSpace: ObservationSpaceSpec = Field(default_factory=ObservationSpaceSpec)
    actionSpace: ActionSpaceSpec = Field(default_factory=ActionSpaceSpec)
    episodeCount: int = 0
    successCount: int = 0
    horizon: Optional[int] = None
    storageUri: str = ""
    manifestVersion: str = MANIFEST_VERSION
    taskType: Optional[str] = None
    sourceJobId: Optional[str] = None
    raw: dict[str, Any] = Field(default_factory=dict, exclude=True)


def _first_non_empty(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _normalize_simulator(raw: dict[str, Any]) -> str:
    value = _first_non_empty(
        raw.get("simulator"),
        raw.get("simulatorBackend"),
        raw.get("simulatorType"),
        raw.get("backend"),
        raw.get("physicsBackend"),
    )
    return value.lower()


def _normalize_robot_type(raw: dict[str, Any]) -> str:
    explicit = _first_non_empty(raw.get("robotType"), raw.get("robot_type"))
    if explicit:
        return explicit

    task_type = _first_non_empty(raw.get("taskType"), raw.get("task_type")).lower()
    if task_type == "dual_arm_cable_manipulation":
        return "dual_fr3"
    if task_type in {"isaac_block_stacking", "isaaclab_franka_stack_cube", "block_stacking"}:
        return "franka_panda"
    if task_type in {"cable_threading", "cable_threading_single_arm"}:
        robots = raw.get("supportedRobotTypes") or raw.get("robotTypes") or []
        if isinstance(robots, list) and robots:
            return str(robots[0])
        return "Panda"
    return ""


def _normalize_data_format(raw: dict[str, Any]) -> str:
    value = _first_non_empty(raw.get("dataFormat"), raw.get("format"), raw.get("mainFormats"))
    if isinstance(raw.get("mainFormats"), list) and raw["mainFormats"]:
        value = str(raw["mainFormats"][0])
    return value.upper() if value else "HDF5"


def _normalize_storage_uri(raw: dict[str, Any]) -> str:
    artifacts = raw.get("artifacts")
    if isinstance(artifacts, dict):
        for key in ("hdf5", "npz", "zarr", "dataset"):
            path = artifacts.get(key)
            if path:
                return str(path)
    return _first_non_empty(
        raw.get("storageUri"),
        raw.get("storagePath"),
        raw.get("datasetFile"),
        raw.get("hdf5"),
    )


def _normalize_observation_space(raw: dict[str, Any]) -> ObservationSpaceSpec:
    obs_raw = raw.get("observationSpace")
    if isinstance(obs_raw, dict):
        keys = [str(k) for k in (obs_raw.get("keys") or [])]
        if not keys:
            keys = [str(k) for k in (raw.get("obsKeys") or raw.get("observationKeys") or [])]
        obs_type = str(obs_raw.get("type") or "low_dim")
        if keys and any("image" in key.lower() for key in keys):
            obs_type = "image" if not any(k for k in keys if "image" not in k.lower()) else "mixed"
        return ObservationSpaceSpec(
            type=obs_type,
            keys=keys,
            dims={str(k): int(v) for k, v in (obs_raw.get("dims") or {}).items()},
        )

    quality = raw.get("quality") if isinstance(raw.get("quality"), dict) else {}
    obs_keys = raw.get("obsKeys") or raw.get("observationKeys") or []
    camera_keys = raw.get("cameraKeys") or raw.get("imageKeys") or []
    if not isinstance(obs_keys, list):
        obs_keys = []
    if camera_keys and not obs_keys:
        obs_keys = list(camera_keys)

    obs_schema = raw.get("observationSchema")
    if obs_schema and not obs_keys:
        obs_keys = list(_default_obs_keys_for_schema(str(obs_schema)))

    has_image = bool(quality.get("hasImage")) or bool(camera_keys)
    obs_type = str(raw.get("observationType") or "")
    if not obs_type:
        obs_type = "image" if has_image else "low_dim"
    if obs_keys and any("image" in key.lower() for key in obs_keys):
        obs_type = "image" if not any(k for k in obs_keys if "image" not in k.lower()) else "mixed"
    elif camera_keys and obs_keys:
        state_keys = [k for k in obs_keys if k not in camera_keys]
        if state_keys:
            obs_type = "mixed"
        else:
            obs_type = "image"
    elif obs_type == "low_dim" and obs_keys:
        obs_type = "mixed" if any("image" in key.lower() for key in obs_keys) else "low_dim"

    dims: dict[str, int] = {}
    obs_dims = raw.get("observationDims") or raw.get("obsDims")
    if isinstance(obs_dims, dict):
        dims = {str(k): int(v) for k, v in obs_dims.items()}

    return ObservationSpaceSpec(type=obs_type, keys=[str(k) for k in obs_keys], dims=dims)


def _default_obs_keys_for_schema(schema: str) -> tuple[str, ...]:
    if schema == "dual_arm_cable_il_v1":
        return ("left_arm_qpos", "right_arm_qpos", "cable_state")
    return ()


def _normalize_action_space(raw: dict[str, Any]) -> ActionSpaceSpec:
    action_raw = raw.get("actionSpace")
    if isinstance(action_raw, dict):
        return ActionSpaceSpec(
            type=str(action_raw.get("type") or "continuous"),
            dim=action_raw.get("dim"),
            supportsSequence=bool(action_raw.get("supportsSequence")),
            horizon=action_raw.get("horizon"),
        )

    action_dim = raw.get("actionDim") or raw.get("action_dim")
    horizon = raw.get("horizon") or raw.get("actionHorizon") or raw.get("action_horizon")
    quality = raw.get("quality") if isinstance(raw.get("quality"), dict) else {}

    supports_sequence = bool(
        raw.get("supportsSequenceActions")
        or raw.get("actionSequence")
        or quality.get("hasTimeline")
        or (horizon is not None and int(horizon) > 1)
    )

    action_schema = str(raw.get("actionSchema") or "")
    if "sequence" in action_schema.lower() or "chunk" in action_schema.lower():
        supports_sequence = True

    dim_value: Optional[int] = None
    if action_dim is not None:
        try:
            dim_value = int(action_dim)
        except (TypeError, ValueError):
            dim_value = None

    horizon_value: Optional[int] = None
    if horizon is not None:
        try:
            horizon_value = int(horizon)
        except (TypeError, ValueError):
            horizon_value = None

    return ActionSpaceSpec(
        type="continuous",
        dim=dim_value,
        supportsSequence=supports_sequence,
        horizon=horizon_value,
    )


def normalize_dataset_manifest(raw_manifest: dict[str, Any]) -> DatasetManifest:
    """将平台原始 manifest 规范化为标准 DatasetManifest。"""
    raw = dict(raw_manifest or {})
    episode_count = int(raw.get("episodeCount") or raw.get("episodes") or raw.get("totalEpisodes") or 0)
    success_count = int(
        raw.get("successCount")
        or raw.get("successfulEpisodes")
        or raw.get("validTrajectories")
        or episode_count
    )

    dataset_id = _first_non_empty(raw.get("datasetId"), raw.get("id"), raw.get("sourceJobId"), "unknown")
    dataset_name = _first_non_empty(
        raw.get("datasetName"),
        raw.get("name"),
        raw.get("displayName"),
        raw.get("sourceRecordName"),
        dataset_id,
    )
    task_name = _first_non_empty(raw.get("taskName"), raw.get("taskDisplayName"), raw.get("taskType"))

    manifest = DatasetManifest(
        datasetId=dataset_id,
        datasetName=dataset_name,
        taskName=task_name,
        simulator=_normalize_simulator(raw),
        robotType=_normalize_robot_type(raw),
        dataFormat=_normalize_data_format(raw),
        observationSpace=_normalize_observation_space(raw),
        actionSpace=_normalize_action_space(raw),
        episodeCount=episode_count,
        successCount=success_count,
        horizon=raw.get("horizon"),
        storageUri=_normalize_storage_uri(raw),
        manifestVersion=str(raw.get("manifestVersion") or MANIFEST_VERSION),
        taskType=_first_non_empty(raw.get("taskType"), raw.get("task_type")) or None,
        sourceJobId=_first_non_empty(raw.get("sourceJobId")) or None,
        raw=raw,
    )
    if manifest.horizon is None and manifest.actionSpace.horizon is not None:
        manifest.horizon = manifest.actionSpace.horizon
    return manifest
