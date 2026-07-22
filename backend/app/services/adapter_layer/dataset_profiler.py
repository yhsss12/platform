from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from app.services.adapter_layer.hdf5_inspector import Hdf5InspectionResult, inspect_hdf5
from app.services.adapter_layer.manifest_schema import DatasetManifest, normalize_dataset_manifest


SIMULATOR_DISPLAY = {
    "mujoco": "MuJoCo",
    "isaac": "Isaac",
    "isaac_lab": "Isaac Lab",
    "isaacsim": "Isaac Sim",
    "physx": "Isaac",
}


def _display_simulator(raw: str) -> str:
    key = (raw or "").lower()
    for token, label in SIMULATOR_DISPLAY.items():
        if token in key:
            return label
    return raw or "unknown"


def _infer_observation_type(keys: list[str], camera_keys: list[str]) -> str:
    if not keys:
        return "unknown"
    if camera_keys and len(camera_keys) == len(keys):
        return "image"
    if camera_keys:
        return "mixed"
    return "low_dim"


def _estimate_state_dim(state_keys: list[str], obs_dims: dict[str, int]) -> int:
    total = 0
    for key in state_keys:
        if key in obs_dims:
            total += int(obs_dims[key])
        else:
            total += 1
    return total


@dataclass
class DatasetProfile:
    datasetId: str = ""
    datasetName: str = ""
    taskName: str = ""
    simulator: str = "unknown"
    robotType: str = "unknown"
    episodeCount: int = 0
    successCount: int = 0
    observationType: str = "unknown"
    observationKeys: list[str] = field(default_factory=list)
    cameraKeys: list[str] = field(default_factory=list)
    imageKeys: list[str] = field(default_factory=list)
    imageShape: Optional[dict[str, int]] = None
    stateDim: int = 0
    actionDim: int = 0
    actionSpace: str = "unknown"
    horizon: int = 0
    format: str = "HDF5"
    storageUri: str = ""
    hasValidationSplit: bool = False
    hasReward: bool = False
    hasSuccess: bool = False
    hasDone: bool = False
    taskType: Optional[str] = None
    attachmentSideChannel: bool = False
    attachmentField: str = "attachment_enabled"
    sideChannelKeys: list[str] = field(default_factory=list)
    attachmentPolicy: Optional[str] = None
    attachmentInputMode: str = "not_used_by_policy"
    attachmentControlMode: str = "eval_controller"
    includeAttachmentObs: bool = False
    controllerType: str = "OSC_POSE"
    jointActionAvailable: bool = False
    preferredPolicySchemaId: str = ""
    evalExecutor: str = ""
    trainedActionMode: str = ""
    observationSchema: str | None = None
    actionSchema: str | None = None
    controllerSchema: str | None = None
    sideChannelSchema: str | None = None
    policySchemas: dict[str, Any] = field(default_factory=dict)
    availableActionKeys: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    inferenceSources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_dataset_profile(
    raw_manifest: dict[str, Any],
    *,
    hdf5_path: Path | str | None = None,
) -> DatasetProfile:
    """基于 manifest 与 HDF5 元信息生成 datasetProfile。"""
    manifest = normalize_dataset_manifest(raw_manifest)
    profile = DatasetProfile(
        datasetId=manifest.datasetId,
        datasetName=manifest.datasetName,
        taskName=manifest.taskName,
        simulator=_display_simulator(manifest.simulator),
        robotType=manifest.robotType or "unknown",
        episodeCount=manifest.episodeCount,
        successCount=manifest.successCount,
        format=manifest.dataFormat or "HDF5",
        storageUri=manifest.storageUri,
        taskType=manifest.taskType,
        inferenceSources=["manifest"],
    )

    attach_manifest = raw_manifest.get("attachmentSideChannel")
    if attach_manifest is None:
        attach_manifest = raw_manifest.get("attachment_side_channel")
    if attach_manifest is not None:
        profile.attachmentSideChannel = bool(attach_manifest)
    if raw_manifest.get("attachmentField") or raw_manifest.get("attachment_field"):
        profile.attachmentField = str(
            raw_manifest.get("attachmentField") or raw_manifest.get("attachment_field")
        )
    side_keys = raw_manifest.get("sideChannelKeys") or raw_manifest.get("side_channel_keys")
    if isinstance(side_keys, list):
        profile.sideChannelKeys = [str(k) for k in side_keys]
    if raw_manifest.get("attachmentPolicy") or raw_manifest.get("attachment_policy"):
        profile.attachmentPolicy = str(
            raw_manifest.get("attachmentPolicy") or raw_manifest.get("attachment_policy")
        )
    if raw_manifest.get("attachmentInputMode"):
        profile.attachmentInputMode = str(raw_manifest["attachmentInputMode"])
    if raw_manifest.get("attachmentControlMode"):
        profile.attachmentControlMode = str(raw_manifest["attachmentControlMode"])
    if raw_manifest.get("includeAttachmentObs") is not None:
        profile.includeAttachmentObs = bool(raw_manifest["includeAttachmentObs"])

    if raw_manifest.get("controller_type") or raw_manifest.get("controllerType"):
        profile.controllerType = str(raw_manifest.get("controllerType") or raw_manifest.get("controller_type"))
    if raw_manifest.get("joint_action_available") is not None:
        profile.jointActionAvailable = bool(raw_manifest.get("joint_action_available"))
    elif raw_manifest.get("jointActionAvailable") is not None:
        profile.jointActionAvailable = bool(raw_manifest.get("jointActionAvailable"))
    if raw_manifest.get("preferredPolicySchemaId"):
        profile.preferredPolicySchemaId = str(raw_manifest["preferredPolicySchemaId"])
    if raw_manifest.get("evalExecutor"):
        profile.evalExecutor = str(raw_manifest["evalExecutor"])
    if raw_manifest.get("trainedActionMode"):
        profile.trainedActionMode = str(raw_manifest["trainedActionMode"])
    obs_schema = raw_manifest.get("observationSchema")
    if isinstance(obs_schema, str):
        profile.observationSchema = obs_schema
    act_schema = raw_manifest.get("actionSchema")
    if isinstance(act_schema, str):
        profile.actionSchema = act_schema
    ctrl_schema = raw_manifest.get("controllerSchema")
    if isinstance(ctrl_schema, str):
        profile.controllerSchema = ctrl_schema
    side_schema = raw_manifest.get("sideChannelSchema")
    if isinstance(side_schema, str):
        profile.sideChannelSchema = side_schema
    policy_schemas = raw_manifest.get("policySchemas")
    if isinstance(policy_schemas, dict):
        profile.policySchemas = dict(policy_schemas)
    action_keys = raw_manifest.get("availableActionKeys") or raw_manifest.get("available_action_keys")
    if isinstance(action_keys, list):
        profile.availableActionKeys = [str(k) for k in action_keys]

    obs_keys = list(manifest.observationSpace.keys)
    manifest_camera_keys = raw_manifest.get("cameraKeys") or raw_manifest.get("imageKeys") or []
    if isinstance(manifest_camera_keys, list) and manifest_camera_keys:
        camera_keys = [str(k) for k in manifest_camera_keys]
    else:
        camera_keys = [k for k in obs_keys if "image" in k.lower()]
    state_keys = [k for k in obs_keys if k not in camera_keys]

    if manifest.observationSpace.type and manifest.observationSpace.type != "low_dim":
        profile.observationType = manifest.observationSpace.type
    elif raw_manifest.get("observationType"):
        profile.observationType = str(raw_manifest["observationType"])
    elif obs_keys:
        profile.observationType = _infer_observation_type(obs_keys, camera_keys)
    else:
        profile.observationType = "unknown"

    if obs_keys:
        profile.observationKeys = obs_keys
        profile.cameraKeys = camera_keys
        profile.imageKeys = list(raw_manifest.get("imageKeys") or camera_keys)
        profile.stateDim = _estimate_state_dim(state_keys, manifest.observationSpace.dims)

    image_shape = raw_manifest.get("imageShape")
    if isinstance(image_shape, dict):
        profile.imageShape = {
            "height": int(image_shape.get("height") or 0),
            "width": int(image_shape.get("width") or 0),
            "channels": int(image_shape.get("channels") or 3),
        }

    if manifest.actionSpace.dim is not None:
        profile.actionDim = int(manifest.actionSpace.dim)
    if manifest.actionSpace.horizon is not None:
        profile.horizon = int(manifest.actionSpace.horizon)
    elif manifest.horizon is not None:
        profile.horizon = int(manifest.horizon)

    split = manifest.raw.get("split") if isinstance(manifest.raw.get("split"), dict) else {}
    if split.get("enabled"):
        profile.hasValidationSplit = True

    quality = manifest.raw.get("quality") if isinstance(manifest.raw.get("quality"), dict) else {}
    if quality.get("hasImage") and not profile.cameraKeys:
        profile.warnings.append("manifest.quality.hasImage=true 但未声明 camera keys")

    resolved_hdf5 = hdf5_path or manifest.storageUri
    if not resolved_hdf5:
        artifacts = raw_manifest.get("artifacts") if isinstance(raw_manifest.get("artifacts"), dict) else {}
        resolved_hdf5 = artifacts.get("hdf5") or raw_manifest.get("hdf5")
    if resolved_hdf5 and not profile.storageUri:
        profile.storageUri = str(resolved_hdf5)

    lerobot_block = raw_manifest.get("lerobot")
    if not isinstance(lerobot_block, dict):
        lerobot_block = raw_manifest.get("lerobotMetadata")
    primary_format = str(raw_manifest.get("primaryFormat") or raw_manifest.get("format") or "").lower()
    available_formats = [
        str(item).lower()
        for item in (raw_manifest.get("availableFormats") or raw_manifest.get("datasetFormats") or [])
    ]
    if isinstance(lerobot_block, dict) and ("lerobot" in available_formats or primary_format == "lerobot"):
        from app.services.pi0_lerobot_loader import resolve_lerobot_path_from_manifest

        lerobot_path = resolve_lerobot_path_from_manifest(raw_manifest)
        if lerobot_path is not None:
            profile.storageUri = str(lerobot_path)
            profile.format = "lerobot"
        if lerobot_block.get("stateDim") is not None:
            profile.stateDim = int(lerobot_block["stateDim"])
        if lerobot_block.get("actionDim") is not None:
            profile.actionDim = int(lerobot_block["actionDim"])
        profile.cameraKeys = ["agentview_image", "robot0_eye_in_hand_image"]
        profile.imageKeys = list(profile.cameraKeys)
        profile.observationKeys = profile.cameraKeys + ["robot0_joint_pos", "robot0_gripper_qpos"]
        profile.observationType = "mixed"
        if lerobot_block.get("robot"):
            profile.robotType = str(lerobot_block["robot"])
        if lerobot_block.get("taskInstruction"):
            profile.taskDescription = str(lerobot_block["taskInstruction"])

    hdf5_inspection: Optional[Hdf5InspectionResult] = None
    if resolved_hdf5 and str(profile.format).upper() == "HDF5":
        hdf5_inspection = inspect_hdf5(resolved_hdf5)
        if hdf5_inspection.source == "hdf5":
            profile.inferenceSources.append("hdf5")

            if hdf5_inspection.episode_count:
                if profile.episodeCount and profile.episodeCount != hdf5_inspection.episode_count:
                    profile.warnings.append("episodeCount 与 HDF5 demo 数量不一致，以 HDF5 为准")
                profile.episodeCount = hdf5_inspection.episode_count

            if hdf5_inspection.observation_keys:
                if profile.observationKeys and set(profile.observationKeys) != set(hdf5_inspection.observation_keys):
                    profile.warnings.append("manifest observationKeys 与 HDF5 不一致，以 HDF5 为准")
                profile.observationKeys = list(hdf5_inspection.observation_keys)
                profile.cameraKeys = list(hdf5_inspection.camera_keys)
                profile.imageKeys = list(hdf5_inspection.camera_keys)
                profile.observationType = _infer_observation_type(
                    profile.observationKeys,
                    profile.cameraKeys,
                )
                if hdf5_inspection.image_shape:
                    profile.imageShape = hdf5_inspection.image_shape

            if hdf5_inspection.action_dim is not None:
                if profile.actionDim and profile.actionDim != hdf5_inspection.action_dim:
                    profile.warnings.append("actionDim 与 HDF5 不一致，以 HDF5 为准")
                profile.actionDim = hdf5_inspection.action_dim

            if hdf5_inspection.state_dim is not None:
                if profile.stateDim and profile.stateDim != hdf5_inspection.state_dim:
                    profile.warnings.append("stateDim 与 HDF5 不一致，以 HDF5 为准")
                profile.stateDim = hdf5_inspection.state_dim

            if hdf5_inspection.horizon is not None:
                if profile.horizon and profile.horizon != hdf5_inspection.horizon:
                    profile.warnings.append("horizon 与 HDF5 不一致，以 HDF5 为准")
                profile.horizon = hdf5_inspection.horizon

            if hdf5_inspection.action_space and profile.actionSpace == "unknown":
                profile.actionSpace = hdf5_inspection.action_space

            profile.hasReward = profile.hasReward or hdf5_inspection.has_reward
            profile.hasDone = profile.hasDone or hdf5_inspection.has_done
            profile.hasSuccess = profile.hasSuccess or hdf5_inspection.has_success
            profile.hasValidationSplit = profile.hasValidationSplit or hdf5_inspection.has_validation_split

            if hdf5_inspection.attachment_side_channel:
                profile.attachmentSideChannel = True
            if hdf5_inspection.side_channel_keys:
                profile.sideChannelKeys = list(hdf5_inspection.side_channel_keys)
            if hdf5_inspection.attachment_field:
                profile.attachmentField = hdf5_inspection.attachment_field
            if hdf5_inspection.attachment_policy:
                profile.attachmentPolicy = hdf5_inspection.attachment_policy
            if hdf5_inspection.attachment_input_mode:
                profile.attachmentInputMode = hdf5_inspection.attachment_input_mode
            if hdf5_inspection.attachment_control_mode:
                profile.attachmentControlMode = hdf5_inspection.attachment_control_mode
            if hdf5_inspection.controller_type:
                profile.controllerType = hdf5_inspection.controller_type
            if hdf5_inspection.joint_action_available:
                profile.jointActionAvailable = True
            if hdf5_inspection.preferred_policy_schema_id:
                profile.preferredPolicySchemaId = hdf5_inspection.preferred_policy_schema_id
            if hdf5_inspection.trained_action_mode:
                profile.trainedActionMode = hdf5_inspection.trained_action_mode
            if hdf5_inspection.eval_executor:
                profile.evalExecutor = hdf5_inspection.eval_executor
            if hdf5_inspection.action_schema_id and not profile.actionSchema:
                profile.actionSchema = hdf5_inspection.action_schema_id
            if hdf5_inspection.available_action_keys:
                profile.availableActionKeys = list(hdf5_inspection.available_action_keys)

        profile.warnings.extend(hdf5_inspection.warnings)

    if profile.robotType == "unknown":
        profile.warnings.append("robotType 未能从 manifest 推断")

    if profile.observationType == "unknown":
        profile.warnings.append("observationType 未能推断，请检查 manifest 或 HDF5 obs 结构")

    if profile.actionDim == 0:
        profile.warnings.append("actionDim 未能推断")

    if profile.actionSpace == "unknown":
        if profile.actionDim == 7:
            profile.actionSpace = "delta_pose"
        elif profile.actionDim == 14:
            profile.actionSpace = "joint_position"
        else:
            profile.warnings.append("actionSpace 未能推断")

    if not profile.storageUri:
        profile.warnings.append("缺少 storageUri / artifacts.hdf5")

    return profile
