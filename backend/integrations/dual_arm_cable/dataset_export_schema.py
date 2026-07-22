from __future__ import annotations

from typing import Any

TASK_TEMPLATE_ID = "dual_arm_cable_manipulation"
TASK_TYPE = "dual_arm_cable_manipulation"
SIMULATOR_BACKEND = "mujoco"
DATASET_FORMAT = "hdf5"
OBSERVATION_SCHEMA = "dual_arm_cable_il_v1"
ACTION_SCHEMA = "dual_arm_bimanual_action_v1"
DEFAULT_TRAINING_BACKENDS = ["torch_bc"]
DEFAULT_CONTROL_FREQUENCY = 20

# Dual Franka FR3 arms × 7 DoF each (manifest may override after inspection).
DEFAULT_ACTION_DIM = 14

DEFAULT_OBS_KEYS = [
    "left_arm_joint_pos",
    "right_arm_joint_pos",
    "left_arm_joint_vel",
    "right_arm_joint_vel",
    "cable_state",
    "overhead_rgb",
]


def build_dataset_manifest(
    *,
    source_job_id: str,
    control_frequency: int,
    num_episodes: int,
    successful_episodes: int,
    obs_keys: list[str],
    action_dim: int,
    action_semantics: str,
    artifacts: dict[str, str],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "taskTemplateId": TASK_TEMPLATE_ID,
        "taskType": TASK_TYPE,
        "simulatorBackend": SIMULATOR_BACKEND,
        "datasetFormat": DATASET_FORMAT,
        "trainable": True,
        "trainingBackends": list(DEFAULT_TRAINING_BACKENDS),
        "observationSchema": OBSERVATION_SCHEMA,
        "actionSchema": ACTION_SCHEMA,
        "actionSemantics": action_semantics,
        "controlFrequency": control_frequency,
        "numEpisodes": num_episodes,
        "successfulEpisodes": successful_episodes,
        "obsKeys": obs_keys,
        "actionDim": action_dim,
        "sourceJobId": source_job_id,
        "artifacts": artifacts,
    }
    if extra:
        payload.update(extra)
    return payload
