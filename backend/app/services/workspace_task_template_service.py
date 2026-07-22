from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

_CABLE_REGISTRY_ID = "task_cable_threading_v1"
_DUAL_ARM_REGISTRY_ID = "task_dual_arm_cable_manipulation_v1"
_ISAAC_BLOCK_STACKING_REGISTRY_ID = "task_isaac_block_stacking_v1"
_ISAACLAB_FRANKA_STACK_CUBE_REGISTRY_ID = "task_isaaclab_franka_stack_cube_v1"
_ISAACSIM_FRANKA_PICK_PLACE_REGISTRY_ID = "task_isaacsim_franka_pick_place_v1"

DEFAULT_TASK_TEMPLATES: list[dict[str, Any]] = [
    {
        "id": "cable_threading_single_arm",
        "name": "线缆穿杆",
        "description": "机械臂完成线缆穿过目标杆的仿真操作任务。",
        "sourceType": "standard_template",
        "taskFamily": "cable_manipulation",
        "taskType": "cable_threading",
        "simulatorType": "mujoco",
        "supportedRobotTypes": ["Panda", "UR5e"],
        "supportedPolicyTypes": ["scripted", "robomimic_bc"],
        "supportedEvaluationModes": ["expert_policy_evaluation", "trained_model_evaluation"],
        "defaultSceneId": "scene_cable_threading_mujoco_v1",
        "defaultMetricProfileId": "metric_cable_success_rate_v1",
        "defaultMetricIds": [
            "metric_cable_success_rate_v1",
        ],
        "registryTaskConfigId": _CABLE_REGISTRY_ID,
        "status": "available",
        "simulatorBackend": "mujoco",
        "supportsDatasetGeneration": True,
        "replayAvailable": True,
        "supportsImportedDemoReplay": False,
    },
    {
        "id": "dual_arm_cable_manipulation",
        "name": "线缆整理",
        "description": "双臂协同完成线缆整理、拖拽与形态控制的仿真操作任务。",
        "sourceType": "standard_template",
        "taskFamily": "cable_manipulation",
        "taskType": "dual_arm_cable_manipulation",
        "simulatorType": "mujoco",
        "supportedRobotTypes": ["dual_fr3"],
        "supportedPolicyTypes": ["scripted", "episode_stability", "torch_bc"],
        "supportedEvaluationModes": ["episode_stability", "trained_model_evaluation"],
        "defaultSceneId": "scene_dual_arm_cable_mujoco_v1",
        "defaultMetricProfileId": "metric_episode_stability_v1",
        "defaultMetricIds": [
            "metric_episode_stability_v1",
        ],
        "registryTaskConfigId": _DUAL_ARM_REGISTRY_ID,
        "status": "available",
        "simulatorBackend": "mujoco",
        "supportsDatasetGeneration": True,
        "replayAvailable": True,
        "supportsImportedDemoReplay": False,
    },
    {
        "id": "isaac_block_stacking",
        "name": "物块堆叠",
        "description": "内部评测与 Robomimic BC 训练适配入口（Isaac Lab rollout 评测）。",
        "sourceType": "standard_template",
        "taskFamily": "manipulation_core",
        "taskType": "block_stacking",
        "simulatorType": "isaac",
        "simulatorBackendLabel": "Isaac Lab",
        "physicsBackend": "physx",
        "defaultEnv": "Isaac-Stack-Cube-Franka-IK-Rel-v0",
        "adapterStatus": "adaptable",
        "requiresExternalRuntime": True,
        "supportedRobotTypes": ["franka_panda"],
        "supportedPolicyTypes": ["robomimic_bc", "teleop"],
        "supportedEvaluationModes": ["trained_model_evaluation"],
        "defaultSceneId": None,
        "defaultMetricProfileId": None,
        "defaultMetricIds": [
            "isaac_stack_success_rate_v1",
            "isaac_stack_mean_reward_v1",
            "isaac_stack_mean_episode_length_v1",
            "isaac_stack_failure_count_v1",
            "isaac_stack_timeout_rate_v1",
        ],
        "registryTaskConfigId": _ISAAC_BLOCK_STACKING_REGISTRY_ID,
        "status": "available",
        "simulatorBackend": "isaac_lab",
        "supportsDatasetGeneration": False,
        "replayAvailable": True,
        "supportsImportedDemoReplay": True,
        "evaluationStatus": "adapter_available",
        "productRole": "evaluation_training_adapter",
        "hiddenFromProductCatalog": True,
    },
    {
        "id": "isaaclab_franka_stack_cube",
        "name": "物块堆叠",
        "description": "基于 Isaac Lab 的 Franka 方块堆叠任务，支持 Mimic 专家数据生成、Robomimic BC 训练与 Isaac Lab rollout 评测。",
        "sourceType": "standard_template",
        "taskFamily": "manipulation_core",
        "taskType": "stacking",
        "simulatorType": "isaac",
        "simulatorBackendLabel": "Isaac Lab",
        "physicsBackend": "physx",
        "defaultEnv": "Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0",
        "expertSource": "Isaac Lab Mimic seed demonstration",
        "adapterStatus": "available",
        "requiresExternalRuntime": True,
        "supportedRobotTypes": ["franka_panda"],
        "supportedPolicyTypes": ["mimic_auto", "teleop_record"],
        "supportedEvaluationModes": ["trained_model_evaluation"],
        "defaultSceneId": None,
        "defaultMetricProfileId": None,
        "defaultMetricIds": [
            "isaac_stack_success_rate_v1",
            "isaac_stack_mean_reward_v1",
            "isaac_stack_mean_episode_length_v1",
            "isaac_stack_failure_count_v1",
            "isaac_stack_timeout_rate_v1",
        ],
        "registryTaskConfigId": _ISAACLAB_FRANKA_STACK_CUBE_REGISTRY_ID,
        "status": "available",
        "simulatorBackend": "isaac_lab",
        "supportsDatasetGeneration": True,
        "replayAvailable": True,
        "supportsImportedDemoReplay": False,
        "datasetFormats": ["hdf5", "zarr"],
        "entrypoint": "integrations/IsaacLabBlockStacking/run/platform_run.py",
        "taskPackagePath": "integrations/IsaacLabBlockStacking",
        "evaluationStatus": "routes_via_isaac_block_stacking",
        "evaluationAdapterTemplateId": "isaac_block_stacking",
        "productSubtitle": "基于 Isaac Lab 的 Franka 方块堆叠任务",
        "demoPriority": 1,
    },
    {
        "id": "isaacsim_franka_pick_place",
        "name": "Franka 物体搬运",
        "description": "Franka 机械臂在 Isaac Sim 中完成官方 pick-and-place 物体搬运任务。",
        "sourceType": "standard_template",
        "taskFamily": "manipulation_core",
        "taskType": "pick_and_place",
        "simulatorType": "isaac",
        "simulatorBackendLabel": "Isaac Sim",
        "physicsBackend": "physx",
        "expertSource": "NVIDIA Isaac Sim 官方 FrankaPickPlace controller",
        "adapterStatus": "available",
        "requiresExternalRuntime": True,
        "supportedRobotTypes": ["franka_panda"],
        "supportedPolicyTypes": [],
        "supportedEvaluationModes": [],
        "defaultSceneId": "scene_isaacsim_franka_pick_place_v1",
        "defaultMetricProfileId": "metric_pick_place_success_rate_v1",
        "defaultMetricIds": [
            "metric_pick_place_success_rate_v1",
        ],
        "registryTaskConfigId": _ISAACSIM_FRANKA_PICK_PLACE_REGISTRY_ID,
        "status": "integration_pending",
        "integrationPendingReason": (
            "Isaac Sim 5.1 adapter API incompatible; replaced by Isaac Lab native task for demo path."
        ),
        "simulatorBackend": "isaacsim",
        "supportsDatasetGeneration": False,
        "replayAvailable": False,
        "demoPriority": 99,
        "supportsImportedDemoReplay": False,
        "datasetFormats": ["json", "npz", "mp4"],
        "entrypoint": (
            "integrations/IsaacSimFrankaPickPlace/"
            "expert/official_franka_pick_place_adapter.py"
        ),
    },
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def list_task_templates_fallback() -> list[dict[str, Any]]:
    """无 DB 表时的静态回退（仅测试 / 未迁移环境）。"""
    from app.services import resource_registry_service as registry

    now = _utc_now_iso()
    rows: list[dict[str, Any]] = []
    for item in DEFAULT_TASK_TEMPLATES:
        row = dict(item)
        registry_id = row.get("registryTaskConfigId")
        if registry_id:
            try:
                registry.ensure_registry_loaded()
                resource = registry.get_resource(registry_id)
                if resource:
                    if resource.get("name"):
                        row.setdefault("name", resource["name"])
                    if resource.get("description"):
                        row["description"] = resource["description"]
                    if resource.get("status"):
                        row["status"] = resource["status"]
                    if resource.get("lastModifiedAt"):
                        row["updatedAt"] = resource["lastModifiedAt"]
                    runner = resource.get("runner") or {}
                    if isinstance(runner, dict):
                        row["runner"] = runner
            except Exception as exc:
                logger.debug("fallback template merge skipped: %s", exc)
        row.setdefault("createdAt", now)
        row.setdefault("updatedAt", row.get("updatedAt") or now)
        rows.append(row)
    return rows


def list_task_templates() -> list[dict[str, Any]]:
    from app.services import task_template_catalog_service as catalog_svc

    return catalog_svc.list_task_templates()


_EXPERT_POLICY_TYPES = frozenset({"scripted", "scripted_expert", "expert", "expert_policy"})


def enrich_task_template_derived_fields(item: dict[str, Any]) -> dict[str, Any]:
    """Derive product capability flags for UI gating (data generation vs evaluation)."""
    supported_policies = item.get("supportedPolicyTypes") or []
    eval_modes = item.get("supportedEvaluationModes") or []
    supports_gen = item.get("supportsDatasetGeneration") is True
    status = str(item.get("status") or "available")

    has_expert_policy = supports_gen and any(
        policy in _EXPERT_POLICY_TYPES for policy in supported_policies
    )
    has_evaluation_runner = bool(eval_modes) and status == "available"
    default_replay_camera = (
        item.get("defaultReplayCamera")
        or item.get("evalDisplayCamera")
        or item.get("recordCamera")
    )
    if not default_replay_camera:
        runner = item.get("runner")
        if isinstance(runner, dict):
            default_replay_camera = runner.get("defaultReplayCamera") or runner.get("recordCamera")

    enriched = {
        **item,
        "hasExpertPolicy": has_expert_policy,
        "hasEvaluationRunner": has_evaluation_runner,
        "supportsDataGeneration": supports_gen,
        "supportsEvaluation": has_evaluation_runner,
    }
    if default_replay_camera:
        enriched["defaultReplayCamera"] = str(default_replay_camera)
    return enriched
