from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ModelCapability:
    """单个训练/评测模型的能力声明。"""

    modelType: str
    displayName: str
    backendKey: str
    requiredSimulators: tuple[str, ...] = ()
    requiredRobotTypes: tuple[str, ...] = ()
    requiredDataFormats: tuple[str, ...] = ("HDF5",)
    requiredObservationTypes: tuple[str, ...] = ("low_dim",)
    requiredObservationKeys: tuple[str, ...] = ()
    prefersSequenceActions: bool = False
    minSuccessEpisodes: int = 1
    minEpisodeCount: int = 1
    priority: int = 100
    status: str = "available"
    downstreamModelType: str = ""
    taskTemplateId: Optional[str] = None
    evaluationPolicyType: str = ""
    defaultMetrics: tuple[str, ...] = ()
    notes: str = ""


def _norm_simulators(*values: str) -> tuple[str, ...]:
    return tuple(v.lower() for v in values)


MODEL_CAPABILITY_REGISTRY: dict[str, ModelCapability] = {
    "robomimic_bc": ModelCapability(
        modelType="robomimic_bc",
        displayName="Robomimic BC",
        backendKey="robomimic_bc",
        downstreamModelType="Robomimic",
        requiredSimulators=_norm_simulators("mujoco"),
        requiredRobotTypes=("Panda", "UR5e", "panda", "ur5e"),
        requiredDataFormats=("HDF5",),
        requiredObservationTypes=("low_dim", "mixed"),
        minSuccessEpisodes=1,
        priority=10,
        taskTemplateId="cable_threading_single_arm",
        evaluationPolicyType="robomimic_bc",
        defaultMetrics=("metric_cable_success_rate_v1",),
        notes="适用于 MuJoCo 单臂低维观测 HDF5 数据集。",
    ),
    "diffusion_policy": ModelCapability(
        modelType="diffusion_policy",
        displayName="Diffusion Policy",
        backendKey="diffusion_policy",
        downstreamModelType="Diffusion Policy",
        requiredSimulators=(),
        requiredRobotTypes=(),
        requiredDataFormats=("HDF5",),
        requiredObservationTypes=("low_dim", "image", "mixed", "unknown"),
        requiredObservationKeys=(),
        prefersSequenceActions=True,
        minSuccessEpisodes=1,
        priority=20,
        taskTemplateId="cable_threading_single_arm",
        evaluationPolicyType="robomimic_bc",
        defaultMetrics=("metric_cable_success_rate_v1",),
        notes="需要双相机图像观测与序列动作支持。",
    ),
    "act": ModelCapability(
        modelType="act",
        displayName="ACT",
        backendKey="act",
        downstreamModelType="ACT",
        requiredSimulators=(),
        requiredRobotTypes=(),
        requiredDataFormats=("HDF5",),
        requiredObservationTypes=("image", "mixed"),
        requiredObservationKeys=(),
        minSuccessEpisodes=1,
        priority=30,
        status="available",
        taskTemplateId="cable_threading_single_arm",
        evaluationPolicyType="act",
        defaultMetrics=("metric_cable_success_rate_v1",),
        notes="需要图像 + proprio 观测的 HDF5 数据集；low_dim-only 数据集不可训练 ACT。",
    ),
    "pi0": ModelCapability(
        modelType="pi0",
        displayName="pi0",
        backendKey="pi0",
        downstreamModelType="pi0",
        requiredSimulators=(),
        requiredRobotTypes=(),
        requiredDataFormats=("HDF5", "lerobot_index", "LeRobot"),
        requiredObservationTypes=("image", "mixed"),
        minSuccessEpisodes=1,
        priority=40,
        status="available",
        taskTemplateId="cable_threading_single_arm",
        evaluationPolicyType="pi0",
        defaultMetrics=("metric_cable_success_rate_v1",),
        notes="需要 openpi 环境与图像观测；平台可将 HDF5 转为 LeRobot index。",
    ),
    "isaac_robomimic_bc": ModelCapability(
        modelType="isaac_robomimic_bc",
        displayName="Isaac Robomimic BC",
        backendKey="isaac_robomimic_bc",
        downstreamModelType="Robomimic",
        requiredSimulators=_norm_simulators("isaac", "isaac_lab", "isaacsim", "physx"),
        requiredRobotTypes=("franka_panda", "Franka", "franka"),
        requiredDataFormats=("HDF5",),
        requiredObservationTypes=("low_dim", "mixed", "image"),
        minSuccessEpisodes=1,
        priority=15,
        taskTemplateId="isaac_block_stacking",
        evaluationPolicyType="robomimic_bc",
        defaultMetrics=("metric_isaac_success_rate_v1",),
        notes="适用于 Isaac Lab 物块堆叠任务。",
    ),
    "torch_bc": ModelCapability(
        modelType="torch_bc",
        displayName="BC (PyTorch)",
        backendKey="torch_bc",
        downstreamModelType="Robomimic",
        requiredSimulators=_norm_simulators("mujoco"),
        requiredRobotTypes=("dual_fr3", "dual_arm"),
        requiredDataFormats=("HDF5",),
        requiredObservationTypes=("low_dim", "mixed"),
        minSuccessEpisodes=1,
        priority=25,
        taskTemplateId="dual_arm_cable_manipulation",
        evaluationPolicyType="torch_bc",
        defaultMetrics=("metric_success_rate_v1", "metric_episode_stability_v1"),
        notes="适用于双臂线缆整理数据集。",
    ),
}


def list_model_capabilities() -> list[ModelCapability]:
    return sorted(MODEL_CAPABILITY_REGISTRY.values(), key=lambda item: item.priority)


def get_model_capability(model_type: str) -> Optional[ModelCapability]:
    key = (model_type or "").strip().lower()
    aliases = {
        "robomimic": "robomimic_bc",
        "diffusion policy": "diffusion_policy",
        "bc": "robomimic_bc",
    }
    key = aliases.get(key, key)
    return MODEL_CAPABILITY_REGISTRY.get(key)
