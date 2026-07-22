"""
dloBench DLO 工具包的公共 API 入口。

本模块是 robosuite.utils.dlo 包的 __init__.py，
负责从各子模块中导出公共 API，使外部代码可以简洁地导入：

  from robosuite.utils.dlo import polyline_length, validate_episode

模块组织：
  cable_state     — 环境状态适配器（鸭子类型分发）
  cable_metrics   — 线缆几何基元（弧长、直度比、覆盖率等）
  episode_schema  — Episode 数据模式定义与校验
  deformable_ravens_tasks  — Deformable Ravens 任务的目标生成与指标
  softgym_rope_tasks       — SoftGym 风格的绳索任务
  rmb_cable_task           — RMB 线缆穿越杆任务
  rmb_chain_task           — RMB 链条挂钩任务
  rmb_operation_presets    — RMB 机器人操作预设
  trajectory_quality       — 轨迹质量检查与报告
  task_logic               — L3 纯逻辑层（任务指标计算）
  controller_adapter       — L2 控制器适配层
  rollout                  — 随机动作 rollout
"""

from robosuite.utils.dlo.cable_state import (
    get_env_cable_keypoints,
    get_env_metrics,
    get_env_success,
)
from robosuite.utils.dlo.cable_metrics import (
    cable_centroid,
    endpoint_distance,
    gripper_to_cable_distance,
    keypoint_goal_coverage,
    line_deviation,
    nearest_cable_segment,
    passed_keypoint_ratio,
    polyline_length,
    straightness_ratio,
    validate_keypoints,
)
from robosuite.utils.dlo.episode_schema import (
    DEFAULT_ACTION_SCHEMA_VERSION,
    DEFAULT_OBSERVATION_SCHEMA_VERSION,
    EpisodeSchemaError,
    validate_episode,
)
from robosuite.utils.dlo.deformable_ravens_tasks import (
    best_ring_target_mapping,
    generate_polyline_target,
    generate_ring_target,
    polygon_area_xy,
    ring_area_metrics,
    target_keypoint_metrics,
)
from robosuite.utils.dlo.softgym_rope_tasks import (
    SOFTGYM_ROPE_CHARACTERS,
    best_bipartite_matching,
    generate_softgym_character_target,
    rope_flatten_metrics,
    softgym_configuration_metrics,
)
from robosuite.utils.dlo.rmb_cable_task import rmb_cable_pass_between_posts_metrics
from robosuite.utils.dlo.rmb_chain_task import rmb_chain_hang_on_hook_metrics
from robosuite.utils.dlo.rmb_operation_presets import (
    RMB_OPERATION_PRESETS,
    get_rmb_operation_preset,
    get_single_arm_pole_offset,
    require_implemented_rmb_preset,
)
from robosuite.utils.dlo.trajectory_quality import (
    DEFAULT_THRESHOLDS,
    quality_report_from_trajectories,
    report_from_file,
    trajectory_quality_report,
)
from robosuite.utils.dlo.hdf5_dataset import (
    HDF5_IMAGE_KEYS,
    HDF5_LEGACY_LOW_DIM_KEYS,
    HDF5_LOW_DIM_KEYS,
    HDF5_TASK_OBS_KEYS,
    POLICY_SCHEMAS,
    PREFERRED_POLICY_SCHEMAS,
    build_hdf5_manifest_fields,
    build_hdf5_manifest_obs_fields,
    derive_gripper_actions,
    derive_joint_delta_actions,
    save_dataset_hdf5,
)

__all__ = [
    # cable_state: 环境状态获取
    "get_env_cable_keypoints",
    "get_env_metrics",
    "get_env_success",
    # episode_schema: 数据模式
    "DEFAULT_ACTION_SCHEMA_VERSION",
    "DEFAULT_OBSERVATION_SCHEMA_VERSION",
    "EpisodeSchemaError",
    # cable_metrics: 几何基元
    "cable_centroid",
    "endpoint_distance",
    "gripper_to_cable_distance",
    "keypoint_goal_coverage",
    "line_deviation",
    "nearest_cable_segment",
    "passed_keypoint_ratio",
    "polyline_length",
    "straightness_ratio",
    "validate_episode",
    "validate_keypoints",
    # deformable_ravens_tasks: 目标生成与指标
    "best_ring_target_mapping",
    "generate_polyline_target",
    "generate_ring_target",
    "polygon_area_xy",
    "ring_area_metrics",
    "target_keypoint_metrics",
    # softgym_rope_tasks: SoftGym 绳索任务
    "SOFTGYM_ROPE_CHARACTERS",
    "best_bipartite_matching",
    "generate_softgym_character_target",
    "rope_flatten_metrics",
    "softgym_configuration_metrics",
    # rmb_*: RMB 任务指标与预设
    "RMB_OPERATION_PRESETS",
    "get_rmb_operation_preset",
    "get_single_arm_pole_offset",
    "require_implemented_rmb_preset",
    "rmb_cable_pass_between_posts_metrics",
    "rmb_chain_hang_on_hook_metrics",
    # trajectory_quality: 轨迹质量检查
    "DEFAULT_THRESHOLDS",
    "quality_report_from_trajectories",
    "report_from_file",
    "trajectory_quality_report",
    # hdf5_dataset: HDF5 数据集保存
    "HDF5_IMAGE_KEYS",
    "HDF5_LEGACY_LOW_DIM_KEYS",
    "HDF5_LOW_DIM_KEYS",
    "HDF5_TASK_OBS_KEYS",
    "POLICY_SCHEMAS",
    "PREFERRED_POLICY_SCHEMAS",
    "build_hdf5_manifest_fields",
    "build_hdf5_manifest_obs_fields",
    "derive_gripper_actions",
    "derive_joint_delta_actions",
    "save_dataset_hdf5",
]
