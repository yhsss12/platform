"""
RMB（RoboManipBaselines）机器人操作预设配置。

本模块存储从 RoboManipBaselines 项目中提取的各机器人型号的
穿线任务参数预设，包括：
  - 各机器人（UR5e、Panda、xArm7 等）的到达高度和持续时间
  - 单臂杆偏移量配置
  - 预设查询和校验函数

用途：
  当使用 RMB 风格的专家策略时，根据机器人型号自动选择合适的参数，
  避免手动调参。每个预设记录了原始 RMB 源码路径，便于追溯。

预设状态：
  - implemented: True 表示已在 robosuite 中实现
  - implemented: False 表示已记录参数但尚未接入 robosuite
"""

# ---------------------------------------------------------------------------
# 机器人预设字典
# ---------------------------------------------------------------------------
# 每个预设包含：
#   source_operation: RMB 源码中对应的操作脚本路径
#   source_env:       RMB 源码中对应的环境脚本路径
#   recommended_robot: 推荐的机器人型号
#   implemented:      是否已在 robosuite 中实现
#   reach_heights:    (到达高度1, 到达高度2) — 用于穿线任务的两阶段高度
#   reach_durations:  (持续时间1, 持续时间2) — 两阶段的时间分配
#   task:             任务描述
#   notes:            备注
RMB_OPERATION_PRESETS = {
    "ur5e": {
        "source_operation": "RoboManipBaselines/robo_manip_baselines/envs/operation/OperationMujocoUR5eCable.py",
        "source_env": "RoboManipBaselines/robo_manip_baselines/envs/mujoco/ur5e/MujocoUR5eCableEnv.py",
        "recommended_robot": "UR5e",
        "implemented": True,
        "reach_heights": (1.02, 0.995),
        "reach_durations": (0.7, 0.3),
        "task": "pass cable between two poles",
        "notes": "首版 robosuite 可运行 preset。",
    },
    "panda": {
        "source_operation": "RoboManipBaselines/robo_manip_baselines/envs/operation/OperationMujocoPandaCable.py",
        "source_env": "RoboManipBaselines/robo_manip_baselines/envs/mujoco/panda/MujocoPandaCableEnv.py",
        "recommended_robot": "Panda",
        "implemented": True,
        "reach_heights": (1.15, 1.058),
        "reach_durations": (1.0, 0.5),
        "task": "pass cable between two poles",
        "notes": "Panda preset 已接入 robosuite，通过 --robot Panda --rmb-robot-preset panda 使用。",
    },
    "xarm7": {
        "source_operation": "RoboManipBaselines/robo_manip_baselines/envs/operation/OperationMujocoXarm7Cable.py",
        "source_env": "RoboManipBaselines/robo_manip_baselines/envs/mujoco/xarm7/MujocoXarm7CableEnv.py",
        "recommended_robot": "xArm7",
        "implemented": False,
        "reach_heights": (1.0, 0.925),
        "reach_durations": (0.7, 0.3),
        "task": "pass cable between two poles",
        "notes": "等待 robosuite robot adapter。",
    },
    "kinovagen3": {
        "source_operation": "RoboManipBaselines/robo_manip_baselines/envs/operation/OperationMujocoKinovaGen3Cable.py",
        "source_env": "RoboManipBaselines/robo_manip_baselines/envs/mujoco/kinovagen3/MujocoKinovaGen3CableEnv.py",
        "recommended_robot": "KinovaGen3",
        "implemented": False,
        "reach_heights": (1.08, 1.04),
        "reach_durations": (0.7, 0.3),
        "task": "pass cable between two poles",
        "notes": "等待 robosuite robot adapter。",
    },
    "crx5ia": {
        "source_operation": "RoboManipBaselines/robo_manip_baselines/envs/operation/OperationMujocoCrx5iaCable.py",
        "source_env": "RoboManipBaselines/robo_manip_baselines/envs/mujoco/crx5ia/MujocoCrx5iaCableEnv.py",
        "recommended_robot": "CRX5iA",
        "implemented": False,
        "reach_heights": (1.2, 1.13),
        "reach_durations": (0.7, 0.3),
        "task": "pass cable between two poles",
        "notes": "等待 robosuite robot adapter。",
    },
    "ur5e_dual": {
        "source_operation": "RoboManipBaselines/robo_manip_baselines/envs/operation/OperationMujocoUR5eDualCable.py",
        "source_env": "RoboManipBaselines/robo_manip_baselines/envs/mujoco/ur5e_dual/MujocoUR5eDualCableEnv.py",
        "recommended_robot": "UR5eDual",
        "implemented": False,
        "reach_heights": (1.02, 0.99),
        "reach_durations": (0.7, 0.3),
        "task": "dual-arm cable manipulation between poles",
        "notes": "双臂 action / teleop / schema 需单独接入。",
    },
    "aloha": {
        "source_operation": "RoboManipBaselines/robo_manip_baselines/envs/operation/OperationMujocoAlohaCable.py",
        "source_env": "RoboManipBaselines/robo_manip_baselines/envs/mujoco/aloha/MujocoAlohaCableEnv.py",
        "recommended_robot": "Aloha",
        "implemented": False,
        "reach_heights": (0.3, 0.2),
        "reach_durations": (0.7, 0.3),
        "task": "dual-arm cable manipulation between poles",
        "notes": "双臂 ALOHA asset / action / teleop 后续接入。",
    },
}

# ---------------------------------------------------------------------------
# 单臂杆偏移量
# ---------------------------------------------------------------------------
# RMB 的多世界评估中，每个世界用不同的杆间距。
# 偏移量是相对于默认杆位置的 (dx, dy, dz) 调整。
# 只有 x 方向有变化（改变两杆之间的水平距离），y 和 z 均为 0。
# 共 6 种间距配置，从 -0.03m 到 +0.12m。
RMB_POLE_OFFSETS_SINGLE_ARM = (
    (-0.03, 0.0, 0.0),
    (0.0, 0.0, 0.0),
    (0.03, 0.0, 0.0),
    (0.06, 0.0, 0.0),
    (0.09, 0.0, 0.0),
    (0.12, 0.0, 0.0),
)


def get_rmb_operation_preset(name):
    """根据名称获取 RMB 操作预设。

    名称不区分大小写。如果名称不存在，抛出 ValueError 并列出
    所有支持的预设名称。

    Args:
        name: 预设名称（如 "ur5e"、"Panda"）。

    Returns:
        预设字典。
    """
    key = str(name).lower()
    if key not in RMB_OPERATION_PRESETS:
        supported = ", ".join(sorted(RMB_OPERATION_PRESETS))
        raise ValueError(f"Unsupported RMB cable preset: {name}. Supported presets: {supported}")
    return RMB_OPERATION_PRESETS[key]


def require_implemented_rmb_preset(name):
    """获取预设并检查是否已在 robosuite 中实现。

    如果预设未实现（implemented=False），抛出 NotImplementedError，
    并附带推荐的机器人型号和原始源码路径。

    Args:
        name: 预设名称。

    Returns:
        已实现的预设字典。
    """
    preset = get_rmb_operation_preset(name)
    if not preset["implemented"]:
        raise NotImplementedError(
            f"rmb_robot_preset='{name}' is documented but not implemented in robosuite yet. "
            f"Recommended robot: {preset['recommended_robot']}. Source: {preset['source_operation']}"
        )
    return preset


def get_single_arm_pole_offset(world_idx):
    """获取单臂任务的杆偏移量。

    Args:
        world_idx: 世界索引（0~5），对应 6 种杆间距配置。

    Returns:
        (dx, dy, dz) 偏移量元组。

    Raises:
        ValueError: 索引超出范围时。
    """
    idx = int(world_idx)
    if idx < 0 or idx >= len(RMB_POLE_OFFSETS_SINGLE_ARM):
        raise ValueError(f"rmb_world_idx must be in [0, {len(RMB_POLE_OFFSETS_SINGLE_ARM) - 1}], got {world_idx}")
    return RMB_POLE_OFFSETS_SINGLE_ARM[idx]
