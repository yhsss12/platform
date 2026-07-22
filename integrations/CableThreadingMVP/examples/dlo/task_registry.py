"""
task_registry.py -- 任务注册表

本文件定义了 dloBench 的全部 10 个任务的元数据（TaskSpec），以及查询路由逻辑。

核心设计：
- TaskSpec 数据类：描述每个任务的来源、默认机器人、支持的环境选项、遥操作/专家入口等
- TASK_SPECS 字典：10 个任务的注册表，是整个系统的"任务目录"
- 路由逻辑：expert_entry 和 teleop_entry 字段决定每个任务由哪个脚本处理

任务来源（source）：
- robosuite_change：原始 robosuite 扩展任务（CableStraighten、CableMoveToTarget 等）
- deformable-ravens：从 deformable-ravens 迁移的任务（CableShape、CableRing 等）
- SoftGym：从 SoftGym 迁移的任务（RopeFlatten、RopeConfiguration）
- RoboManipBaselines：从 RMB 迁移的任务（RMBChainHangOnHook）

任务族（task_family）：
- endpoint_manipulation：端点操作（拾取-拖拽-释放循环）
- shape_matching：形状匹配（线缆变形到目标形状）
- ring_matching：环形匹配（拓扑感知的环形目标）
- threading_between_posts：穿线任务（在柱子间穿过线缆）
- chain_hang_on_hook：挂钩任务
- rope_flatten / rope_configuration：绳子展平/配置
- cable_routing：线缆路径规划
"""

from dataclasses import dataclass, field


# --- 环境选项常量：这些选项通过 env_kwargs 传递给 suite.make() ---
ENV_OPTION_CABLE_MODEL = "cable_model"
ENV_OPTION_GRASP_MODE = "grasp_mode"        # 夹爪模式：attachment（mocap 焊接）或 physical（纯物理接触）
ENV_OPTION_GOAL_FILE = "goal_file"          # 目标形状文件路径
ENV_OPTION_GOAL_CHARACTER = "goal_character" # SoftGym 目标字符（用于 RopeConfiguration）
SOURCE_OPTION_RMB_ROBOT_PRESET = "rmb_robot_preset"          # RMB 机器人预设（如 "ur5e"）
SOURCE_OPTION_RMB_WORLD_IDX = "rmb_world_idx"                # RMB 场景索引
SOURCE_OPTION_RMB_WORLD_RANDOM_SCALE = "rmb_world_random_scale"  # RMB 场景随机缩放

# --- 入口类型常量：决定每个任务由哪个收集脚本处理 ---
TELEOP_DLO_HUMAN = "dlo_human"              # DLO 通用遥操作入口（collect_human.py）
TELEOP_THREADING_HUMAN = "threading_human"   # 穿线任务遥操作入口（cable_threading_collect_human.py）
EXPERT_DLO_ENDPOINT = "dlo_endpoint"         # DLO 通用专家入口（collect_expert_data.py / rollout_composer.py）
EXPERT_THREADING_ENDPOINT = "threading_endpoint"  # 穿线任务专家入口（cable_threading_collect_demos.py）

ALL_CABLE_MODELS = (
    "rmb",
    "segmented",
    "composite_cable",
    "composite_improve",
    "composite_soft",
    "composite_thin",
    "flex",
    "flex_improve",
)


@dataclass(frozen=True)
class TaskSpec:
    """单个任务的元数据规范。

    每个任务在 TASK_SPECS 中有一个 TaskSpec 条目，描述：
    - 元数据：名称、来源、任务族、默认机器人/相机/控制器/线缆模型
    - 选项支持：该任务支持哪些环境选项（cable_model、grasp_mode 等）
    - 路由：teleop_entry 和 expert_entry 决定由哪个脚本处理该任务
    - 录制：是否支持 recording_schema（遥操作数据录制的 schema 配置）
    """
    name: str                        # 任务名称，与 robosuite 环境类名一致
    source: str                      # 任务来源（robosuite_change / deformable-ravens / SoftGym / RoboManipBaselines）
    task_family: str                 # 任务族（endpoint_manipulation / shape_matching / ring_matching 等）
    default_robot: str               # 默认机器人型号（Panda / UR5e）
    default_camera: str = "mainview_ref"       # 默认相机视角
    default_controller: str | None = None      # 默认控制器配置（None 表示使用 BASIC）
    default_cable_model: str = "composite_cable"  # 默认线缆模型后端
    supported_cable_models: tuple[str, ...] = ALL_CABLE_MODELS
    env_options: tuple[str, ...] = (ENV_OPTION_CABLE_MODEL,)   # 该任务支持的环境选项列表
    source_options: tuple[str, ...] = ()       # 该任务的来源特定选项（如 RMB 的 robot_preset）
    source_defaults: dict[str, object] = field(default_factory=dict)  # 来源选项的默认值
    teleop_entry: str | None = None     # 遥操作入口类型（None 表示不支持遥操作）
    expert_entry: str | None = None     # 专家脚本入口类型（None 表示不支持专家脚本）
    supports_recording_schema: bool = False     # 是否支持录制 schema 配置
    default_recording_schema: str = ""          # 默认录制 schema 路径
    default_obs_keys: tuple[str, ...] = ()      # 默认观测键列表

    def supports_option(self, option: str) -> bool:
        """检查该任务是否支持指定的环境选项或来源选项。"""
        return option in self.env_options or option in self.source_options

    def default_for(self, option: str, default=None):
        """获取指定选项的默认值。cable_model 有专用字段，其余从 source_defaults 查找。"""
        if option == ENV_OPTION_CABLE_MODEL:
            return self.default_cable_model
        return self.source_defaults.get(option, default)


# --- 10 个任务的注册表 ---
# 每个 TaskSpec 定义了任务的完整元数据
TASK_SPECS = {
    # --- 端点操作任务（endpoint_manipulation）---
    "CableStraighten": TaskSpec(
        name="CableStraighten",
        source="robosuite_change",
        task_family="endpoint_manipulation",
        default_robot="Panda",
        env_options=(ENV_OPTION_CABLE_MODEL, ENV_OPTION_GRASP_MODE),
        teleop_entry=TELEOP_DLO_HUMAN,
        expert_entry=EXPERT_DLO_ENDPOINT,
        supports_recording_schema=True,
    ),
    "CableMoveToTarget": TaskSpec(
        name="CableMoveToTarget",
        source="robosuite_change",
        task_family="endpoint_manipulation",
        default_robot="Panda",
        env_options=(ENV_OPTION_CABLE_MODEL, ENV_OPTION_GRASP_MODE),
        teleop_entry=TELEOP_DLO_HUMAN,
        expert_entry=EXPERT_DLO_ENDPOINT,
        supports_recording_schema=True,
    ),
    # --- 穿线任务（threading_between_posts）---
    # CableThreading 是穿线任务的统一实现，支持 RMB 兼容模式（通过 rmb_robot_preset 参数）
    "CableThreading": TaskSpec(
        name="CableThreading",
        source="robosuite_change.cable_threading",
        task_family="threading_between_posts",
        default_robot="Panda",
        supported_cable_models=ALL_CABLE_MODELS,
        env_options=(ENV_OPTION_CABLE_MODEL, ENV_OPTION_GRASP_MODE),
        source_options=(SOURCE_OPTION_RMB_ROBOT_PRESET, SOURCE_OPTION_RMB_WORLD_IDX, SOURCE_OPTION_RMB_WORLD_RANDOM_SCALE),
        source_defaults={SOURCE_OPTION_RMB_ROBOT_PRESET: "ur5e"},
        teleop_entry=TELEOP_THREADING_HUMAN,
        expert_entry=EXPERT_THREADING_ENDPOINT,
    ),
    # --- 挂钩任务（chain_hang_on_hook）---
    "RMBChainHangOnHook": TaskSpec(
        name="RMBChainHangOnHook",
        source="RoboManipBaselines",
        task_family="chain_hang_on_hook",
        default_robot="Panda",
        env_options=(ENV_OPTION_CABLE_MODEL,),
        source_options=(SOURCE_OPTION_RMB_ROBOT_PRESET,),
        source_defaults={SOURCE_OPTION_RMB_ROBOT_PRESET: "ur5e"},
        teleop_entry=TELEOP_DLO_HUMAN,
        expert_entry=EXPERT_DLO_ENDPOINT,
        supports_recording_schema=True,
    ),
    # --- 形状匹配任务（shape_matching）---
    # CableShape 统一处理目标可见/隐藏，通过 target_visible 参数控制
    "CableShape": TaskSpec(
        name="CableShape",
        source="deformable-ravens",
        task_family="shape_matching",
        default_robot="Panda",
        default_cable_model="composite_cable",
        env_options=(ENV_OPTION_CABLE_MODEL, ENV_OPTION_GOAL_FILE),
        teleop_entry=TELEOP_DLO_HUMAN,
        expert_entry=EXPERT_DLO_ENDPOINT,
        supports_recording_schema=True,
    ),
    # --- 环形匹配任务（ring_matching）---
    # CableRing 统一处理目标可见/隐藏，通过 target_visible 参数控制
    "CableRing": TaskSpec(
        name="CableRing",
        source="deformable-ravens",
        task_family="ring_matching",
        default_robot="Panda",
        default_cable_model="composite_cable",
        env_options=(ENV_OPTION_CABLE_MODEL, ENV_OPTION_GOAL_FILE),
        teleop_entry=TELEOP_DLO_HUMAN,
        expert_entry=EXPERT_DLO_ENDPOINT,
        supports_recording_schema=True,
    ),
    # --- SoftGym 绳子任务（rope_flatten / rope_configuration）---
    "RopeFlatten": TaskSpec(
        name="RopeFlatten",
        source="SoftGym",
        task_family="rope_flatten",
        default_robot="Panda",
        env_options=(ENV_OPTION_CABLE_MODEL,),
        teleop_entry=TELEOP_DLO_HUMAN,
        expert_entry=EXPERT_DLO_ENDPOINT,
        supports_recording_schema=True,
    ),
    "RopeConfiguration": TaskSpec(
        name="RopeConfiguration",
        source="SoftGym",
        task_family="rope_configuration",
        default_robot="Panda",
        env_options=(ENV_OPTION_CABLE_MODEL, ENV_OPTION_GOAL_FILE, ENV_OPTION_GOAL_CHARACTER),
        teleop_entry=TELEOP_DLO_HUMAN,
        expert_entry=EXPERT_DLO_ENDPOINT,
        supports_recording_schema=True,
    ),
    # --- 路径规划任务（cable_routing）---
    "CableRouting": TaskSpec(
        name="CableRouting",
        source="robosuite_change",
        task_family="cable_routing",
        default_robot="Panda",
        env_options=(ENV_OPTION_CABLE_MODEL,),
        teleop_entry=None,
        expert_entry=EXPERT_DLO_ENDPOINT,
    ),
    # --- 抓取-提起-放置任务 ---
    "CablePickLiftPlace": TaskSpec(
        name="CablePickLiftPlace",
        source="robosuite_change",
        task_family="endpoint_manipulation",
        default_robot="Panda",
        env_options=(ENV_OPTION_CABLE_MODEL, ENV_OPTION_GRASP_MODE),
        teleop_entry=TELEOP_DLO_HUMAN,
        expert_entry=EXPERT_DLO_ENDPOINT,
    ),
    # --- 原子动作可靠性测试 ---
    "CableAtomicTest": TaskSpec(
        name="CableAtomicTest",
        source="robosuite_change",
        task_family="atomic_test",
        default_robot="Panda",
        env_options=(ENV_OPTION_CABLE_MODEL, ENV_OPTION_GRASP_MODE),
        teleop_entry=None,
        expert_entry=EXPERT_DLO_ENDPOINT,
    ),
}

# 所有任务名称的元组，用于 argparse 的 choices 参数
TASKS = tuple(TASK_SPECS)


def get_task_spec(task):
    """根据任务名称获取 TaskSpec，不存在则抛出 ValueError。"""
    try:
        return TASK_SPECS[str(task)]
    except KeyError:
        raise ValueError(f"Unsupported task: {task}")


def task_supports_option(task, option):
    """检查指定任务是否支持某个选项。"""
    return get_task_spec(task).supports_option(option)


def tasks_with_teleop_entry(entry):
    """返回所有使用指定遥操作入口的任务名称集合。"""
    return {name for name, spec in TASK_SPECS.items() if spec.teleop_entry == entry}


def tasks_with_expert_entry(entry):
    """返回所有使用指定专家入口的任务名称集合。"""
    return {name for name, spec in TASK_SPECS.items() if spec.expert_entry == entry}


def expert_tasks():
    """返回所有支持专家脚本的任务名称集合（expert_entry 不为 None）。"""
    return {name for name, spec in TASK_SPECS.items() if spec.expert_entry is not None}
