"""
BaseDLOEnv — 所有 DLO（可变形线性物体）操作任务的统一基类。

本模块是 DLO Benchmark 的核心基础设施，提供：
  - 线缆点读取：从 MuJoCo 引擎提取线缆所有节点的三维坐标
  - 桌面接触度量：判断线缆是否平放在桌面上
  - 夹爪辅助函数：指尖位置、间距、闭合检测、中心校正
  - Flex 抓取系统：顶点直接操控、EMA 跟随、不可延伸性约束
  - 物理抓取系统：双侧接触检测、接触记忆、抬升就绪判断
  - 附着系统：mocap+weld 约束、延迟激活、多点切换
  - 模型加载/重置/观测注册的完整生命周期

子类（CableStraighten、CableThreading 等）继承本类后只需：
  - 覆写 reward()、_check_success()、_compute_metrics() 定义任务逻辑
  - 覆写 _load_model() 添加任务特定场景元素（柱子、路径点等）
  - 覆写 _reset_internal() 实现任务特定重置逻辑

向后兼容：CableBaseEnv = BaseDLOEnv
"""

import xml.etree.ElementTree as ET

import numpy as np

from robosuite.environments.manipulation.cable_in_task import CableInTask
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.dlo.cable_metrics import gripper_to_cable_distance
from robosuite.utils.dlo import physical_grasp as physical_grasp_utils
from robosuite.utils.dlo.task_scene_utils import create_weld_constraint
from robosuite.utils.mjcf_utils import new_body
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler


class BaseDLOEnv(ManipulationEnv):
    """所有 DLO 操作任务的统一基类。

    提供桌面上线缆操作的完整基础设施：模型加载、抓取系统、重置逻辑、观测注册。
    子类只需覆写 reward() 和 _check_success() 即可定义具体任务。
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        base_types="default",
        initialization_noise=None,
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 0.005, 0.0001),
        cable_model="rmb",          # 线缆物理模型类型: "rmb"(刚性链), "flex"(MuJoCo flexcomp), "segmented" 等
        grasp_mode="attachment",     # 抓取模式: "attachment"(mocap 约束) 或 "physical"(物理接触)
        anchor_enabled=True,         # 是否将线缆起点锚定到桌面（False 时两端均可自由移动）
        target_line_visible=False,   # 是否在场景中渲染目标线段的可视化 site
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=True,
        placement_initializer=None,  # 自定义放置采样器，为 None 时使用默认 UniformRandomSampler
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=500,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mjviewer",
        renderer_config=None,
        seed=None,
    ):
        if grasp_mode not in {"attachment", "physical"}:
            raise ValueError(f"Unsupported grasp_mode: {grasp_mode}")
        self.grasp_mode = str(grasp_mode)
        self.anchor_enabled = bool(anchor_enabled)
        self.target_line_visible = bool(target_line_visible)
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        # 桌面位于 z=0.8m 高度，所有线缆位置计算以此为基准
        self.table_offset = np.array((0.0, 0.0, 0.8))
        self.table_top_z = float(self.table_offset[2])

        # 创建 CableInTask（封装线缆配置与重置逻辑）
        # 子类（如 CableThreading）可能已在自己的 __init__ 中创建了 cable_in_task
        if not hasattr(self, 'cable_in_task') or self.cable_in_task is None:
            self.cable_in_task = CableInTask(
                cable_model=cable_model,
                table_full_size=table_full_size,
                table_friction=table_friction,
                table_offset=self.table_offset,
            )
        # 向后兼容别名
        self.cable_model = cable_model
        self.cable_radius = self.cable_in_task.cable_radius
        self.cable_length = self.cable_in_task.cable_length
        self.cable_clearance = self.cable_in_task.cable_clearance
        self.cable_tabletop_offset = self.cable_in_task.cable_tabletop_offset
        self.cable_centerline_z = self.cable_in_task.cable_centerline_z
        self.cable_point_reference_kind = self.cable_in_task.cable_point_reference_kind
        self.cable_point_reference_names = list(self.cable_in_task.cable_point_reference_names)

        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        self.use_object_obs = use_object_obs
        self.placement_initializer = placement_initializer
        # ---- 重置采样参数（向后兼容别名） ----
        self.reset_xy_center = self.cable_in_task.reset_xy_center.copy()
        self.reset_xy_range = self.cable_in_task.reset_xy_range
        self.reset_yaw_range = self.cable_in_task.reset_yaw_range
        self.reset_shape_wave_scale = self.cable_in_task.reset_shape_wave_scale
        self.reset_shape_noise_scale = self.cable_in_task.reset_shape_noise_scale
        self.reset_shape_noise_clip = self.cable_in_task.reset_shape_noise_clip
        self.reset_resample_attempts = self.cable_in_task.reset_resample_attempts
        self.reset_centerline_min_z = self.cable_centerline_z
        # ---- 成功判定阈值 ----
        self.success_table_contact_ratio_threshold = 0.95  # 关键点中至少 95% 需在桌面上才算 "cable_on_table"
        self.table_contact_z_tolerance = 0.025             # 关键点 z 与中心线的最大偏差（判断是否接触桌面）
        self.table_penetration_tolerance = 0.02            # 关键点允许的最大穿透深度

        # 目标线段起点/终点（straighten 任务的对齐目标）
        self.target_start = self.cable_in_task.default_target_start.copy()
        self.target_end = self.cable_in_task.default_target_end.copy()
        self.num_cable_points = None
        # ---- 抓取/附着系统状态 ----
        # attachment_enabled: 是否启用 mocap+weld 约束（每步在 _pre_action 中跟随夹爪）
        self.attachment_enabled = False
        self.attach_offset = np.zeros(3, dtype=float)            # mocap 相对夹爪 grip_site 的偏移
        self.attachment_follow_gain = 0.05                       # 每步跟随插值系数（0~1，越大跟随越快）
        self.attachment_velocity_damping = 0.3                   # 附着时线缆根节点速度阻尼系数（防止弹跳）
        # ---- flex 顶点直接操控状态 ----
        self._flex_grasp_active = False      # 是否启用 flex 顶点直接操控
        self._flex_grasp_vtx_idx = -1        # 被抓取的顶点索引
        self._flex_grasp_offset = np.zeros(3, dtype=float)  # 顶点相对 gripper 的偏移
        self._flex_grasp_vtx_indices = np.zeros(0, dtype=int)  # 被夹持的小段顶点索引
        self._flex_grasp_offsets = np.zeros((0, 3), dtype=float)  # 每个被夹持顶点相对 gripper 的偏移
        self._flex_grasp_segment_local_targets = np.zeros((0, 3), dtype=float)  # 夹持段在 gripper 局部坐标系中的目标点
        self._flex_grasp_support_mode = "segment"  # segment / hanging_arc
        self._flex_mid_grasp_span_radius = 10   # 中段抓取默认夹持半径（顶点数）
        self._flex_mid_grasp_spacing = 0.003    # 夹持段在 gripper 局部坐标系中的点间距
        # ---- 延迟附着：仅在夹爪闭合后才激活 ----
        self._attach_pending = False         # 是否有待激活的附着
        self._attach_pending_params = {}     # 待激活的附着参数
        self._attach_grip_threshold = 0.3    # 夹爪闭合阈值（关节位置 > 此值视为闭合）
        self._attach_max_distance = 0.040    # grip site 到目标点的最大距离（40mm，适配 flex 水平接近策略）
        self._attach_between_fingers_distance = 0.030  # fingerpad box corridor，仍需同时满足 grip site 近距离
        self._flex_grasp_ever_active = False   # 本 episode 中是否曾激活过 flex 抓取
        # 负值扩展 finger axis 检测范围：Panda 夹爪的 fingerpad collision box
        # （8x4x8mm）中心并不在真实夹持面上，夹持面比 collision box 更靠外。
        # -0.50 表示允许投影值延伸到 [−0.50, 1.50]，覆盖 collision box 边界外 50mm。
        self._attach_finger_axis_margin = -0.50
        self.grasp_endpoint_body_names = ("cable_B0", "cable_end")  # 默认端点 body 名称
        self.graspable_body_names = []   # 在 _load_model 中由 CableInTask 填充
        self.graspable_point_count = 0   # 在 _load_model 中由 CableInTask 填充
        self._flex_comp_name = None      # 在 _load_model 中 flex 分支设置
        # ---- Anchor constraint for flex cables (pins first vertex to table) ----
        self.anchor_body_name = "cable_anchor"
        self.anchor_eq_name = "cable_anchor_weld"
        self.anchor_body_id = None
        self.anchor_eq_id = None
        self.anchor_pos = None
        self.anchor_tolerance = 0.02
        # mocap/weld 相关名称和 ID（在 _load_model 和 _setup_references 中填充）
        self.mocap_body_names = []
        self.grasp_eq_names = []
        self.grasp_body_to_index = {}   # body 名称 -> 可抓取索引
        self.grasp_point_to_body = {}   # 可抓取索引 -> body 名称
        self.active_grasp_endpoint_index = 1                     # 当前活跃端点索引（0=起点, 1=终点）
        self.active_grasp_body_name = self.grasp_endpoint_body_names[-1]
        self.active_grasp_point_idx = max(0, self.graspable_point_count - 1)
        # ---- 物理抓取参数 ----
        self.physical_grasp_lift_threshold = 0.03                # 物理抓取成功需抬起的最小高度
        self.physical_grasp_initial_endpoint_z = 0.0             # 重置时记录的端点初始 z（用于计算抬起高度）
        self._physical_grasp_point_idx = -1                      # physical / flex 中点抓取时用于度量的顶点索引
        # ---- 运行时 ID 缓存（在 _setup_references 中填充） ----
        self.mocap_body_ids = []
        self.mocap_ids = []          # MuJoCo mocap 索引（用于直接设置 mocap_pos/quat）
        self.grasp_eq_ids = []       # weld 约束的 equality 索引（用于设置 eq_active）
        self.cable_end_body_id = None
        self.cable_start_body_id = None
        self.cable_shape_joint_names = []   # rmb 线缆的形状关节名列表（如 cable_J1_0, cable_J1_1, ...）
        self.last_reset_summary = {}        # 上次重置的诊断信息（供调试用）

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types=base_types,
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=lite_physics,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
            seed=seed,
        )

    # ------------------------------------------------------------------
    # Default hooks — subclasses override these
    # ------------------------------------------------------------------

    def reward(self, action=None):
        """默认 reward：返回 0（子类必须覆写）。"""
        return 0.0

    def _record_initial_geometry(self):
        """记录重置后的初始几何信息（子类覆写以记录任务特定信息）。"""
        pass

    def _compute_metrics(self):
        """Default metrics: basic grasp + contact info."""
        points = self._get_cable_points()
        table_metrics = self._table_contact_metrics(points)
        try:
            gripper_distance = gripper_to_cable_distance(self._get_gripper_site_position(), points)
        except ValueError:
            gripper_distance = np.inf
        return {
            **table_metrics,
            "grasp_mode": self.grasp_mode,
            "gripper_to_cable_distance": gripper_distance,
            "physical_grasp_contact_count": self._physical_grasp_contact_count(),
            "physical_grasp_lift_height": self._physical_grasp_lift_height(),
            "physical_grasp_success": self._physical_grasp_success(),
            "attachment_eq_active_count": self._attachment_eq_active_count(),
            "flex_grasp_ever_active": bool(self._flex_grasp_ever_active),
            "success": False,
        }

    def _check_success(self):
        """默认成功判定：返回 False（子类必须覆写）。"""
        return False

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self):
        """构建 MuJoCo XML 模型。

        整体流程：
        1. 调用父类 _load_model() 加载机器人基础模型
        2. 创建 TableArena（桌面 + 地板 + 灯光）
        3. 通过 CableInTask.setup_scene() 构建线缆场景
        4. 为每个可抓取 body 创建 weld 约束

        子类可以覆写 _create_arena() 和 _setup_cable_scene(arena) 来自定义。
        """
        super()._load_model()

        # 将机器人放置在桌面边缘
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        arena = self._create_arena()
        self._setup_cable_scene(arena)

    def _create_arena(self):
        """创建桌面 arena。子类可覆写以添加额外场景元素。"""
        arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        arena.set_origin([0, 0, 0])
        return arena

    def _setup_cable_scene(self, arena):
        """在 arena 中构建线缆场景并创建 ManipulationTask。

        子类可覆写以在调用 super() 前后添加额外场景元素（如柱子、路径点等）。
        """
        # 委托给 CableInTask 构建线缆场景
        scene_info = self.cable_in_task.setup_scene(arena, self.placement_initializer)

        # 提取场景信息到 env 属性
        self.cable = scene_info.cable_object
        self._is_flex_cable = scene_info.is_flex
        self.cable_root_joint = scene_info.cable_root_joint
        self.graspable_body_names = scene_info.graspable_body_names
        self.graspable_point_count = scene_info.graspable_point_count
        self.grasp_endpoint_body_names = scene_info.grasp_endpoint_body_names
        self.grasp_point_to_body = scene_info.grasp_point_to_body
        self.grasp_body_to_index = scene_info.grasp_body_to_index
        self.mocap_body_names = scene_info.mocap_body_names
        self.grasp_eq_names = scene_info.grasp_eq_names
        self.num_cable_points = scene_info.num_cable_points
        if self.placement_initializer is None:
            self.placement_initializer = scene_info.placement_initializer
        if scene_info.flex_comp_name is not None:
            self._flex_comp_name = scene_info.flex_comp_name

        # 更新线缆物理参数（使用正式对象覆盖 probe 的值）
        self.cable_point_reference_kind = self.cable.point_reference_kind
        self.cable_point_reference_names = list(self.cable.point_reference_names)
        self.cable_radius = float(self.cable.cable_radius)
        self.cable_tabletop_offset = float(self.cable.tabletop_centerline_offset)
        self.cable_centerline_z = self.table_top_z + self.cable_radius + self.cable_clearance

        # 目标线段（CableInTask 已计算默认值，这里用正式对象更新）
        self.target_start = self.cable_in_task.default_target_start.copy()
        self.target_end = self.cable_in_task.default_target_end.copy()

        if self.graspable_point_count:
            self.active_grasp_point_idx = self.graspable_point_count - 1
            self.active_grasp_body_name = self.grasp_point_to_body[self.active_grasp_point_idx]

        # Create anchor body + endpoint mocap for flex cables
        if self._is_flex_cable:
            anchor_body = new_body(name=self.anchor_body_name, pos=[0, 0, self.table_top_z])
            arena.worldbody.append(anchor_body)
            # 端点 mocap body（用于 attachment 模式的 weld 跟随）
            from robosuite.utils.dlo.task_scene_utils import create_mocap_body
            self._flex_mocap_name = "flex_endpoint_mocap"
            arena.worldbody.append(create_mocap_body(
                self._flex_mocap_name, (0.0, 0.0, self.cable_centerline_z)
            ))

        # 构建 ManipulationTask
        self.model = ManipulationTask(
            mujoco_arena=arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.cable if not self._is_flex_cable else None,
        )

        # CableBaseEnv 保留：weld 约束创建（属于抓取系统）
        for eq_elem in scene_info.equality_elements:
            self.model.equality.append(eq_elem)

        # Create anchor + endpoint weld for flex cables
        if self._is_flex_cable:
            first_flex_body = scene_info.graspable_body_names[0]
            self.model.equality.append(create_weld_constraint(
                self.anchor_eq_name, first_flex_body, self.anchor_body_name, solref="0.01 1",
            ))
            # 端点 weld（初始禁用，attach 时激活）
            # 使用高刚度 solref 防止线缆弹性拉回
            last_flex_body = scene_info.graspable_body_names[-1]
            self._flex_end_weld_name = "flex_end_weld"
            self.model.equality.append(create_weld_constraint(
                self._flex_end_weld_name, self._flex_mocap_name, last_flex_body, solref="0.002 1",
            ))

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_references(self):
        """在 MuJoCo 模型加载完成后，缓存各类 ID 供运行时快速查询。"""
        super()._setup_references()

        # 委托给 CableInTask 解析线缆相关 ID
        ids = self.cable_in_task.resolve_sim_ids(self.sim)
        self.cable_point_ids = ids["cable_point_ids"]
        self._flex_id = ids.get("flex_id")
        self._flex_vertadr = ids.get("flex_vertadr")
        self._flex_vertnum = ids.get("flex_vertnum")

        # 应用视觉/物理修复（geom_group + flex 摩擦力）
        self.cable_in_task.apply_visual_fixes(self.sim)

        self.cable_start_body_id = ids["cable_start_body_id"]
        self.cable_end_body_id = ids["cable_end_body_id"]
        self.cable_shape_joint_names = ids["cable_shape_joint_names"]
        self.num_cable_points = ids["num_cable_points"]

        # CableBaseEnv 保留：mocap/weld ID 解析（属于抓取系统）
        self.mocap_body_ids = []
        self.mocap_ids = []
        for mocap_name in self.mocap_body_names:
            body_id = self.sim.model.body_name2id(mocap_name)
            self.mocap_body_ids.append(body_id)
            self.mocap_ids.append(self.sim.model.body_mocapid[body_id])
        self.grasp_eq_ids = [None for _ in self.grasp_eq_names]
        for eq_id in range(self.sim.model.neq):
            eq_name = self.sim.model.equality(eq_id).name
            for idx, expected_name in enumerate(self.grasp_eq_names):
                if eq_name == expected_name:
                    self.grasp_eq_ids[idx] = eq_id

        # Resolve anchor + endpoint weld IDs for flex cables
        self._flex_end_weld_eq_id = None
        self._flex_mocap_id = None
        if self._is_flex_cable:
            try:
                self.anchor_body_id = self.sim.model.body_name2id(self.anchor_body_name)
            except KeyError:
                self.anchor_body_id = None
            for eq_id in range(self.sim.model.neq):
                eq_name = self.sim.model.equality(eq_id).name
                if eq_name == self.anchor_eq_name:
                    self.anchor_eq_id = eq_id
                if hasattr(self, '_flex_end_weld_name') and eq_name == self._flex_end_weld_name:
                    self._flex_end_weld_eq_id = eq_id
            # 端点 mocap ID
            if hasattr(self, '_flex_mocap_name'):
                try:
                    flex_mocap_body_id = self.sim.model.body_name2id(self._flex_mocap_name)
                    self._flex_mocap_id = self.sim.model.body_mocapid[flex_mocap_body_id]
                except KeyError:
                    self._flex_mocap_id = None

        # 将 placement_initializer 和 cable_root_joint 存入 CableInTask 供重置使用
        self.cable_in_task.set_scene_state("placement_initializer", self.placement_initializer)
        self.cable_in_task.set_scene_state("cable_root_joint", self.cable_root_joint)

    def _setup_observables(self):
        observables = super()._setup_observables()

        if self.use_object_obs:
            modality = "object"

            @sensor(modality=modality)
            def cable_points(obs_cache):
                return self._get_cable_points().reshape(-1)

            @sensor(modality=modality)
            def cable_keypoints(obs_cache):
                return self._get_cable_points().reshape(-1)

            @sensor(modality=modality)
            def target_line(obs_cache):
                return np.concatenate([self.target_start, self.target_end])

            @sensor(modality=modality)
            def task_goal(obs_cache):
                return self._task_goal_obs()

            sensors = [cable_points, cable_keypoints, target_line, task_goal]
            @sensor(modality=modality)
            def cable_end_pos(obs_cache):
                return self._get_cable_end_pos()

            @sensor(modality=modality)
            def attachment_state(obs_cache):
                return np.array(
                    [
                        1.0 if self.attachment_enabled else 0.0,
                        float(self.active_grasp_point_idx),
                    ],
                    dtype=float,
                )

            @sensor(modality=modality)
            def physical_grasp_state(obs_cache):
                return np.array(
                    [
                        float(self._physical_grasp_contact_count()),
                        float(self._physical_grasp_lift_height()),
                        1.0 if self._physical_grasp_success() else 0.0,
                    ],
                    dtype=float,
                )

            sensors.extend([cable_end_pos, attachment_state, physical_grasp_state])
            names = [s.__name__ for s in sensors]

            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_cable_keypoints(self):
        """公共 API：返回线缆关键点位置（供外部脚本调用）。"""
        return self._get_cable_points()

    def get_dlo_metrics(self):
        """公共 API：返回任务指标的副本（供外部脚本调用）。"""
        return dict(self._compute_metrics())

    def get_task_success(self):
        """公共 API：返回任务是否成功（供外部脚本调用）。"""
        return bool(self._check_success())

    def _task_goal_obs(self):
        """返回任务目标观测：[target_start(3), target_end(3)]。"""
        return np.concatenate([self.target_start, self.target_end])

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_internal(self):
        """环境重置的核心方法。

        委托给 CableInTask 处理线缆重置，CableBaseEnv 负责：
        1. 禁用抓取约束和 anchor
        2. 调用 CableInTask.apply_reset()
        3. 对 flex 线缆进行重力沉降 + 锚定
        4. 同步 mocap 到线缆位置
        5. 记录初始几何信息
        """
        super()._reset_internal()
        # 重置时禁用所有抓取约束
        self.attachment_enabled = False
        self._flex_grasp_ever_active = False
        for eq_id in self.grasp_eq_ids:
            if eq_id is not None:
                self.sim.data.eq_active[eq_id] = 0
        # 禁用 anchor（重置期间需要自由沉降）
        if self.anchor_eq_id is not None:
            self.sim.data.eq_active[self.anchor_eq_id] = 0

        # 委托给 CableInTask 处理线缆重置
        reset_info = self.cable_in_task.apply_reset(
            self.sim, self.rng, deterministic=self.deterministic_reset,
        )

        # Flex 线缆：重力沉降后锚定起点（anchor_enabled=False 时跳过）
        if self._is_flex_cable and self.anchor_body_id is not None and self.anchor_enabled:
            self._settle_flex_cable()
            # Zero all flex vertex velocities to prevent drift after anchor activation
            adr = self._flex_vertadr
            nvert = self._flex_vertnum
            if adr is not None and nvert is not None:
                vel_slice = slice(int(adr) * 3, int(adr) * 3 + int(nvert) * 3)
                self.sim.data.qvel[vel_slice] = 0.0
            first_pos = self._get_cable_points()[0].copy()
            self.sim.model.body_pos[self.anchor_body_id] = first_pos
            self.anchor_pos = first_pos.copy()
            self.sim.data.eq_active[self.anchor_eq_id] = 1
            self.sim.forward()

        # CableBaseEnv 负责：grasp 同步（跳过没有 mocap 的子类如 CableThreading）
        if self.mocap_ids:
            self._sync_all_mocaps_to_grasp_bodies()

        # 记录诊断信息
        self.last_reset_summary = {
            **reset_info,
            "grasp_mode": self.grasp_mode,
            "cable_start_pos": self._get_cable_start_pos().copy(),
            "cable_end_pos": self._get_cable_end_pos().copy(),
        }
        if self.graspable_body_names:
            self.physical_grasp_initial_endpoint_z = float(self._get_active_grasp_body_pos()[2])
        self._record_initial_geometry()

    def _settle_flex_cable(self, max_steps=100, settle_vel_threshold=0.005):
        """Run physics steps to let the flex cable settle under gravity.

        After initial shape is set by CableInTask.apply_reset, allow the cable
        to drape naturally under gravity before anchoring the start point.
        Uses velocity-based early stopping: once all vertex velocities are below
        threshold for 10 consecutive steps, settling is complete.
        """
        adr = self._flex_vertadr
        nvert = self._flex_vertnum
        if adr is None or nvert is None:
            return
        vel_slice = slice(int(adr) * 3, int(adr) * 3 + int(nvert) * 3)
        settle_count = 0
        for _ in range(max_steps):
            self.sim.step()
            max_vel = float(np.max(np.abs(self.sim.data.qvel[vel_slice])))
            if max_vel < settle_vel_threshold:
                settle_count += 1
                if settle_count >= 10:
                    break
            else:
                settle_count = 0

    # ------------------------------------------------------------------
    # Pre-action
    # ------------------------------------------------------------------

    def _enforce_flex_inextensible(self):
        """Post-step: dampen velocities on over-stretched edges.

        Instead of modifying qpos directly (which causes instability),
        only dampen velocities along the stretching direction. This gently
        resists stretching without creating position/velocity mismatches.
        """
        nvert = self._flex_vertnum
        adr = self._flex_vertadr
        if adr is None or nvert is None or nvert < 2:
            return
        spacing = getattr(getattr(self, '_cable_in_task', None), 'flex_vertex_spacing', None) or 0.01
        rest_len = spacing
        stretch_threshold = rest_len * 1.05  # 5% tolerance

        obj_body_id = self.sim.model.body_name2id("object")
        body_pos = self.sim.data.body_xpos[obj_body_id]
        body_rot = self.sim.data.body_xmat[obj_body_id].reshape(3, 3)

        verts_world = self.sim.data.flexvert_xpos[adr:adr + nvert].copy().reshape(nvert, 3)
        verts_local = (body_rot.T @ (verts_world - body_pos).T).T

        vel_start = int(adr) * 3
        for i in range(nvert - 1):
            edge = verts_local[i + 1] - verts_local[i]
            edge_len = float(np.linalg.norm(edge))
            if edge_len > stretch_threshold and edge_len > 1e-8:
                # Project vertex velocities onto edge direction and dampen
                edge_dir = edge / edge_len
                for vi in (i, i + 1):
                    vel = self.sim.data.qvel[vel_start + vi * 3: vel_start + vi * 3 + 3]
                    radial_vel = float(np.dot(vel, edge_dir))
                    if radial_vel > 0:  # only dampen stretching velocity
                        self.sim.data.qvel[vel_start + vi * 3: vel_start + vi * 3 + 3] -= edge_dir * radial_vel * 0.5

    def _post_action(self, action):
        """每个控制步执行后调用。强制 flex 线缆不可延伸。"""
        reward, done, info = super()._post_action(action)
        if self._is_flex_cable and self._flex_vertnum is not None:
            self._enforce_flex_inextensible()
        return reward, done, info

    def _pre_action(self, action, policy_step=False):
        """每个控制步执行前调用。如果 attachment 模式启用，更新 mocap 位置跟随夹爪。"""
        super()._pre_action(action, policy_step=policy_step)

        # 夹爪一旦张开，吸附必须立即失效，避免出现“松开后仍悬挂一段时间”的非自然行为。
        if self.attachment_enabled and not self._is_gripper_closed(action):
            self._disable_attachment()

        # 检查待激活附着：仅在夹爪闭合且距离足够近时才真正激活
        if self._attach_pending:
            grip_closed = self._is_gripper_closed(action)
            close_enough = self._is_gripper_close_enough()
            if grip_closed and close_enough:
                self._activate_pending_attachment()

        if self.attachment_enabled or self._flex_grasp_active:
            self._attach_cable_end_to_gripper()

        # flex 物理抓取：更新抓取状态
        if self._is_flex_cable and self.grasp_mode == "physical":
            self._update_flex_grasp_state(action)

    # ------------------------------------------------------------------
    # Grasp / attachment
    # ------------------------------------------------------------------

    def set_attachment_enabled(self, enabled, endpoint_name=None, point_idx=None, body_name=None):
        """启用或禁用线缆端点的 mocap 附着。

        附着机制：
        - 端点抓取：通过 MuJoCo 的 weld 约束将线缆 body 与 mocap body 绑定
        - flex 中点抓取：直接操控 flex vertex qpos（更自然的形变，无直线段）
        - 延迟激活：仅在夹爪闭合且距离足够近时才真正激活

        参数：
            enabled: True 启用附着，False 禁用
            endpoint_name: 指定附着的端点名称（如 "cable_end"）
            point_idx: 指定附着的可抓取点索引（flex 中点时直接操控顶点）
            body_name: 指定附着的 body 名称
        """
        if self.grasp_mode == "physical":
            if not bool(enabled):
                self._disable_attachment()
                return
            self._attach_pending = False
            self._attach_pending_params = {}
            self.attachment_enabled = False
            self._physical_grasp_point_idx = -1
            if point_idx is not None:
                self._set_active_grasp_point(point_idx)
                if self._is_flex_cable:
                    self._physical_grasp_point_idx = int(np.clip(int(point_idx), 0, max(self._flex_vertnum - 1, 0)))
            elif body_name is not None:
                self._set_active_grasp_body(body_name)
            elif endpoint_name is not None:
                self._set_active_grasp_body(endpoint_name)
            return

        # 禁用附着
        if not bool(enabled):
            self._disable_attachment()
        else:
            # 启用附着：存储待激活参数，等夹爪闭合后真正激活
            self._attach_pending = True
            self._attach_pending_params = {
                "endpoint_name": endpoint_name,
                "point_idx": point_idx,
                "body_name": body_name,
            }
            return  # 不立即激活，等 _pre_action 中检查

    def _disable_attachment(self):
        """立即关闭当前所有吸附状态和 weld 约束。"""
        self._dampen_cable_velocities(factor=0.0)
        # flex 线缆：零化顶点速度，防止释放后残余速度导致漂移
        if self._is_flex_cable:
            adr = self._flex_vertadr
            nvert = self._flex_vertnum
            if adr is not None and nvert is not None:
                vel_slice = slice(int(adr) * 3, int(adr) * 3 + int(nvert) * 3)
                self.sim.data.qvel[vel_slice] = 0.0
        self._flex_grasp_active = False
        self._flex_grasp_vtx_idx = -1
        self._flex_grasp_offset = np.zeros(3, dtype=float)
        self._flex_grasp_vtx_indices = np.zeros(0, dtype=int)
        self._flex_grasp_offsets = np.zeros((0, 3), dtype=float)
        self._flex_grasp_segment_local_targets = np.zeros((0, 3), dtype=float)
        self._flex_grasp_support_mode = "segment"
        self._physical_grasp_point_idx = -1
        self._attach_pending = False
        self._attach_pending_params = {}
        self.attachment_enabled = False
        for eq_id in self.grasp_eq_ids:
            if eq_id is not None:
                self.sim.data.eq_active[eq_id] = 0
        # flex 端点 weld 禁用
        if self._is_flex_cable and hasattr(self, '_flex_end_weld_eq_id') and self._flex_end_weld_eq_id is not None:
            self.sim.data.eq_active[self._flex_end_weld_eq_id] = 0
        self.sim.forward()

    def _activate_pending_attachment(self):
        """将待激活的附着正式激活（夹爪已闭合且距离足够近）。"""
        params = self._attach_pending_params
        self._attach_pending = False
        self._attach_pending_params = {}

        endpoint_name = params.get("endpoint_name")
        point_idx = params.get("point_idx")
        body_name = params.get("body_name")
        self.attach_offset = self._gripper_clamp_center_offset()
        self._physical_grasp_point_idx = -1

        # flex cable: resolve endpoint_name to vertex index
        if self._is_flex_cable and point_idx is None and endpoint_name is not None:
            n = self._flex_vertnum
            if n > 0:
                if endpoint_name == "cable_end":
                    point_idx = n - 1
                elif endpoint_name == "cable_start":
                    point_idx = 0

        # flex 顶点抓取
        if self._is_flex_cable and point_idx is not None:
            nvert = self._flex_vertnum
            if 0 <= point_idx <= nvert - 1:
                is_endpoint = (point_idx == 0 or point_idx == nvert - 1)
                if is_endpoint:
                    # 端点抓取：使用 mocap+weld 约束（与 CableThreading 一致）
                    # 这样线缆通过 weld 跟随夹爪，保留物理交互（碰撞、重力、弹性）
                    self._flex_grasp_active = False
                    self._flex_grasp_ever_active = True
                    self._physical_grasp_point_idx = int(point_idx)
                    self._set_active_grasp_point(point_idx)
                    # flex 线缆：使用端点 mocap + weld（子类可覆写 _activate_flex_attachment）
                    if self._is_flex_cable:
                        if hasattr(self, '_activate_flex_attachment'):
                            self._activate_flex_attachment()
                        elif hasattr(self, '_flex_end_weld_eq_id') and self._flex_end_weld_eq_id is not None:
                            flex_end_pos = self._get_cable_end_pos()
                            self.sim.data.mocap_pos[self._flex_mocap_id] = flex_end_pos
                            self.sim.data.mocap_quat[self._flex_mocap_id] = np.array([1.0, 0.0, 0.0, 0.0])
                            self.sim.data.eq_active[self._flex_end_weld_eq_id] = 1
                            self.attachment_enabled = True
                            self.sim.forward()
                        else:
                            self.attachment_enabled = True
                    else:
                        # 非 flex 线缆：使用 grasp_eq_ids 多点 weld 系统
                        for eq_id in self.grasp_eq_ids:
                            if eq_id is not None:
                                self.sim.data.eq_active[eq_id] = 0
                        if self.grasp_eq_ids and 0 <= self.active_grasp_point_idx < len(self.grasp_eq_ids):
                            active_eq_id = self.grasp_eq_ids[self.active_grasp_point_idx]
                            if active_eq_id is not None:
                                self._sync_active_mocap_to_grasp_body()
                                self.sim.data.eq_active[active_eq_id] = 1
                        self.attachment_enabled = True
                        self.sim.forward()
                    return
                else:
                    # 中间点抓取：直接操控顶点（flexcomp 物理引擎处理其余部分）
                    self._physical_grasp_point_idx = int(point_idx)
                    self._flex_grasp_active = True
                    self._flex_grasp_ever_active = True
                    self._flex_grasp_vtx_idx = int(point_idx)
                    grip_pos = self._get_gripper_site_position()
                    vtx_pos = self.sim.data.flexvert_xpos[point_idx].copy()
                    self._flex_grasp_offset = grip_pos - vtx_pos
                    self._flex_grasp_support_mode = "endpoint"
                    self._flex_grasp_vtx_indices = np.array([point_idx], dtype=int)
                    self._flex_grasp_offsets = (grip_pos - vtx_pos).reshape(1, 3)
                    self._flex_grasp_segment_local_targets = np.zeros((1, 3), dtype=float)
                    self.attachment_enabled = True
                    self.sim.forward()
                    return
            self._flex_grasp_active = False

        # 端点 / weld 路径
        if point_idx is not None:
            self._set_active_grasp_point(point_idx)
        if body_name is not None:
            self._set_active_grasp_body(body_name)
        if endpoint_name is not None:
            self._set_active_grasp_body(endpoint_name)
        self.attachment_enabled = True
        for eq_id in self.grasp_eq_ids:
            if eq_id is not None:
                self.sim.data.eq_active[eq_id] = 0
        if self.active_grasp_point_idx >= 0 and self.grasp_eq_ids and self.active_grasp_point_idx < len(self.grasp_eq_ids):
            active_eq_id = self.grasp_eq_ids[self.active_grasp_point_idx]
            if active_eq_id is None:
                raise RuntimeError(f"No attachment equality found for grasp point {self.active_grasp_point_idx}")
            self._sync_active_mocap_to_grasp_body()
            self.sim.data.eq_active[active_eq_id] = 1
        self.sim.forward()

    def _is_gripper_closed(self, action):
        """检查夹爪是否已闭合。优先检查实际关节位置，回退到 action 命令。

        Panda 夹爪关节范围 [0, 0.04]，闭合时 ≈0.035-0.04。
        UR5e/Robotiq 夹爪关节范围更大。
        grip 命令约定：正值=闭合，负值=张开。
        """
        robot = self.robots[0]
        arm = robot.arms[0]
        gripper = robot.gripper[arm]
        if gripper.dof == 0:
            return True
        # 优先：检查实际关节位置（避免命令已发送但物理未到位的误判）
        try:
            joints = gripper.joints
            if joints:
                positions = [float(self.sim.data.qpos[self.sim.model.jnt_qposadr[j]]) for j in joints]
                # 夹爪闭合阈值：取关节范围的 60%
                # Panda: range [0, 0.04] → threshold ≈ 0.024
                # 需要同时检查正负值（不同安装方向）
                closed_threshold = 0.024
                if all(abs(p) > closed_threshold for p in positions):
                    return True
                # 夹爪完全张开（关节接近 0）：不视为闭合
                open_threshold = 0.008
                if all(abs(p) < open_threshold for p in positions):
                    return False
        except (AttributeError, IndexError, KeyError):
            pass
        # 回退：action 命令值
        gripper_val = float(action[-gripper.dof])
        return gripper_val > self._attach_grip_threshold

    def _is_gripper_close_enough(self):
        """检查夹爪与目标抓取点的距离是否在阈值内。

        attachment 模式：只检查距离（weld 约束会处理精确定位）。
        physical 模式：检查距离 + 目标在指间（需要物理接触）。
        """
        if not self._attach_pending:
            return False
        params = self._attach_pending_params

        grip_pos = self._get_gripper_site_position()
        target_pos = self._resolve_attach_target_pos(params)
        if target_pos is None:
            return False
        dist = float(np.linalg.norm(grip_pos - target_pos))
        if dist >= self._attach_max_distance:
            return False
        # attachment 模式：距离足够近即可（weld 约束处理定位）
        if self.grasp_mode != "physical":
            return True
        # physical 模式：需要目标在夹爪指间
        return self._is_point_between_gripper_fingerpads(target_pos)

    def _resolve_attach_target_pos(self, params):
        """根据 pending params 解析附着目标位置。优先用 endpoint_name/body_name。"""
        # 优先：endpoint_name（使用 body 位置；flex 无 body 时用顶点位置）
        endpoint_name = params.get("endpoint_name")
        if endpoint_name is not None:
            # flex 线缆：endpoint 对应首/末顶点
            if self._is_flex_cable:
                n = self._flex_vertnum
                if n > 0:
                    if endpoint_name == "cable_end":
                        return self.sim.data.flexvert_xpos[n - 1].copy()
                    elif endpoint_name == "cable_start":
                        return self.sim.data.flexvert_xpos[0].copy()
            # body 型线缆：通过 body 查找
            body_name = self._endpoint_alias_to_body_name(endpoint_name)
            if body_name is not None:
                try:
                    body_id = self.sim.model.body_name2id(self._resolve_cable_name(body_name, "body"))
                    return self.sim.data.xpos[body_id].copy()
                except (ValueError, KeyError):
                    pass

        # 次选：body_name
        body_name = params.get("body_name")
        if body_name is not None:
            try:
                body_id = self.sim.model.body_name2id(self._resolve_cable_name(body_name, "body"))
                return self.sim.data.xpos[body_id].copy()
            except (ValueError, KeyError):
                pass

        # 末选：point_idx（flex 用顶点，body 型映射到 body）
        point_idx = params.get("point_idx")
        if point_idx is not None:
            if self._is_flex_cable:
                return self.sim.data.flexvert_xpos[point_idx].copy()
            bname = self.grasp_point_to_body.get(int(point_idx))
            if bname is None:
                return None
            try:
                bid = self.sim.model.body_name2id(self._resolve_cable_name(bname, "body"))
                return self.sim.data.xpos[bid].copy()
            except (ValueError, KeyError):
                return None

        return None

    def _attach_cable_end_to_gripper(self):
        """每步更新抓取点位置使其跟随夹爪。

        flex 中点抓取：直接操控 flex vertex qpos，让弹性自然处理形变。
        端点抓取：通过 mocap + weld 约束平滑跟随。
        """
        grip_pos = self._get_gripper_site_position()

        # flex 端点 weld 跟随（attachment 模式）
        # 使用更高的跟随增益防止线缆弹性拉回
        # 注意：physical 模式下 _flex_grasp_active 走顶点操控路径，不走 mocap
        if (self._is_flex_cable and self.attachment_enabled
                and not self._flex_grasp_active
                and hasattr(self, '_flex_mocap_id') and self._flex_mocap_id is not None):
            target_pos = grip_pos + self.attach_offset
            current_pos = self.sim.data.mocap_pos[self._flex_mocap_id].copy()
            flex_follow_gain = min(1.0, self.attachment_follow_gain * 3.0)  # 更高增益
            self.sim.data.mocap_pos[self._flex_mocap_id] = (
                current_pos + flex_follow_gain * (target_pos - current_pos)
            )
            return

        # flex 顶点直接操控模式
        if self._flex_grasp_active:
            # 物理模式：使用多顶点跟随（更稳定）
            if self.grasp_mode == "physical":
                self._flex_tail_follow_gripper()
                return
            # attachment 模式：单顶点操控
            clamp_offset = self._gripper_clamp_center_offset()
            target_world = grip_pos + clamp_offset + self.attach_offset
            adr = self._flex_vertadr

            try:
                obj_body_id = self.sim.model.body_name2id("flex_cable_container")
            except (KeyError, ValueError):
                obj_body_id = self.sim.model.body_name2id("object")
            body_pos = self.sim.data.body_xpos[obj_body_id]
            body_rot = self.sim.data.body_xmat[obj_body_id].reshape(3, 3)
            grip_site = self.robots[0].gripper[self.robots[0].arms[0]].important_sites["grip_site"]
            grip_rot = self.sim.data.site_xmat[self.sim.model.site_name2id(grip_site)].reshape(3, 3)

            if self._flex_grasp_vtx_indices.size == 0:
                self._flex_grasp_vtx_indices = np.array([self._flex_grasp_vtx_idx], dtype=int)
                self._flex_grasp_offsets = self._flex_grasp_offset.reshape(1, 3)
            if self._flex_grasp_segment_local_targets.shape[0] != self._flex_grasp_vtx_indices.size:
                self._flex_grasp_segment_local_targets = np.zeros((self._flex_grasp_vtx_indices.size, 3), dtype=float)

            spacing = getattr(self.cable_in_task, "flex_vertex_spacing", None) or 0.01
            for local_idx, vtx_idx in enumerate(self._flex_grasp_vtx_indices):
                local_target = self._flex_grasp_segment_local_targets[local_idx]
                target_world_i = target_world + grip_rot @ local_target
                body_frame_pos = body_rot.T @ (target_world_i - body_pos)
                grid_rest_x = int(vtx_idx) * spacing
                body_frame_pos[0] -= grid_rest_x
                self.sim.data.qpos[adr + int(vtx_idx) * 3: adr + int(vtx_idx) * 3 + 3] = body_frame_pos

            return

        # mocap + weld 约束模式（端点抓取）
        target_pos = grip_pos + self.attach_offset
        current_pos = self.sim.data.mocap_pos[self.mocap_ids[self.active_grasp_point_idx]].copy()
        self.sim.data.mocap_pos[self.mocap_ids[self.active_grasp_point_idx]] = (
            current_pos + self.attachment_follow_gain * (target_pos - current_pos)
        )
        # flex 电缆没有 root joint，跳过速度阻尼
        if self.cable_root_joint is not None:
            self._dampen_cable_velocities(factor=self.attachment_velocity_damping)

    def _update_flex_grasp_state(self, action):
        """更新 flex 物理抓取状态（参考 CableThreading 实现）。

        激活条件：夹爪闭合 + 夹爪间距接近线缆直径 + 线缆在夹持区域内。
        停用条件：夹爪张开。
        """
        grip_closed = self._is_gripper_closed(action)

        if grip_closed and not self._flex_grasp_active:
            # 检查线缆是否在夹爪附近
            grip_pos = self._get_gripper_site_position()
            points = self._get_cable_points()
            nvert = len(points)

            # 优先抓取端点（与 drag_to 的收敛检查一致）
            # 检查端点是否在夹爪附近
            end_dist = float(np.linalg.norm(points[-1] - grip_pos))
            start_dist = float(np.linalg.norm(points[0] - grip_pos))

            if end_dist < 0.040:
                self._flex_grasp_vtx_idx = nvert - 1
            elif start_dist < 0.040:
                self._flex_grasp_vtx_idx = 0
            else:
                # 回退到最近顶点
                dists = np.linalg.norm(points - grip_pos, axis=1)
                nearest_idx = int(np.argmin(dists))
                if float(dists[nearest_idx]) < 0.040:
                    self._flex_grasp_vtx_idx = nearest_idx
                else:
                    return  # 没有顶点在夹爪附近

            candidate_pos = points[self._flex_grasp_vtx_idx]

            # 检查 1：夹爪间距必须接近线缆直径（gap ≈ 2*cable_radius）
            # 允许更大容差：flex 线缆形变后抓取点可能不完全居中
            gap_width = self._fingerpad_gap_width()
            cable_diameter = 2.0 * float(self.cable_radius)
            if gap_width > cable_diameter + 0.030:
                return  # 夹爪间距远大于线缆直径，不抓取

            # 检查 2：线缆必须在夹爪夹持面正对区域内（放宽走廊半径）
            if not self._is_point_between_gripper_fingerpads(candidate_pos):
                return  # 线缆不在夹爪夹持区域

            self._flex_grasp_initial_z = float(candidate_pos[2])
            self._flex_grasp_active = True
            self._flex_grasp_ever_active = True  # 记录本 episode 中曾激活过
            self.attachment_enabled = True  # 触发 _attach_cable_end_to_gripper 调用
            self.attach_offset = self._gripper_clamp_center_offset()
        elif self._flex_grasp_active and not grip_closed:
            # 停用抓取
            self._flex_grasp_active = False
            self.attachment_enabled = False
            self._flex_grasp_vtx_idx = -1
            self._flex_grasp_initial_z = 0.0

    def _flex_grasp_contact_sides(self):
        """检测 flex 线缆与夹爪指尖的接触（利用 contact.flex 字段）。"""
        from robosuite.utils.dlo import physical_grasp as pg
        left_geoms, right_geoms = pg.get_gripper_fingerpad_geom_groups(self)
        left_set = set(left_geoms)
        right_set = set(right_geoms)
        left_count = 0
        right_count = 0
        for idx in range(self.sim.data.ncon):
            contact = self.sim.data.contact[idx]
            has_flex = contact.flex[0] >= 0 or contact.flex[1] >= 0
            if not has_flex:
                continue
            geom_ids = [g for g in (contact.geom1, contact.geom2) if g >= 0]
            contact_geoms = set()
            for gid in geom_ids:
                name = self.sim.model.geom_id2name(gid)
                if name is not None:
                    contact_geoms.add(name)
            if contact_geoms & left_set:
                left_count += 1
            if contact_geoms & right_set:
                right_count += 1
        return left_count, right_count

    def _flex_tail_follow_gripper(self, follow_gain=0.3):
        """让 flex 线缆自由端多个顶点渐进跟随夹爪。

        增益梯度：抓取点 → 自由端（增益从 follow_gain 线性衰减到 0）。
        锚点方向的顶点不移动（由锚点约束固定）。
        """
        grip_pos = self._get_gripper_site_position()
        clamp_offset = self._gripper_clamp_center_offset()
        target_world = grip_pos + clamp_offset + self.attach_offset

        adr = self._flex_vertadr
        nvert = self._flex_vertnum
        if adr is None or nvert is None or nvert < 2:
            return

        grasp_idx = self._flex_grasp_vtx_idx
        if grasp_idx < 0 or grasp_idx >= nvert:
            return

        # 判断自由端方向：比较首尾端点到当前抓取点的距离
        # 距离远的一端是自由端（未锚定的一端）
        points = self._get_cable_points()
        dist_to_start = abs(grasp_idx - 0)
        dist_to_end = abs(grasp_idx - (nvert - 1))
        if dist_to_end >= dist_to_start:
            # 自由端在尾部（idx 增大方向）
            free_indices = list(range(grasp_idx, nvert))
        else:
            # 自由端在头部（idx 减小方向）
            free_indices = list(range(grasp_idx, -1, -1))

        n_free = len(free_indices)
        if n_free < 1:
            return

        # 获取 flex body 变换
        try:
            obj_body_id = self.sim.model.body_name2id("flex_cable_container")
        except (KeyError, ValueError):
            try:
                obj_body_id = self.sim.model.body_name2id("object")
            except (KeyError, ValueError):
                return
        body_pos = self.sim.data.body_xpos[obj_body_id]
        body_rot = self.sim.data.body_xmat[obj_body_id].reshape(3, 3)

        spacing = getattr(self.cable_in_task, "flex_vertex_spacing", None) or 0.01
        rest_offset = (nvert - 1) * spacing / 2.0

        for i, vtx_idx in enumerate(free_indices):
            # 增益从 follow_gain 线性衰减到 0
            # i=0（抓取点）→ gain=follow_gain, i=n_free-1（自由端末尾）→ gain≈0
            gain = follow_gain * max(0.0, 1.0 - i / max(n_free - 1, 1))

            # 目标位置：抓取点目标 + 沿线缆方向偏移
            # 顶点 i 应该比抓取点低（更靠近原始桌面位置），形成自然悬垂曲线
            target_i = target_world.copy()

            # 转换到 body 局部坐标
            body_frame_pos = body_rot.T @ (target_i - body_pos)
            grid_rest_x = int(vtx_idx) * spacing - rest_offset
            target_qpos = body_frame_pos.copy()
            target_qpos[0] -= grid_rest_x

            # EMA 渐进跟随
            current_qpos = self.sim.data.qpos[adr + int(vtx_idx) * 3: adr + int(vtx_idx) * 3 + 3].copy()
            new_qpos = current_qpos + gain * (target_qpos - current_qpos)
            self.sim.data.qpos[adr + int(vtx_idx) * 3: adr + int(vtx_idx) * 3 + 3] = new_qpos

    def _dampen_cable_velocities(self, factor):
        """缩放线缆 root 和内部形状关节速度。"""
        if self.cable_root_joint is not None:
            joint_qvel = self.sim.data.get_joint_qvel(self.cable_root_joint).copy()
            joint_qvel[:] *= float(factor)
            self.sim.data.set_joint_qvel(self.cable_root_joint, joint_qvel)
        for joint_name in self.cable_shape_joint_names:
            try:
                qvel = self.sim.data.get_joint_qvel(joint_name)
                self.sim.data.set_joint_qvel(joint_name, np.asarray(qvel).copy() * float(factor))
            except Exception:
                continue

    # ------------------------------------------------------------------
    # Cable-point reading（委托给 CableInTask）
    # ------------------------------------------------------------------

    def _get_cable_points(self):
        """读取线缆所有关键点的 3D 位置（形状为 [N, 3]）。"""
        return self.cable_in_task.get_cable_points(self.sim)

    def _get_cable_end_pos(self):
        """返回线缆末端 body 的 3D 位置。"""
        return self.cable_in_task.get_cable_end_pos(self.sim)

    def _get_cable_start_pos(self):
        """返回线缆起始端 body 的 3D 位置。"""
        return self.cable_in_task.get_cable_start_pos(self.sim)

    def _get_active_grasp_endpoint_pos(self):
        """返回当前活跃抓取端点的 3D 位置。"""
        return self._get_active_grasp_body_pos()

    def _active_grasp_endpoint_body_id(self):
        """返回当前活跃抓取端点的 MuJoCo body ID。"""
        return self._active_grasp_body_id()

    def _get_active_grasp_body_pos(self):
        """返回当前活跃抓取 body 的 3D 位置。"""
        return self.sim.data.xpos[self._active_grasp_body_id()].copy()

    def _active_grasp_body_id(self):
        """返回当前活跃抓取 body 的 MuJoCo body ID。"""
        return self.sim.model.body_name2id(self._resolve_cable_name(self.active_grasp_body_name, "body"))

    def _set_active_grasp_endpoint(self, endpoint_name):
        """通过端点名称设置活跃抓取点。"""
        self._set_active_grasp_body(endpoint_name)

    def _set_active_grasp_point(self, point_idx):
        """设置活跃抓取点的索引（0=起点, N-1=终点, 中间值=中间可抓取点）。

        对于只有 2 个可抓取点的线缆模型（如 flex），内部点索引会被映射到最近的端点。
        """
        point_idx = int(point_idx)
        if point_idx < 0 or point_idx >= self.graspable_point_count:
            # 映射内部点到最近的端点
            point_idx = 0 if point_idx <= self.graspable_point_count // 2 else self.graspable_point_count - 1
        self.active_grasp_point_idx = point_idx
        self.active_grasp_body_name = self.grasp_point_to_body[point_idx]
        if self.active_grasp_body_name in self.grasp_endpoint_body_names:
            self.active_grasp_endpoint_index = self.grasp_endpoint_body_names.index(self.active_grasp_body_name)

    def _set_active_grasp_body(self, body_name):
        """通过 body 名称设置活跃抓取点（支持 "cable_start"/"cable_end" 别名）。

        查找逻辑：先直接查找 body_name，如果找不到则尝试端点别名映射。
        设置成功后同步更新 active_grasp_body_name 和 active_grasp_point_idx。
        """
        body_name = str(body_name)
        if body_name not in self.grasp_body_to_index:
            resolved = self._endpoint_alias_to_body_name(body_name)
            if resolved is not None and resolved in self.grasp_body_to_index:
                body_name = resolved
            else:
                raise ValueError(f"Unsupported grasp body: {body_name}")
        self.active_grasp_body_name = body_name
        self.active_grasp_point_idx = self.grasp_body_to_index[body_name]
        if body_name in self.grasp_endpoint_body_names:
            self.active_grasp_endpoint_index = self.grasp_endpoint_body_names.index(body_name)

    def _endpoint_alias_to_body_name(self, endpoint_name):
        """将端点别名解析为真实抓取 body 名称。"""
        endpoint_name = str(endpoint_name)
        if endpoint_name in self.grasp_body_to_index:
            return endpoint_name
        if not self.graspable_body_names:
            return None
        endpoint_aliases = {
            "cable_start": self.graspable_body_names[0],
            "cable_B0": self.graspable_body_names[0],
            "cable_end": self.graspable_body_names[-1],
        }
        return endpoint_aliases.get(endpoint_name)

    def _sync_active_mocap_to_endpoint(self):
        """将当前活跃的 mocap body 同步到对应线缆 body 的当前位置。"""
        self._sync_active_mocap_to_grasp_body()

    def _sync_active_mocap_to_grasp_body(self):
        """将活跃抓取点的 mocap body 移到对应的线缆 body 位置。

        为什么需要同步？mocap body 通过 weld 约束绑定线缆 body。
        如果不同步就激活约束，线缆会因为位置不匹配而产生瞬时力跳变。
        在 set_attachment_enabled(True) 和重置时调用。
        """
        self.sim.data.mocap_pos[self.mocap_ids[self.active_grasp_point_idx]] = self._get_active_grasp_body_pos()
        self.sim.data.mocap_quat[self.mocap_ids[self.active_grasp_point_idx]] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    def _sync_all_mocaps_to_endpoints(self):
        """将所有 mocap body 同步到对应的线缆 body 位置。"""
        self._sync_all_mocaps_to_grasp_bodies()

    def _sync_all_mocaps_to_grasp_bodies(self):
        """将所有 mocap body 同步到对应的线缆 body 位置。

        在重置时调用，确保所有 mocap body 都在线缆 body 的正确位置上。
        即使当前只附着一个端点，其他 mocap 也需要同步（为后续切换端点做准备）。
        """
        if not self.mocap_ids:
            return
        for idx, body_name in enumerate(self.graspable_body_names):
            if idx >= len(self.mocap_ids):
                break
            self.sim.data.mocap_pos[self.mocap_ids[idx]] = self.sim.data.xpos[
                self.sim.model.body_name2id(self._resolve_cable_name(body_name, "body"))
            ].copy()
            self.sim.data.mocap_quat[self.mocap_ids[idx]] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    def translate_cable_xy(self, delta_xy):
        """将线缆整体平移指定的 xy 偏移量（不改变形状）。"""
        self.cable_in_task.translate_cable_xy(self.sim, delta_xy)
        self._sync_all_mocaps_to_grasp_bodies()

    # ------------------------------------------------------------------
    # Gripper / contact helpers
    # ------------------------------------------------------------------

    def _get_gripper_site_position(self):
        """返回夹爪 grip_site 的 3D 位置（夹爪中心点）。"""
        arm = self.robots[0].arms[0]
        grip_site = self.robots[0].gripper[arm].important_sites["grip_site"]
        return np.array(self.sim.data.get_site_xpos(grip_site))

    def _gripper_fingerpad_geom_groups(self):
        """返回左右夹爪指尖的 geom 名称列表（用于碰撞检测）。"""
        return physical_grasp_utils.get_gripper_fingerpad_geom_groups(self)

    def _fingerpad_group_center(self, geom_names):
        """返回一侧 fingerpad collision geoms 的世界坐标中心。"""
        return physical_grasp_utils.get_fingerpad_group_center(self, geom_names)

    def _gripper_fingerpad_midpoint(self):
        """返回两侧 fingerpad 中心点的中点。"""
        return physical_grasp_utils.get_gripper_fingerpad_midpoint(self)

    def _fingerpad_gap_width(self):
        """返回左右 fingerpad 中心之间的距离。"""
        return physical_grasp_utils.get_fingerpad_gap_width(self)

    def _gripper_clamp_center_offset(self):
        """返回 fingerpad 夹持中心相对 grip site 的偏移。"""
        return physical_grasp_utils.get_gripper_clamp_center_offset(self)

    def _is_point_between_gripper_fingerpads(self, point):
        """判断目标点是否位于两侧夹爪夹持面之间。"""
        return physical_grasp_utils.is_point_between_gripper_fingerpads(
            self,
            point,
            max_distance_fallback=0.008,
        )

    def _cable_contact_geoms(self):
        """返回线缆的接触 geom 名称列表（用于物理抓取碰撞检测）。"""
        return physical_grasp_utils.get_cable_contact_geoms(self)

    def _count_contacts_between(self, geom_group, object_geoms):
        """统计两组 geom 之间的接触点数量。

        遍历 MuJoCo 的所有接触点（sim.data.ncon），检查每个接触的两个 geom
        是否分别属于两组。用于判断夹爪指尖是否接触线缆。
        """
        return physical_grasp_utils.count_contacts_between(self, geom_group, object_geoms)

    def _physical_grasp_contact_sides(self):
        """分别检测左右夹爪指尖与线缆的接触数量。

        返回 (left_count, right_count)。物理抓取成功需要两侧都有接触（夹住线缆）。
        """
        return physical_grasp_utils.get_physical_grasp_contact_sides(self)

    def _physical_grasp_contact_count(self):
        """返回夹爪指尖与线缆的总接触点数量。"""
        left_count, right_count = self._physical_grasp_contact_sides()
        return int(left_count + right_count)

    def _physical_grasp_lift_height(self):
        """返回当前活跃抓取 body 相对于重置时初始高度的抬起量。"""
        return float(self._get_physical_grasp_point_pos()[2] - self.physical_grasp_initial_endpoint_z)

    def _get_physical_grasp_point_pos(self):
        """返回物理抓取当前实际抓持点的位置。"""
        if self._is_flex_cable and self._physical_grasp_point_idx >= 0 and self._flex_vertnum is not None:
            point_idx = int(np.clip(self._physical_grasp_point_idx, 0, self._flex_vertnum - 1))
            return np.asarray(self.sim.data.flexvert_xpos[point_idx], dtype=float).copy()
        # composite/rmb 线缆：使用 _physical_grasp_point_idx 索引 cable_points
        if not self._is_flex_cable and self._physical_grasp_point_idx >= 0:
            points = self._get_cable_points()
            point_idx = int(np.clip(self._physical_grasp_point_idx, 0, len(points) - 1))
            return np.asarray(points[point_idx], dtype=float).copy()
        return self._get_active_grasp_body_pos()

    def _attachment_eq_active_count(self):
        """返回当前激活的 weld 约束数量。为 0 表示没有附着（纯物理抓取状态）。"""
        count = int(sum(int(self.sim.data.eq_active[eq_id]) for eq_id in self.grasp_eq_ids if eq_id is not None))
        # flex 端点 weld
        if self._is_flex_cable and hasattr(self, '_flex_end_weld_eq_id') and self._flex_end_weld_eq_id is not None:
            count += int(self.sim.data.eq_active[self._flex_end_weld_eq_id])
        return count

    def _table_contact_metrics(self, points):
        """计算线缆与桌面的接触质量指标。

        判断每个关键点是否"在桌面上"的条件（同时满足）：
        1. 没有严重穿透桌面（z >= 桌面高度 - 穿透容忍度）
        2. 接近线缆中心线高度（|z - centerline_z| <= 接触高度容忍度）

        返回的指标包括：
        - table_contact_ratio: 在桌面上的关键点比例
        - cable_on_table: 是否达到成功阈值（95% 以上关键点在桌面上）
        - max_keypoint_height_above_table: 最高点超出桌面的距离（线缆翘起）
        - max_keypoint_depth_below_table: 最低点低于桌面的距离（穿透）
        """
        z = np.asarray(points, dtype=float)[:, 2]
        table_top_z = float(self.table_top_z)
        centerline_z = float(self.cable_centerline_z)
        centerline_abs_error = np.abs(z - centerline_z)
        not_deep_below_table = z >= table_top_z - self.table_penetration_tolerance
        near_table_centerline = centerline_abs_error <= self.table_contact_z_tolerance
        contact_mask = np.logical_and(not_deep_below_table, near_table_centerline)
        return {
            "table_height_reference": table_top_z,
            "table_top_z": table_top_z,
            "cable_centerline_z": centerline_z,
            "max_keypoint_height_above_table": float(max(0.0, np.max(z - table_top_z))),
            "max_keypoint_depth_below_table": float(max(0.0, np.max(table_top_z - z))),
            "max_keypoint_centerline_z_error": float(np.max(centerline_abs_error)),
            "table_contact_ratio": float(np.mean(contact_mask)),
            "cable_on_table": bool(np.mean(contact_mask) >= self.success_table_contact_ratio_threshold),
        }

    def _physical_grasp_success(self):
        """判断物理抓取是否成功。

        成功条件（全部满足）：
        1. 左右夹爪都有接触（线缆被夹住）
        2. 线缆被抬起超过阈值（0.03m）
        3. 没有激活的 weld 约束（纯物理抓取，非 attachment 模式）
        """
        return bool(self.physical_grasp_lift_ready())

    def physical_grasp_ready(self):
        """判断是否形成了真实的双侧物理夹持。"""
        left_count, right_count = self._physical_grasp_contact_sides()
        cable_diameter = 2.0 * float(self.cable_radius)
        return bool(
            self.grasp_mode == "physical"
            and left_count > 0
            and right_count > 0
            and self._fingerpad_gap_width() <= cable_diameter + 0.020
            and self._is_point_between_gripper_fingerpads(self._get_physical_grasp_point_pos())
        )

    def physical_grasp_lift_ready(self):
        """判断物理夹持后的抬升是否达到成功阈值。"""
        return bool(
            self.physical_grasp_ready()
            and self._physical_grasp_lift_height() >= self.physical_grasp_lift_threshold
            and self._attachment_eq_active_count() == 0
        )

    # ------------------------------------------------------------------
    # Name resolution
    # ------------------------------------------------------------------

    def _resolve_cable_name(self, base_name, kind):
        """解析线缆 body/site 名称（处理命名前缀兼容问题）。"""
        return self.cable_in_task.resolve_cable_name(self.sim, base_name, kind)

    @property
    def _visualizations(self):
        vis_settings = super()._visualizations
        vis_settings.add("grippers")
        return vis_settings


# 向后兼容别名
CableBaseEnv = BaseDLOEnv
