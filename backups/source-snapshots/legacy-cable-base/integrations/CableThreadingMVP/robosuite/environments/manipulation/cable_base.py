"""Historical cable-base snapshot.

CableBaseEnv — 所有线缆操作任务的基类。

本模块提供线缆操作任务的共享基础设施，包括：
  - MuJoCo 模型加载：将桌面、机器人、线缆（rmb/flex/segmented 等）组装成 ManipulationTask
  - 抓取系统：支持两种模式 —— attachment（mocap+weld 约束跟随夹爪）和 physical（物理接触抓取）
  - 重置逻辑：rmb 线缆使用 joint 采样+形状噪声，其他线缆使用 placement_initializer 采样后重力沉降
  - 观测注册：自动注册 cable_points、target_line、attachment_state 等观测
  - 桌面接触度量：用于判断线缆是否平放在桌面上

子类（CableStraighten、CableMoveToTarget 等）继承本类后只需覆写 reward()、_check_success() 和 _compute_metrics()。
"""

import xml.etree.ElementTree as ET

import numpy as np

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import cable_object_factory
from robosuite.models.objects.xml_objects import FlexCableObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.dlo.cable_metrics import gripper_to_cable_distance
from robosuite.utils.mjcf_utils import new_body, new_site, xml_path_completion
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler


class CableBaseEnv(ManipulationEnv):
    """所有线缆操作任务的共享基类。

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
        self.cable_model = cable_model
        self.grasp_mode = str(grasp_mode)
        self.target_line_visible = bool(target_line_visible)
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        # 桌面位于 z=0.8m 高度，所有线缆位置计算以此为基准
        self.table_offset = np.array((0.0, 0.0, 0.8))
        self.table_top_z = float(self.table_offset[2])
        # 用 cable_object_factory 创建一个临时 probe 对象来读取线缆的物理参数
        # （半径、可抓取 body 列表、参考点类型等），避免在 __init__ 阶段加载完整模型
        cable_probe = cable_object_factory(self.cable_model, name="cable_probe")
        self.cable_point_reference_kind = cable_probe.point_reference_kind  # 关键点类型: "flex"/"site"/"body"
        self.cable_point_reference_names = list(cable_probe.point_reference_names)
        self.cable_radius = float(cable_probe.cable_radius)
        self.cable_length = float(getattr(cable_probe, "cable_length", 0.48))
        self.cable_clearance = float(cable_probe.tabletop_centerline_offset) if self.cable_model != "rmb" else 0.002
        self.cable_tabletop_offset = float(cable_probe.tabletop_centerline_offset)
        # 线缆中心线高度 = 桌面高度 + 线缆半径 + 离桌间隙
        self.cable_centerline_z = self.table_top_z + self.cable_radius + self.cable_clearance
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        self.use_object_obs = use_object_obs
        self.placement_initializer = placement_initializer
        # ---- 重置采样参数 ----
        # rmb 线缆重置时，根节点 xy 均匀采样中心和范围；yaw 控制初始朝向；shape 控制形状扰动
        self.reset_xy_center = np.array([-0.22, 0.06], dtype=float)
        self.reset_xy_range = 0.06
        self.reset_yaw_range = 0.8
        self.reset_shape_wave_scale = 0.03
        self.reset_shape_noise_scale = 0.005
        self.reset_shape_noise_clip = 0.06
        self.reset_resample_attempts = 128       # 重采样最大尝试次数（确保线缆不超出桌面边界）
        self.reset_centerline_min_z = self.cable_centerline_z
        # ---- 成功判定阈值 ----
        self.success_table_contact_ratio_threshold = 0.95  # 关键点中至少 95% 需在桌面上才算 "cable_on_table"
        self.table_contact_z_tolerance = 0.025             # 关键点 z 与中心线的最大偏差（判断是否接触桌面）
        self.table_penetration_tolerance = 0.02            # 关键点允许的最大穿透深度

        # 目标线段起点/终点（straighten 任务的对齐目标）
        # 动态计算：目标线段长度 = 85% 线缆长度，确保端点可达
        _half_target = 0.425 * self.cable_length
        self.target_start = np.array([-_half_target, 0.00, self.cable_centerline_z])
        self.target_end = np.array([_half_target, 0.00, self.cable_centerline_z])
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
        self._attach_max_distance = 0.012    # grip site 到目标点的最大距离，避免隔空吸附
        self._attach_between_fingers_distance = 0.030  # fingerpad box corridor，仍需同时满足 grip site 近距离
        self._attach_finger_axis_margin = -0.50        # collision box 中心不等于真实夹持面，允许盒体厚度外延
        self.grasp_endpoint_body_names = ("cable_B0", "cable_end")  # 默认端点 body 名称
        self.graspable_body_names = list(cable_probe.graspable_body_names)  # 可抓取的 body 列表
        self.graspable_point_count = int(cable_probe.graspable_point_count)
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
        3. 通过 cable_object_factory 创建线缆对象（rmb/flex/segmented 等）
        4. 根据线缆类型分两条路径构建模型：
           - flex 电缆：解析 flex_cable.xml，嵌入 extension 插件和 flexcomp body
           - 非 flex 电缆：使用 placement_initializer 放置线缆
        5. 为每个可抓取 body 创建 mocap body + weld 约束
        """
        super()._load_model()

        # 将机器人放置在桌面边缘
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        arena.set_origin([0, 0, 0])

        # 创建正式的线缆对象（不同于 __init__ 中的 probe）
        self.cable = cable_object_factory(self.cable_model, name="cable")
        self._is_flex_cable = isinstance(self.cable, FlexCableObject)

        # 更新线缆物理参数（使用正式对象覆盖 probe 的值）
        self.cable_point_reference_kind = self.cable.point_reference_kind
        self.cable_point_reference_names = list(self.cable.point_reference_names)
        self.cable_radius = float(self.cable.cable_radius)
        self.cable_tabletop_offset = float(self.cable.tabletop_centerline_offset)
        self.cable_centerline_z = self.table_top_z + self.cable_radius + self.cable_clearance

        if self._is_flex_cable:
            # Flex 电缆：动态计算目标线段（100% 线缆长度，确保端点可达且需要拉直）
            _half_target = 0.475 * float(self.cable.cable_length)
            self.target_start = np.array([-_half_target, 0.00, self.cable_centerline_z])
            self.target_end = np.array([_half_target, 0.00, self.cable_centerline_z])
            # Flexcomp 电缆：MuJoCo 会为每个 flex 顶点自动生成 body（flex_cable_0..39）
            # 利用端点 body（flex_cable_0, flex_cable_39）实现 mocap+weld 抓取机制
            self.cable_root_joint = None
            self.graspable_body_names = list(self.cable.graspable_body_names)
            self.graspable_point_count = int(self.cable.graspable_point_count)
            self.grasp_point_to_body = {
                point_idx: self.cable.body_name_for_point_idx(point_idx)
                for point_idx in range(self.graspable_point_count)
            }
            self.grasp_body_to_index = {body_name: idx for idx, body_name in enumerate(self.graspable_body_names)}
            # 设置 flex 端点名称映射，使 "cable_start"/"cable_end" 能正确解析为 flex body 名
            if self.graspable_point_count >= 2:
                self.grasp_endpoint_body_names = (self.graspable_body_names[0], self.graspable_body_names[-1])
            self.mocap_body_names = [f"flex_grasp_mocap_{idx:02d}" for idx in range(self.graspable_point_count)]
            self.grasp_eq_names = [f"flex_grasp_{idx:02d}_mocap_weld" for idx in range(self.graspable_point_count)]
            self.num_cable_points = int(self.cable.flex_vertex_count)
            self._flex_comp_name = self.cable.point_reference_names[0]
            if self.graspable_point_count:
                self.active_grasp_point_idx = self.graspable_point_count - 1
                self.active_grasp_body_name = self.grasp_point_to_body[self.active_grasp_point_idx]

            # 在 arena 中创建 mocap body
            for mocap_name in self.mocap_body_names:
                arena.worldbody.append(new_body(name=mocap_name, mocap="true", pos=(0.0, 0.0, self.cable_centerline_z)))

            # 解析 flex_cable.xml 并嵌入 arena
            flex_xml_path = xml_path_completion("objects/dlo/flex_cable.xml")
            flex_tree = ET.parse(flex_xml_path)
            flex_root = flex_tree.getroot()
            # 复制 extension 插件内容到 arena.extension（避免嵌套 extension 元素）
            for ext in flex_root.findall("extension"):
                for plugin in ext.findall("plugin"):
                    arena.extension.append(plugin)
            # 嵌入 flexcomp body 到 arena worldbody
            # 需要保持完整的 body 层级：外层 body → object → flex_cable_0..39
            flex_worldbody = flex_root.find("worldbody")
            if flex_worldbody is not None:
                for outer_body in flex_worldbody.findall("body"):
                    # 给无名外层 body 添加名称，避免 generate_id_mappings 出错
                    if not outer_body.get("name"):
                        outer_body.set("name", "flex_cable_container")
                    # 将 flex 电缆放置在桌面上方（否则默认在世界原点 z=0）
                    outer_body.set("pos", f"{self.table_offset[0]} {self.table_offset[1]} {self.cable_centerline_z}")
                    arena.worldbody.append(outer_body)

            self.model = ManipulationTask(
                mujoco_arena=arena,
                mujoco_robots=[robot.robot_model for robot in self.robots],
                mujoco_objects=None,
            )

            # 添加 weld 约束：mocap body → flex 端点 body
            for mocap_name, eq_name, body_name in zip(self.mocap_body_names, self.grasp_eq_names, self.graspable_body_names):
                self.model.equality.append(
                    ET.Element(
                        "weld",
                        attrib={
                            "name": eq_name,
                            "body1": mocap_name,
                            "body2": body_name,
                            "relpose": "0 0 0 1 0 0 0",
                            "solref": "0.02 1",
                            "solimp": "0.95 0.99 0.001",
                        },
                    )
                )
        else:
            # ---- 非 flex 电缆路径（rmb, segmented, composite, gemini） ----
            # 这些电缆由多个刚性 body 组成，通过 joint 连接；cable_root_joint 是根节点的自由关节
            self.cable_root_joint = self.cable.joints[-1]
            self.graspable_body_names = list(self.cable.graspable_body_names)
            self.graspable_point_count = int(self.cable.graspable_point_count)
            self.grasp_point_to_body = {
                point_idx: self.cable.body_name_for_point_idx(point_idx)
                for point_idx in range(self.graspable_point_count)
            }
            self.grasp_body_to_index = {body_name: idx for idx, body_name in enumerate(self.graspable_body_names)}
            # 为每个可抓取 body 创建对应的 mocap body 和 weld 约束名称
            self.mocap_body_names = [f"cable_grasp_mocap_{idx:02d}" for idx in range(self.graspable_point_count)]
            self.grasp_eq_names = [f"cable_grasp_{idx:02d}_mocap_weld" for idx in range(self.graspable_point_count)]
            self.num_cable_points = len(self.cable_point_reference_names)
            if self.graspable_point_count:
                self.active_grasp_point_idx = self.graspable_point_count - 1
                self.active_grasp_body_name = self.grasp_point_to_body[self.active_grasp_point_idx]
            # 在 arena 中创建 mocap body（mocap="true" 表示该 body 由 MuJoCo mocap 数据直接驱动，不受物理力影响）
            for mocap_name in self.mocap_body_names:
                arena.worldbody.append(new_body(name=mocap_name, mocap="true", pos=(0.0, 0.0, self.cable_centerline_z)))

            # 设置线缆放置采样器（重置时用于随机化线缆初始位置）
            if self.placement_initializer is not None:
                self.placement_initializer.reset()
                self.placement_initializer.add_objects(self.cable)
            else:
                self.placement_initializer = UniformRandomSampler(
                    name="CableSampler",
                    mujoco_objects=self.cable,
                    x_range=(-0.02, 0.02),
                    y_range=(-0.02, 0.02),
                    rotation=(-0.25, 0.25),
                    rotation_axis="z",
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=True,
                    reference_pos=self.table_offset,
                    z_offset=self.cable_centerline_z - self.table_offset[2],
                    rng=self.rng,
                )

            # 将 arena（桌面）+ 机器人 + 线缆组装成完整的 MuJoCo 任务模型
            self.model = ManipulationTask(
                mujoco_arena=arena,
                mujoco_robots=[robot.robot_model for robot in self.robots],
                mujoco_objects=self.cable,
            )
            # 为每个可抓取 body 创建 weld 约束：mocap body <--weld--> 线缆 body
            # 重置时 eq_active=0（不生效），调用 set_attachment_enabled(True) 后激活
            for mocap_name, eq_name, body_name in zip(self.mocap_body_names, self.grasp_eq_names, self.graspable_body_names):
                self.model.equality.append(
                    ET.Element(
                        "weld",
                        attrib={
                            "name": eq_name,
                            "body1": mocap_name,
                            "body2": self._xml_grasp_body_name(body_name),
                            "relpose": "0 0 0 1 0 0 0",
                            "solref": "0.02 1",
                            "solimp": "0.95 0.99 0.001",
                        },
                    )
                )

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_references(self):
        """在 MuJoCo 模型加载完成后，缓存各类 ID 供运行时快速查询。

        根据线缆类型（flex/site/body）查找关键点的 MuJoCo ID，设置 mocap body ID、
        weld 约束 ID、端点 body ID 等。这些 ID 在 _get_cable_points() 等高频方法中使用。
        """
        super()._setup_references()

        # Enable offscreen rendering of cable geoms (group 0 is disabled by
        # default in robosuite's offscreen renderer, but cable geoms live in
        # group 0).  Move them to group 1 (visual) so they appear in camera
        # observations and recorded videos.
        for i in range(self.sim.model.ngeom):
            name = self.sim.model.geom(i).name
            if name and (name.startswith("cable_") or name.startswith("Flex")):
                self.sim.model.geom_group[i] = 1

        if self.cable_point_reference_kind == "flex":
            # Flexcomp 电缆：查找 flex 组件，读取顶点位置
            flex_name = self.cable_point_reference_names[0]
            flex_id = None
            for i in range(self.sim.model.nflex):
                adr = self.sim.model.name_flexadr[i]
                stored_name = self.sim.model.names[adr:].split(b'\x00')[0].decode('utf-8')
                if stored_name == flex_name or stored_name.endswith(flex_name):
                    flex_id = i
                    break
            if flex_id is None:
                raise KeyError(f"Flex component '{flex_name}' not found in model")
            self._flex_id = flex_id
            self._flex_vertadr = int(self.sim.model.flex_vertadr[flex_id])
            self._flex_vertnum = int(self.sim.model.flex_vertnum[flex_id])
            self.cable_point_ids = list(range(self._flex_vertnum))
            self.num_cable_points = self._flex_vertnum
            # 设置端点 body ID（flex_cable_0, flex_cable_39）
            self.cable_start_body_id = self.sim.model.body_name2id(self.graspable_body_names[0])
            self.cable_end_body_id = self.sim.model.body_name2id(self.graspable_body_names[-1])
            # 设置 mocap 和 weld 约束 ID
            self.mocap_body_ids = []
            self.mocap_ids = []
            for mocap_name in self.mocap_body_names:
                body_id = self.sim.model.body_name2id(mocap_name)
                self.mocap_body_ids.append(body_id)
                self.mocap_ids.append(self.sim.model.body_mocapid[body_id])
            self.cable_shape_joint_names = []
            self.grasp_eq_ids = [None for _ in self.grasp_eq_names]
            for eq_id in range(self.sim.model.neq):
                eq_name = self.sim.model.equality(eq_id).name
                for idx, expected_name in enumerate(self.grasp_eq_names):
                    if eq_name == expected_name:
                        self.grasp_eq_ids[idx] = eq_id
        elif self.cable_point_reference_kind == "site":
            self.cable_point_ids = [self.sim.model.site_name2id(self._resolve_cable_name(name, "site")) for name in self.cable_point_reference_names]
            self._flex_id = None
        elif self.cable_point_reference_kind == "body":
            self.cable_point_ids = [self.sim.model.body_name2id(self._resolve_cable_name(name, "body")) for name in self.cable_point_reference_names]
            self._flex_id = None
        else:
            raise ValueError(f"Unsupported cable point reference kind: {self.cable_point_reference_kind}")

        if self._flex_id is None:
            # Non-flex cable: set up body IDs, mocap, grasp equality
            self.num_cable_points = len(self.cable_point_ids)
            start_body_name = self.graspable_body_names[0] if self.graspable_body_names else self.cable_point_reference_names[0]
            end_body_name = self.graspable_body_names[-1] if self.graspable_body_names else self.cable_point_reference_names[-1]
            self.cable_start_body_id = self.sim.model.body_name2id(self._resolve_cable_name(start_body_name, "body"))
            self.cable_end_body_id = self.sim.model.body_name2id(self._resolve_cable_name(end_body_name, "body"))
            self.mocap_body_ids = []
            self.mocap_ids = []
            for mocap_name in self.mocap_body_names:
                body_id = self.sim.model.body_name2id(mocap_name)
                self.mocap_body_ids.append(body_id)
                self.mocap_ids.append(self.sim.model.body_mocapid[body_id])
            self.cable_shape_joint_names = [
                name
                for name in self.sim.model.joint_names
                if name and name.startswith(self.cable.naming_prefix) and "_J" in name
            ]
            self.grasp_eq_ids = [None for _ in self.grasp_eq_names]
            for eq_id in range(self.sim.model.neq):
                eq_name = self.sim.model.equality(eq_id).name
                for idx, expected_name in enumerate(self.grasp_eq_names):
                    if eq_name == expected_name:
                        self.grasp_eq_ids[idx] = eq_id

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

        根据线缆类型走不同路径：
        1. flex 电缆：父类重置后直接 sim.forward() 触发重力沉降，然后同步 mocap
        2. rmb 电缆：通过 joint 采样位置+形状噪声，计算提升量确保线缆在桌面上方
        3. 其他电缆：使用 placement_initializer 随机放置，然后 drop_cable_to_table 做物理沉降

        三条路径最后都会：禁用所有 weld 约束 → 同步 mocap 到线缆位置 → 记录初始几何信息
        """
        super()._reset_internal()
        # 重置时禁用所有抓取约束
        self.attachment_enabled = False
        for eq_id in self.grasp_eq_ids:
            if eq_id is not None:
                self.sim.data.eq_active[eq_id] = 0

        if self._is_flex_cable:
            # Flexcomp 电缆：设置初始形状后重力沉降，同步 mocap 到端点位置
            self._apply_flex_initial_shape()
            self.sim.forward()
            self._sync_all_mocaps_to_grasp_bodies()
            self.last_reset_summary = {
                "cable_start_pos": self._get_cable_start_pos().copy(),
                "cable_end_pos": self._get_cable_end_pos().copy(),
                "min_centerline_z": float(np.min(self._get_cable_points()[:, 2])),
                "deterministic_reset": bool(self.deterministic_reset),
                "grasp_mode": self.grasp_mode,
            }
            self._record_initial_geometry()
            return

        if self.cable_model == "rmb":
            # ---- rmb 电缆重置路径 ----
            # rmb 是刚性链模型，通过设置根节点 joint (位置+姿态) + 各段形状关节的噪声来初始化
            if self.deterministic_reset:
                # 确定性重置：固定位置，无随机扰动（用于调试/测试）
                root_pos = np.array([self.reset_xy_center[0], self.reset_xy_center[1], self.reset_centerline_min_z], dtype=float)
                root_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
                shape_noise = self._default_rmb_shape_noise()
            else:
                # 随机重置：采样位置、朝向和形状噪声（可能多次重采样确保不超出桌面）
                root_pos, root_quat, shape_noise = self._sample_rmb_reset_state()
            root_pos = self._apply_rmb_reset_state(root_pos, root_quat, shape_noise)
            self._sync_all_mocaps_to_grasp_bodies()
            self.last_reset_summary = {
                "root_pos": root_pos.copy(),
                "root_quat": root_quat.copy(),
                "cable_start_pos": self._get_cable_start_pos().copy(),
                "shape_noise_l2": float(np.linalg.norm(shape_noise)),
                "min_centerline_z": float(np.min(self._get_cable_points()[:, 2])),
                "lift_z": float(root_pos[2] - self.reset_centerline_min_z),
                "cable_end_pos": self._get_cable_end_pos().copy(),
                "endpoint_span_xy": float(np.linalg.norm(self._get_cable_end_pos()[:2] - self._get_cable_start_pos()[:2])),
                "deterministic_reset": bool(self.deterministic_reset),
                "grasp_mode": self.grasp_mode,
            }
            self.physical_grasp_initial_endpoint_z = float(self._get_active_grasp_body_pos()[2])
            self._record_initial_geometry()
            return

        # ---- 非 rmb 非 flex 电缆重置路径（segmented, composite 等） ----
        if not self.deterministic_reset:
            # 使用 placement_initializer 随机放置线缆
            object_placements = self.placement_initializer.sample()
            for obj_pos, obj_quat, obj in object_placements.values():
                joint_name = self.cable_root_joint if obj is self.cable else obj.joints[0]
                self.sim.data.set_joint_qpos(joint_name, np.concatenate([obj_pos, obj_quat]))
                self.sim.data.set_joint_qvel(joint_name, np.zeros(6, dtype=float))
        elif self.cable_model != "rmb":
            yaw = 0.45
            root_pos = np.array([self.reset_xy_center[0], self.reset_xy_center[1] + 0.10, self.cable_centerline_z], dtype=float)
            root_quat = np.array([np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)], dtype=float)
            self.sim.data.set_joint_qpos(self.cable_root_joint, np.concatenate([root_pos, root_quat]))
            self.sim.data.set_joint_qvel(self.cable_root_joint, np.zeros(6, dtype=float))

        # 非 rmb 且非确定性重置时，做物理沉降让线缆自然落到桌面上
        if self.cable_model != "rmb" and not self.deterministic_reset:
            self._drop_cable_to_table()
        self._sync_all_mocaps_to_grasp_bodies()
        self.physical_grasp_initial_endpoint_z = float(self._get_active_grasp_body_pos()[2])
        self._record_initial_geometry()

    def _drop_cable_to_table(self, max_steps=1500, settle_vel_threshold=0.01, settle_steps_required=50):
        """让线缆在重力下自然落到桌面上并等待静止。

        原理：不断调用 sim.step() 推进物理仿真，监测线缆根节点的 z 轴速度。
        当速度连续 settle_steps_required 步低于阈值时认为线缆已静止。
        用于非 rmb 电缆的重置，确保线缆平放在桌面上。
        """
        self.sim.forward()
        settle_count = 0
        for _ in range(max_steps):
            self.sim.step()
            cable_qvel = self.sim.data.get_joint_qvel(self.cable_root_joint)
            z_vel = float(np.abs(cable_qvel[2]))
            if z_vel < settle_vel_threshold:
                settle_count += 1
                if settle_count >= settle_steps_required:
                    break
            else:
                settle_count = 0

    def _default_rmb_shape_noise(self):
        """生成确定性重置用的默认形状噪声。

        使用正弦波 + 锥形缩放生成平滑的弯曲形状：
        - wave: 水平面上的正弦弯曲（模拟自然弯曲）
        - vertical: 垂直面上的微小起伏（模拟线缆不完全平放）
        - taper: 从起点到终点逐渐增大的缩放系数（起点固定，终点更自由）
        - _J0_ 关节控制垂直方向，_J1_ 关节控制水平方向
        """
        joint_count = len(self.cable_shape_joint_names)
        shape_noise = np.zeros(joint_count, dtype=float)
        if not joint_count:
            return shape_noise
        t = np.linspace(0.0, 1.0, joint_count)
        wave = self.reset_shape_wave_scale * np.sin(np.pi * t)
        vertical = 0.2 * self.reset_shape_wave_scale * np.cos(np.pi * t)
        taper = 0.25 + 0.75 * t
        shape_noise = np.clip(taper * wave, -self.reset_shape_noise_clip, self.reset_shape_noise_clip)
        for idx, name in enumerate(self.cable_shape_joint_names):
            if "_J0_" in name:
                shape_noise[idx] = np.clip(
                    taper[idx] * vertical[idx],
                    -self.reset_shape_noise_clip,
                    self.reset_shape_noise_clip,
                )
        return shape_noise

    def _apply_rmb_reset_state(self, root_pos, root_quat, shape_noise):
        """将采样的 rmb 重置状态应用到 MuJoCo 仿真中。

        流程：设置根节点 joint → 设置各形状关节噪声 → sim.forward() 计算正运动学
        → 检查最低点是否在桌面上方 → 如果不够高则提升根节点
        返回最终的 root_pos（可能已被提升）。
        """
        root_pos = np.asarray(root_pos, dtype=float).copy()
        root_quat = np.asarray(root_quat, dtype=float)
        self.sim.data.set_joint_qpos(self.cable_root_joint, np.concatenate([root_pos, root_quat]))
        self.sim.data.set_joint_qvel(self.cable_root_joint, np.zeros(6, dtype=float))
        for joint_name, value in zip(self.cable_shape_joint_names, shape_noise):
            self.sim.data.set_joint_qpos(joint_name, float(value))
            self.sim.data.set_joint_qvel(joint_name, 0.0)
        self.sim.forward()

        # 计算线缆最低点与目标高度的差距，不足时整体提升
        min_centerline_z = float(np.min(self._get_cable_points()[:, 2]))
        lift_z = max(0.0, self.reset_centerline_min_z - min_centerline_z)
        if lift_z > 0.0:
            root_pos[2] += lift_z
            self.sim.data.set_joint_qpos(self.cable_root_joint, np.concatenate([root_pos, root_quat]))
            self.sim.forward()
        return root_pos

    def _sample_rmb_reset_state(self):
        """随机采样 rmb 电缆的重置状态（位置、朝向、形状噪声）。

        采样策略：
        1. 在 reset_xy_center 附近均匀采样根节点 xy 位置
        2. 均匀采样 yaw 朝向角，转为四元数
        3. 生成随机形状噪声：正弦波 + 随机相位 + 高斯抖动 + 锥形缩放
        4. 应用状态后检查所有关键点是否在桌面范围内
        5. 如果超出范围则重新采样（最多 reset_resample_attempts 次）
        6. 始终返回最后一次采样结果（即使不满足约束，作为兜底）
        """
        table_min_xy, table_max_xy = self._table_xy_bounds()
        safe_z = self.reset_centerline_min_z
        best = None

        for _ in range(self.reset_resample_attempts):
            root_xy = self.reset_xy_center + self.rng.uniform(
                low=-self.reset_xy_range,
                high=self.reset_xy_range,
                size=2,
            )
            root_yaw = float(self.rng.uniform(-self.reset_yaw_range, self.reset_yaw_range))
            root_pos = np.array([root_xy[0], root_xy[1], safe_z], dtype=float)
            root_quat = np.array([np.cos(0.5 * root_yaw), 0.0, 0.0, np.sin(0.5 * root_yaw)], dtype=float)

            # 生成随机形状噪声（比确定性版本多了随机相位和高斯抖动）
            joint_count = len(self.cable_shape_joint_names)
            shape_noise = np.zeros(joint_count, dtype=float)
            if joint_count:
                t = np.linspace(0.0, 1.0, joint_count)
                phase = float(self.rng.uniform(-np.pi, np.pi))
                wave = self.reset_shape_wave_scale * np.sin(2.0 * np.pi * t + phase)
                vertical = 0.35 * self.reset_shape_wave_scale * np.cos(np.pi * t + phase)
                taper = 0.25 + 0.75 * t
                jitter = self.rng.normal(loc=0.0, scale=self.reset_shape_noise_scale, size=joint_count)
                shape_noise = np.clip(taper * (wave + jitter), -self.reset_shape_noise_clip, self.reset_shape_noise_clip)
                for idx, name in enumerate(self.cable_shape_joint_names):
                    if "_J0_" in name:
                        shape_noise[idx] = np.clip(
                            taper[idx] * (vertical[idx] + 0.4 * jitter[idx]),
                            -self.reset_shape_noise_clip,
                            self.reset_shape_noise_clip,
                        )
            self.sim.data.set_joint_qpos(self.cable_root_joint, np.concatenate([root_pos, root_quat]))
            self.sim.data.set_joint_qvel(self.cable_root_joint, np.zeros(6, dtype=float))
            for joint_name, value in zip(self.cable_shape_joint_names, shape_noise):
                self.sim.data.set_joint_qpos(joint_name, float(value))
                self.sim.data.set_joint_qvel(joint_name, 0.0)
            self.sim.forward()

            # 提升线缆确保最低点在目标高度之上
            points = self._get_cable_points()
            min_centerline_z = float(np.min(points[:, 2]))
            lift_z = max(0.0, safe_z - min_centerline_z)
            if lift_z > 0.0:
                root_pos = root_pos.copy()
                root_pos[2] += lift_z
                self.sim.data.set_joint_qpos(self.cable_root_joint, np.concatenate([root_pos, root_quat]))
                self.sim.forward()
                points = self._get_cable_points()
            # 检查所有关键点是否在桌面安全区域内
            xy = points[:, :2]
            if np.all(xy[:, 0] >= table_min_xy[0]) and np.all(xy[:, 0] <= table_max_xy[0]) and np.all(
                xy[:, 1] >= table_min_xy[1]
            ) and np.all(xy[:, 1] <= table_max_xy[1]):
                best = (root_pos, root_quat, shape_noise)
                break

        if best is None:
            best = (root_pos, root_quat, shape_noise)
        return best

    def _table_xy_bounds(self):
        """返回桌面安全区域的 xy 边界（留出 8cm 边距防止线缆超出桌面）。"""
        half_x = 0.5 * float(self.table_full_size[0]) - 0.08
        half_y = 0.5 * float(self.table_full_size[1]) - 0.08
        return np.array([-half_x, -half_y], dtype=float), np.array([half_x, half_y], dtype=float)

    # ------------------------------------------------------------------
    # Pre-action
    # ------------------------------------------------------------------

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
            elif not close_enough:
                pass  # 距离太远，等待夹爪靠近

        if self.attachment_enabled:
            self._attach_cable_end_to_gripper()

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
        if bool(enabled) and self.grasp_mode == "physical":
            raise RuntimeError("set_attachment_enabled(True) is disabled when grasp_mode='physical'")

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
        self._flex_grasp_active = False
        self._flex_grasp_vtx_idx = -1
        self._flex_grasp_offset = np.zeros(3, dtype=float)
        self._flex_grasp_vtx_indices = np.zeros(0, dtype=int)
        self._flex_grasp_offsets = np.zeros((0, 3), dtype=float)
        self._flex_grasp_segment_local_targets = np.zeros((0, 3), dtype=float)
        self._flex_grasp_support_mode = "segment"
        self._attach_pending = False
        self._attach_pending_params = {}
        self.attachment_enabled = False
        for eq_id in self.grasp_eq_ids:
            if eq_id is not None:
                self.sim.data.eq_active[eq_id] = 0
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

        # flex 中点抓取：直接操控顶点
        if self._is_flex_cable and point_idx is not None:
            nvert = self._flex_vertnum
            if 0 < point_idx < nvert - 1:
                self._flex_grasp_active = True
                self._flex_grasp_vtx_idx = int(point_idx)
                grip_pos = self._get_gripper_site_position()
                self._flex_grasp_support_mode = "hanging_arc"
                self._flex_grasp_vtx_indices = np.arange(0, nvert, dtype=int)
                span_positions = np.asarray(self.sim.data.flexvert_xpos[self._flex_grasp_vtx_indices], dtype=float)
                self._flex_grasp_offsets = grip_pos[None, :] - span_positions
                vtx_pos = self.sim.data.flexvert_xpos[point_idx].copy()
                self._flex_grasp_offset = grip_pos - vtx_pos
                local_targets = []
                center = float(point_idx)
                spacing = float(self.cable_length) / max(1, nvert - 1)
                half_span = max(center, (nvert - 1) - center)
                for global_idx in self._flex_grasp_vtx_indices:
                    dx = (float(global_idx) - center) * spacing
                    normalized = abs(float(global_idx) - center) / max(half_span, 1.0)
                    dz = -0.18 * (normalized ** 1.35)
                    local_targets.append([dx, 0.0, dz])
                self._flex_grasp_segment_local_targets = np.asarray(local_targets, dtype=float)
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
        if self.active_grasp_point_idx >= 0:
            active_eq_id = self.grasp_eq_ids[self.active_grasp_point_idx]
            if active_eq_id is None:
                raise RuntimeError(f"No attachment equality found for grasp point {self.active_grasp_point_idx}")
            self._sync_active_mocap_to_grasp_body()
            self.sim.data.eq_active[active_eq_id] = 1
        self.sim.forward()

    def _is_gripper_closed(self, action):
        """检查夹爪是否已闭合。通过 action 的 gripper 分量判断。"""
        robot = self.robots[0]
        arm = robot.arms[0]
        gripper_dof = robot.gripper[arm].dof
        if gripper_dof == 0:
            return True
        # action 末尾是 gripper 控制值，正值=闭合
        gripper_val = float(action[-gripper_dof])
        return gripper_val > self._attach_grip_threshold

    def _is_gripper_close_enough(self):
        """检查夹爪与目标抓取点的距离是否在阈值内。"""
        if not self._attach_pending:
            return False
        params = self._attach_pending_params
        point_idx = params.get("point_idx")

        grip_pos = self._get_gripper_site_position()

        # 指定关键点抓取：flex 读取顶点；body 型线缆映射到对应 body。
        if point_idx is not None:
            if self._is_flex_cable:
                target_pos = self.sim.data.flexvert_xpos[point_idx]
            else:
                body_name = self.grasp_point_to_body.get(int(point_idx))
                if body_name is None:
                    return False
                body_id = self.sim.model.body_name2id(self._resolve_cable_name(body_name, "body"))
                target_pos = self.sim.data.xpos[body_id]
            dist = float(np.linalg.norm(grip_pos - target_pos))
            return dist < self._attach_max_distance and self._is_point_between_gripper_fingerpads(target_pos)

        # 端点抓取：用 graspable_body 的位置检查距离
        endpoint_name = params.get("endpoint_name")
        if endpoint_name is not None:
            body_name = self._endpoint_alias_to_body_name(endpoint_name)
            if body_name is not None:
                body_id = self.sim.model.body_name2id(self._resolve_cable_name(body_name, "body"))
                target_pos = self.sim.data.xpos[body_id]
                dist = float(np.linalg.norm(grip_pos - target_pos))
                return dist < self._attach_max_distance and self._is_point_between_gripper_fingerpads(target_pos)

        body_name = params.get("body_name")
        if body_name is not None:
            body_id = self.sim.model.body_name2id(self._resolve_cable_name(body_name, "body"))
            target_pos = self.sim.data.xpos[body_id]
            dist = float(np.linalg.norm(grip_pos - target_pos))
            return dist < self._attach_max_distance and self._is_point_between_gripper_fingerpads(target_pos)

        return False

    def _attach_cable_end_to_gripper(self):
        """每步更新抓取点位置使其跟随夹爪。

        flex 中点抓取：直接操控 flex vertex qpos，让弹性自然处理形变。
        端点抓取：通过 mocap + weld 约束平滑跟随。
        """
        grip_pos = self._get_gripper_site_position()

        # flex 顶点直接操控模式
        if self._flex_grasp_active:
            target_world = grip_pos + self.attach_offset
            adr = self._flex_vertadr

            # 获取 "object" body 的世界变换（flexcomp 所在的 body）
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

            for local_idx, vtx_idx in enumerate(self._flex_grasp_vtx_indices):
                local_target = self._flex_grasp_segment_local_targets[local_idx]
                target_world_i = target_world + grip_rot @ local_target
                local_offset = body_rot.T @ (target_world_i - body_pos)
                self.sim.data.qpos[adr + int(vtx_idx) * 3: adr + int(vtx_idx) * 3 + 3] = local_offset

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
    # Flex cable shape initialization
    # ------------------------------------------------------------------

    def _apply_flex_initial_shape(self):
        """设置 flex 电缆的初始形状（通过 qpos 偏移量）和位置。

        flexcomp 电缆的 DOFs 存储在 data.qpos 中，每个顶点 3 个 DOF（xyz 偏移量）。
        初始 qpos 全为 0，flexcomp grid 定义初始位置（沿 x 轴直线排列）。
        通过设置 qpos 偏移量可以将电缆变形为任意形状。

        固定模式：U 形（sin 波，amplitude=0.04m，沿 y 方向）
        随机模式：随机幅度（0.01-0.05m）+ 随机相位 + 随机频率
        """
        if not self._is_flex_cable:
            return
        nvert = self._flex_vertnum
        if nvert < 3:
            return

        # 1. 设置 XY 位置偏移（移动 flex 容器 body）
        try:
            body_id = self.sim.model.body_name2id("flex_cable_container")
        except ValueError:
            body_id = None
        if body_id is not None:
            if self.deterministic_reset:
                # 固定模式：偏移放置，使端点与目标有偏差，给 oracle 可修正的空间
                cx = float((self.target_start[0] + self.target_end[0]) / 2 + 0.05)
                cy = float((self.target_start[1] + self.target_end[1]) / 2 + 0.03)
            else:
                # 随机模式：基于目标中点 + 随机偏移
                cx = float((self.target_start[0] + self.target_end[0]) / 2 + self.rng.uniform(-self.reset_xy_range, self.reset_xy_range))
                cy = float((self.target_start[1] + self.target_end[1]) / 2 + self.rng.uniform(-self.reset_xy_range, self.reset_xy_range))
            self.sim.model.body_pos[body_id, 0] = cx
            self.sim.model.body_pos[body_id, 1] = cy
            self.sim.model.body_pos[body_id, 2] = self.cable_centerline_z

        # 2. 设置形状偏移（弯曲）：sin 波产生 U 形弯曲
        # 先清零所有形状偏移，避免 grid 初始化的残留偏移叠加
        self.sim.data.qpos[:nvert * 3] = 0.0

        if self.deterministic_reset:
            amplitude = 0.05
            phase = 0.0
            freq = 1.0
        else:
            amplitude = float(self.rng.uniform(0.03, 0.08))
            phase = float(self.rng.uniform(0, 2 * np.pi))
            freq = float(self.rng.choice([1.0, 1.5, 2.0]))

        for i in range(nvert):
            t = i / (nvert - 1)
            self.sim.data.qpos[i * 3 + 1] = amplitude * np.sin(freq * np.pi * t + phase)

    # ------------------------------------------------------------------
    # Cable-point reading
    # ------------------------------------------------------------------

    def _get_cable_points(self):
        """读取线缆所有关键点的 3D 位置（形状为 [N, 3]）。

        根据线缆类型从不同的 MuJoCo 数据源读取：
        - flex: 从 flexvert_xpos 读取 flex 顶点位置（连续内存，效率最高）
        - site: 从 site_xpos 读取 site 位置
        - body: 从 xpos 读取 body 位置
        这是整个系统最高频调用的方法之一，所有任务的 reward 和 metrics 都依赖它。
        """
        if self.cable_point_reference_kind == "flex":
            adr = self._flex_vertadr
            n = self._flex_vertnum
            return np.array(self.sim.data.flexvert_xpos[adr:adr + n].copy().reshape(n, 3))
        if self.cable_point_reference_kind == "site":
            return np.array([self.sim.data.site_xpos[point_id].copy() for point_id in self.cable_point_ids])
        return np.array([self.sim.data.xpos[point_id].copy() for point_id in self.cable_point_ids])

    def _get_cable_end_pos(self):
        """返回线缆末端 body 的 3D 位置。"""
        if self.cable_point_reference_kind == "flex":
            return self._get_cable_points()[-1].copy()
        return self.sim.data.xpos[self.cable_end_body_id].copy()

    def _get_cable_start_pos(self):
        """返回线缆起始端 body 的 3D 位置。"""
        if self.cable_point_reference_kind == "flex":
            return self._get_cable_points()[0].copy()
        return self.sim.data.xpos[self.cable_start_body_id].copy()

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
        for idx, body_name in enumerate(self.graspable_body_names):
            self.sim.data.mocap_pos[self.mocap_ids[idx]] = self.sim.data.xpos[
                self.sim.model.body_name2id(self._resolve_cable_name(body_name, "body"))
            ].copy()
            self.sim.data.mocap_quat[self.mocap_ids[idx]] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    def translate_cable_xy(self, delta_xy):
        """将线缆整体平移指定的 xy 偏移量（不改变形状）。

        用于专家脚本中的粗定位：先平移线缆到大致位置，再精细调整形状。
        直接修改根节点 joint 的位置（RMB/composite）或容器 body 位置（flex），
        并清除所有速度防止惯性导致偏移。
        """
        delta_xy = np.asarray(delta_xy, dtype=float)
        if delta_xy.shape != (2,):
            raise ValueError(f"delta_xy must have shape (2,), got {delta_xy.shape}")
        if self._is_flex_cable:
            # Flex 电缆：移动容器 body 的位置
            body_id = self.sim.model.body_name2id("flex_cable_container")
            self.sim.model.body_pos[body_id, 0] += float(delta_xy[0])
            self.sim.model.body_pos[body_id, 1] += float(delta_xy[1])
            self.sim.forward()
            self._sync_all_mocaps_to_grasp_bodies()
            return
        root_qpos = self.sim.data.get_joint_qpos(self.cable_root_joint).copy()
        root_qpos[0] += float(delta_xy[0])
        root_qpos[1] += float(delta_xy[1])
        self.sim.data.set_joint_qpos(self.cable_root_joint, root_qpos)
        root_qvel = self.sim.data.get_joint_qvel(self.cable_root_joint).copy()
        root_qvel[:] = 0.0
        self.sim.data.set_joint_qvel(self.cable_root_joint, root_qvel)
        for joint_name in self.cable_shape_joint_names:
            self.sim.data.set_joint_qvel(joint_name, 0.0)
        self.sim.forward()
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
        arm = self.robots[0].arms[0]
        gripper = self.robots[0].gripper[arm]
        return [
            list(gripper.important_geoms.get("left_fingerpad", [])),
            list(gripper.important_geoms.get("right_fingerpad", [])),
        ]

    def _fingerpad_group_center(self, geom_names):
        """返回一侧 fingerpad collision geoms 的世界坐标中心。"""
        positions = []
        for geom_name in geom_names:
            try:
                geom_id = self.sim.model.geom_name2id(geom_name)
            except (KeyError, ValueError):
                continue
            positions.append(np.asarray(self.sim.data.geom_xpos[geom_id], dtype=float))
        if not positions:
            return None
        return np.mean(np.asarray(positions, dtype=float), axis=0)

    def _gripper_fingerpad_midpoint(self):
        """返回两侧 fingerpad 中心点的中点。"""
        left_geoms, right_geoms = self._gripper_fingerpad_geom_groups()
        left_center = self._fingerpad_group_center(left_geoms)
        right_center = self._fingerpad_group_center(right_geoms)
        if left_center is None or right_center is None:
            return None
        return 0.5 * (left_center + right_center)

    def _gripper_clamp_center_offset(self):
        """返回 fingerpad 夹持中心相对 grip site 的偏移。"""
        midpoint = self._gripper_fingerpad_midpoint()
        if midpoint is None:
            return np.zeros(3, dtype=float)
        return np.asarray(midpoint, dtype=float) - self._get_gripper_site_position()

    def _is_point_between_gripper_fingerpads(self, point):
        """判断目标点是否位于两侧夹爪夹持面之间。"""
        left_geoms, right_geoms = self._gripper_fingerpad_geom_groups()
        left_center = self._fingerpad_group_center(left_geoms)
        right_center = self._fingerpad_group_center(right_geoms)
        if left_center is None or right_center is None:
            return False

        point = np.asarray(point, dtype=float)
        finger_axis = right_center - left_center
        axis_len_sq = float(np.dot(finger_axis, finger_axis))
        if axis_len_sq < 1e-10:
            return False

        projection = float(np.dot(point - left_center, finger_axis) / axis_len_sq)
        margin = float(self._attach_finger_axis_margin)
        if projection < margin or projection > 1.0 - margin:
            return False

        closest = left_center + projection * finger_axis
        distance_to_clamp_line = float(np.linalg.norm(point - closest))
        max_distance = max(float(self._attach_between_fingers_distance), float(self.cable_radius) + 0.008)
        return distance_to_clamp_line <= max_distance

    def _cable_contact_geoms(self):
        """返回线缆的接触 geom 名称列表（用于物理抓取碰撞检测）。"""
        geoms = list(getattr(self.cable, "contact_geoms", []))
        if geoms:
            return geoms
        # 如果线缆对象没有 contact_geoms 属性，通过名称前缀查找
        return [
            self.sim.model.geom_id2name(geom_id)
            for geom_id in range(self.sim.model.ngeom)
            if (self.sim.model.geom_id2name(geom_id) or "").startswith("cable_")
        ]

    def _count_contacts_between(self, geom_group, object_geoms):
        """统计两组 geom 之间的接触点数量。

        遍历 MuJoCo 的所有接触点（sim.data.ncon），检查每个接触的两个 geom
        是否分别属于两组。用于判断夹爪指尖是否接触线缆。
        """
        geom_group = set(geom_group)
        object_geoms = set(object_geoms)
        count = 0
        for idx in range(self.sim.data.ncon):
            contact = self.sim.data.contact[idx]
            # 跳过无效的 geom id（flex 电缆可能产生 id=-1 的接触）
            if contact.geom1 < 0 or contact.geom2 < 0:
                continue
            name1 = self.sim.model.geom_id2name(contact.geom1)
            name2 = self.sim.model.geom_id2name(contact.geom2)
            if (name1 in geom_group and name2 in object_geoms) or (name2 in geom_group and name1 in object_geoms):
                count += 1
        return count

    def _physical_grasp_contact_sides(self):
        """分别检测左右夹爪指尖与线缆的接触数量。

        返回 (left_count, right_count)。物理抓取成功需要两侧都有接触（夹住线缆）。
        """
        cable_geoms = self._cable_contact_geoms()
        left_geoms, right_geoms = self._gripper_fingerpad_geom_groups()
        left_count = self._count_contacts_between(left_geoms, cable_geoms)
        right_count = self._count_contacts_between(right_geoms, cable_geoms)
        return left_count, right_count

    def _physical_grasp_contact_count(self):
        """返回夹爪指尖与线缆的总接触点数量。"""
        left_count, right_count = self._physical_grasp_contact_sides()
        return int(left_count + right_count)

    def _physical_grasp_lift_height(self):
        """返回当前活跃抓取 body 相对于重置时初始高度的抬起量。"""
        return float(self._get_active_grasp_body_pos()[2] - self.physical_grasp_initial_endpoint_z)

    def _attachment_eq_active_count(self):
        """返回当前激活的 weld 约束数量。为 0 表示没有附着（纯物理抓取状态）。"""
        return int(sum(int(self.sim.data.eq_active[eq_id]) for eq_id in self.grasp_eq_ids if eq_id is not None))

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
        left_count, right_count = self._physical_grasp_contact_sides()
        return bool(
            left_count > 0
            and right_count > 0
            and self._physical_grasp_lift_height() >= self.physical_grasp_lift_threshold
            and self._attachment_eq_active_count() == 0
        )

    # ------------------------------------------------------------------
    # Name resolution
    # ------------------------------------------------------------------

    def _resolve_cable_name(self, base_name, kind):
        """解析线缆 body/site 名称（处理命名前缀兼容问题）。

        不同线缆模型的 body 命名规则不同（有的带前缀如 "cablec_B0"，有的不带如 "cable_B0"）。
        此方法先尝试原始名称，再尝试带前缀的名称，返回第一个在 MuJoCo 模型中找到的名称。
        """
        candidates = [base_name, f"{self.cable.naming_prefix}{base_name}"]
        for name in candidates:
            try:
                if kind == "site":
                    self.sim.model.site_name2id(name)
                elif kind == "body":
                    self.sim.model.body_name2id(name)
                else:
                    raise ValueError(f"Unsupported reference kind: {kind}")
                return name
            except (KeyError, ValueError):
                continue
        raise KeyError(f"Unable to resolve {kind} reference for cable point '{base_name}'")

    def _xml_grasp_body_name(self, body_name):
        """返回 body 在 XML 中的完整名称（用于构建 weld 约束的 body2 属性）。"""
        if str(body_name).startswith(self.cable.naming_prefix):
            return body_name
        if str(body_name).startswith("cablec_"):
            return f"{self.cable.naming_prefix}{body_name}"
        if str(body_name).startswith(("gem_", "flexrefc_")):
            return f"{self.cable.naming_prefix}{body_name}"
        if self.cable.exclude_from_prefixing(body_name):
            return body_name
        return f"{self.cable.naming_prefix}{body_name}"

    @property
    def _visualizations(self):
        vis_settings = super()._visualizations
        vis_settings.add("grippers")
        return vis_settings
