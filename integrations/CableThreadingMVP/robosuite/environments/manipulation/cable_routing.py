"""
CableRouting — 线缆路径规划任务。

任务目标：拖拽线缆一端沿桌面上的路径点（waypoints）依次移动，最终到达终点。
继承 BaseDLOEnv + HasWaypointsMixin，共享 DLO 基础设施和路径点跟踪能力。

核心概念：
  - 路径点序列：route_waypoints 定义了线缆端点需要依次经过的位置
  - 路径进度：current_waypoint_index 跟踪当前目标路径点（到 waypoint_tolerance 内时前进）
  - 附着机制：线缆第一个 keypoint 通过 kinematic 方式跟随夹爪（非物理约束）

成功条件：
  - 路径完成度 >= 100%（到达所有路径点）
  - 端点误差 < 0.03m（最终位置准确）
  - 路径误差 < 0.06m（线缆形状沿路径对齐）
"""

import xml.etree.ElementTree as ET

import numpy as np

from robosuite.environments.manipulation.cable_base import BaseDLOEnv
from robosuite.environments.manipulation.cable_in_task import CableInTask
from robosuite.models.arenas import TableArena
from robosuite.models.objects import cable_object_factory
from robosuite.models.objects.xml_objects import FlexCableObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.dlo.task_scene_utils import create_mocap_body, create_weld_constraint
from robosuite.utils.mjcf_utils import new_body, new_site, xml_path_completion
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler


class CableRouting(BaseDLOEnv):
    """
    Route a cable along a polyline by dragging one endpoint with a robot gripper.

    Inherits BaseDLOEnv for shared DLO infrastructure.
    Uses kinematic attachment between gripper and one cable endpoint.
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        base_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 0.005, 0.0001),
        route_waypoints=None,           # 路径点序列 [(x,y,z), ...]，为 None 时使用默认 L 形路径
        waypoint_tolerance=0.03,        # 路径点到达容差（米）
        attach_offset=None,             # 夹爪到线缆附着点的偏移，为 None 时使用线缆推荐值
        attachment_follow_gain=1.0,     # 每步跟随插值系数（1.0 = 硬跟随）
        attachment_velocity_damping=0.1, # 附着时的速度阻尼系数
        cable_model="rmb",
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=True,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=200,
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
        self.cable_model = cable_model
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0.0, 0.0, 0.8))
        self._cable_in_task = CableInTask(
            cable_model=cable_model,
            table_full_size=table_full_size,
            table_friction=table_friction,
            table_offset=self.table_offset,
        )
        self.cable_in_task = self._cable_in_task  # BaseDLOEnv 兼容别名
        self.cable_point_reference_kind = self._cable_in_task.cable_point_reference_kind
        self.cable_point_reference_names = list(self._cable_in_task.cable_point_reference_names)
        self.cable_radius = self._cable_in_task.cable_radius
        self.cable_clearance = self._cable_in_task.cable_clearance
        self.cable_tabletop_offset = self._cable_in_task.cable_tabletop_offset
        self.cable_centerline_z = self._cable_in_task.cable_centerline_z
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        self.use_object_obs = use_object_obs
        self.placement_initializer = placement_initializer
        self.waypoint_tolerance = waypoint_tolerance
        self.attachment_follow_gain = float(attachment_follow_gain)
        self.attachment_velocity_damping = float(attachment_velocity_damping)
        # 通过 CableInTask 的 probe 对象获取附着参数
        _probe = self._cable_in_task.cable or cable_object_factory(self.cable_model, name="cable_probe")
        if attach_offset is None:
            attach_offset = _probe.recommended_attach_offset
        self.attach_offset = np.array(attach_offset, dtype=float)
        self.cable_root_to_attached_site = np.asarray(_probe.attachment_root_offset, dtype=float)
        self.min_gripper_z = self.table_offset[2] + self.cable_tabletop_offset + abs(min(float(self.attach_offset[2]), 0.0)) + 0.01

        # 默认路径点：L 形路径（先向右，再向右上）
        if route_waypoints is None:
            route_waypoints = [
                (-0.24, -0.12, self.cable_centerline_z),
                (-0.08, -0.12, self.cable_centerline_z),
                (0.06, 0.02, self.cable_centerline_z),
                (0.18, 0.12, self.cable_centerline_z),
            ]
        self.route_waypoints = np.array(route_waypoints, dtype=float)

        self.num_cable_points = None
        self.current_waypoint_index = 0   # 当前目标路径点索引（每步检查是否到达）

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

    def reward(self, action=None):
        """计算路径规划任务的 reward。

        reward = 2 * route_completion - 2 * route_error - 1.5 * endpoint_error
                 - 0.5 * eef_to_target + 2 * success_bonus

        - route_completion: 路径完成度（0~1，到达的路径点比例）
        - route_error: 线缆关键点到最近路径段的平均距离（越小越好）
        - endpoint_error: 线缆端点到最终目标的距离
        - eef_to_target: 夹爪到当前目标路径点的距离（鼓励夹爪跟随路径）
        - success_bonus: 全部达标后给 +2 的奖励
        """
        metrics = self._compute_metrics()
        current_target = self._get_current_grip_target()
        eef_to_target = float(np.linalg.norm(self._get_gripper_site_position() - current_target))
        reward = 2.0 * metrics["route_completion"]
        reward -= 2.0 * metrics["route_error"]
        reward -= 1.5 * metrics["endpoint_error"]
        reward -= 0.5 * eef_to_target
        if metrics["success"]:
            reward += 2.0
        return self.reward_scale * reward

    def _load_model(self):
        """构建路径规划任务的 MuJoCo 模型。"""
        super()._load_model()

    def _create_arena(self):
        """创建带路径点标记的桌面 arena。"""
        arena = super()._create_arena()
        # 路径点可视化标记
        for idx, waypoint in enumerate(self.route_waypoints):
            arena.worldbody.append(
                new_site(
                    name=f"route_wp_{idx:02d}",
                    pos=waypoint,
                    size=(0.01,),
                    rgba=(0.1, 0.8, 0.2, 0.7) if idx < len(self.route_waypoints) - 1 else (0.9, 0.8, 0.1, 0.9),
                )
            )
        return arena

    def _setup_cable_scene(self, arena):
        """在 arena 中构建线缆场景。"""

        # 通过 CableInTask 创建线缆对象并嵌入 arena
        scene_result = self._cable_in_task.embed_in_arena(arena, self.placement_initializer)
        self.cable = scene_result.cable_object
        self._is_flex_cable = scene_result.is_flex
        self.num_cable_points = scene_result.num_cable_points
        self._flex_container_body_name = scene_result.flex_container_body_name

        if self._is_flex_cable:
            self.cable_root_joint = None
            self._flex_mocap_name = "flex_routing_mocap"
            arena.worldbody.append(create_mocap_body(self._flex_mocap_name, (0.0, 0.0, self.cable_centerline_z)))
            self._flex_weld_eq_name = "flex_routing_endpoint_weld"
            self.model = ManipulationTask(
                mujoco_arena=arena,
                mujoco_robots=[robot.robot_model for robot in self.robots],
                mujoco_objects=None,
            )
            self.model.equality.append(create_weld_constraint(
                self._flex_weld_eq_name, self._flex_mocap_name, scene_result.graspable_body_names[0],
            ))
        else:
            self.cable_root_joint = scene_result.cable_root_joint
            self.placement_initializer = scene_result.placement_initializer
            self.model = ManipulationTask(
                mujoco_arena=arena,
                mujoco_robots=[robot.robot_model for robot in self.robots],
                mujoco_objects=self.cable,
            )

    def _setup_references(self):
        """在 MuJoCo 模型加载完成后，缓存各类 ID 供运行时快速查询。"""
        super()._setup_references()
        ids = self._cable_in_task.resolve_sim_ids(self.sim)

        # 应用视觉/物理修复（geom_group + flex 摩擦力）
        self._cable_in_task.apply_visual_fixes(self.sim)
        self._flex_id = ids.get("flex_id")
        self._flex_vertadr = ids.get("flex_vertadr")
        self._flex_vertnum = ids.get("flex_vertnum")
        self.cable_point_ids = ids["cable_point_ids"]
        self.num_cable_points = ids["num_cable_points"]

        if self._flex_id is not None:
            self._flex_mocap_body_id = self.sim.model.body_name2id(self._flex_mocap_name)
            self._flex_mocap_id = self.sim.model.body_mocapid[self._flex_mocap_body_id]
            self._flex_weld_eq_id = None
            for eq_id in range(self.sim.model.neq):
                if self.sim.model.equality(eq_id).name == self._flex_weld_eq_name:
                    self._flex_weld_eq_id = eq_id
                    break
        self.route_site_ids = [self.sim.model.site_name2id(f"route_wp_{i:02d}") for i in range(len(self.route_waypoints))]

    def _setup_observables(self):
        """注册路径规划任务的观测（cable_points, route_waypoints, route_progress 等）。"""
        observables = super()._setup_observables()

        if self.use_object_obs:
            modality = "object"

            @sensor(modality=modality)
            def cable_points(obs_cache):
                return self._get_cable_points().reshape(-1)

            @sensor(modality=modality)
            def route_waypoints(obs_cache):
                return self.route_waypoints.reshape(-1)

            @sensor(modality=modality)
            def attached_endpoint_pos(obs_cache):
                return self._get_cable_points()[0]

            @sensor(modality=modality)
            def current_target_pos(obs_cache):
                return self._get_current_grip_target()

            @sensor(modality=modality)
            def eef_to_current_target(obs_cache):
                return self._get_current_grip_target() - self._get_gripper_site_position()

            @sensor(modality=modality)
            def route_progress(obs_cache):
                denom = max(1, len(self.route_waypoints) - 1)
                return np.array([self.current_waypoint_index / denom], dtype=float)

            sensors = [
                cable_points,
                route_waypoints,
                attached_endpoint_pos,
                current_target_pos,
                eef_to_current_target,
                route_progress,
            ]
            names = [s.__name__ for s in sensors]

            for name, sensor_fn in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=sensor_fn,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _reset_internal(self):
        """重置路径规划任务状态：路径进度归零，放置线缆，设置附着。"""
        super()._reset_internal()
        self.current_waypoint_index = 0
        if self._is_flex_cable:
            # Flex 电缆重置：将容器 body 放置到桌面高度
            container_body_id = self.sim.model.body_name2id(self._flex_container_body_name)
            self.sim.model.body_pos[container_body_id, 2] = self.cable_centerline_z
            # 将容器放到第一个路径点附近
            self.sim.model.body_pos[container_body_id, 0] = self.route_waypoints[0][0]
            self.sim.model.body_pos[container_body_id, 1] = self.route_waypoints[0][1]
            self.sim.forward()
            # 同步 mocap 到 flex 电缆第一个顶点
            first_vertex_pos = self._get_cable_points()[0].copy()
            self.sim.data.mocap_pos[self._flex_mocap_id] = first_vertex_pos
            self.sim.data.mocap_quat[self._flex_mocap_id] = np.array([1.0, 0.0, 0.0, 0.0])
            if self._flex_weld_eq_id is not None:
                self.sim.data.eq_active[self._flex_weld_eq_id] = 1
            self.sim.forward()
        elif not self.deterministic_reset:
            self._cable_in_task.set_scene_state("placement_initializer", self.placement_initializer)
            self._cable_in_task.set_scene_state("cable_root_joint", self.cable_root_joint)
            self._cable_in_task.apply_reset(self.sim, self.rng, deterministic=False)
            self._attach_cable_endpoint_to_gripper()

    def _pre_action(self, action, policy_step=False):
        """每个控制步执行前调用：更新线缆端点跟随夹爪。"""
        super()._pre_action(action, policy_step=policy_step)
        if getattr(self, "_is_flex_cable", False):
            self._flex_pre_action_check(action)
        self._attach_cable_endpoint_to_gripper()

    def _activate_flex_attachment(self):
        """延迟附着激活（flex 路径）：激活 weld 约束 + 同步 mocap。"""
        params = self._attach_pending_params
        self._attach_pending = False
        self._attach_pending_params = {}
        # 端点抓取：激活 weld 约束
        self.attachment_enabled = True
        if hasattr(self, '_flex_weld_eq_id') and self._flex_weld_eq_id is not None:
            self.sim.data.eq_active[self._flex_weld_eq_id] = 1
        if hasattr(self, '_flex_mocap_id') and self._flex_mocap_id is not None:
            try:
                body_id = self.sim.model.body_name2id("flex_cable_0")
                self.sim.data.mocap_pos[self._flex_mocap_id] = self.sim.data.xpos[body_id].copy()
            except (ValueError, KeyError):
                pass
        self.sim.forward()

    def _get_flex_attach_target_pos(self):
        """获取 flex 附着目标位置（用于距离检查）。"""
        if self._is_flex_cable and hasattr(self, '_flex_mocap_id') and self._flex_mocap_id is not None:
            # 端点抓取：目标是 flex_cable_0 的当前位置
            try:
                body_id = self.sim.model.body_name2id("flex_cable_0")
                return self.sim.data.xpos[body_id].copy()
            except (ValueError, KeyError):
                return None
        return None

    def _post_action(self, action):
        """每个控制步结束后调用：更新路径进度、计算 reward、检查 done、收集 metrics。"""
        self._update_route_progress()
        reward = self.reward(action)
        self.done = (self.timestep >= self.horizon) and not self.ignore_done
        info = self._compute_metrics()
        return reward, self.done, info

    def _attach_cable_endpoint_to_gripper(self):
        """将线缆附着点平滑跟随夹爪位置。

        两种实现方式：
        - flex 电缆：通过 mocap body 控制端点（与 CableBaseEnv 类似）
        - rmb 电缆：直接修改根节点 joint 的位置 + 速度阻尼

        cable_root_to_attached_site 补偿线缆根节点到附着 site 的偏移，
        确保夹爪抓住的是线缆端点而不是根节点。
        """
        if self._is_flex_cable:
            # flex 顶点直接操控模式（中点抓取）
            if self._flex_grasp_active:
                self._flex_vertex_follow_gripper()
                return
            # Flex 电缆：通过 mocap 控制端点
            if self._flex_mocap_id is None:
                return
            grip_pos = self._get_gripper_site_position()
            target_pos = grip_pos + self.attach_offset - self.cable_root_to_attached_site
            current_pos = self.sim.data.mocap_pos[self._flex_mocap_id].copy()
            self.sim.data.mocap_pos[self._flex_mocap_id] = (
                current_pos + self.attachment_follow_gain * (target_pos - current_pos)
            )
            return
        if self.cable_root_joint is None:
            return
        grip_pos = self._get_gripper_site_position()
        joint_qpos = self.sim.data.get_joint_qpos(self.cable_root_joint).copy()
        target_pos = grip_pos + self.attach_offset - self.cable_root_to_attached_site
        joint_qpos[:3] = joint_qpos[:3] + self.attachment_follow_gain * (target_pos - joint_qpos[:3])
        self.sim.data.set_joint_qpos(self.cable_root_joint, joint_qpos)
        joint_qvel = self.sim.data.get_joint_qvel(self.cable_root_joint).copy()
        joint_qvel[:] *= self.attachment_velocity_damping
        self.sim.data.set_joint_qvel(self.cable_root_joint, joint_qvel)

    def _get_gripper_site_position(self):
        """返回夹爪 grip_site 的 3D 位置。"""
        arm = self.robots[0].arms[0]
        grip_site = self.robots[0].gripper[arm].important_sites["grip_site"]
        return np.array(self.sim.data.get_site_xpos(grip_site))

    def _get_current_target(self):
        """返回当前目标路径点的位置。"""
        return self.route_waypoints[self.current_waypoint_index]

    def _get_current_grip_target(self):
        """返回夹爪应到达的目标位置（考虑 attach_offset 和最低高度限制）。"""
        target = self._get_current_target() - self.attach_offset
        target = np.array(target, dtype=float)
        target[2] = max(target[2], self.min_gripper_z)
        return target

    def _update_route_progress(self):
        """更新路径进度：检查线缆端点是否到达当前目标路径点。

        使用 while 循环处理一次移动中连续经过多个路径点的情况。
        当端点到当前路径点的距离 > waypoint_tolerance 时停止前进。
        """
        endpoint = self._get_cable_points()[0]
        while self.current_waypoint_index < len(self.route_waypoints) - 1:
            distance = np.linalg.norm(endpoint - self.route_waypoints[self.current_waypoint_index])
            if distance > self.waypoint_tolerance:
                break
            self.current_waypoint_index += 1

    def _get_cable_points(self):
        """读取线缆所有关键点的 3D 位置（形状为 [N, 3]）。"""
        return self._cable_in_task.get_cable_points(self.sim)

    def _resolve_cable_name(self, base_name, kind):
        """解析线缆 body/site 名称（处理命名前缀兼容问题）。"""
        return self._cable_in_task.resolve_cable_name(self.sim, base_name, kind)

    def _point_to_segment_distance(self, points, seg_start, seg_end):
        """计算一组点到线段的最短距离（向量化实现）。

        对每个点，计算其在线段上的投影参数 t（0~1），然后计算到投影点的距离。
        t 被裁剪到 [0, 1] 范围内，确保端点处距离正确。
        """
        seg = seg_end - seg_start
        seg_norm_sq = float(np.dot(seg, seg))
        if seg_norm_sq < 1e-12:
            return np.linalg.norm(points - seg_start, axis=1)
        t = np.clip(((points - seg_start) @ seg) / seg_norm_sq, 0.0, 1.0)
        projection = seg_start + t[:, None] * seg
        return np.linalg.norm(points - projection, axis=1)

    def _compute_metrics(self):
        """计算路径规划任务的完整指标。

        指标说明：
        - route_error: 线缆前半段关键点到最近路径段的平均距离（线缆形状是否沿路径）
        - endpoint_error: 线缆端点到最终目标的距离
        - route_completion: 路径完成度（到达的路径点比例，0~1）
        - success: 全部达标（完成度>=100% + 端点误差<0.03 + 路径误差<0.06）

        只用线缆前半段计算 route_error，因为后半段可能还在上一个路径段上。
        """
        points = self._get_cable_points()
        routed_points = points[: max(4, len(points) // 2)]
        segment_errors = []
        for idx in range(len(self.route_waypoints) - 1):
            segment_errors.append(
                self._point_to_segment_distance(routed_points, self.route_waypoints[idx], self.route_waypoints[idx + 1])
            )
        route_error = float(np.mean(np.min(np.stack(segment_errors, axis=0), axis=0)))
        endpoint_error = float(np.linalg.norm(points[0] - self.route_waypoints[-1]))
        route_completion = float(self.current_waypoint_index / max(1, len(self.route_waypoints) - 1))
        success = bool(route_completion >= 1.0 and endpoint_error < 0.03 and route_error < 0.06)
        return {
            "route_error": route_error,
            "endpoint_error": endpoint_error,
            "route_completion": route_completion,
            "success": success,
        }

    def _check_success(self):
        return self._compute_metrics()["success"]

    @property
    def _visualizations(self):
        vis_settings = super()._visualizations
        vis_settings.add("grippers")
        return vis_settings
