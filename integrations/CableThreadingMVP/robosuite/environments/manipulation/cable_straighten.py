"""
CableStraighten — 线缆拉直任务。

任务目标：将弯曲的线缆拉直并沿桌面上的目标线段对齐。
继承 CableBaseEnv，覆写了 reward 和 _compute_metrics。

成功条件（全部满足）：
  - 线缆中心线误差低于阈值（直线度）
  - 端点距离目标端点足够近
  - 线缆直线度（straightness_ratio）超过阈值
  - 线缆平放在桌面上（table_contact_ratio >= 95%）
  - 没有激活的 weld 约束（已松开夹爪）

核心度量委托给 task_logic.py 中的 straighten_task_metrics() 计算，
包括 centerline_error、endpoint_error、bend_energy 等。
"""

import numpy as np

from robosuite.environments.manipulation.cable_base import CableBaseEnv
from robosuite.utils.dlo.cable_metrics import (
    polyline_length,
    straightness_ratio,
)
from robosuite.utils.dlo.task_logic import DLOTaskState, StraightenTaskSpec, straighten_task_metrics
from robosuite.utils.dlo.task_scene_utils import add_target_sites


class CableStraighten(CableBaseEnv):
    """
    Straighten a cable along a target line segment on a table (supports all cable models).
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
        cable_model="flex",
        grasp_mode="attachment",
        anchor_enabled=True,
        target_line_visible=False,
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
        # ---- 成功判定阈值 ----
        self.success_centerline_threshold = 0.025       # 关键点到目标线的平均距离阈值
        self.success_centerline_max_threshold = 0.045   # 关键点到目标线的最大距离阈值
        self.success_endpoint_threshold = 0.06          # 端点到目标端点的距离阈值
        self.success_straightness_threshold = 0.97      # 直线度阈值（straightness_ratio = 直线距离/折线长度）
        # ---- 初始几何信息（在 _record_initial_geometry 中填充） ----
        self.initial_polyline_length = 0.0
        self.initial_straightness_ratio = 0.0
        self.initial_keypoint_z_median = 0.0

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            gripper_types=gripper_types,
            base_types=base_types,
            initialization_noise=initialization_noise,
            table_full_size=table_full_size,
            table_friction=table_friction,
            cable_model=cable_model,
            grasp_mode=grasp_mode,
            anchor_enabled=anchor_enabled,
            target_line_visible=target_line_visible,
            use_camera_obs=use_camera_obs,
            use_object_obs=use_object_obs,
            reward_scale=reward_scale,
            reward_shaping=reward_shaping,
            placement_initializer=placement_initializer,
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
        """计算拉直任务的 reward。

        reward = -centerline_error - 0.5 * endpoint_error - 0.05 * bend_energy + success_bonus

        - centerline_error: 关键点到目标线的平均距离（主要优化目标）
        - endpoint_error: 两个端点到目标端点的距离之和（端点对齐）
        - bend_energy: 线缆弯曲程度（鼓励拉直，权重较小）
        - success_bonus: 全部达标后给 +1 的奖励
        """
        metrics = self._compute_metrics()
        reward = -metrics["centerline_error"] - 0.5 * metrics["endpoint_error"] - 0.05 * metrics["bend_energy"]
        if metrics["success"]:
            reward += 1.0
        return self.reward_scale * reward

    def _load_model(self):
        """加载模型，可选地在场景中渲染目标线段的可视化标记。"""
        super()._load_model()

        if self.target_line_visible:
            target_samples = np.linspace(self.target_start, self.target_end, 11)
            add_target_sites(
                self.model.mujoco_arena.worldbody,
                target_samples,
                name_prefix="straight_target",
                size=(0.007,),
                rgba=(0.1, 0.8, 0.2, 0.8),
            )

    def _record_initial_geometry(self):
        """记录重置后的初始几何信息（用于计算相对改善量）。

        记录初始折线长度、直线度和 z 中位数，供 _compute_metrics 中的
        task_logic 使用（例如计算长度保持率、直线度改善量等）。
        """
        points = self._get_cable_points()
        self.initial_polyline_length = float(polyline_length(points))
        self.initial_straightness_ratio = float(straightness_ratio(points))
        self.initial_keypoint_z_median = float(np.median(points[:, 2]))

    def _compute_metrics(self):
        """计算拉直任务的完整指标。

        核心逻辑委托给 task_logic.py 中的 straighten_task_metrics()，传入：
        - DLOTaskState: 当前关键点、桌面高度、初始几何信息
        - StraightenTaskSpec: 成功判定阈值

        straighten_task_metrics 返回 centerline_error、endpoint_error、bend_energy、
        straightness_ratio、task_success 等指标。

        本方法额外添加夹爪距离、物理抓取状态等通用指标。
        success 要求 task_success 且没有激活的 weld 约束（已松开夹爪）。
        """
        points = self._get_cable_points()
        attachment_count = self._attachment_eq_active_count()
        task_metrics = straighten_task_metrics(
            DLOTaskState(
                keypoints=points,
                table_top_z=self.table_top_z,
                centerline_z=self.cable_centerline_z,
                initial_polyline_length=self.initial_polyline_length,
                initial_straightness_ratio=self.initial_straightness_ratio,
            ),
            StraightenTaskSpec(
                target_start=self.target_start,
                target_end=self.target_end,
                centerline_threshold=self.success_centerline_threshold,
                centerline_max_threshold=self.success_centerline_max_threshold,
                endpoint_threshold=self.success_endpoint_threshold,
                straightness_threshold=self.success_straightness_threshold,
                table_contact_ratio_threshold=self.success_table_contact_ratio_threshold,
                table_contact_z_tolerance=self.table_contact_z_tolerance,
                table_penetration_tolerance=self.table_penetration_tolerance,
            ),
        )
        from robosuite.utils.dlo.cable_metrics import gripper_to_cable_distance

        try:
            gripper_distance = gripper_to_cable_distance(self._get_gripper_site_position(), points)
        except ValueError:
            gripper_distance = np.inf
        success = bool(task_metrics["task_success"] and attachment_count == 0)
        return {
            **task_metrics,
            "grasp_mode": self.grasp_mode,
            "gripper_to_cable_distance": gripper_distance,
            "physical_grasp_contact_count": self._physical_grasp_contact_count(),
            "physical_grasp_lift_height": self._physical_grasp_lift_height(),
            "physical_grasp_success": self._physical_grasp_success(),
            "attachment_eq_active_count": attachment_count,
            "flex_grasp_ever_active": bool(self._flex_grasp_ever_active),
            "success": success,
        }

    def _check_success(self):
        return self._compute_metrics()["success"]
