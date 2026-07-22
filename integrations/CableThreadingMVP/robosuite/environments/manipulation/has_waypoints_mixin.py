"""HasWaypointsMixin — 路径点跟踪能力混入类。

为 DLO 任务环境提供路径点（waypoints）跟踪：
  - 路径点存储和配置
  - 路径进度跟踪（当前目标路径点）
  - 当前目标计算
  - 路径点可视化（site 标记）
  - 路径相关观测

用法：
    class MyTask(BaseDLOEnv, HasWaypointsMixin):
        def __init__(self, route_waypoints=None, **kwargs):
            super().__init__(**kwargs)
            self._init_waypoints(route_waypoints)

        def _load_model(self):
            super()._load_model()
            self._add_waypoint_sites(self.arena)
"""

import numpy as np

from robosuite.utils.mjcf_utils import CustomMaterial


class HasWaypointsMixin:
    """路径点跟踪能力混入类。"""

    # ---- 默认参数 ----
    waypoint_tolerance: float = 0.03

    # ---- 运行时状态 ----
    route_waypoints: np.ndarray = None  # (N, 3)
    current_waypoint_index: int = 0

    def _init_waypoints(self, route_waypoints=None, cable_centerline_z=0.818):
        """初始化路径点（在子类 __init__ 中调用）。"""
        if route_waypoints is None:
            route_waypoints = [
                (-0.24, -0.12, cable_centerline_z),
                (-0.08, -0.12, cable_centerline_z),
                (0.06, 0.02, cable_centerline_z),
                (0.18, 0.12, cable_centerline_z),
            ]
        self.route_waypoints = np.array(route_waypoints, dtype=float)
        self.current_waypoint_index = 0

    def _add_waypoint_sites(self, arena):
        """向 arena 添加路径点可视化 site（在 _load_model 中调用）。"""
        for i, wp in enumerate(self.route_waypoints):
            rgba = [0.0, 1.0, 0.0, 0.8] if i == 0 else [1.0, 1.0, 0.0, 0.5]
            site = arena.worldbody.find("worldbody")  # placeholder
            # 实际实现需要在 arena.worldbody 中添加 site 元素
            # 简化版：直接在子类中实现

    def _get_current_target(self):
        """返回当前目标路径点的位置。"""
        return self.route_waypoints[self.current_waypoint_index]

    def _get_current_grip_target(self, attach_offset=None, min_gripper_z=None):
        """返回夹爪应到达的目标位置。"""
        target = self._get_current_target().copy()
        if attach_offset is not None:
            target = target - np.asarray(attach_offset, dtype=float)
        if min_gripper_z is not None:
            target[2] = max(target[2], min_gripper_z)
        return target

    def _update_route_progress(self, cable_points):
        """更新路径进度：检查线缆端点是否到达当前目标路径点。"""
        endpoint = cable_points[0]
        while self.current_waypoint_index < len(self.route_waypoints) - 1:
            distance = np.linalg.norm(endpoint - self.route_waypoints[self.current_waypoint_index])
            if distance > self.waypoint_tolerance:
                break
            self.current_waypoint_index += 1

    def _point_to_segment_distance(self, points, seg_start, seg_end):
        """计算一组点到线段的最短距离（向量化实现）。"""
        seg_vec = seg_end - seg_start
        seg_len_sq = float(np.dot(seg_vec, seg_vec))
        if seg_len_sq < 1e-10:
            return np.linalg.norm(points - seg_start, axis=1)
        t = np.clip(np.dot(points - seg_start, seg_vec) / seg_len_sq, 0.0, 1.0)
        projections = seg_start + np.outer(t, seg_vec)
        return np.linalg.norm(points - projections, axis=1)

    @property
    def route_completion(self):
        """返回路径完成比例 (0.0 ~ 1.0)。"""
        if self.route_waypoints is None or len(self.route_waypoints) == 0:
            return 0.0
        return self.current_waypoint_index / max(len(self.route_waypoints) - 1, 1)
