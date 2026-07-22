"""
L3 纯逻辑层：任务指标计算与成功判定。

本模块是 dloBench 五层架构中的 L3（Logic）层，所有函数均为纯函数，
不依赖 MuJoCo 仿真器，不读写环境状态。输入为 numpy 数组和 dataclass
参数，输出为指标字典。

核心职责：
  - 定义任务状态（DLOTaskState / ThreadingTaskState）和任务规格（Spec）
  - 实现三种任务的指标计算管线：straighten / move_to_target / threading
  - 提供几何辅助函数（线段交叉、最近点、通过率等）

设计原则：
  - 所有阈值通过 frozen dataclass 传入，便于超参搜索和序列化
  - 指标返回 dict 而非 dataclass，方便 JSON 序列化和日志记录
  - 失败/非法输入返回带有 metric_status 字段的降级指标，而非抛异常
"""

from dataclasses import dataclass

import numpy as np

from robosuite.utils.dlo.cable_metrics import (
    cable_centroid,
    endpoint_distance as cable_endpoint_distance,
    keypoint_goal_coverage,
    line_deviation as cable_line_deviation,
    passed_keypoint_ratio,
    polyline_length,
    straightness_ratio,
    validate_keypoints,
)


# ---------------------------------------------------------------------------
# 数据类定义：任务状态与任务规格
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DLOTaskState:
    """柔性线缆（DLO）在某一时刻的快照状态。

    用于 straighten 和 move_to_target 两类任务的指标计算。

    Attributes:
        keypoints:       线缆离散采样点坐标，shape (N, 3)。
        table_top_z:     桌面高度（z 坐标），用于判断线缆是否贴合桌面。
        centerline_z:    线缆目标中心线高度，用于计算高度偏差。
        initial_polyline_length: 初始弧长，用于 shape_preservation 计算。
        initial_straightness_ratio: 初始直度比，用于对比拉直前后的改善。
    """
    keypoints: np.ndarray
    table_top_z: float
    centerline_z: float
    initial_polyline_length: float = 0.0
    initial_straightness_ratio: float = 0.0


@dataclass(frozen=True)
class StraightenTaskSpec:
    """拉直任务的成功判定规格（阈值集合）。

    拉直任务要求线缆：贴合桌面、沿目标线段分布、端点对齐、足够直。

    Attributes:
        target_start / target_end: 目标线段的两个端点（XY 平面）。
        centerline_threshold:      平均中心线误差上限（米）。
        centerline_max_threshold:  最大中心线误差上限（米）。
        endpoint_threshold:        端点误差上限（米）。
        straightness_threshold:    直度比下限（0~1，越接近 1 越直）。
        table_contact_ratio_threshold: 桌面接触比例下限。
        table_contact_z_tolerance: z 方向允许偏离桌面的容差。
        table_penetration_tolerance: 允许穿透桌面的深度容差。
    """
    target_start: np.ndarray
    target_end: np.ndarray
    centerline_threshold: float = 0.025
    centerline_max_threshold: float = 0.045
    endpoint_threshold: float = 0.06
    straightness_threshold: float = 0.97
    table_contact_ratio_threshold: float = 0.95
    table_contact_z_tolerance: float = 0.025
    table_penetration_tolerance: float = 0.02


@dataclass(frozen=True)
class MoveToTargetTaskSpec:
    """移动到目标区域的任务规格。

    要求线缆质心落入目标圆内，且足够多的关键点被目标区域覆盖。

    Attributes:
        target_center:              目标圆心坐标 (x, y) 或 (x, y, z)。
        target_radius:              目标圆半径（米）。
        centroid_success_threshold: 质心到目标圆心的距离上限。
        coverage_success_threshold: 关键点覆盖率下限（0~1）。
    """
    target_center: np.ndarray
    target_radius: float
    centroid_success_threshold: float = 0.05
    coverage_success_threshold: float = 0.8


@dataclass(frozen=True)
class ThreadingTaskState:
    """穿线任务的完整状态快照。

    与 DLOTaskState 不同，穿线任务需要更多空间信息：杆位置、锚点、目标点。

    Attributes:
        cable_points:  线缆离散点坐标，shape (N, 3)。
        cable_end_pos: 线缆末端（被抓取端）当前位置。
        anchor_pos:    线缆固定端（锚点）位置。
        pole1_pos / pole2_pos: 两根杆的中心位置。
        endpoint_goal: 线缆末端的目标位置（穿过杆间隙后的位置）。
    """
    cable_points: np.ndarray
    cable_end_pos: np.ndarray
    anchor_pos: np.ndarray
    pole1_pos: np.ndarray
    pole2_pos: np.ndarray
    endpoint_goal: np.ndarray


@dataclass(frozen=True)
class ThreadingTaskSpec:
    """穿线任务的完整规格，包含所有几何阈值。

    穿线任务的核心要求：线缆从两根杆之间穿过，末端到达目标区域，
    线缆在桌面附近平放，且不能与杆发生穿透碰撞。

    Attributes:
        pole_radius:              杆的半径。
        pole_height:              杆的高度（用于计算杆顶 z 坐标）。
        goal_tolerance:           末端到达目标的距离容差。
        height_tolerance:         线缆高度超出杆顶的允许余量。
        thread_cross_threshold:   判定穿线方向的叉积阈值。
        gap_margin:               杆间隙边缘的收缩余量。
        thread_corridor_depth:    穿线走廊沿 y 方向的半深度。
        thread_front_back_margin: 判定线缆在杆前/后方存在的最小距离。
        endpoint_past_gap_margin: 末端必须超过杆间隙的最小距离。
        straightness_tolerance:   穿线后线缆偏离首尾连线的平均距离上限。
        straightness_ratio_threshold: 端点距离 / 弧长的下限。
        low_thread_height_margin: 低穿线判定的 z 方向余量。
        cable_intersection_tolerance: 线缆与杆连线相交的容差。
        table_settle_tolerance:   线缆在桌面上平放的高度散布容差。
        endpoint_table_tolerance: 末端高度与桌面的偏差容差。
        anchor_tolerance:         锚点稳定的最大偏移。
        post_collision_penetration_tolerance: 穿线后允许的杆穿透深度。
    """
    pole_radius: float
    pole_height: float
    goal_tolerance: float = 0.04
    height_tolerance: float = 0.01
    thread_cross_threshold: float = 1e-4
    gap_margin: float = 0.002
    thread_corridor_depth: float = 0.03
    thread_front_back_margin: float = 0.01
    endpoint_past_gap_margin: float = 0.025
    straightness_tolerance: float = 0.03
    straightness_ratio_threshold: float = 0.88
    low_thread_height_margin: float = 0.008
    cable_intersection_tolerance: float = 0.03
    pole_t_margin: float = 0.0
    table_settle_tolerance: float = 0.04
    endpoint_table_tolerance: float = 0.04
    anchor_tolerance: float = 0.02
    post_collision_penetration_tolerance: float = 0.005


# ---------------------------------------------------------------------------
# 桌面接触指标
# ---------------------------------------------------------------------------

def table_contact_metrics(state: DLOTaskState, *, z_tolerance: float, penetration_tolerance: float):
    """计算线缆与桌面的接触情况。

    判定逻辑：一个关键点算"接触桌面"需要同时满足两个条件：
      1. 该点的 z 坐标不低于桌面以下 penetration_tolerance（未严重穿透桌面）
      2. 该点的 z 坐标与目标中心线 z 的偏差不超过 z_tolerance

    返回字典包含：
      - 各种高度统计（最大离桌高度、最大穿透深度、最大中心线偏差）
      - table_contact_ratio: 满足接触条件的关键点比例
      - cable_on_table: 是否整体平放在桌面上（接触比例 >= 95%）
    """
    points = np.asarray(state.keypoints, dtype=float)
    z = points[:, 2]
    above_table = z - float(state.table_top_z)                   # 正值 = 在桌面上方
    centerline_error = np.abs(z - float(state.centerline_z))     # 与目标中心线的偏差
    # 同时满足：未穿透桌面太多 且 接近目标高度
    contact_mask = (above_table >= -float(penetration_tolerance)) & (centerline_error <= float(z_tolerance))
    return {
        "table_height_reference": float(state.table_top_z),
        "table_top_z": float(state.table_top_z),
        "cable_centerline_z": float(state.centerline_z),
        "max_keypoint_height_above_table": float(np.max(np.maximum(above_table, 0.0))),
        "max_keypoint_depth_below_table": float(np.max(np.maximum(-above_table, 0.0))),
        "max_keypoint_centerline_z_error": float(np.max(centerline_error)),
        "table_contact_ratio": float(np.mean(contact_mask)),
        "cable_on_table": bool(np.mean(contact_mask) >= 0.95),
    }


# ---------------------------------------------------------------------------
# 拉直任务指标管线
# ---------------------------------------------------------------------------

def _invalid_straighten_metrics(state: DLOTaskState, spec: StraightenTaskSpec, metric_status: str):
    """当输入数据非法时，返回一组降级指标（所有误差设为 inf，成功为 False）。

    这样调用方不需要额外处理异常，只需检查 metric_status 字段。
    """
    table_metrics = {
        "table_height_reference": float(state.table_top_z),
        "table_top_z": float(state.table_top_z),
        "cable_centerline_z": float(state.centerline_z),
        "max_keypoint_height_above_table": np.inf,
        "max_keypoint_depth_below_table": np.inf,
        "max_keypoint_centerline_z_error": np.inf,
        "table_contact_ratio": 0.0,
        "cable_on_table": False,
    }
    return {
        "metric_status": metric_status,
        "endpoint_distance": 0.0,
        "endpoint_distance_xy": 0.0,
        "polyline_length": 0.0,
        "polyline_length_xy": 0.0,
        "initial_polyline_length": float(state.initial_polyline_length),
        "straightness_ratio": 0.0,
        "initial_straightness_ratio": float(state.initial_straightness_ratio),
        "line_deviation": np.inf,
        "cable_centroid": np.full(3, np.nan),
        "centerline_error": np.inf,
        "centerline_max_error": np.inf,
        "endpoint_error": np.inf,
        "bend_energy": np.inf,
        "task_success": False,
        **table_metrics,
        "success_centerline_threshold": float(spec.centerline_threshold),
        "success_centerline_max_threshold": float(spec.centerline_max_threshold),
        "success_endpoint_threshold": float(spec.endpoint_threshold),
        "success_straightness_threshold": float(spec.straightness_threshold),
        "success_table_contact_ratio_threshold": float(spec.table_contact_ratio_threshold),
    }


def straighten_task_metrics(state: DLOTaskState, spec: StraightenTaskSpec):
    """拉直任务的完整指标计算管线。

    计算流程：
      1. 校验关键点数据合法性
      2. 计算 3D 和 XY 平面的弧长、端点距离、直度比
      3. 计算线缆质心、线偏差
      4. 计算桌面接触指标
      5. 将每个关键点投影到目标线段上，计算中心线误差（平均和最大）
      6. 计算端点误差（考虑线缆两端可能互换方向）
      7. 计算弯曲能量（相邻线段方向变化的累积）
      8. 综合所有阈值判定任务是否成功

    成功条件（全部满足）：
      - 平均中心线误差 <= centerline_threshold
      - 最大中心线误差 <= centerline_max_threshold
      - 端点误差 <= endpoint_threshold
      - 直度比 >= straightness_threshold
      - 桌面接触比例 >= table_contact_ratio_threshold
    """
    points = np.asarray(state.keypoints, dtype=float)
    try:
        validate_keypoints(points)
    except ValueError as exc:
        return _invalid_straighten_metrics(state, spec, str(exc))

    # --- 基础几何量 ---
    endpoint_span = cable_endpoint_distance(points)       # 3D 端点距离
    cable_length = polyline_length(points)                # 3D 弧长
    # XY 平面版本（忽略高度变化，更关注平面形状）
    xy_points = np.column_stack([points[:, :2], np.zeros(points.shape[0], dtype=float)])
    endpoint_span_xy = cable_endpoint_distance(xy_points)
    cable_length_xy = polyline_length(xy_points)
    straightness = straightness_ratio(xy_points)          # XY 平面直度比
    centroid = cable_centroid(points)
    line_error = cable_line_deviation(points, spec.target_start, spec.target_end)

    # --- 桌面接触 ---
    table_metrics = table_contact_metrics(
        state,
        z_tolerance=spec.table_contact_z_tolerance,
        penetration_tolerance=spec.table_penetration_tolerance,
    )
    table_metrics["cable_on_table"] = bool(table_metrics["table_contact_ratio"] >= spec.table_contact_ratio_threshold)

    # --- 中心线误差：将每个点投影到目标线段上，计算垂直距离 ---
    a_xy = np.asarray(spec.target_start, dtype=float)[:2]
    b_xy = np.asarray(spec.target_end, dtype=float)[:2]
    ab = b_xy - a_xy
    ab_norm_sq = np.dot(ab, ab)
    # t 是投影参数，clip 到 [0, 1] 表示投影点在线段上
    t = np.clip(((points[:, :2] - a_xy) @ ab) / ab_norm_sq, 0.0, 1.0)
    projections = a_xy + t[:, None] * ab
    point_errors = np.linalg.norm(points[:, :2] - projections, axis=1)
    centerline_error = float(np.mean(point_errors))
    centerline_max_error = float(np.max(point_errors))

    # --- 端点误差：线缆两端可能互换方向，取较小值 ---
    endpoint_error = min(
        np.linalg.norm(points[0, :2] - a_xy) + np.linalg.norm(points[-1, :2] - b_xy),
        np.linalg.norm(points[0, :2] - b_xy) + np.linalg.norm(points[-1, :2] - a_xy),
    )
    endpoint_error = float(endpoint_error)

    # --- 弯曲能量：相邻单位方向向量之差的范数累加，反映线缆弯曲程度 ---
    segs = np.diff(points, axis=0)
    unit = segs / (np.linalg.norm(segs, axis=1, keepdims=True) + 1e-8)
    bend_energy = float(np.sum(np.linalg.norm(np.diff(unit, axis=0), axis=1)))

    # --- 综合判定 ---
    task_success = bool(
        centerline_error <= spec.centerline_threshold
        and centerline_max_error <= spec.centerline_max_threshold
        and endpoint_error <= spec.endpoint_threshold
        and straightness >= spec.straightness_threshold
        and table_metrics["table_contact_ratio"] >= spec.table_contact_ratio_threshold
    )
    return {
        "metric_status": "ok",
        "endpoint_distance": endpoint_span,
        "endpoint_distance_xy": endpoint_span_xy,
        "polyline_length": cable_length,
        "polyline_length_xy": cable_length_xy,
        "initial_polyline_length": float(state.initial_polyline_length),
        "straightness_ratio": straightness,
        "initial_straightness_ratio": float(state.initial_straightness_ratio),
        "line_deviation": line_error,
        "cable_centroid": centroid,
        "centerline_error": centerline_error,
        "centerline_max_error": centerline_max_error,
        "endpoint_error": endpoint_error,
        "bend_energy": bend_energy,
        "task_success": task_success,
        **table_metrics,
        "success_centerline_threshold": float(spec.centerline_threshold),
        "success_centerline_max_threshold": float(spec.centerline_max_threshold),
        "success_endpoint_threshold": float(spec.endpoint_threshold),
        "success_straightness_threshold": float(spec.straightness_threshold),
        "success_table_contact_ratio_threshold": float(spec.table_contact_ratio_threshold),
    }


# ---------------------------------------------------------------------------
# 移动到目标区域任务指标
# ---------------------------------------------------------------------------

def move_to_target_task_metrics(state: DLOTaskState, spec: MoveToTargetTaskSpec, base_metrics):
    """移动到目标区域任务的指标计算。

    该任务要求将线缆整体移动到桌面上的圆形目标区域。

    计算内容：
      - 质心到目标圆心的 XY 距离
      - 关键点覆盖率（有多少比例的点落在目标圆内）
      - 形状保持度（移动前后弧长变化比例，1.0 = 完全不变）

    成功条件（全部满足）：
      - 质心距离 <= centroid_success_threshold
      - 覆盖率 >= coverage_success_threshold
      - 线缆平放在桌面上（来自 base_metrics）

    Args:
        state: 当前线缆状态。
        spec:  任务规格（目标圆心和半径）。
        base_metrics: 来自 straighten_task_metrics 的基础指标，
                      需要包含 polyline_length、initial_polyline_length、cable_on_table。
    """
    points = np.asarray(state.keypoints, dtype=float)
    centroid = cable_centroid(points)
    centroid_distance = float(np.linalg.norm(centroid[:2] - np.asarray(spec.target_center, dtype=float)[:2]))
    coverage = keypoint_goal_coverage(points, spec.target_center, goal_radius=spec.target_radius)

    # 形状保持度：弧长变化越小越好
    current_length = float(base_metrics.get("polyline_length", 0.0))
    initial_length = float(base_metrics.get("initial_polyline_length", 0.0))
    shape_preservation = 1.0
    if initial_length > 1e-8:
        shape_preservation = max(0.0, 1.0 - abs(current_length - initial_length) / initial_length)

    task_success = bool(
        centroid_distance <= spec.centroid_success_threshold
        and coverage >= spec.coverage_success_threshold
        and bool(base_metrics.get("cable_on_table", False))
    )
    return {
        "cable_centroid": centroid,
        "target_center": np.asarray(spec.target_center, dtype=float).copy(),
        "target_radius": float(spec.target_radius),
        "centroid_distance_to_goal": centroid_distance,
        "keypoint_goal_coverage": coverage,
        "shape_preservation": shape_preservation,
        "task_success": task_success,
        "success": task_success,
        "centroid_success_threshold": float(spec.centroid_success_threshold),
        "coverage_success_threshold": float(spec.coverage_success_threshold),
    }


# ---------------------------------------------------------------------------
# 几何辅助函数
# ---------------------------------------------------------------------------

def _segment_crosses_gap_corridor(a, b, pole1_xy, pole2_xy, *, pole_radius: float, gap_margin: float):
    """判断线段 AB 是否在 XY 平面上穿过两杆之间的间隙走廊。

    间隙走廊的 x 范围 = [pole1.x + radius - margin, pole2.x - radius + margin]。
    仅当线段在 pole_y 处发生 y 方向穿越（即线段两端在杆连线两侧），
    且穿越点的 x 坐标落在走廊内时，才返回 True。

    Returns:
        (crossed: bool, x_cross: float | None)
    """
    pole_y = pole1_xy[1]
    ay = a[1] - pole_y
    by = b[1] - pole_y
    # 线段两端在杆连线同侧，或不穿越 -> 不交叉
    if abs(ay - by) < 1e-8 or ay * by > 0:
        return False, None

    # 计算线段在 y = pole_y 处的参数 t
    t = (pole_y - a[1]) / (b[1] - a[1])
    if t < 0.0 or t > 1.0:
        return False, None

    # 计算穿越点的 x 坐标，检查是否在走廊内
    x_cross = a[0] + t * (b[0] - a[0])
    x_min = min(pole1_xy[0], pole2_xy[0]) + pole_radius - gap_margin
    x_max = max(pole1_xy[0], pole2_xy[0]) - pole_radius + gap_margin
    return x_min <= x_cross <= x_max, x_cross


def _segment_intersection_2d(a, b, c, d):
    """计算两条 2D 线段 AB 和 CD 是否相交，以及交点参数。

    使用参数化方法：
      P = a + t * (b - a),  t in [0,1]
      Q = c + u * (d - c),  u in [0,1]
    当 t, u 同时在 [0,1] 内时两线段相交。

    Returns:
        (intersects: bool, t: float|None, u: float|None, point: ndarray|None)
    """
    ab = b - a
    cd = d - c
    denom = ab[0] * cd[1] - ab[1] * cd[0]
    if abs(denom) < 1e-10:
        return False, None, None, None

    ac = c - a
    t = (ac[0] * cd[1] - ac[1] * cd[0]) / denom
    u = (ac[0] * ab[1] - ac[1] * ab[0]) / denom
    if t < 0.0 or t > 1.0 or u < 0.0 or u > 1.0:
        return False, None, None, None

    point = a + t * ab
    return True, float(t), float(u), point


def _closest_segment_points_2d(a, b, c, d):
    """计算两条 2D 线段 AB 和 CD 上的最近点对及距离。

    算法：枚举 4 种候选组合（C/D 到 AB 的投影 + A/B 到 CD 的投影），
    取距离最小的一组。这是因为线段上最近点可能出现在端点上。

    Returns:
        (distance, t, u, point) — 其中 point 是 AB 上最近点的坐标
    """
    candidates = []

    ab = b - a
    cd = d - c
    ab_norm_sq = float(np.dot(ab, ab))
    cd_norm_sq = float(np.dot(cd, cd))

    # C、D 投影到 AB 线段上
    if ab_norm_sq > 1e-12:
        for point in (c, d):
            t = float(np.clip(np.dot(point - a, ab) / ab_norm_sq, 0.0, 1.0))
            candidates.append((np.linalg.norm((a + t * ab) - point), t, 0.0 if point is c else 1.0, a + t * ab))
    # A、B 投影到 CD 线段上
    if cd_norm_sq > 1e-12:
        for point in (a, b):
            u = float(np.clip(np.dot(point - c, cd) / cd_norm_sq, 0.0, 1.0))
            t = 0.0 if point is a else 1.0
            candidates.append((np.linalg.norm(point - (c + u * cd)), t, u, c + u * cd))

    if not candidates:
        return float(np.linalg.norm(a - c)), 0.0, 0.0, a
    distance, t, u, point = min(candidates, key=lambda item: item[0])
    return float(distance), float(t), float(u), point


def _point_to_segment_distances(points, seg_start, seg_end):
    """批量计算多个点到同一条线段的最短距离。

    对每个点，先投影到线段所在的直线上，再 clip 到线段范围内。
    这是一个向量化操作，比逐点循环快很多。
    """
    segment = seg_end - seg_start
    segment_norm_sq = float(np.dot(segment, segment))
    if segment_norm_sq < 1e-12:
        return np.linalg.norm(points - seg_start, axis=1)
    t = np.clip(((points - seg_start) @ segment) / segment_norm_sq, 0.0, 1.0)
    projection = seg_start + t[:, None] * segment
    return np.linalg.norm(points - projection, axis=1)


def _point_to_polyline_distances(points, vertices):
    """计算每个点到折线（多段线）的最短距离。

    折线由 vertices 定义，每两个相邻顶点构成一段。
    对每段分别计算距离，取最小值。
    """
    distances = []
    for start, end in zip(vertices[:-1], vertices[1:]):
        distances.append(_point_to_segment_distances(points, start, end))
    if not distances:
        return np.linalg.norm(points - vertices[0], axis=1)
    return np.min(np.stack(distances, axis=0), axis=0)


def _exit_side_arclength_ratio(cable_points, gate_center, exit_axis):
    """计算线缆在门（gate）出口侧的弧长占比。

    "门"由 gate_center 和 exit_axis 定义。exit_axis 是出口方向的单位向量。
    将线缆每个采样点投影到 exit_axis 上，位于 gate_center 出口侧（投影 >= 0）
    的弧长占总弧长的比例。

    对于跨越边界的线段，按线性插值计算其在出口侧的部分。

    Returns:
        0.0 ~ 1.0 之间的浮点数
    """
    cable_points = np.asarray(cable_points, dtype=float)
    axis = np.asarray(exit_axis, dtype=float)[:2]
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-12 or len(cable_points) < 2:
        return 0.0
    axis = axis / norm
    # 每个点在 exit_axis 上的投影（正值 = 在出口侧）
    projection = (cable_points[:, :2] - np.asarray(gate_center, dtype=float)[:2]) @ axis
    segments = np.linalg.norm(np.diff(cable_points[:, :2], axis=0), axis=1)
    total_length = float(np.sum(segments))
    if total_length <= 1e-12:
        return float(np.mean(projection >= 0.0))

    # 逐段计算在出口侧的弧长
    passed_length = 0.0
    for idx, segment_length in enumerate(segments):
        start_projection = float(projection[idx])
        end_projection = float(projection[idx + 1])
        if start_projection >= 0.0 and end_projection >= 0.0:
            # 整段在出口侧
            passed_length += float(segment_length)
        elif start_projection < 0.0 <= end_projection:
            # 从非出口侧进入出口侧，按比例计算
            alpha = -start_projection / max(end_projection - start_projection, 1e-12)
            passed_length += float(segment_length) * (1.0 - alpha)
        elif start_projection >= 0.0 > end_projection:
            # 从出口侧离开，按比例计算
            alpha = start_projection / max(start_projection - end_projection, 1e-12)
            passed_length += float(segment_length) * alpha
    return float(np.clip(passed_length / total_length, 0.0, 1.0))


def _expected_exit_side_ratio(gate_center, exit_axis, anchor_pos, endpoint_goal):
    """计算理想的出口侧弧长占比。

    根据锚点和目标点在门两侧的位置，推算理论上应该有多少比例的线缆
    在门的出口侧。用于归一化实际的 passed_ratio。

    如果锚点和目标在门同侧，返回 1.0（避免除零）。
    """
    anchor = np.asarray(anchor_pos, dtype=float)
    goal = np.asarray(endpoint_goal, dtype=float)
    axis = np.asarray(exit_axis, dtype=float)[:2]
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-12:
        return 1.0
    axis = axis / norm
    gate_xy = np.asarray(gate_center, dtype=float)[:2]
    anchor_projection = float((anchor[:2] - gate_xy) @ axis)
    goal_projection = float((goal[:2] - gate_xy) @ axis)
    total_projection = goal_projection - anchor_projection
    if total_projection <= 1e-12:
        return 1.0
    return float(np.clip(goal_projection / total_projection, 1e-6, 1.0))


def threading_geometric_post_collision_count(cable_xy, pole1_xy, pole2_xy, *, pole_radius: float, penetration_tolerance: float):
    """统计穿线完成后线缆与杆的碰撞点数量。

    遍历所有线缆 XY 点，如果某点到杆中心的距离 < (杆半径 - 穿透容差)，
    则该点被认为穿透了杆体。

    Args:
        cable_xy: 线缆 XY 坐标，shape (N, 2)。
        pole1_xy / pole2_xy: 杆的 XY 坐标。
        pole_radius: 杆的半径。
        penetration_tolerance: 允许的穿透深度。

    Returns:
        穿透杆体的线缆点数量。
    """
    clearance_limit = float(pole_radius) - float(penetration_tolerance)
    clearance1 = np.linalg.norm(cable_xy - pole1_xy[None, :], axis=1) < clearance_limit
    clearance2 = np.linalg.norm(cable_xy - pole2_xy[None, :], axis=1) < clearance_limit
    return int(np.count_nonzero(clearance1 | clearance2))


# ---------------------------------------------------------------------------
# 穿线任务指标管线（最复杂的任务）
# ---------------------------------------------------------------------------

def threading_task_metrics(state: ThreadingTaskState, spec: ThreadingTaskSpec, *, post_collision_count: int = 0):
    """穿线任务的完整指标计算管线。

    穿线任务是 dloBench 中最复杂的任务，需要同时检查多个条件：

    1. 穿线检测：线缆是否从两杆之间穿过
       - 遍历线缆每一段，检测与杆连线的 2D 交叉
       - 交叉点必须在杆间隙走廊内
       - 穿线方向必须正确（叉积判定）
       - 交叉点的 z 坐标必须低于杆顶

    2. 末端位置：末端是否到达目标区域，是否已超过杆间隙

    3. 锚点稳定性：线缆固定端是否保持在原位

    4. 桌面平放：线缆是否在桌面上平放（高度散布小）

    5. 直线度：端点距离/弧长足够高，且整体偏离首尾连线足够小

    6. 高度限制：线缆最高点不能超过杆顶

    成功条件（全部满足）：
      - threaded_final: 成功穿线
      - cable_low_intersects_pole_segment: 交叉点在杆低处
      - endpoint_region_final: 末端在目标区域
      - endpoint_past_gap_final: 末端已超过杆间隙
      - straightened_final: 同时满足直度比和首尾连线偏差约束
      - settled_on_table_final: 线缆在桌面平放
      - peak_height_excess <= 0: 线缆未超过杆顶
      - anchor_stable_final: 锚点稳定
    """
    cable_points = validate_keypoints(state.cable_points)
    cable_end_pos = np.asarray(state.cable_end_pos, dtype=float)
    anchor_pos = np.asarray(state.anchor_pos, dtype=float)
    pole1_pos = np.asarray(state.pole1_pos, dtype=float)
    pole2_pos = np.asarray(state.pole2_pos, dtype=float)
    endpoint_goal = np.asarray(state.endpoint_goal, dtype=float)

    # --- 杆顶高度和线缆最高点 ---
    pole_top_z = pole1_pos[2] + float(spec.pole_height) / 2.0
    max_cable_height = float(np.max(cable_points[:, 2]))
    peak_height_excess = max(0.0, max_cable_height - (pole_top_z + spec.height_tolerance))

    # --- 末端位置检查 ---
    endpoint_goal_error = float(np.linalg.norm(cable_end_pos - endpoint_goal))
    # 末端在目标区域内：x 方向对齐，y 方向在杆的远离锚点侧，整体距离在容差内
    anchor_to_gap_y = pole1_pos[1] - anchor_pos[1]
    if abs(anchor_to_gap_y) < 1e-6:
        endpoint_on_far_side = cable_end_pos[1] <= pole1_pos[1] - spec.pole_radius
    else:
        endpoint_on_far_side = bool(
            np.sign(anchor_to_gap_y) * (cable_end_pos[1] - pole1_pos[1]) >= spec.pole_radius
        )
    endpoint_region_final = bool(
        abs(cable_end_pos[0] - endpoint_goal[0]) <= spec.goal_tolerance
        and endpoint_on_far_side
        and endpoint_goal_error <= spec.goal_tolerance
    )
    # 末端是否已超过杆间隙（沿从锚点到门的方向）
    if abs(anchor_to_gap_y) < 1e-6:
        endpoint_past_gap_final = bool(cable_end_pos[1] <= pole1_pos[1] - spec.endpoint_past_gap_margin)
    else:
        endpoint_past_gap_final = bool(
            np.sign(anchor_to_gap_y) * (cable_end_pos[1] - pole1_pos[1]) >= spec.endpoint_past_gap_margin
        )

    # --- 锚点稳定性 ---
    anchor_error = float(np.linalg.norm(cable_points[0] - anchor_pos))
    anchor_stable_final = bool(anchor_error < spec.anchor_tolerance)

    # --- 桌面平放检查 ---
    # 以锚点 z 为参考，检查线缆各点的 z 偏差
    table_reference_z = float(anchor_pos[2])
    height_offsets = np.abs(cable_points[:, 2] - table_reference_z)
    tabletop_spread = float(np.percentile(height_offsets, 90))    # 90th percentile 高度散布
    endpoint_height_error = float(abs(cable_end_pos[2] - table_reference_z))
    settled_on_table_final = bool(
        tabletop_spread <= spec.table_settle_tolerance
        and endpoint_height_error <= spec.endpoint_table_tolerance
    )

    # --- 杆间隙计算 ---
    cable_xy = cable_points[:, :2]
    pole1_xy = pole1_pos[:2]
    pole2_xy = pole2_pos[:2]
    # 线缆到杆的最小净空距离
    clearance1 = np.min(np.linalg.norm(cable_xy - pole1_xy[None, :], axis=1)) - spec.pole_radius
    clearance2 = np.min(np.linalg.norm(cable_xy - pole2_xy[None, :], axis=1)) - spec.pole_radius
    min_pole_clearance = float(min(clearance1, clearance2))

    # 间隙走廊的 x 范围
    corridor_min = min(pole1_xy[0], pole2_xy[0]) + spec.pole_radius - spec.gap_margin
    corridor_max = max(pole1_xy[0], pole2_xy[0]) - spec.pole_radius + spec.gap_margin
    corridor_width = max(corridor_max - corridor_min, 1e-6)
    pole_y = pole1_xy[1]
    # 杆连线附近（y 方向走廊深度 1.5 倍范围内）的线缆点
    band = np.abs(cable_xy[:, 1] - pole_y) <= spec.thread_corridor_depth * 1.5
    if not np.any(band):
        min_outer_clearance = float("inf")
    else:
        x_vals = cable_xy[band, 0]
        # 走廊边缘到最近线缆点的距离
        min_outer_clearance = float(np.min(np.minimum(np.abs(x_vals - corridor_min), np.abs(x_vals - corridor_max))))

    # --- 穿线检测主循环 ---
    # 遍历线缆每一段，检测与杆连线的交叉
    pole_dir = pole2_xy - pole1_xy
    low_thread_z_limit = pole_top_z - spec.low_thread_height_margin
    threaded_final = False
    thread_cross_value = 0.0
    gap_cross_x = np.nan
    gap_cross_z = np.nan
    gap_cross_xy = None
    cable_intersects_pole_segment = False
    cable_low_intersects_pole_segment = False
    thread_completion = 0.0  # 穿线完成度（0~1）

    for idx in range(len(cable_xy) - 1):
        p1 = cable_xy[idx]
        p2 = cable_xy[idx + 1]
        # 检测线段与杆连线的 2D 交叉
        intersects, seg_t, pole_t, cross_xy = _segment_intersection_2d(p1, p2, pole1_xy, pole2_xy)
        if intersects:
            x_cross = float(cross_xy[0])
            # 通过线性插值计算交叉点的 z 坐标
            z_cross = float(cable_points[idx, 2] + seg_t * (cable_points[idx + 1, 2] - cable_points[idx, 2]))
            distance_to_pole_segment = 0.0
        else:
            # 未精确交叉，计算最近距离
            distance_to_pole_segment, seg_t, pole_t, cross_xy = _closest_segment_points_2d(p1, p2, pole1_xy, pole2_xy)
            x_cross = float(cross_xy[0])
            z_cross = float(cable_points[idx, 2] + seg_t * (cable_points[idx + 1, 2] - cable_points[idx, 2]))

        # 判断交叉点是否在杆间隙走廊的有效范围内
        # composite 线缆刚性较大，允许交叉点略微超出杆段范围（pole_t 略 <0 或 >1）
        pole_t_margin = getattr(spec, "pole_t_margin", 0.0)
        in_clear_gap = bool(
            pole_t > -pole_t_margin and pole_t < 1.0 + pole_t_margin
            and distance_to_pole_segment <= spec.cable_intersection_tolerance
        )
        if in_clear_gap:
            cable_intersects_pole_segment = True
            dist_to_corridor = 0.0
        else:
            dist_to_corridor = min(abs(x_cross - corridor_min), abs(x_cross - corridor_max))
        # 更新穿线完成度：交叉点越接近走廊中心，完成度越高
        thread_completion = max(
            thread_completion,
            float(np.clip(1.0 - dist_to_corridor / corridor_width, 0.0, 1.0)),
        )
        if not in_clear_gap:
            continue
        # 通过叉积判定穿线方向
        cable_dir = p2 - p1
        cross = pole_dir[0] * cable_dir[1] - pole_dir[1] * cable_dir[0]
        if abs(cross) > spec.thread_cross_threshold and z_cross <= low_thread_z_limit:
            # 成功穿线：方向正确且高度足够低
            threaded_final = True
            cable_low_intersects_pole_segment = True
            thread_cross_value = float(cross)
            gap_cross_x = float(x_cross)
            gap_cross_z = float(z_cross)
            gap_cross_xy = np.asarray(cross_xy, dtype=float).copy()
            thread_completion = 1.0
            break

    # --- 备用穿线检测：基于空间位置 ---
    # 如果没有精确交叉但线缆点同时出现在杆的前后两侧且在间隙内，也算穿线
    in_gap_mask = (
        (cable_points[:, 0] >= corridor_min)
        & (cable_points[:, 0] <= corridor_max)
        & (np.abs(cable_points[:, 1] - pole1_xy[1]) <= spec.thread_corridor_depth)
        & (cable_points[:, 2] <= pole_top_z + spec.height_tolerance)
    )
    front_present = bool(np.any(cable_points[:, 1] >= pole1_xy[1] + spec.thread_front_back_margin))
    back_present = bool(np.any(cable_points[:, 1] <= pole1_xy[1] - spec.thread_front_back_margin))
    gap_occupied = bool(np.any(in_gap_mask))
    if (not threaded_final) and gap_occupied and front_present and back_present:
        thread_completion = max(thread_completion, 1.0)

    # --- 直线度检查 ---
    # 同时检查直度比和整体偏离首尾连线的平均距离，避免仍明显弯曲时过早判成功。
    cable_arc_length = polyline_length(cable_points)
    straightness_ratio_value = straightness_ratio(cable_points)
    line_deviation_value = cable_line_deviation(cable_points)
    straightened_final = bool(
        straightness_ratio_value >= spec.straightness_ratio_threshold
        and line_deviation_value <= spec.straightness_tolerance
    )

    # 检查起点到终点的连线是否穿过杆间隙
    final_line_crosses_gap, _ = _segment_crosses_gap_corridor(
        cable_points[0, :2],
        cable_end_pos[:2],
        pole1_xy,
        pole2_xy,
        pole_radius=spec.pole_radius,
        gap_margin=spec.gap_margin,
    )

    # --- 综合成功判定 ---
    final_success = bool(
        threaded_final
        and cable_low_intersects_pole_segment
        and endpoint_region_final
        and endpoint_past_gap_final
        and straightened_final
        and settled_on_table_final
        and peak_height_excess <= 1e-6
        and anchor_stable_final
    )

    # --- 通过率指标 ---
    # 计算线缆有多少比例已经通过了门（杆间隙）
    gate_center = 0.5 * (pole1_pos + pole2_pos)
    exit_axis = np.array([0.0, -1.0], dtype=float)
    anchor_to_gate_y = pole1_xy[1] - anchor_pos[1]
    if abs(anchor_to_gate_y) > 1e-6:
        exit_axis = np.array([0.0, np.sign(anchor_to_gate_y)], dtype=float)
    raw_passed_ratio = passed_keypoint_ratio(cable_points, gate_center=gate_center, exit_axis=exit_axis, min_projection=0.0)
    expected_exit_ratio = _expected_exit_side_ratio(gate_center, exit_axis, anchor_pos, endpoint_goal)
    # 取关键点比例和弧长比例的较大值，更鲁棒
    exit_side_ratio = max(raw_passed_ratio, _exit_side_arclength_ratio(cable_points, gate_center, exit_axis))
    passed_ratio = float(np.clip(exit_side_ratio / expected_exit_ratio, 0.0, 1.0))

    # 门附近的横向偏差
    gate_band = np.abs(cable_points[:, 1] - pole1_xy[1]) <= spec.thread_corridor_depth
    if np.any(gate_band):
        gate_deviation = float(np.mean(np.abs(cable_points[gate_band, 0] - gate_center[0])))
    else:
        gate_deviation = float(np.min(np.abs(cable_points[:, 1] - pole1_xy[1])))
    cable_on_table = bool(settled_on_table_final and peak_height_excess <= 1e-6)

    # --- 汇总所有指标 ---
    metrics = {
        "endpoint_goal_error_final": endpoint_goal_error,
        "max_cable_height": max_cable_height,
        "peak_height_excess": peak_height_excess,
        "anchor_error_final": anchor_error,
        "anchor_stable_final": anchor_stable_final,
        "tabletop_spread_final": tabletop_spread,
        "min_pole_clearance_final": min_pole_clearance,
        "endpoint_height_error_final": endpoint_height_error,
        "settled_on_table_final": settled_on_table_final,
        "thread_cross_value": thread_cross_value,
        "gap_cross_x": float(gap_cross_x),
        "gap_cross_z": float(gap_cross_z),
        "threaded_final": threaded_final,
        "endpoint_region_final": endpoint_region_final,
        "endpoint_past_gap_final": endpoint_past_gap_final,
        "final_line_crosses_gap": bool(final_line_crosses_gap),
        "cable_intersects_pole_segment": bool(cable_intersects_pole_segment),
        "cable_low_intersects_pole_segment": bool(cable_low_intersects_pole_segment),
        "straightness_error_final": line_deviation_value,
        "straightness_ratio_final": straightness_ratio_value,
        "polyline_length_final": cable_arc_length,
        "straightened_final": straightened_final,
        "thread_completion": float(thread_completion),
        "final_success": final_success,
        "task_success": final_success,
        "min_outer_clearance_final": min_outer_clearance,
        "passed_keypoint_ratio": float(passed_ratio),
        "gate_deviation": float(gate_deviation),
        "post_collision_count": int(post_collision_count),
        "cable_on_table": cable_on_table,
    }
    # 添加不带 _final 后缀的别名，方便外部代码使用
    metrics.update(
        {
            "endpoint_goal_error": metrics["endpoint_goal_error_final"],
            "anchor_error": metrics["anchor_error_final"],
            "tabletop_spread": metrics["tabletop_spread_final"],
            "min_pole_clearance": metrics["min_pole_clearance_final"],
            "min_outer_clearance": metrics["min_outer_clearance_final"],
            "endpoint_height_error": metrics["endpoint_height_error_final"],
            "settled_on_table": metrics["settled_on_table_final"],
            "threaded": metrics["threaded_final"],
            "endpoint_region": metrics["endpoint_region_final"],
            "endpoint_past_gap": metrics["endpoint_past_gap_final"],
            "straightness_error": metrics["straightness_error_final"],
            "straightness_ratio": metrics["straightness_ratio_final"],
            "polyline_length": metrics["polyline_length_final"],
            "straightened": metrics["straightened_final"],
            "success": metrics["final_success"],
        }
    )
    return metrics
