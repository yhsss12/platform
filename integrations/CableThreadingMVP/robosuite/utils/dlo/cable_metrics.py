"""
线缆几何基元计算库。

本模块提供对 3D 离散线缆点（keypoints）的基础几何运算，包括：
  - 弧长、端点距离、直度比等形状描述子
  - 点到线段/折线的距离计算
  - 关键点覆盖率和门穿越检测

所有函数均接受 shape (N, 3) 的 numpy 数组作为输入，
内部通过 validate_keypoints 确保数据合法性。

数学约定：
  - keypoints 是有序的离散采样点，相邻两点构成一段线段
  - XY 平面用于大多数几何判定，Z 方向用于高度检查
"""

import numpy as np


def validate_keypoints(keypoints, min_points=2):
    """校验关键点数组的合法性。

    检查：
      1. shape 必须是 (N, 3) 的 2D 数组
      2. N >= min_points（至少需要 2 个点才能构成线段）
      3. 所有值必须是有限数（不能有 NaN 或 inf）

    Returns:
        转换后的 float64 数组

    Raises:
        ValueError: 任一检查不通过时
    """
    points = np.asarray(keypoints, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected keypoints with shape (N, 3), got {points.shape}")
    if points.shape[0] < min_points:
        raise ValueError(f"Expected at least {min_points} keypoints, got {points.shape[0]}")
    if not np.all(np.isfinite(points)):
        raise ValueError("Keypoints contain NaN or infinite values")
    return points


def endpoint_distance(keypoints):
    """计算线缆两端点之间的欧氏距离。

    返回 ||points[-1] - points[0]||，即首尾端点的直线距离。
    这是衡量线缆"展开程度"的最简单指标。
    """
    points = validate_keypoints(keypoints)
    return float(np.linalg.norm(points[-1] - points[0]))


def polyline_length(keypoints):
    """计算折线的总弧长。

    弧长 = sum(||points[i+1] - points[i]||)，即所有相邻线段长度之和。
    这是线缆的"真实长度"，始终 >= 端点距离。

    数学上，这是对连续曲线的离散弧长近似。
    """
    points = validate_keypoints(keypoints)
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def straightness_ratio(keypoints, eps=1e-8):
    """计算直度比 = 端点距离 / 弧长。

    值域 [0, 1]：
      - 1.0 表示完全直线（端点距离 == 弧长）
      - 接近 0 表示高度弯曲

    这是衡量线缆"有多直"的标准指标。
    eps 参数防止除零。
    """
    points = validate_keypoints(keypoints)
    return endpoint_distance(points) / max(polyline_length(points), float(eps))


def cable_centroid(keypoints):
    """计算线缆所有关键点的几何中心（质心）。

    返回 shape (3,) 的坐标，即所有点各维度的均值。
    用于衡量线缆的"整体位置"。
    """
    points = validate_keypoints(keypoints)
    return np.mean(points, axis=0)


def line_deviation(keypoints, line_start=None, line_end=None):
    """计算线缆各点到参考线段的平均垂直距离。

    参考线段由 line_start 和 line_end 定义（默认为线缆首尾端点）。
    算法：
      1. 将每个点投影到参考线段所在的直线上（参数 t）
      2. clip t 到 [0, 1]，确保投影点在线段范围内
      3. 计算每个点到其投影点的距离
      4. 返回平均距离

    这衡量了线缆偏离理想直线的程度。
    """
    points = validate_keypoints(keypoints)
    a = points[0] if line_start is None else np.asarray(line_start, dtype=float)
    b = points[-1] if line_end is None else np.asarray(line_end, dtype=float)
    if a.shape != (3,) or b.shape != (3,):
        raise ValueError("line_start and line_end must have shape (3,)")
    ab = b - a
    ab_norm_sq = float(np.dot(ab, ab))
    if ab_norm_sq <= 1e-12:
        raise ValueError("Line endpoints are too close to define a line")
    # 投影参数 t = dot(P-A, AB) / ||AB||^2，clip 到 [0,1]
    t = np.clip(((points - a) @ ab) / ab_norm_sq, 0.0, 1.0)
    projections = a + t[:, None] * ab
    return float(np.mean(np.linalg.norm(points - projections, axis=1)))


def nearest_cable_segment(point, keypoints):
    """找到距离给定点最近的线缆线段。

    遍历所有相邻点对构成的线段，将查询点投影到每段上，
    返回最近线段的索引、距离和最近点坐标。

    Args:
        point: 查询点，shape (3,)。
        keypoints: 线缆关键点，shape (N, 3)。

    Returns:
        (segment_index, distance, closest_point)
        segment_index: 最近线段的起始点索引（0 ~ N-2）
        distance: 查询点到最近线段的距离
        closest_point: 线段上最近点的坐标
    """
    points = validate_keypoints(keypoints)
    query = np.asarray(point, dtype=float)
    if query.shape != (3,):
        raise ValueError(f"Expected point with shape (3,), got {query.shape}")

    starts = points[:-1]
    ends = points[1:]
    segs = ends - starts
    # 投影参数 t，向量化计算所有线段
    denom = np.sum(segs * segs, axis=1)
    safe = np.maximum(denom, 1e-12)
    t = np.clip(np.sum((query - starts) * segs, axis=1) / safe, 0.0, 1.0)
    closest = starts + t[:, None] * segs
    distances = np.linalg.norm(closest - query, axis=1)
    idx = int(np.argmin(distances))
    return idx, float(distances[idx]), closest[idx]


def gripper_to_cable_distance(gripper_pos, keypoints):
    """计算夹爪位置到线缆的最短距离。

    这是 nearest_cable_segment 的便捷封装，只返回距离值。
    用于专家策略中判断夹爪是否足够接近线缆。
    """
    _, distance, _ = nearest_cable_segment(gripper_pos, keypoints)
    return distance


def keypoint_goal_coverage(keypoints, goal_center, goal_radius=None, goal_half_size=None):
    """计算有多少比例的关键点落在目标区域内。

    目标区域可以是圆形（goal_radius）或矩形（goal_half_size）。
    只在 XY 平面上判定，忽略 Z 坐标。

    Args:
        keypoints: 线缆关键点。
        goal_center: 目标区域中心，shape (2,) 或 (3,)。
        goal_radius: 圆形目标的半径（与 goal_half_size 二选一）。
        goal_half_size: 矩形目标的半边长，标量或 shape (2,)。

    Returns:
        0.0 ~ 1.0 之间的覆盖率。
    """
    points = validate_keypoints(keypoints)
    center = np.asarray(goal_center, dtype=float)
    if center.shape not in {(2,), (3,)}:
        raise ValueError("goal_center must have shape (2,) or (3,)")
    center_xy = center[:2]
    point_xy = points[:, :2]

    if goal_radius is not None:
        inside = np.linalg.norm(point_xy - center_xy, axis=1) <= float(goal_radius)
    elif goal_half_size is not None:
        half = np.asarray(goal_half_size, dtype=float)
        if half.shape == ():
            half = np.array([float(half), float(half)], dtype=float)
        if half.shape != (2,):
            raise ValueError("goal_half_size must be scalar or shape (2,)")
        inside = np.all(np.abs(point_xy - center_xy) <= half, axis=1)
    else:
        raise ValueError("Specify either goal_radius or goal_half_size")
    return float(np.mean(inside))


def passed_keypoint_ratio(keypoints, gate_center, exit_axis=(0.0, -1.0), min_projection=0.0):
    """计算有多少比例的关键点已通过"门"（gate）。

    "门"由 gate_center（门中心）和 exit_axis（出口方向）定义。
    一个点被认为"通过了门"当且仅当它在 exit_axis 方向上的投影 >= min_projection。

    用于穿线任务中判断线缆是否已经穿过杆间隙。

    Args:
        keypoints: 线缆关键点。
        gate_center: 门的中心位置。
        exit_axis: 出口方向向量（会被归一化）。
        min_projection: 最小投影值，低于此值的点不算通过。

    Returns:
        0.0 ~ 1.0 之间的比例。
    """
    points = validate_keypoints(keypoints)
    center = np.asarray(gate_center, dtype=float)
    axis = np.asarray(exit_axis, dtype=float)
    if center.shape not in {(2,), (3,)}:
        raise ValueError("gate_center must have shape (2,) or (3,)")
    if axis.shape not in {(2,), (3,)}:
        raise ValueError("exit_axis must have shape (2,) or (3,)")
    axis_xy = axis[:2]
    norm = float(np.linalg.norm(axis_xy))
    if norm <= 1e-12:
        raise ValueError("exit_axis must be non-zero")
    axis_xy = axis_xy / norm
    # 计算每个点在出口方向上的投影
    projection = (points[:, :2] - center[:2]) @ axis_xy
    return float(np.mean(projection >= float(min_projection)))
