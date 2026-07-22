"""
RMB（RoboManipBaselines）线缆穿越杆任务的指标计算。

本模块实现从 RoboManipBaselines 移植的线缆穿越两根杆的任务指标。
该任务要求机器人抓取线缆，从两根竖直杆之间穿过。

成功条件：
  1. 线缆最高点不超过杆顶 + margin（高度限制）
  2. 线缆末端在杆的后方（endpoint_region）
  3. 线缆与杆连线有交叉（gate_intersection）
  4. 交叉方向正确（cross_direction_ok，使用 CCW 测试）

本模块的指标是"稀疏"版本（sparse metrics），只检查关键几何条件，
不涉及复杂的穿线走廊/直线度等详细检查（那是 task_logic.py 的职责）。
"""

import numpy as np

from robosuite.utils.dlo.cable_metrics import validate_keypoints


def _ccw(a, b, c):
    """CCW（Counter-Clockwise）测试的基元。

    判断点 C 是否在向量 AB 的左侧（逆时针方向）。
    数学上，这等价于叉积 (B-A) x (C-A) 的 z 分量是否为正。

    用于快速判断两条线段是否相交（标准的线段相交测试）。
    """
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])


def segment_intersection_2d(a, b, c, d):
    """判断两条 2D 线段 AB 和 CD 是否相交。

    使用 CCW 测试：AB 与 CD 相交当且仅当
      CCW(A,C,D) != CCW(B,C,D) 且 CCW(A,B,C) != CCW(A,B,D)

    这是一个纯布尔测试，不计算交点坐标。
    比参数化方法更快，适合只需判断"是否相交"的场景。
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    d = np.asarray(d, dtype=float)
    return bool(_ccw(a, c, d) != _ccw(b, c, d) and _ccw(a, b, c) != _ccw(a, b, d))


def rmb_cable_pass_between_posts_metrics(
    cable_keypoints,
    cable_end_pos,
    pole1_pos,
    pole2_pos,
    *,
    height_margin=0.01,
    endpoint_y_margin=0.05,
):
    """计算 RMB 线缆穿越杆任务的稀疏指标。

    检查四个方面：
      1. 高度限制：线缆最高点 <= 杆顶 + height_margin
      2. 末端 x 位置：末端 x >= pole2.x（已超过第二根杆）
      3. 末端 y 位置：末端 y <= pole1.y - margin（在杆后方）
      4. 穿越检测：线缆与杆连线有交叉，且方向正确

    穿越方向使用叉积判定：pole_dir x cable_dir > 0 表示正确方向。
    pole_dir = pole2 - pole1，cable_dir = 线缆段方向。

    Args:
        cable_keypoints: 线缆关键点，shape (N, 3)。
        cable_end_pos: 线缆末端位置，shape (3,)。
        pole1_pos / pole2_pos: 两根杆的位置，shape (3,)。
        height_margin: 高度限制的余量。
        endpoint_y_margin: 末端 y 方向的余量。

    Returns:
        dict 包含各项检查结果和综合成功判定（rmb_sparse_success）。
    """
    points = validate_keypoints(cable_keypoints)
    cable_end = np.asarray(cable_end_pos, dtype=float)
    pole1 = np.asarray(pole1_pos, dtype=float)
    pole2 = np.asarray(pole2_pos, dtype=float)
    if cable_end.shape != (3,) or pole1.shape != (3,) or pole2.shape != (3,):
        raise ValueError("cable_end_pos, pole1_pos, and pole2_pos must all have shape (3,)")

    # 1. 高度检查
    max_cable_height = float(np.max(points[:, 2]))
    z_threshold = float(pole1[2] + height_margin)
    height_ok = bool(max_cable_height <= z_threshold)

    # 2 & 3. 末端位置检查
    endpoint_x_ok = bool(cable_end[0] >= pole2[0])
    endpoint_y_ok = bool(cable_end[1] <= pole1[1] - endpoint_y_margin)
    endpoint_region = bool(endpoint_x_ok and endpoint_y_ok)

    # 4. 穿越检测：遍历每一段线缆，检测与杆连线的交叉
    pole1_xy = pole1[:2]
    pole2_xy = pole2[:2]
    pole_dir = pole2_xy - pole1_xy
    intersects = False
    cross_direction_ok = False
    crossing_segment_index = -1
    crossing_value = 0.0

    for idx in range(len(points) - 1):
        p1 = points[idx, :2]
        p2 = points[idx + 1, :2]
        if not segment_intersection_2d(p1, p2, pole1_xy, pole2_xy):
            continue
        intersects = True
        cable_dir = p2 - p1
        # 叉积判定穿越方向
        crossing_value = float(pole_dir[0] * cable_dir[1] - pole_dir[1] * cable_dir[0])
        crossing_segment_index = idx
        if crossing_value > 0.0:
            cross_direction_ok = True
            break

    # 综合判定：所有条件都满足才算成功
    sparse_success = bool(height_ok and endpoint_region and intersects and cross_direction_ok)
    return {
        "rmb_max_cable_height": max_cable_height,
        "rmb_height_threshold": z_threshold,
        "rmb_height_ok": height_ok,
        "rmb_endpoint_x_ok": endpoint_x_ok,
        "rmb_endpoint_y_ok": endpoint_y_ok,
        "rmb_endpoint_region": endpoint_region,
        "rmb_gate_intersection": bool(intersects),
        "rmb_cross_direction_ok": bool(cross_direction_ok),
        "rmb_crossing_segment_index": int(crossing_segment_index),
        "rmb_crossing_value": float(crossing_value),
        "rmb_sparse_success": sparse_success,
    }
