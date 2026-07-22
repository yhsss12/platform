"""
RMB（RoboManipBaselines）链条挂钩任务的指标计算。

本模块实现链条挂到钩子上的任务指标。
该任务要求机器人抓取链条的一端，将其挂到指定位置的钩子上。

成功条件：
  1. 末端靠近钩子：XY 距离 <= hook_radius，高度误差 <= hook_height_tolerance
  2. 挂钩形态：链条的垂直落差 >= min_vertical_drop
     （链条因重力自然下垂，形成悬挂形状）

物理意义：
  成功挂上钩子后，链条一端在钩子附近，其余部分因重力下垂，
  整体呈"倒 U 形"或"J 形"，垂直落差明显。
"""

import numpy as np

from robosuite.utils.dlo.cable_metrics import validate_keypoints


def rmb_chain_hang_on_hook_metrics(
    chain_keypoints,
    chain_end_pos,
    hook_pos,
    *,
    hook_radius=0.05,
    hook_height_tolerance=0.06,
    min_vertical_drop=0.08,
):
    """计算链条挂钩任务的指标。

    Args:
        chain_keypoints: 链条关键点，shape (N, 3)。
        chain_end_pos: 链条末端位置（被抓取端），shape (3,)。
        hook_pos: 钩子位置，shape (3,)。
        hook_radius: 末端到钩子的 XY 距离容差。
        hook_height_tolerance: 末端到钩子的高度容差。
        min_vertical_drop: 链条垂直落差的最小要求。

    Returns:
        dict 包含各项检查结果和综合成功判定（rmb_chain_sparse_success）。
    """
    points = validate_keypoints(chain_keypoints)
    chain_end = np.asarray(chain_end_pos, dtype=float)
    hook = np.asarray(hook_pos, dtype=float)
    if chain_end.shape != (3,) or hook.shape != (3,):
        raise ValueError("chain_end_pos and hook_pos must both have shape (3,)")

    # 末端到钩子的距离（分别计算 XY 和高度）
    end_xy_distance = float(np.linalg.norm(chain_end[:2] - hook[:2]))
    end_height_error = float(abs(chain_end[2] - hook[2]))
    # 垂直落差 = 最高点 - 最低点
    vertical_drop = float(np.max(points[:, 2]) - np.min(points[:, 2]))
    # 链条质心到钩子的 XY 距离（辅助指标）
    center = np.mean(points, axis=0)
    center_to_hook_xy = float(np.linalg.norm(center[:2] - hook[:2]))

    # 条件 1：末端在钩子附近
    end_near_hook = bool(end_xy_distance <= hook_radius and end_height_error <= hook_height_tolerance)
    # 条件 2：有悬挂形态
    hanging_shape = bool(vertical_drop >= min_vertical_drop)
    # 综合判定
    sparse_success = bool(end_near_hook and hanging_shape)

    return {
        "rmb_chain_end_xy_distance": end_xy_distance,
        "rmb_chain_end_height_error": end_height_error,
        "rmb_chain_center_to_hook_xy": center_to_hook_xy,
        "rmb_chain_vertical_drop": vertical_drop,
        "rmb_chain_end_near_hook": end_near_hook,
        "rmb_chain_hanging_shape": hanging_shape,
        "rmb_chain_sparse_success": sparse_success,
        "rmb_chain_hook_radius": float(hook_radius),
        "rmb_chain_hook_height_tolerance": float(hook_height_tolerance),
        "rmb_chain_min_vertical_drop": float(min_vertical_drop),
    }
