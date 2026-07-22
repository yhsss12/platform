"""Shared helpers for cable physical grasp checks.

These helpers intentionally operate on an env instance instead of introducing
another inheritance layer. This keeps the API usable from both CableBaseEnv
and task-specific envs such as CableThreading.
"""

import numpy as np


def get_gripper_fingerpad_geom_groups(env):
    """Return left / right fingerpad geom groups from the active gripper."""
    arm = env.robots[0].arms[0]
    gripper = env.robots[0].gripper[arm]
    return [
        list(gripper.important_geoms.get("left_fingerpad", [])),
        list(gripper.important_geoms.get("right_fingerpad", [])),
    ]


def get_fingerpad_group_center(env, geom_names):
    """Return the world-space center of one fingerpad geom group."""
    positions = []
    for geom_name in geom_names:
        try:
            geom_id = env.sim.model.geom_name2id(geom_name)
        except (KeyError, ValueError):
            continue
        positions.append(np.asarray(env.sim.data.geom_xpos[geom_id], dtype=float))
    if not positions:
        return None
    return np.mean(np.asarray(positions, dtype=float), axis=0)


def get_gripper_fingerpad_midpoint(env):
    """Return the midpoint between the two fingerpad centers."""
    left_geoms, right_geoms = get_gripper_fingerpad_geom_groups(env)
    left_center = get_fingerpad_group_center(env, left_geoms)
    right_center = get_fingerpad_group_center(env, right_geoms)
    if left_center is None or right_center is None:
        return None
    return 0.5 * (left_center + right_center)


def get_fingerpad_gap_width(env):
    """Return the distance between left and right fingerpad centers."""
    left_geoms, right_geoms = get_gripper_fingerpad_geom_groups(env)
    left_center = get_fingerpad_group_center(env, left_geoms)
    right_center = get_fingerpad_group_center(env, right_geoms)
    if left_center is None or right_center is None:
        return np.inf
    return float(np.linalg.norm(right_center - left_center))


def get_gripper_clamp_center_offset(env):
    """Return the offset from grip site to fingerpad midpoint."""
    midpoint = get_gripper_fingerpad_midpoint(env)
    if midpoint is None:
        return np.zeros(3, dtype=float)
    return np.asarray(midpoint, dtype=float) - env._get_gripper_site_position()


def is_point_between_gripper_fingerpads(env, point, *, max_distance_fallback=0.0):
    """Check whether a point lies inside the fingerpad clamp corridor."""
    left_geoms, right_geoms = get_gripper_fingerpad_geom_groups(env)
    left_center = get_fingerpad_group_center(env, left_geoms)
    right_center = get_fingerpad_group_center(env, right_geoms)
    if left_center is None or right_center is None:
        return False

    point = np.asarray(point, dtype=float)
    finger_axis = right_center - left_center
    axis_len_sq = float(np.dot(finger_axis, finger_axis))
    if axis_len_sq < 1e-10:
        return False

    projection = float(np.dot(point - left_center, finger_axis) / axis_len_sq)
    margin = float(getattr(env, "_attach_finger_axis_margin", 0.0))
    if projection < margin or projection > 1.0 - margin:
        return False

    closest = left_center + projection * finger_axis
    distance_to_clamp_line = float(np.linalg.norm(point - closest))
    cable_radius = float(getattr(env, "cable_radius", 0.0))
    corridor_radius = max(
        float(getattr(env, "_attach_between_fingers_distance", 0.0)),
        cable_radius + float(max_distance_fallback),
    )
    result = distance_to_clamp_line <= corridor_radius
    return result


def get_cable_contact_geoms(env):
    """Return cable geom names used for grasp contact detection."""
    geoms = list(getattr(getattr(env, "cable", None), "contact_geoms", []))
    if geoms:
        return geoms
    return [
        env.sim.model.geom_id2name(geom_id)
        for geom_id in range(env.sim.model.ngeom)
        if (env.sim.model.geom_id2name(geom_id) or "").startswith(("cable_", "Flex"))
    ]


def count_contacts_between(env, geom_group, object_geoms):
    """Count contacts between one gripper geom group and cable geoms."""
    geom_group = set(geom_group)
    object_geoms = set(object_geoms)
    count = 0
    for idx in range(env.sim.data.ncon):
        contact = env.sim.data.contact[idx]
        if contact.geom1 < 0 or contact.geom2 < 0:
            if getattr(env, "_is_flex_cable", False):
                name1 = env.sim.model.geom_id2name(contact.geom1) if contact.geom1 >= 0 else None
                name2 = env.sim.model.geom_id2name(contact.geom2) if contact.geom2 >= 0 else None
                if (name1 in geom_group and contact.geom2 < 0) or (name2 in geom_group and contact.geom1 < 0):
                    count += 1
            continue
        name1 = env.sim.model.geom_id2name(contact.geom1)
        name2 = env.sim.model.geom_id2name(contact.geom2)
        if (name1 in geom_group and name2 in object_geoms) or (name2 in geom_group and name1 in object_geoms):
            count += 1
    return count


def get_physical_grasp_contact_sides(env):
    """Return left / right contact counts between fingerpads and cable."""
    cable_geoms = get_cable_contact_geoms(env)
    left_geoms, right_geoms = get_gripper_fingerpad_geom_groups(env)
    left_count = count_contacts_between(env, left_geoms, cable_geoms)
    right_count = count_contacts_between(env, right_geoms, cable_geoms)
    return left_count, right_count
