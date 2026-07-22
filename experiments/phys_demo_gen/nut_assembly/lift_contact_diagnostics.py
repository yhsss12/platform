"""Contact-aware lift rollout diagnostics（MuJoCo contact + gripper qpos）。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from grasp_sim_search import get_sim_nut_pos
from robosuite_env_loader import get_sim_eef_pose4

FINGER_TIP_BODIES = (
    "gripper0_right_finger_joint1_tip",
    "gripper0_right_finger_joint2_tip",
)
CONTACT_DIST_THRESH = 0.012


def _resolve_body_ids(sim: Any, names: tuple[str, ...]) -> list[int]:
    ids: list[int] = []
    for name in names:
        bid = sim.model.body_name2id(name)
        if bid >= 0:
            ids.append(int(bid))
    return ids


def _finger_contact_proxies(env: Any, nut_pos: np.ndarray) -> tuple[int, int]:
    """finger tip 到 nut 的距离代理 contact（left, right）。"""
    sim = env.sim
    tips = _resolve_body_ids(sim, FINGER_TIP_BODIES)
    if len(tips) < 2:
        return 0, 0
    left_dist = float(np.linalg.norm(sim.data.body_xpos[tips[0]] - nut_pos))
    right_dist = float(np.linalg.norm(sim.data.body_xpos[tips[1]] - nut_pos))
    return int(left_dist < CONTACT_DIST_THRESH), int(right_dist < CONTACT_DIST_THRESH)


LEFT_FINGER_PAD_GEOMS = (
    "gripper0_right_finger1_pad_collision",
    "gripper0_right_finger1_collision",
)
RIGHT_FINGER_PAD_GEOMS = (
    "gripper0_right_finger2_pad_collision",
    "gripper0_right_finger2_collision",
)
NUT_GEOM_PREFIXES = ("SquareNut_g", "RoundNut_g")


def _resolve_geom_ids(sim: Any, names: tuple[str, ...]) -> set[int]:
    ids: set[int] = set()
    for name in names:
        gid = sim.model.geom_name2id(name)
        if gid >= 0:
            ids.add(int(gid))
    return ids


def _resolve_nut_geom_ids(sim: Any, env: Any | None = None) -> set[int]:
    ids: set[int] = set()
    active_prefixes = NUT_GEOM_PREFIXES
    if env is not None and hasattr(env, "nuts") and hasattr(env, "nut_id"):
        nut_name = env.nuts[env.nut_id].name
        if "Square" in nut_name:
            active_prefixes = ("SquareNut_g",)
        elif "Round" in nut_name:
            active_prefixes = ("RoundNut_g",)
    for i in range(sim.model.ngeom):
        name = sim.model.geom_id2name(i) or ""
        if any(name.startswith(p) for p in active_prefixes) and "visual" not in name:
            ids.add(i)
    return ids


def _resolve_gripper_qpos_indices(sim: Any) -> tuple[int, int]:
    left = sim.model.joint_name2id("gripper0_right_finger_joint1")
    right = sim.model.joint_name2id("gripper0_right_finger_joint2")
    left_adr = int(sim.model.jnt_qposadr[left])
    right_adr = int(sim.model.jnt_qposadr[right])
    return left_adr, right_adr


@dataclass
class LiftContactDiagnostics:
    """demo_3 lift rollout 诊断字段。"""

    gripper_qpos_before_close: float = 0.0
    gripper_qpos_after_close: float = 0.0
    left_finger_contact_count: int = 0
    right_finger_contact_count: int = 0
    bilateral_contact_steps: int = 0
    contact_duration: int = 0
    eef_nut_xy_at_close: float = 0.0
    eef_nut_z_at_close: float = 0.0
    eef_z_lift_delta: float = 0.0
    nut_z_lift_delta: float = 0.0
    nut_eef_coupling_ratio: float = 0.0
    nut_xy_slip: float = 0.0
    partial_lift_success: bool = False
    per_step_left_contact: list[int] = field(default_factory=list)
    per_step_right_contact: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "gripper_qpos_before_close": self.gripper_qpos_before_close,
            "gripper_qpos_after_close": self.gripper_qpos_after_close,
            "left_finger_contact_count": self.left_finger_contact_count,
            "right_finger_contact_count": self.right_finger_contact_count,
            "bilateral_contact_steps": self.bilateral_contact_steps,
            "contact_duration": self.contact_duration,
            "eef_nut_xy_at_close": self.eef_nut_xy_at_close,
            "eef_nut_z_at_close": self.eef_nut_z_at_close,
            "eef_z_lift_delta": self.eef_z_lift_delta,
            "nut_z_lift_delta": self.nut_z_lift_delta,
            "nut_eef_coupling_ratio": self.nut_eef_coupling_ratio,
            "nut_xy_slip": self.nut_xy_slip,
            "partial_lift_success": self.partial_lift_success,
        }


class LiftContactTracker:
    """逐步跟踪 gripper-nut contact 与 lift 耦合。"""

    def __init__(self, env: Any):
        sim = env.sim
        self._left_geom_ids = _resolve_geom_ids(sim, LEFT_FINGER_PAD_GEOMS)
        self._right_geom_ids = _resolve_geom_ids(sim, RIGHT_FINGER_PAD_GEOMS)
        self._nut_geom_ids = _resolve_nut_geom_ids(sim, env)
        self._left_qpos_adr, self._right_qpos_adr = _resolve_gripper_qpos_indices(sim)
        self._sim = sim

        self.left_finger_contact_count = 0
        self.right_finger_contact_count = 0
        self.bilateral_contact_steps = 0
        self.contact_duration = 0
        self.per_step_left: list[int] = []
        self.per_step_right: list[int] = []

        self.gripper_qpos_before_close: float | None = None
        self.gripper_qpos_after_close: float | None = None
        self.eef_nut_xy_at_close = 0.0
        self.eef_nut_z_at_close = 0.0
        self._nut_xy_at_close: np.ndarray | None = None
        self._eef_z_at_lift_begin: float | None = None
        self._nut_z_at_lift_begin: float | None = None
        self._close_recorded = False
        self._after_close_recorded = False

    def _gripper_qpos_mean(self) -> float:
        qpos = self._sim.data.qpos
        return float(0.5 * (qpos[self._left_qpos_adr] + qpos[self._right_qpos_adr]))

    def _count_finger_contacts(self, env: Any) -> tuple[int, int]:
        left = 0
        right = 0
        nut_ids = self._nut_geom_ids
        nut_pos = get_sim_nut_pos(env)
        if nut_ids:
            for i in range(self._sim.data.ncon):
                con = self._sim.data.contact[i]
                g1, g2 = int(con.geom1), int(con.geom2)
                if g1 in nut_ids and g2 in self._left_geom_ids:
                    left += 1
                elif g2 in nut_ids and g1 in self._left_geom_ids:
                    left += 1
                if g1 in nut_ids and g2 in self._right_geom_ids:
                    right += 1
                elif g2 in nut_ids and g1 in self._right_geom_ids:
                    right += 1
        prox_left, prox_right = _finger_contact_proxies(env, nut_pos)
        return max(left, prox_left), max(right, prox_right)

    def observe_step(
        self,
        env: Any,
        *,
        step: int,
        grasp_idx: int,
        lift_begin: int,
        lift_end: int,
        contact_window_end: int,
    ) -> None:
        left_c, right_c = self._count_finger_contacts(env)
        self.per_step_left.append(left_c)
        self.per_step_right.append(right_c)
        self.left_finger_contact_count += left_c
        self.right_finger_contact_count += right_c
        if left_c > 0 and right_c > 0:
            self.bilateral_contact_steps += 1
        if left_c > 0 or right_c > 0:
            self.contact_duration += 1

        if step == max(0, grasp_idx - 1) and self.gripper_qpos_before_close is None:
            self.gripper_qpos_before_close = self._gripper_qpos_mean()

        if step == grasp_idx and not self._close_recorded:
            self._close_recorded = True
            eef = get_sim_eef_pose4(env)[:3, 3]
            nut = get_sim_nut_pos(env)
            self.eef_nut_xy_at_close = float(np.linalg.norm(eef[:2] - nut[:2]))
            self.eef_nut_z_at_close = float(eef[2] - nut[2])
            self._nut_xy_at_close = nut[:2].copy()

        if step == min(contact_window_end, grasp_idx + 8) and not self._after_close_recorded:
            self._after_close_recorded = True
            self.gripper_qpos_after_close = self._gripper_qpos_mean()

        if step == lift_begin:
            self._eef_z_at_lift_begin = float(get_sim_eef_pose4(env)[2, 3])
            self._nut_z_at_lift_begin = float(get_sim_nut_pos(env)[2])

    def finalize(
        self,
        env: Any,
        *,
        lift_begin: int,
        lift_end: int,
        nut_z_trace: list[float],
        partial_lift_delta_thresh: float = 0.005,
    ) -> LiftContactDiagnostics:
        if self.gripper_qpos_before_close is None:
            self.gripper_qpos_before_close = self._gripper_qpos_mean()
        if self.gripper_qpos_after_close is None:
            self.gripper_qpos_after_close = self._gripper_qpos_mean()

        eef_z_now = float(get_sim_eef_pose4(env)[2, 3])
        nut_pos = get_sim_nut_pos(env)
        nut_z_now = float(nut_pos[2])

        eef_z_lift_delta = 0.0
        nut_z_lift_delta = 0.0
        if self._eef_z_at_lift_begin is not None:
            eef_z_lift_delta = float(eef_z_now - self._eef_z_at_lift_begin)
        if self._nut_z_at_lift_begin is not None:
            nut_z_lift_delta = float(nut_z_now - self._nut_z_at_lift_begin)
        elif lift_begin < len(nut_z_trace) and lift_end < len(nut_z_trace):
            lb = min(max(0, lift_begin), len(nut_z_trace) - 1)
            le = min(max(lb, lift_end), len(nut_z_trace) - 1)
            nut_z_lift_delta = float(max(nut_z_trace[lb : le + 1]) - nut_z_trace[lb])

        nut_xy_slip = 0.0
        if self._nut_xy_at_close is not None:
            nut_xy_slip = float(np.linalg.norm(nut_pos[:2] - self._nut_xy_at_close))

        nut_eef_coupling_ratio = 0.0
        if abs(eef_z_lift_delta) > 1e-6:
            nut_eef_coupling_ratio = float(np.clip(nut_z_lift_delta / eef_z_lift_delta, -2.0, 2.0))

        partial_lift_success = nut_z_lift_delta >= partial_lift_delta_thresh

        return LiftContactDiagnostics(
            gripper_qpos_before_close=float(self.gripper_qpos_before_close),
            gripper_qpos_after_close=float(self.gripper_qpos_after_close),
            left_finger_contact_count=self.left_finger_contact_count,
            right_finger_contact_count=self.right_finger_contact_count,
            bilateral_contact_steps=self.bilateral_contact_steps,
            contact_duration=self.contact_duration,
            eef_nut_xy_at_close=self.eef_nut_xy_at_close,
            eef_nut_z_at_close=self.eef_nut_z_at_close,
            eef_z_lift_delta=eef_z_lift_delta,
            nut_z_lift_delta=nut_z_lift_delta,
            nut_eef_coupling_ratio=nut_eef_coupling_ratio,
            nut_xy_slip=nut_xy_slip,
            partial_lift_success=partial_lift_success,
            per_step_left_contact=self.per_step_left,
            per_step_right_contact=self.per_step_right,
        )
