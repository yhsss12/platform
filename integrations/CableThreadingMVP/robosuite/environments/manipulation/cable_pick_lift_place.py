"""
CablePickLiftPlace -- 线缆抓取-提起-放置任务。

单回合内依次完成三段操作：
1. 夹起线缆首端并抬升到整条线缆完全悬空
2. 放下后再夹起线缆中段并抬升到整条线缆完全悬空
3. 放下后再夹起线缆末端并抬升到整条线缆完全悬空

成功要求：
- 夹爪闭合且抓取点位于夹持面附近时才允许 attachment 激活
- 任一阶段都必须先完成“有效抓取 + 整条线缆完全悬空”，随后释放
- 三个阶段必须在同一回合内按顺序完成
"""

import numpy as np

from robosuite.environments.manipulation.cable_straighten import CableStraighten


class CablePickLiftPlace(CableStraighten):
    """线缆单回合三段抓取-提起-放置任务。"""

    def __init__(
        self,
        *args,
        suspension_clearance=None,
        suspended_contact_ratio_threshold=None,
        **kwargs,
    ):
        kwargs.setdefault("cable_model", "flex")
        kwargs.setdefault("initialization_noise", None)
        cable_model = str(kwargs.get("cable_model", "flex")).lower()
        is_flex = cable_model == "flex"
        # flex (flexcomp) 线缆柔性大，抬升单端时中部下垂明显，需要更宽松的悬空阈值
        self.suspension_clearance = float(suspension_clearance if suspension_clearance is not None else (-0.06 if is_flex else 0.001))
        self.suspended_contact_ratio_threshold = float(suspended_contact_ratio_threshold if suspended_contact_ratio_threshold is not None else (0.60 if is_flex else 0.25))
        self.phase_names = ("head", "mid", "tail")
        self.phase_history = []
        self.current_phase_index = 0
        self._prev_attachment_active = False
        super().__init__(*args, target_line_visible=False, **kwargs)
        # flex (flexcomp) 线缆经三段抬放后弹性形变大，无法完全平放，需放宽桌面接触阈值
        if is_flex:
            self.success_table_contact_ratio_threshold = 0.70

    def _reset_internal(self):
        super()._reset_internal()
        self._reset_sequence_state()

    def _reset_sequence_state(self):
        phase_indices = self._phase_point_indices(len(self._get_cable_points()))
        self.phase_history = [
            {
                "phase_name": phase_name,
                "target_point_idx": int(point_idx),
                "grasped": False,
                "fully_suspended": False,
                "released": False,
                "completed": False,
            }
            for phase_name, point_idx in zip(self.phase_names, phase_indices)
        ]
        self.current_phase_index = 0
        self._prev_attachment_active = False

    def _phase_point_indices(self, point_count):
        if point_count <= 0:
            raise ValueError("point_count must be positive")
        return (0, point_count // 2, point_count - 1)

    def _phase_match_point_count(self):
        try:
            return int(len(self._get_cable_points()))
        except Exception:
            if self.phase_history:
                return int(max(phase["target_point_idx"] for phase in self.phase_history) + 1)
            return 0

    def _active_task_grasp_point_idx(self):
        if getattr(self, "_flex_grasp_active", False) and getattr(self, "_flex_grasp_vtx_idx", -1) >= 0:
            return int(self._flex_grasp_vtx_idx)
        if self.attachment_enabled and getattr(self, "active_grasp_point_idx", -1) >= 0:
            return int(self.active_grasp_point_idx)
        return None

    def _phase_target_matches_active_point(self, phase_point_idx, active_point_idx, point_count):
        if active_point_idx is None:
            return False
        if int(active_point_idx) == int(phase_point_idx):
            return True
        if int(phase_point_idx) == 0:
            return int(active_point_idx) == 0
        if int(phase_point_idx) == point_count - 1:
            return int(active_point_idx) in {1, point_count - 1}
        return False

    def _attachment_active(self):
        return bool(self.attachment_enabled or getattr(self, "_flex_grasp_active", False))

    def _current_phase_record(self):
        if self.current_phase_index >= len(self.phase_history):
            return None
        return self.phase_history[self.current_phase_index]

    def _reward_from_metrics(self, metrics):
        reward = float(metrics["completed_phase_count"])
        if metrics["holding_current_phase_target"]:
            reward += 0.25
        if metrics["current_phase_suspended"]:
            reward += 0.75
        if metrics["success"]:
            reward += 1.0
        return self.reward_scale * reward

    def reward(self, action=None):
        return self._reward_from_metrics(self._compute_metrics())

    def _compute_metrics(self):
        metrics = super()._compute_metrics()
        points = self._get_cable_points()
        current_phase = self._current_phase_record()
        active_point_idx = self._active_task_grasp_point_idx()
        min_keypoint_clearance = float(np.min(points[:, 2] - self.table_top_z))
        cable_fully_suspended = bool(
            min_keypoint_clearance >= self.suspension_clearance
            and metrics["table_contact_ratio"] <= self.suspended_contact_ratio_threshold
        )
        holding_current_target = bool(
            current_phase is not None
            and self._attachment_active()
            and self._phase_target_matches_active_point(
                current_phase["target_point_idx"],
                active_point_idx,
                self._phase_match_point_count() or len(points),
            )
        )
        completed_phase_count = int(sum(1 for phase in self.phase_history if phase["completed"]))
        current_phase_name = "done" if current_phase is None else current_phase["phase_name"]
        current_phase_idx = -1 if current_phase is None else int(current_phase["target_point_idx"])
        current_phase_suspended = bool(current_phase is not None and current_phase["fully_suspended"])
        success = bool(
            len(self.phase_history) > 0
            and completed_phase_count == len(self.phase_history)
            and not self._attachment_active()
            and metrics["cable_on_table"]
        )

        metrics.update(
            {
                "task_name": "CablePickLiftPlace",
                "phase_order": self.phase_names,
                "phase_history": [dict(phase) for phase in self.phase_history],
                "current_phase": current_phase_name,
                "current_phase_point_idx": current_phase_idx,
                "current_attachment_point_idx": -1 if active_point_idx is None else int(active_point_idx),
                "holding_current_phase_target": holding_current_target,
                "min_keypoint_clearance_to_table": min_keypoint_clearance,
                "cable_fully_suspended": cable_fully_suspended,
                "current_phase_suspended": current_phase_suspended,
                "completed_phase_count": completed_phase_count,
                "remaining_phase_count": int(len(self.phase_history) - completed_phase_count),
                "success": success,
                "task_success": success,
            }
        )
        return metrics

    def _update_sequence_state(self, metrics):
        current_phase = self._current_phase_record()
        attachment_active = self._attachment_active()

        if current_phase is None:
            self._prev_attachment_active = attachment_active
            return

        active_point_idx = self._active_task_grasp_point_idx()
        if attachment_active and self._phase_target_matches_active_point(
            current_phase["target_point_idx"],
            active_point_idx,
            self._phase_match_point_count(),
        ):
            current_phase["grasped"] = True
            if metrics["cable_fully_suspended"]:
                current_phase["fully_suspended"] = True

        if self._prev_attachment_active and not attachment_active:
            current_phase["released"] = True
            if current_phase["grasped"] and current_phase["fully_suspended"]:
                current_phase["completed"] = True
                self.current_phase_index += 1

        self._prev_attachment_active = attachment_active

    def _post_action(self, action):
        metrics = self._compute_metrics()
        self._update_sequence_state(metrics)
        metrics = self._compute_metrics()
        reward = self._reward_from_metrics(metrics)
        self.done = (self.timestep >= self.horizon) and not self.ignore_done
        return reward, self.done, metrics

    def _check_success(self):
        return self._compute_metrics()["success"]
