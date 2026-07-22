"""
CableAtomicTest — 原子动作可靠性测试任务。

根据线缆类型的 graspable_point_count 动态决定阶段数：
- flex: 3 阶段（head/mid/tail），物理模式顶点操控
- composite: 3 阶段（head/mid/tail），weld 约束
- composite_soft: 2 阶段（head/tail），仅端点 weld 约束

在每阶段 lift 之后增加 move_eef → lower → release 的放置流程。

成功判据：
  - 所有阶段完成（抓取 + 悬空 + 释放）
  - 每阶段放置误差在阈值内
  - 线缆最终在桌面上
  - 无 attachment 残留
"""

import numpy as np

from robosuite.environments.manipulation.cable_pick_lift_place import CablePickLiftPlace


# 不同线缆类型的阶段名称和放置半径
_CABLE_PHASE_CONFIG = {
    "flex": {
        "phase_names": ("head", "mid", "tail"),
        "placement_radius": (0.08, 0.15),
    },
    "composite": {
        "phase_names": ("head", "mid", "tail"),
        "placement_radius": (0.05, 0.10),
    },
    "composite_soft": {
        "phase_names": ("head", "tail"),
        "placement_radius": (0.05, 0.10),
    },
    "composite_softened": {
        "phase_names": ("head", "tail"),
        "placement_radius": (0.05, 0.10),
    },
    "rmb": {
        "phase_names": ("head", "mid", "tail"),
        "placement_radius": (0.05, 0.10),
    },
}

# 每种线缆的放置误差阈值（米）
_PLACEMENT_ERROR_THRESHOLD = {
    "flex": 0.20,
    "composite": 0.30,
    "composite_soft": 0.30,
    "composite_softened": 0.30,
    "rmb": 0.30,
}


class CableAtomicTest(CablePickLiftPlace):
    """原子动作可靠性测试：抓取 → 搬运 → 精确放置。"""

    def __init__(self, *args, placement_radius_range=None, **kwargs):
        cable_model = str(kwargs.get("cable_model", "flex")).lower()
        config = _CABLE_PHASE_CONFIG.get(cable_model, _CABLE_PHASE_CONFIG["flex"])

        # 保存阶段名配置，不传给基类（基类不接受 phase_names 参数）
        self._atomic_phase_names = config["phase_names"]
        self.placement_radius_range = tuple(placement_radius_range or config["placement_radius"])
        self._place_targets = {}
        self._phase_place_errors = {}
        self._cable_model_type = cable_model

        # flex: 宽松悬空阈值
        if cable_model == "flex":
            kwargs.setdefault("suspension_clearance", -0.10)
            kwargs.setdefault("suspended_contact_ratio_threshold", 0.80)
        # composite_soft: 长链，抬升一端时大部分点仍在桌面
        elif cable_model in ("composite_soft", "composite_softened"):
            kwargs.setdefault("suspension_clearance", -0.05)
            kwargs.setdefault("suspended_contact_ratio_threshold", 0.95)

        super().__init__(*args, **kwargs)

        # 覆盖 CablePickLiftPlace 硬编码的 phase_names
        self.phase_names = self._atomic_phase_names

        # 放宽桌面接触阈值（多次抬放后线缆无法完全平放）
        if cable_model == "flex":
            self.success_table_contact_ratio_threshold = 0.65
        elif cable_model in ("composite", "composite_cable", "rmb"):
            self.success_table_contact_ratio_threshold = 0.80

    def _reset_sequence_state(self):
        """使用动态 phase_names 重建状态（覆盖 CablePickLiftPlace 的硬编码版本）。"""
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
        """根据线缆类型和阶段名动态返回抓取点索引。"""
        n_phases = len(self.phase_names)
        if n_phases == 2:
            # 仅端点：head=首点, tail=末点
            return (0, point_count - 1)
        # 默认 3 阶段：head=首点, mid=中点, tail=末点
        return (0, point_count // 2, point_count - 1)

    def _reset_internal(self):
        super()._reset_internal()
        self._sample_place_targets()
        self._phase_place_errors = {}

    def _sample_place_targets(self):
        """在桌面上为每个阶段采样一个随机放置目标。"""
        points = self._get_cable_points()
        rng = np.random.RandomState()
        table_xy_limit = 0.28

        for phase in self.phase_history:
            idx = phase["target_point_idx"]
            origin_xy = points[idx, :2].copy()
            for _attempt in range(20):
                angle = rng.uniform(0, 2 * np.pi)
                dist = rng.uniform(*self.placement_radius_range)
                target_xy = origin_xy + dist * np.array([np.cos(angle), np.sin(angle)])
                if np.all(np.abs(target_xy) < table_xy_limit):
                    break
            self._place_targets[phase["phase_name"]] = target_xy.astype(float)

    def _compute_metrics(self):
        metrics = super()._compute_metrics()

        # 计算每阶段放置误差
        points = self._get_cable_points()
        for phase in self.phase_history:
            name = phase["phase_name"]
            if phase["completed"] and name in self._place_targets:
                idx = phase["target_point_idx"]
                actual_xy = points[idx, :2]
                target_xy = self._place_targets[name]
                self._phase_place_errors[name] = float(np.linalg.norm(actual_xy - target_xy))

        mean_place_error = (
            float(np.mean(list(self._phase_place_errors.values())))
            if self._phase_place_errors else float("inf")
        )

        # 动态放置误差阈值
        error_threshold = _PLACEMENT_ERROR_THRESHOLD.get(self._cable_model_type, 0.30)
        all_placed = (
            len(self._phase_place_errors) == len(self.phase_history)
            and all(e < error_threshold for e in self._phase_place_errors.values())
        )
        success = bool(
            metrics.get("success", False)
            and all_placed
        )

        metrics.update({
            "task_name": "CableAtomicTest",
            "cable_model_type": self._cable_model_type,
            "phase_count": len(self.phase_names),
            "place_targets": {k: v.tolist() for k, v in self._place_targets.items()},
            "phase_place_errors": dict(self._phase_place_errors),
            "mean_place_error": mean_place_error,
            "placement_error_threshold": error_threshold,
            "success": success,
            "task_success": success,
        })
        return metrics

    def _reward_from_metrics(self, metrics):
        reward = float(metrics.get("completed_phase_count", 0))
        if metrics.get("holding_current_phase_target", False):
            reward += 0.25
        if metrics.get("current_phase_suspended", False):
            reward += 0.75
        mean_err = metrics.get("mean_place_error", float("inf"))
        if mean_err < 0.05:
            reward += 1.5
        elif mean_err < 0.10:
            reward += 1.0
        elif mean_err < 0.15:
            reward += 0.5
        elif mean_err < 0.25:
            reward += 0.25
        if metrics.get("success", False):
            reward += 2.0
        return self.reward_scale * reward

    def get_place_target(self, phase_name):
        return self._place_targets.get(phase_name)

    def _check_success(self):
        return self._compute_metrics()["success"]
