from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from phygen.adapters.base_adapter import BasePhyGenAdapter, TaskSpec

try:
    from phygen.adapters.mimicgen.coffee_repair import (
        load_failed_contexts as _external_load_failed_contexts,
        make_candidate as _external_make_candidate,
    )
except Exception:  # pragma: no cover
    _external_load_failed_contexts = None
    _external_make_candidate = None


class CoffeePreparationAdapter(BasePhyGenAdapter):
    CONTEXT_KEYS = [
        "energy",
        "pod_xy",
        "pod_z",
        "mug_xy",
        "mug_z",
        "machine_xy",
        "stage_progress",
        "drop_penalty",
        "task_order_penalty",
    ]

    CONTEXT_SCALES = {
        "energy": 30.0,
        "pod_xy": 0.10,
        "pod_z": 0.06,
        "mug_xy": 0.12,
        "mug_z": 0.06,
        "machine_xy": 0.35,
        "stage_progress": 1.0,
        "drop_penalty": 0.05,
        "task_order_penalty": 1.0,
    }

    COMPONENT_KEYS = [
        "E_xy",
        "E_transport",
        "E_lift",
        "E_contact",
        "E_stage",
        "E_dynamics",
        "E_slip",
        "E_coupling",
    ]
    COMPONENT_WEIGHTS = [1.2, 1.0, 1.1, 1.1, 1.3, 0.9, 1.0, 1.2]

    RESIDUAL_GROUPS = {
        "geometry": ["E_xy", "E_transport", "E_lift"],
        "contact": ["E_contact", "E_stage"],
        "dynamics": ["E_dynamics", "E_slip"],
        "coupling": ["E_coupling"],
    }

    OPTIONAL_MUJOCO_RESIDUAL_KEYS = {
        "contact": ["contact_residual", "contact_loss", "support_loss", "unstable_contact"],
        "penetration": ["penetration", "max_penetration", "penetration_depth"],
        "slip": ["slip", "slip_residual", "object_slip"],
        "velocity": ["velocity_jump", "object_velocity", "max_object_velocity"],
        "smoothness": ["action_smoothness", "action_delta", "action_jerk"],
    }

    THETA_DISC_KEYS = [
        "select_src_per_subtask",
        "transform_first_robot_pose",
        "interpolate_from_last_target_pose",
        "selection_strategy_nearest_neighbor_object",
        "selection_strategy_random",
    ]

    THETA_CONT_KEYS = [
        "action_noise",
        "num_interpolation_steps",
        "num_fixed_steps",
        "offset_lo",
        "offset_hi",
        "offset_width",
        "offset_center",
        "nn_k",
    ]

    THETA_CONT_INDEPENDENT_KEYS = [
        "action_noise",
        "num_interpolation_steps",
        "num_fixed_steps",
        "offset_lo",
        "offset_hi",
        "nn_k",
    ]
    THETA_CONT_INDEPENDENT_IDXS = [0, 1, 2, 3, 4, 7]
    OFFSET_LO_CONT_IDX = 3
    OFFSET_HI_CONT_IDX = 4
    OFFSET_WIDTH_CONT_IDX = 5
    OFFSET_CENTER_CONT_IDX = 6

    def task_spec(self) -> TaskSpec:
        return TaskSpec(
            task_name="coffee_preparation",
            context_keys=self.CONTEXT_KEYS,
            context_scales=self.CONTEXT_SCALES,
            component_keys=self.COMPONENT_KEYS,
            component_weights=self.COMPONENT_WEIGHTS,
            residual_groups=self.RESIDUAL_GROUPS,
            optional_residual_keys=self.OPTIONAL_MUJOCO_RESIDUAL_KEYS,
            theta_disc_keys=self.THETA_DISC_KEYS,
            theta_cont_keys=self.THETA_CONT_KEYS,
            theta_cont_independent_keys=self.THETA_CONT_INDEPENDENT_KEYS,
            theta_cont_independent_idxs=self.THETA_CONT_INDEPENDENT_IDXS,
            theta_cont_derived_keys=["offset_width", "offset_center"],
            output_model_name="coffee_preparation_failed_conditioned_pinn.pt",
            output_plan_name="pinn_utility_boundary_union_candidate_plan.jsonl",
            output_summary_name="pinn_utility_boundary_union_summary.json",
        )

    @staticmethod
    def _metric(metrics: dict[str, float], keys: list[str], default: float = 0.0) -> float:
        for key in keys:
            if key in metrics and metrics[key] is not None:
                try:
                    return float(metrics[key])
                except (TypeError, ValueError):
                    continue
        return default

    @staticmethod
    def _clip01(value: float) -> float:
        return float(np.clip(value, 0.0, 1.0))

    def theta_to_features(self, theta: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, float]:
        strategy = theta.get("selection_strategy", "nearest_neighbor_object")
        offset = theta.get("offset_range", [10, 20])
        lo, hi = float(offset[0]), float(offset[1])
        theta_disc = np.array(
            [
                float(bool(theta.get("select_src_per_subtask", False))),
                float(bool(theta.get("transform_first_robot_pose", False))),
                float(bool(theta.get("interpolate_from_last_target_pose", True))),
                float(strategy == "nearest_neighbor_object"),
                float(strategy == "random"),
            ],
            dtype=np.float32,
        )
        theta_cont = np.array(
            [
                float(theta.get("action_noise", 0.05)) / 0.10,
                float(theta.get("num_interpolation_steps", 5)) / 15.0,
                float(theta.get("num_fixed_steps", 0)) / 10.0,
                lo / 25.0,
                hi / 25.0,
                (hi - lo) / 25.0,
                (0.5 * (hi + lo)) / 25.0,
                float(theta.get("nn_k", 3)) / 10.0,
            ],
            dtype=np.float32,
        )
        independent = np.array(
            [
                float(theta.get("action_noise", 0.05)) / 0.10,
                float(theta.get("num_interpolation_steps", 5)) / 15.0,
                float(theta.get("num_fixed_steps", 0)) / 10.0,
                lo / 25.0,
                hi / 25.0,
                float(theta.get("nn_k", 3)) / 10.0,
            ],
            dtype=np.float32,
        )
        edge = np.minimum(np.clip(independent, 0.0, 1.0), 1.0 - np.clip(independent, 0.0, 1.0))
        boundary_bonus = float(1.0 - np.mean(edge))
        return theta_disc, theta_cont, boundary_bonus

    def build_residual_targets(self, metrics: dict[str, float] | None, problematic: bool = False) -> np.ndarray:
        if problematic or metrics is None:
            return np.ones(len(self.COMPONENT_KEYS), dtype=np.float32)

        pod_xy = float(metrics.get("pod_xy", 0.10))
        pod_z = float(metrics.get("pod_z", 0.05))
        mug_xy = float(metrics.get("mug_xy", 0.12))
        mug_z = float(metrics.get("mug_z", 0.05))
        machine_xy = float(metrics.get("machine_xy", 0.20))
        stage_progress = float(metrics.get("stage_progress", 0.0))
        drop = float(metrics.get("drop_penalty", 0.0))
        order_pen = float(metrics.get("task_order_penalty", 0.0))

        contact_res = self._metric(metrics, self.OPTIONAL_MUJOCO_RESIDUAL_KEYS["contact"], 0.0)
        penetration = self._metric(metrics, self.OPTIONAL_MUJOCO_RESIDUAL_KEYS["penetration"], 0.0)
        slip_res = self._metric(metrics, self.OPTIONAL_MUJOCO_RESIDUAL_KEYS["slip"], 0.0)
        vel_res = self._metric(metrics, self.OPTIONAL_MUJOCO_RESIDUAL_KEYS["velocity"], 0.0)
        smooth_res = self._metric(metrics, self.OPTIONAL_MUJOCO_RESIDUAL_KEYS["smoothness"], 0.0)

        e_xy = (pod_xy / 0.08 + mug_xy / 0.10 + machine_xy / 0.30) / 3.0
        e_transport = (mug_xy / 0.12 + pod_xy / 0.10 + order_pen) / 3.0
        e_lift = (pod_z / 0.05 + mug_z / 0.06 + drop / 0.04) / 3.0

        pod_align = 0.5 * (pod_xy / 0.06 + pod_z / 0.04)
        mug_align = 0.5 * (mug_xy / 0.08 + mug_z / 0.05)
        e_contact = 0.70 * (pod_align + mug_align) / 2.0 + 0.30 * (contact_res / 1.0 + penetration / 0.010)
        e_stage = (1.0 - stage_progress) + 0.35 * order_pen + 0.20 * (1.0 - mug_align)

        e_dynamics = drop / 0.04 + float(metrics.get("energy", 30.0)) / 60.0 + 0.25 * (vel_res / 0.25)
        e_slip = drop / 0.03 + mug_xy / 0.12 + 0.35 * (slip_res / 0.05)
        e_coupling = max(pod_align, mug_align) + 0.5 * min(pod_align, mug_align) + 0.20 * (smooth_res / 0.20)

        comps = np.array(
            [e_xy, e_transport, e_lift, e_contact, e_stage, e_dynamics, e_slip, e_coupling],
            dtype=np.float32,
        )
        return np.array([self._clip01(float(v)) for v in comps], dtype=np.float32)

    def physical_component_loss(self, pred_components: Any) -> Any:
        import torch
        import torch.nn.functional as F

        e_xy, e_transport, e_lift, e_contact, e_stage, e_dynamics, e_slip, e_coupling = [
            pred_components[:, i : i + 1] for i in range(len(self.COMPONENT_KEYS))
        ]
        contact_target = 0.5 * (e_xy + e_lift)
        stage_target = torch.maximum(e_transport, 1.0 - e_contact)
        dynamics_target = 0.5 * (e_slip + e_stage)
        slip_floor = 0.5 * e_lift
        coupling_floor = torch.maximum(e_contact, 0.5 * (e_xy + e_lift))
        return (
            F.smooth_l1_loss(e_contact, contact_target)
            + F.smooth_l1_loss(e_stage, stage_target)
            + F.smooth_l1_loss(e_dynamics, dynamics_target)
            + torch.relu(slip_floor - e_slip).pow(2).mean()
            + torch.relu(coupling_floor - e_coupling).pow(2).mean()
        ) / 5.0

    def project_theta_cont_manifold(self, theta_cont: Any) -> Any:
        import torch

        theta_cont = theta_cont.clone()
        lo_raw = theta_cont[:, self.OFFSET_LO_CONT_IDX : self.OFFSET_LO_CONT_IDX + 1]
        hi_raw = theta_cont[:, self.OFFSET_HI_CONT_IDX : self.OFFSET_HI_CONT_IDX + 1]
        lo = torch.minimum(lo_raw, hi_raw)
        hi = torch.maximum(lo_raw, hi_raw)
        theta_cont[:, self.OFFSET_LO_CONT_IDX : self.OFFSET_LO_CONT_IDX + 1] = lo
        theta_cont[:, self.OFFSET_HI_CONT_IDX : self.OFFSET_HI_CONT_IDX + 1] = hi
        theta_cont[:, self.OFFSET_WIDTH_CONT_IDX : self.OFFSET_WIDTH_CONT_IDX + 1] = hi - lo
        theta_cont[:, self.OFFSET_CENTER_CONT_IDX : self.OFFSET_CENTER_CONT_IDX + 1] = 0.5 * (lo + hi)
        return torch.clamp(theta_cont, 0.0, 1.0)

    @staticmethod
    def _fallback_make_candidate(index: int, rng: np.random.Generator) -> dict[str, Any]:
        offset_options = [[10, 20], [10, 15], [15, 20], [5, 20], [10, 25], [0, 20], [15, 25]]
        return {
            "candidate_index": int(index),
            "selection_strategy": str(rng.choice(["nearest_neighbor_object", "random"], p=[0.8, 0.2])),
            "select_src_per_subtask": bool(rng.random() < 0.8),
            "transform_first_robot_pose": bool(rng.random() < 0.1),
            "interpolate_from_last_target_pose": bool(rng.random() < 0.9),
            "action_noise": float(rng.choice([0.0, 0.01, 0.02, 0.05, 0.08])),
            "num_interpolation_steps": int(rng.choice([3, 5, 8, 10, 15])),
            "num_fixed_steps": int(rng.choice([0, 1, 2])),
            "offset_range": offset_options[int(rng.integers(0, len(offset_options)))],
            "nn_k": int(rng.choice([1, 3, 5, 10])),
        }

    def _base_make_candidate(self, candidate_index: int, rng: np.random.Generator) -> dict[str, Any]:
        if _external_make_candidate is not None:
            return dict(_external_make_candidate(candidate_index, rng))
        return self._fallback_make_candidate(candidate_index, rng)

    def sample_repair_theta(
        self,
        candidate_index: int,
        rng: np.random.Generator,
        candidate_mode: str = "default",
    ) -> dict[str, Any]:
        if candidate_mode != "safe":
            return self._base_make_candidate(candidate_index, rng)

        theta = self._base_make_candidate(candidate_index, rng)
        theta["num_fixed_steps"] = 0
        if rng.random() < 0.90:
            theta["interpolate_from_last_target_pose"] = True
        if rng.random() < 0.90:
            theta["transform_first_robot_pose"] = False
        if rng.random() < 0.85:
            theta["selection_strategy"] = "nearest_neighbor_object"
            theta["nn_k"] = int(rng.choice([1, 3, 5, 10], p=[0.35, 0.25, 0.25, 0.15]))
        else:
            theta["selection_strategy"] = "random"
            theta["nn_k"] = int(rng.choice([1, 3, 5, 10]))
        theta["select_src_per_subtask"] = bool(rng.random() < 0.82)
        theta["action_noise"] = float(rng.choice([0.0, 0.01, 0.02, 0.05, 0.08], p=[0.12, 0.18, 0.22, 0.30, 0.18]))
        theta["num_interpolation_steps"] = int(rng.choice([3, 5, 8, 10, 15], p=[0.12, 0.28, 0.16, 0.20, 0.24]))
        offset_options = np.array(
            [[10, 20], [10, 15], [15, 20], [5, 20], [10, 25], [0, 20], [15, 25], [0, 15], [5, 15], [10, 10], [15, 15]],
            dtype=int,
        )
        offset_probs = np.array([0.20, 0.15, 0.13, 0.10, 0.09, 0.07, 0.07, 0.06, 0.06, 0.04, 0.03])
        theta["offset_range"] = offset_options[int(rng.choice(len(offset_options), p=offset_probs))].tolist()
        theta["candidate_family"] = (
            f"{theta['selection_strategy']}_per{int(theta['select_src_per_subtask'])}"
            f"_noise{theta['action_noise']}_interp{theta['num_interpolation_steps']}_safe"
        )
        return theta

    def load_failed_contexts(self, failed_hdf5: str | Path, max_failed_demos: int | None = None) -> list[dict[str, Any]]:
        if _external_load_failed_contexts is None:
            raise RuntimeError(
                "load_failed_contexts requires phygen.adapters.mimicgen.coffee_repair"
            )
        return list(_external_load_failed_contexts(str(failed_hdf5), max_failed_demos))
