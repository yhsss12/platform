from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from stack_three_failed_conditioned_mimicgen_repair import (  # type: ignore
        demo_sort_key as _external_demo_sort_key,
        load_failed_contexts as _external_load_failed_contexts,
        make_candidate as _external_make_candidate,
    )
except Exception:  # pragma: no cover - fallback for smoke tests outside the project repo.
    _external_demo_sort_key = None
    _external_load_failed_contexts = None
    _external_make_candidate = None


def demo_sort_key(key: str) -> int:
    """Sort demo ids like demo_12 naturally, with a robust fallback."""
    if _external_demo_sort_key is not None:
        try:
            return int(_external_demo_sort_key(key))
        except Exception:
            pass
    m = re.search(r"(\d+)$", str(key))
    return int(m.group(1)) if m else 0


@dataclass(frozen=True)
class TaskSpec:
    task_name: str
    context_keys: list[str]
    context_scales: dict[str, float]
    component_keys: list[str]
    component_weights: list[float]
    residual_groups: dict[str, list[str]]
    optional_residual_keys: dict[str, list[str]]
    theta_disc_keys: list[str]
    theta_cont_keys: list[str]
    theta_cont_independent_keys: list[str]
    theta_cont_independent_idxs: list[int]
    theta_cont_derived_keys: list[str]
    output_model_name: str = "stack_three_failed_conditioned_pinn.pt"
    output_plan_name: str = "pinn_utility_boundary_union_candidate_plan.jsonl"
    output_summary_name: str = "pinn_utility_boundary_union_summary.json"


class BasePhyGenAdapter:
    """Task/generator adapter interface for PhyGen.

    New MimicGen tasks should subclass this adapter and implement only the
    task-specific pieces: context vector, residual construction, theta features,
    candidate sampling, and optional theta-manifold projection.
    """

    def task_spec(self) -> TaskSpec:
        raise NotImplementedError

    def theta_to_features(self, theta: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, float]:
        """Return (discrete_features, continuous_features, boundary_bonus)."""
        raise NotImplementedError

    def feature_vector(self, context: dict[str, float], theta: dict[str, Any]) -> np.ndarray:
        spec = self.task_spec()
        ctx = np.array(
            [float(context.get(k, 0.0)) / float(spec.context_scales[k]) for k in spec.context_keys],
            dtype=np.float32,
        )
        theta_disc, theta_cont, boundary_bonus = self.theta_to_features(theta)
        return np.concatenate([ctx, theta_disc, theta_cont, np.array([boundary_bonus], dtype=np.float32)])

    def build_residual_targets(self, metrics: dict[str, float] | None, problematic: bool = False) -> np.ndarray:
        raise NotImplementedError

    def component_energy_target(self, components: np.ndarray) -> np.ndarray:
        spec = self.task_spec()
        weights = np.asarray(spec.component_weights, dtype=np.float32)
        return ((components * weights[None, :]).sum(axis=1) / weights.sum()).astype(np.float32)

    def physical_component_loss(self, pred_components: Any) -> Any:
        """Optional differentiable task-specific residual relations.

        The base adapter returns zero.  StackThree overrides this with rigid-body
        compatibility relations.  Future task adapters can add skill, dynamic,
        deformable, or dexterous residual relations without touching the core.
        """
        import torch

        return torch.zeros((), dtype=pred_components.dtype, device=pred_components.device)

    def project_theta_cont_manifold(self, theta_cont: Any) -> Any:
        """Project continuous theta features back to their feasible manifold.

        Default: simple clamp.  Adapters can override to recompute derived
        features such as offset_width and offset_center from independent axes.
        """
        import torch

        return torch.clamp(theta_cont, 0.0, 1.0)

    def sample_repair_theta(
        self,
        candidate_index: int,
        rng: np.random.Generator,
        candidate_mode: str = "default",
    ) -> dict[str, Any]:
        raise NotImplementedError

    def load_failed_contexts(self, failed_hdf5: str | Path, max_failed_demos: int | None = None) -> list[dict[str, Any]]:
        raise NotImplementedError

    def metadata(self) -> dict[str, Any]:
        return asdict(self.task_spec())


def load_records(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("metrics") is None:
                if row.get("problematic"):
                    row["metrics"] = {"energy": 30.0}
                    row["success"] = False
                else:
                    continue
            rows.append(row)
    return rows


def load_success_demo_keys(paths: list[str] | None) -> set[str]:
    keys: set[str] = set()
    if not paths:
        return keys
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("success"):
                    keys.add(row["demo_key"])
    return keys
