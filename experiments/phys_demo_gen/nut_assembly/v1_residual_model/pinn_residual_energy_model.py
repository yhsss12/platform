"""V1-D：PINN-style Physics-Informed Neural Residual Energy Model（非 PDE PINN，非 PINA）。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from residual_dataset import FAILURE_TYPES, FEATURE_NAMES_V1C, OUTCOME_TYPES

ENERGY_WEIGHTS = torch.tensor([3.0, 3.0, 2.0, 2.0, 0.2], dtype=torch.float32)

# V1-C 45-dim feature indices used in explicit physics residuals
FEATURE_IDX = {
    "final_nut_peg_xy": 0,
    "min_nut_peg_xy": 1,
    "final_z_diff": 2,
    "min_yaw_error": 3,
    "action_accel_max": 6,
    "nut_displacement_after_grasp": 27,
    "eef_nut_distance_at_grasp": 28,
    "nut_lift_delta": 42,
    "grasp_success_proxy_feat": 43,
    "lift_success_proxy_feat": 44,
}

XY_THRESHOLD = 0.03
TRANSPORT_THRESHOLD = 0.03
YAW_THRESHOLD = 0.05
Z_TOLERANCE = 0.02
SMOOTH_THRESHOLD = 2.5
ENERGY_MARGIN = 1.0


@dataclass
class PhysicsLossConfig:
    use_phys_components: bool = True
    use_total_consistency: bool = True
    use_margin: bool = True
    w_phys_components: float = 1.0
    w_total_consistency: float = 1.0
    w_margin: float = 0.5
    margin: float = ENERGY_MARGIN

    @classmethod
    def full(cls) -> PhysicsLossConfig:
        return cls()

    @classmethod
    def no_phys_components(cls) -> PhysicsLossConfig:
        return cls(use_phys_components=False)

    @classmethod
    def no_total_consistency(cls) -> PhysicsLossConfig:
        return cls(use_total_consistency=False)

    @classmethod
    def no_margin(cls) -> PhysicsLossConfig:
        return cls(use_margin=False)

    @classmethod
    def supervised_only(cls) -> PhysicsLossConfig:
        return cls(use_phys_components=False, use_total_consistency=False, use_margin=False)


class PINNResidualEnergyModel(nn.Module):
    """MLP + physics-informed losses on Nut Assembly geometric/grasp residuals."""

    def __init__(
        self,
        input_dim: int | None = None,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        num_failure_types: int = len(FAILURE_TYPES),
        num_outcome_types: int = len(OUTCOME_TYPES),
    ):
        super().__init__()
        self.input_dim = input_dim or len(FEATURE_NAMES_V1C)
        self.hidden_dim = hidden_dim

        layers: list[nn.Module] = []
        in_dim = self.input_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)

        self.head_components = nn.Linear(hidden_dim, 5)
        self.head_total = nn.Linear(hidden_dim, 1)
        self.head_success = nn.Linear(hidden_dim, 1)
        self.head_failure = nn.Linear(hidden_dim, num_failure_types)
        self.head_outcome = nn.Linear(hidden_dim, num_outcome_types)
        self.head_grasp_success = nn.Linear(hidden_dim, 1)
        self.head_lift_success = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.backbone(features)
        components = torch.nn.functional.softplus(self.head_components(h))
        total_direct = torch.nn.functional.softplus(self.head_total(h)).squeeze(-1)
        weights = ENERGY_WEIGHTS.to(components.device)
        total_consistent = torch.sum(components * weights, dim=-1)
        return {
            "E_components": components,
            "E_total_direct": total_direct,
            "E_total_consistent": total_consistent,
            "E_total": total_direct,
            "success_logit": self.head_success(h).squeeze(-1),
            "failure_type_logits": self.head_failure(h),
            "outcome_logits": self.head_outcome(h),
            "grasp_success_logit": self.head_grasp_success(h).squeeze(-1),
            "lift_success_logit": self.head_lift_success(h).squeeze(-1),
        }


def compute_physics_components(features: torch.Tensor) -> torch.Tensor:
    """Explicit Nut Assembly normalized energy components from input features."""
    final_xy = features[:, FEATURE_IDX["final_nut_peg_xy"]]
    min_xy = features[:, FEATURE_IDX["min_nut_peg_xy"]]
    final_z = features[:, FEATURE_IDX["final_z_diff"]]
    min_yaw = features[:, FEATURE_IDX["min_yaw_error"]]
    acc_max = features[:, FEATURE_IDX["action_accel_max"]]

    e_xy = final_xy / XY_THRESHOLD
    e_transport = min_xy / TRANSPORT_THRESHOLD
    e_yaw = min_yaw / YAW_THRESHOLD
    e_z = torch.clamp(final_z - Z_TOLERANCE, min=0.0) / Z_TOLERANCE
    e_smooth = acc_max / SMOOTH_THRESHOLD
    return torch.stack([e_xy, e_transport, e_yaw, e_z, e_smooth], dim=-1)


def compute_margin_loss(pred_total: torch.Tensor, success_flag: torch.Tensor, margin: float) -> torch.Tensor:
    succ_mask = success_flag > 0.5
    fail_mask = ~succ_mask
    if not torch.any(succ_mask) or not torch.any(fail_mask):
        return pred_total.new_tensor(0.0)
    succ_mean = pred_total[succ_mask].mean()
    fail_mean = pred_total[fail_mask].mean()
    return torch.relu(margin + succ_mean - fail_mean)


def compute_pinn_losses(
    out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    features: torch.Tensor,
    *,
    physics: PhysicsLossConfig | None = None,
) -> dict[str, torch.Tensor]:
    physics = physics or PhysicsLossConfig.full()
    pred_total = out["E_total"]
    pred_components = out["E_components"]
    target_total = batch["target_E_total"]
    target_components = batch["targets_components"]

    l_energy = nn.functional.mse_loss(pred_total, target_total)
    l_components = nn.functional.mse_loss(pred_components, target_components)
    l_success = nn.functional.binary_cross_entropy_with_logits(out["success_logit"], batch["success_flag"])
    l_failure = nn.functional.cross_entropy(out["failure_type_logits"], batch["failure_type_idx"])
    l_outcome = nn.functional.cross_entropy(out["outcome_logits"], batch["outcome_idx"])
    l_grasp = nn.functional.binary_cross_entropy_with_logits(
        out["grasp_success_logit"], batch["grasp_success_flag"]
    )
    l_lift = nn.functional.binary_cross_entropy_with_logits(
        out["lift_success_logit"], batch["lift_success_flag"]
    )

    weights = ENERGY_WEIGHTS.to(pred_components.device)
    total_from_components = torch.sum(pred_components * weights, dim=-1)
    l_supervised_consistency = nn.functional.mse_loss(pred_total, total_from_components)

    phys_components = compute_physics_components(features)
    l_phys_components = nn.functional.mse_loss(pred_components, phys_components)
    l_total_consistency = nn.functional.mse_loss(pred_total, total_from_components)
    l_margin = compute_margin_loss(pred_total, batch["success_flag"], physics.margin)

    total = (
        l_energy
        + 0.5 * l_components
        + 0.5 * l_success
        + 0.2 * l_failure
        + 0.2 * l_outcome
        + 0.3 * l_grasp
        + 0.3 * l_lift
        + 0.5 * l_supervised_consistency
    )
    if physics.use_phys_components:
        total = total + physics.w_phys_components * l_phys_components
    if physics.use_total_consistency:
        total = total + physics.w_total_consistency * l_total_consistency
    if physics.use_margin:
        total = total + physics.w_margin * l_margin

    return {
        "loss": total,
        "L_energy": l_energy,
        "L_components": l_components,
        "L_success": l_success,
        "L_failure": l_failure,
        "L_outcome": l_outcome,
        "L_grasp": l_grasp,
        "L_lift": l_lift,
        "L_supervised_consistency": l_supervised_consistency,
        "L_phys_components": l_phys_components,
        "L_total_consistency": l_total_consistency,
        "L_margin": l_margin,
    }


def predict_dict(model: PINNResidualEnergyModel, features: torch.Tensor) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        out = model(features)
    return {k: v.cpu().numpy() for k, v in out.items()}
