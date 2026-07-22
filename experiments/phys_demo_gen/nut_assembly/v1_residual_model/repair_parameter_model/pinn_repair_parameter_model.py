"""V1-E：PINN Repair Parameter Residual Field Model。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

import sys
from pathlib import Path

_V1E_DIR = Path(__file__).resolve().parent
_V1_DIR = _V1E_DIR.parent
if str(_V1_DIR) not in sys.path:
    sys.path.insert(0, str(_V1_DIR))

from residual_dataset import FAILURE_TYPES, OUTCOME_TYPES
from repair_dataset import (
    ALL_THETA_KEYS,
    COARSE_FAILURE_MODES,
    CONTEXT_NUMERIC_KEYS,
    DEMO_KEYS,
    GRASP_LIFT_PARAM_KEYS,
    INSERTION_PARAM_KEYS,
    TRANSPORT_PARAM_KEYS,
)

ENERGY_WEIGHTS = torch.tensor([3.0, 3.0, 2.0, 2.0, 0.2], dtype=torch.float32)
ENERGY_MARGIN = 1.0

INPUT_DIM = len(DEMO_KEYS) + (len(COARSE_FAILURE_MODES) - 1) + len(CONTEXT_NUMERIC_KEYS) + 2 * len(ALL_THETA_KEYS)

# Offsets in flat input vector
_DEMO_OFF = 0
_FAIL_OFF = len(DEMO_KEYS)
_CTX_OFF = _FAIL_OFF + (len(COARSE_FAILURE_MODES) - 1)
_THETA_OFF = _CTX_OFF + len(CONTEXT_NUMERIC_KEYS)
_MASK_OFF = _THETA_OFF + len(ALL_THETA_KEYS)

_THETA_SLICE = {
    "insertion": slice(_THETA_OFF, _THETA_OFF + len(INSERTION_PARAM_KEYS)),
    "transport": slice(
        _THETA_OFF + len(INSERTION_PARAM_KEYS),
        _THETA_OFF + len(INSERTION_PARAM_KEYS) + len(TRANSPORT_PARAM_KEYS),
    ),
    "grasp_lift": slice(
        _THETA_OFF + len(INSERTION_PARAM_KEYS) + len(TRANSPORT_PARAM_KEYS),
        _THETA_OFF + len(ALL_THETA_KEYS),
    ),
}


@dataclass
class RepairPhysicsLossConfig:
    use_component_supervision: bool = True
    use_total_consistency: bool = True
    use_success_margin: bool = True
    use_monotonic_repair: bool = True
    w_component_supervision: float = 1.0
    w_total_consistency: float = 1.0
    w_success_margin: float = 0.5
    w_monotonic_repair: float = 0.5
    margin: float = ENERGY_MARGIN

    @classmethod
    def full(cls) -> RepairPhysicsLossConfig:
        return cls()


class PINNRepairParameterModel(nn.Module):
    """Failed demo context + repair theta + mask -> predicted post-repair residual field."""

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dim: int = 160,
        num_layers: int = 4,
        dropout: float = 0.1,
        num_failure_types: int = len(FAILURE_TYPES),
        num_outcome_types: int = len(OUTCOME_TYPES),
    ):
        super().__init__()
        self.input_dim = input_dim
        layers: list[nn.Module] = []
        in_dim = input_dim
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
        total = torch.nn.functional.softplus(self.head_total(h)).squeeze(-1)
        weights = ENERGY_WEIGHTS.to(components.device)
        total_consistent = torch.sum(components * weights, dim=-1)
        return {
            "E_components": components,
            "E_total": total,
            "E_total_consistent": total_consistent,
            "success_logit": self.head_success(h).squeeze(-1),
            "failure_type_logits": self.head_failure(h),
            "outcome_logits": self.head_outcome(h),
            "grasp_success_logit": self.head_grasp_success(h).squeeze(-1),
            "lift_success_logit": self.head_lift_success(h).squeeze(-1),
        }


def _ctx(features: torch.Tensor, name: str) -> torch.Tensor:
    idx = CONTEXT_NUMERIC_KEYS.index(name)
    return features[:, _CTX_OFF + idx]


def compute_repair_direction_score(features: torch.Tensor, source_failure_mode_idx: torch.Tensor) -> torch.Tensor:
    """Heuristic [0,1] score: theta moves residuals in the repair direction for this failure mode."""
    theta = features[:, _THETA_OFF:_MASK_OFF]
    mask = features[:, _MASK_OFF:]
    masked_theta = theta * mask
    score = torch.zeros(features.shape[0], device=features.device)

    ins = source_failure_mode_idx == 1
    if torch.any(ins):
        insert_z = masked_theta[ins, INSERTION_PARAM_KEYS.index("insert_z_offset")]
        z_gain = masked_theta[ins, INSERTION_PARAM_KEYS.index("z_gain")]
        score[ins] = 0.5 * (
            torch.clamp(-insert_z / 0.12, 0.0, 1.0) + torch.clamp((z_gain - 0.55) / 0.45, 0.0, 1.0)
        )

    tr = source_failure_mode_idx == 2
    if torch.any(tr):
        gain = masked_theta[tr, len(INSERTION_PARAM_KEYS) + TRANSPORT_PARAM_KEYS.index("transport_xy_gain")]
        scale = masked_theta[
            tr, len(INSERTION_PARAM_KEYS) + TRANSPORT_PARAM_KEYS.index("transport_xy_offset_scale")
        ]
        score[tr] = 0.5 * (
            torch.clamp((gain - 0.4) / 0.6, 0.0, 1.0) + torch.clamp((scale - 0.5) / 0.75, 0.0, 1.0)
        )

    gr = source_failure_mode_idx == 3
    if torch.any(gr):
        base = len(INSERTION_PARAM_KEYS) + len(TRANSPORT_PARAM_KEYS)
        off = torch.abs(masked_theta[gr, base + GRASP_LIFT_PARAM_KEYS.index("grasp_xy_offset_x")]) + torch.abs(
            masked_theta[gr, base + GRASP_LIFT_PARAM_KEYS.index("grasp_xy_offset_y")]
        )
        lift_h = masked_theta[gr, base + GRASP_LIFT_PARAM_KEYS.index("micro_lift_height")]
        dist0 = _ctx(features, "original_eef_nut_distance")[gr]
        score[gr] = (
            torch.clamp(1.0 - off / 0.08, 0.0, 1.0)
            + torch.clamp(lift_h / 0.08, 0.0, 1.0) * torch.clamp(1.0 - dist0 / 0.08, 0.0, 1.0)
        ) / 2.0

    lf = source_failure_mode_idx == 4
    if torch.any(lf):
        base = len(INSERTION_PARAM_KEYS) + len(TRANSPORT_PARAM_KEYS)
        lift_h = masked_theta[lf, base + GRASP_LIFT_PARAM_KEYS.index("micro_lift_height")]
        lift_steps = masked_theta[lf, base + GRASP_LIFT_PARAM_KEYS.index("lift_steps")]
        score[lf] = 0.5 * (
            torch.clamp(lift_h / 0.10, 0.0, 1.0) + torch.clamp(lift_steps / 30.0, 0.0, 1.0)
        )

    return score


def compute_monotonic_repair_loss(
    pred_total: torch.Tensor,
    original_E_total: torch.Tensor,
    repair_direction_score: torch.Tensor,
) -> torch.Tensor:
    """If theta moves residuals in the right direction, predicted energy should not exceed baseline."""
    target_cap = original_E_total * (1.0 - 0.35 * repair_direction_score)
    return torch.relu(pred_total - target_cap).mean()


def compute_success_margin_loss(
    pred_total: torch.Tensor, success_flag: torch.Tensor, margin: float
) -> torch.Tensor:
    succ = success_flag > 0.5
    fail = ~succ
    if not torch.any(succ) or not torch.any(fail):
        return pred_total.new_tensor(0.0)
    return torch.relu(margin + pred_total[succ].mean() - pred_total[fail].mean())


def compute_repair_pinn_losses(
    out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    physics: RepairPhysicsLossConfig | None = None,
) -> dict[str, torch.Tensor]:
    physics = physics or RepairPhysicsLossConfig.full()
    pred_total = out["E_total"]
    pred_components = out["E_components"]
    target_total = batch["target_E_total"]
    target_components = batch["targets_components"]

    l_energy = nn.functional.mse_loss(pred_total, target_total)
    l_component_supervision = nn.functional.mse_loss(pred_components, target_components)
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
    l_total_consistency = nn.functional.mse_loss(pred_total, total_from_components)
    l_success_margin = compute_success_margin_loss(pred_total, batch["success_flag"], physics.margin)

    repair_score = compute_repair_direction_score(batch["features"], batch["source_failure_mode_idx"])
    l_monotonic = compute_monotonic_repair_loss(
        pred_total, batch["original_E_total"], repair_score
    )

    total = (
        l_energy
        + 0.5 * l_success
        + 0.2 * l_failure
        + 0.2 * l_outcome
        + 0.3 * l_grasp
        + 0.3 * l_lift
    )
    if physics.use_component_supervision:
        total = total + physics.w_component_supervision * l_component_supervision
    if physics.use_total_consistency:
        total = total + physics.w_total_consistency * l_total_consistency
    if physics.use_success_margin:
        total = total + physics.w_success_margin * l_success_margin
    if physics.use_monotonic_repair:
        total = total + physics.w_monotonic_repair * l_monotonic

    return {
        "loss": total,
        "L_energy": l_energy,
        "L_component_supervision": l_component_supervision,
        "L_success": l_success,
        "L_failure": l_failure,
        "L_outcome": l_outcome,
        "L_grasp": l_grasp,
        "L_lift": l_lift,
        "L_total_consistency": l_total_consistency,
        "L_success_margin": l_success_margin,
        "L_monotonic_repair": l_monotonic,
    }


def explicit_repair_energy(features: torch.Tensor) -> torch.Tensor:
    """Physics baseline for candidate pruning (not the primary method)."""
    final_xy = _ctx(features, "original_final_xy")
    min_xy = _ctx(features, "original_min_xy")
    final_z = _ctx(features, "original_final_z_diff")
    min_yaw = _ctx(features, "original_min_yaw_error")
    dist = _ctx(features, "original_eef_nut_distance")

    e_xy = final_xy / 0.03
    e_transport = min_xy / 0.03
    e_yaw = min_yaw / 0.05
    e_z = torch.clamp(final_z - 0.02, min=0.0) / 0.02
    e_smooth = torch.full_like(e_xy, 0.8)

    theta = features[:, _THETA_OFF:_MASK_OFF]
    mask = features[:, _MASK_OFF:]
    mode_idx = batch_mode_from_features(features)

    insert_z = theta[:, INSERTION_PARAM_KEYS.index("insert_z_offset")] * mask[:, INSERTION_PARAM_KEYS.index("insert_z_offset")]
    z_gain = theta[:, INSERTION_PARAM_KEYS.index("z_gain")] * mask[:, INSERTION_PARAM_KEYS.index("z_gain")]
    tr_gain_idx = len(INSERTION_PARAM_KEYS) + TRANSPORT_PARAM_KEYS.index("transport_xy_gain")
    tr_gain = theta[:, tr_gain_idx] * mask[:, tr_gain_idx]
    micro_lift_idx = (
        len(INSERTION_PARAM_KEYS) + len(TRANSPORT_PARAM_KEYS) + GRASP_LIFT_PARAM_KEYS.index("micro_lift_height")
    )
    micro_lift = theta[:, micro_lift_idx] * mask[:, micro_lift_idx]

    e_z = e_z * torch.where(mode_idx == 1, torch.clamp(1.0 + insert_z / 0.04, 0.2, 1.5), torch.ones_like(e_z))
    e_z = e_z * torch.where(mode_idx == 1, torch.clamp(1.3 - z_gain, 0.2, 1.5), torch.ones_like(e_z))
    e_transport = e_transport * torch.where(mode_idx == 2, torch.clamp(1.4 - tr_gain, 0.15, 1.5), torch.ones_like(e_transport))
    e_transport = e_transport * torch.where(
        mode_idx >= 3, torch.clamp(dist / 0.08, 0.2, 1.5) * torch.clamp(1.1 - micro_lift / 0.08, 0.3, 1.0), torch.ones_like(e_transport)
    )

    weights = ENERGY_WEIGHTS.to(features.device)
    components = torch.stack([e_xy, e_transport, e_yaw, e_z, e_smooth], dim=-1)
    return torch.sum(components * weights, dim=-1)


def batch_mode_from_features(features: torch.Tensor) -> torch.Tensor:
    fail_onehot = features[:, _FAIL_OFF : _FAIL_OFF + (len(COARSE_FAILURE_MODES) - 1)]
    if fail_onehot.shape[1] == 0:
        return torch.zeros(features.shape[0], dtype=torch.long, device=features.device)
    return fail_onehot.argmax(dim=-1) + 1


def predict_dict(model: PINNRepairParameterModel, features: torch.Tensor) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        out = model(features)
    return {k: v.detach().cpu().numpy() for k, v in out.items()}
