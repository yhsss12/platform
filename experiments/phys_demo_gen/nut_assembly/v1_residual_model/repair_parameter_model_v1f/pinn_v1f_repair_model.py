"""V1-F：Uncertainty-aware PINN Repair Parameter Residual Field Model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path

_V1F_DIR = Path(__file__).resolve().parent
_V1_DIR = _V1F_DIR.parent
_V1E_DIR = _V1_DIR / "repair_parameter_model"
for path in (_V1_DIR, _V1E_DIR, _V1F_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from residual_dataset import FAILURE_TYPES, OUTCOME_TYPES
from v1f_repair_dataset import (
    ALL_THETA_KEYS_V1F,
    CONTEXT_NUMERIC_KEYS,
    COARSE_FAILURE_MODES,
    DEMO_KEYS,
    GRASP_LIFT_PARAM_KEYS,
    INSERTION_PARAM_KEYS,
    LIFT_EXTRA_PARAM_KEYS,
    TRANSPORT_PARAM_KEYS,
    V1F_COMPONENT_NAMES,
)

# E_xy, E_transport, E_yaw, E_z, E_grasp, E_lift, E_smooth
ENERGY_WEIGHTS_V1F = torch.tensor([3.0, 3.0, 2.0, 2.0, 2.0, 2.0, 0.2], dtype=torch.float32)
ENERGY_MARGIN = 1.0

INPUT_DIM_V1F = (
    len(DEMO_KEYS) + (len(COARSE_FAILURE_MODES) - 1) + len(CONTEXT_NUMERIC_KEYS) + 2 * len(ALL_THETA_KEYS_V1F)
)

_DEMO_OFF = 0
_FAIL_OFF = len(DEMO_KEYS)
_CTX_OFF = _FAIL_OFF + (len(COARSE_FAILURE_MODES) - 1)
_THETA_OFF = _CTX_OFF + len(CONTEXT_NUMERIC_KEYS)
_MASK_OFF = _THETA_OFF + len(ALL_THETA_KEYS_V1F)


@dataclass
class V1FPhysicsLossConfig:
    use_component_supervision: bool = True
    use_lift_residual_supervision: bool = True
    use_total_consistency: bool = True
    use_success_margin: bool = True
    use_uncertainty_nll: bool = True
    use_monotonic_repair: bool = True
    w_component: float = 1.0
    w_lift_residual: float = 0.5
    w_total_consistency: float = 1.0
    w_success_margin: float = 0.5
    w_uncertainty: float = 0.3
    w_monotonic: float = 0.3
    margin: float = ENERGY_MARGIN

    @classmethod
    def full(cls) -> V1FPhysicsLossConfig:
        return cls()

    @classmethod
    def aligned_finetune(cls) -> V1FPhysicsLossConfig:
        """更稳的 aligned 微调：降低 uncertainty / monotonic 权重。"""
        return cls(w_uncertainty=0.05, w_monotonic=0.1, w_success_margin=0.3)

    @classmethod
    def balanced(cls) -> V1FPhysicsLossConfig:
        """plus-balanced：强化 success / ranking，保留 consistency + uncertainty。"""
        return cls(
            w_component=1.0,
            w_lift_residual=0.4,
            w_total_consistency=1.0,
            w_success_margin=0.4,
            w_uncertainty=0.05,
            w_monotonic=0.05,
        )

    @classmethod
    def hundredbase(cls) -> V1FPhysicsLossConfig:
        """V1-F-100Base：balanced 基础上略增 uncertainty / consistency。"""
        return cls(
            w_component=1.0,
            w_lift_residual=0.4,
            w_total_consistency=1.0,
            w_success_margin=0.45,
            w_uncertainty=0.08,
            w_monotonic=0.05,
        )


@dataclass
class V1GStage1PhysicsLossConfig(V1FPhysicsLossConfig):
    """V1-G-stage1-p1xy：E_transport / E_xy 辅助 physics loss + E_lift soft loss。"""

    lambda_transport: float = 0.05
    lambda_xy: float = 0.05
    lambda_lift: float = 0.02
    use_physics_aux_loss: bool = True

    @classmethod
    def stage1_p1xy(cls) -> V1GStage1PhysicsLossConfig:
        return cls(
            w_component=1.0,
            w_lift_residual=0.0,
            w_total_consistency=1.0,
            w_success_margin=0.45,
            w_uncertainty=0.08,
            w_monotonic=0.05,
            lambda_transport=0.05,
            lambda_xy=0.05,
            lambda_lift=0.02,
            use_physics_aux_loss=True,
        )


@dataclass
class V1GStage1LiteP1P2Config(V1FPhysicsLossConfig):
    """V1-G-stage1-lite-p1p2：更低 physics 权重 + 略提高 retention。"""

    lambda_transport: float = 0.02
    lambda_xy: float = 0.02
    lambda_lift: float = 0.01
    lambda_retention: float = 0.35
    use_physics_aux_loss: bool = True

    @classmethod
    def lite_p1p2(cls) -> V1GStage1LiteP1P2Config:
        return cls(
            w_component=1.0,
            w_lift_residual=0.0,
            w_total_consistency=1.0,
            w_success_margin=0.45,
            w_uncertainty=0.08,
            w_monotonic=0.05,
            lambda_transport=0.02,
            lambda_xy=0.02,
            lambda_lift=0.01,
            lambda_retention=0.35,
            use_physics_aux_loss=True,
        )


# V1F component index: E_xy=0, E_transport=1, E_lift=5
_V1F_IDX_XY = 0
_V1F_IDX_TRANSPORT = 1
_V1F_IDX_LIFT = 5


class PINNV1FRepairModel(nn.Module):
    """Failed context + repair θ + failure embedding + mask -> energy field + uncertainty."""

    def __init__(
        self,
        input_dim: int = INPUT_DIM_V1F,
        hidden_dim: int = 192,
        num_layers: int = 5,
        dropout: float = 0.15,
        num_failure_types: int = len(FAILURE_TYPES),
        num_outcome_types: int = len(OUTCOME_TYPES),
        num_components: int = len(V1F_COMPONENT_NAMES),
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_components = num_components
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)

        self.head_components = nn.Linear(hidden_dim, num_components)
        self.head_total = nn.Linear(hidden_dim, 1)
        self.head_log_var = nn.Linear(hidden_dim, 1)
        self.head_success = nn.Linear(hidden_dim, 1)
        self.head_failure = nn.Linear(hidden_dim, num_failure_types)
        self.head_outcome = nn.Linear(hidden_dim, num_outcome_types)
        self.head_grasp_success = nn.Linear(hidden_dim, 1)
        self.head_lift_success = nn.Linear(hidden_dim, 1)
        self.head_lift_residuals = nn.Linear(hidden_dim, 5)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.backbone(features)
        components = F.softplus(self.head_components(h))
        total = F.softplus(self.head_total(h)).squeeze(-1)
        log_var = torch.clamp(self.head_log_var(h).squeeze(-1), min=-8.0, max=8.0)
        weights = ENERGY_WEIGHTS_V1F.to(components.device)
        total_consistent = torch.sum(components * weights, dim=-1)
        uncertainty = torch.exp(0.5 * log_var)
        return {
            "E_components": components,
            "E_total": total,
            "E_total_consistent": total_consistent,
            "log_var": log_var,
            "uncertainty": uncertainty,
            "success_logit": self.head_success(h).squeeze(-1),
            "failure_type_logits": self.head_failure(h),
            "outcome_logits": self.head_outcome(h),
            "grasp_success_logit": self.head_grasp_success(h).squeeze(-1),
            "lift_success_logit": self.head_lift_success(h).squeeze(-1),
            "lift_residuals": F.softplus(self.head_lift_residuals(h)),
        }


def _ctx(features: torch.Tensor, name: str) -> torch.Tensor:
    idx = CONTEXT_NUMERIC_KEYS.index(name)
    return features[:, _CTX_OFF + idx]


def compute_v1f_repair_direction_score(
    features: torch.Tensor, source_failure_mode_idx: torch.Tensor
) -> torch.Tensor:
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
        off = len(INSERTION_PARAM_KEYS)
        gain = masked_theta[tr, off + TRANSPORT_PARAM_KEYS.index("transport_xy_gain")]
        scale = masked_theta[tr, off + TRANSPORT_PARAM_KEYS.index("transport_xy_offset_scale")]
        score[tr] = 0.5 * (
            torch.clamp((gain - 0.4) / 0.6, 0.0, 1.0) + torch.clamp((scale - 0.5) / 0.75, 0.0, 1.0)
        )

    gr = source_failure_mode_idx == 3
    if torch.any(gr):
        off = len(INSERTION_PARAM_KEYS) + len(TRANSPORT_PARAM_KEYS)
        micro = masked_theta[gr, off + GRASP_LIFT_PARAM_KEYS.index("micro_lift_height")]
        score[gr] = torch.clamp((micro - 0.04) / 0.08, 0.0, 1.0)

    lift = source_failure_mode_idx == 4
    if torch.any(lift):
        off = len(INSERTION_PARAM_KEYS) + len(TRANSPORT_PARAM_KEYS)
        micro = masked_theta[lift, off + GRASP_LIFT_PARAM_KEYS.index("micro_lift_height")]
        lift_steps = masked_theta[lift, off + len(GRASP_LIFT_PARAM_KEYS) + LIFT_EXTRA_PARAM_KEYS.index("micro_lift_steps")]
        score[lift] = 0.5 * (
            torch.clamp((micro - 0.04) / 0.08, 0.0, 1.0)
            + torch.clamp((lift_steps - 10.0) / 30.0, 0.0, 1.0)
        )
    return score


def explicit_v1f_repair_energy(features: torch.Tensor) -> torch.Tensor:
    """Heuristic explicit energy for baseline comparison."""
    orig_xy = _ctx(features, "original_final_xy")
    orig_z = _ctx(features, "original_final_z_diff")
    orig_e = _ctx(features, "original_E_total_norm")
    theta = features[:, _THETA_OFF:_MASK_OFF]
    mask = features[:, _MASK_OFF:]
    masked = theta * mask

    insert_z = masked[:, INSERTION_PARAM_KEYS.index("insert_z_offset")]
    z_gain = masked[:, INSERTION_PARAM_KEYS.index("z_gain")]
    micro_lift = masked[
        :, len(INSERTION_PARAM_KEYS) + len(TRANSPORT_PARAM_KEYS) + GRASP_LIFT_PARAM_KEYS.index("micro_lift_height")
    ]

    repair_bonus = (
        torch.clamp(-insert_z, 0.0, 0.12) * 40.0
        + torch.clamp(z_gain - 0.55, 0.0, 0.45) * 30.0
        + torch.clamp(micro_lift - 0.04, 0.0, 0.08) * 50.0
    )
    return torch.clamp(orig_e - repair_bonus + orig_xy * 10.0 + torch.abs(orig_z + 0.02) * 20.0, min=0.0)


def compute_v1f_pinn_losses(
    out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    physics: V1FPhysicsLossConfig | None = None,
) -> dict[str, torch.Tensor]:
    physics = physics or V1FPhysicsLossConfig.full()
    losses: dict[str, torch.Tensor] = {}

    if physics.use_component_supervision:
        losses["component_mse"] = F.mse_loss(out["E_components"], batch["targets_components"])
    else:
        losses["component_mse"] = torch.tensor(0.0, device=out["E_total"].device)

    if physics.use_lift_residual_supervision:
        losses["lift_residual_mse"] = F.mse_loss(out["lift_residuals"], batch["lift_residuals"])
    else:
        losses["lift_residual_mse"] = torch.tensor(0.0, device=out["E_total"].device)

    if physics.use_total_consistency:
        losses["total_consistency"] = F.mse_loss(out["E_total"], out["E_total_consistent"])
    else:
        losses["total_consistency"] = torch.tensor(0.0, device=out["E_total"].device)

    losses["total_supervision"] = F.mse_loss(out["E_total"], batch["target_E_total"])

    if physics.use_uncertainty_nll:
        log_var = torch.clamp(out["log_var"], min=-8.0, max=8.0)
        var = torch.exp(log_var).clamp(min=1e-4, max=1e4)
        nll = 0.5 * (torch.log(var) + (batch["target_E_total"] - out["E_total"]) ** 2 / var)
        losses["uncertainty_nll"] = nll.mean()
    else:
        losses["uncertainty_nll"] = torch.tensor(0.0, device=out["E_total"].device)

    if physics.use_success_margin:
        success = batch["success_flag"]
        margin_loss = success * F.relu(out["E_total"] - physics.margin) + (1.0 - success) * F.relu(
            physics.margin - out["E_total"]
        )
        losses["success_margin"] = margin_loss.mean()
    else:
        losses["success_margin"] = torch.tensor(0.0, device=out["E_total"].device)

    losses["success_bce"] = F.binary_cross_entropy_with_logits(out["success_logit"], batch["success_flag"])
    losses["failure_ce"] = F.cross_entropy(out["failure_type_logits"], batch["failure_type_idx"])
    losses["grasp_bce"] = F.binary_cross_entropy_with_logits(
        out["grasp_success_logit"], batch["grasp_success_flag"]
    )
    losses["lift_bce"] = F.binary_cross_entropy_with_logits(out["lift_success_logit"], batch["lift_success_flag"])

    if physics.use_monotonic_repair:
        direction = compute_v1f_repair_direction_score(batch["features"], batch["source_failure_mode_idx"])
        baseline = batch["original_E_total"]
        predicted_drop = baseline - out["E_total"]
        mono = direction * F.relu(-predicted_drop)
        losses["monotonic_repair"] = mono.mean()
    else:
        losses["monotonic_repair"] = torch.tensor(0.0, device=out["E_total"].device)

    losses["loss"] = (
        physics.w_component * losses["component_mse"]
        + physics.w_lift_residual * losses["lift_residual_mse"]
        + physics.w_total_consistency * losses["total_consistency"]
        + losses["total_supervision"]
        + physics.w_uncertainty * losses["uncertainty_nll"]
        + physics.w_success_margin * losses["success_margin"]
        + 0.3 * losses["success_bce"]
        + 0.2 * losses["failure_ce"]
        + 0.2 * losses["grasp_bce"]
        + 0.2 * losses["lift_bce"]
        + physics.w_monotonic * losses["monotonic_repair"]
    )
    return losses


FAILURE_TYPE_RESIDUAL_WEIGHTS = torch.tensor([1.0, 1.4, 1.5, 1.2, 1.3, 1.0], dtype=torch.float32)


def _focal_bce_with_logits(logits: torch.Tensor, targets: torch.Tensor, *, gamma: float = 2.0) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = torch.sigmoid(logits)
    pt = targets * probs + (1.0 - targets) * (1.0 - probs)
    return ((1.0 - pt) ** gamma * bce).mean()


def _pairwise_ranking_loss(
    e_total: torch.Tensor,
    demo_group_id: torch.Tensor,
    target_e: torch.Tensor,
    ranking_eligible: torch.Tensor,
) -> torch.Tensor:
    loss_terms: list[torch.Tensor] = []
    eligible = ranking_eligible > 0.5
    if not torch.any(eligible):
        return torch.tensor(0.0, device=e_total.device)

    for gid in torch.unique(demo_group_id[eligible]):
        if int(gid) < 0:
            continue
        mask = eligible & (demo_group_id == gid)
        if int(mask.sum()) < 2:
            continue
        pred = e_total[mask]
        tgt = target_e[mask]
        n = pred.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                if tgt[i] == tgt[j]:
                    continue
                if tgt[i] < tgt[j]:
                    loss_terms.append(F.relu(pred[i] - pred[j] + 0.5))
                else:
                    loss_terms.append(F.relu(pred[j] - pred[i] + 0.5))
    if not loss_terms:
        return torch.tensor(0.0, device=e_total.device)
    return torch.stack(loss_terms).mean()


def compute_v1f_balanced_losses(
    out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    physics: V1FPhysicsLossConfig | None = None,
) -> dict[str, torch.Tensor]:
    physics = physics or V1FPhysicsLossConfig.balanced()
    losses = compute_v1f_pinn_losses(out, batch, physics=physics)

    sample_weight = batch.get("sample_weight")
    if sample_weight is not None:
        w = sample_weight / sample_weight.mean().clamp(min=1e-6)
        losses["total_supervision"] = (w * (out["E_total"] - batch["target_E_total"]) ** 2).mean()
    else:
        losses["total_supervision"] = F.mse_loss(out["E_total"], batch["target_E_total"])

    ft_idx = batch["source_failure_mode_idx"].clamp(min=0, max=len(FAILURE_TYPE_RESIDUAL_WEIGHTS) - 1)
    ft_w = FAILURE_TYPE_RESIDUAL_WEIGHTS.to(out["E_total"].device)[ft_idx]
    comp_err = (out["E_components"] - batch["targets_components"]) ** 2
    losses["failure_weighted_component"] = (ft_w.unsqueeze(-1) * comp_err).mean()

    losses["success_focal"] = _focal_bce_with_logits(out["success_logit"], batch["success_flag"], gamma=2.0)

    ranking_eligible = batch.get("ranking_supervision_eligible", torch.ones_like(batch["success_flag"]))
    demo_group_id = batch.get("demo_group_id", torch.full_like(batch["success_flag"], -1, dtype=torch.long))
    losses["pairwise_ranking"] = _pairwise_ranking_loss(
        out["E_total"], demo_group_id.long(), batch["target_E_total"], ranking_eligible
    )

    losses["loss"] = (
        physics.w_component * losses["failure_weighted_component"]
        + physics.w_lift_residual * losses["lift_residual_mse"]
        + physics.w_total_consistency * losses["total_consistency"]
        + losses["total_supervision"]
        + physics.w_uncertainty * losses["uncertainty_nll"]
        + physics.w_success_margin * losses["success_margin"]
        + 0.5 * losses["success_focal"]
        + 0.15 * losses["pairwise_ranking"]
        + 0.15 * losses["failure_ce"]
        + 0.15 * losses["grasp_bce"]
        + 0.15 * losses["lift_bce"]
        + physics.w_monotonic * losses["monotonic_repair"]
    )
    return losses


def compute_v1f_100base_losses(
    out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    physics: V1FPhysicsLossConfig | None = None,
) -> dict[str, torch.Tensor]:
    """V1-F-100Base：balanced losses + old demo retention。"""
    physics = physics or V1FPhysicsLossConfig.hundredbase()
    losses = compute_v1f_balanced_losses(out, batch, physics=physics)

    old_ret = batch.get("old_demo_retention")
    if old_ret is not None:
        err = (out["E_total"] - batch["target_E_total"]) ** 2
        retention_w = 1.0 + 2.5 * old_ret
        losses["old_demo_retention"] = (retention_w * err).mean()
    else:
        losses["old_demo_retention"] = torch.tensor(0.0, device=out["E_total"].device)

    losses["loss"] = losses["loss"] + 0.3 * losses["old_demo_retention"]
    return losses


def compute_v1g_stage1_losses(
    out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    physics: V1GStage1PhysicsLossConfig | None = None,
) -> dict[str, torch.Tensor]:
    """V1-G-stage1-p1xy：100Base base + P1/P2 physics aux（不含 P3/P4）。"""
    physics = physics or V1GStage1PhysicsLossConfig.stage1_p1xy()
    losses = compute_v1f_100base_losses(out, batch, physics=physics)

    if physics.use_physics_aux_loss:
        pred = out["E_components"]
        tgt = batch["targets_components"]
        losses["physics_aux_transport"] = F.mse_loss(
            pred[:, _V1F_IDX_TRANSPORT], tgt[:, _V1F_IDX_TRANSPORT]
        )
        losses["physics_aux_xy"] = F.mse_loss(pred[:, _V1F_IDX_XY], tgt[:, _V1F_IDX_XY])
        losses["physics_aux_lift"] = F.mse_loss(pred[:, _V1F_IDX_LIFT], tgt[:, _V1F_IDX_LIFT])
        aux = (
            physics.lambda_transport * losses["physics_aux_transport"]
            + physics.lambda_xy * losses["physics_aux_xy"]
            + physics.lambda_lift * losses["physics_aux_lift"]
        )
        losses["physics_aux_total"] = aux
        losses["loss"] = losses["loss"] + aux
    else:
        z = torch.tensor(0.0, device=out["E_total"].device)
        losses["physics_aux_transport"] = z
        losses["physics_aux_xy"] = z
        losses["physics_aux_lift"] = z
        losses["physics_aux_total"] = z
    return losses


def compute_v1g_stage1_lite_losses(
    out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    physics: V1GStage1LiteP1P2Config | None = None,
) -> dict[str, torch.Tensor]:
    """V1-G-stage1-lite-p1p2：100Base + 轻量 P1/P2 physics aux + 略提高 retention。"""
    physics = physics or V1GStage1LiteP1P2Config.lite_p1p2()
    losses = compute_v1f_100base_losses(out, batch, physics=physics)
    losses["loss"] = (
        losses["loss"]
        - 0.3 * losses["old_demo_retention"]
        + physics.lambda_retention * losses["old_demo_retention"]
    )

    if physics.use_physics_aux_loss:
        pred = out["E_components"]
        tgt = batch["targets_components"]
        losses["physics_aux_transport"] = F.mse_loss(
            pred[:, _V1F_IDX_TRANSPORT], tgt[:, _V1F_IDX_TRANSPORT]
        )
        losses["physics_aux_xy"] = F.mse_loss(pred[:, _V1F_IDX_XY], tgt[:, _V1F_IDX_XY])
        losses["physics_aux_lift"] = F.mse_loss(pred[:, _V1F_IDX_LIFT], tgt[:, _V1F_IDX_LIFT])
        aux = (
            physics.lambda_transport * losses["physics_aux_transport"]
            + physics.lambda_xy * losses["physics_aux_xy"]
            + physics.lambda_lift * losses["physics_aux_lift"]
        )
        losses["physics_aux_total"] = aux
        losses["loss"] = losses["loss"] + aux
    else:
        z = torch.tensor(0.0, device=out["E_total"].device)
        losses["physics_aux_transport"] = z
        losses["physics_aux_xy"] = z
        losses["physics_aux_lift"] = z
        losses["physics_aux_total"] = z
    return losses


def compute_v1f_100base_r1_losses(
    out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    physics: V1FPhysicsLossConfig | None = None,
) -> dict[str, torch.Tensor]:
    """V1-F-100Base-R1：demo_uid 分组；success reference 不参与 focal / pairwise。"""
    physics = physics or V1FPhysicsLossConfig.hundredbase()
    losses = compute_v1f_pinn_losses(out, batch, physics=physics)

    is_cal = batch.get("is_success_reference")
    if is_cal is None:
        repair_mask = torch.ones_like(batch["success_flag"])
    else:
        repair_mask = (is_cal <= 0.5).float()

    sample_weight = batch.get("sample_weight")
    if sample_weight is not None:
        w = sample_weight / sample_weight.mean().clamp(min=1e-6)
        sup = w * (out["E_total"] - batch["target_E_total"]) ** 2
        losses["total_supervision"] = sup.mean()
    else:
        losses["total_supervision"] = F.mse_loss(out["E_total"], batch["target_E_total"])

    ft_idx = batch["source_failure_mode_idx"].clamp(min=0, max=len(FAILURE_TYPE_RESIDUAL_WEIGHTS) - 1)
    ft_w = FAILURE_TYPE_RESIDUAL_WEIGHTS.to(out["E_total"].device)[ft_idx]
    comp_err = (out["E_components"] - batch["targets_components"]) ** 2
    losses["failure_weighted_component"] = (ft_w.unsqueeze(-1) * comp_err).mean()

    focal = F.binary_cross_entropy_with_logits(out["success_logit"], batch["success_flag"], reduction="none")
    focal_probs = torch.sigmoid(out["success_logit"])
    focal_pt = batch["success_flag"] * focal_probs + (1.0 - batch["success_flag"]) * (1.0 - focal_probs)
    focal_weighted = (1.0 - focal_pt) ** 2 * focal
    if repair_mask.sum() > 0:
        losses["success_focal"] = (focal_weighted * repair_mask).sum() / repair_mask.sum().clamp(min=1.0)
    else:
        losses["success_focal"] = torch.tensor(0.0, device=out["E_total"].device)

    ranking_eligible = batch.get("ranking_supervision_eligible", torch.ones_like(batch["success_flag"]))
    ranking_eligible = ranking_eligible * repair_mask
    demo_group_id = batch.get("demo_group_id", torch.full_like(batch["success_flag"], -1, dtype=torch.long))
    losses["pairwise_ranking"] = _pairwise_ranking_loss(
        out["E_total"], demo_group_id.long(), batch["target_E_total"], ranking_eligible
    )

    old_ret = batch.get("old_demo_retention")
    if old_ret is not None and old_ret.sum() > 0:
        err = (out["E_total"] - batch["target_E_total"]) ** 2
        retention_w = 1.0 + 2.5 * old_ret
        ret_mask = (old_ret > 0.5).float()
        losses["old_demo_retention"] = (retention_w * err * ret_mask).sum() / ret_mask.sum().clamp(min=1.0)
    else:
        losses["old_demo_retention"] = torch.tensor(0.0, device=out["E_total"].device)

    losses["loss"] = (
        physics.w_component * losses["failure_weighted_component"]
        + physics.w_lift_residual * losses["lift_residual_mse"]
        + physics.w_total_consistency * losses["total_consistency"]
        + losses["total_supervision"]
        + physics.w_uncertainty * losses["uncertainty_nll"]
        + physics.w_success_margin * losses["success_margin"]
        + 0.5 * losses["success_focal"]
        + 0.15 * losses["pairwise_ranking"]
        + 0.15 * losses["failure_ce"]
        + 0.15 * losses["grasp_bce"]
        + 0.15 * losses["lift_bce"]
        + physics.w_monotonic * losses["monotonic_repair"]
        + 0.3 * losses["old_demo_retention"]
    )
    return losses
