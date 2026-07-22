from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from phygen.adapters.base_adapter import BasePhyGenAdapter
from phygen.core.residual_field_model import (
    PINN_BETA,
    PINN_FAILURE_ENERGY_FLOOR,
    PINN_SUCCESS_CORRECTION_SCALE,
    PINN_SUCCESS_ENERGY_THRESHOLD,
    FeatureLayout,
    RepairParameterResidualFieldPINN,
    build_mlp_selector,
)


@dataclass
class TrainingResult:
    model: Any
    input_dim: int
    layout: FeatureLayout
    history: list[dict[str, float]]


class PhyGenTrainer:
    def __init__(self, adapter: BasePhyGenAdapter) -> None:
        self.adapter = adapter
        self.spec = adapter.task_spec()
        self.layout = FeatureLayout(
            context_dim=len(self.spec.context_keys),
            theta_disc_dim=len(self.spec.theta_disc_keys),
            theta_cont_dim=len(self.spec.theta_cont_keys),
        )

    def train_model(
        self,
        records: list[dict[str, Any]],
        epochs: int,
        lr: float,
        use_component_loss: bool = False,
        component_weight: float = 0.45,
        use_true_pinn: bool = False,
        pinn_weight: float = 0.35,
        use_standard_pinn: bool = False,
        standard_pinn_weight: float = 0.35,
    ) -> TrainingResult:
        import torch
        import torch.nn.functional as F

        independent_theta_idxs = torch.tensor(self.spec.theta_cont_independent_idxs, dtype=torch.long)

        x = np.stack([self.adapter.feature_vector(r["context_metrics"], r["theta"]) for r in records]).astype(np.float32)
        energy = np.array([float(r["metrics"].get("energy", 30.0)) for r in records], dtype=np.float32)
        e_target = np.clip(energy, 0.0, 30.0) / 30.0
        success = np.array([float(r["success"]) for r in records], dtype=np.float32)
        component_target = np.stack(
            [self.adapter.build_residual_targets(r.get("metrics"), bool(r.get("problematic"))) for r in records]
        ).astype(np.float32)

        xt = torch.from_numpy(x)
        et = torch.from_numpy(e_target).unsqueeze(1)
        st = torch.from_numpy(success).unsqueeze(1)
        ct = torch.from_numpy(component_target)

        use_component_head = use_component_loss or use_true_pinn or use_standard_pinn
        if use_component_head:
            model = RepairParameterResidualFieldPINN.build(self.layout, len(self.spec.component_keys))
        else:
            model = build_mlp_selector(self.layout.input_dim)

        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
        pos = float(st.sum().item())
        pos_weight = torch.tensor([(len(st) - pos) / max(pos, 1.0)], dtype=torch.float32)
        comp_weights = torch.tensor(self.spec.component_weights, dtype=torch.float32)
        comp_energy_target = torch.from_numpy(self.adapter.component_energy_target(component_target)).unsqueeze(1)

        history: list[dict[str, float]] = []
        pde_enabled = use_true_pinn or use_standard_pinn

        for epoch in range(epochs):
            xt_epoch = xt.detach().clone().requires_grad_(pde_enabled)
            raw_pred = model(xt_epoch)
            if isinstance(raw_pred, dict):
                pred = raw_pred["energy_success"]
                pred_components = raw_pred["components"]
                success_correction = raw_pred.get("success_correction")
            else:
                pred = raw_pred
                pred_components = None
                success_correction = None

            e_pred = pred[:, :1]
            s_logit = pred[:, 1:2]
            loss_e = F.smooth_l1_loss(e_pred, et)
            loss_s = F.binary_cross_entropy_with_logits(s_logit, st, pos_weight=pos_weight)
            loss_success_correction = torch.zeros((), dtype=torch.float32)
            if success_correction is not None:
                loss_success_correction = (success_correction / PINN_SUCCESS_CORRECTION_SCALE).pow(2).mean()

            s_from_e = torch.sigmoid((PINN_SUCCESS_ENERGY_THRESHOLD - e_pred) * PINN_BETA)
            loss_cons = F.binary_cross_entropy(s_from_e, st)

            success_margin = torch.relu(e_pred - PINN_SUCCESS_ENERGY_THRESHOLD) * st
            fail_margin = torch.relu(PINN_FAILURE_ENERGY_FLOOR - e_pred) * (1.0 - st)
            loss_margin = success_margin.mean() + fail_margin.mean()

            loss_components = torch.zeros((), dtype=torch.float32)
            loss_component_total = torch.zeros((), dtype=torch.float32)
            loss_component_success = torch.zeros((), dtype=torch.float32)
            loss_total_consistency = torch.zeros((), dtype=torch.float32)
            loss_physics_residual = torch.zeros((), dtype=torch.float32)
            loss_boundary = torch.zeros((), dtype=torch.float32)
            loss_differential = torch.zeros((), dtype=torch.float32)
            loss_hjb_pde = torch.zeros((), dtype=torch.float32)
            loss_collocation_transport_pde = torch.zeros((), dtype=torch.float32)

            if pred_components is not None:
                loss_components = F.smooth_l1_loss(pred_components, ct)
                comp_e = (pred_components * comp_weights[None, :]).sum(dim=1, keepdim=True) / comp_weights.sum()
                loss_component_total = F.smooth_l1_loss(comp_e, comp_energy_target)
                loss_component_success = F.binary_cross_entropy(
                    torch.sigmoid((PINN_SUCCESS_ENERGY_THRESHOLD - comp_e) * PINN_BETA), st
                )

                if pde_enabled:
                    loss_total_consistency = F.smooth_l1_loss(e_pred, comp_e)
                    loss_physics_residual = self.adapter.physical_component_loss(pred_components)

                    comp_mean = pred_components.mean(dim=1, keepdim=True)
                    success_boundary = st * (e_pred.pow(2) + pred_components.pow(2).mean(dim=1, keepdim=True))
                    failure_boundary = (1.0 - st) * (
                        torch.relu(0.08 - comp_mean).pow(2) + torch.relu(PINN_FAILURE_ENERGY_FLOOR - e_pred).pow(2)
                    )
                    loss_boundary = success_boundary.mean() + failure_boundary.mean()

                    p_success = torch.sigmoid(s_logit)
                    grad_e = torch.autograd.grad(
                        e_pred.sum(),
                        xt_epoch,
                        create_graph=True,
                        retain_graph=True,
                    )[0][:, self.layout.theta_cont_start : self.layout.theta_cont_end][:, independent_theta_idxs]
                    grad_p = torch.autograd.grad(
                        p_success.sum(),
                        xt_epoch,
                        create_graph=True,
                        retain_graph=True,
                    )[0][:, self.layout.theta_cont_start : self.layout.theta_cont_end][:, independent_theta_idxs]
                    differential_residual = grad_p + PINN_BETA * p_success * (1.0 - p_success) * grad_e
                    loss_differential = differential_residual.pow(2).mean()

                if use_standard_pinn:
                    x_col = xt.detach().clone()
                    z_block = x_col[:, self.layout.theta_cont_start : self.layout.theta_cont_end].clone()
                    z_noise_independent = 0.035 * torch.randn_like(z_block[:, independent_theta_idxs])
                    z_block[:, independent_theta_idxs] = torch.clamp(
                        z_block[:, independent_theta_idxs] + z_noise_independent,
                        0.0,
                        1.0,
                    )
                    x_col[:, self.layout.theta_cont_start : self.layout.theta_cont_end] = self.adapter.project_theta_cont_manifold(z_block)
                    x_col.requires_grad_(True)
                    raw_col = model(x_col)
                    col_pred = raw_col["energy_success"] if isinstance(raw_col, dict) else raw_col
                    col_components = raw_col["components"] if isinstance(raw_col, dict) else None
                    col_v = col_pred[:, :1]
                    col_p = torch.sigmoid(col_pred[:, 1:2])
                    col_grad_v = torch.autograd.grad(
                        col_v.sum(), x_col, create_graph=True, retain_graph=True
                    )[0][:, self.layout.theta_cont_start : self.layout.theta_cont_end][:, independent_theta_idxs]
                    col_grad_p = torch.autograd.grad(
                        col_p.sum(), x_col, create_graph=True, retain_graph=True
                    )[0][:, self.layout.theta_cont_start : self.layout.theta_cont_end][:, independent_theta_idxs]
                    if col_components is not None:
                        col_source = (col_components * comp_weights[None, :]).sum(dim=1, keepdim=True) / comp_weights.sum()
                    else:
                        col_source = col_v.detach().clamp_min(0.0)
                    hjb_residual = 0.5 * col_grad_v.pow(2).sum(dim=1, keepdim=True) - col_source
                    loss_hjb_pde = hjb_residual.pow(2).mean()
                    col_transport_residual = col_grad_p + PINN_BETA * col_p * (1.0 - col_p) * col_grad_v
                    loss_collocation_transport_pde = col_transport_residual.pow(2).mean()

            loss = (
                loss_e
                + 0.45 * loss_s
                + 0.30 * loss_cons
                + 0.20 * loss_margin
                + 0.05 * loss_success_correction
                + component_weight * loss_components
                + 0.25 * component_weight * loss_component_total
                + 0.20 * component_weight * loss_component_success
                + pinn_weight * loss_total_consistency
                + pinn_weight * loss_physics_residual
                + 0.50 * pinn_weight * loss_boundary
                + 0.25 * pinn_weight * loss_differential
                + standard_pinn_weight * loss_hjb_pde
                + standard_pinn_weight * loss_collocation_transport_pde
            )
            opt.zero_grad()
            loss.backward()
            opt.step()

            if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0:
                history.append(
                    {
                        "epoch": float(epoch + 1),
                        "loss": float(loss.item()),
                        "loss_energy": float(loss_e.item()),
                        "loss_success": float(loss_s.item()),
                        "loss_consistency": float(loss_cons.item()),
                        "loss_margin": float(loss_margin.item()),
                        "loss_success_correction_regularizer": float(loss_success_correction.item()),
                        "loss_components": float(loss_components.item()),
                        "loss_component_total": float(loss_component_total.item()),
                        "loss_component_success": float(loss_component_success.item()),
                        "loss_total_consistency": float(loss_total_consistency.item()),
                        "loss_physics_residual": float(loss_physics_residual.item()),
                        "loss_boundary": float(loss_boundary.item()),
                        "loss_differential_independent_theta_cont": float(loss_differential.item()),
                        "loss_hjb_pde_independent_theta_cont": float(loss_hjb_pde.item()),
                        "loss_collocation_transport_pde_independent_theta_cont": float(loss_collocation_transport_pde.item()),
                    }
                )

        return TrainingResult(model=model, input_dim=self.layout.input_dim, layout=self.layout, history=history)
