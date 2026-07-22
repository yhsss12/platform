#!/usr/bin/env python3
"""
Train a stack repair-parameter residual-field PINN from feedback rollouts
and emit a utility / boundary-union candidate plan for the next true rollout.

This version treats repair as a value-field learning problem over the continuous
MimicGen repair-parameter space.  The network learns a residual energy field
V(c,d,z), physics residual source q(c,d,z), and a success probability mainly
induced by V.  PDE/collocation losses are applied only to continuous repair
parameters z, while discrete MimicGen choices d are used only as conditions.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from stack_failed_conditioned_mimicgen_repair import demo_sort_key, load_failed_contexts, make_candidate


CONTEXT_KEYS = [
    "energy",
    "ab_xy",
    "ab_z",
    "ca_xy",
    "ca_z",
    "cb_xy",
    "cb_z",
    "c_minus_a",
    "drop_penalty",
]

COMPONENT_KEYS = [
    "E_xy",
    "E_transport",
    "E_lift",
    "E_contact",
    "E_bilateral",
    "E_dynamics",
    "E_slip",
    "E_coupling",
]

COMPONENT_WEIGHTS = np.array([1.2, 1.0, 1.2, 1.1, 0.8, 0.8, 1.0, 1.2], dtype=np.float32)


# Repair-parameter PINN layout.  The feature vector remains backward
# compatible with the original script, but the PINN PDE is applied only
# to the continuous repair-parameter block instead of to all theta bits.
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

THETA_DISC_START = len(CONTEXT_KEYS)
THETA_DISC_END = THETA_DISC_START + len(THETA_DISC_KEYS)
THETA_CONT_START = THETA_DISC_END
THETA_CONT_END = THETA_CONT_START + len(THETA_CONT_KEYS)
BOUNDARY_BONUS_INDEX = THETA_CONT_END
PINN_BETA = 8.0
PINN_SUCCESS_ENERGY_THRESHOLD = 0.25
PINN_FAILURE_ENERGY_FLOOR = 0.12
PINN_SUCCESS_CORRECTION_SCALE = 0.25


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


def theta_to_features(theta: dict[str, Any]) -> tuple[np.ndarray, float]:
    strategy = theta.get("selection_strategy", "nearest_neighbor_object")
    offset = theta.get("offset_range", [10, 20])
    lo, hi = float(offset[0]), float(offset[1])
    vals = np.array(
        [
            float(bool(theta.get("select_src_per_subtask", False))),
            float(bool(theta.get("transform_first_robot_pose", False))),
            float(bool(theta.get("interpolate_from_last_target_pose", True))),
            float(strategy == "nearest_neighbor_object"),
            float(strategy == "random"),
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
    continuous = np.array(
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
    edge = np.minimum(np.clip(continuous, 0.0, 1.0), 1.0 - np.clip(continuous, 0.0, 1.0))
    boundary_bonus = float(1.0 - np.mean(edge))
    return vals, boundary_bonus


def component_targets_from_metrics(metrics: dict[str, float] | None, problematic: bool = False) -> np.ndarray:
    """Eight StackThree physics residual proxies scaled to roughly [0, 1]."""
    if problematic or metrics is None:
        return np.ones(len(COMPONENT_KEYS), dtype=np.float32)
    ab_xy = float(metrics.get("ab_xy", 0.18))
    ab_z = float(metrics.get("ab_z", 0.12))
    ca_xy = float(metrics.get("ca_xy", 0.18))
    ca_z = float(metrics.get("ca_z", 0.12))
    cb_xy = float(metrics.get("cb_xy", 0.18))
    cb_z = float(metrics.get("cb_z", 0.12))
    c_minus_a = float(metrics.get("c_minus_a", -0.05))
    drop = float(metrics.get("drop_penalty", max(0.0, 0.018 - c_minus_a)))

    a_stack = 0.5 * (ab_xy / 0.030 + ab_z / 0.020)
    c_on_a = 0.5 * (ca_xy / 0.045 + ca_z / 0.025)
    c_global = 0.5 * (cb_xy / 0.075 + cb_z / 0.050)
    e_xy = (ab_xy / 0.030 + ca_xy / 0.045 + cb_xy / 0.075) / 3.0
    e_transport = (ca_xy / 0.070 + cb_xy / 0.090 + max(0.0, 0.02 - c_minus_a) / 0.040) / 3.0
    e_lift = (ab_z / 0.025 + ca_z / 0.030 + cb_z / 0.060 + drop / 0.020) / 4.0
    e_contact = 0.5 * (a_stack + c_on_a)
    # Bilateral is a proxy for object-response consistency: good hand/source
    # behavior should improve both stacks instead of only one relation.
    e_bilateral = abs(a_stack - c_on_a) + 0.25 * c_global
    e_dynamics = drop / 0.020 + float(metrics.get("energy", 30.0)) / 60.0
    e_slip = drop / 0.015 + max(0.0, 0.035 - c_minus_a) / 0.050 + ca_xy / 0.080
    e_coupling = max(a_stack, c_on_a) + 0.5 * min(a_stack, c_on_a)
    comps = np.array(
        [e_xy, e_transport, e_lift, e_contact, e_bilateral, e_dynamics, e_slip, e_coupling],
        dtype=np.float32,
    )
    return np.clip(comps, 0.0, 1.0)


def physics_residual_targets_from_metrics(metrics: dict[str, float] | None, problematic: bool = False) -> np.ndarray:
    """Backward-compatible wrapper for StackThree physics residual targets.

    The current implementation uses physically interpretable rollout-metric
    proxies.  The wrapper name makes it explicit that these targets can later
    be replaced or augmented by MuJoCo contact, penetration, velocity, and
    smoothness residuals without changing the training loop.
    """
    return component_targets_from_metrics(metrics, problematic)


def component_energy_target(components: np.ndarray) -> np.ndarray:
    weighted = components * COMPONENT_WEIGHTS[None, :]
    return (weighted.sum(axis=1) / COMPONENT_WEIGHTS.sum()).astype(np.float32)


def make_safe_candidate(index: int, rng: np.random.Generator) -> dict[str, Any]:
    """Stable StackThree theta pool learned from feedback diagnostics."""
    theta = make_candidate(index, rng)
    theta["num_fixed_steps"] = 0

    # These two settings almost always avoid rollout exceptions in the feedback
    # data while preserving useful source-subtask recombination.
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
        [
            [10, 20],
            [10, 15],
            [15, 20],
            [5, 20],
            [10, 25],
            [0, 20],
            [15, 25],
            [0, 15],
            [5, 15],
            [10, 10],
            [15, 15],
        ],
        dtype=int,
    )
    offset_probs = np.array([0.20, 0.15, 0.13, 0.10, 0.09, 0.07, 0.07, 0.06, 0.06, 0.04, 0.03])
    theta["offset_range"] = offset_options[int(rng.choice(len(offset_options), p=offset_probs))].tolist()
    theta["candidate_family"] = (
        f"{theta['selection_strategy']}_per{int(theta['select_src_per_subtask'])}"
        f"_noise{theta['action_noise']}_interp{theta['num_interpolation_steps']}_safe"
    )
    return theta


def feature_vector(context: dict[str, float], theta: dict[str, Any]) -> np.ndarray:
    ctx_scales = {
        "energy": 30.0,
        "ab_xy": 0.12,
        "ab_z": 0.08,
        "ca_xy": 0.16,
        "ca_z": 0.08,
        "cb_xy": 0.18,
        "cb_z": 0.12,
        "c_minus_a": 0.12,
        "drop_penalty": 0.06,
    }
    ctx = np.array(
        [float(context.get(k, 0.0)) / ctx_scales[k] for k in CONTEXT_KEYS],
        dtype=np.float32,
    )
    theta_feat, boundary_bonus = theta_to_features(theta)
    return np.concatenate([ctx, theta_feat, np.array([boundary_bonus], dtype=np.float32)])


def train_model(
    records: list[dict[str, Any]],
    epochs: int,
    lr: float,
    use_component_loss: bool = False,
    component_weight: float = 0.45,
    use_true_pinn: bool = False,
    pinn_weight: float = 0.35,
    use_standard_pinn: bool = False,
    standard_pinn_weight: float = 0.35,
):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    x = np.stack([feature_vector(r["context_metrics"], r["theta"]) for r in records]).astype(np.float32)
    energy = np.array([float(r["metrics"]["energy"]) for r in records], dtype=np.float32)
    e_target = np.clip(energy, 0.0, 30.0) / 30.0
    success = np.array([float(r["success"]) for r in records], dtype=np.float32)
    component_target = np.stack(
        [physics_residual_targets_from_metrics(r.get("metrics"), bool(r.get("problematic"))) for r in records]
    ).astype(np.float32)

    xt = torch.from_numpy(x)
    et = torch.from_numpy(e_target).unsqueeze(1)
    st = torch.from_numpy(success).unsqueeze(1)
    ct = torch.from_numpy(component_target)

    use_component_head = use_component_loss or use_true_pinn or use_standard_pinn

    if use_component_head:
        class RepairParameterPINN(nn.Module):
            """Repair-Parameter Residual Field PINN.

            Mathematical object:
                V_phi(c, d, z): residual value / cost-to-success field
                q_phi(c, d, z): weighted physics residual source
                p_phi(c, d, z): success probability induced mainly by V

            c is the failed-trajectory context, d is the discrete MimicGen
            repair mode, and z is the continuous repair-parameter vector.  The
            PDE residuals are only defined on z.  This makes the model a
            residual-field model instead of a generic MLP with arbitrary PINN
            losses attached to mixed discrete/continuous features.
            """

            def __init__(self, input_dim: int):
                super().__init__()
                self.input_dim = input_dim
                self.context_dim = len(CONTEXT_KEYS)
                self.theta_disc_dim = len(THETA_DISC_KEYS)
                self.theta_cont_dim = len(THETA_CONT_KEYS)
                self.boundary_dim = 1
                self.model_family = "repair_parameter_residual_field_pinn"

                self.discrete_encoder = nn.Sequential(
                    nn.Linear(self.theta_disc_dim, 32),
                    nn.SiLU(),
                    nn.LayerNorm(32),
                    nn.Linear(32, 32),
                    nn.SiLU(),
                )
                self.backbone = nn.Sequential(
                    nn.Linear(self.context_dim + 32 + self.theta_cont_dim + self.boundary_dim, 128),
                    nn.SiLU(),
                    nn.LayerNorm(128),
                    nn.Linear(128, 96),
                    nn.SiLU(),
                    nn.LayerNorm(96),
                )
                self.head_value = nn.Linear(96, 1)
                self.head_success_correction = nn.Linear(96, 1)
                self.head_components = nn.Linear(96, len(COMPONENT_KEYS))

            def split(self, inp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                context = inp[:, :THETA_DISC_START]
                theta_disc = inp[:, THETA_DISC_START:THETA_DISC_END]
                theta_cont = inp[:, THETA_CONT_START:THETA_CONT_END]
                boundary = inp[:, BOUNDARY_BONUS_INDEX : BOUNDARY_BONUS_INDEX + 1]
                return context, theta_disc, theta_cont, boundary

            def forward(self, inp: torch.Tensor) -> dict[str, torch.Tensor]:
                context, theta_disc, theta_cont, boundary = self.split(inp)
                theta_disc_emb = self.discrete_encoder(theta_disc)
                h = self.backbone(torch.cat([context, theta_disc_emb, theta_cont, boundary], dim=1))

                # V is a normalized non-negative residual value field in [0, 1].
                # Success probability is induced mainly by V through
                # sigmoid(beta * (tau - V)); the correction head is deliberately
                # small and regularized so that success prediction does not become
                # an independent black-box classifier detached from the field.
                value = torch.sigmoid(self.head_value(h))
                correction = PINN_SUCCESS_CORRECTION_SCALE * torch.tanh(self.head_success_correction(h))
                success_logit = PINN_BETA * (PINN_SUCCESS_ENERGY_THRESHOLD - value) + correction
                components = torch.sigmoid(self.head_components(h))
                return {
                    "energy_success": torch.cat([value, success_logit], dim=1),
                    "components": components,
                    "residual_value": value,
                    "success_correction": correction,
                }

        model = RepairParameterPINN(x.shape[1])
    else:
        model = nn.Sequential(
            nn.Linear(x.shape[1], 128),
            nn.SiLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 96),
            nn.SiLU(),
            nn.LayerNorm(96),
            nn.Linear(96, 2),
        )

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    pos = float(st.sum().item())
    pos_weight = torch.tensor([(len(st) - pos) / max(pos, 1.0)], dtype=torch.float32)
    comp_weights = torch.from_numpy(COMPONENT_WEIGHTS).float()
    comp_energy_target = torch.from_numpy(component_energy_target(component_target)).unsqueeze(1)

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
        loss_success_correction = torch.zeros((), dtype=torch.float32)
        loss_s = F.binary_cross_entropy_with_logits(s_logit, st, pos_weight=pos_weight)
        if success_correction is not None:
            loss_success_correction = (success_correction / PINN_SUCCESS_CORRECTION_SCALE).pow(2).mean()

        # Physics consistency: lower residual energy should imply higher success probability.
        s_from_e = torch.sigmoid((PINN_SUCCESS_ENERGY_THRESHOLD - e_pred) * PINN_BETA)
        loss_cons = F.binary_cross_entropy(s_from_e, st)

        # Soft margin: successful candidates should have low residual energy, failures should
        # not collapse to an artificially near-zero residual field.
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
                # PINN residual 1: scalar value field V should be consistent with
                # the weighted physical residual source q.
                loss_total_consistency = F.smooth_l1_loss(e_pred, comp_e)

                e_xy, e_transport, e_lift, e_contact, e_bilateral, e_dynamics, e_slip, e_coupling = [
                    pred_components[:, i : i + 1] for i in range(len(COMPONENT_KEYS))
                ]
                # Differentiable physical compatibility between component fields.
                contact_target = 0.5 * (e_xy + e_lift)
                bilateral_target = torch.abs(e_transport - e_lift)
                dynamics_target = 0.5 * (e_slip + e_bilateral)
                slip_floor = 0.5 * e_lift
                coupling_floor = torch.maximum(e_contact, 0.5 * (e_xy + e_lift))
                loss_contact_res = F.smooth_l1_loss(e_contact, contact_target)
                loss_bilateral_res = F.smooth_l1_loss(e_bilateral, bilateral_target)
                loss_dynamics_res = F.smooth_l1_loss(e_dynamics, dynamics_target)
                loss_slip_res = torch.relu(slip_floor - e_slip).pow(2).mean()
                loss_coupling_res = torch.relu(coupling_floor - e_coupling).pow(2).mean()
                loss_physics_residual = (
                    loss_contact_res
                    + loss_bilateral_res
                    + loss_dynamics_res
                    + loss_slip_res
                    + loss_coupling_res
                ) / 5.0

                # Boundary conditions on the residual manifold.
                comp_mean = pred_components.mean(dim=1, keepdim=True)
                success_boundary = st * (e_pred.pow(2) + pred_components.pow(2).mean(dim=1, keepdim=True))
                failure_boundary = (1.0 - st) * (
                    torch.relu(0.08 - comp_mean).pow(2) + torch.relu(PINN_FAILURE_ENERGY_FLOOR - e_pred).pow(2)
                )
                loss_boundary = success_boundary.mean() + failure_boundary.mean()

                # Differential consistency only on continuous repair parameters z.
                # If p = sigmoid(beta * (tau - V)), then:
                # grad_z p + beta * p * (1-p) * grad_z V = 0.
                p_success = torch.sigmoid(s_logit)
                grad_e = torch.autograd.grad(
                    e_pred.sum(),
                    xt_epoch,
                    create_graph=True,
                    retain_graph=True,
                )[0][:, THETA_CONT_START:THETA_CONT_END]
                grad_p = torch.autograd.grad(
                    p_success.sum(),
                    xt_epoch,
                    create_graph=True,
                    retain_graph=True,
                )[0][:, THETA_CONT_START:THETA_CONT_END]
                differential_residual = grad_p + PINN_BETA * p_success * (1.0 - p_success) * grad_e
                loss_differential = differential_residual.pow(2).mean()

            if use_standard_pinn:
                # Standard PINN collocation: enforce governing PDEs at unlabeled
                # relaxed continuous-theta points.  Context and discrete MimicGen
                # repair mode stay fixed; only z is perturbed.
                x_col = xt.detach().clone()
                z_noise = 0.035 * torch.randn_like(x_col[:, THETA_CONT_START:THETA_CONT_END])
                x_col[:, THETA_CONT_START:THETA_CONT_END] = torch.clamp(
                    x_col[:, THETA_CONT_START:THETA_CONT_END] + z_noise,
                    0.0,
                    1.0,
                )
                x_col.requires_grad_(True)
                raw_col = model(x_col)
                col_pred = raw_col["energy_success"] if isinstance(raw_col, dict) else raw_col
                col_components = raw_col["components"] if isinstance(raw_col, dict) else None
                col_v = col_pred[:, :1]
                col_p = torch.sigmoid(col_pred[:, 1:2])
                col_grad_v = torch.autograd.grad(
                    col_v.sum(),
                    x_col,
                    create_graph=True,
                    retain_graph=True,
                )[0][:, THETA_CONT_START:THETA_CONT_END]
                col_grad_p = torch.autograd.grad(
                    col_p.sum(),
                    x_col,
                    create_graph=True,
                    retain_graph=True,
                )[0][:, THETA_CONT_START:THETA_CONT_END]

                if col_components is not None:
                    col_source = (col_components * comp_weights[None, :]).sum(dim=1, keepdim=True) / comp_weights.sum()
                else:
                    col_source = col_v.detach().clamp_min(0.0)

                # Governing equation 1: stationary HJB/Eikonal-style value-field
                # residual over continuous repair-parameter space z.
                # V(c,d,z)=E_pred(c,d,z), q(c,d,z)=weighted component residual source.
                hjb_residual = 0.5 * col_grad_v.pow(2).sum(dim=1, keepdim=True) - col_source
                loss_hjb_pde = hjb_residual.pow(2).mean()

                # Governing equation 2: success probability is transported along
                # the value-field gradient in z-space.
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
                    "loss_differential_theta_cont": float(loss_differential.item()),
                    "loss_hjb_pde_theta_cont": float(loss_hjb_pde.item()),
                    "loss_collocation_transport_pde_theta_cont": float(loss_collocation_transport_pde.item()),
                }
            )
    return model, x.shape[1], history


def load_checkpoint_model(checkpoint_path: str | Path) -> tuple[Any, int, dict[str, Any]]:
    import torch
    import torch.nn as nn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)
    input_dim = int(ckpt.get("input_dim", BOUNDARY_BONUS_INDEX + 1))
    use_component_head = bool(ckpt.get("component_keys")) or ckpt.get("model_family") == "repair_parameter_residual_field_pinn"

    if use_component_head:
        class RepairParameterPINN(nn.Module):
            def __init__(self, input_dim: int):
                super().__init__()
                self.input_dim = input_dim
                self.context_dim = len(CONTEXT_KEYS)
                self.theta_disc_dim = len(THETA_DISC_KEYS)
                self.theta_cont_dim = len(THETA_CONT_KEYS)
                self.boundary_dim = 1
                self.model_family = "repair_parameter_residual_field_pinn"
                self.discrete_encoder = nn.Sequential(
                    nn.Linear(self.theta_disc_dim, 32),
                    nn.SiLU(),
                    nn.LayerNorm(32),
                    nn.Linear(32, 32),
                    nn.SiLU(),
                )
                self.backbone = nn.Sequential(
                    nn.Linear(self.context_dim + 32 + self.theta_cont_dim + self.boundary_dim, 128),
                    nn.SiLU(),
                    nn.LayerNorm(128),
                    nn.Linear(128, 96),
                    nn.SiLU(),
                    nn.LayerNorm(96),
                )
                self.head_value = nn.Linear(96, 1)
                self.head_success_correction = nn.Linear(96, 1)
                self.head_components = nn.Linear(96, len(COMPONENT_KEYS))

            def split(self, inp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                context = inp[:, :THETA_DISC_START]
                theta_disc = inp[:, THETA_DISC_START:THETA_DISC_END]
                theta_cont = inp[:, THETA_CONT_START:THETA_CONT_END]
                boundary = inp[:, BOUNDARY_BONUS_INDEX : BOUNDARY_BONUS_INDEX + 1]
                return context, theta_disc, theta_cont, boundary

            def forward(self, inp: torch.Tensor) -> dict[str, torch.Tensor]:
                context, theta_disc, theta_cont, boundary = self.split(inp)
                theta_disc_emb = self.discrete_encoder(theta_disc)
                h = self.backbone(torch.cat([context, theta_disc_emb, theta_cont, boundary], dim=1))
                value = torch.sigmoid(self.head_value(h))
                correction = PINN_SUCCESS_CORRECTION_SCALE * torch.tanh(self.head_success_correction(h))
                success_logit = PINN_BETA * (PINN_SUCCESS_ENERGY_THRESHOLD - value) + correction
                components = torch.sigmoid(self.head_components(h))
                return {
                    "energy_success": torch.cat([value, success_logit], dim=1),
                    "components": components,
                    "residual_value": value,
                    "success_correction": correction,
                }

        model = RepairParameterPINN(input_dim)
    else:
        model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.SiLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 96),
            nn.SiLU(),
            nn.LayerNorm(96),
            nn.Linear(96, 2),
        )

    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    model.runtime_device = str(device)
    return model, input_dim, ckpt


def predict(model: Any, contexts: list[dict[str, float]], thetas: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    import torch

    x = np.stack([feature_vector(c, t) for c, t in zip(contexts, thetas)]).astype(np.float32)
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        raw_pred = model(torch.from_numpy(x).to(device))
        if isinstance(raw_pred, dict):
            pred = raw_pred["energy_success"]
            comps = raw_pred["components"].cpu().numpy()
        else:
            pred = raw_pred
            comps = None
        e = pred[:, 0].cpu().numpy() * 30.0
        p = torch.sigmoid(pred[:, 1]).cpu().numpy()
        if comps is not None:
            comp_e = component_energy_target(comps) * 30.0
            e = 0.55 * e + 0.45 * comp_e
    return e, p


def unique_union(candidates: list[dict[str, Any]], budget: int) -> list[dict[str, Any]]:
    utility_quota = max(1, int(math.ceil(0.6 * budget)))
    by_utility = sorted(candidates, key=lambda r: r["utility_score"])
    by_boundary = sorted(candidates, key=lambda r: r["boundary_score"])
    out: list[dict[str, Any]] = []
    seen: set[int] = set()
    for row in by_utility[:utility_quota] + by_boundary:
        idx = int(row["candidate_index"])
        if idx in seen:
            continue
        seen.add(idx)
        out.append(row)
        if len(out) >= budget:
            break
    return out


def offline_selector_report(records: list[dict[str, Any]], model: Any, budget: int, boundary_weight: float) -> list[dict[str, Any]]:
    demos = sorted({r["demo_key"] for r in records}, key=demo_sort_key)
    rows: list[dict[str, Any]] = []
    for demo in demos:
        group = [dict(r) for r in records if r["demo_key"] == demo]
        contexts = [r["context_metrics"] for r in group]
        thetas = [r["theta"] for r in group]
        pred_e, pred_p = predict(model, contexts, thetas)
        for rec, pe, pp in zip(group, pred_e, pred_p):
            _, boundary_bonus = theta_to_features(rec["theta"])
            rec["pred_energy"] = float(pe)
            rec["pred_success_prob"] = float(pp)
            rec["utility_score"] = float(pe - 8.0 * pp)
            rec["boundary_score"] = float(pe - 6.0 * pp - boundary_weight * boundary_bonus)
        top = unique_union(group, budget)
        rows.append(
            {
                "demo_key": demo,
                "num_candidates": len(group),
                "oracle_success": bool(any(r["success"] for r in group)),
                "selector_success": bool(any(r["success"] for r in top)),
                "selected_candidate_indices": [int(r["candidate_index"]) for r in top],
                "selected_successes": [bool(r["success"]) for r in top],
                "selected_energies": [float(r["metrics"]["energy"]) for r in top],
            }
        )
    return rows


def build_candidate_plan(
    records: list[dict[str, Any]],
    model: Any,
    out_path: Path,
    pool_size: int,
    budget: int,
    seed: int,
    start_index: int,
    boundary_weight: float,
    include_repaired: bool,
    candidate_mode: str,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        grouped.setdefault(rec["demo_key"], []).append(rec)

    plan_rows: list[dict[str, Any]] = []
    for demo_key in sorted(grouped, key=demo_sort_key):
        already_repaired = any(r["success"] for r in grouped[demo_key])
        if already_repaired and not include_repaired:
            continue
        context = grouped[demo_key][0]["context_metrics"]
        rng = np.random.default_rng(seed + demo_sort_key(demo_key) * 1009)
        pool: list[dict[str, Any]] = []
        for i in range(pool_size):
            candidate_index = start_index + i
            if candidate_mode == "safe":
                theta = make_safe_candidate(candidate_index, rng)
            else:
                theta = make_candidate(candidate_index, rng)
            pool.append({"candidate_index": candidate_index, "theta": theta})
        pred_e, pred_p = predict(model, [context] * len(pool), [p["theta"] for p in pool])
        for row, pe, pp in zip(pool, pred_e, pred_p):
            _, boundary_bonus = theta_to_features(row["theta"])
            row["pred_energy"] = float(pe)
            row["pred_success_prob"] = float(pp)
            row["utility_score"] = float(pe - 8.0 * pp)
            row["boundary_score"] = float(pe - 6.0 * pp - boundary_weight * boundary_bonus)
        selected = unique_union(pool, budget)
        for rank, row in enumerate(selected):
            row["planner_rank"] = rank
            row["planner_score"] = float(min(row["utility_score"], row["boundary_score"]))
        plan_rows.append({"demo_key": demo_key, "candidates": selected})

    with out_path.open("w", encoding="utf-8") as f:
        for row in plan_rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    return {
        "candidate_plan": str(out_path),
        "target_demo_count": len(plan_rows),
        "pool_size": pool_size,
        "budget": budget,
        "include_repaired": include_repaired,
        "candidate_mode": candidate_mode,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feedback-jsonl", default=None)
    parser.add_argument("--checkpoint", default=None, help="Load a trained .pt checkpoint and only generate a candidate plan.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--budget", type=int, default=5)
    parser.add_argument("--pool-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=9701)
    parser.add_argument("--start-index", type=int, default=100)
    parser.add_argument("--boundary-weight", type=float, default=1.5)
    parser.add_argument("--include-repaired", action="store_true")
    parser.add_argument("--target-failed-hdf5", default=None)
    parser.add_argument("--target-success-hdf5", default=None)
    parser.add_argument("--target-max-failed-demos", type=int, default=0)
    parser.add_argument("--target-exclude-success-jsonl", action="append", default=None)
    parser.add_argument("--candidate-mode", choices=["default", "safe"], default="default")
    parser.add_argument("--use-component-loss", action="store_true")
    parser.add_argument("--component-weight", type=float, default=0.45)
    parser.add_argument("--true-pinn", action="store_true")
    parser.add_argument("--pinn-weight", type=float, default=0.35)
    parser.add_argument("--standard-pinn", action="store_true")
    parser.add_argument("--standard-pinn-weight", type=float, default=0.35)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.checkpoint:
        if not args.target_failed_hdf5:
            parser.error("--target-failed-hdf5 is required when using --checkpoint")
        model, input_dim, ckpt = load_checkpoint_model(args.checkpoint)
        history = []
        offline = []
        records = []
        if args.target_success_hdf5 and "set_success_anchors" in globals():
            set_success_anchors(args.target_success_hdf5)
        target_contexts = load_failed_contexts(
            args.target_failed_hdf5,
            args.target_max_failed_demos if args.target_max_failed_demos > 0 else None,
        )
        exclude_success = load_success_demo_keys(args.target_exclude_success_jsonl)
        target_contexts = [ctx for ctx in target_contexts if ctx["demo_key"] not in exclude_success]
        plan_records = [
            {
                "demo_key": ctx["demo_key"],
                "context_metrics": ctx["context_metrics"],
                "success": False,
            }
            for ctx in target_contexts
        ]
    else:
        if not args.feedback_jsonl:
            parser.error("--feedback-jsonl is required unless --checkpoint is provided")
        records = load_records(args.feedback_jsonl)
        if not records:
            raise RuntimeError("No usable feedback records found")

        model, input_dim, history = train_model(
            records,
            epochs=args.epochs,
            lr=args.lr,
            use_component_loss=args.use_component_loss or args.true_pinn or args.standard_pinn,
            component_weight=args.component_weight,
            use_true_pinn=args.true_pinn,
            pinn_weight=args.pinn_weight,
            use_standard_pinn=args.standard_pinn,
            standard_pinn_weight=args.standard_pinn_weight,
        )
        offline = offline_selector_report(records, model, budget=args.budget, boundary_weight=args.boundary_weight)
        plan_records = records
        if args.target_success_hdf5 and "set_success_anchors" in globals():
            set_success_anchors(args.target_success_hdf5)
        if args.target_failed_hdf5:
            target_contexts = load_failed_contexts(
                args.target_failed_hdf5,
                args.target_max_failed_demos if args.target_max_failed_demos > 0 else None,
            )
            exclude_success = load_success_demo_keys(args.target_exclude_success_jsonl)
            target_contexts = [ctx for ctx in target_contexts if ctx["demo_key"] not in exclude_success]
            plan_records = [
                {
                    "demo_key": ctx["demo_key"],
                    "context_metrics": ctx["context_metrics"],
                    "success": False,
                }
                for ctx in target_contexts
            ]
    plan_info = build_candidate_plan(
        records=plan_records,
        model=model,
        out_path=out_dir / "pinn_utility_boundary_union_candidate_plan.jsonl",
        pool_size=args.pool_size,
        budget=args.budget,
        seed=args.seed,
        start_index=args.start_index,
        boundary_weight=args.boundary_weight,
        include_repaired=args.include_repaired,
        candidate_mode=args.candidate_mode,
    )

    if args.checkpoint:
        checkpoint_summary = {
            "mode": "checkpoint_inference",
            "checkpoint": args.checkpoint,
            "runtime_device": getattr(model, "runtime_device", "unknown"),
            "target_failed_hdf5": args.target_failed_hdf5,
            "target_success_hdf5": args.target_success_hdf5,
            "candidate_plan": plan_info["candidate_plan"],
            "target_demo_count": plan_info["target_demo_count"],
            "pool_size": args.pool_size,
            "budget": args.budget,
            "candidate_mode": args.candidate_mode,
        }
        with (out_dir / "checkpoint_inference_summary.json").open("w", encoding="utf-8") as f:
            json.dump(checkpoint_summary, f, indent=2)
        print(json.dumps(checkpoint_summary, indent=2))
        return

    import torch

    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": input_dim,
            "context_keys": CONTEXT_KEYS,
            "component_keys": COMPONENT_KEYS if (args.use_component_loss or args.true_pinn or args.standard_pinn) else [],
            "continuous_theta_keys": THETA_CONT_KEYS,
            "discrete_theta_keys": THETA_DISC_KEYS,
            "pde_applied_to": "theta_cont_only",
            "model_family": "repair_parameter_residual_field_pinn" if (args.use_component_loss or args.true_pinn or args.standard_pinn) else "mlp_selector",
            "residual_field_definition": {
                "context": "failed-trajectory context c",
                "discrete_repair_mode": "MimicGen discrete repair choices d",
                "continuous_repair_parameters": "continuous repair vector z",
                "value_field": "V(c,d,z): normalized residual energy / cost-to-success field",
                "physics_source": "q(c,d,z): weighted physics residual source from component residuals",
                "success_probability": "p(c,d,z)=sigmoid(beta*(tau-V)+small_correction)",
            },
            "use_component_loss": args.use_component_loss or args.true_pinn or args.standard_pinn,
            "true_pinn": args.true_pinn,
            "standard_pinn": args.standard_pinn,
        },
        out_dir / "stack_three_failed_conditioned_pinn.pt",
    )
    summary = {
        "method": (
            "standard_pinn_hjb_repair_field"
            if args.standard_pinn
            else (
                "true_pinn_utility_boundary_union"
                if args.true_pinn
                else ("component_pinn_utility_boundary_union" if args.use_component_loss else "pinn_utility_boundary_union")
            )
        ),
        "true_pinn": args.true_pinn,
        "standard_pinn": args.standard_pinn,
        "use_component_loss": args.use_component_loss or args.true_pinn or args.standard_pinn,
        "component_keys": COMPONENT_KEYS if (args.use_component_loss or args.true_pinn or args.standard_pinn) else [],
        "component_weight": args.component_weight,
        "pinn_weight": args.pinn_weight,
        "standard_pinn_weight": args.standard_pinn_weight,
        "model_family": "repair_parameter_residual_field_pinn" if (args.use_component_loss or args.true_pinn or args.standard_pinn) else "mlp_selector",
        "paper_method_name": "RP-RF / Repair-Parameter Residual Field PINN",
        "residual_field_definition": {
            "c": "failed-trajectory context",
            "d": "discrete MimicGen repair mode / conditional embedding",
            "z": "continuous repair parameters, the only variables used for PDE derivatives",
            "V(c,d,z)": "residual energy / cost-to-success value field",
            "q(c,d,z)": "weighted physics residual source from geometric, transport, lift, contact, slip, dynamics, and coupling components",
            "p(c,d,z)": "success probability induced mainly by V through sigmoid(beta*(tau-V)) with a small regularized correction",
        },
        "continuous_theta_keys": THETA_CONT_KEYS,
        "discrete_theta_keys": THETA_DISC_KEYS,
        "pde_applied_to": "theta_cont_only",
        "backward_compatible_outputs": True,
        "governing_equations": (
            [
                "V(c,d,z): residual cost-to-success field over continuous repair parameters",
                "q(c,d,z)=weighted_sum(component_residuals)",
                "p(c,d,z)=sigmoid(beta*(tau - V(c,d,z)) + epsilon(c,d,z))",
                "0.5 * ||grad_z V||^2 - q(c,d,z) = 0",
                "grad_z p + beta * p * (1-p) * grad_z V = 0",
            ]
            if args.standard_pinn
            else []
        ),
        "loss_terms": [
            "supervised_energy",
            "success_bce",
            "energy_success_consistency",
            "energy_margin",
            "regularized_success_correction",
            "component_residual_supervision" if (args.use_component_loss or args.true_pinn or args.standard_pinn) else "disabled_component_residual_supervision",
            "energy_component_consistency" if (args.true_pinn or args.standard_pinn) else "disabled_energy_component_consistency",
            "physical_component_relations" if (args.true_pinn or args.standard_pinn) else "disabled_physical_component_relations",
            "boundary_conditions" if (args.true_pinn or args.standard_pinn) else "disabled_boundary_conditions",
            "theta_cont_differential_consistency" if (args.true_pinn or args.standard_pinn) else "disabled_theta_cont_differential_consistency",
            "theta_cont_hjb_eikonal_collocation" if args.standard_pinn else "disabled_theta_cont_hjb_eikonal_collocation",
            "theta_cont_success_transport_collocation" if args.standard_pinn else "disabled_theta_cont_success_transport_collocation",
        ],
        "num_feedback_records": len(records),
        "num_feedback_success": int(sum(1 for r in records if r["success"])),
        "num_feedback_demos": len({r["demo_key"] for r in records}),
        "offline_budget": args.budget,
        "offline_oracle_demo_success": int(sum(1 for r in offline if r["oracle_success"])),
        "offline_selector_demo_success": int(sum(1 for r in offline if r["selector_success"])),
        "offline_per_demo": offline,
        "target_failed_hdf5": args.target_failed_hdf5,
        "loss_history": history,
        **plan_info,
    }
    (out_dir / "pinn_utility_boundary_union_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
