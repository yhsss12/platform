#!/usr/bin/env python3
"""
Universal task-conditioned RP-RF PINN selector for MimicGen repair feedback.

This script trains one shared repair-parameter residual-field PINN across
multiple tasks. Task-specific code is only used to load failed contexts and to
generate task-appropriate candidate repair parameters; the model weights,
V/q/p output semantics, losses, and selector are shared.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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

THETA_DISC_KEYS = [
    "select_src_per_subtask",
    "transform_first_robot_pose",
    "interpolate_from_last_target_pose",
    "selection_strategy_nearest_neighbor_object",
    "selection_strategy_random",
]

THETA_CONT_INDEPENDENT_KEYS = [
    "action_noise",
    "num_interpolation_steps",
    "num_fixed_steps",
    "offset_lo",
    "offset_hi",
    "nn_k",
]

THETA_CONT_DERIVED_KEYS = [
    "offset_width",
    "offset_center",
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


def theta_cont_abs_index(key: str) -> int:
    return THETA_CONT_START + THETA_CONT_KEYS.index(key)


INDEPENDENT_Z_INDICES = [theta_cont_abs_index(k) for k in THETA_CONT_INDEPENDENT_KEYS]
DERIVED_Z_INDICES = [theta_cont_abs_index(k) for k in THETA_CONT_DERIVED_KEYS]


def parse_task_feedback(items: list[str]) -> list[tuple[str, Path]]:
    pairs: list[tuple[str, Path]] = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected TASK=PATH for --task-feedback, got: {item}")
        task, path = item.split("=", 1)
        task = task.strip()
        if not task:
            raise ValueError(f"Empty task name in --task-feedback {item}")
        pairs.append((task, Path(path)))
    return pairs


def load_records(path: str | Path, task: str) -> list[dict[str, Any]]:
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
            row["task"] = task
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
    return component_targets_from_metrics(metrics, problematic)


def component_energy_target(components: np.ndarray) -> np.ndarray:
    weighted = components * COMPONENT_WEIGHTS[None, :]
    return (weighted.sum(axis=1) / COMPONENT_WEIGHTS.sum()).astype(np.float32)


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
    ctx = np.array([float(context.get(k, 0.0)) / ctx_scales[k] for k in CONTEXT_KEYS], dtype=np.float32)
    theta_feat, boundary_bonus = theta_to_features(theta)
    return np.concatenate([ctx, theta_feat, np.array([boundary_bonus], dtype=np.float32)])


def project_derived_theta_features(x_tensor: torch.Tensor) -> torch.Tensor:
    lo_idx = theta_cont_abs_index("offset_lo")
    hi_idx = theta_cont_abs_index("offset_hi")
    width_idx = theta_cont_abs_index("offset_width")
    center_idx = theta_cont_abs_index("offset_center")
    lo_raw = x_tensor[:, lo_idx : lo_idx + 1]
    hi_raw = x_tensor[:, hi_idx : hi_idx + 1]
    lo = torch.minimum(lo_raw, hi_raw).clamp(0.0, 1.0)
    hi = torch.maximum(lo_raw, hi_raw).clamp(0.0, 1.0)
    projected = x_tensor.clone()
    projected[:, lo_idx : lo_idx + 1] = lo
    projected[:, hi_idx : hi_idx + 1] = hi
    projected[:, width_idx : width_idx + 1] = torch.clamp(hi - lo, 0.0, 1.0)
    projected[:, center_idx : center_idx + 1] = torch.clamp(0.5 * (hi + lo), 0.0, 1.0)
    return projected


def weighted_mean(loss: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    while weights.ndim < loss.ndim:
        weights = weights.unsqueeze(-1)
    return (loss * weights).sum() / weights.sum().clamp_min(1e-6)


class FiLMResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.cond = nn.Linear(cond_dim, hidden_dim * 2)

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.cond(cond).chunk(2, dim=-1)
        y = self.fc(F.silu(self.norm(h)))
        y = y * (1.0 + 0.15 * torch.tanh(gamma)) + 0.15 * torch.tanh(beta)
        return h + y


class UniversalRepairParameterPINN(nn.Module):
    def __init__(
        self,
        num_tasks: int,
        input_dim: int,
        task_emb_dim: int = 32,
        hidden_dim: int = 256,
        num_blocks: int = 5,
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.input_dim = input_dim
        self.task_emb_dim = task_emb_dim
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.context_dim = len(CONTEXT_KEYS)
        self.theta_disc_dim = len(THETA_DISC_KEYS)
        self.theta_cont_dim = len(THETA_CONT_KEYS)
        self.boundary_dim = 1
        self.model_family = "universal_task_conditioned_repair_parameter_residual_field_pinn"

        self.task_embedding = nn.Embedding(num_tasks, task_emb_dim)
        self.context_encoder = nn.Sequential(
            nn.Linear(self.context_dim, 96),
            nn.SiLU(),
            nn.LayerNorm(96),
            nn.Linear(96, 96),
            nn.SiLU(),
        )
        self.discrete_encoder = nn.Sequential(
            nn.Linear(self.theta_disc_dim, 48),
            nn.SiLU(),
            nn.LayerNorm(48),
            nn.Linear(48, 48),
            nn.SiLU(),
        )
        self.cont_encoder = nn.Sequential(
            nn.Linear(self.theta_cont_dim, 64),
            nn.SiLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 64),
            nn.SiLU(),
        )
        self.boundary_encoder = nn.Sequential(nn.Linear(self.boundary_dim, 16), nn.SiLU())
        fused_dim = 96 + 48 + 64 + 16 + task_emb_dim
        self.input_proj = nn.Sequential(nn.Linear(fused_dim, hidden_dim), nn.SiLU(), nn.LayerNorm(hidden_dim))
        cond_dim = task_emb_dim + 48
        self.blocks = nn.ModuleList([FiLMResidualBlock(hidden_dim, cond_dim) for _ in range(num_blocks)])
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.head_value = nn.Linear(hidden_dim, 1)
        self.head_q = nn.Linear(hidden_dim, 1)
        self.head_success_correction = nn.Linear(hidden_dim, 1)
        self.head_components = nn.Linear(hidden_dim, len(COMPONENT_KEYS))

    def split(self, inp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        context = inp[:, :THETA_DISC_START]
        theta_disc = inp[:, THETA_DISC_START:THETA_DISC_END]
        theta_cont = inp[:, THETA_CONT_START:THETA_CONT_END]
        boundary = inp[:, BOUNDARY_BONUS_INDEX : BOUNDARY_BONUS_INDEX + 1]
        return context, theta_disc, theta_cont, boundary

    def forward(self, inp: torch.Tensor, task_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        context, theta_disc, theta_cont, boundary = self.split(inp)
        task_emb = self.task_embedding(task_ids.long())
        context_emb = self.context_encoder(context)
        disc_emb = self.discrete_encoder(theta_disc)
        cont_emb = self.cont_encoder(theta_cont)
        boundary_emb = self.boundary_encoder(boundary)
        h = self.input_proj(torch.cat([context_emb, disc_emb, cont_emb, boundary_emb, task_emb], dim=1))
        cond = torch.cat([task_emb, disc_emb], dim=1)
        for block in self.blocks:
            h = block(h, cond)
        h = F.silu(self.final_norm(h))

        value = torch.sigmoid(self.head_value(h))
        q_source = torch.sigmoid(self.head_q(h))
        correction = PINN_SUCCESS_CORRECTION_SCALE * torch.tanh(self.head_success_correction(h))
        success_logit = PINN_BETA * (PINN_SUCCESS_ENERGY_THRESHOLD - value) + correction
        components = torch.sigmoid(self.head_components(h))
        return {
            "energy_success": torch.cat([value, success_logit], dim=1),
            "residual_value": value,
            "physics_source": q_source,
            "success_correction": correction,
            "components": components,
        }


def make_training_arrays(records: list[dict[str, Any]], task_to_id: dict[str, int]):
    x = np.stack([feature_vector(r["context_metrics"], r["theta"]) for r in records]).astype(np.float32)
    task_ids = np.array([task_to_id[r["task"]] for r in records], dtype=np.int64)
    energy = np.array([float(r["metrics"]["energy"]) for r in records], dtype=np.float32)
    e_target = np.clip(energy, 0.0, 30.0) / 30.0
    success = np.array([float(r["success"]) for r in records], dtype=np.float32)
    component_target = np.stack(
        [physics_residual_targets_from_metrics(r.get("metrics"), bool(r.get("problematic"))) for r in records]
    ).astype(np.float32)

    task_counts: dict[str, int] = {}
    for r in records:
        task_counts[r["task"]] = task_counts.get(r["task"], 0) + 1
    task_weights = np.array(
        [len(records) / (len(task_counts) * max(task_counts[r["task"]], 1)) for r in records],
        dtype=np.float32,
    )
    return x, task_ids, e_target, success, component_target, task_weights, task_counts


def train_model(
    records: list[dict[str, Any]],
    task_to_id: dict[str, int],
    epochs: int,
    lr: float,
    component_weight: float,
    pinn_weight: float,
    standard_pinn_weight: float,
    hidden_dim: int,
    num_blocks: int,
    task_emb_dim: int,
    pde_collocation_noise: float,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    x, task_ids, e_target, success, component_target, task_weights, task_counts = make_training_arrays(records, task_to_id)
    xt = torch.from_numpy(x).to(device)
    task_t = torch.from_numpy(task_ids).to(device)
    et = torch.from_numpy(e_target).unsqueeze(1).to(device)
    st = torch.from_numpy(success).unsqueeze(1).to(device)
    ct = torch.from_numpy(component_target).to(device)
    wt = torch.from_numpy(task_weights).unsqueeze(1).to(device)
    comp_weights = torch.from_numpy(COMPONENT_WEIGHTS).float().to(device)
    comp_energy_target = torch.from_numpy(component_energy_target(component_target)).unsqueeze(1).to(device)

    model = UniversalRepairParameterPINN(
        num_tasks=len(task_to_id),
        input_dim=x.shape[1],
        task_emb_dim=task_emb_dim,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
    ).to(device)
    model.runtime_device = str(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    pos = float(st.sum().item())
    pos_weight = torch.tensor([(len(st) - pos) / max(pos, 1.0)], dtype=torch.float32, device=device)

    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        xt_epoch = xt.detach().clone().requires_grad_(True)
        raw_pred = model(xt_epoch, task_t)
        pred = raw_pred["energy_success"]
        pred_components = raw_pred["components"]
        q_source = raw_pred["physics_source"]
        success_correction = raw_pred["success_correction"]
        e_pred = pred[:, :1]
        s_logit = pred[:, 1:2]

        loss_e = weighted_mean(F.smooth_l1_loss(e_pred, et, reduction="none"), wt)
        loss_s_raw = F.binary_cross_entropy_with_logits(s_logit, st, pos_weight=pos_weight, reduction="none")
        loss_s = weighted_mean(loss_s_raw, wt)
        loss_success_correction = weighted_mean((success_correction / PINN_SUCCESS_CORRECTION_SCALE).pow(2), wt)

        s_from_e = torch.sigmoid((PINN_SUCCESS_ENERGY_THRESHOLD - e_pred) * PINN_BETA)
        loss_cons = weighted_mean(F.binary_cross_entropy(s_from_e, st, reduction="none"), wt)
        success_margin = torch.relu(e_pred - PINN_SUCCESS_ENERGY_THRESHOLD) * st
        fail_margin = torch.relu(PINN_FAILURE_ENERGY_FLOOR - e_pred) * (1.0 - st)
        loss_margin = weighted_mean(success_margin + fail_margin, wt)

        loss_components = weighted_mean(F.smooth_l1_loss(pred_components, ct, reduction="none").mean(dim=1, keepdim=True), wt)
        comp_e = (pred_components * comp_weights[None, :]).sum(dim=1, keepdim=True) / comp_weights.sum()
        loss_component_total = weighted_mean(F.smooth_l1_loss(comp_e, comp_energy_target, reduction="none"), wt)
        loss_q_supervised = weighted_mean(F.smooth_l1_loss(q_source, comp_energy_target, reduction="none"), wt)
        loss_component_success = weighted_mean(
            F.binary_cross_entropy(torch.sigmoid((PINN_SUCCESS_ENERGY_THRESHOLD - comp_e) * PINN_BETA), st, reduction="none"),
            wt,
        )
        loss_total_consistency = weighted_mean(F.smooth_l1_loss(e_pred, q_source, reduction="none"), wt)

        e_xy, e_transport, e_lift, e_contact, e_bilateral, e_dynamics, e_slip, e_coupling = [
            pred_components[:, i : i + 1] for i in range(len(COMPONENT_KEYS))
        ]
        contact_target = 0.5 * (e_xy + e_lift)
        bilateral_target = torch.abs(e_transport - e_lift)
        dynamics_target = 0.5 * (e_slip + e_bilateral)
        slip_floor = 0.5 * e_lift
        coupling_floor = torch.maximum(e_contact, 0.5 * (e_xy + e_lift))
        loss_physics_residual = (
            weighted_mean(F.smooth_l1_loss(e_contact, contact_target, reduction="none"), wt)
            + weighted_mean(F.smooth_l1_loss(e_bilateral, bilateral_target, reduction="none"), wt)
            + weighted_mean(F.smooth_l1_loss(e_dynamics, dynamics_target, reduction="none"), wt)
            + weighted_mean(torch.relu(slip_floor - e_slip).pow(2), wt)
            + weighted_mean(torch.relu(coupling_floor - e_coupling).pow(2), wt)
        ) / 5.0

        comp_mean = pred_components.mean(dim=1, keepdim=True)
        success_boundary = st * (e_pred.pow(2) + pred_components.pow(2).mean(dim=1, keepdim=True))
        failure_boundary = (1.0 - st) * (
            torch.relu(0.08 - comp_mean).pow(2) + torch.relu(PINN_FAILURE_ENERGY_FLOOR - e_pred).pow(2)
        )
        loss_boundary = weighted_mean(success_boundary + failure_boundary, wt)

        p_success = torch.sigmoid(s_logit)
        grad_e = torch.autograd.grad(e_pred.sum(), xt_epoch, create_graph=True, retain_graph=True)[0][:, INDEPENDENT_Z_INDICES]
        grad_p = torch.autograd.grad(p_success.sum(), xt_epoch, create_graph=True, retain_graph=True)[0][
            :, INDEPENDENT_Z_INDICES
        ]
        differential_residual = grad_p + PINN_BETA * p_success * (1.0 - p_success) * grad_e
        loss_differential = weighted_mean(differential_residual.pow(2).mean(dim=1, keepdim=True), wt)

        x_col = project_derived_theta_features(xt.detach().clone())
        z_noise = pde_collocation_noise * torch.randn_like(x_col[:, INDEPENDENT_Z_INDICES])
        x_col[:, INDEPENDENT_Z_INDICES] = torch.clamp(x_col[:, INDEPENDENT_Z_INDICES] + z_noise, 0.0, 1.0)
        x_col = project_derived_theta_features(x_col).requires_grad_(True)
        raw_col = model(x_col, task_t)
        col_pred = raw_col["energy_success"]
        col_v = col_pred[:, :1]
        col_p = torch.sigmoid(col_pred[:, 1:2])
        col_q = raw_col["physics_source"]
        col_grad_v = torch.autograd.grad(col_v.sum(), x_col, create_graph=True, retain_graph=True)[0][
            :, INDEPENDENT_Z_INDICES
        ]
        col_grad_p = torch.autograd.grad(col_p.sum(), x_col, create_graph=True, retain_graph=True)[0][
            :, INDEPENDENT_Z_INDICES
        ]
        hjb_residual = 0.5 * col_grad_v.pow(2).sum(dim=1, keepdim=True) - col_q
        loss_hjb_pde = weighted_mean(hjb_residual.pow(2), wt)
        col_transport_residual = col_grad_p + PINN_BETA * col_p * (1.0 - col_p) * col_grad_v
        loss_collocation_transport_pde = weighted_mean(col_transport_residual.pow(2).mean(dim=1, keepdim=True), wt)

        loss = (
            loss_e
            + 0.45 * loss_s
            + 0.30 * loss_cons
            + 0.20 * loss_margin
            + 0.05 * loss_success_correction
            + component_weight * loss_components
            + 0.25 * component_weight * loss_component_total
            + 0.20 * component_weight * loss_component_success
            + 0.35 * component_weight * loss_q_supervised
            + pinn_weight * loss_total_consistency
            + pinn_weight * loss_physics_residual
            + 0.50 * pinn_weight * loss_boundary
            + 0.25 * pinn_weight * loss_differential
            + standard_pinn_weight * loss_hjb_pde
            + standard_pinn_weight * loss_collocation_transport_pde
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0:
            history.append(
                {
                    "epoch": float(epoch + 1),
                    "loss": float(loss.item()),
                    "loss_energy": float(loss_e.item()),
                    "loss_success": float(loss_s.item()),
                    "loss_v_to_p_consistency": float(loss_cons.item()),
                    "loss_margin": float(loss_margin.item()),
                    "loss_success_correction_regularizer": float(loss_success_correction.item()),
                    "loss_components": float(loss_components.item()),
                    "loss_component_total": float(loss_component_total.item()),
                    "loss_q_supervised": float(loss_q_supervised.item()),
                    "loss_component_success": float(loss_component_success.item()),
                    "loss_total_consistency_V_q": float(loss_total_consistency.item()),
                    "loss_physics_residual": float(loss_physics_residual.item()),
                    "loss_boundary": float(loss_boundary.item()),
                    "loss_differential_theta_cont": float(loss_differential.item()),
                    "loss_hjb_pde_theta_cont": float(loss_hjb_pde.item()),
                    "loss_collocation_transport_pde_theta_cont": float(loss_collocation_transport_pde.item()),
                }
            )
    return model, x.shape[1], history, task_counts


@torch.no_grad()
def predict(
    model: UniversalRepairParameterPINN,
    task: str,
    task_to_id: dict[str, int],
    contexts: list[dict[str, float]],
    thetas: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    device = next(model.parameters()).device
    x = np.stack([feature_vector(c, t) for c, t in zip(contexts, thetas)]).astype(np.float32)
    xt = torch.from_numpy(x).to(device)
    task_id = torch.full((len(x),), int(task_to_id[task]), dtype=torch.long, device=device)
    raw = model(xt, task_id)
    pred = raw["energy_success"]
    e = pred[:, 0].detach().cpu().numpy()
    p = torch.sigmoid(pred[:, 1]).detach().cpu().numpy()
    return e, p


def unique_union(candidates: list[dict[str, Any]], budget: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for score_name in ["utility_score", "boundary_score"]:
        for row in sorted(candidates, key=lambda r: (r[score_name], r["candidate_index"])):
            key = json.dumps(row["theta"], sort_keys=True)
            if key in seen:
                continue
            selected.append(row)
            seen.add(key)
            if len(selected) >= budget:
                return selected
    return selected


def offline_selector_report(
    records: list[dict[str, Any]],
    model: UniversalRepairParameterPINN,
    task_to_id: dict[str, int],
    budget: int,
    boundary_weight: float,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for rec in records:
        grouped.setdefault((rec["task"], rec["demo_key"]), []).append(rec)

    report: list[dict[str, Any]] = []
    for (task, demo_key), rows in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        context = rows[0]["context_metrics"]
        pred_e, pred_p = predict(model, task, task_to_id, [context] * len(rows), [r["theta"] for r in rows])
        scored: list[dict[str, Any]] = []
        for local_idx, (rec, pe, pp) in enumerate(zip(rows, pred_e, pred_p)):
            _, boundary_bonus = theta_to_features(rec["theta"])
            scored.append(
                {
                    "candidate_index": int(rec.get("candidate_index", local_idx)),
                    "theta": rec["theta"],
                    "success": bool(rec["success"]),
                    "utility_score": float(pe - 8.0 * pp),
                    "boundary_score": float(pe - 6.0 * pp - boundary_weight * boundary_bonus),
                }
            )
        selected = unique_union(scored, budget)
        report.append(
            {
                "task": task,
                "demo_key": demo_key,
                "oracle_success": any(r["success"] for r in rows),
                "selector_success": any(r["success"] for r in selected),
                "num_candidates": len(rows),
            }
        )
    return report


def import_task_modules(task: str):
    raise RuntimeError(
        "Task adapters are bundled in universal_rprf_inference.py. "
        "Use that entry point for rollout evaluation."
    )


def build_candidate_plan(
    task: str,
    task_to_id: dict[str, int],
    records: list[dict[str, Any]],
    model: UniversalRepairParameterPINN,
    out_path: Path,
    pool_size: int,
    budget: int,
    seed: int,
    start_index: int,
    boundary_weight: float,
    include_repaired: bool,
    candidate_mode: str,
    demo_sort_key,
    make_candidate,
    make_safe_candidate,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        grouped.setdefault(rec["demo_key"], []).append(rec)

    plan_rows: list[dict[str, Any]] = []
    for demo_key in sorted(grouped, key=demo_sort_key):
        already_repaired = any(bool(r.get("success")) for r in grouped[demo_key])
        if already_repaired and not include_repaired:
            continue
        context = grouped[demo_key][0]["context_metrics"]
        rng = np.random.default_rng(seed + demo_sort_key(demo_key) * 1009)
        pool: list[dict[str, Any]] = []
        for i in range(pool_size):
            candidate_index = start_index + i
            theta = make_safe_candidate(candidate_index, rng) if candidate_mode == "safe" else make_candidate(candidate_index, rng)
            pool.append({"candidate_index": candidate_index, "theta": theta})
        pred_e, pred_p = predict(model, task, task_to_id, [context] * len(pool), [p["theta"] for p in pool])
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
        "target_task": task,
        "target_demo_count": len(plan_rows),
        "pool_size": pool_size,
        "budget": budget,
        "include_repaired": include_repaired,
        "candidate_mode": candidate_mode,
    }


def save_checkpoint(
    path: Path,
    model: UniversalRepairParameterPINN,
    input_dim: int,
    task_to_id: dict[str, int],
    args: argparse.Namespace,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "input_dim": input_dim,
            "task_to_id": task_to_id,
            "context_keys": CONTEXT_KEYS,
            "component_keys": COMPONENT_KEYS,
            "continuous_theta_keys": THETA_CONT_KEYS,
            "independent_z_keys": THETA_CONT_INDEPENDENT_KEYS,
            "derived_z_keys": THETA_CONT_DERIVED_KEYS,
            "discrete_theta_keys": THETA_DISC_KEYS,
            "trained_device": getattr(model, "runtime_device", "unknown"),
            "model_family": "universal_task_conditioned_repair_parameter_residual_field_pinn",
            "task_emb_dim": args.task_emb_dim,
            "hidden_dim": args.hidden_dim,
            "num_blocks": args.num_blocks,
            "v_q_p_outputs": True,
            "p_induced_by_v": True,
        },
        path,
    )


def load_checkpoint(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    model = UniversalRepairParameterPINN(
        num_tasks=len(ckpt["task_to_id"]),
        input_dim=int(ckpt["input_dim"]),
        task_emb_dim=int(ckpt.get("task_emb_dim", 32)),
        hidden_dim=int(ckpt.get("hidden_dim", 256)),
        num_blocks=int(ckpt.get("num_blocks", 5)),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.runtime_device = str(device)
    model.eval()
    return model, int(ckpt["input_dim"]), dict(ckpt["task_to_id"]), ckpt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-feedback", action="append", default=[], help="Repeated TASK=feedback_candidates.jsonl")
    parser.add_argument("--checkpoint", default=None, help="Load an existing universal checkpoint and only build a plan.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=450)
    parser.add_argument("--lr", type=float, default=1.5e-3)
    parser.add_argument("--budget", type=int, default=5)
    parser.add_argument("--pool-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=9701)
    parser.add_argument("--start-index", type=int, default=100000)
    parser.add_argument("--boundary-weight", type=float, default=1.5)
    parser.add_argument("--candidate-mode", choices=["default", "safe"], default="safe")
    parser.add_argument("--include-repaired", action="store_true")
    parser.add_argument("--target-task", default=None)
    parser.add_argument("--target-failed-hdf5", default=None)
    parser.add_argument("--target-success-hdf5", default=None)
    parser.add_argument("--target-max-failed-demos", type=int, default=0)
    parser.add_argument("--target-exclude-success-jsonl", action="append", default=None)
    parser.add_argument("--component-weight", type=float, default=0.45)
    parser.add_argument("--pinn-weight", type=float, default=0.35)
    parser.add_argument("--standard-pinn-weight", type=float, default=0.35)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-blocks", type=int, default=5)
    parser.add_argument("--task-emb-dim", type=int, default=32)
    parser.add_argument("--pde-collocation-noise", type=float, default=0.035)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    records: list[dict[str, Any]] = []
    task_counts: dict[str, int] = {}
    history: list[dict[str, float]] = []
    if args.checkpoint:
        model, input_dim, task_to_id, ckpt = load_checkpoint(Path(args.checkpoint), device)
        trained_device = ckpt.get("trained_device", "unknown")
    else:
        pairs = parse_task_feedback(args.task_feedback)
        if not pairs:
            raise RuntimeError("Provide at least one --task-feedback TASK=PATH or --checkpoint")
        for task, path in pairs:
            task_rows = load_records(path, task)
            if not task_rows:
                print(f"[warn] no usable records for {task}: {path}", file=sys.stderr)
                continue
            records.extend(task_rows)
        if not records:
            raise RuntimeError("No usable feedback records found")
        task_names = sorted({r["task"] for r in records})
        task_to_id = {task: i for i, task in enumerate(task_names)}
        model, input_dim, history, task_counts = train_model(
            records=records,
            task_to_id=task_to_id,
            epochs=args.epochs,
            lr=args.lr,
            component_weight=args.component_weight,
            pinn_weight=args.pinn_weight,
            standard_pinn_weight=args.standard_pinn_weight,
            hidden_dim=args.hidden_dim,
            num_blocks=args.num_blocks,
            task_emb_dim=args.task_emb_dim,
            pde_collocation_noise=args.pde_collocation_noise,
        )
        trained_device = getattr(model, "runtime_device", "unknown")
        save_checkpoint(out_dir / "universal_rprf_pinn.pt", model, input_dim, task_to_id, args)

    offline: list[dict[str, Any]] = []
    if records:
        offline = offline_selector_report(records, model, task_to_id, budget=args.budget, boundary_weight=args.boundary_weight)

    plan_info: dict[str, Any] | None = None
    if args.target_task and args.target_failed_hdf5:
        if args.target_task not in task_to_id:
            raise RuntimeError(f"Target task {args.target_task} is not in trained task_to_id: {sorted(task_to_id)}")
        repair_module, selector_module = import_task_modules(args.target_task)
        if args.target_success_hdf5 and hasattr(repair_module, "set_success_anchors"):
            repair_module.set_success_anchors(args.target_success_hdf5)
        target_contexts = repair_module.load_failed_contexts(
            args.target_failed_hdf5,
            args.target_max_failed_demos if args.target_max_failed_demos > 0 else None,
        )
        exclude_success = load_success_demo_keys(args.target_exclude_success_jsonl)
        target_contexts = [ctx for ctx in target_contexts if ctx["demo_key"] not in exclude_success]
        plan_records = [
            {"task": args.target_task, "demo_key": ctx["demo_key"], "context_metrics": ctx["context_metrics"], "success": False}
            for ctx in target_contexts
        ]
        plan_info = build_candidate_plan(
            task=args.target_task,
            task_to_id=task_to_id,
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
            demo_sort_key=repair_module.demo_sort_key,
            make_candidate=repair_module.make_candidate,
            make_safe_candidate=selector_module.make_safe_candidate,
        )

    task_offline: dict[str, dict[str, int]] = {}
    for row in offline:
        item = task_offline.setdefault(row["task"], {"demos": 0, "oracle_success": 0, "selector_success": 0})
        item["demos"] += 1
        item["oracle_success"] += int(bool(row["oracle_success"]))
        item["selector_success"] += int(bool(row["selector_success"]))

    summary = {
        "method": "universal_task_conditioned_rp_rf_pinn",
        "paper_method_name": "Universal Task-Conditioned RP-RF PINN",
        "model_family": "universal_task_conditioned_repair_parameter_residual_field_pinn",
        "trained_device": trained_device,
        "runtime_device": str(device),
        "task_to_id": task_to_id,
        "num_tasks": len(task_to_id),
        "task_counts": task_counts,
        "num_feedback_records": len(records),
        "num_feedback_success": int(sum(1 for r in records if r.get("success"))) if records else None,
        "offline_by_task": task_offline,
        "offline_oracle_demo_success": int(sum(1 for r in offline if r["oracle_success"])) if offline else None,
        "offline_selector_demo_success": int(sum(1 for r in offline if r["selector_success"])) if offline else None,
        "offline_demo_count": len(offline),
        "plan_info": plan_info,
        "history": history,
        "v_q_p_outputs": {
            "V": "normalized residual value / cost-to-success field",
            "q": "task-conditioned physics residual source",
            "p": "success probability induced mainly by V plus a small regularized correction",
        },
        "pde_applied_to": "independent_z_only",
        "independent_z_keys": THETA_CONT_INDEPENDENT_KEYS,
        "derived_z_keys": THETA_CONT_DERIVED_KEYS,
        "continuous_theta_keys": THETA_CONT_KEYS,
        "discrete_theta_keys": THETA_DISC_KEYS,
        "selector": {
            "candidate_mode": args.candidate_mode,
            "budget": args.budget,
            "pool_size": args.pool_size,
            "boundary_weight": args.boundary_weight,
        },
        "architecture": {
            "task_embedding": args.task_emb_dim,
            "hidden_dim": args.hidden_dim,
            "num_film_residual_blocks": args.num_blocks,
            "conditioning": "task_embedding + discrete_repair_embedding via FiLM residual blocks",
        },
    }
    with (out_dir / "universal_rprf_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=True, indent=2)
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
