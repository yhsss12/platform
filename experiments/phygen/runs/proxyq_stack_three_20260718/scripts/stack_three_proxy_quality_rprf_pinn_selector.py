#!/usr/bin/env python3
"""
Train the StackThree PhyGen main-method repair-parameter residual-field PINN
from feedback rollouts and emit a utility / boundary-union candidate plan for
the next true rollout.

This is the paper-facing main method version.  It intentionally removes the
many ablation/version switches from the runnable interface and fixes the core
method as:

    TaskAdapter -> PhyGen residual-field core -> V/q/p outputs
    -> independent-z PINN consistency -> utility/boundary top-k selection

Candidate generation, utility/boundary selection, and output plan format remain
compatible with the earlier StackThree selector to minimize performance drift.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np

from stack_three_failed_conditioned_mimicgen_repair import demo_sort_key, load_failed_contexts, make_candidate


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

# PhyGen-v1 repair-parameter schema.  The original selector concatenated all
# theta features and differentiated through the whole block.  For a
# paper-facing PINN formulation, we explicitly separate discrete generator
# choices d from relaxed-continuous repair parameters z and apply differential
# constraints only to z in the main method.
THETA_FEATURE_KEYS = [
    "select_src_per_subtask",
    "transform_first_robot_pose",
    "interpolate_from_last_target_pose",
    "strategy_nearest_neighbor_object",
    "strategy_random",
    "action_noise",
    "num_interpolation_steps",
    "num_fixed_steps",
    "offset_low",
    "offset_high",
    "offset_width",
    "offset_mid",
    "nn_k",
]
DISCRETE_REPAIR_KEYS = THETA_FEATURE_KEYS[:5]
# Only the independent relaxed-continuous repair variables form the default
# PINN manifold z.  offset_width and offset_mid remain useful input features
# for the selector, but they are derived from offset_low/high and are not
# differentiated through by default.  This avoids treating dependent feature
# coordinates as independent physical coordinates in the paper formulation.
INDEPENDENT_RELAXED_CONTINUOUS_REPAIR_KEYS = [
    "action_noise",
    "num_interpolation_steps",
    "num_fixed_steps",
    "offset_low",
    "offset_high",
    "nn_k",
]
DERIVED_REPAIR_FEATURE_KEYS = ["offset_width", "offset_mid"]
RELAXED_CONTINUOUS_REPAIR_KEYS = INDEPENDENT_RELAXED_CONTINUOUS_REPAIR_KEYS
INDEPENDENT_RELAXED_CONTINUOUS_THETA_REL_INDICES = [
    THETA_FEATURE_KEYS.index(k) for k in INDEPENDENT_RELAXED_CONTINUOUS_REPAIR_KEYS
]
DERIVED_THETA_REL_INDICES = [THETA_FEATURE_KEYS.index(k) for k in DERIVED_REPAIR_FEATURE_KEYS]
SUCCESS_RESIDUAL_THRESHOLD = 0.25
SUCCESS_RESIDUAL_BETA = 8.0

# Paper main-method configuration.  These are fixed on purpose: this script is
# meant to run the PhyGen main method rather than a menu of experimental
# variants.  This entry point always represents the method claimed in the
# paper.
MAIN_USE_COMPONENT_LOSS = True
MAIN_USE_TRUE_PINN = True
MAIN_USE_STANDARD_PINN = True
MAIN_SUCCESS_HEAD_MODE = "residual_induced"
MAIN_PINN_GRADIENT_DOMAIN = "continuous"
MAIN_ENABLE_ENERGY_LOSS = True
MAIN_ENABLE_CONSISTENCY_LOSS = True
MAIN_ENABLE_MARGIN_LOSS = True
MAIN_ENABLE_PHYSICS_RESIDUAL_LOSS = True
MAIN_ENABLE_BOUNDARY_LOSS = True
MAIN_ENABLE_DIFFERENTIAL_LOSS = True
MAIN_ENABLE_HJB_PDE_LOSS = True
MAIN_ENABLE_TRANSPORT_PDE_LOSS = True
MAIN_METHOD_NAME = "phygen_residual_field_pinn_main"


def theta_abs_index(feature_key: str) -> int:
    """Absolute input-feature index for a theta feature."""
    return len(CONTEXT_KEYS) + THETA_FEATURE_KEYS.index(feature_key)


def pinn_feature_indices(domain: str) -> list[int]:
    """Absolute input-feature indices used by PINN differential losses."""
    theta_start = len(CONTEXT_KEYS)
    if domain == "all":
        return list(range(theta_start, theta_start + len(THETA_FEATURE_KEYS)))
    if domain in {"continuous", "independent_z"}:
        return [theta_start + i for i in INDEPENDENT_RELAXED_CONTINUOUS_THETA_REL_INDICES]
    raise ValueError(f"Unsupported pinn gradient domain: {domain}")


def project_derived_theta_features(x_tensor: Any) -> Any:
    """Project derived theta features back to the feasible StackThree manifold.

    The feature vector stores offset_low, offset_high, offset_width, and
    offset_mid as normalized values.  Collocation perturbs independent z
    coordinates only, so width/mid must be recomputed before the model sees
    the relaxed point.  This function keeps the projection differentiable.
    """
    import torch

    low_idx = theta_abs_index("offset_low")
    high_idx = theta_abs_index("offset_high")
    width_idx = theta_abs_index("offset_width")
    mid_idx = theta_abs_index("offset_mid")
    lo_raw = x_tensor[:, low_idx : low_idx + 1]
    hi_raw = x_tensor[:, high_idx : high_idx + 1]
    lo = torch.minimum(lo_raw, hi_raw)
    hi = torch.maximum(lo_raw, hi_raw)
    x_projected = x_tensor.clone()
    x_projected[:, low_idx : low_idx + 1] = lo
    x_projected[:, high_idx : high_idx + 1] = hi
    x_projected[:, width_idx : width_idx + 1] = torch.clamp(hi - lo, 0.0, 1.0)
    x_projected[:, mid_idx : mid_idx + 1] = torch.clamp(0.5 * (hi + lo), 0.0, 1.0)
    return x_projected


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


@dataclass(frozen=True)
class TaskAdapterSpec:
    """Method-facing schema for a PhyGen task adapter.

    For v1.1 we keep a single StackThree implementation, but the schema makes
    explicit which parts should become task-specific when moving to MimicGen-12
    and SoftMimicGen adapters.
    """

    name: str
    context_keys: list[str]
    theta_feature_keys: list[str]
    discrete_repair_keys: list[str]
    independent_z_keys: list[str]
    derived_theta_keys: list[str]
    component_keys: list[str]


class StackThreeTaskAdapter:
    """StackThree adapter for PhyGenCore.

    The adapter isolates task-specific context scaling, theta featurization, and
    residual-target construction from the shared residual-field core.  Existing
    helper functions remain as wrappers for backward compatibility.
    """

    spec = TaskAdapterSpec(
        name="stack_three",
        context_keys=CONTEXT_KEYS,
        theta_feature_keys=THETA_FEATURE_KEYS,
        discrete_repair_keys=DISCRETE_REPAIR_KEYS,
        independent_z_keys=INDEPENDENT_RELAXED_CONTINUOUS_REPAIR_KEYS,
        derived_theta_keys=DERIVED_REPAIR_FEATURE_KEYS,
        component_keys=COMPONENT_KEYS,
    )

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

    def context_vector(self, context: dict[str, float]) -> np.ndarray:
        return np.array(
            [float(context.get(k, 0.0)) / self.ctx_scales[k] for k in self.spec.context_keys],
            dtype=np.float32,
        )

    def theta_features(self, theta: dict[str, Any]) -> tuple[np.ndarray, float]:
        return theta_to_features(theta)

    def feature_vector(self, context: dict[str, float], theta: dict[str, Any]) -> np.ndarray:
        theta_feat, boundary_bonus = self.theta_features(theta)
        return np.concatenate(
            [self.context_vector(context), theta_feat, np.array([boundary_bonus], dtype=np.float32)]
        )

    def component_targets(self, metrics: dict[str, float] | None, problematic: bool = False) -> np.ndarray:
        return component_targets_from_metrics(metrics, problematic)

    def component_energy_target(self, components: np.ndarray) -> np.ndarray:
        return component_energy_target(components)


STACK_THREE_ADAPTER = StackThreeTaskAdapter()


def feature_vector(context: dict[str, float], theta: dict[str, Any]) -> np.ndarray:
    return STACK_THREE_ADAPTER.feature_vector(context, theta)


def compute_proxy_quality_targets(
    records: list[dict[str, Any]],
    x: np.ndarray,
    neighbor_count: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Training-free, feedback-only proxy for robust demonstration quality.

    The proxy intentionally uses no downstream policy.  It combines a local
    Bayesian lower confidence bound on rollout success with physical margin,
    repair regularity, and moderate coverage of the observed repair manifold.
    """
    n = len(records)
    labels = np.asarray([float(r["success"]) for r in records], dtype=np.float32)
    energy = np.asarray(
        [float((r.get("metrics") or {}).get("energy", 30.0)) for r in records],
        dtype=np.float32,
    )

    # Standardized feedback geometry gives a non-parametric local success
    # posterior.  The queried record is excluded to avoid target leakage.
    scale = np.std(x, axis=0).astype(np.float32)
    scale[scale < 1e-4] = 1.0
    xs = (x - np.mean(x, axis=0, keepdims=True)) / scale[None, :]
    dist2 = np.sum((xs[:, None, :] - xs[None, :, :]) ** 2, axis=2)
    np.fill_diagonal(dist2, np.inf)
    k = max(1, min(int(neighbor_count), n - 1))
    neighbor_idx = np.argpartition(dist2, kth=k - 1, axis=1)[:, :k]
    neighbor_dist2 = np.take_along_axis(dist2, neighbor_idx, axis=1)
    finite = neighbor_dist2[np.isfinite(neighbor_dist2)]
    bandwidth2 = float(np.median(finite)) if finite.size else 1.0
    bandwidth2 = max(bandwidth2, 1e-4)
    weights = np.exp(-0.5 * neighbor_dist2 / bandwidth2).astype(np.float32)
    local_success = labels[neighbor_idx]
    weighted_success = np.sum(weights * local_success, axis=1)
    effective_n = np.sum(weights, axis=1)
    alpha_post = 1.0 + weighted_success
    beta_post = 1.0 + effective_n - weighted_success
    posterior_mean = alpha_post / (alpha_post + beta_post)
    posterior_var = (alpha_post * beta_post) / (
        (alpha_post + beta_post) ** 2 * (alpha_post + beta_post + 1.0)
    )
    # Normal approximation to the 10% Beta posterior quantile.  This avoids a
    # scipy dependency and remains conservative in sparsely sampled regions.
    success_lcb = np.clip(posterior_mean - 1.2815516 * np.sqrt(posterior_var), 0.0, 1.0).astype(np.float32)

    physics_quality = np.exp(-np.clip(energy, 0.0, 30.0) / 10.0).astype(np.float32)
    regularity = np.empty(n, dtype=np.float32)
    for i, r in enumerate(records):
        theta = r.get("theta") or {}
        noise = float(theta.get("action_noise", 0.05))
        interp = float(theta.get("num_interpolation_steps", 5.0))
        fixed = float(theta.get("num_fixed_steps", 0.0))
        problematic = float(bool(r.get("problematic")))
        noise_score = math.exp(-noise / 0.10)
        interp_score = 0.5 + 0.5 * min(max(interp / 10.0, 0.0), 1.0)
        fixed_score = math.exp(-fixed / 8.0)
        regularity[i] = float(noise_score * interp_score * fixed_score * (1.0 - problematic))

    success_idx = np.flatnonzero(labels > 0.5)
    if len(success_idx) >= 2:
        ds = np.sqrt(np.maximum(dist2[:, success_idx], 0.0))
        for col, idx in enumerate(success_idx):
            ds[idx, col] = np.inf
        nearest_success = np.min(ds, axis=1)
        finite_success = nearest_success[np.isfinite(nearest_success)]
        center = float(np.median(finite_success))
        spread = float(np.quantile(finite_success, 0.75) - np.quantile(finite_success, 0.25))
        spread = max(spread, 0.25)
        coverage = np.exp(-0.5 * ((nearest_success - center) / spread) ** 2)
        coverage[~np.isfinite(coverage)] = 0.0
    else:
        coverage = np.ones(n, dtype=np.float32)
    coverage = np.asarray(coverage, dtype=np.float32)

    quality = (
        0.50 * success_lcb
        + 0.25 * physics_quality
        + 0.15 * regularity
        + 0.10 * coverage
    ).astype(np.float32)

    # Compare candidates only within the same failed-demo context.  One
    # high-vs-low pair per demo is stable and avoids O(n^2) ranking noise.
    grouped: dict[str, list[int]] = {}
    for idx, record in enumerate(records):
        grouped.setdefault(str(record["demo_key"]), []).append(idx)
    pairs: list[tuple[int, int]] = []
    for indices in grouped.values():
        if len(indices) < 2:
            continue
        ordered = sorted(indices, key=lambda idx: float(quality[idx]))
        lo, hi = ordered[0], ordered[-1]
        if float(quality[hi] - quality[lo]) >= 0.03:
            pairs.append((hi, lo))
    pair_array = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    stats = {
        "definition": "0.50*success_lcb + 0.25*physics + 0.15*repair_regularity + 0.10*coverage",
        "neighbor_count": k,
        "kernel_bandwidth_squared": bandwidth2,
        "num_ranking_pairs": int(len(pair_array)),
        "success_lcb_mean": float(success_lcb.mean()),
        "success_lcb_min": float(success_lcb.min()),
        "success_lcb_max": float(success_lcb.max()),
        "proxy_quality_mean": float(quality.mean()),
        "proxy_quality_min": float(quality.min()),
        "proxy_quality_max": float(quality.max()),
    }
    return success_lcb, quality, pair_array, stats


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
    success_head_mode: str = "legacy",
    pinn_gradient_domain: str = "continuous",
    enable_energy_loss: bool = True,
    enable_consistency_loss: bool = True,
    enable_margin_loss: bool = True,
    enable_physics_residual_loss: bool = True,
    enable_boundary_loss: bool = True,
    enable_differential_loss: bool = True,
    enable_hjb_pde_loss: bool = True,
    enable_transport_pde_loss: bool = True,
    use_proxy_quality: bool = False,
    proxy_calibration_weight: float = 0.10,
    proxy_ranking_weight: float = 0.20,
    proxy_ranking_margin: float = 0.10,
    proxy_neighbor_count: int = 32,
):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = np.stack([feature_vector(r["context_metrics"], r["theta"]) for r in records]).astype(np.float32)
    energy = np.array([float(r["metrics"]["energy"]) for r in records], dtype=np.float32)
    e_target = np.clip(energy, 0.0, 30.0) / 30.0
    success = np.array([float(r["success"]) for r in records], dtype=np.float32)
    component_target = np.stack(
        [component_targets_from_metrics(r.get("metrics"), bool(r.get("problematic"))) for r in records]
    ).astype(np.float32)
    proxy_lcb, proxy_quality, proxy_pairs, proxy_stats = compute_proxy_quality_targets(
        records, x, neighbor_count=proxy_neighbor_count
    )

    xt = torch.from_numpy(x).to(device)
    et = torch.from_numpy(e_target).unsqueeze(1).to(device)
    st = torch.from_numpy(success).unsqueeze(1).to(device)
    ct = torch.from_numpy(component_target).to(device)
    proxy_lcb_t = torch.from_numpy(proxy_lcb).unsqueeze(1).to(device)
    proxy_pairs_t = torch.from_numpy(proxy_pairs).long().to(device)

    use_component_head = use_component_loss or use_true_pinn or use_standard_pinn

    if use_component_head:
        class ComponentScorer(nn.Module):
            """PhyGen-v1 residual-field core for StackThree.

            Backward-compatible keys are preserved:
            - energy_success[:, 0] is V_raw, the residual value field.
            - energy_success[:, 1] is either the legacy success logit or a
              residual-field correction eps, depending on success_head_mode.

            Paper-facing keys expose the intended V/q/p semantics directly.
            """

            def __init__(self, input_dim: int, mode: str = "legacy"):
                super().__init__()
                if mode not in {"legacy", "residual_induced"}:
                    raise ValueError(f"Unsupported success_head_mode: {mode}")
                self.success_head_mode = mode
                self.backbone = nn.Sequential(
                    nn.Linear(input_dim, 128),
                    nn.SiLU(),
                    nn.LayerNorm(128),
                    nn.Linear(128, 96),
                    nn.SiLU(),
                    nn.LayerNorm(96),
                )
                self.head_energy_success = nn.Linear(96, 2)
                self.head_components = nn.Linear(96, len(COMPONENT_KEYS))

            def forward(self, inp: torch.Tensor) -> dict[str, torch.Tensor]:
                h = self.backbone(inp)
                es = self.head_energy_success(h)
                V = es[:, :1]
                eps_or_logit = es[:, 1:2]
                components = torch.sigmoid(self.head_components(h))
                comp_weights_local = torch.as_tensor(
                    COMPONENT_WEIGHTS, dtype=components.dtype, device=components.device
                )
                q_total = (components * comp_weights_local[None, :]).sum(dim=1, keepdim=True) / comp_weights_local.sum()
                induced_logit = SUCCESS_RESIDUAL_BETA * (SUCCESS_RESIDUAL_THRESHOLD - V) + eps_or_logit
                p_logit = induced_logit if self.success_head_mode == "residual_induced" else eps_or_logit
                return {
                    # legacy-compatible tensors
                    "energy_success": torch.cat([V, p_logit], dim=1),
                    "components": components,
                    # PhyGen-v1 semantics
                    "V": V,
                    "q_components": components,
                    "q_total": q_total,
                    "p_logit": p_logit,
                    "p": torch.sigmoid(p_logit),
                    "eps": eps_or_logit,
                    "success_head_mode": self.success_head_mode,
                }

        model = ComponentScorer(x.shape[1], mode=success_head_mode)
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
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    pos = float(st.sum().item())
    pos_weight = torch.tensor([(len(st) - pos) / max(pos, 1.0)], dtype=torch.float32, device=device)
    comp_weights = torch.from_numpy(COMPONENT_WEIGHTS).float().to(device)
    comp_energy_target = torch.from_numpy(component_energy_target(component_target)).unsqueeze(1).to(device)

    history: list[dict[str, float]] = []
    if pinn_gradient_domain not in {"continuous", "all"}:
        raise ValueError("pinn_gradient_domain must be 'continuous' or 'all'")
    pinn_indices = pinn_feature_indices(pinn_gradient_domain)

    for epoch in range(epochs):
        need_pinn_grad = use_true_pinn or use_standard_pinn
        xt_base = xt.detach().clone().requires_grad_(need_pinn_grad)
        if need_pinn_grad and pinn_gradient_domain in {"continuous", "independent_z"}:
            xt_epoch = project_derived_theta_features(xt_base)
        else:
            xt_epoch = xt_base
        raw_pred = model(xt_epoch)
        if isinstance(raw_pred, dict):
            pred = raw_pred["energy_success"]
            pred_components = raw_pred["components"]
        else:
            pred = raw_pred
            pred_components = None
        e_pred = pred[:, :1]
        s_logit = pred[:, 1:2]
        loss_e = F.smooth_l1_loss(e_pred, et)
        loss_s = F.binary_cross_entropy_with_logits(s_logit, st, pos_weight=pos_weight)
        # Physics consistency: lower residual energy should imply higher success probability.
        s_from_e = torch.sigmoid((SUCCESS_RESIDUAL_THRESHOLD - e_pred) * SUCCESS_RESIDUAL_BETA)
        loss_cons = F.binary_cross_entropy(s_from_e, st)
        # Soft margin: successful candidates should have low residual energy.
        success_margin = torch.relu(e_pred - SUCCESS_RESIDUAL_THRESHOLD) * st
        fail_margin = torch.relu(0.12 - e_pred) * (1.0 - st)
        loss_margin = success_margin.mean() + fail_margin.mean()
        loss_components = torch.zeros((), dtype=torch.float32, device=device)
        loss_component_total = torch.zeros((), dtype=torch.float32, device=device)
        loss_component_success = torch.zeros((), dtype=torch.float32, device=device)
        loss_total_consistency = torch.zeros((), dtype=torch.float32, device=device)
        loss_physics_residual = torch.zeros((), dtype=torch.float32, device=device)
        loss_boundary = torch.zeros((), dtype=torch.float32, device=device)
        loss_differential = torch.zeros((), dtype=torch.float32, device=device)
        loss_hjb_pde = torch.zeros((), dtype=torch.float32, device=device)
        loss_collocation_transport_pde = torch.zeros((), dtype=torch.float32, device=device)
        loss_proxy_calibration = torch.zeros((), dtype=torch.float32, device=device)
        loss_proxy_ranking = torch.zeros((), dtype=torch.float32, device=device)
        if use_proxy_quality:
            p_success_proxy = torch.sigmoid(s_logit)
            loss_proxy_calibration = F.mse_loss(p_success_proxy, proxy_lcb_t)
            if proxy_pairs_t.numel() > 0:
                utility = p_success_proxy - 0.50 * e_pred
                utility_hi = utility[proxy_pairs_t[:, 0]]
                utility_lo = utility[proxy_pairs_t[:, 1]]
                loss_proxy_ranking = torch.relu(
                    proxy_ranking_margin - utility_hi + utility_lo
                ).mean()
        if pred_components is not None:
            loss_components = F.smooth_l1_loss(pred_components, ct)
            comp_e = (pred_components * comp_weights[None, :]).sum(dim=1, keepdim=True) / comp_weights.sum()
            loss_component_total = F.smooth_l1_loss(comp_e, comp_energy_target)
            loss_component_success = F.binary_cross_entropy(torch.sigmoid((SUCCESS_RESIDUAL_THRESHOLD - comp_e) * SUCCESS_RESIDUAL_BETA), st)

            if use_true_pinn or use_standard_pinn:
                # PINN residual 1: predicted scalar energy must be the
                # weighted integral of the predicted physical components.
                loss_total_consistency = F.smooth_l1_loss(e_pred, comp_e)

                e_xy, e_transport, e_lift, e_contact, e_bilateral, e_dynamics, e_slip, e_coupling = [
                    pred_components[:, i : i + 1] for i in range(len(COMPONENT_KEYS))
                ]
                # PINN residual 2: differentiable physical relations between
                # component fields. These are soft constraints, not labels.
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

                # PINN residual 3: boundary conditions. Successful rollouts
                # should lie near the zero-residual manifold; failures should
                # not be assigned a near-zero component field.
                comp_mean = pred_components.mean(dim=1, keepdim=True)
                success_boundary = st * pred_components.pow(2).mean(dim=1, keepdim=True)
                failure_boundary = (1.0 - st) * torch.relu(0.08 - comp_mean).pow(2)
                loss_boundary = success_boundary.mean() + failure_boundary.mean()

                # PINN residual 4: differential consistency between the
                # energy field and the success-probability field:
                # if p = sigmoid(k * (tau - E)), then
                # grad_theta p + k p (1-p) grad_theta E = 0.
                p_success = torch.sigmoid(s_logit)
                grad_e = torch.autograd.grad(
                    e_pred.sum(),
                    xt_base,
                    create_graph=True,
                    retain_graph=True,
                )[0][:, pinn_indices]
                grad_p = torch.autograd.grad(
                    p_success.sum(),
                    xt_base,
                    create_graph=True,
                    retain_graph=True,
                )[0][:, pinn_indices]
                differential_residual = grad_p + SUCCESS_RESIDUAL_BETA * p_success * (1.0 - p_success) * grad_e
                loss_differential = differential_residual.pow(2).mean()

            if use_standard_pinn:
                # Standard PINN collocation: enforce governing PDEs at
                # unlabeled relaxed theta-domain collocation points.
                x_col_base = xt.detach().clone()
                theta_noise = 0.035 * torch.randn_like(x_col_base[:, pinn_indices])
                x_col_base[:, pinn_indices] = torch.clamp(
                    x_col_base[:, pinn_indices] + theta_noise,
                    0.0,
                    1.0,
                )
                x_col_base.requires_grad_(True)
                if pinn_gradient_domain in {"continuous", "independent_z"}:
                    x_col_model = project_derived_theta_features(x_col_base)
                else:
                    x_col_model = x_col_base
                raw_col = model(x_col_model)
                col_pred = raw_col["energy_success"] if isinstance(raw_col, dict) else raw_col
                col_components = raw_col["components"] if isinstance(raw_col, dict) else None
                col_e = col_pred[:, :1]
                col_p = torch.sigmoid(col_pred[:, 1:2])
                col_grad_e = torch.autograd.grad(
                    col_e.sum(),
                    x_col_base,
                    create_graph=True,
                    retain_graph=True,
                )[0][:, pinn_indices]
                col_grad_p = torch.autograd.grad(
                    col_p.sum(),
                    x_col_base,
                    create_graph=True,
                    retain_graph=True,
                )[0][:, pinn_indices]

                if col_components is not None:
                    col_source = (col_components * comp_weights[None, :]).sum(dim=1, keepdim=True) / comp_weights.sum()
                else:
                    col_source = col_e.detach().clamp_min(0.0)

                # Governing equation 1: stationary HJB / Eikonal residual
                # over the repair-parameter domain. V=E is a value field;
                # q is the component residual source.
                hjb_residual = 0.5 * col_grad_e.pow(2).sum(dim=1, keepdim=True) - col_source.detach()
                loss_hjb_pde = hjb_residual.pow(2).mean()

                # Governing equation 2: success probability is transported
                # along the energy field according to p=sigmoid(beta(tau-E)).
                col_transport_residual = col_grad_p + SUCCESS_RESIDUAL_BETA * col_p * (1.0 - col_p) * col_grad_e
                loss_collocation_transport_pde = col_transport_residual.pow(2).mean()
        loss = (
            (1.0 if enable_energy_loss else 0.0) * loss_e
            + 0.45 * loss_s
            + (0.30 if enable_consistency_loss else 0.0) * loss_cons
            + (0.20 if enable_margin_loss else 0.0) * loss_margin
            + component_weight * loss_components
            + 0.25 * component_weight * loss_component_total
            + 0.20 * component_weight * loss_component_success
            + pinn_weight * loss_total_consistency
            + (pinn_weight if enable_physics_residual_loss else 0.0) * loss_physics_residual
            + (0.50 * pinn_weight if enable_boundary_loss else 0.0) * loss_boundary
            + (0.25 * pinn_weight if enable_differential_loss else 0.0) * loss_differential
            + (standard_pinn_weight if enable_hjb_pde_loss else 0.0) * loss_hjb_pde
            + (standard_pinn_weight if enable_transport_pde_loss else 0.0) * loss_collocation_transport_pde
            + (proxy_calibration_weight if use_proxy_quality else 0.0) * loss_proxy_calibration
            + (proxy_ranking_weight if use_proxy_quality else 0.0) * loss_proxy_ranking
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
                    "loss_components": float(loss_components.item()),
                    "loss_component_total": float(loss_component_total.item()),
                    "loss_component_success": float(loss_component_success.item()),
                    "loss_total_consistency": float(loss_total_consistency.item()),
                    "loss_physics_residual": float(loss_physics_residual.item()),
                    "loss_boundary": float(loss_boundary.item()),
                    "loss_differential": float(loss_differential.item()),
                    "loss_hjb_pde": float(loss_hjb_pde.item()),
                    "loss_collocation_transport_pde": float(loss_collocation_transport_pde.item()),
                    "loss_proxy_calibration": float(loss_proxy_calibration.item()),
                    "loss_proxy_ranking": float(loss_proxy_ranking.item()),
                }
            )
    return model, x.shape[1], history, proxy_stats


def predict(model: Any, contexts: list[dict[str, float]], thetas: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    import torch

    x = np.stack([feature_vector(c, t) for c, t in zip(contexts, thetas)]).astype(np.float32)
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        raw_pred = model(torch.from_numpy(x).to(device))
        if isinstance(raw_pred, dict):
            pred = raw_pred["energy_success"]
            comps = raw_pred["components"].cpu().numpy()
            p_tensor = raw_pred.get("p", torch.sigmoid(pred[:, 1]))
        else:
            pred = raw_pred
            comps = None
            p_tensor = torch.sigmoid(pred[:, 1])
        e = pred[:, 0].cpu().numpy() * 30.0
        p = p_tensor.cpu().numpy().reshape(-1)
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



def main_method_loss_switches() -> dict[str, bool]:
    return {
        "energy_loss": MAIN_ENABLE_ENERGY_LOSS,
        "consistency_loss": MAIN_ENABLE_CONSISTENCY_LOSS,
        "margin_loss": MAIN_ENABLE_MARGIN_LOSS,
        "physics_residual_loss": MAIN_ENABLE_PHYSICS_RESIDUAL_LOSS,
        "boundary_loss": MAIN_ENABLE_BOUNDARY_LOSS,
        "differential_loss": MAIN_ENABLE_DIFFERENTIAL_LOSS,
        "hjb_pde_loss": MAIN_ENABLE_HJB_PDE_LOSS,
        "transport_pde_loss": MAIN_ENABLE_TRANSPORT_PDE_LOSS,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train the StackThree PhyGen main-method residual-field PINN and "
            "emit a candidate plan for true rollout validation."
        )
    )
    parser.add_argument("--feedback-jsonl", required=True)
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
    parser.add_argument("--target-max-failed-demos", type=int, default=0)
    parser.add_argument("--target-exclude-success-jsonl", action="append", default=None)
    # This remains a data-generation setting rather than a model-version switch.
    # "safe" is the default because it matches the stable candidate pool used by
    # the current strongest StackThree repair pipeline.
    parser.add_argument("--candidate-mode", choices=["default", "safe"], default="safe")
    # Weights are kept tunable because they are scalar training hyperparameters,
    # not alternative methods.
    parser.add_argument("--component-weight", type=float, default=0.45)
    parser.add_argument("--pinn-weight", type=float, default=0.35)
    parser.add_argument("--collocation-weight", dest="standard_pinn_weight", type=float, default=0.35)
    parser.add_argument("--proxy-quality", action="store_true")
    parser.add_argument("--proxy-calibration-weight", type=float, default=0.10)
    parser.add_argument("--proxy-ranking-weight", type=float, default=0.20)
    parser.add_argument("--proxy-ranking-margin", type=float, default=0.10)
    parser.add_argument("--proxy-neighbor-count", type=int, default=32)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(args.feedback_jsonl)
    if not records:
        raise RuntimeError("No usable feedback records found")

    import torch

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model, input_dim, history, proxy_stats = train_model(
        records,
        epochs=args.epochs,
        lr=args.lr,
        use_component_loss=MAIN_USE_COMPONENT_LOSS,
        component_weight=args.component_weight,
        use_true_pinn=MAIN_USE_TRUE_PINN,
        pinn_weight=args.pinn_weight,
        use_standard_pinn=MAIN_USE_STANDARD_PINN,
        standard_pinn_weight=args.standard_pinn_weight,
        success_head_mode=MAIN_SUCCESS_HEAD_MODE,
        pinn_gradient_domain=MAIN_PINN_GRADIENT_DOMAIN,
        enable_energy_loss=MAIN_ENABLE_ENERGY_LOSS,
        enable_consistency_loss=MAIN_ENABLE_CONSISTENCY_LOSS,
        enable_margin_loss=MAIN_ENABLE_MARGIN_LOSS,
        enable_physics_residual_loss=MAIN_ENABLE_PHYSICS_RESIDUAL_LOSS,
        enable_boundary_loss=MAIN_ENABLE_BOUNDARY_LOSS,
        enable_differential_loss=MAIN_ENABLE_DIFFERENTIAL_LOSS,
        enable_hjb_pde_loss=MAIN_ENABLE_HJB_PDE_LOSS,
        enable_transport_pde_loss=MAIN_ENABLE_TRANSPORT_PDE_LOSS,
        use_proxy_quality=args.proxy_quality,
        proxy_calibration_weight=args.proxy_calibration_weight,
        proxy_ranking_weight=args.proxy_ranking_weight,
        proxy_ranking_margin=args.proxy_ranking_margin,
        proxy_neighbor_count=args.proxy_neighbor_count,
    )
    offline = offline_selector_report(records, model, budget=args.budget, boundary_weight=args.boundary_weight)
    plan_records = records
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
        out_path=out_dir / "phygen_main_candidate_plan.jsonl",
        pool_size=args.pool_size,
        budget=args.budget,
        seed=args.seed,
        start_index=args.start_index,
        boundary_weight=args.boundary_weight,
        include_repaired=args.include_repaired,
        candidate_mode=args.candidate_mode,
    )

    checkpoint_payload = {
        "state_dict": model.state_dict(),
        "input_dim": input_dim,
        "method": MAIN_METHOD_NAME,
        "context_keys": CONTEXT_KEYS,
        "component_keys": COMPONENT_KEYS,
        "theta_feature_keys": THETA_FEATURE_KEYS,
        "discrete_repair_keys": DISCRETE_REPAIR_KEYS,
        "relaxed_continuous_repair_keys": RELAXED_CONTINUOUS_REPAIR_KEYS,
        "independent_z_repair_keys": INDEPENDENT_RELAXED_CONTINUOUS_REPAIR_KEYS,
        "derived_theta_feature_keys": DERIVED_REPAIR_FEATURE_KEYS,
        "task_adapter": STACK_THREE_ADAPTER.spec.name,
        "active_loss_switches": main_method_loss_switches(),
        "pinn_gradient_domain": MAIN_PINN_GRADIENT_DOMAIN,
        "success_head_mode": MAIN_SUCCESS_HEAD_MODE,
        "success_residual_threshold": SUCCESS_RESIDUAL_THRESHOLD,
        "success_residual_beta": SUCCESS_RESIDUAL_BETA,
        "use_component_loss": MAIN_USE_COMPONENT_LOSS,
        "true_pinn": MAIN_USE_TRUE_PINN,
        "standard_pinn": MAIN_USE_STANDARD_PINN,
        "proxy_quality": args.proxy_quality,
        "proxy_quality_config": {
            "calibration_weight": args.proxy_calibration_weight,
            "ranking_weight": args.proxy_ranking_weight,
            "ranking_margin": args.proxy_ranking_margin,
            "neighbor_count": args.proxy_neighbor_count,
            **proxy_stats,
        },
    }
    torch.save(checkpoint_payload, out_dir / "stack_three_phygen_main.pt")

    grad_symbol = "z" if MAIN_PINN_GRADIENT_DOMAIN == "continuous" else "theta"
    summary = {
        "method": MAIN_METHOD_NAME + ("_proxy_quality" if args.proxy_quality else "_baseline"),
        "paper_main_method": True,
        "task_adapter": STACK_THREE_ADAPTER.spec.name,
        "candidate_mode": args.candidate_mode,
        "active_loss_switches": main_method_loss_switches(),
        "theta_feature_keys": THETA_FEATURE_KEYS,
        "discrete_repair_keys": DISCRETE_REPAIR_KEYS,
        "relaxed_continuous_repair_keys": RELAXED_CONTINUOUS_REPAIR_KEYS,
        "independent_z_repair_keys": INDEPENDENT_RELAXED_CONTINUOUS_REPAIR_KEYS,
        "derived_theta_feature_keys": DERIVED_REPAIR_FEATURE_KEYS,
        "pinn_gradient_domain": MAIN_PINN_GRADIENT_DOMAIN,
        "success_head_mode": MAIN_SUCCESS_HEAD_MODE,
        "success_residual_threshold": SUCCESS_RESIDUAL_THRESHOLD,
        "success_residual_beta": SUCCESS_RESIDUAL_BETA,
        "model_semantics": {
            "V": "residual value / cost-to-success field",
            "q": "structured physics residual source from component heads",
            "p": "residual-induced success probability p=sigmoid(beta*(tau-V)+eps)",
            "selector_score": "A(c,theta)=pred_energy - lambda_p*pred_success_prob - lambda_b*boundary_bonus",
        },
        "loss_groups": {
            "supervised_rollout": ["loss_energy", "loss_success", "loss_consistency", "loss_margin"],
            "residual_components": ["loss_components", "loss_component_total", "loss_component_success"],
            "field_consistency": ["loss_total_consistency", "loss_physics_residual", "loss_differential"],
            "boundary_collocation": ["loss_boundary", "loss_hjb_pde", "loss_collocation_transport_pde"],
        },
        "component_weight": args.component_weight,
        "pinn_weight": args.pinn_weight,
        "standard_pinn_weight": args.standard_pinn_weight,
        "governing_equations": [
            "theta=(d,z), where d is discrete repair mode and z contains independent relaxed-continuous repair parameters; offset_width/mid are derived features",
            "V(c,d,z)=E_pred(c,d,z)",
            "q(c,d,z)=weighted_sum(E_xy,E_transport,E_lift,E_contact,E_bilateral,E_dynamics,E_slip,E_coupling)",
            f"0.5 * ||grad_{grad_symbol} V||^2 - q(c,d,z) = 0",
            f"grad_{grad_symbol} p + beta * p * (1-p) * grad_{grad_symbol} V = 0",
        ],
        "num_feedback_records": len(records),
        "num_feedback_success": int(sum(1 for r in records if r["success"])),
        "num_feedback_demos": len({r["demo_key"] for r in records}),
        "training_device": str(next(model.parameters()).device),
        "proxy_quality": args.proxy_quality,
        "proxy_quality_config": {
            "calibration_weight": args.proxy_calibration_weight,
            "ranking_weight": args.proxy_ranking_weight,
            "ranking_margin": args.proxy_ranking_margin,
            "neighbor_count": args.proxy_neighbor_count,
            **proxy_stats,
        },
        "offline_budget": args.budget,
        "offline_oracle_demo_success": int(sum(1 for r in offline if r["oracle_success"])),
        "offline_selector_demo_success": int(sum(1 for r in offline if r["selector_success"])),
        "offline_per_demo": offline,
        "target_failed_hdf5": args.target_failed_hdf5,
        "loss_history": history,
        **plan_info,
    }
    (out_dir / "phygen_main_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
