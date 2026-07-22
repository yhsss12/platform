"""Residual-Guided CEM Trajectory Refinement（V2-A offline proxy）。"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from energy_model import (
    NEAR_PEG_XY_THRESH,
    ON_PEG_XY_THRESH,
    Z_INSERT_RISK_THRESH,
    score_candidate_trajectory,
)
from extract_features import NutAssemblyFeatures
from trajectory_parameterization import (
    TrajectoryProxy,
    apply_theta_to_proxy_features,
    clip_theta_vector,
    suggest_initial_theta,
    theta_bounds_as_arrays,
    vector_to_theta,
)


def classify_cem_outcome(
    energy_before: float,
    energy_after: float,
    failure_type_before: str,
    failure_type_after: str,
    features_after: NutAssemblyFeatures,
) -> str:
    """CEM 后候选状态分类（proxy-level，非仿真验证）。"""
    drop_ratio = (energy_before - energy_after) / max(energy_before, 1e-9)
    if drop_ratio < 0.05:
        return failure_type_after

    min_xy = features_after.min_nut_peg_xy_distance
    final_z = features_after.final_nut_peg_z_difference

    if (
        min_xy <= NEAR_PEG_XY_THRESH
        and final_z <= Z_INSERT_RISK_THRESH
        and min_xy <= ON_PEG_XY_THRESH * 1.5
    ):
        return "candidate_ready"
    if drop_ratio >= 0.30:
        return "lower_energy_candidate"
    return failure_type_after


def refine_trajectory_cem(
    proxy: TrajectoryProxy,
    score_fn: Callable[[NutAssemblyFeatures], dict[str, Any]] | None = None,
    n_samples: int = 128,
    elite_frac: float = 0.1,
    num_iters: int = 5,
    seed: int = 0,
    init_std_scale: float = 0.25,
) -> dict[str, Any]:
    """
    对单条 demo 运行 CEM，优化 proxy theta 使 E_total_norm 下降。

    这是 offline proxy refinement，不是 MuJoCo rollout。
    """
    rng = np.random.default_rng(seed)
    lows, highs = theta_bounds_as_arrays()
    span = highs - lows

    if score_fn is None:
        def score_fn(features: NutAssemblyFeatures) -> dict[str, Any]:
            return score_candidate_trajectory(features)

    baseline_features = apply_theta_to_proxy_features(proxy, np.zeros_like(lows))
    before = score_fn(baseline_features)

    mean = suggest_initial_theta(proxy)
    std = span * init_std_scale
    std[6] = min(std[6], 0.15)  # speed_scale 窄一些

    best_vector = mean.copy()
    best_score = before
    iteration_history: list[dict[str, Any]] = []

    for iteration in range(num_iters):
        samples = rng.normal(loc=mean, scale=std, size=(n_samples, len(mean)))
        for index in range(n_samples):
            samples[index] = clip_theta_vector(samples[index])

        scores: list[dict[str, Any]] = []
        energies = np.empty(n_samples, dtype=float)
        for index in range(n_samples):
            features = apply_theta_to_proxy_features(proxy, samples[index])
            score = score_fn(features)
            scores.append(score)
            energies[index] = score["E_total_norm"]

        elite_count = max(1, int(n_samples * elite_frac))
        elite_idx = np.argsort(energies)[:elite_count]
        elite_vectors = samples[elite_idx]
        elite_energies = energies[elite_idx]

        mean = elite_vectors.mean(axis=0)
        std = elite_vectors.std(axis=0) + 1e-6
        std = np.minimum(std, span * 0.35)

        iter_best_idx = int(elite_idx[0])
        if energies[iter_best_idx] < best_score["E_total_norm"]:
            best_vector = samples[iter_best_idx].copy()
            best_score = scores[iter_best_idx]

        iteration_history.append(
            {
                "iteration": iteration,
                "best_energy": float(energies[iter_best_idx]),
                "mean_energy": float(energies.mean()),
                "elite_mean_energy": float(elite_energies.mean()),
                "elite_min_energy": float(elite_energies.min()),
                "best_theta": vector_to_theta(samples[iter_best_idx]),
            }
        )

    after_features = apply_theta_to_proxy_features(proxy, best_vector)
    after = score_fn(after_features)
    failure_type_after = classify_cem_outcome(
        before["E_total_norm"],
        after["E_total_norm"],
        before["failure_type"],
        after["failure_type"],
        after_features,
    )

    energy_drop_ratio = (before["E_total_norm"] - after["E_total_norm"]) / max(
        before["E_total_norm"], 1e-9
    )

    return {
        "demo_key": proxy.demo_key,
        "label": proxy.label,
        "best_theta": vector_to_theta(best_vector),
        "best_theta_vector": best_vector.tolist(),
        "energy_before": before["E_total_norm"],
        "energy_after": after["E_total_norm"],
        "energy_drop_ratio": float(energy_drop_ratio),
        "components_before": before["components"],
        "components_after": after["components"],
        "failure_type_before": before["failure_type"],
        "failure_type_after_raw": after["failure_type"],
        "failure_type_after": failure_type_after,
        "optimization_targets_before": before["optimization_targets"],
        "optimization_targets_after": after["optimization_targets"],
        "residual_before": {
            "final_xy": baseline_features.final_nut_peg_xy_distance,
            "min_xy": baseline_features.min_nut_peg_xy_distance,
            "final_z": baseline_features.final_nut_peg_z_difference,
            "min_yaw": baseline_features.min_nut_peg_yaw_error,
        },
        "residual_after": {
            "final_xy": after_features.final_nut_peg_xy_distance,
            "min_xy": after_features.min_nut_peg_xy_distance,
            "final_z": after_features.final_nut_peg_z_difference,
            "min_yaw": after_features.min_nut_peg_yaw_error,
        },
        "iteration_history": iteration_history,
        "cem_config": {
            "n_samples": n_samples,
            "elite_frac": elite_frac,
            "num_iters": num_iters,
            "seed": seed,
        },
    }
