#!/usr/bin/env python3
"""V1-F：对比评估 V1-E / V1-F / explicit / random repair 排序效率。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

_V1F_DIR = Path(__file__).resolve().parent
_V1_DIR = _V1F_DIR.parent
_V1E_DIR = _V1_DIR / "repair_parameter_model"
_EXPERIMENT_DIR = _V1_DIR.parent
for path in (_V1_DIR, _V1E_DIR, _V1F_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from group_split_utils import top_k_refined_success_rate  # noqa: E402
from pinn_repair_parameter_model import PINNRepairParameterModel, explicit_repair_energy  # noqa: E402
from pinn_v1f_repair_model import PINNV1FRepairModel, explicit_v1f_repair_energy  # noqa: E402
from repair_dataset import load_repair_npz  # noqa: E402
from v1f_repair_dataset import DEMO_KEYS, load_v1f_npz  # noqa: E402

DEFAULT_V1F_DATASET = _EXPERIMENT_DIR / "outputs" / "v1f_repair_parameter_model" / "repair_parameter_dataset_v1f.npz"
DEFAULT_V1E_MODEL = _EXPERIMENT_DIR / "outputs" / "v1_repair_parameter_model" / "model.pt"
DEFAULT_V1F_MODEL = _EXPERIMENT_DIR / "outputs" / "v1f_repair_parameter_model" / "model_v1f.pt"
DEFAULT_OUTPUT = _EXPERIMENT_DIR / "outputs" / "v1f_repair_parameter_model"

POOLS = {
    "demo_4_insertion": {"demo_idx": 4, "failure_modes": {1}},
    "demo_1_transport": {"demo_idx": 1, "failure_modes": {2}},
    "demo_2_grasp": {"demo_idx": 2, "failure_modes": {3}},
    "demo_3_lift": {"demo_idx": 3, "failure_modes": {4}},
    "demo_0_transport": {"demo_idx": 0, "failure_modes": {2}},
}


def _load_v1e_model(path: Path) -> PINNRepairParameterModel:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = PINNRepairParameterModel(
        input_dim=ckpt["input_dim"],
        hidden_dim=ckpt["hidden_dim"],
        num_layers=ckpt["num_layers"],
        dropout=ckpt.get("dropout", 0.1),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def _load_v1f_model(path: Path) -> PINNV1FRepairModel:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = PINNV1FRepairModel(
        input_dim=ckpt["input_dim"],
        hidden_dim=ckpt["hidden_dim"],
        num_layers=ckpt["num_layers"],
        dropout=ckpt.get("dropout", 0.1),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def _v1f_to_v1e_features(features_v1f: np.ndarray, v1e_dim: int) -> np.ndarray:
    """将 V1-F 特征投影到 V1-E 输入维度（截断/填充）供 V1-E 模型打分。"""
    if features_v1f.shape[1] == v1e_dim:
        return features_v1f
    if features_v1f.shape[1] > v1e_dim:
        return features_v1f[:, :v1e_dim]
    pad = np.zeros((features_v1f.shape[0], v1e_dim - features_v1f.shape[1]), dtype=np.float32)
    return np.concatenate([features_v1f, pad], axis=1)


def _top_k_contains_argmin(scores: np.ndarray, values: np.ndarray, k: int) -> float:
    if len(values) == 0:
        return 0.0
    k = min(k, len(values))
    best_idx = int(np.argmin(values))
    order = np.argsort(scores)
    return float(best_idx in order[:k])


def _success_at_k(scores: np.ndarray, success: np.ndarray, k: int) -> float:
    k = min(k, len(scores))
    order = np.argsort(scores)
    return float(np.any(success[order[:k]] > 0.5))


def _rollouts_per_success(scores: np.ndarray, success: np.ndarray) -> float:
    order = np.argsort(scores)
    for rank, idx in enumerate(order, start=1):
        if success[idx] > 0.5:
            return float(rank)
    return float(len(scores) + 1)


def _repair_success_rate(scores: np.ndarray, success: np.ndarray, budget: int) -> float:
    budget = min(budget, len(scores))
    order = np.argsort(scores)
    return float(np.mean(success[order[:budget]] > 0.5))


def _evaluate_pool(
    *,
    features: np.ndarray,
    true_energy: np.ndarray,
    refined_success: np.ndarray,
    success: np.ndarray,
    v1e_model: PINNRepairParameterModel | None,
    v1f_model: PINNV1FRepairModel,
    v1e_dim: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    x_v1f = torch.from_numpy(features)
    with torch.no_grad():
        v1f_out = v1f_model(x_v1f)
        v1f_scores = v1f_out["E_total"].numpy()
        v1f_uncertainty = v1f_out["uncertainty"].numpy()
        # uncertainty-aware: penalize high uncertainty
        v1f_ua_scores = v1f_scores + 0.5 * v1f_uncertainty

    explicit_scores = explicit_v1f_repair_energy(x_v1f).numpy()
    random_scores = rng.random(len(features))

    methods: dict[str, np.ndarray] = {
        "random": random_scores,
        "explicit_energy": explicit_scores,
        "v1f_pinn": v1f_scores,
        "v1f_uncertainty_aware": v1f_ua_scores,
    }

    if v1e_model is not None:
        x_v1e = torch.from_numpy(_v1f_to_v1e_features(features, v1e_dim))
        with torch.no_grad():
            v1e_scores = v1e_model(x_v1e)["E_total"].numpy()
        methods["v1e_pinn"] = v1e_scores

    out: dict[str, Any] = {}
    for name, scores in methods.items():
        out[name] = {
            "success_at_k": {
                "at_1": _success_at_k(scores, success, 1),
                "at_3": _success_at_k(scores, success, 3),
                "at_5": _success_at_k(scores, success, 5),
                "at_10": _success_at_k(scores, success, 10),
                "at_20": _success_at_k(scores, success, 20),
            },
            "refined_success_at_k": {
                "at_1": top_k_refined_success_rate(scores, refined_success, 1),
                "at_3": top_k_refined_success_rate(scores, refined_success, 3),
                "at_5": top_k_refined_success_rate(scores, refined_success, 5),
                "at_10": top_k_refined_success_rate(scores, refined_success, 10),
                "at_20": top_k_refined_success_rate(scores, refined_success, 20),
            },
            "repair_success_rate": {
                "budget_10": _repair_success_rate(scores, success, 10),
                "budget_20": _repair_success_rate(scores, success, 20),
            },
            "rollouts_per_success": _rollouts_per_success(scores, success),
            "best_E_total": float(np.min(true_energy[np.argsort(scores)])),
            "lowest_true_E_at_k": {
                "at_1": _top_k_contains_argmin(scores, true_energy, 1),
                "at_5": _top_k_contains_argmin(scores, true_energy, 5),
                "at_20": _top_k_contains_argmin(scores, true_energy, 20),
            },
        }
    return out


def _lodo_eval(
    bundle: dict[str, Any],
    v1f_model: PINNV1FRepairModel,
    v1e_model: PINNRepairParameterModel | None,
    v1e_dim: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    demo_idx = bundle["demo_idx"]
    results: dict[str, Any] = {}
    for i, demo_key in enumerate(DEMO_KEYS):
        mask = demo_idx == i
        if not np.any(mask):
            continue
        pool = _evaluate_pool(
            features=bundle["features"][mask],
            true_energy=bundle["target_E_total"][mask],
            refined_success=bundle["refined_success_flag"][mask],
            success=bundle["success_flag"][mask],
            v1e_model=v1e_model,
            v1f_model=v1f_model,
            v1e_dim=v1e_dim,
            rng=rng,
        )
        results[demo_key] = pool
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="V1-F repair evaluation vs V1-E / explicit / random")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_V1F_DATASET)
    parser.add_argument("--v1e-model", type=Path, default=DEFAULT_V1E_MODEL)
    parser.add_argument("--v1f-model", type=Path, default=DEFAULT_V1F_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    bundle = load_v1f_npz(args.dataset)
    v1f_model = _load_v1f_model(args.v1f_model)
    v1e_model = _load_v1e_model(args.v1e_model) if args.v1e_model.exists() else None
    v1e_dim = 65
    if v1e_model and args.v1e_model.exists():
        ckpt = torch.load(args.v1e_model, map_location="cpu", weights_only=False)
        v1e_dim = int(ckpt["input_dim"])

    rng = np.random.default_rng(args.seed)
    per_pool: dict[str, Any] = {}
    for pool_name, cfg in POOLS.items():
        mask = bundle["demo_idx"] == cfg["demo_idx"]
        if cfg.get("failure_modes"):
            mask = mask & np.isin(bundle["source_failure_mode_idx"], list(cfg["failure_modes"]))
        if not np.any(mask):
            continue
        per_pool[pool_name] = _evaluate_pool(
            features=bundle["features"][mask],
            true_energy=bundle["target_E_total"][mask],
            refined_success=bundle["refined_success_flag"][mask],
            success=bundle["success_flag"][mask],
            v1e_model=v1e_model,
            v1f_model=v1f_model,
            v1e_dim=v1e_dim,
            rng=rng,
        )

    lodo = _lodo_eval(bundle, v1f_model, v1e_model, v1e_dim, rng)

    demo3 = per_pool.get("demo_3_lift", {})
    demo3_improvement = {}
    if demo3:
        for method in ("v1e_pinn", "v1f_pinn", "random"):
            if method in demo3:
                demo3_improvement[method] = demo3[method]["success_at_k"]

    report = {
        "task": "v1f_repair_evaluation",
        "dataset": str(args.dataset),
        "num_samples": int(len(bundle["features"])),
        "v1e_model": str(args.v1e_model) if v1e_model else None,
        "v1f_model": str(args.v1f_model),
        "per_pool": per_pool,
        "leave_one_demo_out": lodo,
        "demo_3_lift_improvement": demo3_improvement,
        "summary_comparison": {
            pool: {
                "v1f_vs_v1e_success_at_20": (
                    per_pool[pool]["v1f_pinn"]["success_at_k"]["at_20"]
                    - per_pool[pool].get("v1e_pinn", {}).get("success_at_k", {}).get("at_20", 0.0)
                )
                if "v1f_pinn" in per_pool[pool]
                else None
            }
            for pool in per_pool
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "v1f_evaluation_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(out_path), "demo_3": demo3_improvement}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
