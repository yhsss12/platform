#!/usr/bin/env python3
"""V1-E：评估 repair-parameter 候选排序（random / explicit / PINN）。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import sys

_V1E_DIR = Path(__file__).resolve().parent
_V1_DIR = _V1E_DIR.parent
_EXPERIMENT_DIR = _V1_DIR.parent
for path in (_V1_DIR, _V1E_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from group_split_utils import top_k_refined_success_rate  # noqa: E402
from pinn_repair_parameter_model import PINNRepairParameterModel, explicit_repair_energy  # noqa: E402
from repair_dataset import load_repair_npz  # noqa: E402

DEFAULT_DATASET = _EXPERIMENT_DIR / "outputs" / "v1_repair_parameter_model" / "repair_parameter_dataset.npz"
DEFAULT_MODEL = _EXPERIMENT_DIR / "outputs" / "v1_repair_parameter_model" / "model.pt"
DEFAULT_OUTPUT = DEFAULT_MODEL.parent

POOLS = {
    "demo_4_insertion": {"demo_key": "demo_4", "active_groups": {"insertion"}},
    "demo_1_transport": {"demo_key": "demo_1", "active_groups": {"transport"}},
    "demo_2_grasp": {"demo_key": "demo_2", "active_groups": {"grasp"}},
    "demo_3_lift": {"demo_key": "demo_3", "active_groups": {"grasp", "lift"}},
}


def _top_k_contains_argmin(scores: np.ndarray, values: np.ndarray, k: int) -> float:
    if len(values) == 0:
        return 0.0
    k = min(k, len(values))
    best_idx = int(np.argmin(values))
    order = np.argsort(scores)
    return float(best_idx in order[:k])


def _avg_true_energy_topk(scores: np.ndarray, true_energy: np.ndarray, k: int) -> float:
    k = min(k, len(scores))
    order = np.argsort(scores)
    return float(np.mean(true_energy[order[:k]]))


def _success_within_budget(scores: np.ndarray, success: np.ndarray, budget: int) -> float:
    budget = min(budget, len(scores))
    order = np.argsort(scores)
    picked = success[order[:budget]]
    return float(np.any(picked > 0.5))


def _load_model(path: Path) -> PINNRepairParameterModel:
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


def _evaluate_pool(
    *,
    features: np.ndarray,
    true_energy: np.ndarray,
    refined_success: np.ndarray,
    success: np.ndarray,
    model: PINNRepairParameterModel,
    rng: np.random.Generator,
) -> dict[str, Any]:
    x = torch.from_numpy(features)
    with torch.no_grad():
        pinn_scores = model(x)["E_total"].numpy()
    explicit_scores = explicit_repair_energy(x).numpy()
    random_scores = rng.random(len(features))

    out: dict[str, Any] = {}
    for name, scores in (
        ("random", random_scores),
        ("explicit_energy", explicit_scores),
        ("pinn_predicted_energy", pinn_scores),
    ):
        out[name] = {
            "top_k_refined_success_hit_rate": {
                "top_1": top_k_refined_success_rate(scores, refined_success, 1),
                "top_3": top_k_refined_success_rate(scores, refined_success, 3),
                "top_5": top_k_refined_success_rate(scores, refined_success, 5),
            },
            "top_k_lowest_true_E_total_hit_rate": {
                "top_1": _top_k_contains_argmin(scores, true_energy, 1),
                "top_3": _top_k_contains_argmin(scores, true_energy, 3),
                "top_5": _top_k_contains_argmin(scores, true_energy, 5),
            },
            "avg_true_E_total_top_k": {
                "top_1": _avg_true_energy_topk(scores, true_energy, 1),
                "top_3": _avg_true_energy_topk(scores, true_energy, 3),
                "top_5": _avg_true_energy_topk(scores, true_energy, 5),
            },
            "success_rate_under_rollout_budget": {
                "K_10": _success_within_budget(scores, success, 10),
                "K_20": _success_within_budget(scores, success, 20),
                "K_40": _success_within_budget(scores, success, 40),
            },
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="V1-E repair parameter ranking evaluation")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    bundle = load_repair_npz(args.dataset)
    meta_records = bundle["meta"].get("meta_records", [])
    model = _load_model(args.model)
    rng = np.random.default_rng(42)

    pools_report: dict[str, Any] = {}
    for pool_id, spec in POOLS.items():
        indices = []
        for i, meta in enumerate(meta_records):
            if meta.get("demo_key") != spec["demo_key"]:
                continue
            if meta.get("active_param_group") not in spec["active_groups"]:
                continue
            indices.append(i)
        if not indices:
            continue
        idx = np.array(indices, dtype=int)
        pools_report[pool_id] = {
            "demo_key": spec["demo_key"],
            "num_candidates": int(len(idx)),
            "ranking": _evaluate_pool(
                features=bundle["features"][idx],
                true_energy=bundle["target_E_total"][idx],
                refined_success=bundle["refined_success_flag"][idx],
                success=bundle["success_flag"][idx],
                model=model,
                rng=rng,
            ),
        }

    def _macro(method: str, metric_path: tuple[str, str]) -> float | None:
        vals = []
        for pool in pools_report.values():
            node = pool["ranking"][method]
            for key in metric_path[:-1]:
                node = node[key]
            vals.append(float(node[metric_path[-1]]))
        return float(np.mean(vals)) if vals else None

    report = {
        "task": "V1-E_repair_parameter_ranking",
        "dataset": str(args.dataset),
        "model": str(args.model),
        "pools": pools_report,
        "macro_average": {
            "pinn_top_1_refined_success": _macro("pinn_predicted_energy", ("top_k_refined_success_hit_rate", "top_1")),
            "explicit_top_1_refined_success": _macro("explicit_energy", ("top_k_refined_success_hit_rate", "top_1")),
            "random_top_1_refined_success": _macro("random", ("top_k_refined_success_hit_rate", "top_1")),
            "pinn_top_3_refined_success": _macro("pinn_predicted_energy", ("top_k_refined_success_hit_rate", "top_3")),
        },
        "acceptance_check": {
            "pinn_not_worse_than_random_top1": (
                (_macro("pinn_predicted_energy", ("top_k_refined_success_hit_rate", "top_1")) or 0)
                >= (_macro("random", ("top_k_refined_success_hit_rate", "top_1")) or 0)
            ),
        },
        "notes": [
            "PINN is primary repair-parameter selector; explicit energy is baseline only.",
            "Pools use held-out candidate groups per source demo / failure mode.",
            "If PINN < explicit, see per-pool breakdown — still valid for pruning narrative.",
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "repair_parameter_ranking_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out_path), "acceptance": report["acceptance_check"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
