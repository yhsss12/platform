#!/usr/bin/env python3
"""Evaluate PhyGen selector on held-out feedback jsonl with baselines."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phygen.adapters.base_adapter import demo_sort_key
from phygen.adapters.registry import get_adapter
from phygen.core.selector import attach_selector_scores, predict_with_details, unique_union


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _demo_key(row: dict[str, Any]) -> str:
    return str(row.get("source_demo_key", row.get("demo_key", "unknown")))


def _theta_key(theta: dict[str, Any]) -> str:
    offset = theta.get("offset_range", [0, 0])
    return (
        f"{theta.get('selection_strategy')}|per={int(bool(theta.get('select_src_per_subtask')))}|"
        f"noise={theta.get('action_noise')}|off={offset[0]}-{offset[1]}"
    )


def _select_random(candidates: list[dict[str, Any]], budget: int, rng: random.Random) -> list[dict[str, Any]]:
    if budget <= 0:
        return []
    pool = list(candidates)
    rng.shuffle(pool)
    return pool[:budget]


def _select_lowest_energy(candidates: list[dict[str, Any]], budget: int) -> list[dict[str, Any]]:
    return sorted(candidates, key=lambda r: float(r["metrics"].get("energy", 30.0)))[:budget]


def _select_highest_success_prob(candidates: list[dict[str, Any]], budget: int) -> list[dict[str, Any]]:
    return sorted(candidates, key=lambda r: -float(r.get("pred_success_prob", 0.0)))[:budget]


def _group_success(selected: list[dict[str, Any]]) -> bool:
    return bool(selected) and any(bool(r.get("success")) for r in selected)


def evaluate_selector(
    *,
    adapter: Any,
    model: Any,
    records: list[dict[str, Any]],
    budgets: list[int],
    boundary_weight: float,
    uncertainty_weight: float,
    random_seed: int,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        grouped[_demo_key(rec)].append(dict(rec))

    rng = random.Random(random_seed)
    all_scored: list[dict[str, Any]] = []
    for demo in sorted(grouped.keys(), key=demo_sort_key):
        group = grouped[demo]
        contexts = [r["context_metrics"] for r in group]
        thetas = [r["theta"] for r in group]
        pred_e, pred_p, details = predict_with_details(adapter, model, contexts, thetas)
        attach_selector_scores(adapter, group, pred_e, pred_p, details, boundary_weight, uncertainty_weight)
        all_scored.extend(group)

    per_demo: list[dict[str, Any]] = []
    by_budget: dict[str, Any] = {}

    for budget in budgets:
        oracle_demo_success = 0
        selector_demo_success = 0
        random_demo_success = 0
        energy_demo_success = 0
        prob_demo_success = 0
        topk_success_counts = {"selector": [], "random": [], "lowest_energy": [], "highest_success_prob": []}

        for demo in sorted(grouped.keys(), key=demo_sort_key):
            group = [r for r in all_scored if _demo_key(r) == demo]
            oracle = bool(any(r["success"] for r in group))
            if oracle:
                oracle_demo_success += 1

            sel = unique_union(group, budget)
            rnd = _select_random(group, budget, rng)
            eng = _select_lowest_energy(group, budget)
            prob = _select_highest_success_prob(group, budget)

            selector_ok = _group_success(sel)
            random_ok = _group_success(rnd)
            energy_ok = _group_success(eng)
            prob_ok = _group_success(prob)

            selector_demo_success += int(selector_ok)
            random_demo_success += int(random_ok)
            energy_demo_success += int(energy_ok)
            prob_demo_success += int(prob_ok)

            topk_success_counts["selector"].append(int(any(r["success"] for r in sel)))
            topk_success_counts["random"].append(int(any(r["success"] for r in rnd)))
            topk_success_counts["lowest_energy"].append(int(any(r["success"] for r in eng)))
            topk_success_counts["highest_success_prob"].append(int(any(r["success"] for r in prob)))

            if budget == budgets[0]:
                per_demo.append(
                    {
                        "demo_key": demo,
                        "num_candidates": len(group),
                        "oracle_success": oracle,
                        f"selector_top{budget}_success": selector_ok,
                        f"random_top{budget}_success": random_ok,
                        f"lowest_energy_top{budget}_success": energy_ok,
                        f"highest_success_prob_top{budget}_success": prob_ok,
                    }
                )
            else:
                for row in per_demo:
                    if row["demo_key"] == demo:
                        row[f"selector_top{budget}_success"] = selector_ok
                        row[f"random_top{budget}_success"] = random_ok
                        row[f"lowest_energy_top{budget}_success"] = energy_ok
                        row[f"highest_success_prob_top{budget}_success"] = prob_ok

        num_demos = len(grouped)
        by_budget[str(budget)] = {
            "budget": budget,
            "num_demos": num_demos,
            "oracle_demo_success": oracle_demo_success,
            "selector_demo_success": selector_demo_success,
            "random_demo_success": random_demo_success,
            "lowest_energy_demo_success": energy_demo_success,
            "highest_success_prob_demo_success": prob_demo_success,
            "selector_topk_success_rate": selector_demo_success / max(num_demos, 1),
            "random_topk_success_rate": random_demo_success / max(num_demos, 1),
            "lowest_energy_topk_success_rate": energy_demo_success / max(num_demos, 1),
            "highest_success_prob_topk_success_rate": prob_demo_success / max(num_demos, 1),
            "selector_vs_random_delta": (selector_demo_success - random_demo_success) / max(num_demos, 1),
        }

    theta_stats: dict[str, dict[str, Any]] = {}
    for row in all_scored:
        key = _theta_key(row["theta"])
        bucket = theta_stats.setdefault(key, {"count": 0, "success": 0, "theta": row["theta"]})
        bucket["count"] += 1
        bucket["success"] += int(bool(row.get("success")))
    for bucket in theta_stats.values():
        bucket["success_rate"] = bucket["success"] / max(bucket["count"], 1)

    return {
        "num_records": len(records),
        "num_demos": len(grouped),
        "budgets": budgets,
        "by_budget": by_budget,
        "per_source_demo": per_demo,
        "per_theta_success": sorted(
            [
                {
                    "theta_key": k,
                    "count": v["count"],
                    "success": v["success"],
                    "success_rate": v["success_rate"],
                    "theta": v["theta"],
                }
                for k, v in theta_stats.items()
            ],
            key=lambda x: (-x["count"], x["theta_key"]),
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="coffee_preparation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--feedback-jsonl", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--budgets", default="1,3,5")
    parser.add_argument("--boundary-weight", type=float, default=1.5)
    parser.add_argument("--uncertainty-weight", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=9701)
    args = parser.parse_args()

    import torch
    from phygen.core.residual_field_model import FeatureLayout, RepairParameterResidualFieldPINN, build_mlp_selector

    adapter = get_adapter(args.task)
    checkpoint_path = (ROOT / args.checkpoint).resolve()
    feedback_path = (ROOT / args.feedback_jsonl).resolve()
    output_path = (ROOT / args.output).resolve()

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    layout = FeatureLayout(
        context_dim=int(ckpt["layout"]["context_dim"]),
        theta_disc_dim=int(ckpt["layout"]["theta_disc_dim"]),
        theta_cont_dim=int(ckpt["layout"]["theta_cont_dim"]),
    )
    component_keys = ckpt.get("component_keys") or []
    if component_keys:
        model = RepairParameterResidualFieldPINN.build(layout, len(component_keys))
    else:
        model = build_mlp_selector(layout.input_dim)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    records = _load_jsonl(feedback_path)
    budgets = [int(x.strip()) for x in args.budgets.split(",") if x.strip()]
    report = evaluate_selector(
        adapter=adapter,
        model=model,
        records=records,
        budgets=budgets,
        boundary_weight=args.boundary_weight,
        uncertainty_weight=args.uncertainty_weight,
        random_seed=args.seed,
    )
    report["checkpoint"] = str(checkpoint_path)
    report["feedback_jsonl"] = str(feedback_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
