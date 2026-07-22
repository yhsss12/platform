from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from phygen.adapters.base_adapter import BasePhyGenAdapter, demo_sort_key


def predict_with_details(
    adapter: BasePhyGenAdapter,
    model: Any,
    contexts: list[dict[str, float]],
    thetas: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    import torch

    x = np.stack([adapter.feature_vector(c, t) for c, t in zip(contexts, thetas)]).astype(np.float32)
    model.eval()
    with torch.no_grad():
        raw_pred = model(torch.from_numpy(x))
        if isinstance(raw_pred, dict):
            pred = raw_pred["energy_success"]
            comps = raw_pred["components"].cpu().numpy()
        else:
            pred = raw_pred
            comps = None
        raw_e = pred[:, 0].cpu().numpy() * 30.0
        p = torch.sigmoid(pred[:, 1]).cpu().numpy()
        if comps is not None:
            comp_e = adapter.component_energy_target(comps) * 30.0
            e = 0.55 * raw_e + 0.45 * comp_e
            residual_disagreement = np.abs(raw_e - comp_e) / 30.0
            component_spread = np.std(comps, axis=1)
        else:
            comp_e = raw_e.copy()
            e = raw_e
            residual_disagreement = np.zeros_like(e, dtype=np.float32)
            component_spread = np.zeros_like(e, dtype=np.float32)
        decision_ambiguity = 1.0 - np.abs(p - 0.5) * 2.0
        uncertainty = 30.0 * np.clip(
            0.45 * residual_disagreement + 0.35 * component_spread + 0.20 * decision_ambiguity,
            0.0,
            1.0,
        )
        details = {
            "raw_energy": raw_e.astype(np.float32),
            "component_energy": comp_e.astype(np.float32),
            "residual_disagreement": residual_disagreement.astype(np.float32),
            "component_spread": component_spread.astype(np.float32),
            "decision_ambiguity": decision_ambiguity.astype(np.float32),
            "uncertainty": uncertainty.astype(np.float32),
        }
    return e.astype(np.float32), p.astype(np.float32), details


def predict(adapter: BasePhyGenAdapter, model: Any, contexts: list[dict[str, float]], thetas: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    e, p, _ = predict_with_details(adapter, model, contexts, thetas)
    return e, p


def attach_selector_scores(
    adapter: BasePhyGenAdapter,
    rows: list[dict[str, Any]],
    pred_e: np.ndarray,
    pred_p: np.ndarray,
    details: dict[str, np.ndarray],
    boundary_weight: float,
    uncertainty_weight: float,
) -> None:
    for i, (row, pe, pp) in enumerate(zip(rows, pred_e, pred_p)):
        _, _, boundary_bonus = adapter.theta_to_features(row["theta"])
        uncertainty = float(details["uncertainty"][i])
        row["pred_energy"] = float(pe)
        row["pred_success_prob"] = float(pp)
        row["uncertainty_score"] = uncertainty
        row["residual_disagreement"] = float(details["residual_disagreement"][i])
        row["component_spread"] = float(details["component_spread"][i])
        row["decision_ambiguity"] = float(details["decision_ambiguity"][i])
        row["utility_score"] = float(pe - 8.0 * pp)
        row["boundary_score"] = float(pe - 6.0 * pp - boundary_weight * boundary_bonus)
        uncertainty_bonus = min(uncertainty, 8.0)
        row["uncertainty_bonus"] = float(uncertainty_bonus)
        row["acquisition_score"] = float(pe - 8.0 * pp - uncertainty_weight * uncertainty_bonus)


def unique_union(candidates: list[dict[str, Any]], budget: int) -> list[dict[str, Any]]:
    if budget <= 0:
        return []
    exploitation_quota = max(1, int(math.ceil(0.6 * budget)))
    boundary_quota = max(0, int(math.floor(0.2 * budget)))
    uncertainty_quota = max(0, budget - exploitation_quota - boundary_quota)
    by_acquisition = sorted(candidates, key=lambda r: r.get("acquisition_score", r["utility_score"]))
    by_boundary = sorted(candidates, key=lambda r: r["boundary_score"])
    by_uncertainty = sorted(candidates, key=lambda r: -float(r.get("uncertainty_score", 0.0)))

    out: list[dict[str, Any]] = []
    seen: set[int] = set()

    def add_from(rows: list[dict[str, Any]], quota: int) -> None:
        if quota <= 0:
            return
        added = 0
        for row in rows:
            idx = int(row["candidate_index"])
            if idx in seen:
                continue
            seen.add(idx)
            out.append(row)
            added += 1
            if added >= quota or len(out) >= budget:
                break

    add_from(by_acquisition, exploitation_quota)
    add_from(by_boundary, boundary_quota)
    add_from(by_uncertainty, uncertainty_quota)
    add_from(by_acquisition, budget - len(out))
    add_from(by_boundary, budget - len(out))
    add_from(by_uncertainty, budget - len(out))
    return out[:budget]


def offline_selector_report(
    adapter: BasePhyGenAdapter,
    records: list[dict[str, Any]],
    model: Any,
    budget: int,
    boundary_weight: float,
    uncertainty_weight: float = 0.3,
) -> list[dict[str, Any]]:
    demos = sorted({r["demo_key"] for r in records}, key=demo_sort_key)
    rows: list[dict[str, Any]] = []
    for demo in demos:
        group = [dict(r) for r in records if r["demo_key"] == demo]
        contexts = [r["context_metrics"] for r in group]
        thetas = [r["theta"] for r in group]
        pred_e, pred_p, details = predict_with_details(adapter, model, contexts, thetas)
        attach_selector_scores(adapter, group, pred_e, pred_p, details, boundary_weight, uncertainty_weight)
        top = unique_union(group, budget)
        rows.append(
            {
                "demo_key": demo,
                "num_candidates": len(group),
                "oracle_success": bool(any(r["success"] for r in group)),
                "selector_success": bool(any(r["success"] for r in top)),
                "selected_candidate_indices": [int(r["candidate_index"]) for r in top],
                "selected_successes": [bool(r["success"]) for r in top],
                "selected_energies": [float(r["metrics"].get("energy", 30.0)) for r in top],
                "selected_acquisition_scores": [float(r.get("acquisition_score", r["utility_score"])) for r in top],
                "selected_uncertainty_scores": [float(r.get("uncertainty_score", 0.0)) for r in top],
            }
        )
    return rows


def build_candidate_plan(
    adapter: BasePhyGenAdapter,
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
    uncertainty_weight: float = 0.3,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        grouped.setdefault(rec["demo_key"], []).append(rec)

    plan_rows: list[dict[str, Any]] = []
    for demo_key in sorted(grouped, key=demo_sort_key):
        already_repaired = any(r.get("success", False) for r in grouped[demo_key])
        if already_repaired and not include_repaired:
            continue
        context = grouped[demo_key][0]["context_metrics"]
        rng = np.random.default_rng(seed + demo_sort_key(demo_key) * 1009)
        pool: list[dict[str, Any]] = []
        for i in range(pool_size):
            candidate_index = start_index + i
            theta = adapter.sample_repair_theta(candidate_index, rng, candidate_mode=candidate_mode)
            pool.append({"candidate_index": candidate_index, "theta": theta})
        pred_e, pred_p, details = predict_with_details(adapter, model, [context] * len(pool), [p["theta"] for p in pool])
        attach_selector_scores(adapter, pool, pred_e, pred_p, details, boundary_weight, uncertainty_weight)
        selected = unique_union(pool, budget)
        for rank, row in enumerate(selected):
            row["planner_rank"] = rank
            row["planner_score"] = float(row.get("acquisition_score", min(row["utility_score"], row["boundary_score"])))
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
        "selector": "quota_acquisition_boundary_uncertainty_union",
        "selector_quotas": {"acquisition": 0.6, "boundary": 0.2, "uncertainty": 0.2},
        "uncertainty_weight": uncertainty_weight,
        "uncertainty_bonus_cap": 8.0,
    }
