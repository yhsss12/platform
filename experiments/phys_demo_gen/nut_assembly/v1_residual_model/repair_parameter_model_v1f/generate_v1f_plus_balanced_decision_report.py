#!/usr/bin/env python3
"""Task 4 完成后：V1-F-aligned-plus-balanced decision report。"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

DEFAULT_EVAL_REPORT = (
    Path(__file__).resolve().parents[2]
    / "outputs"
    / "v1f_aligned_plus_balanced"
    / "evaluation"
    / "v1f_plus_balanced_evaluation_report.json"
)
DEFAULT_REPAIRABILITY = (
    Path(__file__).resolve().parents[2]
    / "outputs"
    / "v1f_aligned_plus"
    / "repairability_audit"
    / "new_demo_repairability_report.json"
)
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[2]
    / "outputs"
    / "v1f_aligned_plus_balanced"
    / "evaluation"
    / "v1f_plus_balanced_decision_report.json"
)

MODELS = ("aligned-original", "aligned-plus", "aligned-plus-balanced")
METHODS = ("v1f_plain_top_k", "random_top_k", "explicit_top_k")


def _mean(vals: list[float]) -> float | None:
    return float(statistics.mean(vals)) if vals else None


def _get_row(rows: list[dict[str, Any]], **filters: Any) -> dict[str, Any] | None:
    for r in rows:
        if all(r.get(k) == v for k, v in filters.items()):
            return r
    return None


def _filter_rows(rows: list[dict[str, Any]], **filters: Any) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        if all(r.get(k) == v for k, v in filters.items()):
            out.append(r)
    return out


def build_decision_report(
    eval_report: dict[str, Any],
    repairability_report: dict[str, Any],
) -> dict[str, Any]:
    rows = eval_report["results"]
    repair_by_demo = {d["source_demo"]: d for d in repairability_report["per_demo"]}
    repairable_keys = set(repairability_report.get("repairable_demo_keys", []))
    hard_keys = set(repairability_report.get("hard_but_improvable_demo_keys", []))

    # 1. Old demo regression
    old_regression: dict[str, Any] = {}
    for demo_key in ("demo_4", "demo_2", "demo_3"):
        per_model = {}
        for model in MODELS:
            row = _get_row(
                rows,
                demo_group="old",
                demo_key=demo_key,
                model_label=model,
                selection_method="v1f_plain_top_k",
            )
            if row:
                per_model[model] = {
                    "repair_rate_at_20": row["metrics"]["repair_rate_at_20"],
                    "success_at_k": row["metrics"].get("success_at_k", {}),
                    "best_E_total": row["metrics"].get("best_E_total"),
                    "num_successes": row["metrics"].get("num_successes_written"),
                }
        old_regression[demo_key] = per_model

    demo_3_balanced = _get_row(
        rows, demo_group="old", demo_key="demo_3", model_label="aligned-plus-balanced", selection_method="v1f_plain_top_k"
    )
    demo_3_note = "no-positive-lift-candidate" if (
        demo_3_balanced and demo_3_balanced["metrics"].get("num_successes_written", 0) == 0
    ) else "has_success_candidate"

    # 2. New demos aggregate
    new_plain = _filter_rows(rows, demo_group="new", selection_method="v1f_plain_top_k")
    new_by_model: dict[str, list[float]] = defaultdict(list)
    new_repairable_by_model: dict[str, list[float]] = defaultdict(list)
    for r in new_plain:
        new_by_model[r["model_label"]].append(float(r["metrics"]["repair_rate_at_20"]))
        if r["demo_key"] in repairable_keys:
            new_repairable_by_model[r["model_label"]].append(float(r["metrics"]["repair_rate_at_20"]))

    hard_improvement: list[dict[str, Any]] = []
    for demo_key in sorted(hard_keys):
        audit = repair_by_demo.get(demo_key, {})
        per_model_best_e: dict[str, float | None] = {}
        for model in MODELS:
            row = _get_row(
                rows, demo_group="new", demo_key=demo_key, model_label=model, selection_method="v1f_plain_top_k"
            )
            per_model_best_e[model] = float(row["metrics"]["best_E_total"]) if row else None
        hard_improvement.append(
            {
                "demo_key": demo_key,
                "avg_E_before": audit.get("avg_E_before"),
                "rollout_best_E_after": audit.get("best_E_after"),
                "energy_drop_ratio": audit.get("energy_drop_ratio"),
                "eval_best_E_by_model": per_model_best_e,
            }
        )

    # 3. By failure_type
    by_failure_type: dict[str, dict[str, Any]] = {}
    for ft in ("transport_failed", "insertion_failed", "alignment_failed"):
        ft_rows = [r for r in new_plain if r.get("failure_type") == ft]
        by_failure_type[ft] = {
            model: _mean([float(r["metrics"]["repair_rate_at_20"]) for r in ft_rows if r["model_label"] == model])
            for model in MODELS
        }
        by_failure_type[ft]["random"] = _mean(
            [
                float(r["metrics"]["repair_rate_at_20"])
                for r in rows
                if r["demo_group"] == "new"
                and r.get("failure_type") == ft
                and r["selection_method"] == "random_top_k"
                and r["model_label"] == "aligned-plus-balanced"
            ]
        )
        by_failure_type[ft]["explicit_energy"] = _mean(
            [
                float(r["metrics"]["repair_rate_at_20"])
                for r in rows
                if r["demo_group"] == "new"
                and r.get("failure_type") == ft
                and r["selection_method"] == "explicit_top_k"
                and r["model_label"] == "aligned-plus-balanced"
            ]
        )

    # 4. Model comparison table
    model_comparison: dict[str, Any] = {}
    for model in MODELS:
        plain = _filter_rows(rows, model_label=model, selection_method="v1f_plain_top_k")
        old_d4 = _get_row(rows, demo_group="old", demo_key="demo_4", model_label=model, selection_method="v1f_plain_top_k")
        old_d2 = _get_row(rows, demo_group="old", demo_key="demo_2", model_label=model, selection_method="v1f_plain_top_k")
        new_r = [r for r in plain if r["demo_group"] == "new"]
        model_comparison[model] = {
            "old_demo_4_repair_rate_at_20": old_d4["metrics"]["repair_rate_at_20"] if old_d4 else None,
            "old_demo_2_repair_rate_at_20": old_d2["metrics"]["repair_rate_at_20"] if old_d2 else None,
            "new_all_avg_repair_rate_at_20": _mean([float(r["metrics"]["repair_rate_at_20"]) for r in new_r]),
            "new_repairable_avg_repair_rate_at_20": _mean(
                [float(r["metrics"]["repair_rate_at_20"]) for r in new_r if r["demo_key"] in repairable_keys]
            ),
        }
    for method in ("random_top_k", "explicit_top_k"):
        label = "random" if method == "random_top_k" else "explicit_energy"
        subset = _filter_rows(rows, selection_method=method, model_label="aligned-plus-balanced")
        new_r = [r for r in subset if r["demo_group"] == "new"]
        model_comparison[label] = {
            "new_all_avg_repair_rate_at_20": _mean([float(r["metrics"]["repair_rate_at_20"]) for r in new_r]),
            "new_repairable_avg_repair_rate_at_20": _mean(
                [float(r["metrics"]["repair_rate_at_20"]) for r in new_r if r["demo_key"] in repairable_keys]
            ),
        }

    # 5. Final decision
    d4_orig = old_regression.get("demo_4", {}).get("aligned-original", {}).get("repair_rate_at_20")
    d4_bal = old_regression.get("demo_4", {}).get("aligned-plus-balanced", {}).get("repair_rate_at_20")
    d2_orig = old_regression.get("demo_2", {}).get("aligned-original", {}).get("repair_rate_at_20")
    d2_bal = old_regression.get("demo_2", {}).get("aligned-plus-balanced", {}).get("repair_rate_at_20")
    new_all_orig = model_comparison["aligned-original"]["new_all_avg_repair_rate_at_20"]
    new_all_bal = model_comparison["aligned-plus-balanced"]["new_all_avg_repair_rate_at_20"]
    new_all_plus = model_comparison["aligned-plus"]["new_all_avg_repair_rate_at_20"]
    new_rep_bal = model_comparison["aligned-plus-balanced"]["new_repairable_avg_repair_rate_at_20"]
    new_rep_random = model_comparison.get("random", {}).get("new_repairable_avg_repair_rate_at_20")

    old_ok = (d4_bal is not None and d4_bal >= 0.70) and (d2_bal is not None and d2_bal >= 0.20)
    new_improved = (
        new_all_bal is not None
        and new_all_orig is not None
        and new_all_bal > new_all_orig
        and (new_rep_bal or 0) > (new_rep_random or 0)
    )
    transport_bal = by_failure_type.get("transport_failed", {}).get("aligned-plus-balanced")
    transport_plus = by_failure_type.get("transport_failed", {}).get("aligned-plus")
    insertion_bal = by_failure_type.get("insertion_failed", {}).get("aligned-plus-balanced")
    insertion_plus = by_failure_type.get("insertion_failed", {}).get("aligned-plus")
    ft_improved = (
        (transport_bal is not None and transport_plus is not None and transport_bal > transport_plus)
        or (insertion_bal is not None and insertion_plus is not None and insertion_bal > insertion_plus)
    )

    can_be_default = old_ok and new_improved and ft_improved

    reasons: list[str] = []
    if not old_ok:
        if d4_bal is not None and d4_bal < 0.70:
            reasons.append(f"old demo_4 regression: balanced repair_rate@20={d4_bal:.3f} < 0.70")
        if d2_bal is not None and d2_bal < 0.20:
            reasons.append(f"old demo_2 regression: balanced repair_rate@20={d2_bal:.3f} < 0.20")
    if not new_improved:
        reasons.append(
            f"new demo limited gain: balanced all={new_all_bal}, original={new_all_orig}, plus={new_all_plus}"
        )
    if not ft_improved:
        reasons.append("transport/insertion not clearly better than plus under balanced model")

    if can_be_default:
        recommendation = "balanced_can_be_default"
        root_cause = None
    elif not old_ok:
        recommendation = "keep_aligned_original_as_default"
        root_cause = "old_demo_regression"
    elif not new_improved:
        recommendation = "keep_aligned_original_as_default"
        root_cause = "new_demo_no_ranking_gain"
    else:
        recommendation = "keep_aligned_original_as_default"
        root_cause = "sampler_refiner_coverage_or_failure_type_gap"

    return {
        "task": "v1f_plus_balanced_decision_report",
        "default_model_policy": "do_not_change_default_until_explicit_approval",
        "recommended_default_model": "model_v1f_aligned_original.pt",
        "balanced_can_be_default": can_be_default,
        "recommendation": recommendation,
        "root_cause_if_not_default": root_cause,
        "reasons": reasons,
        "1_old_demo_regression_check": {
            "demo_4_repair_rate_at_20_by_model": {
                m: old_regression.get("demo_4", {}).get(m, {}).get("repair_rate_at_20") for m in MODELS
            },
            "demo_2_repair_rate_at_20_by_model": {
                m: old_regression.get("demo_2", {}).get(m, {}).get("repair_rate_at_20") for m in MODELS
            },
            "demo_3_status": demo_3_note,
            "thresholds": {"demo_4_min": 0.70, "demo_2_min": 0.20},
        },
        "2_new_demos_summary": {
            "all_new_failed_avg_repair_rate_at_20_by_model": {
                m: model_comparison[m]["new_all_avg_repair_rate_at_20"] for m in MODELS
            },
            "repairable_avg_repair_rate_at_20_by_model": {
                m: model_comparison[m]["new_repairable_avg_repair_rate_at_20"] for m in MODELS
            },
            "hard_but_improvable_energy": hard_improvement,
        },
        "3_by_failure_type": by_failure_type,
        "4_model_comparison": model_comparison,
        "5_final_decision": {
            "balanced_can_be_default": can_be_default,
            "recommendation": recommendation,
            "root_cause_if_not_default": root_cause,
            "explicit_note": "Plus model also remains non-default; aligned-original stays production default.",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate V1-F-plus-balanced decision report")
    parser.add_argument("--eval-report", type=Path, default=DEFAULT_EVAL_REPORT)
    parser.add_argument("--repairability-report", type=Path, default=DEFAULT_REPAIRABILITY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.eval_report.exists():
        raise SystemExit(f"Eval report not found: {args.eval_report}")

    eval_report = json.loads(args.eval_report.read_text(encoding="utf-8"))
    repairability = json.loads(args.repairability_report.read_text(encoding="utf-8"))
    decision = build_decision_report(eval_report, repairability)
    args.output.write_text(json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "recommendation": decision["recommendation"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
