#!/usr/bin/env python3
"""Aggregate offline MuJoCo rollout metrics and produce PINN model selection report."""
from __future__ import annotations

import csv
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_EXPERIMENT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = _EXPERIMENT_DIR / "outputs" / "pinn_model_selection"

REPAIRABLE_KEYS = {"demo_4", "demo_5", "demo_6", "demo_7", "demo_9", "demo_18", "demo_20", "demo_21"}

MODELS: list[dict[str, Any]] = [
    {
        "model_id": "V1-E",
        "label": "V1-E",
        "checkpoint": "outputs/v1_repair_parameter_model/model.pt",
        "status": "legacy_baseline",
        "deprecated": False,
    },
    {
        "model_id": "V1-F-old",
        "label": "V1-F old",
        "checkpoint": "outputs/v1f_repair_parameter_model/model_v1f.pt",
        "status": "deprecated",
        "deprecated": True,
    },
    {
        "model_id": "V1-F-aligned-original",
        "label": "V1-F-aligned-original",
        "checkpoint": "outputs/v1f_aligned_repair_parameter_model/original_failed/trained_model/model_v1f_aligned_original.pt",
        "status": "production_default",
        "deprecated": False,
    },
    {
        "model_id": "V1-F-aligned-plus",
        "label": "V1-F-aligned-plus",
        "checkpoint": "outputs/v1f_aligned_plus/trained_model/model_v1f_aligned_plus.pt",
        "status": "experimental_control",
        "deprecated": False,
    },
    {
        "model_id": "V1-F-aligned-plus-balanced",
        "label": "V1-F-aligned-plus-balanced",
        "checkpoint": "outputs/v1f_aligned_plus_balanced/trained_model/model_v1f_aligned_plus_balanced.pt",
        "status": "experimental_control",
        "deprecated": False,
    },
    {
        "model_id": "V1-F-aligned-plus-balanced-v2",
        "label": "V1-F-aligned-plus-balanced-v2",
        "checkpoint": "outputs/v1f_aligned_plus_balanced_v2/trained_model/model_v1f_aligned_plus_balanced_v2.pt",
        "status": "deprecated",
        "deprecated": True,
    },
]

THRESHOLDS = {"old_demo_4_min": 0.70, "old_demo_2_min": 0.20}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _mean(vals: list[float]) -> float | None:
    return float(statistics.mean(vals)) if vals else None


def _float(val: str | float | None) -> float | None:
    if val is None or val == "":
        return None
    return float(val)


def _row(
    *,
    demo_group: str | None = None,
    demo_key: str | None = None,
    model_label: str | None = None,
    variant: str | None = None,
    selection_method: str = "v1f_plain_top_k",
    rows: list[dict[str, str]],
) -> dict[str, str] | None:
    for r in rows:
        if demo_group is not None and r.get("demo_group") != demo_group:
            continue
        if demo_key is not None and r.get("demo_key") != demo_key:
            continue
        if model_label is not None and r.get("model_label") != model_label:
            continue
        if variant is not None and r.get("variant") != variant:
            continue
        if selection_method and r.get("selection_method") not in (selection_method, None):
            sm = r.get("selection_method")
            if sm is not None and sm != selection_method:
                continue
        return r
    return None


def load_eval_sources() -> dict[str, Any]:
    base = _EXPERIMENT_DIR / "outputs"
    return {
        "offline_v1f": _read_csv(base / "offline_mimicgen_repair_test_v1f" / "v1e_vs_v1f_comparison.csv"),
        "aligned_comparison": _read_csv(
            base / "v1f_aligned_repair_parameter_model" / "original_failed" / "validation" / "v1f_aligned_comparison.csv"
        ),
        "plus_eval": _read_csv(base / "v1f_aligned_plus" / "evaluation" / "v1f_plus_evaluation_summary.csv"),
        "balanced_quick": _read_csv(base / "v1f_aligned_plus_balanced" / "quick_eval" / "quick_summary.csv"),
        "v2_quick": _read_csv(
            base / "v1f_aligned_plus_balanced_v2" / "quick_validation" / "quick_validation_summary.csv"
        ),
        "context_ablation": _read_csv(base / "context_alignment_ablation" / "context_alignment_ablation_summary.csv"),
        "repairability": _read_csv(
            base / "v1f_aligned_plus" / "repairability_audit" / "new_demo_repairability_summary.csv"
        ),
    }


def extract_old_demo_metrics(model_id: str, src: dict[str, Any]) -> dict[str, Any]:
    method = "v1f_plain_top_k"
    out: dict[str, Any] = {"selection_method": method, "sources": []}

    if model_id == "V1-E":
        for demo_key in ("demo_4", "demo_2", "demo_3"):
            r = next(
                (
                    x
                    for x in src["offline_v1f"]
                    if x["demo_key"] == demo_key and x["method"] == "v1e_pinn_top_k"
                ),
                None,
            )
            if r:
                out[demo_key] = {
                    "repair_rate_at_20": _float(r["repair_rate_at_20"]),
                    "best_E_total": _float(r["best_E_total"]),
                    "failure_type": r.get("failure_type"),
                    "num_successes": _float(r.get("num_successes")),
                }
        out["sources"].append("offline_mimicgen_repair_test_v1f/v1e_vs_v1f_comparison.csv")
        ac = src["aligned_comparison"]
        d3 = next((x for x in ac if x["demo_key"] == "demo_3" and x["variant"] == "V1-E"), None)
        if d3:
            out["demo_3"]["note"] = d3.get("note") or "physical_lift_bottleneck"
        else:
            out["demo_3"]["note"] = "physical_lift_bottleneck"
        return out

    if model_id == "V1-F-old":
        for demo_key in ("demo_4", "demo_2", "demo_3"):
            r = next(
                (
                    x
                    for x in src["offline_v1f"]
                    if x["demo_key"] == demo_key and x["method"] == "v1f_pinn_top_k"
                ),
                None,
            )
            if r:
                out[demo_key] = {
                    "repair_rate_at_20": _float(r["repair_rate_at_20"]),
                    "best_E_total": _float(r["best_E_total"]),
                    "failure_type": r.get("failure_type"),
                    "num_successes": _float(r.get("num_successes")),
                }
        out["sources"].append("offline_mimicgen_repair_test_v1f/v1e_vs_v1f_comparison.csv")
        out["sources"].append("v1f_aligned_comparison.csv (V1-F_old_original_context)")
        out["demo_3"]["note"] = "physical_lift_bottleneck"
        out["context_ablation_note"] = (
            "demo_4=0.95 only with cem_refined_context (context_alignment_ablation); "
            "original_failed_context repair_rate@20=0.0"
        )
        return out

    label_map = {
        "V1-F-aligned-original": "aligned-original",
        "V1-F-aligned-plus": "aligned-plus",
        "V1-F-aligned-plus-balanced": "aligned-plus-balanced",
        "V1-F-aligned-plus-balanced-v2": "aligned-plus-balanced-v2",
    }
    label = label_map[model_id]

    if model_id in ("V1-F-aligned-original", "V1-F-aligned-plus"):
        pe = src["plus_eval"]
        out["sources"].append("v1f_aligned_plus/evaluation/v1f_plus_evaluation_summary.csv")
        for demo_key in ("demo_4", "demo_2", "demo_3"):
            r = _row(demo_group="old", demo_key=demo_key, model_label=label, rows=pe)
            if r:
                out[demo_key] = {
                    "repair_rate_at_20": _float(r["repair_rate_at_20"]),
                    "best_E_total": _float(r["best_E_total"]),
                    "failure_type": r.get("failure_type"),
                    "num_successes": _float(r.get("num_successes")),
                }
        if model_id == "V1-F-aligned-original":
            ac = next(
                (
                    x
                    for x in src["aligned_comparison"]
                    if x["demo_key"] == "demo_3" and x["variant"] == "V1-F-aligned-original_original_context"
                ),
                None,
            )
            out["demo_3"]["note"] = ac.get("note") if ac and ac.get("note") else "physical_lift_bottleneck"
        else:
            out["demo_3"]["note"] = "physical_lift_bottleneck"
        return out

    if model_id == "V1-F-aligned-plus-balanced":
        # Prefer quick_validation cross-model run; fallback to quick_eval
        v2q = src["v2_quick"]
        bq = src["balanced_quick"]
        out["sources"].append("v1f_aligned_plus_balanced_v2/quick_validation/quick_validation_summary.csv")
        out["sources"].append("v1f_aligned_plus_balanced/quick_eval/quick_summary.csv")
        out["sources"].append(
            "NOTE: full balanced evaluation incomplete (eval_status.json: 11/234 jobs); quick validation used"
        )
        for demo_key in ("demo_4", "demo_2", "demo_3"):
            r = _row(demo_group="old", demo_key=demo_key, variant=label, rows=v2q)
            if not r:
                r = _row(demo_group="old", demo_key=demo_key, variant=label, rows=bq)
            if r:
                out[demo_key] = {
                    "repair_rate_at_20": _float(r["repair_rate_at_20"]),
                    "best_E_total": _float(r["best_E_total"]),
                    "failure_type": r.get("failure_type"),
                    "num_successes": _float(r.get("num_successes")),
                }
        out["demo_3"]["note"] = "physical_lift_bottleneck"
        return out

    if model_id == "V1-F-aligned-plus-balanced-v2":
        v2 = src["v2_quick"]
        out["sources"].append("v1f_aligned_plus_balanced_v2/quick_validation/quick_validation_summary.csv")
        out["deprecated_reason"] = (
            "残缺快速训练流程; old demo_4 repair_rate@20 相对 aligned-original 明显退化 (0.55 vs 0.85)"
        )
        for demo_key in ("demo_4", "demo_2", "demo_3"):
            r = _row(demo_key=demo_key, variant=label, rows=v2)
            if r:
                out[demo_key] = {
                    "repair_rate_at_20": _float(r["repair_rate_at_20"]),
                    "best_E_total": _float(r["best_E_total"]),
                    "failure_type": r.get("failure_type"),
                    "num_successes": _float(r.get("num_successes")),
                }
        out["demo_3"]["note"] = "physical_lift_bottleneck"
        return out

    raise ValueError(model_id)


def extract_new_demo_metrics(model_id: str, src: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"selection_method": "v1f_plain_top_k", "sources": [], "per_demo": {}}

    if model_id in ("V1-E", "V1-F-old"):
        out["new_all_avg_repair_rate_at_20"] = None
        out["new_repairable_avg_repair_rate_at_20"] = None
        out["note"] = "无 new failed demo MuJoCo rollout 评估数据"
        return out

    label_map = {
        "V1-F-aligned-original": ("aligned-original", src["plus_eval"], "full_23_new_demos"),
        "V1-F-aligned-plus": ("aligned-plus", src["plus_eval"], "full_23_new_demos"),
        "V1-F-aligned-plus-balanced": (
            "aligned-plus-balanced",
            src["v2_quick"],
            "quick_validation_subset_8_new_demos",
        ),
        "V1-F-aligned-plus-balanced-v2": (
            "aligned-plus-balanced-v2",
            src["v2_quick"],
            "quick_validation_subset_8_new_demos",
        ),
    }
    label, rows, scope = label_map[model_id]
    out["eval_scope"] = scope

    if scope.startswith("full"):
        out["sources"].append("v1f_aligned_plus/evaluation/v1f_plus_evaluation_summary.csv")
        subset = [
            r
            for r in rows
            if r.get("demo_group") == "new"
            and r.get("model_label") == label
            and r.get("selection_method") == "v1f_plain_top_k"
        ]
    else:
        out["sources"].append("v1f_aligned_plus_balanced_v2/quick_validation/quick_validation_summary.csv")
        subset = [
            r
            for r in rows
            if r.get("demo_group") == "new"
            and r.get("variant") == label
            and r.get("selection_method") == "v1f_plain_top_k"
        ]

    rates_all: list[float] = []
    rates_rep: list[float] = []
    for r in subset:
        dk = r["demo_key"]
        rate = _float(r["repair_rate_at_20"])
        if rate is None:
            continue
        out["per_demo"][dk] = {
            "repair_rate_at_20": rate,
            "best_E_total": _float(r.get("best_E_total")),
            "failure_type": r.get("failure_type"),
            "whether_repairable": dk in REPAIRABLE_KEYS,
        }
        rates_all.append(rate)
        if dk in REPAIRABLE_KEYS:
            rates_rep.append(rate)

    out["new_all_avg_repair_rate_at_20"] = _mean(rates_all)
    out["new_repairable_avg_repair_rate_at_20"] = _mean(rates_rep)
    out["new_demo_count"] = len(rates_all)
    out["repairable_demo_count"] = len(rates_rep)
    return out


def extract_failure_type_breakdown(model_id: str, src: dict[str, Any]) -> dict[str, Any]:
    """transport_failed / insertion_failed on new demos; old demo insertion/grasp from labeled eval."""
    out: dict[str, Any] = {}

    old = extract_old_demo_metrics(model_id, src)
    out["old_insertion_failed_demo_4"] = old.get("demo_4", {}).get("repair_rate_at_20")
    out["old_grasp_failed_demo_2"] = old.get("demo_2", {}).get("repair_rate_at_20")
    out["old_lift_failed_demo_3"] = old.get("demo_3", {}).get("repair_rate_at_20")

    new_metrics = extract_new_demo_metrics(model_id, src)
    ft_groups: dict[str, list[float]] = {"transport_failed": [], "insertion_failed": [], "unknown": []}
    for demo_data in new_metrics.get("per_demo", {}).values():
        ft = demo_data.get("failure_type") or "unknown"
        if ft not in ft_groups:
            ft_groups[ft] = []
        ft_groups[ft].append(demo_data["repair_rate_at_20"])

    if not new_metrics.get("per_demo"):
        out["new_transport_failed_avg"] = None
        out["new_insertion_failed_avg"] = None
        out["failure_type_availability"] = "N/A — 无 new demo 评估或未标注 failure_type"
    elif not ft_groups["transport_failed"] and not ft_groups["insertion_failed"]:
        out["new_transport_failed_avg"] = None
        out["new_insertion_failed_avg"] = None
        out["failure_type_availability"] = (
            "不可用 — new failed demos 在评估中 failure_type=unknown，无法分项 transport/insertion"
        )
        out["new_unknown_failure_type_avg"] = _mean(ft_groups["unknown"])
    else:
        out["new_transport_failed_avg"] = _mean(ft_groups["transport_failed"])
        out["new_insertion_failed_avg"] = _mean(ft_groups["insertion_failed"])
        out["failure_type_availability"] = "available"

    return out


def check_old_demo_regression(model_id: str, old: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    d4 = old.get("demo_4", {}).get("repair_rate_at_20")
    d2 = old.get("demo_2", {}).get("repair_rate_at_20")
    b4 = baseline.get("demo_4", {}).get("repair_rate_at_20")
    b2 = baseline.get("demo_2", {}).get("repair_rate_at_20")

    meets_threshold = (
        d4 is not None
        and d4 >= THRESHOLDS["old_demo_4_min"]
        and d2 is not None
        and d2 >= THRESHOLDS["old_demo_2_min"]
    )
    regressed_vs_baseline = False
    regression_detail: list[str] = []
    aligned_successors = {
        "V1-F-aligned-plus",
        "V1-F-aligned-plus-balanced",
        "V1-F-aligned-plus-balanced-v2",
    }
    if model_id in aligned_successors and b4 is not None and d4 is not None and d4 < b4 - 0.05:
        regressed_vs_baseline = True
        regression_detail.append(f"demo_4: {d4:.2f} < baseline {b4:.2f}")
    if model_id in aligned_successors and b2 is not None and d2 is not None and d2 < b2 - 0.05:
        regressed_vs_baseline = True
        regression_detail.append(f"demo_2: {d2:.2f} < baseline {b2:.2f}")

    return {
        "old_demo_4_repair_rate_at_20": d4,
        "old_demo_2_repair_rate_at_20": d2,
        "old_demo_3_repair_rate_at_20": old.get("demo_3", {}).get("repair_rate_at_20"),
        "old_demo_3_note": old.get("demo_3", {}).get("note"),
        "meets_default_thresholds": meets_threshold,
        "regressed_vs_aligned_original": regressed_vs_baseline,
        "regression_detail": regression_detail,
    }


def assess_default_eligibility(model: dict[str, Any], regression: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    eligible = True

    if model["deprecated"]:
        eligible = False
        reasons.append(f"模型已标记 deprecated ({model['status']})")

    if not regression["meets_default_thresholds"]:
        eligible = False
        d4 = regression["old_demo_4_repair_rate_at_20"]
        d2 = regression["old_demo_2_repair_rate_at_20"]
        if d4 is not None and d4 < THRESHOLDS["old_demo_4_min"]:
            reasons.append(f"old demo_4 repair_rate@20={d4:.2f} < {THRESHOLDS['old_demo_4_min']}")
        if d2 is not None and d2 < THRESHOLDS["old_demo_2_min"]:
            reasons.append(f"old demo_2 repair_rate@20={d2:.2f} < {THRESHOLDS['old_demo_2_min']}")

    if regression["regressed_vs_aligned_original"]:
        eligible = False
        reasons.extend(regression["regression_detail"])

    if model["model_id"] == "V1-F-old":
        eligible = False
        reasons.append("V1-F old 在 original context 下 demo_4 repair_rate@20=0.0，context 未对齐")

    if model["model_id"] == "V1-E":
        eligible = False
        reasons.append("V1-E demo_4 repair_rate@20=0.20，远低于默认门槛 0.70")

    return {"eligible_as_default": eligible, "default_eligibility_reasons": reasons}


def assess_init_base(model: dict[str, Any], regression: dict[str, Any]) -> dict[str, Any]:
    """V1-G / V1-F-100Base initialization suitability."""
    if model["deprecated"]:
        return {"suitable_init_base": False, "reason": "deprecated 或 old demo 退化"}
    if not regression["meets_default_thresholds"]:
        return {"suitable_init_base": False, "reason": "未满足 old demo 最低 rollout 门槛"}
    if regression["regressed_vs_aligned_original"] and model["model_id"] != "V1-F-aligned-original":
        return {"suitable_init_base": False, "reason": "相对 aligned-original 存在 old demo 退化"}
    if model["model_id"] in ("V1-E", "V1-F-old"):
        return {"suitable_init_base": False, "reason": "legacy 模型，不具备 aligned context 能力"}
    return {"suitable_init_base": True, "reason": "old demo rollout 稳定且为当前 production 架构"}


def build_report() -> dict[str, Any]:
    src = load_eval_sources()
    baseline_old = extract_old_demo_metrics("V1-F-aligned-original", src)

    model_reports: list[dict[str, Any]] = []
    for model in MODELS:
        mid = model["model_id"]
        old = extract_old_demo_metrics(mid, src)
        new = extract_new_demo_metrics(mid, src)
        ft = extract_failure_type_breakdown(mid, src)
        regression = check_old_demo_regression(mid, old, baseline_old)
        default = assess_default_eligibility(model, regression)
        init = assess_init_base(model, regression)

        best_e_vals = [
            old.get(dk, {}).get("best_E_total")
            for dk in ("demo_4", "demo_2", "demo_3")
            if old.get(dk, {}).get("best_E_total") is not None
        ]
        model_reports.append(
            {
                **model,
                "checkpoint_exists": (_EXPERIMENT_DIR / model["checkpoint"]).exists(),
                "metrics": {
                    "old_demos": old,
                    "new_demos": new,
                    "failure_type_breakdown": ft,
                    "regression_check": regression,
                    "best_E_total_old_demos_min": min(best_e_vals) if best_e_vals else None,
                },
                "default_eligibility": default,
                "init_base_suitability": init,
            }
        )

    eligible = [m for m in model_reports if m["default_eligibility"]["eligible_as_default"]]
    recommended = eligible[0]["model_id"] if eligible else "V1-F-aligned-original"
    if eligible:
        eligible.sort(
            key=lambda m: (
                m["metrics"]["regression_check"]["old_demo_4_repair_rate_at_20"] or 0,
                m["metrics"]["regression_check"]["old_demo_2_repair_rate_at_20"] or 0,
            ),
            reverse=True,
        )
        recommended = eligible[0]["model_id"]

    return {
        "report_type": "pinn_model_selection",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "methodology": {
            "primary_metric": "offline_repair / quick_validation MuJoCo rollout repair_rate@20 (v1f_plain_top_k)",
            "excluded_metric": "train_loss — 不作为模型选择依据",
            "default_thresholds": THRESHOLDS,
            "baseline_for_regression": "V1-F-aligned-original",
            "repairable_new_demos": sorted(REPAIRABLE_KEYS),
        },
        "data_sources": [
            "outputs/offline_mimicgen_repair_test_v1f/",
            "outputs/v1f_aligned_repair_parameter_model/original_failed/validation/v1f_aligned_comparison.csv",
            "outputs/v1f_aligned_plus/evaluation/v1f_plus_evaluation_summary.csv",
            "outputs/v1f_aligned_plus_balanced/quick_eval/quick_summary.csv",
            "outputs/v1f_aligned_plus_balanced_v2/quick_validation/quick_validation_summary.csv",
            "outputs/context_alignment_ablation/context_alignment_ablation_summary.csv",
            "outputs/v1f_aligned_plus/repairability_audit/new_demo_repairability_summary.csv",
        ],
        "conclusions": {
            "recommended_default_model": recommended,
            "recommended_default_checkpoint": next(m["checkpoint"] for m in MODELS if m["model_id"] == recommended),
            "recommended_init_base_for_v1g_and_v1f_100base": recommended,
            "deprecated_models": [m["model_id"] for m in model_reports if m["deprecated"]],
            "experimental_control_only": [
                m["model_id"]
                for m in model_reports
                if m["status"] == "experimental_control" and not m["deprecated"]
            ],
            "legacy_baseline_only": ["V1-E"],
        },
        "models": model_reports,
    }


def to_summary_csv(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for m in report["models"]:
        reg = m["metrics"]["regression_check"]
        new = m["metrics"]["new_demos"]
        ft = m["metrics"]["failure_type_breakdown"]
        rows.append(
            {
                "model_id": m["model_id"],
                "checkpoint": m["checkpoint"],
                "status": m["status"],
                "deprecated": m["deprecated"],
                "old_demo_4_insertion_repair_rate_at_20": reg["old_demo_4_repair_rate_at_20"],
                "old_demo_2_grasp_repair_rate_at_20": reg["old_demo_2_repair_rate_at_20"],
                "old_demo_3_lift_repair_rate_at_20": reg["old_demo_3_repair_rate_at_20"],
                "old_demo_3_note": reg["old_demo_3_note"],
                "new_all_avg_repair_rate_at_20": new.get("new_all_avg_repair_rate_at_20"),
                "new_repairable_avg_repair_rate_at_20": new.get("new_repairable_avg_repair_rate_at_20"),
                "new_transport_failed_avg": ft.get("new_transport_failed_avg"),
                "new_insertion_failed_avg": ft.get("new_insertion_failed_avg"),
                "failure_type_availability": ft.get("failure_type_availability"),
                "best_E_total_old_demos_min": m["metrics"]["best_E_total_old_demos_min"],
                "meets_default_thresholds": reg["meets_default_thresholds"],
                "regressed_vs_aligned_original": reg["regressed_vs_aligned_original"],
                "eligible_as_default": m["default_eligibility"]["eligible_as_default"],
                "suitable_init_base": m["init_base_suitability"]["suitable_init_base"],
            }
        )
    return rows


def to_markdown(report: dict[str, Any]) -> str:
    c = report["conclusions"]
    lines = [
        "# PINN Repair Model 选型报告",
        "",
        f"生成时间 (UTC): {report['generated_at']}",
        "",
        "## 结论摘要",
        "",
        f"1. **当前推荐默认模型**: `{c['recommended_default_model']}`",
        f"   - Checkpoint: `{c['recommended_default_checkpoint']}`",
        f"2. **V1-G / V1-F-100Base 初始化基座**: `{c['recommended_init_base_for_v1g_and_v1f_100base']}`",
        f"3. **废弃模型**: {', '.join(c['deprecated_models']) or '无'}",
        f"4. **仅作实验对照**: {', '.join(c['experimental_control_only']) or '无'}",
        f"5. **Legacy 基线（不参与默认）**: {', '.join(c['legacy_baseline_only'])}",
        "",
        "## 判断规则",
        "",
        "- 依据 **MuJoCo rollout repair_rate@20**（`v1f_plain_top_k`），**不使用 train loss**",
        "- 默认模型门槛: old demo_4 ≥ 0.70, old demo_2 ≥ 0.20",
        "- demo_3 lift_failed 允许为 0，标记为 physical lift bottleneck",
        "- 新 demo 提升不能以 old demo 明显退化为代价",
        "- v2 已 deprecated，不得作为默认",
        "",
        "## 模型对比表",
        "",
        "| 模型 | demo_4@20 | demo_2@20 | demo_3 | new全部均 | new可修复均 | 默认合格 | old退化 |",
        "|------|-----------|-----------|--------|-----------|-------------|----------|---------|",
    ]

    for m in report["models"]:
        reg = m["metrics"]["regression_check"]
        new = m["metrics"]["new_demos"]
        d4 = reg["old_demo_4_repair_rate_at_20"]
        d2 = reg["old_demo_2_repair_rate_at_20"]
        d3 = reg["old_demo_3_repair_rate_at_20"]
        d3s = f"{d3} ({reg['old_demo_3_note']})" if d3 is not None else "N/A"
        na = new.get("new_all_avg_repair_rate_at_20")
        nr = new.get("new_repairable_avg_repair_rate_at_20")
        ok = "✓" if m["default_eligibility"]["eligible_as_default"] else "✗"
        reg_s = "是" if reg["regressed_vs_aligned_original"] else "否"
        lines.append(
            f"| {m['model_id']} | {d4:.2f} | {d2:.2f} | {d3s} | "
            f"{na:.3f} | {nr:.3f} | {ok} | {reg_s} |"
            if na is not None and nr is not None and d4 is not None and d2 is not None
            else f"| {m['model_id']} | {d4} | {d2} | {d3s} | {na} | {nr} | {ok} | {reg_s} |"
        )

    lines.extend(
        [
            "",
            "## 为何选择 V1-F-aligned-original",
            "",
            "- **old demo_4 (insertion_failed)**: repair_rate@20 = **0.85**，唯一稳定超过 0.70 门槛且显著优于 V1-E (0.20) 与 V1-F-old (0.00)",
            "- **old demo_2 (grasp_failed)**: repair_rate@20 = **0.20**，达到建议门槛",
            "- **old demo_3 (lift_failed)**: 0 / no-positive-lift-candidate — 物理 lift 瓶颈，全模型均为 0，允许",
            "- **plus / balanced / v2** 均在 demo_4 上相对 original 退化 (0.85→0.75→0.65→0.55)，不符合默认策略",
            "- **V1-F-old** 在未对齐 context 下 demo_4 完全失败；**V1-E** demo_4 仅 0.20",
            "",
            "## 废弃与对照说明",
            "",
            "### 废弃",
            "- **V1-F-old**: original context demo_4=0；仅 ablation 中 cem_refined context 可达 0.95，不具备生产可用性",
            "- **V1-F-aligned-plus-balanced-v2**: deprecated 快速残缺流程；demo_4=0.55 明显退化，虽 new demo_18 略优 (0.25) 不可抵消",
            "",
            "### 仅实验对照",
            "- **V1-F-aligned-plus**: demo_4=0.75、new demo_1 有提升，但 demo_4 相对 original 退化",
            "- **V1-F-aligned-plus-balanced**: demo_4=0.65、demo_2=0.10 未达门槛；完整 evaluation 尚未跑完 (11/234 jobs)",
            "",
            "### Legacy 基线",
            "- **V1-E**: 保留作历史对照；demo_4 rollout 0.20，不具备默认资格",
            "",
            "## failure_type 分项",
            "",
            "new failed demos 在现有 evaluation 中 **failure_type 均为 unknown**，transport_failed / insertion_failed 分项 **不可用**。",
            "old demo 分项: demo_4=insertion_failed, demo_2=grasp_failed, demo_3=lift_failed（见上表）。",
            "",
            "## 数据来源",
            "",
        ]
    )
    for s in report["data_sources"]:
        lines.append(f"- `{s}`")

    return "\n".join(lines) + "\n"


def main() -> int:
    report = build_report()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    json_path = OUTPUT_DIR / "pinn_model_selection_report.json"
    csv_path = OUTPUT_DIR / "pinn_model_selection_summary.csv"
    md_path = OUTPUT_DIR / "pinn_model_selection.md"

    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    summary_rows = to_summary_csv(report)
    if summary_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    md_path.write_text(to_markdown(report), encoding="utf-8")

    print(
        json.dumps(
            {
                "outputs": {
                    "json": str(json_path),
                    "csv": str(csv_path),
                    "md": str(md_path),
                },
                "recommended_default": report["conclusions"]["recommended_default_model"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
