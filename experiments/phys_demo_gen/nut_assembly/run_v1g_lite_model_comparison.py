#!/usr/bin/env python3
"""V1-G-lite vs aligned-original：multi-seed rollout 对比 + 验收判定。"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_EXPERIMENT_DIR = Path(__file__).resolve().parent
_OFFLINE_DIR = _EXPERIMENT_DIR / "offline_mimicgen_repair_test"
_V1F_DIR = _EXPERIMENT_DIR / "v1_residual_model" / "repair_parameter_model_v1f"
for path in (_EXPERIMENT_DIR, _OFFLINE_DIR, _V1F_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config import (  # noqa: E402
    DEFAULT_CEM_REPORT,
    DEFAULT_FAILED_HDF5,
    DEFAULT_PINN_MODEL,
    DEFAULT_SUCCESS_REFERENCE_JSONL,
    DEMO_3_V1G_LITE_DIAGNOSTIC,
)
from physics_residual_repair import is_physics_residual_repair_enabled  # noqa: E402
from run_insertion_gated_multi_seed_validation import (  # noqa: E402
    DEFAULT_SEEDS,
    STRATEGIES,
    _aggregate_strategy,
    run_single_config,
)
from v1f_plus_utils import DEFAULT_ALIGNED_MODEL  # noqa: E402

DEFAULT_OUT_DIR = _EXPERIMENT_DIR / "outputs" / "v1g_stage1_lite_p1p2"
REGRESSION_THRESHOLD_PP = 0.05


def _strategy_metrics_from_run(run: dict[str, Any], strategy: str) -> dict[str, Any]:
    s = run["strategies"][strategy]
    summary_like = {
        "partial_success_rate": s["partial_success_rate"],
        "final_success_rate": s["final_success_rate"],
        "transport_success_rate": s["transport_success_rate"],
        "xy_alignment_success_rate": s["xy_alignment_success_rate"],
        "lift_success_rate": s["lift_success_rate"],
    }
    reject_reasons = s.get("insertion_gate_reject_reasons", {})
    insertion_gate_reject_count = int(sum(reject_reasons.values())) if reject_reasons else 0
    return {
        "selected_count": s["selected_count"],
        "gate_pass_rate": s["gate_pass_rate"],
        "partial_success": summary_like["partial_success_rate"],
        "final_success": summary_like["final_success_rate"],
        "transport_success": summary_like["transport_success_rate"],
        "xy_success": summary_like["xy_alignment_success_rate"],
        "lift_success": summary_like["lift_success_rate"],
        "E_transport_improved_rate": _infer_improved_rate(s, "transport"),
        "E_xy_improved_rate": _infer_improved_rate(s, "xy"),
        "E_lift_degraded_rate": _infer_lift_degraded_rate(s),
        "mean_raw_total_score": s.get("mean_raw_total_score", 0.0),
        "mean_ranking_score": s.get("mean_ranking_score", 0.0),
        "insertion_gate_reject_count": insertion_gate_reject_count,
        "insertion_gate_reject_reasons": reject_reasons,
        "failure_reason_cluster": s.get("failure_reason_counts", {}),
    }


def _infer_improved_rate(strategy_result: dict[str, Any], kind: str) -> float:
    """从 multi_seed run 字段推断改善率（若缺失则返回 0）。"""
    key = f"E_{kind}_improved_rate"
    if key in strategy_result:
        return float(strategy_result[key])
    imp_key = f"E_{kind}_improvement"
    block = strategy_result.get(imp_key)
    if isinstance(block, dict) and "rate" in block:
        return float(block["rate"])
    return 0.0


def _infer_lift_degraded_rate(strategy_result: dict[str, Any]) -> float:
    if "E_lift_degraded_rate" in strategy_result:
        return float(strategy_result["E_lift_degraded_rate"])
    block = strategy_result.get("E_lift_degraded")
    if isinstance(block, dict) and "rate" in block:
        return float(block["rate"])
    return 0.0


def run_model_rollout_validation(
    *,
    model_label: str,
    model_path: Path,
    demos: list[str],
    num_samples: int,
    top_k: int,
    seeds: list[int],
    failed_hdf5: Path,
    cem_report: Path,
    v1e_model: Path,
    success_reference: Path,
) -> dict[str, Any]:
    all_runs: list[dict[str, Any]] = []
    for demo_key in demos:
        for seed in seeds:
            print(
                f"[rollout] {model_label} {demo_key} n={num_samples} k={top_k} seed={seed}",
                flush=True,
            )
            run = run_single_config(
                demo_key=demo_key,
                num_samples=num_samples,
                top_k=top_k,
                seed=seed,
                failed_hdf5=failed_hdf5,
                cem_report=cem_report,
                aligned_model=model_path,
                v1e_model=v1e_model,
                success_reference_jsonl=success_reference,
            )
            run["model_label"] = model_label
            run["model_path"] = str(model_path)
            all_runs.append(run)

    aggregates: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    per_cell: dict[str, dict[str, dict[str, Any]]] = defaultdict(lambda: defaultdict(dict))

    for demo_key in demos:
        demo_runs = [r for r in all_runs if r["demo_key"] == demo_key]
        aggregates[demo_key] = {strat: _aggregate_strategy(demo_runs, strat) for strat in STRATEGIES}
        for strat in STRATEGIES:
            per_cell[demo_key][strat] = {
                "runs": [_strategy_metrics_from_run(r, strat) for r in demo_runs],
                "aggregate": aggregates[demo_key][strat],
            }

    return {
        "schema": "v1g_lite_rollout_validation_v1",
        "model_label": model_label,
        "model_path": str(model_path),
        "demos": demos,
        "num_samples": num_samples,
        "top_k": top_k,
        "seeds": seeds,
        "strategies": list(STRATEGIES),
        "runs": all_runs,
        "aggregates": aggregates,
        "metrics_by_demo_strategy": per_cell,
    }


def _mean_final(payload: dict[str, Any], demo: str, strategy: str) -> float:
    return float(payload["aggregates"][demo][strategy]["mean_final_success_rate"])


def _mean_partial(payload: dict[str, Any], demo: str, strategy: str) -> float:
    return float(payload["aggregates"][demo][strategy]["mean_partial_success_rate"])


def evaluate_acceptance(
    *,
    aligned: dict[str, Any],
    v1g: dict[str, Any],
    checkpoint_integrity_ok: bool,
) -> dict[str, Any]:
    regressions: list[dict[str, Any]] = []
    demo2_improvements: list[dict[str, Any]] = []

    for strategy in STRATEGIES:
        aligned_d4 = _mean_final(aligned, "demo_4", strategy)
        v1g_d4 = _mean_final(v1g, "demo_4", strategy)
        delta_d4 = v1g_d4 - aligned_d4
        if delta_d4 < -REGRESSION_THRESHOLD_PP:
            regressions.append(
                {
                    "demo": "demo_4",
                    "strategy": strategy,
                    "aligned_final": aligned_d4,
                    "v1g_final": v1g_d4,
                    "delta_pp": delta_d4,
                    "severity": "regression",
                }
            )

        aligned_d2_final = _mean_final(aligned, "demo_2", strategy)
        v1g_d2_final = _mean_final(v1g, "demo_2", strategy)
        aligned_d2_partial = _mean_partial(aligned, "demo_2", strategy)
        v1g_d2_partial = _mean_partial(v1g, "demo_2", strategy)
        residual_better = (
            v1g["aggregates"]["demo_2"][strategy].get("mean_gate_pass_rate", 0)
            >= aligned["aggregates"]["demo_2"][strategy].get("mean_gate_pass_rate", 0)
        )
        demo2_improvements.append(
            {
                "strategy": strategy,
                "delta_final": v1g_d2_final - aligned_d2_final,
                "delta_partial": v1g_d2_partial - aligned_d2_partial,
                "residual_or_gate_better": residual_better,
            }
        )

    demo2_better = any(
        imp["delta_final"] > 0 or imp["delta_partial"] > 0 or imp["residual_or_gate_better"]
        for imp in demo2_improvements
    )
    demo4_not_worse = not regressions
    candidate_ok = (
        checkpoint_integrity_ok
        and demo2_better
        and demo4_not_worse
    )

    insertion_effective = _mean_final(v1g, "demo_4", "physics_residual_insertion_gated_top_k") >= _mean_final(
        v1g, "demo_4", "v1f_plain_top_k"
    )

    return {
        "checkpoint_integrity_ok": checkpoint_integrity_ok,
        "demo_2_improvement_detected": demo2_better,
        "demo_2_details": demo2_improvements,
        "demo_4_regression_detected": bool(regressions),
        "demo_4_regressions": regressions,
        "regression_threshold_pp": REGRESSION_THRESHOLD_PP,
        "insertion_gate_effective_on_v1g": insertion_effective,
        "passes_candidate_criteria": candidate_ok,
        "recommend_candidate_model": candidate_ok,
        "recommend_replace_aligned_original": False,
        "aligned_original_remains_default": True,
        "v1g_lite_status": "experimental_candidate" if candidate_ok else "experimental_opt_in",
    }


def build_comparison_table(
    aligned: dict[str, Any],
    v1g: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for demo in aligned["demos"]:
        for strategy in STRATEGIES:
            a_agg = aligned["aggregates"][demo][strategy]
            v_agg = v1g["aggregates"][demo][strategy]
            rows.append(
                {
                    "demo": demo,
                    "strategy": strategy,
                    "aligned_mean_final": a_agg["mean_final_success_rate"],
                    "v1g_mean_final": v_agg["mean_final_success_rate"],
                    "delta_final": v_agg["mean_final_success_rate"] - a_agg["mean_final_success_rate"],
                    "aligned_mean_partial": a_agg["mean_partial_success_rate"],
                    "v1g_mean_partial": v_agg["mean_partial_success_rate"],
                    "delta_partial": v_agg["mean_partial_success_rate"] - a_agg["mean_partial_success_rate"],
                    "aligned_mean_gate_pass": a_agg["mean_gate_pass_rate"],
                    "v1g_mean_gate_pass": v_agg["mean_gate_pass_rate"],
                }
            )
    return rows


def write_comparison_md(
    *,
    payload: dict[str, Any],
    path: Path,
) -> None:
    acceptance = payload["acceptance"]
    table = payload["comparison_table"]
    lines = [
        "# V1-G-stage1-lite-p1p2 vs aligned-original 对比报告",
        "",
        "## 模型",
        "",
        f"- aligned-original: `{payload['models']['aligned_original']}`",
        f"- V1-G-stage1-lite-p1p2: `{payload['models']['v1g_lite']}`",
        f"- checkpoint 完整性: {'通过' if acceptance['checkpoint_integrity_ok'] else '失败'}",
        "",
        "## 对比表（mean final / partial，multi-seed 聚合）",
        "",
        "| demo | strategy | aligned final | v1g final | Δfinal | aligned partial | v1g partial | Δpartial |",
        "|------|----------|---------------|-----------|--------|-----------------|-------------|----------|",
    ]
    for row in table:
        lines.append(
            f"| {row['demo']} | {row['strategy']} | "
            f"{row['aligned_mean_final']:.0%} | {row['v1g_mean_final']:.0%} | {row['delta_final']:+.0%} | "
            f"{row['aligned_mean_partial']:.0%} | {row['v1g_mean_partial']:.0%} | {row['delta_partial']:+.0%} |"
        )

    lines.extend(
        [
            "",
            "## demo_3 诊断标签（不参与主验收）",
            "",
            f"- repairability: `{DEMO_3_V1G_LITE_DIAGNOSTIC['repairability']}`",
            f"- failure_stage: `{DEMO_3_V1G_LITE_DIAGNOSTIC['failure_stage']}`",
            f"- failure_reason: `{DEMO_3_V1G_LITE_DIAGNOSTIC['failure_reason']}`",
            "",
            "## 最终结论",
            "",
        ]
    )

    has_benefit = acceptance["demo_2_improvement_detected"]
    has_regression = acceptance["demo_4_regression_detected"]
    insertion_ok = acceptance["insertion_gate_effective_on_v1g"]

    if has_benefit and not has_regression:
        conclusion = (
            "V1-G-stage1-lite-p1p2 可作为 experimental candidate model 继续扩大验证，"
            "但 aligned-original 仍为默认模型。"
        )
    elif has_regression:
        conclusion = (
            "V1-G-stage1-lite-p1p2 不替换 aligned-original，仅保留为实验分支；"
            "后续需要继续提高 retention 或进一步降低 physics loss 权重。"
        )
    else:
        conclusion = (
            "V1-G-stage1-lite-p1p2 未明显优于 aligned-original，保留为 experimental / opt-in；"
            "aligned-original 继续为默认模型。"
        )

    lines.extend(
        [
            f"1. **相比 aligned-original 是否有收益**: {'是（主要在 demo_2）' if has_benefit else '否或有限'}",
            f"2. **收益主要体现在**: demo_2 transport/xy/lift/partial/final（见上表）",
            f"3. **demo_4 regression**: {'是' if has_regression else '否'}",
            f"4. **insertion gate 在新模型下是否仍有效**: {'是' if insertion_ok else '否或不稳定'}",
            f"5. **是否建议继续训练**: {'可继续轻量迭代' if has_regression else '可扩大验证' if has_benefit else '暂缓，先调 retention/权重'}",
            f"6. **是否建议替换 aligned-original**: 否（无论验收结果）",
            f"7. **下一步建议**: "
            + (
                "扩大 demo_2/demo_4 multi-seed 池验证；保持 insertion gate opt-in。"
                if has_benefit and not has_regression
                else "提高 lambda_retention 或降低 physics λ；复跑 lite fine-tune。"
                if has_regression
                else "维持 aligned-original 默认；V1-G-lite 仅 opt-in 对比。"
            ),
            "",
            f"**总评**: {conclusion}",
            "",
            f"- 验收通过 candidate 标准: {acceptance['passes_candidate_criteria']}",
            f"- V1-G-lite 状态: `{acceptance['v1g_lite_status']}`",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="V1-G-lite model comparison")
    parser.add_argument("--aligned-model", type=Path, default=DEFAULT_ALIGNED_MODEL)
    parser.add_argument("--v1g-model", type=Path, required=True)
    parser.add_argument("--failed-hdf5", type=Path, default=DEFAULT_FAILED_HDF5)
    parser.add_argument("--cem-report", type=Path, default=DEFAULT_CEM_REPORT)
    parser.add_argument("--v1e-model", type=Path, default=DEFAULT_PINN_MODEL)
    parser.add_argument("--success-reference", type=Path, default=DEFAULT_SUCCESS_REFERENCE_JSONL)
    parser.add_argument("--demos", nargs="+", default=["demo_2", "demo_4"])
    parser.add_argument("--num-samples", type=int, default=400)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--checkpoint-integrity", type=Path, default=None)
    parser.add_argument("--enable-physics-residual-repair", action="store_true")
    args = parser.parse_args()

    if args.enable_physics_residual_repair:
        os.environ["enable_physics_residual_repair"] = "true"
    if not is_physics_residual_repair_enabled():
        print("ERROR: --enable-physics-residual-repair required", flush=True)
        return 2

    os.environ.setdefault("MUJOCO_GL", "egl")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    integrity_path = args.checkpoint_integrity or (args.output_dir / "checkpoint_integrity.json")
    integrity = json.loads(integrity_path.read_text(encoding="utf-8")) if integrity_path.exists() else {}
    integrity_ok = bool(integrity.get("aligned_original_unchanged", False))

    aligned_payload = run_model_rollout_validation(
        model_label="aligned-original",
        model_path=args.aligned_model,
        demos=args.demos,
        num_samples=args.num_samples,
        top_k=args.top_k,
        seeds=args.seeds,
        failed_hdf5=args.failed_hdf5,
        cem_report=args.cem_report,
        v1e_model=args.v1e_model,
        success_reference=args.success_reference,
    )
    v1g_payload = run_model_rollout_validation(
        model_label="V1-G-stage1-lite-p1p2",
        model_path=args.v1g_model,
        demos=args.demos,
        num_samples=args.num_samples,
        top_k=args.top_k,
        seeds=args.seeds,
        failed_hdf5=args.failed_hdf5,
        cem_report=args.cem_report,
        v1e_model=args.v1e_model,
        success_reference=args.success_reference,
    )

    aligned_out = args.output_dir / "rollout_validation_aligned_original.json"
    v1g_out = args.output_dir / "rollout_validation_v1g_lite.json"
    aligned_out.write_text(json.dumps(aligned_payload, indent=2, default=str), encoding="utf-8")
    v1g_out.write_text(json.dumps(v1g_payload, indent=2, default=str), encoding="utf-8")

    acceptance = evaluate_acceptance(
        aligned=aligned_payload,
        v1g=v1g_payload,
        checkpoint_integrity_ok=integrity_ok,
    )
    comparison_table = build_comparison_table(aligned_payload, v1g_payload)

    report = {
        "schema": "v1g_lite_model_comparison_v1",
        "models": {
            "aligned_original": str(args.aligned_model),
            "v1g_lite": str(args.v1g_model),
        },
        "validation": {
            "demos": args.demos,
            "num_samples": args.num_samples,
            "top_k": args.top_k,
            "seeds": args.seeds,
            "strategies": list(STRATEGIES),
        },
        "comparison_table": comparison_table,
        "acceptance": acceptance,
        "model_summary": {
            "demo_2_v1g_better_than_aligned": acceptance["demo_2_improvement_detected"],
            "demo_4_v1g_not_below_aligned": not acceptance["demo_4_regression_detected"],
            "regression_present": acceptance["demo_4_regression_detected"],
            "regression_locations": acceptance["demo_4_regressions"],
            "recommend_v1g_as_candidate": acceptance["recommend_candidate_model"],
            "keep_aligned_as_default": True,
        },
        "outputs": {
            "rollout_validation_aligned_original": str(aligned_out),
            "rollout_validation_v1g_lite": str(v1g_out),
        },
    }

    json_path = args.output_dir / "model_comparison_report.json"
    md_path = args.output_dir / "model_comparison_report.md"
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    write_comparison_md(payload=report, path=md_path)
    print(json.dumps({"json": str(json_path), "md": str(md_path), "acceptance": acceptance}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
