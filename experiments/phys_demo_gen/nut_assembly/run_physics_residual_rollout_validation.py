#!/usr/bin/env python3
"""Physics residual repair rollout 验证：demo_2/demo_4 正式对比 + demo_3 诊断。"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
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
    DEFAULT_ROLLOUT_VALIDATION_JSON,
    DEFAULT_ROLLOUT_VALIDATION_MD,
    DEFAULT_ROLLOUT_VALIDATION_OUTPUT_DIR,
    DEFAULT_SUCCESS_REFERENCE_JSONL,
    DEMO_REPAIR_CONFIGS,
)
from physics_residual_repair import (  # noqa: E402
    build_candidate_record,
    build_physics_repair_context,
    is_physics_residual_repair_enabled,
    select_indices_by_insertion_gated_ranking,
    select_indices_by_ranking_score,
    summarize_rollout_strategy,
)
from physics_residuals import (
    candidate_beats_original,
    candidate_passes_p1p2_gate,
    check_source_consistency,
    compute_physics_residuals,
)
from repair_common_v1f import (  # noqa: E402
    extract_baseline_context_v1f,
    sample_repair_candidates_v1f,
    score_repair_candidates_v1f,
    select_candidate_indices_v1f,
)
from repair_rollout import run_repair_rollout  # noqa: E402
from rollout_outcome_evaluator import evaluate_rollout_outcome  # noqa: E402
from run_physics_residual_repair_validation import _run_original_baseline_rollout  # noqa: E402
from v1f_plus_utils import DEFAULT_ALIGNED_MODEL  # noqa: E402

FORMAL_ROLLOUT_DEMOS = ("demo_2", "demo_4")
DIAGNOSTIC_DEMO = "demo_3"

STRATEGY_SPECS: dict[str, dict[str, Any]] = {
    "aligned-original": {
        "selection": "v1f_plain_top_k",
        "use_physics_gate": False,
        "gate_mode": None,
        "formal": True,
    },
    "physics_residual_top_k": {
        "selection": "physics_ranking",
        "use_physics_gate": False,
        "gate_mode": None,
        "formal": True,
    },
    "physics_residual_gated_top_k": {
        "selection": "physics_ranking",
        "use_physics_gate": True,
        "gate_mode": "full",
        "formal": True,
        "primary_rollout": True,
    },
    "physics_residual_p1p2_gated_top_k": {
        "selection": "physics_ranking",
        "use_physics_gate": True,
        "gate_mode": "p1p2",
        "formal": False,
        "diagnostic_only": True,
    },
    "physics_residual_insertion_gated_top_k": {
        "selection": "insertion_gated_ranking",
        "use_physics_gate": True,
        "gate_mode": "p1p2_insertion",
        "formal": False,
        "diagnostic_only": True,
    },
}


def _select_indices_for_strategy(
    *,
    strategy: str,
    candidates: list[dict[str, Any]],
    pinn_top: list[int],
    context: dict[str, Any],
    original_breakdown: dict[str, Any],
    original_rollout: dict[str, Any],
    top_k: int,
) -> list[int]:
    spec = STRATEGY_SPECS[strategy]
    if spec["selection"] == "v1f_plain_top_k":
        return list(pinn_top[:top_k])
    if spec["selection"] == "insertion_gated_ranking":
        selected, _ = select_indices_by_insertion_gated_ranking(
            candidates,
            context=context,
            top_k=top_k,
            original_breakdown=original_breakdown,
            original_rollout=original_rollout,
        )
        return selected
    return select_indices_by_ranking_score(
        candidates,
        context=context,
        top_k=top_k,
        require_gate=bool(spec["use_physics_gate"]),
        original_breakdown=original_breakdown,
        gate_mode=str(spec.get("gate_mode") or "full"),
    )


def _build_rollout_record(
    *,
    demo_key: str,
    strategy: str,
    rank: int,
    candidate_index: int,
    rollout: dict[str, Any],
    breakdown: dict[str, Any],
    original_breakdown: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    outcome = evaluate_rollout_outcome(rollout, context)
    base = build_candidate_record(
        label=f"{strategy}_{rank:02d}",
        demo_key=demo_key,
        strategy=strategy,
        rollout=rollout,
        breakdown=breakdown,
        original_breakdown=original_breakdown,
        candidate_index=candidate_index,
    )
    spec = STRATEGY_SPECS.get(strategy, {})
    if spec.get("gate_mode") == "p1p2":
        src = check_source_consistency(breakdown, original_breakdown)
        passed, gate_checks = candidate_passes_p1p2_gate(
            breakdown, original_breakdown, source_consistency=src
        )
        base["physics_gate_passed"] = passed
        base["physics_gate_checks"] = gate_checks
        base["gate_mode"] = "p1p2"
    base.update(outcome)
    return base


def run_demo_rollout_validation(
    *,
    demo_key: str,
    cfg: dict[str, Any],
    failed_hdf5: Path,
    cem_report: Path,
    aligned_model: Path,
    v1e_model: Path,
    success_reference_jsonl: Path,
    strategies: tuple[str, ...],
    num_samples: int,
    top_k: int,
    seed: int,
) -> dict[str, Any]:
    base_context = extract_baseline_context_v1f(
        failed_hdf5=failed_hdf5,
        demo_key=demo_key,
        failure_type=cfg["failure_type"],
        search_kind=cfg["search_kind"],
    )
    context = build_physics_repair_context(
        base_context=base_context,
        success_reference_jsonl=success_reference_jsonl,
    )

    original_rollout = _run_original_baseline_rollout(
        demo_key=demo_key,
        cfg=cfg,
        failed_hdf5=failed_hdf5,
    )
    original_breakdown = compute_physics_residuals(original_rollout, context)
    original_outcome = evaluate_rollout_outcome(original_rollout, context)

    candidates = sample_repair_candidates_v1f(
        search_kind=cfg["search_kind"],
        n_samples=num_samples,
        seed=seed + hash(demo_key) % 10000,
    )
    score_repair_candidates_v1f(
        context=base_context,
        candidates=candidates,
        active=cfg["active"],
        v1e_model_path=v1e_model,
        v1f_model_path=aligned_model,
    )
    pinn_top = select_candidate_indices_v1f(
        candidates,
        method="v1f_plain_top_k",
        top_k=top_k,
        rng=random.Random(seed),
    )

    for idx in pinn_top:
        if candidates[idx].get("rollout"):
            continue
        candidates[idx]["rollout"] = run_repair_rollout(
            failed_hdf5=failed_hdf5,
            demo_key=demo_key,
            search_kind=cfg["search_kind"],
            cem_report=cem_report if cfg["search_kind"] in ("insertion", "transport") else None,
            candidate=candidates[idx],
        )

    strategy_records: dict[str, list[dict[str, Any]]] = {}
    strategy_summary: dict[str, Any] = {}

    for strategy in strategies:
        selected = _select_indices_for_strategy(
            strategy=strategy,
            candidates=candidates,
            pinn_top=pinn_top,
            context=context,
            original_breakdown=original_breakdown,
            original_rollout=original_rollout,
            top_k=top_k,
        )
        records: list[dict[str, Any]] = []
        for rank, idx in enumerate(selected, start=1):
            rollout = candidates[idx]["rollout"]
            br = compute_physics_residuals(rollout, context)
            records.append(
                _build_rollout_record(
                    demo_key=demo_key,
                    strategy=strategy,
                    rank=rank,
                    candidate_index=idx,
                    rollout=rollout,
                    breakdown=br,
                    original_breakdown=original_breakdown,
                    context=context,
                )
            )
        strategy_records[strategy] = records
        strategy_summary[strategy] = summarize_rollout_strategy(
            demo_key=demo_key,
            strategy=strategy,
            rollout_records=records,
            original_breakdown=original_breakdown,
        )
        print(
            f"[rollout] {demo_key}/{strategy} n={len(records)} "
            f"partial={strategy_summary[strategy]['partial_success_rate']:.0%} "
            f"final={strategy_summary[strategy]['final_success_rate']:.0%}",
            flush=True,
        )

    return {
        "demo_key": demo_key,
        "failure_type": cfg["failure_type"],
        "original_baseline": {
            **original_outcome,
            "raw_total_score": original_breakdown["raw_total_score"],
            "ranking_score": original_breakdown["ranking_score"],
        },
        "strategies": strategy_summary,
        "records": strategy_records,
    }


def generate_rollout_validation_report_md(
    *,
    results: dict[str, Any],
    output_path: Path,
) -> None:
    lines = [
        "# Physics Residual Repair Rollout 验证报告",
        "",
        "- aligned-original checkpoint: **未修改**",
        "- physics residual repair: **显式启用**（非默认）",
        "- 正式 rollout: demo_2, demo_4",
        "- demo_3: 仅 `physics_residual_p1p2_gated_top_k` 诊断",
        "",
        "## 1. Original Baseline",
        "",
    ]

    for demo_key, demo_res in results.get("formal", {}).items():
        ob = demo_res["original_baseline"]
        lines.append(
            f"- **{demo_key}**: transport={ob['transport_success']}, xy={ob['xy_alignment_success']}, "
            f"partial={ob['partial_success']}, final={ob['final_success']}, "
            f"reason={ob['failure_reason']}, raw_score={ob['raw_total_score']:.3f}"
        )
    lines.append("")

    lines.extend(["## 2. 三策略对比（demo_2 / demo_4）", ""])
    for demo_key in FORMAL_ROLLOUT_DEMOS:
        demo_res = results.get("formal", {}).get(demo_key, {})
        lines.append(f"### {demo_key}")
        lines.append("")
        lines.append(
            "| 策略 | partial | final | transport | xy | lift | E_transport↑ | E_xy↑ | E_lift↑ | mean raw | mean rank |"
        )
        lines.append("|------|---------|-------|-----------|----|----|-------------|--------|---------|----------|-----------|")
        for strategy in ("aligned-original", "physics_residual_top_k", "physics_residual_gated_top_k"):
            s = demo_res.get("strategies", {}).get(strategy, {})
            if not s:
                continue
            lines.append(
                f"| {strategy} | {s.get('partial_success_rate', 0):.0%} | {s.get('final_success_rate', 0):.0%} | "
                f"{s.get('transport_success_rate', 0):.0%} | {s.get('xy_alignment_success_rate', 0):.0%} | "
                f"{s.get('lift_success_rate', 0):.0%} | "
                f"{s.get('E_transport_improvement', {}).get('rate', 0):.0%} | "
                f"{s.get('E_xy_improvement', {}).get('rate', 0):.0%} | "
                f"{s.get('E_lift_improvement', {}).get('rate', 0):.0%} | "
                f"{s.get('mean_raw_total_score', 0):.3f} | {s.get('mean_ranking_score', 0):.3f} |"
            )
        lines.append("")

    lines.extend(["## 3. Residual 改善是否转化为 Success", ""])
    for demo_key in FORMAL_ROLLOUT_DEMOS:
        demo_res = results.get("formal", {}).get(demo_key, {})
        gated = demo_res.get("strategies", {}).get("physics_residual_gated_top_k", {})
        aligned = demo_res.get("strategies", {}).get("aligned-original", {})
        ob = demo_res.get("original_baseline", {})
        delta_partial = gated.get("partial_success_rate", 0) - aligned.get("partial_success_rate", 0)
        delta_final = gated.get("final_success_rate", 0) - aligned.get("final_success_rate", 0)
        translated = (
            gated.get("E_transport_improvement", {}).get("rate", 0) >= 0.5
            and (
                delta_partial > 0
                or delta_final > 0
                or gated.get("best_partial_success")
            )
        )
        lines.append(f"### {demo_key}")
        lines.append("")
        lines.append(
            f"- original baseline: partial={ob.get('partial_success')}, final={ob.get('final_success')}"
        )
        lines.append(
            f"- aligned-original 策略: partial={aligned.get('partial_success_rate', 0):.0%}, "
            f"final={aligned.get('final_success_rate', 0):.0%}"
        )
        lines.append(
            f"- **physics_residual_gated_top_k**（优先）: partial={gated.get('partial_success_rate', 0):.0%}, "
            f"final={gated.get('final_success_rate', 0):.0%}, gate_pass={gated.get('gate_pass_rate', 0):.0%}"
        )
        lines.append(
            f"- residual→success 转化: **{'是' if translated else '否/部分'}** "
            f"(Δpartial={delta_partial:+.0%}, Δfinal={delta_final:+.0%})"
        )
        if demo_key == "demo_2":
            topk = demo_res.get("strategies", {}).get("physics_residual_top_k", {})
            lines.append(
                f"- demo_2 top_k E_lift 退化率较高 ({1 - topk.get('E_lift_improvement', {}).get('rate', 0):.0%})，"
                f"已优先 gated 策略"
            )
        lines.append("")

    if DIAGNOSTIC_DEMO in results.get("diagnostic", {}):
        d3 = results["diagnostic"][DIAGNOSTIC_DEMO]
        p1p2 = d3.get("strategies", {}).get("physics_residual_p1p2_gated_top_k", {})
        full = results.get("formal", {}).get(DIAGNOSTIC_DEMO, {}).get("strategies", {}).get(
            "physics_residual_gated_top_k", {}
        )
        lines.extend(["## 4. demo_3 诊断（P1/P2 gated）", ""])
        lines.append(
            f"- P1/P2 gated 候选数: {p1p2.get('num_rollouts', 0)} "
            f"(full gated 正式策略候选: {full.get('num_rollouts', 'N/A') if full else '未跑正式'})"
        )
        lines.append(
            f"- partial={p1p2.get('partial_success_rate', 0):.0%}, "
            f"transport↑={p1p2.get('E_transport_improvement', {}).get('rate', 0):.0%}, "
            f"source_consistency={p1p2.get('source_consistency_rate', 0):.0%}"
        )
        lines.append("")

    lines.extend(["## 5. 结论", ""])
    recommendations = results.get("recommendations", {})
    for key, val in recommendations.items():
        lines.append(f"- **{key}**: {val}")
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def _build_recommendations(results: dict[str, Any]) -> dict[str, str]:
    formal = results.get("formal", {})
    rollout_ready = []
    v1g_ready = []
    for demo_key in FORMAL_ROLLOUT_DEMOS:
        gated = formal.get(demo_key, {}).get("strategies", {}).get("physics_residual_gated_top_k", {})
        if gated.get("best_partial_success") or gated.get("final_success_rate", 0) > 0:
            rollout_ready.append(demo_key)
        if (
            gated.get("E_transport_improvement", {}).get("rate", 0) >= 0.8
            and gated.get("E_xy_improvement", {}).get("rate", 0) >= 0.8
        ):
            v1g_ready.append(demo_key)

    d3 = results.get("diagnostic", {}).get(DIAGNOSTIC_DEMO, {})
    p1p2 = d3.get("strategies", {}).get("physics_residual_p1p2_gated_top_k", {})
    return {
        "continue_rollout": ", ".join(rollout_ready) if rollout_ready else "暂无 demo 获得 final success 提升",
        "prefer_strategy": "physics_residual_gated_top_k（demo_2 避免 top_k 的 E_lift 退化）",
        "demo_3_next_step": (
            f"P1/P2 gated 产生 {p1p2.get('num_rollouts', 0)} 候选；"
            "若需正式 rollout 需放宽 lift gate 或扩大 PINN 池"
        ),
        "v1g_physics_loss": (
            f"可对 {', '.join(v1g_ready)} 试点 E_transport+E_xy 辅助 loss"
            if v1g_ready
            else "暂不建议全量 V1-G；先解决 demo_3 gate 空集"
        ),
    }


def run_rollout_validation(
    *,
    failed_hdf5: Path,
    cem_report: Path,
    aligned_model: Path,
    v1e_model: Path,
    success_reference_jsonl: Path,
    output_json: Path,
    report_md: Path,
    num_samples: int,
    top_k: int,
    seed: int,
) -> dict[str, Any]:
    os.environ["enable_physics_residual_repair"] = "true"

    formal_strategies = (
        "aligned-original",
        "physics_residual_top_k",
        "physics_residual_gated_top_k",
    )
    diagnostic_strategies = ("physics_residual_p1p2_gated_top_k",)

    formal_results: dict[str, Any] = {}
    for demo_key in FORMAL_ROLLOUT_DEMOS:
        formal_results[demo_key] = run_demo_rollout_validation(
            demo_key=demo_key,
            cfg=DEMO_REPAIR_CONFIGS[demo_key],
            failed_hdf5=failed_hdf5,
            cem_report=cem_report,
            aligned_model=aligned_model,
            v1e_model=v1e_model,
            success_reference_jsonl=success_reference_jsonl,
            strategies=formal_strategies,
            num_samples=num_samples,
            top_k=top_k,
            seed=seed,
        )

    diagnostic_results: dict[str, Any] = {}
    diagnostic_results[DIAGNOSTIC_DEMO] = run_demo_rollout_validation(
        demo_key=DIAGNOSTIC_DEMO,
        cfg=DEMO_REPAIR_CONFIGS[DIAGNOSTIC_DEMO],
        failed_hdf5=failed_hdf5,
        cem_report=cem_report,
        aligned_model=aligned_model,
        v1e_model=v1e_model,
        success_reference_jsonl=success_reference_jsonl,
        strategies=diagnostic_strategies,
        num_samples=num_samples,
        top_k=top_k,
        seed=seed,
    )

    payload_pre = {"formal": formal_results, "diagnostic": diagnostic_results}
    recommendations = _build_recommendations(payload_pre)

    all_records: list[dict[str, Any]] = []
    summary_by_strategy: dict[str, dict[str, Any]] = {}
    for section, demos in (("formal", formal_results), ("diagnostic", diagnostic_results)):
        for demo_key, demo_res in demos.items():
            summary_by_strategy[demo_key] = {
                "section": section,
                "original_baseline": demo_res["original_baseline"],
                **demo_res["strategies"],
            }
            for strategy, recs in demo_res["records"].items():
                all_records.extend(recs)

    payload = {
        "schema": "nut_assembly_physics_residual_rollout_validation_v1",
        "enable_physics_residual_repair": True,
        "aligned_original_checkpoint_preserved": True,
        "aligned_original_model": str(aligned_model),
        "formal_demos": list(FORMAL_ROLLOUT_DEMOS),
        "diagnostic_demos": [DIAGNOSTIC_DEMO],
        "primary_rollout_strategy": "physics_residual_gated_top_k",
        "summary_by_strategy": summary_by_strategy,
        "recommendations": recommendations,
        "formal": {
            k: {
                "original_baseline": v["original_baseline"],
                "strategies": v["strategies"],
            }
            for k, v in formal_results.items()
        },
        "diagnostic": {
            k: {
                "original_baseline": v["original_baseline"],
                "strategies": v["strategies"],
            }
            for k, v in diagnostic_results.items()
        },
        "records": all_records,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    generate_rollout_validation_report_md(results=payload_pre | {"recommendations": recommendations}, output_path=report_md)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Physics residual rollout validation")
    parser.add_argument("--failed-hdf5", type=Path, default=DEFAULT_FAILED_HDF5)
    parser.add_argument("--cem-report", type=Path, default=DEFAULT_CEM_REPORT)
    parser.add_argument("--aligned-original-model", type=Path, default=DEFAULT_ALIGNED_MODEL)
    parser.add_argument("--v1e-model", type=Path, default=DEFAULT_PINN_MODEL)
    parser.add_argument("--success-reference", type=Path, default=DEFAULT_SUCCESS_REFERENCE_JSONL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ROLLOUT_VALIDATION_OUTPUT_DIR)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_ROLLOUT_VALIDATION_JSON)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_ROLLOUT_VALIDATION_MD)
    parser.add_argument("--num-samples", type=int, default=80)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--enable-physics-residual-repair", action="store_true")
    args = parser.parse_args()

    if args.enable_physics_residual_repair:
        os.environ["enable_physics_residual_repair"] = "true"
    if not is_physics_residual_repair_enabled():
        print("ERROR: set --enable-physics-residual-repair", flush=True)
        return 2

    os.environ.setdefault("MUJOCO_GL", "egl")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    result = run_rollout_validation(
        failed_hdf5=args.failed_hdf5,
        cem_report=args.cem_report,
        aligned_model=args.aligned_original_model,
        v1e_model=args.v1e_model,
        success_reference_jsonl=args.success_reference,
        output_json=args.output_json,
        report_md=args.report_md,
        num_samples=args.num_samples,
        top_k=args.top_k,
        seed=args.seed,
    )
    print(json.dumps({"json": str(args.output_json), "md": str(args.report_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
