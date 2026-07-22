#!/usr/bin/env python3
"""demo_3 诊断：扩大候选池 + P1/P2 gated + source-aware ranking。"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
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
    DEFAULT_PHYSICS_RESIDUAL_OUTPUT_DIR,
    DEFAULT_SUCCESS_REFERENCE_JSONL,
    DEMO_REPAIR_CONFIGS,
)
from physics_residual_repair import (
    build_physics_repair_context,
    is_physics_residual_repair_enabled,
    select_indices_by_ranking_score,
    summarize_rollout_strategy,
)
from physics_residuals import (
    RESIDUAL_KEYS,
    check_source_consistency,
    compute_effective_ranking_score,
    compute_physics_residuals,
)
from repair_common_v1f import (
    extract_baseline_context_v1f,
    sample_repair_candidates_v1f,
    score_repair_candidates_v1f,
    select_candidate_indices_v1f,
)
from repair_rollout import run_repair_rollout
from rollout_outcome_evaluator import evaluate_rollout_outcome
from run_physics_residual_repair_validation import _run_original_baseline_rollout
from v1f_plus_utils import DEFAULT_ALIGNED_MODEL

DEFAULT_MD = DEFAULT_PHYSICS_RESIDUAL_OUTPUT_DIR / "demo_3_diagnostic_report.md"
DEFAULT_JSON = DEFAULT_PHYSICS_RESIDUAL_OUTPUT_DIR / "demo_3_diagnostic_report.json"

POOL_CONFIGS = (
    {"num_samples": 200, "top_k": 20},
    {"num_samples": 400, "top_k": 30},
)


def run_demo_3_pool(
    *,
    num_samples: int,
    top_k: int,
    seed: int,
    failed_hdf5: Path,
    cem_report: Path,
    aligned_model: Path,
    v1e_model: Path,
) -> dict[str, Any]:
    demo_key = "demo_3"
    cfg = DEMO_REPAIR_CONFIGS[demo_key]
    base_ctx = extract_baseline_context_v1f(
        failed_hdf5=failed_hdf5,
        demo_key=demo_key,
        failure_type=cfg["failure_type"],
        search_kind=cfg["search_kind"],
    )
    ctx = build_physics_repair_context(base_context=base_ctx, success_reference_jsonl=DEFAULT_SUCCESS_REFERENCE_JSONL)

    original_rollout = _run_original_baseline_rollout(demo_key=demo_key, cfg=cfg, failed_hdf5=failed_hdf5)
    original_br = compute_physics_residuals(original_rollout, ctx)
    original_outcome = evaluate_rollout_outcome(original_rollout, ctx)

    candidates = sample_repair_candidates_v1f(
        search_kind=cfg["search_kind"], n_samples=num_samples, seed=seed + 777
    )
    score_repair_candidates_v1f(
        context=base_ctx,
        candidates=candidates,
        active=cfg["active"],
        v1e_model_path=v1e_model,
        v1f_model_path=aligned_model,
    )
    pinn_top = select_candidate_indices_v1f(
        candidates, method="v1f_plain_top_k", top_k=top_k, rng=random.Random(seed)
    )
    for idx in pinn_top:
        if candidates[idx].get("rollout"):
            continue
        candidates[idx]["rollout"] = run_repair_rollout(
            failed_hdf5=failed_hdf5,
            demo_key=demo_key,
            search_kind=cfg["search_kind"],
            cem_report=None,
            candidate=candidates[idx],
        )

    selected = select_indices_by_ranking_score(
        candidates,
        context=ctx,
        top_k=top_k,
        require_gate=True,
        original_breakdown=original_br,
        gate_mode="p1p2",
    )

    records: list[dict[str, Any]] = []
    for rank, idx in enumerate(selected, start=1):
        rollout = candidates[idx]["rollout"]
        br = compute_physics_residuals(rollout, ctx)
        outcome = evaluate_rollout_outcome(rollout, ctx)
        src = check_source_consistency(br, original_br)
        eff_score, eff_keys, eff_meta = compute_effective_ranking_score(br, original_br)
        records.append(
            {
                "rank": rank,
                "candidate_index": idx,
                "raw_breakdown": br,
                "effective_ranking_score": eff_score,
                "effective_ranking_keys": eff_keys,
                "effective_ranking_meta": eff_meta,
                "source_consistency": src,
                "outcome": outcome,
            }
        )

    rollout_records = []
    for rec in records:
        ob = rec["outcome"]
        rollout_records.append(
            {
                **ob,
                "raw_total_score": rec["raw_breakdown"]["raw_total_score"],
                "ranking_score": rec["effective_ranking_score"],
                "physics_gate_passed": True,
                "source_consistent": rec["source_consistency"]["source_consistent"],
                "fallback_rate": sum(
                    1 for k in RESIDUAL_KEYS if rec["raw_breakdown"]["residuals"][k]["source"] == "fallback"
                )
                / len(RESIDUAL_KEYS),
                "delta_E_transport": rec["raw_breakdown"]["residuals"]["E_transport"]["normalized"]
                - original_br["residuals"]["E_transport"]["normalized"],
                "delta_E_xy": rec["raw_breakdown"]["residuals"]["E_xy"]["normalized"]
                - original_br["residuals"]["E_xy"]["normalized"],
                "delta_E_lift": rec["raw_breakdown"]["residuals"]["E_lift"]["normalized"]
                - original_br["residuals"]["E_lift"]["normalized"],
                "delta_total_score": rec["raw_breakdown"]["raw_total_score"] - original_br["raw_total_score"],
                "delta_ranking_score": rec["effective_ranking_score"]
                - compute_effective_ranking_score(original_br, original_br)[0],
            }
        )

    summary = summarize_rollout_strategy(
        demo_key=demo_key,
        strategy="physics_residual_p1p2_gated_top_k",
        rollout_records=rollout_records,
        original_breakdown=original_br,
    )
    summary["pinn_pool_size"] = len(pinn_top)
    summary["num_samples"] = num_samples
    summary["top_k"] = top_k

    return {
        "pool": {"num_samples": num_samples, "top_k": top_k},
        "original_baseline": {**original_outcome, "raw_breakdown": original_br},
        "selected_count": len(selected),
        "summary": summary,
        "records": records,
    }


def diagnose_root_cause(pool_results: list[dict[str, Any]]) -> dict[str, str]:
    latest = pool_results[-1]
    selected = latest["selected_count"]
    pinn_pool = latest["summary"].get("pinn_pool_size", 0)
    gate_pass = latest["summary"].get("gate_pass_rate", 0)
    partial = latest["summary"].get("partial_success_rate", 0)
    final = latest["summary"].get("final_success_rate", 0)
    reasons = Counter()
    for rec in latest.get("records", []):
        reasons[rec["outcome"]["failure_reason"]] += 1

    if selected == 0:
        root = "lift_gate_too_strict"
        detail = "P1/P2 gated 在扩大池后仍无候选通过 gate"
    elif selected < latest["pool"]["top_k"] * 0.5:
        root = "lift_gate_too_strict"
        detail = f"仅 {selected}/{latest['pool']['top_k']} 候选通过 gate"
    elif pinn_pool < latest["pool"]["top_k"]:
        root = "candidate_pool_insufficient"
        detail = f"PINN prescreen 仅产出 {pinn_pool} 条 rollout"
    elif partial >= 0.5 and final == 0:
        root = "insertion_contact_stage_failure"
        detail = f"partial={partial:.0%} 但 final=0；failure reasons: {dict(reasons)}"
    else:
        root = "mixed"
        detail = f"selected={selected}, partial={partial:.0%}, final={final:.0%}"

    return {"root_cause": root, "detail": detail, "failure_reason_counts": dict(reasons)}


def write_md(payload: dict[str, Any], path: Path) -> None:
    diag = payload["diagnosis"]
    lines = [
        "# demo_3 Physics Residual 诊断报告",
        "",
        f"- root_cause: **{diag['root_cause']}**",
        f"- detail: {diag['detail']}",
        "",
        "## 候选池扫描",
        "",
    ]
    for pr in payload["pool_results"]:
        s = pr["summary"]
        lines.append(
            f"- samples={pr['pool']['num_samples']} top_k={pr['pool']['top_k']}: "
            f"selected={pr['selected_count']} partial={s.get('partial_success_rate', 0):.0%} "
            f"final={s.get('final_success_rate', 0):.0%} gate_pass={s.get('gate_pass_rate', 0):.0%}"
        )
    lines.extend(
        [
            "",
            "## Source Consistency 修复",
            "",
            "- gate / effective ranking 仅使用 source 一致 residual",
            "- 不一致项保留于 raw_breakdown，不参与 gate 与 ranking_score",
            "",
            "## 判断",
            "",
        ]
    )
    if diag["root_cause"] == "lift_gate_too_strict":
        lines.append("- 主要瓶颈：**lift gate 过严**（非候选池不足）")
    elif diag["root_cause"] == "candidate_pool_insufficient":
        lines.append("- 主要瓶颈：**候选池/rollout 不足**")
    elif diag["root_cause"] == "insertion_contact_stage_failure":
        lines.append("- 主要瓶颈：**insertion/contact 阶段失败**（transport/xy 改善但未 final success）")
    else:
        lines.append(f"- 综合判断：{diag['detail']}")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="demo_3 diagnostic")
    parser.add_argument("--failed-hdf5", type=Path, default=DEFAULT_FAILED_HDF5)
    parser.add_argument("--cem-report", type=Path, default=DEFAULT_CEM_REPORT)
    parser.add_argument("--aligned-model", type=Path, default=DEFAULT_ALIGNED_MODEL)
    parser.add_argument("--v1e-model", type=Path, default=DEFAULT_PINN_MODEL)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--enable-physics-residual-repair", action="store_true")
    args = parser.parse_args()

    if args.enable_physics_residual_repair:
        os.environ["enable_physics_residual_repair"] = "true"
    if not is_physics_residual_repair_enabled():
        print("ERROR: --enable-physics-residual-repair required", flush=True)
        return 2

    os.environ.setdefault("MUJOCO_GL", "egl")
    pool_results = []
    for pc in POOL_CONFIGS:
        print(f"[demo_3] pool samples={pc['num_samples']} top_k={pc['top_k']}", flush=True)
        pool_results.append(
            run_demo_3_pool(
                num_samples=pc["num_samples"],
                top_k=pc["top_k"],
                seed=args.seed,
                failed_hdf5=args.failed_hdf5,
                cem_report=args.cem_report,
                aligned_model=args.aligned_model,
                v1e_model=args.v1e_model,
            )
        )

    diagnosis = diagnose_root_cause(pool_results)
    payload = {
        "demo_key": "demo_3",
        "strategy": "physics_residual_p1p2_gated_top_k",
        "source_consistency_policy": "inconsistent residuals excluded from gate and effective ranking",
        "pool_results": pool_results,
        "diagnosis": diagnosis,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    write_md(payload, args.output_md)
    print(json.dumps({"json": str(args.output_json), "md": str(args.output_md), "diagnosis": diagnosis}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
