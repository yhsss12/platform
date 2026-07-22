#!/usr/bin/env python3
"""demo_3 insertion-stage local repair 验证与三策略对比。"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
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
from insertion_residuals import INSERTION_RESIDUAL_KEYS, compute_insertion_residuals
from insertion_stage_repair import (
    compute_insertion_stage_objective,
    is_insertion_stage_repair_enabled,
    run_insertion_stage_local_search,
)
from physics_residual_repair import (
    build_candidate_record,
    build_physics_repair_context,
    is_physics_residual_repair_enabled,
    select_indices_by_insertion_gated_ranking,
    summarize_rollout_strategy,
)
from physics_residuals import compute_physics_residuals
from repair_common_v1f import (
    extract_baseline_context_v1f,
    sample_repair_candidates_v1f,
    score_repair_candidates_v1f,
    select_candidate_indices_v1f,
)
from repair_rollout import run_repair_rollout
from rollout_outcome_evaluator import evaluate_rollout_outcome
from run_insertion_gated_multi_seed_validation import _pool_gate_stats, _select_indices
from run_physics_residual_repair_validation import _run_original_baseline_rollout
from v1f_plus_utils import DEFAULT_ALIGNED_MODEL

DEFAULT_OUTPUT = DEFAULT_PHYSICS_RESIDUAL_OUTPUT_DIR / "demo_3_insertion_repair_breakdown.json"
DEFAULT_MD = DEFAULT_PHYSICS_RESIDUAL_OUTPUT_DIR / "demo_3_insertion_repair_breakdown.md"

STRATEGIES = (
    "v1f_plain_top_k",
    "physics_residual_insertion_gated_top_k",
    "insertion_stage_repair",
)

POOL_CONFIGS = (
    {"num_samples": 200, "top_k": 20},
    {"num_samples": 400, "top_k": 30},
)
DEFAULT_SEEDS = (0, 1, 2, 3, 4)


def _summarize_records(
    *,
    demo_key: str,
    strategy: str,
    records: list[dict[str, Any]],
    original_breakdown: dict[str, Any],
    gate_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary = summarize_rollout_strategy(
        demo_key=demo_key,
        strategy=strategy,
        rollout_records=records,
        original_breakdown=original_breakdown,
    )
    summary["selected_count"] = len(records)
    if gate_stats:
        summary["gate_pass_rate"] = gate_stats.get("gate_pass_rate", summary.get("gate_pass_rate", 0.0))
        summary["gate_pass_count"] = gate_stats.get("gate_pass_count", 0)
        summary["insertion_gate_reject_reasons"] = gate_stats.get("insertion_gate_reject_reasons", {})
        summary["rejected_p1p2"] = gate_stats.get("rejected_p1p2", 0)
        summary["rejected_insertion"] = gate_stats.get("rejected_insertion", 0)

    ins_stats: dict[str, list[float]] = {k: [] for k in INSERTION_RESIDUAL_KEYS}
    failure_reasons: Counter[str] = Counter()
    for rec in records:
        reason = rec.get("insertion_failure_reason") or rec.get("failure_reason") or "unknown"
        failure_reasons[str(reason)] += 1
        ibr = rec.get("insertion_residuals") or {}
        for key in INSERTION_RESIDUAL_KEYS:
            if key in ibr:
                ins_stats[key].append(float(ibr[key].get("normalized", ibr[key]) if isinstance(ibr[key], dict) else ibr[key]))
            elif "raw_breakdown" in rec:
                ins_stats[key].append(float(rec["raw_breakdown"]["residuals"][key]["normalized"]))

    summary["insertion_residual_stats"] = {
        k: {
            "mean": float(statistics.mean(v)) if v else 0.0,
            "min": float(min(v)) if v else 0.0,
            "max": float(max(v)) if v else 0.0,
        }
        for k, v in ins_stats.items()
    }
    summary["failure_reason_counts"] = dict(failure_reasons)
    return summary


def run_demo3_config(
    *,
    num_samples: int,
    top_k: int,
    seed: int,
    failed_hdf5: Path,
    cem_report: Path,
    aligned_model: Path,
    v1e_model: Path,
    success_reference_jsonl: Path,
    local_search_evals: int,
    max_repair_candidates: int,
) -> dict[str, Any]:
    demo_key = "demo_3"
    cfg = DEMO_REPAIR_CONFIGS[demo_key]
    base_ctx = extract_baseline_context_v1f(
        failed_hdf5=failed_hdf5,
        demo_key=demo_key,
        failure_type=cfg["failure_type"],
        search_kind=cfg["search_kind"],
    )
    ctx = build_physics_repair_context(base_context=base_ctx, success_reference_jsonl=success_reference_jsonl)

    original_rollout = _run_original_baseline_rollout(demo_key=demo_key, cfg=cfg, failed_hdf5=failed_hdf5)
    original_breakdown = compute_physics_residuals(original_rollout, ctx)

    candidates = sample_repair_candidates_v1f(
        search_kind=cfg["search_kind"],
        n_samples=num_samples,
        seed=seed + hash(demo_key) % 10000,
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

    strategy_results: dict[str, Any] = {}

    # 1) plain top-k
    plain_indices = list(pinn_top[:top_k])
    plain_records: list[dict[str, Any]] = []
    for rank, idx in enumerate(plain_indices, start=1):
        rollout = candidates[idx]["rollout"]
        br = compute_physics_residuals(rollout, ctx)
        ibr = compute_insertion_residuals(rollout, ctx)
        rec = build_candidate_record(
            label=f"plain_{rank:02d}",
            demo_key=demo_key,
            strategy="v1f_plain_top_k",
            rollout=rollout,
            breakdown=br,
            original_breakdown=original_breakdown,
            candidate_index=idx,
        )
        rec.update(evaluate_rollout_outcome(rollout, ctx))
        rec["insertion_residuals"] = ibr["residuals"]
        rec["insertion_failure_reason"] = rec.get("failure_reason")
        plain_records.append(rec)
    strategy_results["v1f_plain_top_k"] = _summarize_records(
        demo_key=demo_key,
        strategy="v1f_plain_top_k",
        records=plain_records,
        original_breakdown=original_breakdown,
        gate_stats={"gate_pass_rate": 1.0, "gate_pass_count": len(plain_indices), "pool_size": top_k},
    )

    # 2) insertion gated
    gated_indices, gate_records = _select_indices(
        strategy="physics_residual_insertion_gated_top_k",
        candidates=candidates,
        pinn_top=pinn_top,
        context=ctx,
        original_breakdown=original_breakdown,
        original_rollout=original_rollout,
        top_k=top_k,
    )
    _, gate_stats = _pool_gate_stats(
        strategy="physics_residual_insertion_gated_top_k",
        candidates=candidates,
        pinn_top=pinn_top,
        context=ctx,
        original_breakdown=original_breakdown,
        original_rollout=original_rollout,
    )
    gated_records: list[dict[str, Any]] = []
    for rank, idx in enumerate(gated_indices, start=1):
        rollout = candidates[idx]["rollout"]
        br = compute_physics_residuals(rollout, ctx)
        ibr = compute_insertion_residuals(rollout, ctx)
        rec = build_candidate_record(
            label=f"gated_{rank:02d}",
            demo_key=demo_key,
            strategy="physics_residual_insertion_gated_top_k",
            rollout=rollout,
            breakdown=br,
            original_breakdown=original_breakdown,
            candidate_index=idx,
        )
        rec.update(evaluate_rollout_outcome(rollout, ctx))
        rec["insertion_residuals"] = ibr["residuals"]
        rec["insertion_failure_reason"] = rec.get("failure_reason")
        gated_records.append(rec)
    strategy_results["physics_residual_insertion_gated_top_k"] = _summarize_records(
        demo_key=demo_key,
        strategy="physics_residual_insertion_gated_top_k",
        records=gated_records,
        original_breakdown=original_breakdown,
        gate_stats=gate_stats,
    )

    # 3) insertion stage local repair on partial-success PINN candidates
    partial_indices = [
        idx
        for idx in pinn_top
        if evaluate_rollout_outcome(candidates[idx]["rollout"], ctx).get("partial_success")
    ][:max_repair_candidates]

    repair_records: list[dict[str, Any]] = []
    repair_details: list[dict[str, Any]] = []
    for rank, idx in enumerate(partial_indices, start=1):
        cand = candidates[idx]
        search = run_insertion_stage_local_search(
            failed_hdf5=failed_hdf5,
            demo_key=demo_key,
            grasp_lift=cand.get("grasp_lift") or cand,
            lift_extra=cand.get("lift_extra"),
            context=ctx,
            max_evals=local_search_evals,
            seed=seed + idx,
        )
        rollout = search["best_rollout"]
        br = compute_physics_residuals(rollout, ctx)
        ibr = compute_insertion_residuals(rollout, ctx)
        obj = compute_insertion_stage_objective(rollout, context=ctx)
        rec = build_candidate_record(
            label=f"stage_repair_{rank:02d}",
            demo_key=demo_key,
            strategy="insertion_stage_repair",
            rollout=rollout,
            breakdown=br,
            original_breakdown=original_breakdown,
            candidate_index=idx,
        )
        rec.update(evaluate_rollout_outcome(rollout, ctx))
        rec["insertion_residuals"] = ibr["residuals"]
        rec["insertion_failure_reason"] = rollout.get("insertion_failure_reason")
        rec["insertion_stage_objective"] = obj
        rec["repair_params"] = search["best"]["repair_params"]
        repair_records.append(rec)
        repair_details.append(
            {
                "source_candidate_index": idx,
                "num_local_evals": search["num_evals"],
                "best_objective": search["best"]["objective"],
                "best_outcome": search["best"]["outcome"],
                "best_repair_params": search["best"]["repair_params"],
            }
        )

    repair_records.sort(
        key=lambda r: (
            not r.get("final_success"),
            r.get("insertion_stage_objective", {}).get("objective_score", 1e9),
        )
    )
    repair_records = repair_records[:top_k]
    strategy_results["insertion_stage_repair"] = _summarize_records(
        demo_key=demo_key,
        strategy="insertion_stage_repair",
        records=repair_records,
        original_breakdown=original_breakdown,
        gate_stats={
            "gate_pass_rate": len(repair_records) / max(len(partial_indices), 1),
            "gate_pass_count": len(repair_records),
            "pool_size": len(partial_indices),
            "partial_source_count": len(partial_indices),
        },
    )
    strategy_results["insertion_stage_repair"]["local_search_details"] = repair_details

    print(
        f"[demo_3] seed={seed} n={num_samples} k={top_k} "
        f"plain={strategy_results['v1f_plain_top_k']['final_success_rate']:.0%} "
        f"gated={strategy_results['physics_residual_insertion_gated_top_k']['final_success_rate']:.0%} "
        f"repair={strategy_results['insertion_stage_repair']['final_success_rate']:.0%}",
        flush=True,
    )

    return {
        "demo_key": demo_key,
        "num_samples": num_samples,
        "top_k": top_k,
        "seed": seed,
        "strategies": strategy_results,
        "partial_candidate_count": len(partial_indices),
    }


def _aggregate_runs(runs: list[dict[str, Any]], strategy: str) -> dict[str, Any]:
    summaries = [r["strategies"][strategy] for r in runs]
    finals = [s["final_success_rate"] for s in summaries]
    partials = [s["partial_success_rate"] for s in summaries]
    reason_cluster: Counter[str] = Counter()
    for s in summaries:
        reason_cluster.update(s.get("failure_reason_counts", {}))
    return {
        "runs": len(summaries),
        "mean_final_success_rate": float(statistics.mean(finals)) if finals else 0.0,
        "mean_partial_success_rate": float(statistics.mean(partials)) if partials else 0.0,
        "max_final_success_rate": float(max(finals)) if finals else 0.0,
        "failure_reason_cluster": dict(reason_cluster.most_common()),
        "still_all_final_zero": all(f == 0.0 for f in finals),
    }


def write_md(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# demo_3 Insertion-Stage Local Repair 验证",
        "",
        f"- checkpoint: `{payload['meta']['checkpoint']}`",
        f"- aligned-original preserved: {payload['meta']['aligned_original_preserved']}",
        f"- insertion_stage_repair: opt-in (`enable_insertion_stage_repair=true`)",
        "",
        "## 验收",
        "",
    ]
    acc = payload["acceptance"]
    lines.extend(
        [
            f"- partial success 保持 100%: {acc['partial_success_maintained']}",
            f"- final success 提升: {acc['final_success_improved']} (plain {acc['plain_mean_final']:.0%} → repair {acc['repair_mean_final']:.0%})",
            f"- failure 原因更具体: {acc['failure_reasons_more_specific']}",
            f"- 仍全部 final=0: {acc['all_final_zero']}",
            "",
            "## 三策略聚合",
            "",
            "| 策略 | mean partial | mean final | max final |",
            "|------|--------------|------------|-----------|",
        ]
    )
    for strat in STRATEGIES:
        agg = payload["aggregates"][strat]
        lines.append(
            f"| {strat} | {agg['mean_partial_success_rate']:.0%} | "
            f"{agg['mean_final_success_rate']:.0%} | {agg['max_final_success_rate']:.0%} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="demo_3 insertion stage repair validation")
    parser.add_argument("--failed-hdf5", type=Path, default=DEFAULT_FAILED_HDF5)
    parser.add_argument("--cem-report", type=Path, default=DEFAULT_CEM_REPORT)
    parser.add_argument("--aligned-model", type=Path, default=DEFAULT_ALIGNED_MODEL)
    parser.add_argument("--v1e-model", type=Path, default=DEFAULT_PINN_MODEL)
    parser.add_argument("--success-reference", type=Path, default=DEFAULT_SUCCESS_REFERENCE_JSONL)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--local-search-evals", type=int, default=30)
    parser.add_argument("--max-repair-candidates", type=int, default=5)
    parser.add_argument("--enable-physics-residual-repair", action="store_true")
    parser.add_argument("--enable-insertion-stage-repair", action="store_true")
    args = parser.parse_args()

    if args.enable_physics_residual_repair:
        os.environ["enable_physics_residual_repair"] = "true"
    if args.enable_insertion_stage_repair:
        os.environ["enable_insertion_stage_repair"] = "true"
    if not is_physics_residual_repair_enabled():
        print("ERROR: --enable-physics-residual-repair required", flush=True)
        return 2
    if not is_insertion_stage_repair_enabled():
        print("ERROR: --enable-insertion-stage-repair required", flush=True)
        return 2

    os.environ.setdefault("MUJOCO_GL", "egl")
    all_runs: list[dict[str, Any]] = []
    for pool in POOL_CONFIGS:
        for seed in args.seeds:
            all_runs.append(
                run_demo3_config(
                    num_samples=pool["num_samples"],
                    top_k=pool["top_k"],
                    seed=seed,
                    failed_hdf5=args.failed_hdf5,
                    cem_report=args.cem_report,
                    aligned_model=args.aligned_model,
                    v1e_model=args.v1e_model,
                    success_reference_jsonl=args.success_reference,
                    local_search_evals=args.local_search_evals,
                    max_repair_candidates=args.max_repair_candidates,
                )
            )

    aggregates = {strat: _aggregate_runs(all_runs, strat) for strat in STRATEGIES}
    plain_agg = aggregates["v1f_plain_top_k"]
    repair_agg = aggregates["insertion_stage_repair"]
    specific_reasons = {
        "insertion_jamming",
        "insertion_axis_misalignment",
        "insertion_vertical_approach_error",
        "insertion_depth_error",
        "insertion_final_pose_error",
        "insertion_contact_unstable",
    }
    repair_reasons = set(repair_agg["failure_reason_cluster"].keys())
    payload = {
        "schema": "demo_3_insertion_repair_breakdown_v1",
        "meta": {
            "checkpoint": str(args.aligned_model),
            "aligned_original_preserved": True,
            "physics_residual_repair_opt_in": True,
            "insertion_stage_repair_opt_in": True,
            "pool_configs": list(POOL_CONFIGS),
            "seeds": list(args.seeds),
            "strategies": list(STRATEGIES),
        },
        "runs": all_runs,
        "aggregates": aggregates,
        "acceptance": {
            "partial_success_maintained": all(
                r["strategies"]["insertion_stage_repair"]["partial_success_rate"] >= 0.99 for r in all_runs
            ),
            "final_success_improved": repair_agg["mean_final_success_rate"] > plain_agg["mean_final_success_rate"],
            "plain_mean_final": plain_agg["mean_final_success_rate"],
            "repair_mean_final": repair_agg["mean_final_success_rate"],
            "gated_mean_final": aggregates["physics_residual_insertion_gated_top_k"]["mean_final_success_rate"],
            "failure_reasons_more_specific": bool(repair_reasons & specific_reasons),
            "all_final_zero": repair_agg["still_all_final_zero"],
            "failure_reason_cluster_repair": repair_agg["failure_reason_cluster"],
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    write_md(payload, args.output_md)
    print(json.dumps({"json": str(args.output_json), "md": str(args.output_md), "acceptance": payload["acceptance"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
