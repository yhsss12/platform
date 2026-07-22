#!/usr/bin/env python3
"""Multi-seed / multi-pool insertion-aware gate 验证。"""
from __future__ import annotations

import argparse
import json
import os
import random
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
    DEFAULT_PHYSICS_RESIDUAL_OUTPUT_DIR,
    DEFAULT_SUCCESS_REFERENCE_JSONL,
    DEMO_REPAIR_CONFIGS,
)
from insertion_residuals import INSERTION_RESIDUAL_KEYS, compute_insertion_residuals  # noqa: E402
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

DEFAULT_JSON = DEFAULT_PHYSICS_RESIDUAL_OUTPUT_DIR / "insertion_gated_multi_seed_report.json"
DEFAULT_MD = DEFAULT_PHYSICS_RESIDUAL_OUTPUT_DIR / "insertion_gated_multi_seed_report.md"

STRATEGIES = (
    "v1f_plain_top_k",
    "physics_residual_gated_top_k",
    "physics_residual_p1p2_gated_top_k",
    "physics_residual_insertion_gated_top_k",
)

POOL_CONFIGS = (
    {"num_samples": 200, "top_k": 20},
    {"num_samples": 400, "top_k": 30},
)

DEFAULT_SEEDS = (0, 1, 2, 3, 4)


def _pool_gate_stats(
    *,
    strategy: str,
    candidates: list[dict[str, Any]],
    pinn_top: list[int],
    context: dict[str, Any],
    original_breakdown: dict[str, Any],
    original_rollout: dict[str, Any],
) -> tuple[list[dict[str, Any]] | None, dict[str, Any]]:
    """在 PINN prescreen 池上统计 gate pass / reject。"""
    pool_size = len(pinn_top)
    if strategy == "v1f_plain_top_k":
        return None, {"pool_size": pool_size, "gate_pass_count": pool_size, "gate_pass_rate": 1.0}

    if strategy == "physics_residual_insertion_gated_top_k":
        _, gate_records = select_indices_by_insertion_gated_ranking(
            candidates,
            context=context,
            top_k=pool_size,
            original_breakdown=original_breakdown,
            original_rollout=original_rollout,
        )
        accepted = sum(1 for r in gate_records if r.get("accepted"))
        reject_reasons: Counter[str] = Counter()
        for rec in gate_records:
            if rec.get("accepted"):
                continue
            stage = str(rec.get("rejection_stage", "unknown"))
            reason = rec.get("reason")
            if isinstance(reason, list):
                for item in reason:
                    reject_reasons[f"{stage}:{item}"] += 1
            else:
                reject_reasons[f"{stage}:{reason}"] += 1
        return gate_records, {
            "pool_size": pool_size,
            "gate_pass_count": accepted,
            "gate_pass_rate": float(accepted / max(pool_size, 1)),
            "insertion_gate_reject_reasons": dict(reject_reasons),
            "rejected_p1p2": sum(1 for r in gate_records if r.get("rejection_stage") == "p1p2_gate"),
            "rejected_insertion": sum(1 for r in gate_records if r.get("rejection_stage") == "insertion_gate"),
        }

    passed = 0
    reject_reasons = Counter()
    for idx in pinn_top:
        rollout = candidates[idx].get("rollout")
        if not rollout:
            continue
        br = compute_physics_residuals(rollout, context)
        src = check_source_consistency(br, original_breakdown)
        if strategy == "physics_residual_p1p2_gated_top_k":
            ok, checks = candidate_passes_p1p2_gate(br, original_breakdown, source_consistency=src)
        else:
            ok, checks = candidate_beats_original(br, original_breakdown, respect_source_consistency=True)
        if ok:
            passed += 1
        else:
            for k, v in checks.items():
                if not v and not k.endswith("_skipped_source_mismatch"):
                    reject_reasons[f"physics_gate:{k}"] += 1

    return None, {
        "pool_size": pool_size,
        "gate_pass_count": passed,
        "gate_pass_rate": float(passed / max(pool_size, 1)),
        "insertion_gate_reject_reasons": dict(reject_reasons),
    }


def _select_indices(
    *,
    strategy: str,
    candidates: list[dict[str, Any]],
    pinn_top: list[int],
    context: dict[str, Any],
    original_breakdown: dict[str, Any],
    original_rollout: dict[str, Any],
    top_k: int,
) -> tuple[list[int], list[dict[str, Any]] | None]:
    if strategy == "v1f_plain_top_k":
        return list(pinn_top[:top_k]), None
    if strategy == "physics_residual_gated_top_k":
        return (
            select_indices_by_ranking_score(
                candidates,
                context=context,
                top_k=top_k,
                require_gate=True,
                original_breakdown=original_breakdown,
                gate_mode="full",
            ),
            None,
        )
    if strategy == "physics_residual_p1p2_gated_top_k":
        return (
            select_indices_by_ranking_score(
                candidates,
                context=context,
                top_k=top_k,
                require_gate=True,
                original_breakdown=original_breakdown,
                gate_mode="p1p2",
            ),
            None,
        )
    if strategy == "physics_residual_insertion_gated_top_k":
        return select_indices_by_insertion_gated_ranking(
            candidates,
            context=context,
            top_k=top_k,
            original_breakdown=original_breakdown,
            original_rollout=original_rollout,
        )
    raise ValueError(strategy)


def _insertion_residual_stats(records: list[dict[str, Any]], context: dict[str, Any]) -> dict[str, Any]:
    if not records:
        return {k: {"mean": 0.0, "min": 0.0, "max": 0.0, "p50": 0.0} for k in INSERTION_RESIDUAL_KEYS}
    per_key: dict[str, list[float]] = {k: [] for k in INSERTION_RESIDUAL_KEYS}
    for rec in records:
        rollout = rec.get("_rollout") or {}
        ibr = compute_insertion_residuals(rollout, context)
        for key in INSERTION_RESIDUAL_KEYS:
            per_key[key].append(float(ibr["residuals"][key]["normalized"]))
    out: dict[str, Any] = {}
    for key, vals in per_key.items():
        if not vals:
            out[key] = {"mean": 0.0, "min": 0.0, "max": 0.0, "p50": 0.0}
        else:
            sv = sorted(vals)
            out[key] = {
                "mean": float(statistics.mean(vals)),
                "min": float(min(vals)),
                "max": float(max(vals)),
                "p50": float(sv[len(sv) // 2]),
            }
    return out


def run_single_config(
    *,
    demo_key: str,
    num_samples: int,
    top_k: int,
    seed: int,
    failed_hdf5: Path,
    cem_report: Path,
    aligned_model: Path,
    v1e_model: Path,
    success_reference_jsonl: Path,
) -> dict[str, Any]:
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
            cem_report=cem_report if cfg["search_kind"] in ("insertion", "transport") else None,
            candidate=candidates[idx],
        )

    strategy_results: dict[str, Any] = {}
    for strategy in STRATEGIES:
        gate_records, gate_stats = _pool_gate_stats(
            strategy=strategy,
            candidates=candidates,
            pinn_top=pinn_top,
            context=ctx,
            original_breakdown=original_breakdown,
            original_rollout=original_rollout,
        )
        selected, sel_gate_records = _select_indices(
            strategy=strategy,
            candidates=candidates,
            pinn_top=pinn_top,
            context=ctx,
            original_breakdown=original_breakdown,
            original_rollout=original_rollout,
            top_k=top_k,
        )

        records: list[dict[str, Any]] = []
        for rank, idx in enumerate(selected, start=1):
            rollout = candidates[idx]["rollout"]
            br = compute_physics_residuals(rollout, ctx)
            rec = build_candidate_record(
                label=f"{strategy}_{rank:02d}",
                demo_key=demo_key,
                strategy=strategy,
                rollout=rollout,
                breakdown=br,
                original_breakdown=original_breakdown,
                candidate_index=idx,
            )
            rec.update(evaluate_rollout_outcome(rollout, ctx))
            rec["_rollout"] = rollout
            if strategy == "physics_residual_p1p2_gated_top_k":
                src = check_source_consistency(br, original_breakdown)
                passed, checks = candidate_passes_p1p2_gate(br, original_breakdown, source_consistency=src)
                rec["physics_gate_passed"] = passed
                rec["physics_gate_checks"] = checks
            elif strategy == "physics_residual_insertion_gated_top_k":
                rec["physics_gate_passed"] = True
            records.append(rec)

        summary = summarize_rollout_strategy(
            demo_key=demo_key,
            strategy=strategy,
            rollout_records=records,
            original_breakdown=original_breakdown,
        )
        insertion_stats = _insertion_residual_stats(records, ctx)
        for rec in records:
            rec.pop("_rollout", None)

        ig_reject = gate_stats.get("insertion_gate_reject_reasons", {})
        if sel_gate_records is not None and strategy == "physics_residual_insertion_gated_top_k":
            extra_reject: Counter[str] = Counter()
            for rec in sel_gate_records:
                if rec.get("accepted"):
                    continue
                stage = str(rec.get("rejection_stage", "unknown"))
                reason = rec.get("reason")
                if isinstance(reason, list):
                    for item in reason:
                        extra_reject[f"{stage}:{item}"] += 1
            ig_reject = dict(Counter(ig_reject) + extra_reject)

        lift_degraded_n = sum(1 for r in records if float(r.get("delta_E_lift", 0.0)) > 0)
        strategy_results[strategy] = {
            "selected_count": len(selected),
            "gate_pass_rate": gate_stats["gate_pass_rate"],
            "gate_pass_count": gate_stats["gate_pass_count"],
            "pool_size": gate_stats["pool_size"],
            "partial_success_rate": summary["partial_success_rate"],
            "final_success_rate": summary["final_success_rate"],
            "transport_success_rate": summary["transport_success_rate"],
            "xy_alignment_success_rate": summary["xy_alignment_success_rate"],
            "lift_success_rate": summary["lift_success_rate"],
            "mean_raw_total_score": summary.get("mean_raw_total_score", 0.0),
            "mean_ranking_score": summary.get("mean_ranking_score", 0.0),
            "E_transport_improved_rate": summary.get("E_transport_improvement", {}).get("rate", 0.0),
            "E_xy_improved_rate": summary.get("E_xy_improvement", {}).get("rate", 0.0),
            "E_lift_degraded_rate": float(lift_degraded_n / max(len(records), 1)),
            "failure_reason_counts": summary.get("failure_reason_counts", {}),
            "insertion_gate_reject_reasons": ig_reject,
            "insertion_residual_stats": insertion_stats,
            "rejected_p1p2": gate_stats.get("rejected_p1p2", 0),
            "rejected_insertion": gate_stats.get("rejected_insertion", 0),
        }
        print(
            f"[ms] {demo_key} n={num_samples} k={top_k} seed={seed} {strategy} "
            f"sel={len(selected)} final={summary['final_success_rate']:.0%}",
            flush=True,
        )

    return {
        "demo_key": demo_key,
        "num_samples": num_samples,
        "top_k": top_k,
        "seed": seed,
        "pinn_top_indices": pinn_top,
        "strategies": strategy_results,
    }


def _aggregate_strategy(runs: list[dict[str, Any]], strategy: str) -> dict[str, Any]:
    vals = [r["strategies"][strategy] for r in runs]
    finals = [v["final_success_rate"] for v in vals]
    partials = [v["partial_success_rate"] for v in vals]
    reject_merged: Counter[str] = Counter()
    for v in vals:
        reject_merged.update(v.get("insertion_gate_reject_reasons", {}))

    return {
        "runs": len(vals),
        "mean_final_success_rate": float(statistics.mean(finals)) if finals else 0.0,
        "mean_partial_success_rate": float(statistics.mean(partials)) if partials else 0.0,
        "min_final_success_rate": float(min(finals)) if finals else 0.0,
        "max_final_success_rate": float(max(finals)) if finals else 0.0,
        "mean_selected_count": float(statistics.mean(v["selected_count"] for v in vals)),
        "mean_gate_pass_rate": float(statistics.mean(v["gate_pass_rate"] for v in vals)),
        "insertion_gate_reject_reasons_total": dict(reject_merged),
    }


def _demo4_insertion_vs_plain(all_runs: list[dict[str, Any]]) -> dict[str, Any]:
    wins = 0
    ties = 0
    losses = 0
    deltas: list[float] = []
    for run in all_runs:
        if run["demo_key"] != "demo_4":
            continue
        plain = run["strategies"]["v1f_plain_top_k"]["final_success_rate"]
        ins = run["strategies"]["physics_residual_insertion_gated_top_k"]["final_success_rate"]
        delta = ins - plain
        deltas.append(delta)
        if delta > 1e-9:
            wins += 1
        elif delta < -1e-9:
            losses += 1
        else:
            ties += 1
    return {
        "comparison": "physics_residual_insertion_gated_top_k vs v1f_plain_top_k",
        "runs": len(deltas),
        "insertion_higher_count": wins,
        "tie_count": ties,
        "plain_higher_count": losses,
        "mean_delta_final": float(statistics.mean(deltas)) if deltas else 0.0,
        "stable_above_plain": wins > losses and float(statistics.mean(deltas)) > 0 if deltas else False,
    }


def _demo3_failure_cluster(all_runs: list[dict[str, Any]]) -> dict[str, Any]:
    runs = [r for r in all_runs if r["demo_key"] == "demo_3"]
    cluster: Counter[str] = Counter()
    all_final_zero = True
    for run in runs:
        for strat in STRATEGIES:
            s = run["strategies"][strat]
            if s["final_success_rate"] > 0:
                all_final_zero = False
            cluster.update(s.get("failure_reason_counts", {}))
    return {
        "all_final_zero_across_runs": all_final_zero,
        "failure_reason_cluster": dict(cluster.most_common()),
        "dominant_failure": cluster.most_common(1)[0][0] if cluster else "none",
    }


def write_md(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Insertion-Aware Gate Multi-Seed 验证报告",
        "",
        f"- checkpoint: `{payload['meta']['checkpoint']}`",
        f"- aligned-original preserved: {payload['meta']['aligned_original_preserved']}",
        f"- physics residual repair: opt-in only (`--enable-physics-residual-repair`)",
        "",
        "## demo_4：insertion_gated vs plain top-k",
        "",
    ]
    d4 = payload["analysis"]["demo_4_insertion_vs_plain"]
    lines.extend(
        [
            f"- 对比 runs: {d4['runs']}",
            f"- insertion 更高: {d4['insertion_higher_count']} / tie {d4['tie_count']} / plain 更高 {d4['plain_higher_count']}",
            f"- mean Δfinal: {d4['mean_delta_final']:.1%}",
            f"- **稳定高于 plain**: {d4['stable_above_plain']}",
            "",
            "## demo_3：final=0 失败聚类",
            "",
        ]
    )
    d3 = payload["analysis"]["demo_3_failure_cluster"]
    lines.extend(
        [
            f"- 全部 run final=0: {d3['all_final_zero_across_runs']}",
            f"- 主导 failure: `{d3['dominant_failure']}`",
            f"- 聚类: {d3['failure_reason_cluster']}",
            "",
            "## 聚合结果（按 demo × pool × strategy）",
            "",
        ]
    )

    for demo_key in payload["meta"]["demos"]:
        lines.append(f"### {demo_key}")
        lines.append("")
        for pool_key, pool_data in payload["aggregates"][demo_key].items():
            lines.append(f"#### pool {pool_key}")
            lines.append("")
            lines.append("| strategy | mean final | mean partial | mean selected | mean gate pass |")
            lines.append("|----------|------------|--------------|---------------|----------------|")
            for strat in STRATEGIES:
                agg = pool_data[strat]
                lines.append(
                    f"| {strat} | {agg['mean_final_success_rate']:.0%} | "
                    f"{agg['mean_partial_success_rate']:.0%} | {agg['mean_selected_count']:.1f} | "
                    f"{agg['mean_gate_pass_rate']:.0%} |"
                )
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-seed insertion gate validation")
    parser.add_argument("--failed-hdf5", type=Path, default=DEFAULT_FAILED_HDF5)
    parser.add_argument("--cem-report", type=Path, default=DEFAULT_CEM_REPORT)
    parser.add_argument("--aligned-model", type=Path, default=DEFAULT_ALIGNED_MODEL)
    parser.add_argument("--v1e-model", type=Path, default=DEFAULT_PINN_MODEL)
    parser.add_argument("--success-reference", type=Path, default=DEFAULT_SUCCESS_REFERENCE_JSONL)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--demos", nargs="+", default=["demo_3", "demo_4"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--enable-physics-residual-repair", action="store_true")
    args = parser.parse_args()

    if args.enable_physics_residual_repair:
        os.environ["enable_physics_residual_repair"] = "true"
    if not is_physics_residual_repair_enabled():
        print("ERROR: --enable-physics-residual-repair required", flush=True)
        return 2

    os.environ.setdefault("MUJOCO_GL", "egl")
    all_runs: list[dict[str, Any]] = []

    for demo_key in args.demos:
        for pool in POOL_CONFIGS:
            for seed in args.seeds:
                print(
                    f"[run] {demo_key} samples={pool['num_samples']} top_k={pool['top_k']} seed={seed}",
                    flush=True,
                )
                all_runs.append(
                    run_single_config(
                        demo_key=demo_key,
                        num_samples=pool["num_samples"],
                        top_k=pool["top_k"],
                        seed=seed,
                        failed_hdf5=args.failed_hdf5,
                        cem_report=args.cem_report,
                        aligned_model=args.aligned_model,
                        v1e_model=args.v1e_model,
                        success_reference_jsonl=args.success_reference,
                    )
                )

    aggregates: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for demo_key in args.demos:
        for pool in POOL_CONFIGS:
            pool_key = f"samples={pool['num_samples']}_top_k={pool['top_k']}"
            pool_runs = [
                r
                for r in all_runs
                if r["demo_key"] == demo_key
                and r["num_samples"] == pool["num_samples"]
                and r["top_k"] == pool["top_k"]
            ]
            aggregates[demo_key][pool_key] = {
                strat: _aggregate_strategy(pool_runs, strat) for strat in STRATEGIES
            }

    payload = {
        "schema": "insertion_gated_multi_seed_report_v1",
        "meta": {
            "checkpoint": str(args.aligned_model),
            "aligned_original_preserved": True,
            "enable_physics_residual_repair": True,
            "opt_in_only": True,
            "demos": list(args.demos),
            "pool_configs": list(POOL_CONFIGS),
            "seeds": list(args.seeds),
            "strategies": list(STRATEGIES),
        },
        "runs": all_runs,
        "aggregates": aggregates,
        "analysis": {
            "demo_4_insertion_vs_plain": _demo4_insertion_vs_plain(all_runs),
            "demo_3_failure_cluster": _demo3_failure_cluster(all_runs),
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    write_md(payload, args.output_md)
    print(json.dumps({"json": str(args.output_json), "md": str(args.output_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
