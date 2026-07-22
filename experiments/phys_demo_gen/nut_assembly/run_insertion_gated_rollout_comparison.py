#!/usr/bin/env python3
"""demo_3/demo_4 四策略 rollout 对比 + insertion_gate_breakdown.json。"""
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
    DEFAULT_PHYSICS_RESIDUAL_OUTPUT_DIR,
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

DEFAULT_GATE_JSON = DEFAULT_PHYSICS_RESIDUAL_OUTPUT_DIR / "insertion_gate_breakdown.json"
DEFAULT_REPORT_JSON = DEFAULT_PHYSICS_RESIDUAL_OUTPUT_DIR / "insertion_gated_rollout_comparison.json"
DEFAULT_REPORT_MD = DEFAULT_PHYSICS_RESIDUAL_OUTPUT_DIR / "insertion_gated_rollout_comparison.md"

COMPARISON_STRATEGIES = (
    "aligned-original",
    "physics_residual_gated_top_k",
    "physics_residual_p1p2_gated_top_k",
    "physics_residual_insertion_gated_top_k",
)


def _select_for_strategy(
    *,
    strategy: str,
    candidates: list[dict[str, Any]],
    pinn_top: list[int],
    context: dict[str, Any],
    original_breakdown: dict[str, Any],
    original_rollout: dict[str, Any],
    top_k: int,
) -> tuple[list[int], list[dict[str, Any]] | None]:
    if strategy == "aligned-original":
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
    raise ValueError(f"unknown strategy: {strategy}")


def run_demo_comparison(
    *,
    demo_key: str,
    failed_hdf5: Path,
    cem_report: Path,
    aligned_model: Path,
    v1e_model: Path,
    success_reference_jsonl: Path,
    num_samples: int,
    top_k: int,
    seed: int,
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
    original_outcome = evaluate_rollout_outcome(original_rollout, ctx)

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

    strategy_summary: dict[str, Any] = {}
    strategy_records: dict[str, list[dict[str, Any]]] = {}
    gate_breakdown: dict[str, Any] = {"demo_key": demo_key, "candidates": []}

    for strategy in COMPARISON_STRATEGIES:
        selected, gate_records = _select_for_strategy(
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
            if strategy == "physics_residual_p1p2_gated_top_k":
                src = check_source_consistency(br, original_breakdown)
                passed, checks = candidate_passes_p1p2_gate(br, original_breakdown, source_consistency=src)
                rec["physics_gate_passed"] = passed
                rec["physics_gate_checks"] = checks
                rec["gate_mode"] = "p1p2"
            records.append(rec)

        strategy_records[strategy] = records
        strategy_summary[strategy] = summarize_rollout_strategy(
            demo_key=demo_key,
            strategy=strategy,
            rollout_records=records,
            original_breakdown=original_breakdown,
        )
        print(
            f"[compare] {demo_key}/{strategy} n={len(records)} "
            f"partial={strategy_summary[strategy]['partial_success_rate']:.0%} "
            f"final={strategy_summary[strategy]['final_success_rate']:.0%}",
            flush=True,
        )

        if gate_records is not None:
            gate_breakdown["candidates"] = gate_records
            gate_breakdown["accepted_count"] = sum(1 for r in gate_records if r.get("accepted"))
            gate_breakdown["rejected_p1p2"] = sum(
                1 for r in gate_records if r.get("rejection_stage") == "p1p2_gate"
            )
            gate_breakdown["rejected_insertion"] = sum(
                1 for r in gate_records if r.get("rejection_stage") == "insertion_gate"
            )

    return {
        "demo_key": demo_key,
        "config": {
            "checkpoint": str(aligned_model),
            "num_samples": num_samples,
            "top_k": top_k,
            "seed": seed,
            "failed_hdf5": str(failed_hdf5),
        },
        "original_baseline": {**original_outcome, "raw_total_score": original_breakdown["raw_total_score"]},
        "strategies": strategy_summary,
        "records": strategy_records,
        "insertion_gate_breakdown": gate_breakdown if demo_key in ("demo_3", "demo_4") else None,
    }


def write_md(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Insertion-Aware Gated Rollout 对比",
        "",
        f"- checkpoint: `{payload['meta']['checkpoint']}`",
        f"- aligned-original preserved: {payload['meta']['aligned_original_preserved']}",
        "",
        "## 策略对比",
        "",
    ]
    for demo_key in payload["demos"]:
        lines.append(f"### {demo_key}")
        lines.append("")
        lines.append("| 策略 | partial | final | n |")
        lines.append("|------|---------|-------|---|")
        for strat in COMPARISON_STRATEGIES:
            s = payload["results"][demo_key]["strategies"].get(strat, {})
            lines.append(
                f"| {strat} | {s.get('partial_success_rate', 0):.0%} | "
                f"{s.get('final_success_rate', 0):.0%} | {s.get('num_rollouts', 0)} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Insertion gated rollout comparison")
    parser.add_argument("--failed-hdf5", type=Path, default=DEFAULT_FAILED_HDF5)
    parser.add_argument("--cem-report", type=Path, default=DEFAULT_CEM_REPORT)
    parser.add_argument("--aligned-model", type=Path, default=DEFAULT_ALIGNED_MODEL)
    parser.add_argument("--v1e-model", type=Path, default=DEFAULT_PINN_MODEL)
    parser.add_argument("--success-reference", type=Path, default=DEFAULT_SUCCESS_REFERENCE_JSONL)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--gate-json", type=Path, default=DEFAULT_GATE_JSON)
    parser.add_argument("--demos", nargs="+", default=["demo_3", "demo_4"])
    parser.add_argument("--num-samples", type=int, default=80)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--enable-physics-residual-repair", action="store_true")
    args = parser.parse_args()

    if args.enable_physics_residual_repair:
        os.environ["enable_physics_residual_repair"] = "true"
    if not is_physics_residual_repair_enabled():
        print("ERROR: --enable-physics-residual-repair required", flush=True)
        return 2

    os.environ.setdefault("MUJOCO_GL", "egl")
    results: dict[str, Any] = {}
    gate_payload: dict[str, Any] = {"schema": "insertion_gate_breakdown_v1", "demos": {}}

    for demo_key in args.demos:
        print(f"[start] {demo_key}", flush=True)
        demo_result = run_demo_comparison(
            demo_key=demo_key,
            failed_hdf5=args.failed_hdf5,
            cem_report=args.cem_report,
            aligned_model=args.aligned_model,
            v1e_model=args.v1e_model,
            success_reference_jsonl=args.success_reference,
            num_samples=args.num_samples,
            top_k=args.top_k,
            seed=args.seed,
        )
        results[demo_key] = demo_result
        if demo_result.get("insertion_gate_breakdown"):
            gate_payload["demos"][demo_key] = demo_result["insertion_gate_breakdown"]

    payload = {
        "schema": "insertion_gated_rollout_comparison_v1",
        "meta": {
            "checkpoint": str(args.aligned_model),
            "aligned_original_preserved": True,
            "strategies": list(COMPARISON_STRATEGIES),
            "num_samples": args.num_samples,
            "top_k": args.top_k,
            "seed": args.seed,
        },
        "demos": list(args.demos),
        "results": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    args.gate_json.write_text(json.dumps(gate_payload, indent=2, default=str), encoding="utf-8")
    write_md(payload, args.output_md)
    print(json.dumps({"json": str(args.output_json), "gate": str(args.gate_json), "md": str(args.output_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
