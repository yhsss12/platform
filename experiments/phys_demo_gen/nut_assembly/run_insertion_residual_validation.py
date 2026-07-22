#!/usr/bin/env python3
"""demo_4 insertion residual 增强验证：对比排序前后 final success。"""
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
from insertion_residuals import (  # noqa: E402
    INSERTION_RESIDUAL_KEYS,
    combined_demo4_ranking_score,
    compute_insertion_residuals,
)
from physics_residual_repair import build_physics_repair_context, is_physics_residual_repair_enabled
from physics_residuals import compute_effective_ranking_score, compute_physics_residuals
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

DEFAULT_OUTPUT = DEFAULT_PHYSICS_RESIDUAL_OUTPUT_DIR / "insertion_residual_breakdown.json"


def _rollout_pool(
    *,
    demo_key: str,
    cfg: dict[str, Any],
    failed_hdf5: Path,
    cem_report: Path,
    aligned_model: Path,
    v1e_model: Path,
    context: dict[str, Any],
    num_samples: int,
    top_k: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[int], dict[str, Any]]:
    candidates = sample_repair_candidates_v1f(
        search_kind=cfg["search_kind"], n_samples=num_samples, seed=seed + hash(demo_key) % 10000
    )
    score_repair_candidates_v1f(
        context=context,
        candidates=candidates,
        active=cfg["active"],
        v1e_model_path=v1e_model,
        v1f_model_path=aligned_model,
    )
    pinn_top = select_candidate_indices_v1f(
        candidates, method="v1f_plain_top_k", top_k=top_k, rng=random.Random(seed)
    )
    for idx in pinn_top:
        if not candidates[idx].get("rollout"):
            candidates[idx]["rollout"] = run_repair_rollout(
                failed_hdf5=failed_hdf5,
                demo_key=demo_key,
                search_kind=cfg["search_kind"],
                cem_report=cem_report,
                candidate=candidates[idx],
            )
    original = _run_original_baseline_rollout(demo_key=demo_key, cfg=cfg, failed_hdf5=failed_hdf5)
    physics_ctx = build_physics_repair_context(
        base_context=context, success_reference_jsonl=DEFAULT_SUCCESS_REFERENCE_JSONL
    )
    original_br = compute_physics_residuals(original, physics_ctx)
    return candidates, pinn_top, original_br


def _select_by_physics_only(
    candidates: list[dict[str, Any]],
    indices: list[int],
    *,
    context: dict[str, Any],
    original_br: dict[str, Any],
    top_k: int,
) -> list[int]:
    scored: list[tuple[int, float]] = []
    for idx in indices:
        br = compute_physics_residuals(candidates[idx]["rollout"], context)
        eff, _, _ = compute_effective_ranking_score(br, original_br)
        scored.append((idx, eff))
    scored.sort(key=lambda x: x[1])
    return [i for i, _ in scored[:top_k]]


def _select_by_physics_plus_insertion(
    candidates: list[dict[str, Any]],
    indices: list[int],
    *,
    context: dict[str, Any],
    original_br: dict[str, Any],
    top_k: int,
) -> list[int]:
    scored: list[tuple[int, float]] = []
    for idx in indices:
        rollout = candidates[idx]["rollout"]
        pbr = compute_physics_residuals(rollout, context)
        ibr = compute_insertion_residuals(rollout, context)
        eff, _, _ = compute_effective_ranking_score(pbr, original_br)
        pbr_eff = dict(pbr)
        pbr_eff["ranking_score"] = eff
        score = combined_demo4_ranking_score(pbr_eff, ibr)
        scored.append((idx, score))
    scored.sort(key=lambda x: x[1])
    return [i for i, _ in scored[:top_k]]


def _summarize_rollouts(
    candidates: list[dict[str, Any]],
    indices: list[int],
    *,
    context: dict[str, Any],
    original_br: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    finals = []
    records = []
    for rank, idx in enumerate(indices, start=1):
        rollout = candidates[idx]["rollout"]
        outcome = evaluate_rollout_outcome(rollout, context)
        pbr = compute_physics_residuals(rollout, context)
        ibr = compute_insertion_residuals(rollout, context)
        eff, eff_keys, eff_meta = compute_effective_ranking_score(pbr, original_br)
        rec = {
            "label": f"{label}_{rank:02d}",
            "candidate_index": idx,
            "final_success": outcome["final_success"],
            "partial_success": outcome["partial_success"],
            "failure_reason": outcome["failure_reason"],
            "physics_ranking_score": eff,
            "effective_ranking_keys": eff_keys,
            "effective_ranking_meta": eff_meta,
            "insertion_total_score": ibr["insertion_total_score"],
            "insertion_residuals": ibr["residuals"],
            "physics_residuals": pbr["residuals"],
        }
        records.append(rec)
        finals.append(outcome["final_success"])
    n = max(len(finals), 1)
    return {
        "strategy": label,
        "num_candidates": len(indices),
        "final_success_rate": float(sum(finals) / n),
        "partial_success_rate": float(sum(r["partial_success"] for r in records) / n),
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="demo_4 insertion residual validation")
    parser.add_argument("--failed-hdf5", type=Path, default=DEFAULT_FAILED_HDF5)
    parser.add_argument("--cem-report", type=Path, default=DEFAULT_CEM_REPORT)
    parser.add_argument("--aligned-model", type=Path, default=DEFAULT_ALIGNED_MODEL)
    parser.add_argument("--v1e-model", type=Path, default=DEFAULT_PINN_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
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
    demo_key = "demo_4"
    cfg = DEMO_REPAIR_CONFIGS[demo_key]
    base_ctx = extract_baseline_context_v1f(
        failed_hdf5=args.failed_hdf5,
        demo_key=demo_key,
        failure_type=cfg["failure_type"],
        search_kind=cfg["search_kind"],
    )
    ctx = build_physics_repair_context(
        base_context=base_ctx, success_reference_jsonl=DEFAULT_SUCCESS_REFERENCE_JSONL
    )

    candidates, pinn_top, original_br = _rollout_pool(
        demo_key=demo_key,
        cfg=cfg,
        failed_hdf5=args.failed_hdf5,
        cem_report=args.cem_report,
        aligned_model=args.aligned_model,
        v1e_model=args.v1e_model,
        context=base_ctx,
        num_samples=args.num_samples,
        top_k=args.top_k,
        seed=args.seed,
    )

    before_indices = _select_by_physics_only(
        candidates, pinn_top, context=ctx, original_br=original_br, top_k=args.top_k
    )
    after_indices = _select_by_physics_plus_insertion(
        candidates, pinn_top, context=ctx, original_br=original_br, top_k=args.top_k
    )

    before = _summarize_rollouts(candidates, before_indices, context=ctx, original_br=original_br, label="physics_only")
    after = _summarize_rollouts(
        candidates, after_indices, context=ctx, original_br=original_br, label="physics_plus_insertion"
    )

    payload = {
        "demo_key": demo_key,
        "insertion_residual_keys": list(INSERTION_RESIDUAL_KEYS),
        "baseline_final_success_rate_physics_only": before["final_success_rate"],
        "enhanced_final_success_rate": after["final_success_rate"],
        "final_success_delta": after["final_success_rate"] - before["final_success_rate"],
        "before": before,
        "after": after,
        "original_baseline": {
            "raw_total_score": original_br["raw_total_score"],
            "residuals": original_br["residuals"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "delta_final": payload["final_success_delta"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
