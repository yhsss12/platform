#!/usr/bin/env python3
"""Task 6：V1-F-aligned-plus 评估（old demos + new failed demos）。"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

_EXPERIMENT_DIR = Path(__file__).resolve().parents[2]
_V1_DIR = _EXPERIMENT_DIR / "v1_residual_model"
_V1F_DIR = _V1_DIR / "repair_parameter_model_v1f"
_OFFLINE_DIR = _EXPERIMENT_DIR / "offline_mimicgen_repair_test"
for path in (_EXPERIMENT_DIR, _V1_DIR, _V1F_DIR, _OFFLINE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config import DEFAULT_CEM_REPORT, DEFAULT_FAILED_HDF5, DEFAULT_PINN_MODEL, DEMO_REPAIR_CONFIGS  # noqa: E402
from repair_common_v1f import (  # noqa: E402
    extract_repair_context_v1f,
    sample_repair_candidates_v1f,
    score_repair_candidates_v1f,
    select_candidate_indices_v1f,
    summarize_method_results_v1f,
    top_k_diversity_stats,
)
from repair_rollout import run_repair_rollout  # noqa: E402
from pinn_v1f_inference import clear_v1f_model_cache  # noqa: E402
from v1f_plus_utils import (  # noqa: E402
    DEFAULT_ALIGNED_MODEL,
    DEFAULT_FAILED_HDF5 as NEW_FAILED_HDF5,
    DEFAULT_PLUS_OUTPUT,
    list_demo_keys,
    load_failure_map,
    search_kind_for_failure,
)

OLD_DEMO_KEYS = ("demo_4", "demo_2", "demo_3")
OLD_FAILED_HDF5 = DEFAULT_FAILED_HDF5
DEFAULT_AUDIT = _EXPERIMENT_DIR / "outputs" / "new_100_demo_audit" / "new_demo_audit_report.json"
METHODS = ("v1f_plain_top_k", "random_top_k", "explicit_top_k")


def _repair_cfg_for_new_demo(demo_key: str, failure_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    info = failure_map.get(demo_key, {})
    coarse = info.get("coarse_failure_type", "transport_failed")
    search_kind = info.get("search_kind", search_kind_for_failure(coarse))
    active = search_kind if search_kind in ("insertion", "transport", "lift") else "grasp"
    return {
        "failure_type": coarse,
        "active": active,
        "search_kind": search_kind,
        "label": "failed",
        "rough_failure_type": info.get("rough_failure_type", "unknown"),
    }


def run_offline_repair(
    *,
    demo_key: str,
    cfg: dict[str, Any],
    v1f_model: Path,
    failed_hdf5: Path,
    cem_report: Path,
    selection_method: str,
    num_samples: int,
    top_k: int,
    seed: int,
    v1e_model: Path,
    model_label: str,
    demo_group: str,
) -> dict[str, Any]:
    pool_seed = seed + hash(demo_key) % 10000
    rng = random.Random(seed)
    context = extract_repair_context_v1f(
        context_source="original_failed_context",
        failed_hdf5=failed_hdf5,
        demo_key=demo_key,
        failure_type=cfg["failure_type"],
        search_kind=cfg["search_kind"],
        cem_report=cem_report,
    )
    candidates = sample_repair_candidates_v1f(
        search_kind=cfg["search_kind"], n_samples=num_samples, seed=pool_seed
    )
    clear_v1f_model_cache()
    score_repair_candidates_v1f(
        context=context,
        candidates=candidates,
        active=cfg["active"],
        v1e_model_path=v1e_model,
        v1f_model_path=v1f_model,
    )
    indices = select_candidate_indices_v1f(candidates, method=selection_method, top_k=top_k, rng=rng)
    rollout_results = []
    for idx in indices:
        rollout_results.append(
            run_repair_rollout(
                failed_hdf5=failed_hdf5,
                demo_key=demo_key,
                search_kind=cfg["search_kind"],
                cem_report=cem_report if cfg["search_kind"] in ("insertion", "transport") else None,
                candidate=candidates[idx],
            )
        )
    metrics = summarize_method_results_v1f(rollout_results, method=selection_method, rollout_budget=top_k)
    return {
        "demo_group": demo_group,
        "demo_key": demo_key,
        "failure_type": cfg.get("rough_failure_type", cfg["failure_type"]),
        "coarse_failure_type": cfg["failure_type"],
        "search_kind": cfg["search_kind"],
        "model_label": model_label,
        "selection_method": selection_method,
        "metrics": metrics,
    }


def evaluate_acceptance(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def _get(group: str, demo: str, model: str, method: str) -> dict[str, Any] | None:
        for r in rows:
            if (
                r["demo_group"] == group
                and r["demo_key"] == demo
                and r["model_label"] == model
                and r["selection_method"] == method
            ):
                return r
        return None

    d4_orig = _get("old", "demo_4", "aligned-plus", "v1f_plain_top_k")
    d2_orig = _get("old", "demo_2", "aligned-plus", "v1f_plain_top_k")
    new_plus = [r for r in rows if r["demo_group"] == "new" and r["model_label"] == "aligned-plus" and r["selection_method"] == "v1f_plain_top_k"]
    new_random = [r for r in rows if r["demo_group"] == "new" and r["selection_method"] == "random_top_k"]
    new_plus_by_ft: dict[str, list[float]] = {}
    new_random_by_ft: dict[str, list[float]] = {}
    for r in new_plus:
        new_plus_by_ft.setdefault(r["failure_type"], []).append(float(r["metrics"]["repair_rate_at_20"]))
    for r in new_random:
        if r["model_label"] == "aligned-plus":
            new_random_by_ft.setdefault(r["failure_type"], []).append(float(r["metrics"]["repair_rate_at_20"]))

    plus_avg = float(sum(r["metrics"]["repair_rate_at_20"] for r in new_plus) / max(len(new_plus), 1))
    random_avg = float(
        sum(r["metrics"]["repair_rate_at_20"] for r in new_random if r["model_label"] == "aligned-plus")
        / max(len([r for r in new_random if r["model_label"] == "aligned-plus"]), 1)
    )

    transport_plus = new_plus_by_ft.get("transport_failed", [])
    insertion_plus = new_plus_by_ft.get("insertion_failed", [])
    transport_random = new_random_by_ft.get("transport_failed", [])
    insertion_random = new_random_by_ft.get("insertion_failed", [])

    def _mean(vals: list[float]) -> float:
        return float(sum(vals) / max(len(vals), 1))

    return {
        "old_demo_4_repair_rate_at_20": float(d4_orig["metrics"]["repair_rate_at_20"]) if d4_orig else None,
        "old_demo_2_repair_rate_at_20": float(d2_orig["metrics"]["repair_rate_at_20"]) if d2_orig else None,
        "new_plus_avg_repair_rate_at_20": plus_avg,
        "new_random_avg_repair_rate_at_20": random_avg,
        "checks": {
            "old_demo_4_ge_0_70": (d4_orig is not None and float(d4_orig["metrics"]["repair_rate_at_20"]) >= 0.70),
            "old_demo_2_ge_0_20": (d2_orig is not None and float(d2_orig["metrics"]["repair_rate_at_20"]) >= 0.20),
            "new_plus_better_than_random": plus_avg > random_avg,
            "transport_or_insertion_improved": (
                (_mean(transport_plus) > _mean(transport_random) if transport_plus else False)
                or (_mean(insertion_plus) > _mean(insertion_random) if insertion_plus else False)
            ),
        },
        "all_pass": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate V1-F-aligned-plus")
    parser.add_argument("--aligned-original-model", type=Path, default=DEFAULT_ALIGNED_MODEL)
    parser.add_argument("--aligned-plus-model", type=Path, default=DEFAULT_PLUS_OUTPUT / "trained_model" / "model_v1f_aligned_plus.pt")
    parser.add_argument("--old-failed-hdf5", type=Path, default=OLD_FAILED_HDF5)
    parser.add_argument("--new-failed-hdf5", type=Path, default=NEW_FAILED_HDF5)
    parser.add_argument("--audit-report", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--cem-report", type=Path, default=DEFAULT_CEM_REPORT)
    parser.add_argument("--v1e-model", type=Path, default=DEFAULT_PINN_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PLUS_OUTPUT / "evaluation")
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--new-demo-limit", type=int, default=0, help="0 = all new demos")
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    failure_map = load_failure_map(args.audit_report)
    new_demo_keys = list_demo_keys(args.new_failed_hdf5)
    if args.new_demo_limit > 0:
        new_demo_keys = new_demo_keys[: args.new_demo_limit]

    rows: list[dict[str, Any]] = []
    eval_plan: list[tuple[str, str, Path, dict[str, Any], str]] = []
    for demo_key in OLD_DEMO_KEYS:
        eval_plan.append(("old", demo_key, args.old_failed_hdf5, DEMO_REPAIR_CONFIGS[demo_key], "aligned-original"))
        eval_plan.append(("old", demo_key, args.old_failed_hdf5, DEMO_REPAIR_CONFIGS[demo_key], "aligned-plus"))
    for demo_key in new_demo_keys:
        cfg = _repair_cfg_for_new_demo(demo_key, failure_map)
        eval_plan.append(("new", demo_key, args.new_failed_hdf5, cfg, "aligned-original"))
        eval_plan.append(("new", demo_key, args.new_failed_hdf5, cfg, "aligned-plus"))

    for group, demo_key, hdf5, cfg, model_label in eval_plan:
        model_path = args.aligned_original_model if model_label == "aligned-original" else args.aligned_plus_model
        if not model_path.exists():
            print(f"skip missing model {model_path}", flush=True)
            continue
        for method in METHODS:
            print(f"eval {group}/{demo_key} {model_label} {method}", flush=True)
            rows.append(
                run_offline_repair(
                    demo_key=demo_key,
                    cfg=cfg,
                    v1f_model=model_path,
                    failed_hdf5=hdf5,
                    cem_report=args.cem_report,
                    selection_method=method,
                    num_samples=args.num_samples,
                    top_k=args.top_k,
                    seed=args.seed,
                    v1e_model=args.v1e_model,
                    model_label=model_label,
                    demo_group=group,
                )
            )

    acceptance = evaluate_acceptance(rows)
    acceptance["all_pass"] = all(acceptance["checks"].values())

    report = {
        "task": "v1f_aligned_plus_evaluation",
        "aligned_original_model": str(args.aligned_original_model),
        "aligned_plus_model": str(args.aligned_plus_model),
        "old_demo_keys": list(OLD_DEMO_KEYS),
        "new_demo_keys": new_demo_keys,
        "results": rows,
        "acceptance": acceptance,
    }
    report_path = args.output_dir / "v1f_plus_evaluation_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    csv_rows = []
    for r in rows:
        m = r["metrics"]
        sk = m.get("success_at_k", {})
        csv_rows.append(
            {
                "demo_group": r["demo_group"],
                "demo_key": r["demo_key"],
                "failure_type": r["failure_type"],
                "model_label": r["model_label"],
                "selection_method": r["selection_method"],
                "repair_rate_at_20": m.get("repair_rate_at_20"),
                "success_at_1": sk.get("at_1"),
                "success_at_3": sk.get("at_3"),
                "success_at_5": sk.get("at_5"),
                "success_at_10": sk.get("at_10"),
                "success_at_20": sk.get("at_20"),
                "rollouts_per_success": m.get("rollouts_per_success"),
                "best_E_total": m.get("best_E_total"),
                "num_successes": m.get("num_successes_written"),
            }
        )
    csv_path = args.output_dir / "v1f_plus_evaluation_summary.csv"
    if csv_rows:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)

    print(json.dumps({"report": str(report_path), "acceptance": acceptance}, indent=2))
    return 0 if acceptance["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
