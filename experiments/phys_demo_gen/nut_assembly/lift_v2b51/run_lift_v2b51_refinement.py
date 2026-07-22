#!/usr/bin/env python3
"""V2-B5.1：demo_3 contact-aware lift targeted search。"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

_EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
_V2B51_DIR = Path(__file__).resolve().parent
for path in (_EXPERIMENT_DIR, _V2B51_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from lift_v2b51_sim_search import (  # noqa: E402
    execute_lift_v2b51_rollout,
    iter_lift_v2b51_candidates,
)
from lift_v2b51_refiner import CONTACT_AWARE_SEED_PARAMS, LiftV2B51Params  # noqa: E402

DEFAULT_FAILED_HDF5 = _EXPERIMENT_DIR.parents[2] / "mnt" / "data" / "demo_failed.hdf5"
DEFAULT_OUTPUT = _EXPERIMENT_DIR / "outputs" / "lift_v2b51"
PARTIAL_LIFT_DELTA_GOAL = 0.005


def _score_candidate(result: dict[str, Any]) -> tuple[float, float, float]:
    """partial > lift proxy > contact energy（越小越好）。"""
    partial = 1.0 if result.get("partial_lift_success") else 0.0
    nut_lift = float(result.get("nut_z_lift_delta", result.get("nut_lift_phase_delta", 0.0)))
    contact_total = float(result.get("E_contact_aware_total", 999.0))
    return (partial, nut_lift, -contact_total)


def main() -> int:
    parser = argparse.ArgumentParser(description="V2-B5.1 contact-aware lift search for demo_3")
    parser.add_argument("--failed-hdf5", type=Path, default=DEFAULT_FAILED_HDF5)
    parser.add_argument("--demo-key", default="demo_3")
    parser.add_argument("--max-evals", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / "lift_v2b51_rollout_samples.jsonl"
    diag_path = args.output_dir / "lift_v2b51_diagnostics_summary.json"
    report_path = args.output_dir / "lift_v2b51_report.json"

    baseline = execute_lift_v2b51_rollout(
        str(args.failed_hdf5), args.demo_key, "failed", LiftV2B51Params(), rollout_kind="baseline"
    )
    seed_rollout = execute_lift_v2b51_rollout(
        str(args.failed_hdf5),
        args.demo_key,
        "failed",
        CONTACT_AWARE_SEED_PARAMS,
        rollout_kind="contact_aware_seed",
    )

    records: list[dict[str, Any]] = [baseline, seed_rollout]
    best = seed_rollout if _score_candidate(seed_rollout) > _score_candidate(baseline) else baseline

    for i, params in enumerate(iter_lift_v2b51_candidates(max_evals=args.max_evals, seed=args.seed)):
        result = execute_lift_v2b51_rollout(
            str(args.failed_hdf5), args.demo_key, "failed", params, rollout_kind="lift_v2b51_search"
        )
        result["search_index"] = i
        result["seed"] = args.seed
        records.append(result)
        if _score_candidate(result) > _score_candidate(best):
            best = result
        if (i + 1) % 50 == 0:
            n_partial = sum(1 for r in records if r.get("partial_lift_success"))
            print(
                f"  demo_3 v2b51: {i + 1}/{args.max_evals} partial={n_partial} best_nut_lift={best.get('nut_z_lift_delta', 0):.4f}",
                flush=True,
            )

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for rec in records:
            slim = {k: v for k, v in rec.items() if not k.startswith("per_step_")}
            handle.write(json.dumps(slim, default=str) + "\n")

    n_partial = sum(1 for r in records if r.get("partial_lift_success"))
    n_lift = sum(1 for r in records if r.get("lift_success_proxy"))
    n_success = sum(1 for r in records if r.get("success_flag"))
    n_bilateral = sum(1 for r in records if int(r.get("bilateral_contact_steps", 0)) > 0)

    diag_summary = {
        "baseline": {k: baseline.get(k) for k in baseline if k.startswith(("gripper_", "left_", "right_", "bilateral_", "contact_", "eef_", "nut_", "partial_"))},
        "best": {k: best.get(k) for k in best if k.startswith(("gripper_", "left_", "right_", "bilateral_", "contact_", "eef_", "nut_", "partial_", "E_contact"))},
        "best_outcome": best.get("outcome_label"),
        "best_nut_z_lift_delta": best.get("nut_z_lift_delta"),
        "num_with_bilateral_contact": n_bilateral,
    }
    diag_path.write_text(json.dumps(diag_summary, indent=2), encoding="utf-8")

    report = {
        "task": "lift_v2b51_contact_aware_refinement",
        "demo_key": args.demo_key,
        "failure_type": "transport_failed",
        "secondary_failure_type": "lift_underdeveloped",
        "legacy_failure_type": "lift_failed",
        "max_evals": args.max_evals,
        "seed": args.seed,
        "partial_lift_delta_goal_m": PARTIAL_LIFT_DELTA_GOAL,
        "baseline": {
            "nut_z_lift_delta": baseline.get("nut_z_lift_delta"),
            "partial_lift_success": baseline.get("partial_lift_success"),
            "bilateral_contact_steps": baseline.get("bilateral_contact_steps"),
            "E_contact_aware_total": baseline.get("E_contact_aware_total"),
        },
        "search_summary": {
            "num_rollouts": len(records),
            "num_partial_lift_success": n_partial,
            "num_lift_success_proxy": n_lift,
            "num_refined_success": n_success,
            "num_with_bilateral_contact": n_bilateral,
            "partial_lift_success_rate": float(n_partial / max(len(records), 1)),
        },
        "best_partial_lift": {
            "search_index": best.get("search_index"),
            "outcome_label": best.get("outcome_label"),
            "nut_z_lift_delta": best.get("nut_z_lift_delta"),
            "nut_lift_phase_delta": best.get("nut_lift_phase_delta"),
            "partial_lift_success": best.get("partial_lift_success"),
            "bilateral_contact_steps": best.get("bilateral_contact_steps"),
            "contact_duration": best.get("contact_duration"),
            "nut_eef_coupling_ratio": best.get("nut_eef_coupling_ratio"),
            "lift_v2b51_params": best.get("lift_v2b51_params"),
            "enabled_templates": best.get("enabled_templates"),
            "E_contact_aware_total": best.get("E_contact_aware_total"),
        },
        "acceptance": {
            "has_partial_lift_success": n_partial > 0,
            "best_nut_lift_delta_ge_goal": float(best.get("nut_z_lift_delta", 0.0)) >= PARTIAL_LIFT_DELTA_GOAL,
            "has_refined_success": n_success > 0,
            "object_poses_not_modified": all(not r.get("object_poses_modified", True) for r in records),
        },
        "templates": [
            "lower_approach",
            "squeeze_close",
            "contact_settle",
            "micro_lift",
            "reclose_after_micro_lift",
            "slow_lift",
            "two_stage_lift",
            "lateral_correction",
        ],
        "contact_energies": [
            "E_contact_presence",
            "E_bilateral_contact",
            "E_contact_duration",
            "E_lift_follow",
            "E_nut_eef_coupling",
            "E_lift_stability",
            "E_slow_lift_smoothness",
        ],
        "outputs": {
            "rollout_samples_jsonl": str(jsonl_path),
            "diagnostics_summary_json": str(diag_path),
            "report_json": str(report_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(report_path), "acceptance": report["acceptance"]}, indent=2))
    return 0 if report["acceptance"]["has_partial_lift_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
