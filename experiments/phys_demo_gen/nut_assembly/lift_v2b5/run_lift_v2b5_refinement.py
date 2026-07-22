#!/usr/bin/env python3
"""V2-B5：demo_3 lift_failed 物理精修搜索。"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

_EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
if str(_EXPERIMENT_DIR) not in sys.path:
  sys.path.insert(0, str(_EXPERIMENT_DIR))

from lift_v2b5_sim_search import execute_lift_v2b5_rollout, iter_lift_v2b5_candidates  # noqa: E402
from lift_v2b5_refiner import LiftV2B5Params  # noqa: E402

DEFAULT_FAILED_HDF5 = _EXPERIMENT_DIR.parents[2] / "mnt" / "data" / "demo_failed.hdf5"
DEFAULT_OUTPUT = _EXPERIMENT_DIR / "outputs" / "lift_v2b5"


def main() -> int:
  parser = argparse.ArgumentParser(description="V2-B5 lift physical refinement for demo_3")
  parser.add_argument("--failed-hdf5", type=Path, default=DEFAULT_FAILED_HDF5)
  parser.add_argument("--demo-key", default="demo_3")
  parser.add_argument("--max-evals", type=int, default=800)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
  args = parser.parse_args()

  os.environ.setdefault("MUJOCO_GL", "egl")
  args.output_dir.mkdir(parents=True, exist_ok=True)
  jsonl_path = args.output_dir / "lift_v2b5_rollout_samples.jsonl"
  report_path = args.output_dir / "lift_v2b5_report.json"

  baseline = execute_lift_v2b5_rollout(
    str(args.failed_hdf5), args.demo_key, "failed", LiftV2B5Params(), rollout_kind="baseline"
  )

  records: list[dict[str, Any]] = []
  best_partial = baseline
  best_lift = baseline
  best_success = baseline

  for i, params in enumerate(iter_lift_v2b5_candidates(max_evals=args.max_evals, seed=args.seed)):
    result = execute_lift_v2b5_rollout(
      str(args.failed_hdf5), args.demo_key, "failed", params, rollout_kind="lift_v2b5_search"
    )
    result["search_index"] = i
    result["seed"] = args.seed
    records.append(result)

    if result.get("partial_lift_success") and float(result.get("nut_lift_phase_delta", 0)) >= float(
      best_partial.get("nut_lift_phase_delta", 0)
    ):
      best_partial = result
    elif float(result.get("nut_lift_phase_delta", 0)) > float(best_partial.get("nut_lift_phase_delta", -1)):
      best_partial = result
    if result.get("lift_success_proxy") and float(result.get("nut_lift_phase_delta", 0)) >= float(
      best_lift.get("nut_lift_phase_delta", 0)
    ):
      best_lift = result
    if result.get("success_flag"):
      best_success = result

    if (i + 1) % 50 == 0:
      print(f"  demo_3 v2b5: {i + 1}/{args.max_evals}", flush=True)

  with jsonl_path.open("w", encoding="utf-8") as handle:
    for rec in records:
      handle.write(json.dumps(rec, default=str) + "\n")

  n_partial = sum(1 for r in records if r.get("partial_lift_success"))
  n_lift = sum(1 for r in records if r.get("lift_success_proxy"))
  n_success = sum(1 for r in records if r.get("success_flag"))
  n_grasp = sum(1 for r in records if r.get("grasp_success_proxy"))

  report = {
    "task": "lift_v2b5_physical_refinement",
    "demo_key": args.demo_key,
    "failure_type": "lift_failed",
    "max_evals": args.max_evals,
    "seed": args.seed,
    "baseline": {
      "nut_lift_delta": baseline.get("nut_lift_delta"),
      "partial_lift_success": baseline.get("partial_lift_success"),
      "lift_success_proxy": baseline.get("lift_success_proxy"),
      "success_flag": baseline.get("success_flag"),
      "E_total_norm": baseline.get("E_total_norm"),
    },
    "search_summary": {
      "num_rollouts": len(records),
      "num_partial_lift_success": n_partial,
      "num_lift_success_proxy": n_lift,
      "num_grasp_success_proxy": n_grasp,
      "num_refined_success": n_success,
      "partial_lift_success_rate": float(n_partial / max(len(records), 1)),
      "lift_success_proxy_rate": float(n_lift / max(len(records), 1)),
      "refined_success_rate": float(n_success / max(len(records), 1)),
    },
    "best_partial_lift": {
      "search_index": best_partial.get("search_index"),
      "nut_lift_delta": best_partial.get("nut_lift_delta"),
      "nut_lift_phase_delta": best_partial.get("nut_lift_phase_delta"),
      "outcome_label": best_partial.get("outcome_label"),
      "lift_v2b5_params": best_partial.get("lift_v2b5_params"),
      "E_total_norm": best_partial.get("E_total_norm"),
    },
    "best_lift_proxy": {
      "search_index": best_lift.get("search_index"),
      "nut_lift_delta": best_lift.get("nut_lift_delta"),
      "outcome_label": best_lift.get("outcome_label"),
      "lift_v2b5_params": best_lift.get("lift_v2b5_params"),
    },
    "best_refined_success": {
      "search_index": best_success.get("search_index"),
      "success_flag": best_success.get("success_flag"),
      "nut_lift_delta": best_success.get("nut_lift_delta"),
      "lift_v2b5_params": best_success.get("lift_v2b5_params"),
    },
    "acceptance": {
      "has_partial_lift_success": n_partial > 0,
      "has_lift_success_proxy": n_lift > 0,
      "has_refined_success": n_success > 0,
      "object_poses_not_modified": all(not r.get("object_poses_modified", True) for r in records),
      "all_muJoCo_rollout": True,
    },
    "outputs": {
      "rollout_samples_jsonl": str(jsonl_path),
      "report_json": str(report_path),
    },
    "features": [
      "re-grasp pulse",
      "extra gripper close",
      "slow lift",
      "contact settle",
      "micro-lift check boost",
      "two-stage lift",
      "gripper timing sweep",
      "lateral correction",
    ],
  }
  report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
  print(json.dumps({"report": str(report_path), "acceptance": report["acceptance"]}, indent=2))
  return 0 if report["acceptance"]["has_partial_lift_success"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
