#!/usr/bin/env python3
"""V1-F-100Base：对 77 条 success demo 做 baseline replay，生成 success reference 样本。"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

_V1F_DIR = Path(__file__).resolve().parent
for path in (_V1F_DIR.parent.parent, _V1F_DIR.parent, _V1F_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_v1f_repair_dataset import _rollout_record_to_sample  # noqa: E402
from osc_action_converter import SimLoopParams  # noqa: E402
from sim_in_loop_refiner import build_refined_waypoints_from_hdf5, execute_waypoint_rollout  # noqa: E402
from v1f_100base_utils import DEFAULT_CEM_REPORT, DEFAULT_SUCCESS_HDF5, DEFAULT_SUCCESS_REFERENCE  # noqa: E402
from v1f_plus_utils import list_demo_keys, load_theta_or_default  # noqa: E402
from v1f_repair_dataset import extract_failed_context  # noqa: E402


def _success_context(demo_key: str, rollout: dict[str, Any]) -> dict[str, Any]:
    ctx = extract_failed_context(rollout, demo_key=demo_key, failure_type="success")
    ctx["source_failure_type"] = "success"
    return ctx


def run_success_reference(
    *,
    success_hdf5: Path,
    cem_report: Path,
    output: Path,
    perturb_budget: int = 2,
    seed: int = 0,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []
    output.parent.mkdir(parents=True, exist_ok=True)

    for demo_key in list_demo_keys(success_hdf5):
        theta = load_theta_or_default(cem_report, demo_key, search_kind="insertion")
        _proxy, _orig, target_eef, gripper = build_refined_waypoints_from_hdf5(
            str(success_hdf5), demo_key, "success", theta
        )
        baseline = execute_waypoint_rollout(
            str(success_hdf5),
            demo_key,
            "success",
            target_eef,
            gripper,
            SimLoopParams(),
            rollout_kind="v1f_100base_success_baseline",
            theta=theta,
        )
        baseline["success_flag"] = True
        context = _success_context(demo_key, baseline)
        row = {
            "source_file": str(success_hdf5),
            "source_demo": demo_key,
            "original_failure_type": "success",
            "context_mode": "success_reference",
            "repair_theta": theta if isinstance(theta, dict) else {},
            "success_flag": True,
            "E_xy": float(baseline.get("E_xy_norm", 0.0)),
            "E_transport": float(baseline.get("E_transport_norm", 0.0)),
            "E_yaw": float(baseline.get("E_yaw_norm", 0.0)),
            "E_z": float(baseline.get("E_z_norm", 0.0)),
            "E_grasp": float(baseline.get("E_grasp_norm", 0.0)),
            "E_lift": float(baseline.get("E_lift_norm", 0.0)),
            "E_smooth": float(baseline.get("E_smooth_norm", 0.0)),
            "E_total": float(baseline.get("E_total_norm", 0.0)),
            "context": context,
            "rollout": baseline,
            "active": "insertion",
            "demo_key": demo_key,
            "source": "v1f_100base_success_reference",
        }
        records.append(row)

        for j in range(perturb_budget):
            sim = SimLoopParams(insertion_speed_scale=float(rng.choice([0.5, 0.75, 1.0])))
            perturbed = execute_waypoint_rollout(
                str(success_hdf5),
                demo_key,
                "success",
                target_eef,
                gripper,
                sim,
                rollout_kind="v1f_100base_success_perturb",
                theta=theta,
            )
            perturbed["success_flag"] = True
            prow = dict(row)
            prow["rollout"] = perturbed
            prow["E_total"] = float(perturbed.get("E_total_norm", 0.0))
            prow["source"] = "v1f_100base_success_perturb"
            prow["context"] = _success_context(demo_key, perturbed)
            records.append(prow)

        print(f"[100base-success] {demo_key} E={row['E_total']:.3f}", flush=True)

    with output.open("w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec, default=str) + "\n")
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="V1-F-100Base success reference sampling")
    parser.add_argument("--success-hdf5", type=Path, default=DEFAULT_SUCCESS_HDF5)
    parser.add_argument("--cem-report", type=Path, default=DEFAULT_CEM_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_SUCCESS_REFERENCE)
    parser.add_argument("--perturb-budget", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    records = run_success_reference(
        success_hdf5=args.success_hdf5,
        cem_report=args.cem_report,
        output=args.output,
        perturb_budget=args.perturb_budget,
        seed=args.seed,
    )
    e_vals = [r["E_total"] for r in records if r["source"] == "v1f_100base_success_reference"]
    summary = {
        "num_samples": len(records),
        "num_success_demos": len(e_vals),
        "success_E_total_mean": float(np.mean(e_vals)) if e_vals else None,
        "success_E_total_p95": float(np.percentile(e_vals, 95)) if e_vals else None,
        "output": str(args.output),
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
