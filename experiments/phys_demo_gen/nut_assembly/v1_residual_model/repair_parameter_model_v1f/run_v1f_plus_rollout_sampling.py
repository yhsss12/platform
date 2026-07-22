#!/usr/bin/env python3
"""Task 3：对新 failed demo 做 MuJoCo repair rollout 采样。"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

_V1F_DIR = Path(__file__).resolve().parent
_V1_DIR = _V1F_DIR.parent
_EXPERIMENT_DIR = _V1_DIR.parent
for path in (_EXPERIMENT_DIR, _V1_DIR, _V1F_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from grasp_sim_search import execute_grasp_rollout, iter_grasp_candidates  # noqa: E402
from grasp_waypoint_builder import GraspSearchParams  # noqa: E402
from lift_energy_model import merge_v1f_energy_targets  # noqa: E402
from lift_sim_search import execute_lift_rollout, iter_lift_candidates  # noqa: E402
from lift_waypoint_refiner import LiftRepairParams  # noqa: E402
from osc_action_converter import SEARCH_SPACE, SimLoopParams  # noqa: E402
from sim_in_loop_refiner import (  # noqa: E402
    build_refined_waypoints_from_hdf5,
    execute_waypoint_rollout,
)
from transport_sim_search import execute_transport_rollout, iter_transport_candidates  # noqa: E402
from transport_waypoint_builder import TransportSearchParams  # noqa: E402
from v1f_plus_utils import (  # noqa: E402
    DEFAULT_CEM_REPORT,
    DEFAULT_FAILED_HDF5,
    DEFAULT_PLUS_OUTPUT,
    list_demo_keys,
    load_failure_map,
    load_theta_or_default,
)
from v1f_repair_dataset import extract_failed_context  # noqa: E402

DEFAULT_AUDIT = _EXPERIMENT_DIR / "outputs" / "new_100_demo_audit" / "new_demo_audit_report.json"


def _baseline_rollout(hdf5: Path, demo_key: str, active: str, cem_report: Path) -> dict[str, Any]:
    label = "failed"
    if active == "insertion":
        theta = load_theta_or_default(cem_report, demo_key, search_kind="insertion")
        _proxy, _orig, target_eef, gripper = build_refined_waypoints_from_hdf5(
            str(hdf5), demo_key, label, theta
        )
        return execute_waypoint_rollout(
            str(hdf5), demo_key, label, target_eef, gripper, SimLoopParams(),
            rollout_kind="baseline_original",
            theta=theta,
        )
    if active == "transport":
        theta = load_theta_or_default(cem_report, demo_key, search_kind="transport")
        return execute_transport_rollout(
            str(hdf5), demo_key, label, theta, TransportSearchParams(), rollout_kind="baseline_original"
        )
    if active == "grasp":
        return execute_grasp_rollout(str(hdf5), demo_key, label, GraspSearchParams(), rollout_kind="baseline_original")
    return execute_lift_rollout(str(hdf5), demo_key, label, LiftRepairParams(), rollout_kind="baseline_original")


def _sample_rollout(
    hdf5: Path,
    demo_key: str,
    active: str,
    params: Any,
    cem_report: Path,
    seed: int,
    index: int,
) -> dict[str, Any]:
    label = "failed"
    if active == "insertion":
        theta = load_theta_or_default(cem_report, demo_key, search_kind="insertion")
        _proxy, _orig, target_eef, gripper = build_refined_waypoints_from_hdf5(
            str(hdf5), demo_key, label, theta
        )
        sim_params = SimLoopParams(**{k: params[k] for k in SEARCH_SPACE if k in params})
        result = execute_waypoint_rollout(
            str(hdf5),
            demo_key,
            label,
            target_eef,
            gripper,
            sim_params,
            rollout_kind="v1f_plus_insertion_repair",
            theta=theta,
        )
        result["sim_params"] = sim_params.to_dict()
    elif active == "transport":
        theta = load_theta_or_default(cem_report, demo_key, search_kind="transport")
        result = execute_transport_rollout(
            str(hdf5),
            demo_key,
            label,
            theta,
            params,
            rollout_kind="v1f_plus_transport_repair",
        )
        result.update(merge_v1f_energy_targets(result))
    elif active == "grasp":
        result = execute_grasp_rollout(
            str(hdf5), demo_key, label, params, rollout_kind="v1f_plus_grasp_repair"
        )
        result.update(merge_v1f_energy_targets(result))
    else:
        result = execute_lift_rollout(
            str(hdf5), demo_key, label, params, rollout_kind="v1f_plus_lift_repair"
        )

    result["sampling_seed"] = seed
    result["sampling_index"] = index
    result["active_param_group"] = active
    result["demo_key"] = demo_key
    result["object_poses_modified"] = False
    result["success_from_rollout"] = True
    return result


def _iter_candidates(active: str, budget: int, seed: int, demo_key: str):
    if active == "insertion":
        rng = np.random.default_rng(seed + hash(demo_key) % 10000)
        for _ in range(budget):
            yield {k: rng.choice(SEARCH_SPACE[k]) for k in SEARCH_SPACE}
    elif active == "transport":
        yield from iter_transport_candidates(mode="random", max_evals=budget, seed=seed)
    elif active == "grasp":
        yield from iter_grasp_candidates(mode="random", max_evals=budget, seed=seed)
    else:
        yield from iter_lift_candidates(mode="random", max_evals=budget, seed=seed)


def _rollout_record_to_jsonl_row(
    *,
    source_file: Path,
    demo_key: str,
    failure_type: str,
    context: dict[str, Any],
    rollout: dict[str, Any],
    active: str,
) -> dict[str, Any]:
    repair_theta: dict[str, Any] = {}
    if active == "insertion":
        repair_theta = rollout.get("sim_params", {})
    elif active == "transport":
        repair_theta = rollout.get("transport_params", rollout.get("params", {}))
        if hasattr(repair_theta, "to_dict"):
            repair_theta = repair_theta.to_dict()
    elif active == "grasp":
        repair_theta = rollout.get("grasp_params", {})
        if hasattr(repair_theta, "to_dict"):
            repair_theta = repair_theta.to_dict()
    else:
        repair_theta = rollout.get("lift_params", {})

    return {
        "source_file": str(source_file),
        "source_demo": demo_key,
        "original_failure_type": failure_type,
        "context_mode": "original_failed",
        "repair_theta": repair_theta,
        "success_flag": bool(rollout.get("success_flag")),
        "E_xy": float(rollout.get("E_xy_norm", 0.0)),
        "E_transport": float(rollout.get("E_transport_norm", 0.0)),
        "E_yaw": float(rollout.get("E_yaw_norm", 0.0)),
        "E_z": float(rollout.get("E_z_norm", 0.0)),
        "E_grasp": float(rollout.get("E_grasp_norm", 0.0)),
        "E_lift": float(rollout.get("E_lift_norm", 0.0)),
        "E_smooth": float(rollout.get("E_smooth_norm", 0.0)),
        "E_total": float(rollout.get("E_total_norm", 0.0)),
        "final_xy": float(rollout.get("final_nut_peg_xy", rollout.get("final_xy", 0.0))),
        "final_z_diff": float(rollout.get("final_z_diff", 0.0)),
        "object_poses_modified": False,
        "success_from_rollout": True,
        "context": context,
        "rollout": rollout,
        "active": active,
        "demo_key": demo_key,
        "source": "v1f_plus_rollout_sampling",
    }


def run_plus_rollout_sampling(
    *,
    failed_hdf5: Path,
    audit_report: Path,
    cem_report: Path,
    output: Path,
    seed: int = 0,
    min_budget: int = 100,
    max_budget: int = 200,
    partial_dir: Path | None = None,
) -> list[dict[str, Any]]:
    failure_map = load_failure_map(audit_report)
    records: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)

    for demo_key in list_demo_keys(failed_hdf5):
        info = failure_map.get(demo_key, {})
        sampler = info.get("sampler", "mixed")
        coarse = info.get("coarse_failure_type", "transport_failed")
        rough = info.get("rough_failure_type", "unknown")
        budget = int(rng.integers(min_budget, max_budget + 1))

        if sampler == "mixed":
            half = budget // 2
            plan = [("transport", half), ("insertion", budget - half)]
        else:
            plan = [(sampler, budget)]

        print(f"[v1f-plus-rollout] {demo_key} sampler={sampler} budget={budget}", flush=True)
        demo_records: list[dict[str, Any]] = []

        for active, active_budget in plan:
            baseline = _baseline_rollout(failed_hdf5, demo_key, active, cem_report)
            context = extract_failed_context(baseline, demo_key=demo_key, failure_type=coarse)
            for i, params in enumerate(_iter_candidates(active, active_budget, seed, demo_key)):
                rollout = _sample_rollout(failed_hdf5, demo_key, active, params, cem_report, seed, i)
                row = _rollout_record_to_jsonl_row(
                    source_file=failed_hdf5,
                    demo_key=demo_key,
                    failure_type=rough,
                    context=context,
                    rollout=rollout,
                    active=active,
                )
                demo_records.append(row)
                records.append(row)
                if (i + 1) % 25 == 0:
                    print(f"  {demo_key}/{active}: {i + 1}/{active_budget}", flush=True)

        if partial_dir:
            partial_dir.mkdir(parents=True, exist_ok=True)
            partial_path = partial_dir / f"rollout_partial_{demo_key}.jsonl"
            with partial_path.open("w", encoding="utf-8") as handle:
                for rec in demo_records:
                    handle.write(json.dumps(rec, default=str) + "\n")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec, default=str) + "\n")
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="V1-F-aligned-plus rollout sampling on new failed demos")
    parser.add_argument("--failed-hdf5", type=Path, default=DEFAULT_FAILED_HDF5)
    parser.add_argument("--audit-report", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--cem-report", type=Path, default=DEFAULT_CEM_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_PLUS_OUTPUT / "new_rollout_samples.jsonl")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-budget", type=int, default=100)
    parser.add_argument("--max-budget", type=int, default=200)
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    partial_dir = args.output.parent / "rollout_partials"
    records = run_plus_rollout_sampling(
        failed_hdf5=args.failed_hdf5,
        audit_report=args.audit_report,
        cem_report=args.cem_report,
        output=args.output,
        seed=args.seed,
        min_budget=args.min_budget,
        max_budget=args.max_budget,
        partial_dir=partial_dir,
    )
    summary = {
        "num_samples": len(records),
        "per_demo": {},
        "success_count": sum(1 for r in records if r["success_flag"]),
        "output": str(args.output),
    }
    for rec in records:
        dk = rec["demo_key"]
        summary["per_demo"][dk] = summary["per_demo"].get(dk, 0) + 1
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
