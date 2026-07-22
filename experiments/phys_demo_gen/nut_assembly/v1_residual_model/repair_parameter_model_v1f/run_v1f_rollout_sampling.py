#!/usr/bin/env python3
"""V1-F：通过 MuJoCo rollout 扩充 repair-parameter 数据集。"""
from __future__ import annotations

import argparse
import json
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

from grasp_sim_search import GRASP_SEARCH_SPACE, execute_grasp_rollout, iter_grasp_candidates  # noqa: E402
from grasp_waypoint_builder import GraspSearchParams  # noqa: E402
from lift_energy_model import merge_v1f_energy_targets  # noqa: E402
from lift_sim_search import execute_lift_rollout, iter_lift_candidates  # noqa: E402
from lift_waypoint_refiner import LiftRepairParams  # noqa: E402
from osc_action_converter import SEARCH_SPACE, SimLoopParams  # noqa: E402
from sim_in_loop_refiner import (  # noqa: E402
    build_refined_waypoints_from_hdf5,
    execute_waypoint_rollout,
    load_best_theta,
)
from transport_sim_search import execute_transport_rollout, iter_transport_candidates  # noqa: E402
from transport_waypoint_builder import TransportSearchParams  # noqa: E402
from v1f_repair_dataset import extract_failed_context, infer_coarse_failure_mode  # noqa: E402

DEFAULT_CEM_REPORT = _EXPERIMENT_DIR / "outputs" / "cem_refinement" / "cem_refinement_report.json"
DEFAULT_FAILED_HDF5 = _EXPERIMENT_DIR.parents[2] / "mnt" / "data" / "demo_failed.hdf5"
DEFAULT_OUTPUT = _EXPERIMENT_DIR / "outputs" / "v1f_repair_parameter_model" / "rollout_samples.jsonl"

# 每 demo 采样配额（总计约 1260，可调到 2000+）
DEFAULT_SAMPLING_PLAN: dict[str, tuple[str, int]] = {
    "demo_0": ("transport", 150),
    "demo_1": ("transport", 150),
    "demo_2": ("grasp", 200),
    "demo_3": ("lift", 500),
    "demo_4": ("insertion", 200),
}


def _baseline_rollout(hdf5: Path, demo_key: str, active: str) -> dict[str, Any]:
    label = "failed"
    if active == "insertion":
        theta = load_best_theta(DEFAULT_CEM_REPORT)
        _proxy, _orig, target_eef, gripper = build_refined_waypoints_from_hdf5(
            str(hdf5), demo_key, label, theta
        )
        return execute_waypoint_rollout(
            str(hdf5), demo_key, label, target_eef, gripper, SimLoopParams(),
            rollout_kind="baseline_original",
            theta=theta,
        )
    if active == "transport":
        theta = load_best_theta(DEFAULT_CEM_REPORT, demo_key)
        params = TransportSearchParams()
        return execute_transport_rollout(
            str(hdf5), demo_key, label, theta, params, rollout_kind="baseline_original"
        )
    if active == "grasp":
        params = GraspSearchParams()
        return execute_grasp_rollout(str(hdf5), demo_key, label, params, rollout_kind="baseline_original")
    params = LiftRepairParams()
    return execute_lift_rollout(str(hdf5), demo_key, label, params, rollout_kind="baseline_original")


def _sample_rollout(
    hdf5: Path,
    demo_key: str,
    active: str,
    params: Any,
    seed: int,
    index: int,
) -> dict[str, Any]:
    label = "failed"
    if active == "insertion":
        theta = load_best_theta(DEFAULT_CEM_REPORT)
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
            rollout_kind="v1f_insertion_repair",
            theta=theta,
        )
        result["sim_params"] = sim_params.to_dict()
    elif active == "transport":
        theta = load_best_theta(DEFAULT_CEM_REPORT, demo_key)
        result = execute_transport_rollout(
            str(hdf5),
            demo_key,
            label,
            theta,
            params,
            rollout_kind="v1f_transport_repair",
        )
        result.update(merge_v1f_energy_targets(result))
    elif active == "grasp":
        result = execute_grasp_rollout(
            str(hdf5), demo_key, label, params, rollout_kind="v1f_grasp_repair"
        )
        result.update(merge_v1f_energy_targets(result))
    else:
        result = execute_lift_rollout(
            str(hdf5), demo_key, label, params, rollout_kind="v1f_lift_repair"
        )

    result["sampling_seed"] = seed
    result["sampling_index"] = index
    result["active_param_group"] = active
    result["demo_key"] = demo_key
    return result


def run_sampling(
    *,
    hdf5: Path,
    plan: dict[str, tuple[str, int]],
    seed: int,
    partial_dir: Path | None = None,
    demo_filter: set[str] | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for demo_key, (active, budget) in plan.items():
        if demo_filter and demo_key not in demo_filter:
            partial_path = (partial_dir / f"rollout_partial_{demo_key}.jsonl") if partial_dir else None
            if partial_path and partial_path.exists():
                with partial_path.open(encoding="utf-8") as handle:
                    for line in handle:
                        if line.strip():
                            records.append(json.loads(line))
            continue
        print(f"[v1f-rollout] {demo_key} active={active} budget={budget}", flush=True)
        baseline = _baseline_rollout(hdf5, demo_key, active)
        context = extract_failed_context(
            baseline,
            demo_key=demo_key,
            failure_type=infer_coarse_failure_mode(demo_key=demo_key),
        )
        if active == "insertion":
            rng = np.random.default_rng(seed + hash(demo_key) % 10000)
            candidates = [
                {k: rng.choice(SEARCH_SPACE[k]) for k in SEARCH_SPACE}
                for _ in range(budget)
            ]
            iterator = candidates
        elif active == "transport":
            iterator = list(iter_transport_candidates(mode="random", max_evals=budget, seed=seed))
        elif active == "grasp":
            iterator = list(iter_grasp_candidates(mode="random", max_evals=budget, seed=seed))
        else:
            iterator = list(iter_lift_candidates(mode="random", max_evals=budget, seed=seed))

        for i, params in enumerate(iterator):
            if active == "insertion":
                rollout = _sample_rollout(hdf5, demo_key, active, params, seed, i)
            else:
                rollout = _sample_rollout(hdf5, demo_key, active, params, seed, i)
            record = {
                "context": context,
                "rollout": rollout,
                "active": active,
                "demo_key": demo_key,
                "source": "v1f_rollout_sampling",
            }
            records.append(record)
            if (i + 1) % 25 == 0:
                print(f"  {demo_key}: {i + 1}/{budget}", flush=True)

        if partial_dir:
            partial_dir.mkdir(parents=True, exist_ok=True)
            partial_path = partial_dir / f"rollout_partial_{demo_key}.jsonl"
            with partial_path.open("w", encoding="utf-8") as handle:
                demo_records = [r for r in records if r["demo_key"] == demo_key]
                for rec in demo_records:
                    handle.write(json.dumps(rec, default=str) + "\n")
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="V1-F rollout sampling for repair dataset expansion")
    parser.add_argument("--failed-hdf5", type=Path, default=DEFAULT_FAILED_HDF5)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--demo-3-budget", type=int, default=500, help="Extra budget for demo_3 lift")
    parser.add_argument("--demo-keys", type=str, default="", help="Comma-separated demo keys to run (others load partial)")
    args = parser.parse_args()

    plan = dict(DEFAULT_SAMPLING_PLAN)
    plan["demo_3"] = ("lift", args.demo_3_budget)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial_dir = args.output.parent / "rollout_partials"
    demo_filter = {k.strip() for k in args.demo_keys.split(",") if k.strip()} or None

    records = run_sampling(
        hdf5=args.failed_hdf5,
        plan=plan,
        seed=args.seed,
        partial_dir=partial_dir,
        demo_filter=demo_filter,
    )
    with args.output.open("w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec, default=str) + "\n")

    summary = {
        "num_samples": len(records),
        "per_demo": {},
        "plan": {k: {"active": v[0], "budget": v[1]} for k, v in plan.items()},
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
