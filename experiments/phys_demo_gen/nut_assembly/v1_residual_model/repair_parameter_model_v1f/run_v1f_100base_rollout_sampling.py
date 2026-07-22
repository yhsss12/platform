#!/usr/bin/env python3
"""V1-F-100Base：repairability-aware failed demo rollout + targeted extra sampling。"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

_V1F_DIR = Path(__file__).resolve().parent
_EXPERIMENT_DIR = _V1F_DIR.parent.parent
for path in (_EXPERIMENT_DIR, _V1F_DIR.parent, _V1F_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from audit_new_demo_repairability import audit_repairability  # noqa: E402
from run_v1f_plus_rollout_sampling import (  # noqa: E402
    _baseline_rollout,
    _iter_candidates,
    _rollout_record_to_jsonl_row,
    _sample_rollout,
)
from v1f_100base_utils import (  # noqa: E402
    DEFAULT_AUDIT_REPORT,
    DEFAULT_CEM_REPORT,
    DEFAULT_FAILED_HDF5,
    DEFAULT_FAILED_ROLLOUT,
    DEFAULT_REPAIRABILITY,
    DEFAULT_TARGETED_ROLLOUT,
    REPAIRABILITY_BUDGET,
    TARGETED_EXTRA_BUDGET,
)
from v1f_plus_utils import list_demo_keys, load_failure_map  # noqa: E402
from v1f_repair_dataset import extract_failed_context  # noqa: E402


def _audit_heuristic_budget(info: dict[str, Any], rng: np.random.Generator) -> tuple[int, int]:
    """Phase-1 budget from residual audit difficulty (before rollout repairability)."""
    category = str(info.get("rough_failure_type", "alignment_failed"))
    if category in ("grasp_failed", "lift_failed"):
        lo, hi = REPAIRABILITY_BUDGET["hard_but_improvable"]
    elif category == "insertion_failed":
        lo, hi = REPAIRABILITY_BUDGET["repairable"]
    else:
        lo, hi = REPAIRABILITY_BUDGET["default"]
    b = int(rng.integers(lo, hi + 1))
    return b, b


def run_failed_rollout_phase1(
    *,
    failed_hdf5: Path,
    audit_report: Path,
    cem_report: Path,
    output: Path,
    seed: int,
) -> list[dict[str, Any]]:
    failure_map = load_failure_map(audit_report)
    records: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)

    for demo_key in list_demo_keys(failed_hdf5):
        info = failure_map.get(demo_key, {})
        sampler = info.get("sampler", "mixed")
        coarse = info.get("coarse_failure_type", "transport_failed")
        rough = info.get("rough_failure_type", "transport_failed")
        budget, _ = _audit_heuristic_budget(info, rng)

        if sampler == "mixed":
            half = budget // 2
            plan = [("transport", half), ("insertion", budget - half)]
        else:
            plan = [(sampler, budget)]

        print(f"[100base-rollout-p1] {demo_key} rough={rough} coarse={coarse} budget={budget}", flush=True)
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
                row["source"] = "v1f_100base_failed_rollout"
                records.append(row)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec, default=str) + "\n")
    return records


def run_targeted_phase2(
    *,
    failed_hdf5: Path,
    audit_report: Path,
    cem_report: Path,
    phase1_jsonl: Path,
    repairability_report: Path,
    output: Path,
    seed: int,
) -> list[dict[str, Any]]:
    failure_map = load_failure_map(audit_report)
    repair_report = json.loads(repairability_report.read_text(encoding="utf-8"))
    repair_by_demo = {r["source_demo"]: r for r in repair_report["per_demo"]}
    records: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed + 17)

    targets = [
        d for d, r in repair_by_demo.items()
        if r.get("whether_repairable") in ("repairable", "hard_but_improvable")
    ]
    print(f"[100base-rollout-p2] targeted demos: {len(targets)}", flush=True)

    for demo_key in targets:
        info = failure_map.get(demo_key, {})
        whether = repair_by_demo[demo_key].get("whether_repairable", "repairable")
        extra = TARGETED_EXTRA_BUDGET.get(whether, 40)
        sampler = info.get("sampler", "mixed")
        coarse = info.get("coarse_failure_type", "transport_failed")
        rough = info.get("rough_failure_type", "transport_failed")

        if sampler == "mixed":
            half = extra // 2
            plan = [("transport", half), ("insertion", extra - half)]
        else:
            plan = [(sampler, extra)]

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
                row["source"] = "v1f_100base_targeted_rollout"
                records.append(row)

    with output.open("w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec, default=str) + "\n")
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="V1-F-100Base repairability-aware rollout")
    parser.add_argument("--failed-hdf5", type=Path, default=DEFAULT_FAILED_HDF5)
    parser.add_argument("--audit-report", type=Path, default=DEFAULT_AUDIT_REPORT)
    parser.add_argument("--cem-report", type=Path, default=DEFAULT_CEM_REPORT)
    parser.add_argument("--phase1-output", type=Path, default=DEFAULT_FAILED_ROLLOUT)
    parser.add_argument("--phase2-output", type=Path, default=DEFAULT_TARGETED_ROLLOUT)
    parser.add_argument("--repairability-report", type=Path, default=DEFAULT_REPAIRABILITY)
    parser.add_argument("--skip-phase1", action="store_true")
    parser.add_argument("--skip-phase2", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")

    if not args.skip_phase1:
        p1 = run_failed_rollout_phase1(
            failed_hdf5=args.failed_hdf5,
            audit_report=args.audit_report,
            cem_report=args.cem_report,
            output=args.phase1_output,
            seed=args.seed,
        )
        report, _ = audit_repairability(args.phase1_output)
        args.repairability_report.parent.mkdir(parents=True, exist_ok=True)
        args.repairability_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({"phase1_samples": len(p1), "repairability": str(args.repairability_report)}, indent=2))

    if not args.skip_phase2 and args.repairability_report.exists():
        p2 = run_targeted_phase2(
            failed_hdf5=args.failed_hdf5,
            audit_report=args.audit_report,
            cem_report=args.cem_report,
            phase1_jsonl=args.phase1_output,
            repairability_report=args.repairability_report,
            output=args.phase2_output,
            seed=args.seed,
        )
        print(json.dumps({"phase2_samples": len(p2)}, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
