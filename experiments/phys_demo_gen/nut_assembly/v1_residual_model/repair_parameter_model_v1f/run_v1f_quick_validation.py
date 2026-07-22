#!/usr/bin/env python3
"""Task 5：Quick validation，对比 original / balanced / balanced-v2。"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

_EXPERIMENT_DIR = Path(__file__).resolve().parents[2]
_V1F_DIR = _EXPERIMENT_DIR / "v1_residual_model" / "repair_parameter_model_v1f"
for path in (_EXPERIMENT_DIR, _EXPERIMENT_DIR / "v1_residual_model", _V1F_DIR, _EXPERIMENT_DIR / "offline_mimicgen_repair_test"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_v1f_plus_evaluation import _repair_cfg_for_new_demo, run_offline_repair  # noqa: E402
from run_v1f_quick_evaluation import NEW_REPAIRABLE, OLD_DEMOS, job_key, load_partial  # noqa: E402
from config import DEFAULT_CEM_REPORT, DEFAULT_FAILED_HDF5, DEFAULT_PINN_MODEL, DEMO_REPAIR_CONFIGS  # noqa: E402
from run_v1f_plus_evaluation import _repair_cfg_for_new_demo  # noqa: E402
from v1f_plus_utils import DEFAULT_ALIGNED_MODEL, DEFAULT_FAILED_HDF5 as NEW_FAILED_HDF5, load_failure_map  # noqa: E402

DEFAULT_BALANCED = _EXPERIMENT_DIR / "outputs" / "v1f_aligned_plus_balanced" / "trained_model" / "model_v1f_aligned_plus_balanced.pt"
DEFAULT_V2 = _EXPERIMENT_DIR / "outputs" / "v1f_aligned_plus_balanced_v2" / "trained_model" / "model_v1f_aligned_plus_balanced_v2.pt"
DEFAULT_OUTPUT = _EXPERIMENT_DIR / "outputs" / "v1f_aligned_plus_balanced_v2" / "quick_validation"
DEFAULT_AUDIT = _EXPERIMENT_DIR / "outputs" / "new_100_demo_audit" / "new_demo_audit_report.json"

VALIDATION_PLAN = [
    ("aligned-original", "v1f_plain_top_k", "aligned-original"),
    ("aligned-plus-balanced", "v1f_plain_top_k", "aligned-plus-balanced"),
    ("aligned-plus-balanced-v2", "v1f_plain_top_k", "aligned-plus-balanced-v2"),
    ("random", "random_top_k", "aligned-plus-balanced-v2"),
    ("explicit", "explicit_top_k", "aligned-plus-balanced-v2"),
]


def evaluate_checks(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def rate(group: str, demo: str, variant: str) -> float | None:
        for r in rows:
            if r["demo_group"] == group and r["demo_key"] == demo and r["variant"] == variant:
                return float(r["repair_rate_at_20"])
        return None

    def avg(group: str, variant: str, demos: tuple[str, ...] | None = None) -> float:
        vals = []
        for r in rows:
            if r["demo_group"] == group and r["variant"] == variant:
                if demos is None or r["demo_key"] in demos:
                    vals.append(float(r["repair_rate_at_20"]))
        return float(sum(vals) / max(len(vals), 1))

    def by_ft(group: str, variant: str, ft: str) -> float:
        vals = [
            float(r["repair_rate_at_20"])
            for r in rows
            if r["demo_group"] == group and r["variant"] == variant and r.get("failure_type") == ft
        ]
        return float(sum(vals) / max(len(vals), 1))

    d4_v2 = rate("old", "demo_4", "aligned-plus-balanced-v2")
    d2_v2 = rate("old", "demo_2", "aligned-plus-balanced-v2")
    d3_v2 = rate("old", "demo_3", "aligned-plus-balanced-v2")
    new_all_v1 = avg("new", "aligned-plus-balanced")
    new_all_v2 = avg("new", "aligned-plus-balanced-v2")
    new_rep_v1 = avg("new", "aligned-plus-balanced", NEW_REPAIRABLE)
    new_rep_v2 = avg("new", "aligned-plus-balanced-v2", NEW_REPAIRABLE)

    checks = {
        "demo_4_ge_0_70": d4_v2 is not None and d4_v2 >= 0.70,
        "demo_2_ge_0_20": d2_v2 is not None and d2_v2 >= 0.20,
        "demo_3_no_lift": d3_v2 is not None and d3_v2 == 0.0,
        "new_repairable_improved": new_rep_v2 > new_rep_v1,
        "transport_improved": by_ft("new", "aligned-plus-balanced-v2", "transport_failed") > by_ft("new", "aligned-plus-balanced", "transport_failed"),
        "insertion_improved": by_ft("new", "aligned-plus-balanced-v2", "insertion_failed") > by_ft("new", "aligned-plus-balanced", "insertion_failed"),
    }
    return {
        "demo_4_repair_rate_v2": d4_v2,
        "demo_2_repair_rate_v2": d2_v2,
        "demo_3_repair_rate_v2": d3_v2,
        "new_all_avg_v1_balanced": new_all_v1,
        "new_all_avg_v2": new_all_v2,
        "new_repairable_avg_v1": new_rep_v1,
        "new_repairable_avg_v2": new_rep_v2,
        "by_failure_type": {
            "transport_failed": {
                "v1_balanced": by_ft("new", "aligned-plus-balanced", "transport_failed"),
                "v2": by_ft("new", "aligned-plus-balanced-v2", "transport_failed"),
            },
            "insertion_failed": {
                "v1_balanced": by_ft("new", "aligned-plus-balanced", "insertion_failed"),
                "v2": by_ft("new", "aligned-plus-balanced-v2", "insertion_failed"),
            },
        },
        "checks": checks,
        "all_pass": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Quick validation for balanced-v2")
    parser.add_argument("--aligned-original-model", type=Path, default=DEFAULT_ALIGNED_MODEL)
    parser.add_argument("--aligned-balanced-model", type=Path, default=DEFAULT_BALANCED)
    parser.add_argument("--aligned-v2-model", type=Path, default=DEFAULT_V2)
    parser.add_argument("--old-failed-hdf5", type=Path, default=DEFAULT_FAILED_HDF5)
    parser.add_argument("--new-failed-hdf5", type=Path, default=NEW_FAILED_HDF5)
    parser.add_argument("--audit-report", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--cem-report", type=Path, default=DEFAULT_CEM_REPORT)
    parser.add_argument("--v1e-model", type=Path, default=DEFAULT_PINN_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = args.output_dir / "validation_partial.jsonl"
    done = load_partial(partial_path) if args.resume else {}

    model_map = {
        "aligned-original": args.aligned_original_model,
        "aligned-plus-balanced": args.aligned_balanced_model,
        "aligned-plus-balanced-v2": args.aligned_v2_model,
    }
    failure_map = load_failure_map(args.audit_report)
    rows: list[dict[str, Any]] = list(done.values())

    jobs: list[dict[str, Any]] = []
    for demo_key in OLD_DEMOS:
        for variant, method, model_label in VALIDATION_PLAN:
            jobs.append(("old", demo_key, variant, method, model_label, DEMO_REPAIR_CONFIGS[demo_key], args.old_failed_hdf5))
    for demo_key in NEW_REPAIRABLE:
        cfg = _repair_cfg_for_new_demo(demo_key, failure_map)
        for variant, method, model_label in VALIDATION_PLAN:
            jobs.append(("new", demo_key, variant, method, model_label, cfg, args.new_failed_hdf5))

    for demo_group, demo_key, variant, method, model_label, cfg, hdf5 in jobs:
        key = job_key(demo_group, demo_key, variant)
        if key in done:
            continue
        model_path = model_map[model_label]
        if not model_path.exists():
            print(f"skip missing model {model_label}", flush=True)
            continue
        print(f"validate {demo_group}/{demo_key} {variant}", flush=True)
        result = run_offline_repair(
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
            demo_group=demo_group,
        )
        row = {
            "job_key": key,
            "demo_group": demo_group,
            "demo_key": demo_key,
            "variant": variant,
            "selection_method": method,
            "failure_type": result.get("failure_type", cfg.get("failure_type")),
            "repair_rate_at_20": result["metrics"]["repair_rate_at_20"],
            "success_at_20": result["metrics"].get("success_at_k", {}).get("at_20"),
            "best_E_total": result["metrics"].get("best_E_total"),
            "num_successes": result["metrics"].get("num_successes_written"),
        }
        done[key] = row
        rows.append(row)
        with partial_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, default=str) + "\n")

    csv_path = args.output_dir / "quick_validation_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    acceptance = evaluate_checks(rows)
    report = {"rows": rows, "acceptance": acceptance}
    (args.output_dir / "quick_validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"csv": str(csv_path), "acceptance": acceptance}, indent=2))
    return 0 if acceptance["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
