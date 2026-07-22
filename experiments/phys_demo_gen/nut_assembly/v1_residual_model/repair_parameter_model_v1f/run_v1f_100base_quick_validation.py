#!/usr/bin/env python3
"""V1-F-100Base quick validation vs aligned-original baseline."""
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

from config import DEFAULT_CEM_REPORT, DEFAULT_FAILED_HDF5, DEFAULT_PINN_MODEL, DEMO_REPAIR_CONFIGS  # noqa: E402
from run_v1f_plus_evaluation import _repair_cfg_for_new_demo, run_offline_repair  # noqa: E402
from run_v1f_quick_evaluation import NEW_REPAIRABLE, OLD_DEMOS, job_key, load_partial  # noqa: E402
from v1f_100base_utils import DEFAULT_100BASE_OUTPUT, DEFAULT_ALIGNED_MODEL, DEFAULT_AUDIT_REPORT, DEFAULT_EVAL_REPORT  # noqa: E402
from v1f_plus_utils import DEFAULT_FAILED_HDF5 as NEW_FAILED_HDF5, load_failure_map  # noqa: E402

DEFAULT_100BASE_MODEL = DEFAULT_100BASE_OUTPUT / "trained_model" / "model_v1f_100base.pt"

VALIDATION_JOBS: list[tuple[str, str]] = [
    *[("old", demo_key) for demo_key in OLD_DEMOS],
    *[("new", demo_key) for demo_key in NEW_REPAIRABLE],
]

VALIDATION_PLAN: list[tuple[str, str, str]] = [
    ("aligned-original", "v1f_plain_top_k", "aligned-original"),
    ("v1f-100base", "v1f_plain_top_k", "v1f-100base"),
]

FAILURE_TYPES = ("transport_failed", "insertion_failed", "alignment_failed", "grasp_failed", "lift_failed")


def evaluate_acceptance(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def rate(group: str, demo: str, variant: str) -> float | None:
        for r in rows:
            if r["demo_group"] == group and r["demo_key"] == demo and r["variant"] == variant:
                return float(r["repair_rate_at_20"])
        return None

    def avg_repairable(variant: str) -> float | None:
        vals = [
            float(r["repair_rate_at_20"])
            for r in rows
            if r["demo_group"] == "new" and r["demo_key"] in NEW_REPAIRABLE and r["variant"] == variant
        ]
        if not vals:
            return None
        return float(sum(vals) / len(vals))

    def by_failure_type(group: str, variant: str, failure_type: str) -> dict[str, Any]:
        vals = [
            float(r["repair_rate_at_20"])
            for r in rows
            if r["demo_group"] == group and r["variant"] == variant and r.get("failure_type") == failure_type
        ]
        if not vals:
            return {"count": 0, "avg_repair_rate_at_20": None, "available": False}
        return {
            "count": len(vals),
            "avg_repair_rate_at_20": float(sum(vals) / len(vals)),
            "available": True,
        }

    d4_orig = rate("old", "demo_4", "aligned-original")
    d4_100 = rate("old", "demo_4", "v1f-100base")
    d2_orig = rate("old", "demo_2", "aligned-original")
    d2_100 = rate("old", "demo_2", "v1f-100base")
    d3_orig = rate("old", "demo_3", "aligned-original")
    d3_100 = rate("old", "demo_3", "v1f-100base")
    new_orig = avg_repairable("aligned-original")
    new_100 = avg_repairable("v1f-100base")

    by_ft: dict[str, dict[str, Any]] = {}
    for ft in FAILURE_TYPES:
        by_ft[ft] = {
            "aligned-original": by_failure_type("new", "aligned-original", ft),
            "v1f-100base": by_failure_type("new", "v1f-100base", ft),
        }

    checks = {
        "demo_4_ge_0_70": d4_100 is not None and d4_100 >= 0.70,
        "demo_2_ge_0_20": d2_100 is not None and d2_100 >= 0.20,
        "demo_3_lift_bottleneck_ok": d3_100 is not None and d3_100 == 0.0,
        "new_repairable_improved_vs_original": (
            new_orig is not None and new_100 is not None and new_100 > new_orig
        ),
    }
    return {
        "old_demo_4": {
            "aligned-original": d4_orig,
            "v1f-100base": d4_100,
            "threshold": 0.70,
            "passed": checks["demo_4_ge_0_70"],
        },
        "old_demo_2": {
            "aligned-original": d2_orig,
            "v1f-100base": d2_100,
            "threshold": 0.20,
            "passed": checks["demo_2_ge_0_20"],
        },
        "old_demo_3": {
            "aligned-original": d3_orig,
            "v1f-100base": d3_100,
            "failure_type": "transport_failed",
            "secondary_failure_type": "lift_underdeveloped",
            "legacy_failure_type": "lift_failed",
            "passed": checks["demo_3_lift_bottleneck_ok"],
        },
        "new_repairable_avg": {
            "aligned-original": new_orig,
            "v1f-100base": new_100,
            "improved": checks["new_repairable_improved_vs_original"],
        },
        "by_failure_type": by_ft,
        "checks": checks,
        "all_passed": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="V1-F-100Base quick validation")
    parser.add_argument("--model-100base", type=Path, default=DEFAULT_100BASE_MODEL)
    parser.add_argument("--aligned-original-model", type=Path, default=DEFAULT_ALIGNED_MODEL)
    parser.add_argument("--old-failed-hdf5", type=Path, default=DEFAULT_FAILED_HDF5)
    parser.add_argument("--new-failed-hdf5", type=Path, default=NEW_FAILED_HDF5)
    parser.add_argument("--audit-report", type=Path, default=DEFAULT_AUDIT_REPORT)
    parser.add_argument("--cem-report", type=Path, default=DEFAULT_CEM_REPORT)
    parser.add_argument("--v1e-model", type=Path, default=DEFAULT_PINN_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_EVAL_REPORT)
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial_path = args.output.parent / "quick_validation_partial.jsonl"
    done = load_partial(partial_path) if args.resume else {}
    rows: list[dict[str, Any]] = list(done.values())

    model_map = {
        "aligned-original": args.aligned_original_model,
        "v1f-100base": args.model_100base,
    }
    failure_map = load_failure_map(args.audit_report)

    jobs: list[tuple[str, str, str, str, str, dict[str, Any], Path]] = []
    for demo_group, demo_key in VALIDATION_JOBS:
        hdf5 = args.old_failed_hdf5 if demo_group == "old" else args.new_failed_hdf5
        if demo_group == "old":
            cfg = DEMO_REPAIR_CONFIGS[demo_key]
        else:
            cfg = _repair_cfg_for_new_demo(demo_key, failure_map)
        for variant, method, model_label in VALIDATION_PLAN:
            jobs.append((demo_group, demo_key, variant, method, model_label, cfg, hdf5))

    for demo_group, demo_key, variant, method, model_label, cfg, hdf5 in jobs:
        key = job_key(demo_group, demo_key, variant)
        if key in done:
            continue
        model_path = model_map[model_label]
        if not model_path.exists():
            raise SystemExit(f"Model checkpoint missing: {model_path}")

        print(f"[100base-val] {key}", flush=True)
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
            "coarse_failure_type": result.get("coarse_failure_type", cfg.get("failure_type")),
            "repair_rate_at_20": result["metrics"]["repair_rate_at_20"],
            "success_at_20": result["metrics"].get("success_at_k", {}).get("at_20"),
            "best_E_total": result["metrics"].get("best_E_total"),
            "num_successes": result["metrics"].get("num_successes_written"),
        }
        done[key] = row
        rows.append(row)
        with partial_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, default=str) + "\n")

    acceptance = evaluate_acceptance(rows)
    report = {
        "task": "v1f_100base_quick_validation",
        "variants": [variant for variant, _, _ in VALIDATION_PLAN],
        "acceptance": acceptance,
        "results": rows,
    }
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_path = args.output.parent / "quick_validation_summary.csv"
    if rows:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(json.dumps({"output": str(args.output), "acceptance": acceptance}, indent=2))
    return 0 if acceptance["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
