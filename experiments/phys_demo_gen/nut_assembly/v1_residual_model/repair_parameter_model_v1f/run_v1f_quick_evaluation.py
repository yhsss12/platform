#!/usr/bin/env python3
"""Quick evaluation：old demo_4/2/3 + new repairable demos，4 种 selection 对比。"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from datetime import datetime, timezone
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
from run_v1f_plus_evaluation import _repair_cfg_for_new_demo, run_offline_repair  # noqa: E402
from v1f_plus_utils import DEFAULT_ALIGNED_MODEL, DEFAULT_FAILED_HDF5 as NEW_FAILED_HDF5, load_failure_map  # noqa: E402

DEFAULT_BALANCED_MODEL = (
    _EXPERIMENT_DIR / "outputs" / "v1f_aligned_plus_balanced" / "trained_model" / "model_v1f_aligned_plus_balanced.pt"
)
DEFAULT_OUTPUT = _EXPERIMENT_DIR / "outputs" / "v1f_aligned_plus_balanced" / "quick_eval"
DEFAULT_AUDIT = _EXPERIMENT_DIR / "outputs" / "new_100_demo_audit" / "new_demo_audit_report.json"

OLD_DEMOS = ("demo_4", "demo_2", "demo_3")
NEW_REPAIRABLE = ("demo_4", "demo_5", "demo_6", "demo_7", "demo_9", "demo_18", "demo_20", "demo_21")

EVAL_PLAN: list[tuple[str, str, str, str]] = [
    ("aligned-original", "v1f_plain_top_k", "aligned-original", "v1f"),
    ("aligned-plus-balanced", "v1f_plain_top_k", "aligned-plus-balanced", "v1f"),
    ("random", "random_top_k", "aligned-plus-balanced", "random"),
    ("explicit", "explicit_top_k", "aligned-plus-balanced", "explicit"),
]


def job_key(demo_group: str, demo_key: str, variant: str) -> str:
    return f"{demo_group}|{demo_key}|{variant}"


def load_partial(path: Path) -> dict[str, dict[str, Any]]:
    done: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                done[row["job_key"]] = row
    return done


def run_quick_eval(
    *,
    aligned_original: Path,
    aligned_balanced: Path,
    old_hdf5: Path,
    new_hdf5: Path,
    audit_report: Path,
    cem_report: Path,
    v1e_model: Path,
    output_dir: Path,
    num_samples: int,
    top_k: int,
    seed: int,
    resume: bool,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = output_dir / "quick_partial.jsonl"
    done = load_partial(partial_path) if resume else {}

    model_map = {
        "aligned-original": aligned_original,
        "aligned-plus-balanced": aligned_balanced,
    }
    failure_map = load_failure_map(audit_report)

    jobs: list[dict[str, Any]] = []
    for demo_key in OLD_DEMOS:
        for variant, method, model_label, _ in EVAL_PLAN:
            jobs.append(
                {
                    "demo_group": "old",
                    "demo_key": demo_key,
                    "variant": variant,
                    "method": method,
                    "model_label": model_label,
                    "cfg": DEMO_REPAIR_CONFIGS[demo_key],
                    "hdf5": old_hdf5,
                }
            )
    for demo_key in NEW_REPAIRABLE:
        cfg = _repair_cfg_for_new_demo(demo_key, failure_map)
        for variant, method, model_label, _ in EVAL_PLAN:
            jobs.append(
                {
                    "demo_group": "new",
                    "demo_key": demo_key,
                    "variant": variant,
                    "method": method,
                    "model_label": model_label,
                    "cfg": cfg,
                    "hdf5": new_hdf5,
                }
            )

    rows: list[dict[str, Any]] = list(done.values())
    for job in jobs:
        key = job_key(job["demo_group"], job["demo_key"], job["variant"])
        if key in done:
            continue
        model_path = model_map[job["model_label"]]
        if not model_path.exists():
            continue
        print(f"quick eval {job['demo_group']}/{job['demo_key']} {job['variant']}", flush=True)
        result = run_offline_repair(
            demo_key=job["demo_key"],
            cfg=job["cfg"],
            v1f_model=model_path,
            failed_hdf5=job["hdf5"],
            cem_report=cem_report,
            selection_method=job["method"],
            num_samples=num_samples,
            top_k=top_k,
            seed=seed,
            v1e_model=v1e_model,
            model_label=job["model_label"],
            demo_group=job["demo_group"],
        )
        row = {
            "job_key": key,
            "demo_group": job["demo_group"],
            "demo_key": job["demo_key"],
            "variant": job["variant"],
            "selection_method": job["method"],
            "failure_type": result.get("failure_type", job["cfg"].get("failure_type")),
            "repair_rate_at_20": result["metrics"]["repair_rate_at_20"],
            "success_at_1": result["metrics"].get("success_at_k", {}).get("at_1"),
            "success_at_3": result["metrics"].get("success_at_k", {}).get("at_3"),
            "success_at_5": result["metrics"].get("success_at_k", {}).get("at_5"),
            "success_at_10": result["metrics"].get("success_at_k", {}).get("at_10"),
            "success_at_20": result["metrics"].get("success_at_k", {}).get("at_20"),
            "rollouts_per_success": result["metrics"].get("rollouts_per_success"),
            "best_E_total": result["metrics"].get("best_E_total"),
            "num_successes": result["metrics"].get("num_successes_written"),
        }
        done[key] = row
        rows.append(row)
        with partial_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, default=str) + "\n")

    csv_path = output_dir / "quick_summary.csv"
    fieldnames = [
        "demo_group", "demo_key", "variant", "selection_method", "failure_type",
        "repair_rate_at_20", "success_at_1", "success_at_3", "success_at_5",
        "success_at_10", "success_at_20", "rollouts_per_success", "best_E_total", "num_successes",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["demo_group"], r["demo_key"], r["variant"])))

    (output_dir / "quick_eval_report.json").write_text(
        json.dumps({"num_jobs": len(jobs), "num_completed": len(rows), "rows": rows}, indent=2, default=str),
        encoding="utf-8",
    )
    return csv_path


def main() -> int:
    parser = argparse.ArgumentParser(description="V1-F quick evaluation")
    parser.add_argument("--aligned-original-model", type=Path, default=DEFAULT_ALIGNED_MODEL)
    parser.add_argument("--aligned-balanced-model", type=Path, default=DEFAULT_BALANCED_MODEL)
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
    csv_path = run_quick_eval(
        aligned_original=args.aligned_original_model,
        aligned_balanced=args.aligned_balanced_model,
        old_hdf5=args.old_failed_hdf5,
        new_hdf5=args.new_failed_hdf5,
        audit_report=args.audit_report,
        cem_report=args.cem_report,
        v1e_model=args.v1e_model,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        top_k=args.top_k,
        seed=args.seed,
        resume=args.resume,
    )
    print(json.dumps({"csv": str(csv_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
