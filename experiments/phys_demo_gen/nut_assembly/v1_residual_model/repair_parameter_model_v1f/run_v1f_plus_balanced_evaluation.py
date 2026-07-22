#!/usr/bin/env python3
"""Task 4：V1-F-aligned-plus-balanced 评估（original / plus / balanced），支持 resume。"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

_EXPERIMENT_DIR = Path(__file__).resolve().parents[2]
_V1_DIR = _EXPERIMENT_DIR / "v1_residual_model"
_V1F_DIR = _V1_DIR / "repair_parameter_model_v1f"
_OFFLINE_DIR = _EXPERIMENT_DIR / "offline_mimicgen_repair_test"
for path in (_EXPERIMENT_DIR, _V1_DIR, _V1F_DIR, _OFFLINE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config import DEFAULT_CEM_REPORT, DEFAULT_FAILED_HDF5, DEFAULT_PINN_MODEL, DEMO_REPAIR_CONFIGS  # noqa: E402
from generate_v1f_plus_balanced_decision_report import build_decision_report  # noqa: E402
from run_v1f_plus_evaluation import (  # noqa: E402
    METHODS,
    OLD_DEMO_KEYS,
    _repair_cfg_for_new_demo,
    run_offline_repair,
)
from v1f_plus_utils import (  # noqa: E402
    DEFAULT_ALIGNED_MODEL,
    DEFAULT_FAILED_HDF5 as NEW_FAILED_HDF5,
    DEFAULT_PLUS_OUTPUT,
    list_demo_keys,
    load_failure_map,
)

DEFAULT_AUDIT = _EXPERIMENT_DIR / "outputs" / "new_100_demo_audit" / "new_demo_audit_report.json"
DEFAULT_REPAIRABILITY = (
    _EXPERIMENT_DIR / "outputs" / "v1f_aligned_plus" / "repairability_audit" / "new_demo_repairability_report.json"
)
DEFAULT_BALANCED_MODEL = (
    _EXPERIMENT_DIR / "outputs" / "v1f_aligned_plus_balanced" / "trained_model" / "model_v1f_aligned_plus_balanced.pt"
)
DEFAULT_OUTPUT = _EXPERIMENT_DIR / "outputs" / "v1f_aligned_plus_balanced" / "evaluation"

MODEL_LABELS = ("aligned-original", "aligned-plus", "aligned-plus-balanced")
PARTIAL_JSONL = "eval_partial_results.jsonl"
STATUS_JSON = "eval_status.json"


def job_key(*, demo_group: str, demo_key: str, model_label: str, selection_method: str) -> str:
    return f"{demo_group}|{demo_key}|{model_label}|{selection_method}"


def load_repairability_labels(path: Path) -> dict[str, str]:
    report = json.loads(path.read_text(encoding="utf-8"))
    return {row["source_demo"]: row["whether_repairable"] for row in report["per_demo"]}


def _mean_rate(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return float(sum(r["metrics"]["repair_rate_at_20"] for r in rows) / len(rows))


def evaluate_acceptance(rows: list[dict[str, Any]], repairability: dict[str, str]) -> dict[str, Any]:
    def _get(group: str, demo: str, model: str, method: str = "v1f_plain_top_k") -> dict[str, Any] | None:
        for r in rows:
            if (
                r["demo_group"] == group
                and r["demo_key"] == demo
                and r["model_label"] == model
                and r["selection_method"] == method
            ):
                return r
        return None

    d4 = _get("old", "demo_4", "aligned-plus-balanced")
    d2 = _get("old", "demo_2", "aligned-plus-balanced")
    new_balanced = [
        r
        for r in rows
        if r["demo_group"] == "new" and r["model_label"] == "aligned-plus-balanced" and r["selection_method"] == "v1f_plain_top_k"
    ]
    new_plus = [
        r for r in rows if r["demo_group"] == "new" and r["model_label"] == "aligned-plus" and r["selection_method"] == "v1f_plain_top_k"
    ]
    new_random = [
        r for r in rows if r["demo_group"] == "new" and r["model_label"] == "aligned-plus-balanced" and r["selection_method"] == "random_top_k"
    ]
    repairable_keys = {k for k, v in repairability.items() if v == "repairable"}
    new_repairable_balanced = [r for r in new_balanced if r["demo_key"] in repairable_keys]
    new_repairable_random = [r for r in new_random if r["demo_key"] in repairable_keys]

    def _by_ft(model_rows: list[dict[str, Any]], ft: str) -> list[float]:
        return [float(r["metrics"]["repair_rate_at_20"]) for r in model_rows if r["failure_type"] == ft]

    def _mean(vals: list[float]) -> float:
        return float(sum(vals) / max(len(vals), 1))

    transport_balanced = _by_ft(new_balanced, "transport_failed")
    transport_plus = _by_ft(new_plus, "transport_failed")
    transport_random = _by_ft(new_random, "transport_failed")
    insertion_balanced = _by_ft(new_balanced, "insertion_failed")
    insertion_plus = _by_ft(new_plus, "insertion_failed")
    insertion_random = _by_ft(new_random, "insertion_failed")

    checks = {
        "old_demo_4_ge_0_70": d4 is not None and float(d4["metrics"]["repair_rate_at_20"]) >= 0.70,
        "old_demo_2_ge_0_20": d2 is not None and float(d2["metrics"]["repair_rate_at_20"]) >= 0.20,
        "new_repairable_balanced_better_than_random": _mean_rate(new_repairable_balanced) > _mean_rate(new_repairable_random),
        "transport_or_insertion_beats_plus_and_random": (
            (_mean(transport_balanced) > max(_mean(transport_plus), _mean(transport_random)) if transport_balanced else False)
            or (_mean(insertion_balanced) > max(_mean(insertion_plus), _mean(insertion_random)) if insertion_balanced else False)
        ),
    }
    return {
        "old_demo_4_repair_rate_at_20_balanced": float(d4["metrics"]["repair_rate_at_20"]) if d4 else None,
        "old_demo_2_repair_rate_at_20_balanced": float(d2["metrics"]["repair_rate_at_20"]) if d2 else None,
        "new_all_avg_balanced": _mean_rate(new_balanced),
        "new_all_avg_plus": _mean_rate(new_plus),
        "new_repairable_avg_balanced": _mean_rate(new_repairable_balanced),
        "new_repairable_avg_random": _mean_rate(new_repairable_random),
        "by_failure_type": {
            "transport_failed": {"balanced": _mean(transport_balanced), "plus": _mean(transport_plus), "random": _mean(transport_random)},
            "insertion_failed": {"balanced": _mean(insertion_balanced), "plus": _mean(insertion_plus), "random": _mean(insertion_random)},
        },
        "checks": checks,
        "all_pass": all(checks.values()),
    }


def load_partial_rows(partial_path: Path) -> dict[str, dict[str, Any]]:
    done: dict[str, dict[str, Any]] = {}
    if not partial_path.exists():
        return done
    with partial_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = row.get("job_key") or job_key(
                demo_group=row["demo_group"],
                demo_key=row["demo_key"],
                model_label=row["model_label"],
                selection_method=row["selection_method"],
            )
            done[key] = row
    return done


def append_partial_row(partial_path: Path, row: dict[str, Any]) -> None:
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    with partial_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, default=str) + "\n")


def write_status(status_path: Path, payload: dict[str, Any]) -> None:
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def iter_eval_jobs(
    *,
    model_paths: dict[str, Path],
    new_demo_keys: list[str],
    failure_map: dict[str, dict[str, Any]],
    repairability: dict[str, str],
    old_failed_hdf5: Path,
    new_failed_hdf5: Path,
) -> Iterator[dict[str, Any]]:
    for model_label in MODEL_LABELS:
        if model_label not in model_paths or not model_paths[model_label].exists():
            continue
        for demo_key in OLD_DEMO_KEYS:
            for method in METHODS:
                yield {
                    "demo_group": "old",
                    "demo_key": demo_key,
                    "model_label": model_label,
                    "selection_method": method,
                    "cfg": DEMO_REPAIR_CONFIGS[demo_key],
                    "failed_hdf5": old_failed_hdf5,
                    "repairability_label": "",
                }
        for demo_key in new_demo_keys:
            cfg = _repair_cfg_for_new_demo(demo_key, failure_map)
            for method in METHODS:
                yield {
                    "demo_group": "new",
                    "demo_key": demo_key,
                    "model_label": model_label,
                    "selection_method": method,
                    "cfg": cfg,
                    "failed_hdf5": new_failed_hdf5,
                    "repairability_label": repairability.get(demo_key, "unknown"),
                }


def finalize_outputs(
    *,
    rows: list[dict[str, Any]],
    repairability: dict[str, str],
    model_paths: dict[str, Path],
    repairability_report: Path,
    output_dir: Path,
    status_path: Path,
    partial_path: Path,
    write_decision_report: bool,
) -> dict[str, Any]:
    acceptance = evaluate_acceptance(rows, repairability)
    acceptance["all_pass"] = all(acceptance["checks"].values())

    report = {
        "task": "v1f_aligned_plus_balanced_evaluation",
        "models": {k: str(v) for k, v in model_paths.items()},
        "repairability_report": str(repairability_report),
        "partial_results": str(partial_path),
        "results": rows,
        "acceptance": acceptance,
    }
    report_path = output_dir / "v1f_plus_balanced_evaluation_report.json"
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
                "repairability_label": r.get("repairability_label", ""),
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
    csv_path = output_dir / "v1f_plus_balanced_evaluation_summary.csv"
    if csv_rows:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)

    decision_path = output_dir / "v1f_plus_balanced_decision_report.json"
    if write_decision_report:
        repairability_doc = json.loads(repairability_report.read_text(encoding="utf-8"))
        decision = build_decision_report(report, repairability_doc)
        decision_path.write_text(json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8")

    write_status(
        status_path,
        {
            "state": "completed",
            "completed_jobs": len(rows),
            "total_jobs": len(rows),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "report": str(report_path),
            "csv": str(csv_path),
            "decision_report": str(decision_path) if write_decision_report else None,
            "acceptance": acceptance,
        },
    )
    return {
        "report": str(report_path),
        "csv": str(csv_path),
        "decision_report": str(decision_path) if write_decision_report else None,
        "acceptance": acceptance,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate V1-F-aligned-plus-balanced (resumable)")
    parser.add_argument("--aligned-original-model", type=Path, default=DEFAULT_ALIGNED_MODEL)
    parser.add_argument("--aligned-plus-model", type=Path, default=DEFAULT_PLUS_OUTPUT / "trained_model" / "model_v1f_aligned_plus.pt")
    parser.add_argument("--aligned-balanced-model", type=Path, default=DEFAULT_BALANCED_MODEL)
    parser.add_argument("--old-failed-hdf5", type=Path, default=DEFAULT_FAILED_HDF5)
    parser.add_argument("--new-failed-hdf5", type=Path, default=NEW_FAILED_HDF5)
    parser.add_argument("--audit-report", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--repairability-report", type=Path, default=DEFAULT_REPAIRABILITY)
    parser.add_argument("--cem-report", type=Path, default=DEFAULT_CEM_REPORT)
    parser.add_argument("--v1e-model", type=Path, default=DEFAULT_PINN_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="Resume from eval_partial_results.jsonl")
    parser.add_argument("--no-decision-report", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = args.output_dir / PARTIAL_JSONL
    status_path = args.output_dir / STATUS_JSON

    failure_map = load_failure_map(args.audit_report)
    repairability = load_repairability_labels(args.repairability_report)
    new_demo_keys = list_demo_keys(args.new_failed_hdf5)
    model_paths = {
        "aligned-original": args.aligned_original_model,
        "aligned-plus": args.aligned_plus_model,
        "aligned-plus-balanced": args.aligned_balanced_model,
    }

    done = load_partial_rows(partial_path) if args.resume else {}
    if args.resume and done:
        print(f"[resume] loaded {len(done)} completed jobs from {partial_path}", flush=True)
    elif args.resume:
        print(f"[resume] no partial file yet, starting fresh: {partial_path}", flush=True)

    jobs = list(
        iter_eval_jobs(
            model_paths=model_paths,
            new_demo_keys=new_demo_keys,
            failure_map=failure_map,
            repairability=repairability,
            old_failed_hdf5=args.old_failed_hdf5,
            new_failed_hdf5=args.new_failed_hdf5,
        )
    )
    total_jobs = len(jobs)
    rows = list(done.values())

    write_status(
        status_path,
        {
            "state": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "total_jobs": total_jobs,
            "completed_jobs": len(done),
            "partial_results": str(partial_path),
            "resume_enabled": args.resume,
        },
    )

    for i, job in enumerate(jobs, start=1):
        key = job_key(
            demo_group=job["demo_group"],
            demo_key=job["demo_key"],
            model_label=job["model_label"],
            selection_method=job["selection_method"],
        )
        if key in done:
            continue

        print(
            f"eval [{len(done)+1}/{total_jobs}] {job['demo_group']}/{job['demo_key']} "
            f"{job['model_label']} {job['selection_method']}",
            flush=True,
        )
        row = run_offline_repair(
            demo_key=job["demo_key"],
            cfg=job["cfg"],
            v1f_model=model_paths[job["model_label"]],
            failed_hdf5=job["failed_hdf5"],
            cem_report=args.cem_report,
            selection_method=job["selection_method"],
            num_samples=args.num_samples,
            top_k=args.top_k,
            seed=args.seed,
            v1e_model=args.v1e_model,
            model_label=job["model_label"],
            demo_group=job["demo_group"],
        )
        row["repairability_label"] = job["repairability_label"]
        row["job_key"] = key
        done[key] = row
        rows.append(row)
        append_partial_row(partial_path, row)
        write_status(
            status_path,
            {
                "state": "running",
                "total_jobs": total_jobs,
                "completed_jobs": len(done),
                "last_job": key,
                "partial_results": str(partial_path),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    summary = finalize_outputs(
        rows=rows,
        repairability=repairability,
        model_paths=model_paths,
        repairability_report=args.repairability_report,
        output_dir=args.output_dir,
        status_path=status_path,
        partial_path=partial_path,
        write_decision_report=not args.no_decision_report,
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["acceptance"]["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
