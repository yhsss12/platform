#!/usr/bin/env python3
"""Task 1：Square_D0 新 demo(1) / demo_failed(1) 数据审计。"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[5]
_NUT_EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
if str(_NUT_EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_NUT_EXPERIMENT_ROOT))

from nut_assembly_residual_audit import run_audit  # noqa: E402

from v1f_plus_utils import DEFAULT_FAILED_HDF5, DEFAULT_PLUS_OUTPUT, DEFAULT_SUCCESS_HDF5  # noqa: E402

DEFAULT_OUTPUT = _REPO_ROOT / "experiments" / "phys_demo_gen" / "nut_assembly" / "outputs" / "new_100_demo_audit"


def _build_audit_report(raw: dict[str, Any]) -> dict[str, Any]:
    success_rows = [r for r in raw["per_demo_residuals"] if r["file_label"] == "success"]
    failed_rows = [r for r in raw["per_demo_residuals"] if r["file_label"] == "failed"]
    return {
        "task": "new_square_d0_demo_audit",
        "source_env": "Square_D0",
        "files": raw["files"],
        "demo_counts": {
            "success": len(success_rows),
            "failed": len(failed_rows),
            "total": len(success_rows) + len(failed_rows),
        },
        "demo_lengths": {
            "success": {r["demo_key"]: r["trajectory_length"] for r in success_rows},
            "failed": {r["demo_key"]: r["trajectory_length"] for r in failed_rows},
        },
        "success_failed_distribution": {
            "success": len(success_rows),
            "failed": len(failed_rows),
        },
        "failure_type_counts": raw.get("classification_counts", {}),
        "failed_demo_classification": raw.get("failed_demo_classification", {}),
        "per_demo_summary": [
            {
                "demo_key": r["demo_key"],
                "file_label": r["file_label"],
                "trajectory_length": r["trajectory_length"],
                "final_nut_peg_xy": r["final_nut_peg_xy_distance"],
                "min_nut_peg_xy": r["min_nut_peg_xy_distance"],
                "final_z_diff": r["final_nut_peg_z_difference"],
                "grasp_signal_index": r.get("grasp_signal_index"),
                "grasp_signal_length": r.get("grasp_signal_length"),
                "rough_failure_type": r.get("failure_category"),
            }
            for r in raw["per_demo_residuals"]
        ],
        "residual_statistics": raw.get("residual_statistics", {}),
        "success_vs_failed_comparison": raw.get("success_vs_failed_comparison", []),
        "thresholds": raw.get("thresholds", {}),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit new Square_D0 demo(1) datasets")
    parser.add_argument("--success", type=Path, default=DEFAULT_SUCCESS_HDF5)
    parser.add_argument("--failed", type=Path, default=DEFAULT_FAILED_HDF5)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = run_audit(str(args.success), str(args.failed), str(args.output_dir / "_tmp"))
    report = _build_audit_report(raw)

    report_path = args.output_dir / "new_demo_audit_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    failed_rows = [
        {
            "demo_key": r["demo_key"],
            "trajectory_length": r["trajectory_length"],
            "final_nut_peg_xy": r["final_nut_peg_xy_distance"],
            "min_nut_peg_xy": r["min_nut_peg_xy_distance"],
            "final_z_diff": r["final_nut_peg_z_difference"],
            "grasp_signal_index": r.get("grasp_signal_index"),
            "grasp_signal_length": r.get("grasp_signal_length"),
            "rough_failure_type": r.get("failure_category"),
        }
        for r in raw["per_demo_residuals"]
        if r["file_label"] == "failed"
    ]
    csv_path = args.output_dir / "new_failed_demo_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(failed_rows[0].keys()))
        writer.writeheader()
        writer.writerows(failed_rows)

    print(json.dumps({"report": str(report_path), "csv": str(csv_path), "counts": report["demo_counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
