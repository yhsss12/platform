#!/usr/bin/env python3
"""对 Nut Assembly HDF5 demo 运行 V0.5 物理能量审计（含归一化能量与贡献比）。"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from energy_model import (
    DEFAULT_WEIGHTS,
    SMOOTHNESS_ABNORMAL_FACTOR,
    XY_THRESHOLD,
    YAW_THRESHOLD,
    Z_TOLERANCE,
    EnergyBreakdown,
    compute_total_energy,
    score_candidate_trajectory,
)
from extract_features import NutAssemblyFeatures, load_features_from_hdf5

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SUCCESS = PROJECT_ROOT / "mnt/data/demo.hdf5"
DEFAULT_FAILED = PROJECT_ROOT / "mnt/data/demo_failed.hdf5"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "outputs"

RAW_KEYS = ["E_xy", "E_transport", "E_yaw", "E_z", "E_smooth", "E_total"]
NORM_KEYS = [
    "E_xy_norm",
    "E_transport_norm",
    "E_yaw_norm",
    "E_z_norm",
    "E_smooth_norm",
    "E_total_norm",
]
CONTRIB_KEYS = [
    "contribution_xy",
    "contribution_transport",
    "contribution_yaw",
    "contribution_z",
    "contribution_smooth",
]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_stats(rows: list[EnergyBreakdown], keys: list[str]) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for key in keys:
        values = np.array([getattr(row, key) for row in rows], dtype=float)
        stats[key] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
            "var": float(values.var()),
        }
    return stats


def _comparison_table(
    success_stats: dict[str, dict[str, float]],
    failed_stats: dict[str, dict[str, float]],
    keys: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in keys:
        success = success_stats.get(key, {})
        failed = failed_stats.get(key, {})
        success_mean = success.get("mean", float("nan"))
        failed_mean = failed.get("mean", float("nan"))
        rows.append(
            {
                "energy_term": key,
                "success_mean": success_mean,
                "success_std": success.get("std", float("nan")),
                "success_min": success.get("min", float("nan")),
                "success_max": success.get("max", float("nan")),
                "failed_mean": failed_mean,
                "failed_std": failed.get("std", float("nan")),
                "failed_min": failed.get("min", float("nan")),
                "failed_max": failed.get("max", float("nan")),
                "mean_delta": failed_mean - success_mean,
                "separation_ratio": failed_mean / success_mean if success_mean > 1e-9 else float("inf"),
            }
        )
    rows.sort(key=lambda item: item["separation_ratio"], reverse=True)
    return rows


def _contribution_by_failure_type(rows: list[EnergyBreakdown]) -> dict[str, dict[str, float]]:
    groups: dict[str, list[EnergyBreakdown]] = {}
    for row in rows:
        groups.setdefault(row.failure_type, []).append(row)
    summary: dict[str, dict[str, float]] = {}
    for failure_type, group in groups.items():
        summary[failure_type] = {
            "count": len(group),
            "mean_contribution_xy": float(np.mean([item.contribution_xy for item in group])),
            "mean_contribution_transport": float(np.mean([item.contribution_transport for item in group])),
            "mean_contribution_yaw": float(np.mean([item.contribution_yaw for item in group])),
            "mean_contribution_z": float(np.mean([item.contribution_z for item in group])),
            "mean_contribution_smooth": float(np.mean([item.contribution_smooth for item in group])),
        }
    return summary


def run_energy_audit(
    success_path: str,
    failed_path: str,
    output_dir: str,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    all_features: list[NutAssemblyFeatures] = []
    all_features.extend(load_features_from_hdf5(success_path, "success"))
    all_features.extend(load_features_from_hdf5(failed_path, "failed"))

    success_smooth = [
        compute_total_energy(feature).E_smooth
        for feature in all_features
        if feature.label == "success"
    ]
    smoothness_threshold = (
        float(np.percentile(success_smooth, 95) * SMOOTHNESS_ABNORMAL_FACTOR) if success_smooth else None
    )

    breakdowns = [
        compute_total_energy(feature, smoothness_threshold=smoothness_threshold)
        for feature in all_features
    ]

    success_rows = [row for row in breakdowns if row.label == "success"]
    failed_rows = [row for row in breakdowns if row.label == "failed"]

    success_raw_stats = _aggregate_stats(success_rows, RAW_KEYS)
    failed_raw_stats = _aggregate_stats(failed_rows, RAW_KEYS)
    success_norm_stats = _aggregate_stats(success_rows, NORM_KEYS)
    failed_norm_stats = _aggregate_stats(failed_rows, NORM_KEYS)

    raw_comparison = _comparison_table(success_raw_stats, failed_raw_stats, RAW_KEYS)
    norm_comparison = _comparison_table(success_norm_stats, failed_norm_stats, NORM_KEYS)
    contribution_summary = _contribution_by_failure_type(breakdowns)

    demo_records = [row.to_dict() for row in breakdowns]
    candidate_scores = {
        row.demo_key + "@" + row.label: score_candidate_trajectory(
            next(feature for feature in all_features if feature.demo_key == row.demo_key and feature.label == row.label),
            smoothness_threshold=smoothness_threshold,
        )
        for row in breakdowns
    }

    transport_failed = [row for row in failed_rows if row.failure_type == "transport_failed"]
    insertion_failed = [row for row in failed_rows if row.failure_type == "insertion_failed"]

    report = {
        "task": "Square_D0 / NutAssembly",
        "model_version": "V0.5_normalized_physics_energy",
        "weights": DEFAULT_WEIGHTS,
        "normalization_scales": {
            "xy_threshold": XY_THRESHOLD,
            "transport_threshold": XY_THRESHOLD,
            "yaw_threshold": YAW_THRESHOLD,
            "z_success_target": -0.021,
            "z_tolerance": Z_TOLERANCE,
            "smooth_threshold": 2.5,
        },
        "inputs": {
            "success_hdf5": success_path,
            "failed_hdf5": failed_path,
        },
        "acceptance_checks": {
            "success_mean_E_total_lt_failed_mean": bool(
                success_raw_stats["E_total"]["mean"] < failed_raw_stats["E_total"]["mean"]
            ),
            "success_mean_E_total_norm_lt_failed_mean": bool(
                success_norm_stats["E_total_norm"]["mean"] < failed_norm_stats["E_total_norm"]["mean"]
            ),
            "failed_demo_0_to_3_transport_failed": bool(
                all(
                    next(row.failure_type for row in failed_rows if row.demo_key == f"demo_{index}")
                    == "transport_failed"
                    for index in range(4)
                )
            ),
            "failed_demo_4_insertion_failed": bool(
                next(row.failure_type for row in failed_rows if row.demo_key == "demo_4") == "insertion_failed"
            ),
            "transport_dominated_by_xy_transport": bool(
                len(transport_failed) > 0
                and float(
                    np.mean([row.contribution_xy + row.contribution_transport for row in transport_failed])
                )
                > 0.5
            ),
            "insertion_dominated_by_z": bool(
                len(insertion_failed) > 0
                and float(np.mean([row.contribution_z for row in insertion_failed])) > 0.3
            ),
            "success_low_normalized_components": bool(success_norm_stats["E_total_norm"]["mean"] < 5.0),
        },
        "energy_statistics": {
            "raw": {"success": success_raw_stats, "failed": failed_raw_stats},
            "normalized": {"success": success_norm_stats, "failed": failed_norm_stats},
        },
        "success_vs_failed_energy_comparison": raw_comparison,
        "success_vs_failed_normalized_comparison": norm_comparison,
        "contribution_by_failure_type": contribution_summary,
        "candidate_scoring_examples": candidate_scores,
        "demos": demo_records,
        "failure_type_counts": {},
    }

    failure_counts: dict[str, int] = {}
    for row in breakdowns:
        failure_counts[row.failure_type] = failure_counts.get(row.failure_type, 0) + 1
    report["failure_type_counts"] = failure_counts

    json_path = output / "energy_report.json"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    summary_path = output / "energy_summary.csv"
    _write_csv(summary_path, demo_records)

    comparison_path = output / "success_vs_failed_energy_comparison.csv"
    _write_csv(comparison_path, norm_comparison)

    print(f"Wrote {json_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {comparison_path}")
    print()
    print("Acceptance checks:")
    for name, passed in report["acceptance_checks"].items():
        print(f"  {'PASS' if passed else 'FAIL'}: {name}")
    print()
    print(f"E_total       success mean = {success_raw_stats['E_total']['mean']:.4f}")
    print(f"E_total       failed  mean = {failed_raw_stats['E_total']['mean']:.4f}")
    print(f"E_total_norm  success mean = {success_norm_stats['E_total_norm']['mean']:.4f}")
    print(f"E_total_norm  failed  mean = {failed_norm_stats['E_total_norm']['mean']:.4f}")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Nut Assembly V0.5 physics energy audit")
    parser.add_argument("--success", default=str(DEFAULT_SUCCESS))
    parser.add_argument("--failed", default=str(DEFAULT_FAILED))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    run_energy_audit(args.success, args.failed, args.output_dir)


if __name__ == "__main__":
    main()
