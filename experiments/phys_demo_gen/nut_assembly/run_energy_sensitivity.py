#!/usr/bin/env python3
"""V0.5 能量敏感性检查：对 failed demo_4 做虚拟残差修正，验证 CEM 优化方向。"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from energy_model import Z_SUCCESS_TARGET, clone_features_with_overrides, compute_total_energy, score_candidate_trajectory
from extract_features import load_features_from_hdf5

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FAILED = PROJECT_ROOT / "mnt/data/demo_failed.hdf5"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "outputs"


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


def _scenario_row(
    scenario_id: str,
    description: str,
    features,
    baseline_total_norm: float,
) -> dict[str, Any]:
    breakdown = compute_total_energy(features)
    score = score_candidate_trajectory(features)
    delta = breakdown.E_total_norm - baseline_total_norm
    return {
        "scenario_id": scenario_id,
        "description": description,
        "final_xy": features.final_nut_peg_xy_distance,
        "min_xy": features.min_nut_peg_xy_distance,
        "final_z_diff": features.final_nut_peg_z_difference,
        "min_yaw": features.min_nut_peg_yaw_error,
        "E_xy_norm": breakdown.E_xy_norm,
        "E_transport_norm": breakdown.E_transport_norm,
        "E_yaw_norm": breakdown.E_yaw_norm,
        "E_z_norm": breakdown.E_z_norm,
        "E_smooth_norm": breakdown.E_smooth_norm,
        "E_total_norm": breakdown.E_total_norm,
        "delta_E_total_norm": delta,
        "failure_type": breakdown.failure_type,
        "optimization_targets": ",".join(score["optimization_targets"]),
    }


def run_sensitivity_check(failed_path: str, output_dir: str) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    failed_features = load_features_from_hdf5(failed_path, "failed")
    demo_4 = next(feature for feature in failed_features if feature.demo_key == "demo_4")

    baseline = compute_total_energy(demo_4)
    baseline_norm = baseline.E_total_norm

    scenarios: list[dict[str, Any]] = []

    scenarios.append(
        {**_scenario_row("baseline", "failed demo_4 原始残差", demo_4, baseline_norm), "category": "baseline"}
    )

    # 1. 将 final_z_diff 往 -0.021 靠近
    for target_z, tag in [
        (0.05, "z_partial_improve"),
        (0.0, "z_near_table"),
        (Z_SUCCESS_TARGET, "z_success_target"),
    ]:
        patched = clone_features_with_overrides(demo_4, final_nut_peg_z_difference=target_z)
        row = _scenario_row(
            f"z_fix_{tag}", f"virtual final_z_diff -> {target_z:.3f}", patched, baseline_norm
        )
        row["category"] = "z"
        scenarios.append(row)

    # 2. 将 min_xy / final_xy 往 0.03 靠近
    for target_xy, tag in [
        (0.03, "xy_on_threshold"),
        (0.02, "xy_close"),
        (0.005, "xy_success_like"),
    ]:
        patched = clone_features_with_overrides(
            demo_4,
            min_nut_peg_xy_distance=target_xy,
            final_nut_peg_xy_distance=min(demo_4.final_nut_peg_xy_distance, target_xy + 0.005),
        )
        row = _scenario_row(
            f"xy_fix_{tag}", f"virtual min_xy -> {target_xy:.3f}", patched, baseline_norm
        )
        row["category"] = "xy"
        scenarios.append(row)

    # 3. 将 min_yaw 往 0.05 以内靠近
    # demo_4 原始 min_yaw≈0.019 已较好，先虚拟恶化再验证修正方向
    yaw_worsened = clone_features_with_overrides(
        demo_4,
        min_nut_peg_yaw_error=0.25,
        final_nut_peg_yaw_error=0.35,
    )
    yaw_baseline_breakdown = compute_total_energy(yaw_worsened)
    yaw_baseline_norm = yaw_baseline_breakdown.E_total_norm

    scenarios.append(
        {
            **_scenario_row(
                "yaw_baseline_worsened",
                "virtual min_yaw worsened to 0.250 for sensitivity",
                yaw_worsened,
                yaw_baseline_norm,
            ),
            "category": "yaw",
        }
    )

    for target_yaw, tag in [
        (0.08, "yaw_near_threshold"),
        (0.04, "yaw_mid"),
        (0.02, "yaw_good"),
        (0.005, "yaw_success_like"),
    ]:
        patched = clone_features_with_overrides(
            yaw_worsened,
            min_nut_peg_yaw_error=target_yaw,
            final_nut_peg_yaw_error=min(yaw_worsened.final_nut_peg_yaw_error, target_yaw + 0.05),
        )
        row = _scenario_row(
            f"yaw_fix_{tag}",
            f"virtual min_yaw -> {target_yaw:.3f} (from worsened 0.250)",
            patched,
            yaw_baseline_norm,
        )
        row["category"] = "yaw"
        scenarios.append(row)

    z_rows = [row for row in scenarios if row.get("category") == "z"]
    xy_rows = [row for row in scenarios if row.get("category") == "xy"]
    yaw_rows = [row for row in scenarios if row.get("category") == "yaw" and row["scenario_id"].startswith("yaw_fix_")]

    z_energy_decreases = all(row["E_z_norm"] < baseline.E_z_norm for row in z_rows) and all(
        row["E_total_norm"] < baseline_norm for row in z_rows
    )

    xy_energy_decreases = all(
        row["E_xy_norm"] < baseline.E_xy_norm and row["E_transport_norm"] < baseline.E_transport_norm
        for row in xy_rows
    ) and all(row["E_total_norm"] < baseline_norm for row in xy_rows)

    yaw_baseline_row = next(row for row in scenarios if row["scenario_id"] == "yaw_baseline_worsened")
    yaw_energy_decreases = all(row["E_yaw_norm"] < yaw_baseline_row["E_yaw_norm"] for row in yaw_rows) and all(
        row["E_total_norm"] < yaw_baseline_row["E_total_norm"] for row in yaw_rows
    )

    per_category_lower = (
        z_energy_decreases
        and xy_energy_decreases
        and yaw_energy_decreases
    )

    report = {
        "task": "Square_D0 / NutAssembly",
        "model_version": "V0.5_sensitivity_check",
        "target_demo": "demo_4",
        "baseline": {
            "final_xy": demo_4.final_nut_peg_xy_distance,
            "min_xy": demo_4.min_nut_peg_xy_distance,
            "final_z_diff": demo_4.final_nut_peg_z_difference,
            "min_yaw": demo_4.min_nut_peg_yaw_error,
            "E_total_norm": baseline_norm,
            "E_z_norm": baseline.E_z_norm,
            "E_xy_norm": baseline.E_xy_norm,
            "E_transport_norm": baseline.E_transport_norm,
            "E_yaw_norm": baseline.E_yaw_norm,
        },
        "scenarios": scenarios,
        "acceptance_checks": {
            "z_correction_lowers_insertion_energy": bool(z_energy_decreases),
            "xy_correction_lowers_transport_xy_energy": bool(xy_energy_decreases),
            "yaw_correction_lowers_yaw_energy": bool(yaw_energy_decreases),
            "all_corrections_lower_E_total_norm": bool(per_category_lower),
        },
        "interpretation": {
            "z": "朝 z_success_target=-0.021 修正可显著降低 E_z_norm 与 E_total_norm",
            "xy": "朝 transport_threshold=0.03 修正可降低 E_xy_norm / E_transport_norm",
            "yaw": "先将 min_yaw 虚拟恶化至 0.25，再朝 yaw_threshold 修正可降低 E_yaw_norm",
        },
    }

    json_path = output / "sensitivity_report.json"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    csv_path = output / "sensitivity_summary.csv"
    _write_csv(csv_path, scenarios)

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print()
    print("Sensitivity acceptance checks:")
    for name, passed in report["acceptance_checks"].items():
        print(f"  {'PASS' if passed else 'FAIL'}: {name}")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Nut Assembly V0.5 energy sensitivity check")
    parser.add_argument("--failed", default=str(DEFAULT_FAILED))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    run_sensitivity_check(args.failed, args.output_dir)


if __name__ == "__main__":
    main()
