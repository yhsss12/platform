#!/usr/bin/env python3
"""
Square_D0 / Nut Assembly 物理残差审计脚本（独立运行，不修改平台代码）。

用法:
  python nut_assembly_residual_audit.py \
    --success /path/to/demo.hdf5 \
    --failed /path/to/demo_failed.hdf5 \
    --output-dir /path/to/output
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
LEGACY_DATA_ROOT = REPO_ROOT / "mnt" / "data"

# NutAssembly on_peg 与 Square nut 四重对称
XY_CLOSE_THRESH = 0.03
MIN_XY_TRANSPORT_THRESH = 0.10
YAW_ALIGN_THRESH = 0.05
Z_INSERTED_THRESH = -0.015
FINAL_XY_INSERT_THRESH = 0.01


def _list_demo_keys(data_grp: h5py.Group) -> list[str]:
    return sorted(data_grp.keys(), key=lambda k: int(k.split("_")[-1]))


def _group_fields(grp: h5py.Group, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in grp.keys():
        path = f"{prefix}/{key}" if prefix else key
        obj = grp[key]
        if isinstance(obj, h5py.Dataset):
            out[path] = {"shape": list(obj.shape), "dtype": str(obj.dtype)}
        else:
            out.update(_group_fields(obj, path))
    return out


def _yaw_from_rot(rot: np.ndarray) -> np.ndarray:
    return np.arctan2(rot[:, 1, 0], rot[:, 0, 0])


def _square_yaw_error(nut_rot: np.ndarray, peg_rot: np.ndarray) -> np.ndarray:
    """方螺母四重旋转对称：误差折叠到 [-pi/4, pi/4]。"""
    d = _yaw_from_rot(nut_rot) - _yaw_from_rot(peg_rot)
    d = (d + np.pi / 4) % (np.pi / 2) - np.pi / 4
    return np.abs(d)


def _smoothness_stats(series: np.ndarray) -> dict[str, float]:
    if len(series) < 2:
        return {"mean": 0.0, "std": 0.0, "max": 0.0, "min": 0.0}
    first = np.diff(series, axis=0)
    second = np.diff(first, axis=0) if len(first) > 1 else np.zeros((0, series.shape[1]))
    v1 = np.linalg.norm(first, axis=1)
    v2 = np.linalg.norm(second, axis=1) if len(second) else np.array([0.0])
    return {
        "velocity_mean": float(np.mean(v1)),
        "velocity_std": float(np.std(v1)),
        "velocity_max": float(np.max(v1)),
        "velocity_min": float(np.min(v1)),
        "acceleration_mean": float(np.mean(v2)),
        "acceleration_std": float(np.std(v2)),
        "acceleration_max": float(np.max(v2)),
        "acceleration_min": float(np.min(v2)),
    }


def _gripper_stats(gripper: np.ndarray, actions: np.ndarray) -> dict[str, float]:
    g = gripper.squeeze()
    a_g = actions[:, -1]
    return {
        "gripper_action_mean": float(np.mean(g)),
        "gripper_action_std": float(np.std(g)),
        "gripper_action_min": float(np.min(g)),
        "gripper_action_max": float(np.max(g)),
        "action_gripper_mean": float(np.mean(a_g)),
        "action_gripper_std": float(np.std(a_g)),
        "action_gripper_min": float(np.min(a_g)),
        "action_gripper_max": float(np.max(a_g)),
        "fraction_gripper_closed": float(np.mean(g < 0)),
    }


def _grasp_index(grasp_signal: np.ndarray) -> int | None:
    sig = grasp_signal.squeeze()
    idx = np.where(sig > 0.5)[0]
    return int(idx[0]) if len(idx) else None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _classify_failed(residuals: dict[str, float], success_ref: dict[str, float]) -> str:
    """对 failed demo 做粗分类。"""
    min_xy = residuals["min_nut_peg_xy_distance"]
    final_xy = residuals["final_nut_peg_xy_distance"]
    final_z = residuals["final_nut_peg_z_difference"]
    min_yaw = residuals["min_nut_peg_yaw_error"]
    action_jerk = residuals["action_acceleration_mean"]
    eef_acc = residuals["eef_acceleration_mean"]

    smooth_thresh = max(
        success_ref.get("action_acceleration_mean_p95", 0.5),
        success_ref.get("eef_acceleration_mean_p95", 0.003) * 50,
    )
    is_smooth = action_jerk > smooth_thresh or eef_acc > success_ref.get("eef_acceleration_mean_p95", 0.003)

    if min_xy > MIN_XY_TRANSPORT_THRESH:
        return "transport_failed"
    if min_yaw > YAW_ALIGN_THRESH:
        return "alignment_failed"
    if final_xy > FINAL_XY_INSERT_THRESH or final_z > Z_INSERTED_THRESH:
        return "insertion_failed"
    if is_smooth:
        return "smoothness_issue"
    return "alignment_failed"


def _inspect_file(path: str, label: str) -> dict[str, Any]:
    with h5py.File(path, "r") as f:
        demos = _list_demo_keys(f["data"])
        first = f[f"data/{demos[0]}"]
        file_info = {
            "path": path,
            "label": label,
            "task": "Square_D0 / NutAssembly",
            "demo_count": len(demos),
            "demo_lengths": {},
            "actions_shape": list(first["actions"].shape),
            "datagen_info_fields": _group_fields(first["datagen_info"]),
            "object_poses_fields": _group_fields(first["datagen_info/object_poses"]),
            "obs_fields": _group_fields(first["obs"]),
        }
        for dk in demos:
            file_info["demo_lengths"][dk] = int(f[f"data/{dk}/actions"].shape[0])
    return file_info


def _compute_demo_residuals(demo_grp: h5py.Group, demo_key: str) -> dict[str, Any]:
    actions = demo_grp["actions"][:]
    nut = demo_grp["datagen_info/object_poses/square_nut"][:]
    peg = demo_grp["datagen_info/object_poses/square_peg"][:]
    eef = demo_grp["datagen_info/eef_pose"][:]
    target = demo_grp["datagen_info/target_pose"][:]
    gripper = demo_grp["datagen_info/gripper_action"][:]
    grasp = demo_grp["datagen_info/subtask_term_signals/grasp"][:]

    nut_pos = nut[:, :3, 3]
    peg_pos = peg[:, :3, 3]
    xy_dist = np.linalg.norm(nut_pos[:, :2] - peg_pos[:, :2], axis=1)
    z_diff = nut_pos[:, 2] - peg_pos[:, 2]
    yaw_err = _square_yaw_error(nut[:, :3, :3], peg[:, :3, :3])

    closest_idx = int(np.argmin(xy_dist))
    eef_pos = eef[:, :3, 3]
    target_pos = target[:, :3, 3]
    eef_target_dist = np.linalg.norm(eef_pos - target_pos, axis=1)

    action_sm = _smoothness_stats(actions)
    eef_sm = _smoothness_stats(eef_pos)
    grip_stats = _gripper_stats(gripper, actions)
    grasp_idx = _grasp_index(grasp)

    return {
        "demo_key": demo_key,
        "trajectory_length": int(actions.shape[0]),
        "final_nut_peg_xy_distance": float(xy_dist[-1]),
        "min_nut_peg_xy_distance": float(xy_dist.min()),
        "final_nut_peg_z_difference": float(z_diff[-1]),
        "min_nut_peg_z_difference": float(z_diff.min()),
        "final_nut_peg_yaw_error": float(yaw_err[-1]),
        "min_nut_peg_yaw_error": float(yaw_err.min()),
        "yaw_error_at_closest_xy": float(yaw_err[closest_idx]),
        "closest_xy_index": closest_idx,
        "eef_target_final_distance": float(eef_target_dist[-1]),
        "eef_target_min_distance": float(eef_target_dist.min()),
        "eef_target_mean_distance": float(eef_target_dist.mean()),
        "action_velocity_mean": action_sm["velocity_mean"],
        "action_velocity_std": action_sm["velocity_std"],
        "action_velocity_max": action_sm["velocity_max"],
        "action_acceleration_mean": action_sm["acceleration_mean"],
        "action_acceleration_std": action_sm["acceleration_std"],
        "action_acceleration_max": action_sm["acceleration_max"],
        "eef_velocity_mean": eef_sm["velocity_mean"],
        "eef_velocity_std": eef_sm["velocity_std"],
        "eef_velocity_max": eef_sm["velocity_max"],
        "eef_acceleration_mean": eef_sm["acceleration_mean"],
        "eef_acceleration_std": eef_sm["acceleration_std"],
        "eef_acceleration_max": eef_sm["acceleration_max"],
        "grasp_signal_index": grasp_idx,
        "grasp_signal_length": int(np.sum(grasp.squeeze() > 0.5)),
        **grip_stats,
    }


def _aggregate_stats(rows: list[dict[str, Any]], numeric_keys: list[str]) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for key in numeric_keys:
        vals = np.array([r[key] for r in rows if r.get(key) is not None], dtype=float)
        if len(vals) == 0:
            continue
        stats[key] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "var": float(np.var(vals)),
            "p95": float(np.percentile(vals, 95)),
        }
    return stats


def _comparison_table(
    success_stats: dict[str, dict[str, float]],
    failed_stats: dict[str, dict[str, float]],
    residual_keys: list[str],
) -> list[dict[str, Any]]:
    rows = []
    for key in residual_keys:
        s = success_stats.get(key, {})
        f = failed_stats.get(key, {})
        s_mean = s.get("mean", np.nan)
        f_mean = f.get("mean", np.nan)
        pooled_std = np.sqrt((s.get("var", 0) + f.get("var", 0)) / 2) or 1e-9
        cohens_d = abs(f_mean - s_mean) / pooled_std if pooled_std > 0 else 0.0
        rows.append(
            {
                "residual": key,
                "success_mean": s_mean,
                "success_std": s.get("std", np.nan),
                "success_min": s.get("min", np.nan),
                "success_max": s.get("max", np.nan),
                "failed_mean": f_mean,
                "failed_std": f.get("std", np.nan),
                "failed_min": f.get("min", np.nan),
                "failed_max": f.get("max", np.nan),
                "mean_delta": f_mean - s_mean,
                "separation_score": cohens_d,
            }
        )
    rows.sort(key=lambda r: r["separation_score"], reverse=True)
    return rows


RESIDUAL_KEYS = [
    "final_nut_peg_xy_distance",
    "min_nut_peg_xy_distance",
    "final_nut_peg_z_difference",
    "min_nut_peg_z_difference",
    "final_nut_peg_yaw_error",
    "min_nut_peg_yaw_error",
    "yaw_error_at_closest_xy",
    "eef_target_final_distance",
    "eef_target_min_distance",
    "action_velocity_mean",
    "action_acceleration_mean",
    "action_acceleration_max",
    "eef_velocity_mean",
    "eef_acceleration_mean",
    "eef_acceleration_max",
    "gripper_action_mean",
    "fraction_gripper_closed",
]


def run_audit(success_path: str, failed_path: str, output_dir: str) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    success_info = _inspect_file(success_path, "success")
    failed_info = _inspect_file(failed_path, "failed")

    all_rows: list[dict[str, Any]] = []
    for path, label in [(success_path, "success"), (failed_path, "failed")]:
        with h5py.File(path, "r") as f:
            for dk in _list_demo_keys(f["data"]):
                row = _compute_demo_residuals(f[f"data/{dk}"], dk)
                row["file_label"] = label
                row["source_file"] = path
                all_rows.append(row)

    success_rows = [r for r in all_rows if r["file_label"] == "success"]
    failed_rows = [r for r in all_rows if r["file_label"] == "failed"]

    success_stats = _aggregate_stats(success_rows, RESIDUAL_KEYS)
    failed_stats = _aggregate_stats(failed_rows, RESIDUAL_KEYS)

    success_ref = {
        "action_acceleration_mean_p95": success_stats.get("action_acceleration_mean", {}).get("p95", 0.5),
        "eef_acceleration_mean_p95": success_stats.get("eef_acceleration_mean", {}).get("p95", 0.003),
    }
    for row in failed_rows:
        row["failure_category"] = _classify_failed(row, success_ref)
    for row in success_rows:
        row["failure_category"] = "success"

    comparison = _comparison_table(success_stats, failed_stats, RESIDUAL_KEYS)
    top_separators = [r["residual"] for r in comparison[:5]]

    report = {
        "task": "Square_D0 / NutAssembly",
        "files": {
            "success": success_info,
            "failed": failed_info,
        },
        "thresholds": {
            "xy_close_thresh": XY_CLOSE_THRESH,
            "min_xy_transport_thresh": MIN_XY_TRANSPORT_THRESH,
            "yaw_align_thresh": YAW_ALIGN_THRESH,
            "z_inserted_thresh": Z_INSERTED_THRESH,
            "final_xy_insert_thresh": FINAL_XY_INSERT_THRESH,
        },
        "per_demo_residuals": all_rows,
        "residual_statistics": {
            "success": success_stats,
            "failed": failed_stats,
        },
        "success_vs_failed_comparison": comparison,
        "failed_demo_classification": {
            r["demo_key"]: r["failure_category"] for r in failed_rows
        },
        "classification_counts": {},
        "recommendations": {
            "top_discriminative_residuals": top_separators,
            "pinn_energy_model_v1": {
                "inputs": [
                    "nut_pos (xy,z) relative to peg",
                    "square_yaw_error (4-fold symmetric)",
                    "eef_pose relative to target_pose",
                    "gripper_action / grasp phase indicator",
                    "action_velocity (first difference of actions)",
                    "eef_velocity (first difference of eef position)",
                ],
                "outputs": [
                    "nut_peg_xy_distance",
                    "nut_peg_z_difference",
                    "square_yaw_error",
                    "insertion_energy (composite: xy + z + yaw)",
                    "action_smoothness_penalty (L2 jerk)",
                ],
            },
            "cem_trajectory_refinement_targets": [
                "final_nut_peg_xy_distance (primary transport/insertion)",
                "min_nut_peg_xy_distance (transport quality)",
                "final_nut_peg_z_difference (insertion depth)",
                "min_nut_peg_yaw_error (alignment before insert)",
                "action_acceleration_mean / max (smoothness constraint)",
                "eef_target_final_distance (tracking residual)",
            ],
        },
    }

    cat_counts: dict[str, int] = {}
    for r in failed_rows:
        cat = r["failure_category"]
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    report["classification_counts"] = cat_counts

    # JSON
    json_path = out / "residual_report.json"
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2, ensure_ascii=False)

    # Per-demo CSV
    csv_path = out / "residual_summary.csv"
    _write_csv(csv_path, all_rows)

    # Comparison CSV
    cmp_path = out / "success_vs_failed_comparison.csv"
    _write_csv(cmp_path, comparison)

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {cmp_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Nut Assembly physical residual audit")
    parser.add_argument(
        "--success",
        default=str(LEGACY_DATA_ROOT / "demo.hdf5"),
    )
    parser.add_argument(
        "--failed",
        default=str(LEGACY_DATA_ROOT / "demo_failed.hdf5"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(LEGACY_DATA_ROOT / "residual_audit"),
    )
    args = parser.parse_args()
    run_audit(args.success, args.failed, args.output_dir)


if __name__ == "__main__":
    main()
