#!/usr/bin/env python3
"""V2-B2：refined demo_4 rollout 验证（原始 failed replay vs refined theta rollout）。"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

_EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(_EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENT_DIR))

from replay_original_demo import replay_demo_rollout
from rollout_refined_demo import rollout_refined_demo
from robosuite_env_loader import check_environment

DEFAULT_FAILED = _EXPERIMENT_DIR.parents[2] / "mnt" / "data" / "demo_failed.hdf5"
DEFAULT_CEM_REPORT = _EXPERIMENT_DIR / "outputs" / "cem_refinement" / "cem_refinement_report.json"
DEFAULT_OUTPUT = _EXPERIMENT_DIR / "outputs" / "refined_rollout"
SUCCESS_Z_TARGET = -0.021


def load_best_theta(cem_report_path: Path, demo_key: str = "demo_4") -> dict:
    report = json.loads(cem_report_path.read_text(encoding="utf-8"))
    for item in report.get("results", []):
        if item.get("demo_key") == demo_key:
            return item["best_theta"]
    raise KeyError(f"{demo_key} not found in {cem_report_path}")


def classify_outcome(
    refined: dict,
    original: dict,
    success_z_target: float = SUCCESS_Z_TARGET,
) -> str:
    if refined.get("success_flag"):
        return "refined_success"
    z_improved = abs(refined["final_z_diff"] - success_z_target) < abs(
        original["final_z_diff"] - success_z_target
    )
    energy_improved = refined["E_total_norm"] < original["E_total_norm"]
    if z_improved and energy_improved:
        return "improved_but_failed"
    return "no_improvement"


def main() -> int:
    parser = argparse.ArgumentParser(description="V2-B2 refined rollout for failed demo_4")
    parser.add_argument("--failed-hdf5", type=Path, default=DEFAULT_FAILED)
    parser.add_argument("--cem-report", type=Path, default=DEFAULT_CEM_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--demo-key", default="demo_4")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    video_dir = args.output_dir / "videos"
    video_dir.mkdir(exist_ok=True)

    env_check = check_environment()
    report: dict = {
        "task": "V2-B2_refined_rollout",
        "phase": "demo_4_only",
        "environment_check": env_check,
        "demo_key": args.demo_key,
        "success_z_target": SUCCESS_Z_TARGET,
    }

    if not env_check["available"]:
        report["blocked"] = True
        report["block_reason"] = env_check["blockers"]
        out = args.output_dir / "refined_rollout_report.json"
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 1

    theta = load_best_theta(args.cem_report, args.demo_key)
    report["best_theta"] = theta

    original = replay_demo_rollout(
        str(args.failed_hdf5),
        args.demo_key,
        "failed",
        video_path=None if args.no_video else video_dir / "original_failed_demo_4.mp4",
        record_video=not args.no_video,
    )
    refined = rollout_refined_demo(
        str(args.failed_hdf5),
        args.demo_key,
        "failed",
        theta,
        video_path=None if args.no_video else video_dir / "refined_demo_4.mp4",
        record_video=not args.no_video,
    )

    outcome = classify_outcome(refined, original)
    refined["outcome_label"] = outcome

    residual = {
        "demo_key": args.demo_key,
        "success_z_target": SUCCESS_Z_TARGET,
        "before": {
            "final_nut_peg_xy": original["final_nut_peg_xy"],
            "final_z_diff": original["final_z_diff"],
            "min_yaw_error": original["min_yaw_error"],
            "E_total_norm": original["E_total_norm"],
            "success_flag": original["success_flag"],
            "failure_guess": original["failure_guess"],
        },
        "after": {
            "final_nut_peg_xy": refined["final_nut_peg_xy"],
            "final_z_diff": refined["final_z_diff"],
            "min_yaw_error": refined["min_yaw_error"],
            "E_total_norm": refined["E_total_norm"],
            "success_flag": refined["success_flag"],
            "failure_guess": refined["failure_guess"],
        },
        "delta": {
            "final_z_diff_toward_target": abs(original["final_z_diff"] - SUCCESS_Z_TARGET)
            - abs(refined["final_z_diff"] - SUCCESS_Z_TARGET),
            "E_total_norm_drop": original["E_total_norm"] - refined["E_total_norm"],
        },
        "outcome_label": outcome,
    }

    report["original_replay"] = original
    report["refined_rollout"] = refined
    report["outcome_label"] = outcome
    report["acceptance"] = {
        "E_total_norm_lower_than_original": refined["E_total_norm"] < original["E_total_norm"],
        "final_z_diff_improved_toward_target": abs(refined["final_z_diff"] - SUCCESS_Z_TARGET)
        < abs(original["final_z_diff"] - SUCCESS_Z_TARGET),
        "videos_generated": not args.no_video,
        "object_poses_not_forged": refined.get("object_poses_modified") is False,
        "outcome": outcome,
    }

    (args.output_dir / "refined_rollout_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "residual_before_after.json").write_text(
        json.dumps(residual, indent=2),
        encoding="utf-8",
    )

    summary_path = args.output_dir / "refined_rollout_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "variant",
                "final_nut_peg_xy",
                "final_z_diff",
                "min_yaw_error",
                "E_total_norm",
                "success_flag",
                "failure_guess",
                "video_path",
            ],
        )
        writer.writeheader()
        for variant, row in [("original_failed_replay", original), ("refined_rollout", refined)]:
            writer.writerow(
                {
                    "variant": variant,
                    "final_nut_peg_xy": row["final_nut_peg_xy"],
                    "final_z_diff": row["final_z_diff"],
                    "min_yaw_error": row["min_yaw_error"],
                    "E_total_norm": row["E_total_norm"],
                    "success_flag": row["success_flag"],
                    "failure_guess": row["failure_guess"],
                    "video_path": row.get("video_path"),
                }
            )

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
