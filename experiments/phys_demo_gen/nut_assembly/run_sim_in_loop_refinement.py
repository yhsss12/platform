#!/usr/bin/env python3
"""V2-B2.5：Simulator-in-the-loop Local Refinement（failed demo_4）。"""
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

from osc_action_converter import SimLoopParams
from sim_in_loop_refiner import (
    SUCCESS_Z_TARGET,
    _json_safe_theta,
    classify_outcome,
    load_best_theta,
    pick_best_candidate,
    run_original_waypoint_rollout,
    run_refined_waypoint_rollout,
    run_sim_in_loop_search,
)
from robosuite_env_loader import check_environment

DEFAULT_FAILED = _EXPERIMENT_DIR.parents[2] / "mnt" / "data" / "demo_failed.hdf5"
DEFAULT_CEM_REPORT = _EXPERIMENT_DIR / "outputs" / "cem_refinement" / "cem_refinement_report.json"
DEFAULT_OUTPUT = _EXPERIMENT_DIR / "outputs" / "sim_in_loop_refinement"


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if not rows:
        return
    fields = fieldnames or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _candidate_row(result: dict, rank: int | None = None) -> dict:
    row = {
        "rank": rank,
        "rollout_kind": result.get("rollout_kind"),
        "success_flag": result.get("success_flag"),
        "outcome_label": result.get("outcome_label"),
        "score": result.get("score"),
        "E_total_norm": result.get("E_total_norm"),
        "E_xy_norm": result.get("E_xy_norm"),
        "E_transport_norm": result.get("E_transport_norm"),
        "E_yaw_norm": result.get("E_yaw_norm"),
        "E_z_norm": result.get("E_z_norm"),
        "final_nut_peg_xy": result.get("final_nut_peg_xy"),
        "min_nut_peg_xy": result.get("min_nut_peg_xy"),
        "final_z_diff": result.get("final_z_diff"),
        "min_yaw_error": result.get("min_yaw_error"),
        "failure_guess": result.get("failure_guess"),
        "video_path": result.get("video_path"),
    }
    sim = result.get("sim_params") or {}
    for key, val in sim.items():
        row[f"sim_{key}"] = val
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="V2-B2.5 sim-in-the-loop refinement for demo_4")
    parser.add_argument("--failed-hdf5", type=Path, default=DEFAULT_FAILED)
    parser.add_argument("--cem-report", type=Path, default=DEFAULT_CEM_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--demo-key", default="demo_4")
    parser.add_argument("--search-mode", choices=["random", "grid"], default="random")
    parser.add_argument("--max-evals", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--top-k-videos", type=int, default=3)
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    video_dir = args.output_dir / "videos"
    video_dir.mkdir(exist_ok=True)

    env_check = check_environment()
    report: dict = {
        "task": "V2-B2.5_sim_in_loop_refinement",
        "demo_key": args.demo_key,
        "success_z_target": SUCCESS_Z_TARGET,
        "environment_check": env_check,
        "search_mode": args.search_mode,
        "max_evals": args.max_evals,
        "baseline_method": "current_controller_closed_loop_waypoint_rollout",
        "notes": [
            "Does NOT use HDF5 raw actions as baseline (V2-B1.5 fidelity issue).",
            "original_waypoint_rollout and refined_waypoint_rollout share osc_action_converter.",
            "Interpretation: residual_improvement_validation unless success_flag=true.",
        ],
    }

    if not env_check["available"]:
        report["blocked"] = True
        report["block_reason"] = env_check["blockers"]
        (args.output_dir / "sim_in_loop_refinement_report.json").write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(report, indent=2))
        return 1

    theta = load_best_theta(args.cem_report, args.demo_key)
    report["v2a_best_theta"] = _json_safe_theta(theta)

    record = not args.no_video
    original = run_original_waypoint_rollout(
        str(args.failed_hdf5),
        args.demo_key,
        "failed",
        video_path=video_dir / "original_waypoint_demo_4.mp4",
        record_video=record,
    )
    original["outcome_label"] = "baseline"

    refined_default = run_refined_waypoint_rollout(
        str(args.failed_hdf5),
        args.demo_key,
        "failed",
        theta,
        SimLoopParams(),
        video_path=None,
        record_video=False,
    )
    refined_default["outcome_label"] = classify_outcome(refined_default, original)

    candidates, _search_meta = run_sim_in_loop_search(
        str(args.failed_hdf5),
        args.demo_key,
        "failed",
        theta,
        mode=args.search_mode,
        max_evals=args.max_evals,
        seed=args.seed,
        record_videos=False,
    )
    best = pick_best_candidate(candidates)
    best["outcome_label"] = classify_outcome(best, original)

    best_video = video_dir / "best_refined_demo_4.mp4"
    best_with_video = run_refined_waypoint_rollout(
        str(args.failed_hdf5),
        args.demo_key,
        "failed",
        theta,
        SimLoopParams(**best["sim_params"]),
        video_path=best_video,
        record_video=record,
    )
    best_with_video["outcome_label"] = best["outcome_label"]
    best_with_video["video_path"] = str(best_video)

    ranked = sorted(
        candidates,
        key=lambda row: (not row.get("success_flag", False), row["score"]),
    )
    top_k = ranked[:10]
    top_video_paths: list[str] = []
    if record:
        for rank, row in enumerate(ranked[: args.top_k_videos], start=1):
            dest = video_dir / f"top_{rank:02d}.mp4"
            rerun = run_refined_waypoint_rollout(
                str(args.failed_hdf5),
                args.demo_key,
                "failed",
                theta,
                SimLoopParams(**row["sim_params"]),
                video_path=dest,
                record_video=True,
            )
            top_video_paths.append(str(dest))
            row["video_path"] = str(dest)
            row["reroll_success_flag"] = rerun["success_flag"]

    residual = {
        "demo_key": args.demo_key,
        "success_z_target": SUCCESS_Z_TARGET,
        "baseline": "original_waypoint_rollout",
        "before": {
            "rollout_kind": original["rollout_kind"],
            "final_nut_peg_xy": original["final_nut_peg_xy"],
            "final_z_diff": original["final_z_diff"],
            "E_total_norm": original["E_total_norm"],
            "success_flag": original["success_flag"],
            "failure_guess": original["failure_guess"],
        },
        "after": {
            "rollout_kind": "sim_in_loop_best",
            "sim_params": best["sim_params"],
            "final_nut_peg_xy": best["final_nut_peg_xy"],
            "final_z_diff": best["final_z_diff"],
            "E_total_norm": best["E_total_norm"],
            "success_flag": best["success_flag"],
            "failure_guess": best["failure_guess"],
        },
        "refined_default_waypoint": {
            "E_total_norm": refined_default["E_total_norm"],
            "final_z_diff": refined_default["final_z_diff"],
            "success_flag": refined_default["success_flag"],
            "outcome_label": refined_default["outcome_label"],
        },
        "delta": {
            "E_total_norm_drop": original["E_total_norm"] - best["E_total_norm"],
            "final_z_diff_toward_target": abs(original["final_z_diff"] - SUCCESS_Z_TARGET)
            - abs(best["final_z_diff"] - SUCCESS_Z_TARGET),
        },
        "outcome_label": best["outcome_label"],
    }

    report["original_waypoint_rollout"] = original
    report["refined_waypoint_rollout_default"] = refined_default
    report["best_sim_in_loop"] = best_with_video
    report["best_sim_params"] = best["sim_params"]
    report["top_10_candidates"] = top_k
    report["residual_before_after_file"] = str(args.output_dir / "residual_before_after.json")
    report["acceptance"] = {
        "same_current_controller": True,
        "object_poses_not_forged": True,
        "best_E_lower_than_original_waypoint": best["E_total_norm"] < original["E_total_norm"],
        "best_z_improved_toward_target": abs(best["final_z_diff"] - SUCCESS_Z_TARGET)
        < abs(original["final_z_diff"] - SUCCESS_Z_TARGET),
        "outcome": best["outcome_label"],
        "refined_success": best["success_flag"],
    }
    report["interpretation"] = {
        "validation_type": "current_env_refined_success"
        if best["success_flag"]
        else "residual_improvement_validation",
        "can_claim_full_success": bool(best["success_flag"]),
    }

    (args.output_dir / "sim_in_loop_refinement_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "residual_before_after.json").write_text(
        json.dumps(residual, indent=2),
        encoding="utf-8",
    )

    summary_rows = [
        _candidate_row(original, rank=None),
        _candidate_row(refined_default, rank=None),
        _candidate_row(best_with_video, rank=1),
    ]
    _write_csv(args.output_dir / "sim_in_loop_refinement_summary.csv", summary_rows)

    top_rows = [_candidate_row(row, rank=i + 1) for i, row in enumerate(top_k)]
    _write_csv(args.output_dir / "top_candidates.csv", top_rows)

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
