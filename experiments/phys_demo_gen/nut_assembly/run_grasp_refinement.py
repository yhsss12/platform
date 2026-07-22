#!/usr/bin/env python3
"""V2-B4：grasp_failed demo_2 / demo_3 grasp-stage sim-in-loop refinement。"""
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

from grasp_sim_search import (
    GraspSearchParams,
    classify_grasp_outcome,
    diagnose_grasp_failure_reason,
    evaluate_grasp_levels,
    execute_grasp_rollout,
    pick_best_grasp_candidate,
    run_grasp_search,
    summarize_effective_grasp_params,
)
from robosuite_env_loader import check_environment
from sim_in_loop_refiner import SUCCESS_Z_TARGET

DEFAULT_FAILED = _EXPERIMENT_DIR.parents[2] / "mnt" / "data" / "demo_failed.hdf5"
DEFAULT_OUTPUT = _EXPERIMENT_DIR / "outputs" / "grasp_refinement"
GRASP_DEMO_KEYS = ["demo_2", "demo_3"]


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if not rows:
        return
    fields = fieldnames or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _candidate_row(result: dict, rank: int | None = None, demo_key: str = "") -> dict:
    row = {
        "demo_key": demo_key or result.get("demo_name"),
        "rank": rank,
        "rollout_kind": result.get("rollout_kind"),
        "success_flag": result.get("success_flag"),
        "outcome": result.get("outcome_label"),
        "search_score": result.get("search_score", result.get("score")),
        "E_total_norm": result.get("E_total_norm"),
        "E_xy_norm": result.get("E_xy_norm"),
        "E_transport_norm": result.get("E_transport_norm"),
        "final_nut_peg_xy": result.get("final_nut_peg_xy"),
        "min_nut_peg_xy": result.get("min_nut_peg_xy"),
        "nut_displacement_total": result.get("nut_displacement_total"),
        "nut_displacement_after_grasp": result.get("nut_displacement_after_grasp"),
        "nut_lift_delta": result.get("nut_lift_delta"),
        "eef_nut_distance_at_grasp": result.get("eef_nut_distance_at_grasp"),
        "min_eef_nut_distance": result.get("min_eef_nut_distance"),
        "gripper_closed_fraction": result.get("gripper_closed_fraction"),
        "grasp_success_proxy": result.get("grasp_success_proxy"),
        "lift_success_proxy": result.get("lift_success_proxy"),
        "failure_guess": result.get("failure_guess"),
        "failure_reason": result.get("failure_reason"),
        "seed": result.get("seed"),
        "video_path": result.get("video_path"),
    }
    params = result.get("grasp_params") or {}
    for key, val in params.items():
        row[f"grasp_{key}"] = val
    return row


def refine_one_demo(
    *,
    failed_hdf5: Path,
    demo_key: str,
    output_dir: Path,
    search_mode: str,
    max_evals: int,
    seeds: list[int],
    record_video: bool,
) -> dict:
    video_dir = output_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    original = execute_grasp_rollout(
        str(failed_hdf5),
        demo_key,
        "failed",
        GraspSearchParams(),
        rollout_kind="grasp_original_waypoint",
        video_path=video_dir / f"original_{demo_key}.mp4",
        record_video=record_video,
    )
    original["outcome_label"] = "baseline"
    original["failure_reason"] = diagnose_grasp_failure_reason(original, original)

    all_candidates: list[dict] = []
    seed_metas: list[dict] = []
    for seed in seeds:
        candidates, meta = run_grasp_search(
            str(failed_hdf5),
            demo_key,
            "failed",
            mode=search_mode,
            max_evals=max_evals,
            seed=seed,
        )
        all_candidates.extend(candidates)
        seed_metas.append({"seed": seed, **meta})

    best = pick_best_grasp_candidate(all_candidates)
    best["outcome_label"] = classify_grasp_outcome(best, original)
    best["failure_reason"] = diagnose_grasp_failure_reason(best, original)

    best_video = video_dir / f"best_refined_{demo_key}.mp4"
    best_with_video = execute_grasp_rollout(
        str(failed_hdf5),
        demo_key,
        "failed",
        GraspSearchParams(**best["grasp_params"]),
        rollout_kind="grasp_best_refined",
        video_path=best_video,
        record_video=record_video,
    )
    best_with_video["outcome_label"] = best["outcome_label"]
    best_with_video["failure_reason"] = best["failure_reason"]
    best_with_video["search_score"] = best["search_score"]
    best_with_video["video_path"] = str(best_video)

    levels = evaluate_grasp_levels(original, best_with_video)
    effective_params = summarize_effective_grasp_params(all_candidates, top_k=10)

    ranked = sorted(
        all_candidates,
        key=lambda row: (
            not row.get("success_flag", False),
            not row.get("grasp_success_proxy", False),
            row.get("search_score", row.get("score", 1e9)),
        ),
    )
    top10 = ranked[:10]

    return {
        "demo_key": demo_key,
        "failure_mode": "grasp_failed",
        "original_waypoint_rollout": original,
        "best_grasp_refined": best_with_video,
        "best_grasp_params": best["grasp_params"],
        "acceptance_levels": levels,
        "outcome": best_with_video["outcome_label"],
        "failure_reason": best_with_video["failure_reason"],
        "effective_params": effective_params,
        "search": {
            "mode": search_mode,
            "max_evals_per_seed": max_evals,
            "seeds": seeds,
            "total_evals": len(all_candidates),
            "seed_metas": seed_metas,
        },
        "comparison": {
            "original_nut_displacement_after_grasp": original["nut_displacement_after_grasp"],
            "best_nut_displacement_after_grasp": best_with_video["nut_displacement_after_grasp"],
            "original_nut_lift_delta": original.get("nut_lift_delta", 0.0),
            "best_nut_lift_delta": best_with_video.get("nut_lift_delta", 0.0),
            "original_min_nut_peg_xy": original["min_nut_peg_xy"],
            "best_min_nut_peg_xy": best_with_video["min_nut_peg_xy"],
            "original_eef_nut_distance_at_grasp": original.get("eef_nut_distance_at_grasp"),
            "best_eef_nut_distance_at_grasp": best_with_video.get("eef_nut_distance_at_grasp"),
            "original_grasp_success_proxy": original.get("grasp_success_proxy"),
            "best_grasp_success_proxy": best_with_video.get("grasp_success_proxy"),
            "delta_nut_displacement_after_grasp": best_with_video["nut_displacement_after_grasp"]
            - original["nut_displacement_after_grasp"],
            "delta_min_nut_peg_xy": original["min_nut_peg_xy"] - best_with_video["min_nut_peg_xy"],
        },
        "top_10_candidates": top10,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="V2-B4 grasp refinement for demo_2/demo_3")
    parser.add_argument("--failed-hdf5", type=Path, default=DEFAULT_FAILED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--demo-keys", default=",".join(GRASP_DEMO_KEYS))
    parser.add_argument("--search-mode", choices=["random", "grid"], default="random")
    parser.add_argument("--max-evals", type=int, default=80)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    demo_keys = [k.strip() for k in args.demo_keys.split(",") if k.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    record_video = not args.no_video

    env_check = check_environment()
    report: dict = {
        "task": "V2-B4_grasp_failed_sim_in_loop_refinement",
        "demo_keys": demo_keys,
        "success_z_target": SUCCESS_Z_TARGET,
        "environment_check": env_check,
        "baseline_method": "current_controller_closed_loop_waypoint_rollout",
        "search_scoring": (
            "4*grasp_distance + 4*no_lift + 3*no_nut_motion + 2*E_transport + 1*E_xy + 0.2*E_smooth"
        ),
        "notes": [
            "Targets grasp_failed demo_2/demo_3 diagnosed in V2-B3.",
            "Does NOT forge object_poses or set final state as success.",
            "Goal: stable grasp + lift + enter transport; outputs feed V1-C.",
        ],
    }

    if not env_check["available"]:
        report["blocked"] = True
        report["block_reason"] = env_check["blockers"]
        (args.output_dir / "grasp_refinement_report.json").write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(report, indent=2))
        return 1

    per_demo: dict[str, dict] = {}
    summary_rows: list[dict] = []
    top_rows: list[dict] = []
    per_demo_best: dict[str, dict] = {}

    for demo_key in demo_keys:
        demo_result = refine_one_demo(
            failed_hdf5=args.failed_hdf5,
            demo_key=demo_key,
            output_dir=args.output_dir,
            search_mode=args.search_mode,
            max_evals=args.max_evals,
            seeds=seeds,
            record_video=record_video,
        )
        per_demo[demo_key] = demo_result

        original = demo_result["original_waypoint_rollout"]
        best = demo_result["best_grasp_refined"]
        levels = demo_result["acceptance_levels"]

        summary_rows.append(
            {
                "demo_key": demo_key,
                "outcome": demo_result["outcome"],
                "failure_reason": demo_result["failure_reason"],
                "success_flag": best["success_flag"],
                "level_g1": levels["level_g1_nut_motion_improved_50pct"],
                "level_g2": levels["level_g2_lift_or_grasp_proxy"],
                "level_g3": levels["level_g3_transport_improved_30pct"],
                "original_nut_displacement_after_grasp": original["nut_displacement_after_grasp"],
                "best_nut_displacement_after_grasp": best["nut_displacement_after_grasp"],
                "nut_displacement_improvement_ratio": levels["nut_displacement_after_grasp_improvement_ratio"],
                "original_nut_lift_delta": original.get("nut_lift_delta", 0.0),
                "best_nut_lift_delta": best.get("nut_lift_delta", 0.0),
                "original_min_nut_peg_xy": original["min_nut_peg_xy"],
                "best_min_nut_peg_xy": best["min_nut_peg_xy"],
                "min_xy_reduction_ratio": levels["min_nut_peg_xy_reduction_ratio"],
                "original_grasp_success_proxy": original.get("grasp_success_proxy"),
                "best_grasp_success_proxy": best.get("grasp_success_proxy"),
                "best_grasp_params": json.dumps(demo_result["best_grasp_params"]),
                "video_original": original.get("video_path"),
                "video_best": best.get("video_path"),
            }
        )

        per_demo_best[demo_key] = {
            "outcome": demo_result["outcome"],
            "failure_reason": demo_result["failure_reason"],
            "success_flag": best["success_flag"],
            "acceptance_levels": levels,
            "best_grasp_params": demo_result["best_grasp_params"],
            "comparison": demo_result["comparison"],
            "effective_params": demo_result["effective_params"],
        }

        for rank, row in enumerate(demo_result["top_10_candidates"], start=1):
            top_rows.append(_candidate_row(row, rank=rank, demo_key=demo_key))

    report["per_demo"] = per_demo
    report["aggregate"] = {
        "num_demos": len(demo_keys),
        "refined_success_count": sum(
            1 for d in per_demo.values() if d["outcome"] == "refined_success"
        ),
        "grasp_improved_but_failed_count": sum(
            1 for d in per_demo.values() if d["outcome"] == "grasp_improved_but_failed"
        ),
        "grasp_no_improvement_count": sum(
            1 for d in per_demo.values() if d["outcome"] == "grasp_no_improvement"
        ),
        "level_g1_pass_count": sum(
            1 for d in per_demo.values() if d["acceptance_levels"]["level_g1_nut_motion_improved_50pct"]
        ),
        "level_g2_pass_count": sum(
            1 for d in per_demo.values() if d["acceptance_levels"]["level_g2_lift_or_grasp_proxy"]
        ),
        "level_g3_pass_count": sum(
            1 for d in per_demo.values() if d["acceptance_levels"]["level_g3_transport_improved_30pct"]
        ),
    }

    (args.output_dir / "grasp_refinement_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "per_demo_best.json").write_text(
        json.dumps(per_demo_best, indent=2),
        encoding="utf-8",
    )
    _write_csv(args.output_dir / "grasp_refinement_summary.csv", summary_rows)
    _write_csv(args.output_dir / "top_candidates.csv", top_rows)

    print(json.dumps(report["aggregate"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
