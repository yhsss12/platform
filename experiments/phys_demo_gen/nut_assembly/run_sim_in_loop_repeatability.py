#!/usr/bin/env python3
"""V2-B2.6：Sim-in-loop repeatability study（failed demo_4）。"""
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
    classify_outcome,
    load_best_theta,
    pick_best_candidate,
    result_to_summary_row,
    run_original_waypoint_rollout,
    run_refined_waypoint_rollout,
    run_sim_in_loop_search,
    summarize_repeatability_runs,
)
from robosuite_env_loader import check_environment

DEFAULT_FAILED = _EXPERIMENT_DIR.parents[2] / "mnt" / "data" / "demo_failed.hdf5"
DEFAULT_CEM_REPORT = _EXPERIMENT_DIR / "outputs" / "cem_refinement" / "cem_refinement_report.json"
DEFAULT_OUTPUT = _EXPERIMENT_DIR / "outputs" / "sim_in_loop_repeatability"

DEFAULT_SEEDS = [0, 1, 2, 3, 4]
DEFAULT_MAX_EVALS = [40, 80, 120]
VIDEO_MAX_EVALS = 80


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _analyze_instability(runs: list[dict], by_max_evals_stats: dict[int, dict]) -> list[str]:
    notes: list[str] = []
    by_max_evals: dict[int, list[dict]] = {}
    for row in runs:
        by_max_evals.setdefault(int(row["max_evals"]), []).append(row)

    for max_evals, group in sorted(by_max_evals.items()):
        rate = sum(bool(r["success_flag"]) for r in group) / len(group)
        stats = by_max_evals_stats.get(max_evals, {})
        z_std = float(stats.get("final_z_diff_std", 0.0))
        if rate < 1.0 and rate > 0.0:
            notes.append(
                f"max_evals={max_evals}: success_rate={rate:.2f} — "
                "partial success suggests search budget / stochastic sampling sensitivity."
            )
        elif rate == 0.0:
            notes.append(
                f"max_evals={max_evals}: all seeds failed — likely insufficient eval budget "
                "or high variance in closed-loop insertion dynamics."
            )
        if z_std > 0.03:
            notes.append(
                f"max_evals={max_evals}: final_z_diff std={z_std:.4f} — insertion depth not fully stable."
            )
    return notes


def main() -> int:
    parser = argparse.ArgumentParser(description="V2-B2.6 repeatability study for demo_4")
    parser.add_argument("--failed-hdf5", type=Path, default=DEFAULT_FAILED)
    parser.add_argument("--cem-report", type=Path, default=DEFAULT_CEM_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--demo-key", default="demo_4")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--max-evals-list", type=int, nargs="+", default=DEFAULT_MAX_EVALS)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    video_dir = args.output_dir / "videos"
    video_dir.mkdir(exist_ok=True)

    env_check = check_environment()
    report: dict = {
        "task": "V2-B2.6_sim_in_loop_repeatability",
        "demo_key": args.demo_key,
        "success_z_target": SUCCESS_Z_TARGET,
        "seeds": args.seeds,
        "max_evals_list": args.max_evals_list,
        "environment_check": env_check,
    }

    if not env_check["available"]:
        report["blocked"] = True
        report["block_reason"] = env_check["blockers"]
        (args.output_dir / "repeatability_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 1

    theta = load_best_theta(args.cem_report, args.demo_key)
    hdf5 = str(args.failed_hdf5)
    record = not args.no_video

    original = run_original_waypoint_rollout(hdf5, args.demo_key, "failed", record_video=False)
    original["outcome_label"] = "baseline"

    refined = run_refined_waypoint_rollout(
        hdf5, args.demo_key, "failed", theta, SimLoopParams(), record_video=False
    )
    refined["outcome_label"] = classify_outcome(refined, original)

    all_runs: list[dict] = []
    summary_rows: list[dict] = []
    by_max_evals_stats: dict[int, dict] = {}

    for max_evals in args.max_evals_list:
        group_runs: list[dict] = []
        for seed in args.seeds:
            candidates, search_meta = run_sim_in_loop_search(
                hdf5,
                args.demo_key,
                "failed",
                theta,
                max_evals=max_evals,
                seed=seed,
                scoring_mode="energy_full",
            )
            best = pick_best_candidate(candidates, scoring_mode="energy_full")
            best["eval_count_to_best"] = search_meta["eval_count_to_best"]
            best["outcome_label"] = classify_outcome(best, original)

            video_path = None
            if record and max_evals == VIDEO_MAX_EVALS:
                video_path = video_dir / f"best_seed_{seed}.mp4"
                best_video = run_refined_waypoint_rollout(
                    hdf5,
                    args.demo_key,
                    "failed",
                    theta,
                    SimLoopParams(**best["sim_params"]),
                    video_path=video_path,
                    record_video=True,
                )
                best["video_path"] = str(video_path)
                best["reroll_success_flag"] = best_video["success_flag"]

            run_record = {
                **result_to_summary_row(
                    best,
                    seed=seed,
                    max_evals=max_evals,
                    scoring_mode="energy_full",
                ),
                "sim_params": best["sim_params"],
            }
            all_runs.append(run_record)
            group_runs.append(best)
            summary_rows.append(run_record)

        stats = summarize_repeatability_runs(group_runs)
        by_max_evals_stats[max_evals] = stats

    overall_stats = summarize_repeatability_runs(
        [r for r in all_runs if isinstance(r.get("success_flag"), (bool, int))]
    )
    # Rebuild from best dicts for overall - use summary rows converted back
    best_dicts = []
    for row in summary_rows:
        best_dicts.append(
            {
                "success_flag": row["success_flag"],
                "E_total_norm": row["E_total_norm"],
                "final_z_diff": row["final_z_diff"],
                "outcome_label": row["outcome"],
            }
        )
    overall_stats = summarize_repeatability_runs(best_dicts)

    instability_notes = _analyze_instability(summary_rows, by_max_evals_stats)

    report["reference_rollouts"] = {
        "original_waypoint": {
            "success_flag": original["success_flag"],
            "E_total_norm": original["E_total_norm"],
            "final_z_diff": original["final_z_diff"],
        },
        "refined_v2a_theta": {
            "success_flag": refined["success_flag"],
            "E_total_norm": refined["E_total_norm"],
            "final_z_diff": refined["final_z_diff"],
            "outcome": refined["outcome_label"],
        },
    }
    report["runs"] = all_runs
    report["statistics"] = {
        "overall": overall_stats,
        "by_max_evals": by_max_evals_stats,
    }
    report["instability_analysis"] = instability_notes
    report["acceptance"] = {
        "n_seeds": len(args.seeds),
        "n_max_evals_settings": len(args.max_evals_list),
        "overall_success_rate": overall_stats.get("success_rate"),
        "E_total_norm_mean": overall_stats.get("E_total_norm_mean"),
        "E_total_norm_std": overall_stats.get("E_total_norm_std"),
        "final_z_diff_near_target": overall_stats.get("final_z_diff_distance_to_target_mean"),
        "real_sim_rollout_only": True,
    }

    (args.output_dir / "repeatability_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_csv(args.output_dir / "repeatability_summary.csv", summary_rows)

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
