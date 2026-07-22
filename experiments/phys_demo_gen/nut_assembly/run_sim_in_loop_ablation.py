#!/usr/bin/env python3
"""V2-B2.6：Sim-in-loop ablation study（failed demo_4）。"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

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
)
from robosuite_env_loader import check_environment

DEFAULT_FAILED = _EXPERIMENT_DIR.parents[2] / "mnt" / "data" / "demo_failed.hdf5"
DEFAULT_CEM_REPORT = _EXPERIMENT_DIR / "outputs" / "cem_refinement" / "cem_refinement_report.json"
DEFAULT_OUTPUT = _EXPERIMENT_DIR / "outputs" / "sim_in_loop_ablation"

ABLATION_METHODS: dict[str, dict[str, Any]] = {
    "A_original_waypoint": {
        "description": "Original eef waypoint, no V2-A theta, no sim search",
        "needs_search": False,
        "use_theta": False,
        "scoring_mode": None,
    },
    "B_v2a_theta_only": {
        "description": "V2-A best_theta refined waypoint, default SimLoopParams, no search",
        "needs_search": False,
        "use_theta": True,
        "scoring_mode": None,
    },
    "C_random_search_without_energy": {
        "description": "Random sim param search; best picked by random score (not energy)",
        "needs_search": True,
        "use_theta": True,
        "scoring_mode": "random",
    },
    "D_energy_guided_search_full": {
        "description": "Random sim param search; best picked by full E_total_norm",
        "needs_search": True,
        "use_theta": True,
        "scoring_mode": "energy_full",
    },
    "E_energy_without_z": {
        "description": "Search ranked without E_z_norm term",
        "needs_search": True,
        "use_theta": True,
        "scoring_mode": "energy_without_z",
    },
    "F_energy_without_xy_transport": {
        "description": "Search ranked without E_xy_norm and E_transport_norm",
        "needs_search": True,
        "use_theta": True,
        "scoring_mode": "energy_without_xy_transport",
    },
    "G_energy_without_yaw": {
        "description": "Search ranked without E_yaw_norm term",
        "needs_search": True,
        "use_theta": True,
        "scoring_mode": "energy_without_yaw",
    },
    "H_energy_without_smooth": {
        "description": "Search ranked without E_smooth_norm term",
        "needs_search": True,
        "use_theta": True,
        "scoring_mode": "energy_without_smooth",
    },
}


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_ablation_method(
    method_id: str,
    cfg: dict[str, Any],
    hdf5: str,
    demo_key: str,
    theta: dict,
    original: dict,
    *,
    max_evals: int,
    seed: int,
    video_path: Path | None,
    record_video: bool,
) -> dict[str, Any]:
    if not cfg["needs_search"]:
        if cfg["use_theta"]:
            result = run_refined_waypoint_rollout(
                hdf5,
                demo_key,
                "failed",
                theta,
                SimLoopParams(),
                video_path=video_path,
                record_video=record_video and video_path is not None,
            )
        else:
            result = run_original_waypoint_rollout(
                hdf5,
                demo_key,
                "failed",
                video_path=video_path,
                record_video=record_video and video_path is not None,
            )
        result["eval_count_to_best"] = 1
        result["method_id"] = method_id
        result["scoring_mode"] = cfg["scoring_mode"]
        result["outcome_label"] = classify_outcome(result, original)
        return result

    scoring_mode = cfg["scoring_mode"]
    candidates, search_meta = run_sim_in_loop_search(
        hdf5,
        demo_key,
        "failed",
        theta,
        max_evals=max_evals,
        seed=seed,
        scoring_mode=scoring_mode,
    )
    best = pick_best_candidate(candidates, scoring_mode=scoring_mode)
    best["eval_count_to_best"] = search_meta["eval_count_to_best"]
    best["method_id"] = method_id
    best["scoring_mode"] = scoring_mode
    best["outcome_label"] = classify_outcome(best, original)

    if record_video and video_path is not None:
        best_video = run_refined_waypoint_rollout(
            hdf5,
            demo_key,
            "failed",
            theta,
            SimLoopParams(**best["sim_params"]),
            video_path=video_path,
            record_video=True,
        )
        best["video_path"] = str(video_path)
        best["reroll_success_flag"] = best_video["success_flag"]

    return best


def _ablation_conclusions(results: dict[str, dict]) -> dict[str, Any]:
    full = results.get("D_energy_guided_search_full", {})
    random_m = results.get("C_random_search_without_energy", {})
    no_z = results.get("E_energy_without_z", {})
    no_xy = results.get("F_energy_without_xy_transport", {})
    no_smooth = results.get("H_energy_without_smooth", {})
    original = results.get("A_original_waypoint", {})
    v2a = results.get("B_v2a_theta_only", {})

    def better(a: dict, b: dict) -> bool:
        if not a or not b:
            return False
        if a.get("success_flag") and not b.get("success_flag"):
            return True
        if a.get("success_flag") == b.get("success_flag"):
            return float(a.get("E_total_norm", 1e9)) < float(b.get("E_total_norm", 1e9))
        return False

    return {
        "full_energy_beats_random": better(full, random_m),
        "full_energy_beats_without_z": better(full, no_z),
        "without_z_worse_for_insertion": float(no_z.get("E_total_norm", 0)) > float(full.get("E_total_norm", 0))
        or (not no_z.get("success_flag") and full.get("success_flag")),
        "without_xy_transport_small_effect_demo4": abs(
            float(no_xy.get("E_total_norm", 0)) - float(full.get("E_total_norm", 0))
        )
        < 2.0,
        "without_smooth_small_effect": abs(
            float(no_smooth.get("E_total_norm", 0)) - float(full.get("E_total_norm", 0))
        )
        < 1.0,
        "v2a_theta_helps_vs_original": better(v2a, original),
        "full_search_helps_vs_v2a": better(full, v2a),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="V2-B2.6 ablation study for demo_4")
    parser.add_argument("--failed-hdf5", type=Path, default=DEFAULT_FAILED)
    parser.add_argument("--cem-report", type=Path, default=DEFAULT_CEM_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--demo-key", default="demo_4")
    parser.add_argument("--max-evals", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    video_dir = args.output_dir / "videos"
    video_dir.mkdir(exist_ok=True)

    env_check = check_environment()
    report: dict = {
        "task": "V2-B2.6_sim_in_loop_ablation",
        "demo_key": args.demo_key,
        "success_z_target": SUCCESS_Z_TARGET,
        "max_evals_per_search_method": args.max_evals,
        "seed": args.seed,
        "methods": ABLATION_METHODS,
        "environment_check": env_check,
    }

    if not env_check["available"]:
        report["blocked"] = True
        report["block_reason"] = env_check["blockers"]
        (args.output_dir / "ablation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 1

    theta = load_best_theta(args.cem_report, args.demo_key)
    hdf5 = str(args.failed_hdf5)
    record = not args.no_video

    original = run_original_waypoint_rollout(hdf5, args.demo_key, "failed", record_video=False)
    original["outcome_label"] = "baseline"

    method_results: dict[str, dict] = {}
    summary_rows: list[dict] = []

    for method_id, cfg in ABLATION_METHODS.items():
        slug = method_id.split("_", 1)[0].lower() + "_" + method_id.split("_", 1)[1]
        video_path = video_dir / f"{method_id}.mp4" if record else None
        result = run_ablation_method(
            method_id,
            cfg,
            hdf5,
            args.demo_key,
            theta,
            original,
            max_evals=args.max_evals,
            seed=args.seed,
            video_path=video_path,
            record_video=record,
        )
        method_results[method_id] = result
        summary_rows.append(
            result_to_summary_row(
                result,
                method_id=method_id,
                description=cfg["description"],
                scoring_mode=cfg["scoring_mode"],
                max_evals=args.max_evals if cfg["needs_search"] else 1,
            )
        )

    conclusions = _ablation_conclusions(method_results)
    ranked = sorted(
        [r for r in method_results.values() if r.get("scoring_mode") == "energy_full" or r.get("method_id") == "D_energy_guided_search_full"]
        + [method_results["D_energy_guided_search_full"]],
        key=lambda r: (not r.get("success_flag", False), r.get("E_total_norm", 1e9)),
    )
    top_by_method = sorted(
        method_results.values(),
        key=lambda r: (not r.get("success_flag", False), r.get("E_total_norm", 1e9)),
    )

    report["reference_original_waypoint"] = {
        k: original[k]
        for k in ["success_flag", "E_total_norm", "final_z_diff", "final_nut_peg_xy", "failure_guess"]
    }
    report["method_results"] = method_results
    report["conclusions"] = conclusions
    report["acceptance"] = {
        "full_energy_beats_random": conclusions["full_energy_beats_random"],
        "without_z_hurts_insertion": conclusions["without_z_worse_for_insertion"],
        "real_sim_rollout_only": True,
        "includes_original_refined_best_tiers": True,
    }

    (args.output_dir / "ablation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_csv(args.output_dir / "ablation_summary.csv", summary_rows)
    _write_csv(
        args.output_dir / "top_candidates_by_method.csv",
        [
            result_to_summary_row(r, method_id=r.get("method_id"), rank=i + 1)
            for i, r in enumerate(top_by_method)
        ],
    )

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
