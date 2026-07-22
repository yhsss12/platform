#!/usr/bin/env python3
"""V2-B1.5：Replay Fidelity Calibration — 诊断 open-loop action replay 为何不能复现 success。"""
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

from replay_fidelity_utils import (
    compare_controller_configs,
    diagnose_demo_fidelity,
    extract_runtime_controller_info,
    run_action_replay_check,
    run_final_state_check,
    run_state_sequence_check,
    save_json,
    timeline_summary_rows,
)
from robosuite_env_loader import check_environment, create_env_from_metadata, load_demo_rollout_data

DEFAULT_SUCCESS = _EXPERIMENT_DIR.parents[2] / "mnt" / "data" / "demo.hdf5"
DEFAULT_FAILED = _EXPERIMENT_DIR.parents[2] / "mnt" / "data" / "demo_failed.hdf5"
DEFAULT_OUTPUT = _EXPERIMENT_DIR / "outputs" / "replay_fidelity"

JOBS = [
    {
        "hdf5_key": "success_hdf5",
        "demo_key": "demo_0",
        "label": "success",
        "state_seq_video": "state_sequence_success_demo_0.mp4",
        "action_video": "action_replay_success_demo_0.mp4",
    },
    {
        "hdf5_key": "failed_hdf5",
        "demo_key": "demo_4",
        "label": "failed",
        "state_seq_video": None,
        "action_video": "action_replay_failed_demo_4.mp4",
    },
]


def run_demo_suite(
    hdf5_path: Path,
    demo_key: str,
    label: str,
    output_dir: Path,
    *,
    record_videos: bool,
    state_seq_video: str | None,
    action_video: str | None,
) -> dict:
    video_dir = output_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"{label}_{demo_key}".replace("demo_", "demo_")
    state_metrics_path = output_dir / f"state_sequence_metrics_{label}_{demo_key}.json"
    action_metrics_path = output_dir / f"action_replay_metrics_{label}_{demo_key}.json"

    final_state = run_final_state_check(str(hdf5_path), demo_key, label)

    state_sequence = run_state_sequence_check(
        str(hdf5_path),
        demo_key,
        label,
        video_path=video_dir / state_seq_video if record_videos and state_seq_video else None,
        record_video=record_videos and state_seq_video is not None,
    )
    # 写入 timeline 单独文件（不含逐步 timeline 的 summary 也保留在 report）
    save_json(state_metrics_path, state_sequence)

    action_replay = run_action_replay_check(
        str(hdf5_path),
        demo_key,
        label,
        video_path=video_dir / action_video if record_videos and action_video else None,
        record_video=record_videos and action_video is not None,
    )
    save_json(action_metrics_path, action_replay)

    demo = load_demo_rollout_data(str(hdf5_path), demo_key, label)
    build = create_env_from_metadata(demo.env_args, for_video=False)
    runtime_ctrl = extract_runtime_controller_info(build.env)
    build.env.close()
    controller_comparison = compare_controller_configs(demo.env_args, runtime_ctrl)

    diagnosis = diagnose_demo_fidelity(
        final_state,
        state_sequence,
        action_replay,
        controller_comparison,
    )

    return {
        "demo_key": demo_key,
        "label": label,
        "source_file": str(hdf5_path),
        "hdf5_env_args": demo.env_args,
        "controller_comparison": controller_comparison,
        "final_state_check": final_state,
        "state_sequence_check": {
            k: v
            for k, v in state_sequence.items()
            if k != "timeline"
        },
        "state_sequence_metrics_file": str(state_metrics_path),
        "action_replay_check": {
            k: v
            for k, v in action_replay.items()
            if k != "timeline"
        },
        "action_replay_metrics_file": str(action_metrics_path),
        "diagnosis": diagnosis,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="V2-B1.5 replay fidelity calibration")
    parser.add_argument("--success-hdf5", type=Path, default=DEFAULT_SUCCESS)
    parser.add_argument("--failed-hdf5", type=Path, default=DEFAULT_FAILED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    env_check = check_environment()
    report: dict = {
        "task": "V2-B1.5_replay_fidelity_calibration",
        "environment_check": env_check,
        "demos": [],
        "acceptance": {},
        "global_conclusions": {},
    }

    if not env_check["available"]:
        report["blocked"] = True
        report["block_reason"] = env_check["blockers"]
        save_json(args.output_dir / "replay_fidelity_report.json", report)
        print(json.dumps(report, indent=2))
        return 1

    hdf5_map = {
        "success_hdf5": args.success_hdf5,
        "failed_hdf5": args.failed_hdf5,
    }

    summary_rows: list[dict] = []
    for job in JOBS:
        hdf5_path = hdf5_map[job["hdf5_key"]]
        demo_result = run_demo_suite(
            hdf5_path,
            job["demo_key"],
            job["label"],
            args.output_dir,
            record_videos=not args.no_video,
            state_seq_video=job["state_seq_video"],
            action_video=job["action_video"],
        )
        report["demos"].append(demo_result)
        summary_rows.extend(
            timeline_summary_rows(
                job["demo_key"],
                job["label"],
                {
                    "final_state": demo_result["final_state_check"],
                    "state_sequence": demo_result["state_sequence_check"],
                    "action_replay": demo_result["action_replay_check"],
                },
            )
        )

    success_demo = next(d for d in report["demos"] if d["demo_key"] == "demo_0")
    failed_demo = next(d for d in report["demos"] if d["demo_key"] == "demo_4")

    report["acceptance"] = {
        "success_demo_0_final_state_success": success_demo["final_state_check"]["final_state_success_flag"],
        "success_demo_0_action_replay_success": success_demo["action_replay_check"]["final_success_flag"],
        "success_demo_0_diagnosis": success_demo["diagnosis"]["primary_diagnosis"],
        "failed_demo_4_final_state_failed": not failed_demo["final_state_check"]["final_state_success_flag"],
        "failed_demo_4_action_replay_failed": not failed_demo["action_replay_check"]["final_success_flag"],
        "failed_demo_4_consistent_failure": (
            not failed_demo["final_state_check"]["final_state_success_flag"]
            and not failed_demo["action_replay_check"]["final_success_flag"]
        ),
    }

    report["global_conclusions"] = {
        "why_success_demo_0_action_replay_not_success": success_demo["diagnosis"]["summary"],
        "v2b_trust_boundary": (
            "State-sequence replay is trustworthy (HDF5 states reproduce success). "
            "Open-loop action replay is NOT sufficient for full success validation."
            if success_demo["diagnosis"]["primary_diagnosis"] == "controller_action_replay_fidelity_issue"
            else success_demo["diagnosis"]["summary"]
        ),
        "continue_refined_rollout": success_demo["diagnosis"]["continue_refined_rollout"],
        "refined_rollout_interpretation": success_demo["diagnosis"]["refined_rollout_interpretation"],
        "recommended_next_steps": [
            "Compare legacy vs composite OSC controller closed-loop response at each step",
            "Try closed-loop tracking of recorded eef_pose waypoints instead of open-loop actions",
            "Optional: enable demo XML replay if robosuite version matched collection environment",
        ],
    }

    save_json(args.output_dir / "replay_fidelity_report.json", report)

    summary_path = args.output_dir / "replay_fidelity_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        if summary_rows:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
