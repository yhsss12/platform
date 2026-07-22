#!/usr/bin/env python3
"""V2-B1：RoboSuite rollout 冒烟 — 重放 success demo_0 与 failed demo_4。"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(_EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENT_DIR))

from replay_original_demo import replay_demo_rollout
from robosuite_env_loader import check_environment

DEFAULT_SUCCESS = _EXPERIMENT_DIR.parents[2] / "mnt" / "data" / "demo.hdf5"
DEFAULT_FAILED = _EXPERIMENT_DIR.parents[2] / "mnt" / "data" / "demo_failed.hdf5"
DEFAULT_OUTPUT = _EXPERIMENT_DIR / "outputs" / "rollout_smoke"


def main() -> int:
    parser = argparse.ArgumentParser(description="V2-B1 rollout smoke: replay original demos")
    parser.add_argument("--success-hdf5", type=Path, default=DEFAULT_SUCCESS)
    parser.add_argument("--failed-hdf5", type=Path, default=DEFAULT_FAILED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    video_dir = args.output_dir / "videos"
    video_dir.mkdir(exist_ok=True)

    env_check = check_environment()
    report: dict = {
        "task": "V2-B1_rollout_smoke",
        "environment_check": env_check,
        "results": [],
        "acceptance": {},
    }

    if not env_check["available"]:
        report["blocked"] = True
        report["block_reason"] = env_check["blockers"]
        out_path = args.output_dir / "rollout_result.json"
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 1

    jobs = [
        (str(args.success_hdf5), "demo_0", "success", video_dir / "success_demo_0.mp4"),
        (str(args.failed_hdf5), "demo_4", "failed", video_dir / "failed_demo_4.mp4"),
    ]

    for hdf5_path, demo_key, label, video_path in jobs:
        try:
            result = replay_demo_rollout(
                hdf5_path,
                demo_key,
                label,
                video_path=None if args.no_video else video_path,
                record_video=not args.no_video,
            )
            report["results"].append(result)
        except Exception as exc:
            report["results"].append(
                {
                    "demo_name": demo_key,
                    "source_file": hdf5_path,
                    "blocked": True,
                    "block_reason": str(exc),
                    "replay_success": False,
                }
            )

    success_result = next((r for r in report["results"] if r.get("demo_name") == "demo_0"), None)
    failed_result = next((r for r in report["results"] if r.get("demo_name") == "demo_4"), None)

    report["acceptance"] = {
        "env_loaded": env_check["available"],
        "success_demo_replayed": success_result is not None and not success_result.get("blocked"),
        "success_demo_near_success": bool(success_result and success_result.get("metrics_near_success")),
        "success_replay_success_flag": bool(success_result and success_result.get("replay_success")),
        "failed_demo_4_replayed": failed_result is not None and not failed_result.get("blocked"),
        "failed_demo_4_still_failed": bool(
            failed_result
            and not failed_result.get("success_flag")
            and failed_result.get("failure_guess") in {"insertion_failed", "unknown_failed", "alignment_failed"}
        ),
    }

    out_path = args.output_dir / "rollout_result.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if env_check["available"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
