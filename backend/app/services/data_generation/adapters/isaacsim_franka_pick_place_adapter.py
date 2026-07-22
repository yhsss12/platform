#!/usr/bin/env python3
"""CLI entry for Isaac Sim Franka Pick Place data generation (platform-compatible layout)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.isaacsim_franka_pick_place_data_worker import execute_job  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Isaac Sim Franka Pick Place data generation")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-trajectory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--job-id", default="data_gen_cli_local")
    args = parser.parse_args()

    job_dir = Path(args.output_dir).expanduser().resolve()
    job_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("metadata", "logs", "results", "episodes", "videos"):
        (job_dir / sub).mkdir(parents=True, exist_ok=True)

    config = {
        "episodes": max(1, min(args.episodes, 5)),
        "seed": args.seed,
        "save_video": args.save_video,
        "save_trajectory": args.save_trajectory,
        "headless": args.headless,
    }
    config_path = job_dir / "metadata" / "job_config.json"
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    result = execute_job(job_dir, args.job_id, config)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
