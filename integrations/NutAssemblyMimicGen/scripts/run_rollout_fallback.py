#!/usr/bin/env python3
"""Run robosuite rollout in cable-threading-mvp subprocess (MimicGen fallback path)."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

_INTEGRATION_ROOT = Path(__file__).resolve().parents[1]
if str(_INTEGRATION_ROOT) not in sys.path:
    sys.path.insert(0, str(_INTEGRATION_ROOT))

from utils.hdf5_writer import save_rollout_hdf5
from utils.runtime_env import build_rollout_subprocess_env, collect_rollout_runtime_diagnostics
from utils.robosuite_rollout import rollout_episodes


def main() -> int:
    parser = argparse.ArgumentParser(description="NutAssembly robosuite rollout fallback worker")
    parser.add_argument("--job-root", required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--env-name", default="Square_D0")
    parser.add_argument("--render-video", action="store_true")
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    job_root = Path(args.job_root).resolve()
    log_path = job_root / "logs" / "fallback_rollout.log"
    video_path = job_root / "videos" / "generate.mp4"
    hdf5_out = job_root / "datasets" / "nut_assembly_generated.hdf5"
    output_json = Path(args.output_json)

    log_lines = ["=== robosuite rollout fallback (cable-threading-mvp) ==="]
    log_lines.extend(collect_rollout_runtime_diagnostics())

    try:
        rollout_result = rollout_episodes(
            env_name=args.env_name,
            episodes=args.episodes,
            seed=args.seed,
            horizon=args.horizon,
            render_video=args.render_video,
            video_path=video_path if args.render_video else None,
            debug_log_path=job_root / "logs" / "stage_debug.jsonl",
        )
        hdf5_info = save_rollout_hdf5(
            hdf5_out,
            rollout_result["episodes"],
            env_name=rollout_result["runtimeEnvName"],
            generation_mode="robosuite_rollout",
            policy_mode=str(rollout_result.get("policyMode") or "partial_scripted"),
        )
        payload = {
            "ok": True,
            "rolloutResult": {
                "runtimeEnvName": rollout_result.get("runtimeEnvName"),
                "successEpisodes": rollout_result.get("successEpisodes"),
                "failedEpisodes": rollout_result.get("failedEpisodes"),
                "successRate": rollout_result.get("successRate"),
                "policyMode": rollout_result.get("policyMode"),
                "episodesGenerated": rollout_result.get("episodesGenerated"),
                "failureDistribution": rollout_result.get("failureDistribution"),
                "videoResult": rollout_result.get("videoResult"),
            },
            "hdf5Info": hdf5_info,
            "hdf5Path": str(hdf5_out),
        }
        log_lines.append("fallback_rollout=success")
        output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        return 0
    except Exception as exc:
        tb = traceback.format_exc()
        log_lines.extend([f"fallback_rollout=failed", f"error={exc}", tb])
        log_path.write_text("\n".join(log_lines), encoding="utf-8")
        output_json.write_text(
            json.dumps({"ok": False, "error": str(exc), "traceback": tb}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
