#!/usr/bin/env python3
"""在单臂线缆仿真环境中 rollout Diffusion Policy checkpoint。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

LAB_ROOT = Path(__file__).resolve().parents[1]
CABLE_ROOT = LAB_ROOT.parents[1]
sys.path.insert(0, str(LAB_ROOT))
sys.path.insert(0, str(CABLE_ROOT))

from dp_lab.policy_runtime import DiffusionPolicyAdapter
from examples.cable_threading.utils import aggregate_rows, make_env, rollout_policy_episode, write_results_csv


def main() -> int:
    parser = argparse.ArgumentParser(description="DP lab simulation rollout")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--robot", type=str, default="Panda")
    parser.add_argument("--cable-model", type=str, default="composite_cable")
    parser.add_argument("--difficulty", type=str, default="easy")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else LAB_ROOT / "outputs" / "rollout"
    out_dir.mkdir(parents=True, exist_ok=True)

    image_camera_names = ["agentview", "robot0_eye_in_hand"]
    env = make_env(
        robot=args.robot,
        cable_model=args.cable_model,
        difficulty=args.difficulty,
        horizon=args.horizon,
        seed=args.seed,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=image_camera_names,
        camera_heights=[256] * len(image_camera_names),
        camera_widths=[256] * len(image_camera_names),
    )

    policy = DiffusionPolicyAdapter(args.checkpoint, device=args.device)
    summaries = []
    successes = 0

    try:
        for episode in range(args.episodes):
            episode_seed = args.seed + episode
            summary, _ = rollout_policy_episode(
                env,
                policy,
                episode_index=episode,
                seed=episode_seed,
                policy_name="diffusion_policy",
            )
            summaries.append(summary)
            if summary.get("final_success"):
                successes += 1
            print(
                f"episode={episode} success={summary.get('final_success')} "
                f"steps={summary.get('steps')} return={summary.get('return', 0):.3f}"
            )
    finally:
        env.close()

    success_rate = successes / max(args.episodes, 1)
    agg = aggregate_rows(summaries)
    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "episodes": args.episodes,
        "successes": successes,
        "success_rate": success_rate,
        "aggregate": agg,
        "robot": args.robot,
        "cable_model": args.cable_model,
        "seed": args.seed,
    }
    result_path = out_dir / "rollout_summary.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = out_dir / "rollout_results.csv"
    write_results_csv(csv_path, summaries)

    print(f"success_rate={success_rate:.1%} ({successes}/{args.episodes})")
    print(f"saved: {result_path}")
    print(f"saved: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
