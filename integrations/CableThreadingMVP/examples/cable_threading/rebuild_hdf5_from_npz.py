#!/usr/bin/env python3
"""Rebuild HDF5 from NPZ using recorded trajectory replay (fixes oracle re-export bug)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.cable_threading.hdf5_replay import load_npz_trajectories, replay_trajectory_collect_obs
from examples.cable_threading.utils import make_env
from robosuite.utils.dlo.hdf5_dataset import save_dataset_hdf5


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild cable threading HDF5 from NPZ")
    parser.add_argument("--npz", required=True, help="Source dataset.npz path")
    parser.add_argument("--hdf5-out", required=True, help="Output dataset.hdf5 path")
    parser.add_argument("--robot", default="Panda")
    parser.add_argument("--cable-model", default="composite_cable")
    parser.add_argument("--difficulty", default="easy")
    parser.add_argument("--grasp-mode", default="attachment")
    parser.add_argument("--horizon", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    trajectories, episode_metadata, metadata = load_npz_trajectories(args.npz)
    metadata = dict(metadata)
    metadata.update(
        {
            "robot": args.robot,
            "cable_model": args.cable_model,
            "difficulty": args.difficulty,
            "grasp_mode": args.grasp_mode,
            "attachment_side_channel": args.grasp_mode == "attachment",
            "attachment_field": "attachment_enabled",
            "side_channel_keys": ["attachment_enabled"] if args.grasp_mode == "attachment" else [],
            "horizon": args.horizon,
            "seed": args.seed,
        }
    )

    image_camera_names = ["agentview", "robot0_eye_in_hand"]
    raw_obs_trajectories = []
    for ep_idx, (traj, meta) in enumerate(zip(trajectories, episode_metadata)):
        ep_seed = int(meta.get("seed", args.seed + ep_idx))
        env = make_env(
            robot=args.robot,
            cable_model=args.cable_model,
            grasp_mode=args.grasp_mode,
            difficulty=args.difficulty,
            horizon=args.horizon,
            seed=ep_seed,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            camera_names=image_camera_names,
        )
        try:
            raw_traj = replay_trajectory_collect_obs(env, traj, seed=ep_seed)
            raw_obs_trajectories.append(raw_traj)
            print(f"[rebuild] episode {ep_idx}: {len(raw_traj)} steps (seed={ep_seed})")
        finally:
            env.close()

    hdf5_path = Path(args.hdf5_out)
    save_dataset_hdf5(
        hdf5_path,
        raw_obs_trajectories,
        metadata=metadata,
        episode_metadata=episode_metadata,
    )

    manifest_path = hdf5_path.with_name("dataset.manifest.json")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {}
    manifest.update(
        {
            "dataset": str(Path(args.npz).resolve()),
            "hdf5": str(hdf5_path.resolve()),
            "grasp_mode": args.grasp_mode,
            "attachment_side_channel": args.grasp_mode == "attachment",
            "attachment_field": "attachment_enabled",
            "side_channel_keys": ["attachment_enabled"] if args.grasp_mode == "attachment" else [],
            "num_successful": len(trajectories),
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved_manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
