from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def _success(env: Any) -> bool:
    try:
        result = env.is_success()
        if isinstance(result, dict):
            return bool(result.get("task") or result.get("success"))
        return bool(result)
    except Exception:
        try:
            return bool(env._check_success())
        except Exception:
            return False


def _frame(env: Any) -> np.ndarray | None:
    try:
        image = env.render(mode="rgb_array", height=480, width=640, camera_name="agentview")
        return np.asarray(image, dtype=np.uint8)
    except Exception:
        try:
            return np.flipud(
                np.asarray(env.sim.render(camera_name="agentview", height=480, width=640), dtype=np.uint8)
            )
        except Exception:
            return None


def _write_video(path: Path, frames: list[np.ndarray]) -> bool:
    if not frames:
        return False
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(path), frames, fps=20)
    return path.is_file() and path.stat().st_size > 0


def run(args: argparse.Namespace) -> int:
    job_root = Path(args.job_root).resolve()
    status_path = job_root / "status.json"
    checkpoint = Path(args.checkpoint).resolve()
    started = _now()
    base = {
        "evalJobId": args.eval_job_id,
        "taskType": "nut_assembly",
        "evaluationMode": "trained_model_evaluation",
        "status": "running",
        "phase": "evaluating",
        "progress": 0.0,
        "currentEpisode": 0,
        "totalEpisodes": args.episodes,
        "message": "螺母装配模型评测运行中",
        "startedAt": started,
        "updatedAt": started,
        "metrics": {"modelAssetId": args.model_asset_id or None},
        "artifacts": {},
    }
    _write_json(status_path, base)

    from robomimic.utils import file_utils as FileUtils
    from robomimic.utils import torch_utils as TorchUtils

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    policy, checkpoint_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=str(checkpoint), device=device, verbose=False
    )
    # robomimic 0.3 的 EnvRobosuite 包装仍依赖已淘汰的 mujoco_py。
    # 本任务训练数据来自 robosuite NutAssemblySquare，直接创建同构环境，
    # 保留原始 obs 字典供 RolloutPolicy 归一化和选键。
    import robosuite

    env = robosuite.make(
        env_name="NutAssemblySquare",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=bool(args.record),
        use_camera_obs=False,
        control_freq=20,
        horizon=args.horizon,
    )

    episode_rows: list[dict[str, Any]] = []
    successes = 0
    try:
        for episode in range(args.episodes):
            np.random.seed(args.seed + episode)
            obs = env.reset()
            policy.start_episode()
            frames: list[np.ndarray] = []
            total_reward = 0.0
            succeeded = False
            length = 0
            for step in range(args.horizon):
                action = policy(ob=obs)
                obs, reward, done, _ = env.step(action)
                total_reward += float(reward)
                length = step + 1
                if args.record and step % 5 == 0:
                    image = _frame(env)
                    if image is not None:
                        frames.append(image)
                if _success(env):
                    succeeded = True
                    break
                if done:
                    break
            if succeeded:
                successes += 1
            video_rel = None
            if args.record and frames:
                candidate = job_root / "videos" / f"episode_{episode:02d}.mp4"
                if _write_video(candidate, frames):
                    video_rel = str(candidate.relative_to(job_root))
            episode_rows.append(
                {
                    "episodeIndex": episode,
                    "success": succeeded,
                    "reward": total_reward,
                    "episodeLength": length,
                    "seed": args.seed + episode,
                    "videoPath": video_rel,
                }
            )
            completed = episode + 1
            live = dict(base)
            live.update(
                {
                    "progress": completed / args.episodes,
                    "currentEpisode": completed,
                    "updatedAt": _now(),
                    "metrics": {
                        "modelAssetId": args.model_asset_id or None,
                        "completedEpisodes": completed,
                        "successfulEpisodes": successes,
                        "successRate": successes / completed,
                    },
                }
            )
            _write_json(status_path, live)
    finally:
        env.close()

    success_rate = successes / args.episodes
    aggregate = {
        "evalJobId": args.eval_job_id,
        "taskType": "nut_assembly",
        "taskTemplateId": "nut_assembly_single_arm",
        "evaluationMode": "trained_model_evaluation",
        "status": "completed",
        "backendType": "robomimic_bc",
        "episodeCount": args.episodes,
        "successEpisodes": successes,
        "failureCount": args.episodes - successes,
        "successRate": success_rate,
        "horizon": args.horizon,
        "seed": args.seed,
        "checkpointPath": str(checkpoint),
        "modelAssetId": args.model_asset_id or None,
    }
    _write_json(job_root / "results" / "aggregate_result.json", aggregate)
    _write_json(job_root / "results" / "per_episode_results.json", {"episodes": episode_rows})
    videos = [row["videoPath"] for row in episode_rows if row.get("videoPath")]
    finished = dict(base)
    finished.update(
        {
            "status": "completed",
            "phase": "completed",
            "progress": 1.0,
            "currentEpisode": args.episodes,
            "message": "螺母装配模型评测已完成",
            "updatedAt": _now(),
            "finishedAt": _now(),
            "metrics": aggregate,
            "artifacts": {"videos": videos, "resultsJson": "results/aggregate_result.json"},
        }
    )
    _write_json(status_path, finished)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-root", required=True)
    parser.add_argument("--eval-job-id", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-asset-id", default="")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()
    try:
        return run(args)
    except Exception as exc:
        job_root = Path(args.job_root).resolve()
        payload = {
            "evalJobId": args.eval_job_id,
            "taskType": "nut_assembly",
            "evaluationMode": "trained_model_evaluation",
            "status": "failed",
            "phase": "failed",
            "progress": 0.0,
            "currentEpisode": 0,
            "totalEpisodes": args.episodes,
            "message": f"螺母装配模型评测失败: {exc}",
            "error": str(exc),
            "updatedAt": _now(),
            "finishedAt": _now(),
            "metrics": {},
            "artifacts": {},
        }
        _write_json(job_root / "status.json", payload)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
