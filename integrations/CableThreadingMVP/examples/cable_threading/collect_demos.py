import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.cable_threading.utils import (
    DEFAULT_OBS_KEYS,
    SUPPORTED_CABLE_MODELS,
    SUPPORTED_DIFFICULTIES,
    aggregate_rows,
    make_env,
    rollout_expert_episode,
    save_dataset,
    write_results_csv,
)
from examples.dlo.common import add_grasp_mode_arg, apply_scene_randomization, jsonable, resolve_scene_randomization
from robosuite.utils.dlo.episode_schema import formal_metadata
from robosuite.utils.dlo.trajectory_quality import quality_report_from_trajectories, write_quality_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="CableThreading")
    parser.add_argument("--episodes", type=int, default=50, help="最大尝试回合数")
    parser.add_argument("--min-success", type=int, default=None, help="目标成功回合数（未达标前持续采集，受 --episodes 上限限制）")
    parser.add_argument("--horizon", type=int, default=None, help="每 episode 最大步数（默认按 20Hz/250 步自动缩放）")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--robot", type=str, default="UR5e")
    parser.add_argument("--cable-model", choices=SUPPORTED_CABLE_MODELS, default="rmb")
    add_grasp_mode_arg(parser)
    parser.add_argument("--difficulty", choices=SUPPORTED_DIFFICULTIES, default="easy")
    parser.add_argument("--control-freq", type=float, default=20, help="控制频率 Hz（默认 20）")
    parser.add_argument("--out", type=str, default="datasets/cable_threading/ur5e_rmb_scripted_train.npz")
    parser.add_argument("--results-out", type=str, default="results/cable_threading_collect.csv")
    parser.add_argument("--failures-out", type=str, default="results/cable_threading_collect_failures.json")
    parser.add_argument("--quality-out", default=None, help="Optional JSON trajectory quality report path.")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--camera", type=str, default="frontview")
    parser.add_argument("--render-sleep", type=float, default=0.0)
    parser.add_argument("--scene-randomization", choices=["random", "fixed"], default="random")
    parser.add_argument("--fixed-initial-state", action="store_true")
    parser.add_argument("--random-initial-state", action="store_true")
    parser.add_argument("--record-video", action="store_true", help="录制视频")
    parser.add_argument("--video-output", default=None, help="视频输出路径")
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--video-width", type=int, default=1280)
    parser.add_argument("--video-height", type=int, default=720)
    parser.add_argument("--video-quality", choices=["standard", "high", "ultra"], default="high")
    parser.add_argument("--video-output-dir", default=None, help="视频输出目录，默认 ~/Videos/rms")
    parser.add_argument("--hdf5-out", default=None, help="HDF5 数据集输出路径（自动启用 offscreen rendering 采集图像）")
    parser.add_argument("--lerobot-out", default=None, help="LeRobot v3.0 数据集输出目录（自动启用 offscreen rendering 采集图像）")
    args = parser.parse_args()

    # horizon 自动缩放：未指定时按控制频率等比调整（基准 20Hz / 250 步）
    _BASE_FREQ = 20
    _BASE_HORIZON = 250
    if args.horizon is None:
        args.horizon = int(_BASE_HORIZON * args.control_freq / _BASE_FREQ)

    # --hdf5-out / --lerobot-out 自动启用 offscreen rendering 和 camera obs
    hdf5_enabled = bool(args.hdf5_out)
    lerobot_enabled = bool(args.lerobot_out)
    need_image_data = hdf5_enabled or lerobot_enabled
    image_camera_names = ["agentview", "robot0_eye_in_hand"] if need_image_data else None

    camera = args.camera or "frontview"
    need_offscreen = args.render or args.record_video or need_image_data
    # Merge video camera with HDF5/image cameras (deduplicate, preserve order)
    _cams = []
    if need_image_data and image_camera_names:
        _cams.extend(image_camera_names)
    if args.record_video and camera not in _cams:
        _cams.append(camera)
    env_camera_names = _cams or None
    env = make_env(
        env_name=args.env,
        robot=args.robot,
        horizon=args.horizon,
        seed=args.seed,
        has_renderer=args.render,
        has_offscreen_renderer=need_offscreen,
        render_camera=camera,
        cable_model=args.cable_model,
        grasp_mode=args.grasp_mode,
        difficulty=args.difficulty,
        use_camera_obs=args.record_video or need_image_data,
        camera_names=env_camera_names,
        camera_widths=[args.video_width] * len(env_camera_names) if env_camera_names else None,
        camera_heights=[args.video_height] * len(env_camera_names) if env_camera_names else None,
        control_freq=args.control_freq,
    )
    scene_randomization = resolve_scene_randomization(args, default="random")
    apply_scene_randomization(env, scene_randomization)

    # Video setup（每 episode 独立视频文件）
    record_video = args.record_video
    video_camera = camera
    video_dir = None
    video_stamp = None
    if record_video:
        import imageio.v2 as iio
        from datetime import datetime
        video_dir = Path(args.video_output_dir or "~/Videos/rms").expanduser()
        video_dir.mkdir(parents=True, exist_ok=True)
        video_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"video_output_dir: {video_dir}")

    trajectories = []
    episode_metadata = []
    rows = []
    failures = []

    # Base metadata computed before try so HDF5 save in finally can use it
    base_dataset_metadata = formal_metadata(
        env_name=args.env,
        robot=args.robot,
        controller="default",
        seed=args.seed,
        horizon=args.horizon,
        scene_randomization=scene_randomization,
        policy="robot_endpoint_oracle",
        cable_model=args.cable_model,
        grasp_mode=args.grasp_mode,
        difficulty=args.difficulty,
        success_semantics="final_only",
        obs_keys=DEFAULT_OBS_KEYS,
        attachment_side_channel=args.grasp_mode == "attachment",
        attachment_field="attachment_enabled",
        attachment_policy="recorded_or_controller" if args.grasp_mode == "attachment" else None,
        attachment_input_mode="not_used_by_policy",
        attachment_control_mode="eval_controller",
        include_attachment_obs=False,
        env_params={
            "pole_spacing": 0.05,
            "pole_radius": 0.01,
            "pole_height": 0.06,
        },
        data_version="cable_threading_v1",
    )

    min_success = args.min_success
    max_attempts = args.episodes
    # 如果指定了 --min-success 但 --episodes 未显式设置，自动放大上限
    if min_success is not None and args.episodes == 50:
        max_attempts = max(args.episodes, min_success * 10)

    try:
        episode = 0
        while episode < max_attempts:
            if min_success is not None and len(trajectories) >= min_success:
                break
            episode_seed = args.seed + episode
            summary, phase_log, transitions = rollout_expert_episode(
                env,
                render_sleep=args.render_sleep,
                obs_keys=DEFAULT_OBS_KEYS,
                record_trajectory=True,
                record_raw_obs=need_image_data,
                episode_index=episode,
                seed=episode_seed,
            )
            rows.append(summary)
            print(episode, summary)

            # Video frame capture: replay the episode to record frames（每 episode 独立文件）
            if record_video and transitions:
                ep_video_path = str(video_dir / f"{args.env}_{video_stamp}_ep{episode}.mp4")
                fps = args.video_fps
                quality_map = {"standard": 7, "high": 9, "ultra": 10}
                q = quality_map.get(args.video_quality, 9)
                video_writer = iio.get_writer(
                    ep_video_path, format="FFMPEG", mode="I",
                    fps=fps, codec="libx264", quality=q,
                    pixelformat="yuv420p", macro_block_size=16,
                )
                img_key = f"{video_camera}_image"
                env.rng = np.random.default_rng(episode_seed)
                obs0 = env.reset()
                if episode == 0:
                    print(f"[video] img_key={img_key}, in_obs={img_key in obs0}, obs_keys={[k for k in obs0 if 'image' in k]}")
                prev_attach = False
                ep_frames = 0
                for t in transitions:
                    want_attach = bool(t.get("attachment_enabled", False))
                    if want_attach and not prev_attach:
                        env.set_attachment_enabled(True)
                        if getattr(env, "_attach_pending", False) and hasattr(env, "_activate_flex_attachment"):
                            env._activate_flex_attachment()
                    elif not want_attach and prev_attach:
                        env.set_attachment_enabled(False)
                    prev_attach = want_attach
                    action = np.asarray(t["action"], dtype=np.float32)
                    obs, _, _, _ = env.step(action)
                    if img_key in obs:
                        video_writer.append_data(obs[img_key][::-1])
                        ep_frames += 1
                video_writer.close()
                print(f"[video] episode {episode}: wrote {ep_frames}/{len(transitions)} frames → {ep_video_path}")

            if summary["final_success"]:
                trajectories.append(transitions)
                episode_metadata.append(
                    {
                        "episode": episode,
                        "seed": episode_seed,
                        "phase_log": phase_log,
                        "summary": summary,
                        "reset_summary": jsonable(getattr(env, "last_reset_summary", {})),
                        "scene_randomization": scene_randomization,
                    }
                )
            else:
                failures.append(
                    {
                        "episode": episode,
                        "seed": episode_seed,
                        "summary": summary,
                        "phase_log": phase_log,
                        "reset_summary": jsonable(getattr(env, "last_reset_summary", {})),
                        "scene_randomization": scene_randomization,
                    }
                )
            episode += 1
    finally:
        # HDF5 / LeRobot：直接写入采集轨迹（禁止 reset 后 replay 重跑 expert）
        if need_image_data and trajectories:
            if hdf5_enabled:
                try:
                    from robosuite.utils.dlo.hdf5_dataset import (
                        HDF5_IMAGE_KEYS,
                        HDF5_LOW_DIM_KEYS,
                        HDF5_TASK_OBS_KEYS,
                        save_dataset_hdf5,
                        validate_hdf5_trajectory_actions,
                    )

                    hdf5_metadata = dict(base_dataset_metadata)
                    if args.grasp_mode == "attachment":
                        hdf5_metadata["side_channel_keys"] = ["attachment_enabled"]
                    save_dataset_hdf5(
                        args.hdf5_out,
                        trajectories,
                        image_keys=list(HDF5_IMAGE_KEYS),
                        low_dim_keys=list(HDF5_LOW_DIM_KEYS),
                        task_obs_keys=list(HDF5_TASK_OBS_KEYS),
                        metadata=hdf5_metadata,
                        episode_metadata=episode_metadata,
                    )
                    validation = validate_hdf5_trajectory_actions(args.hdf5_out, trajectories)
                    if not validation["ok"]:
                        raise RuntimeError(
                            "HDF5 validation failed: " + "; ".join(validation["issues"])
                        )
                    print(
                        f"[hdf5] validated source trajectories "
                        f"(max_diff={validation['max_action_diff']})"
                    )
                except Exception as e:
                    import traceback

                    print(f"[hdf5] ERROR during save: {e}")
                    traceback.print_exc()
                    hdf5_path = Path(args.hdf5_out).expanduser()
                    if hdf5_path.exists():
                        hdf5_path.unlink()
                        print(f"[hdf5] deleted corrupt file: {hdf5_path}")
            # LeRobot v3.0 保存（仍需要 raw_obs，已由采集阶段写入）
            if lerobot_enabled:
                try:
                    from robosuite.utils.dlo.lerobot_dataset import save_dataset_lerobot
                    save_dataset_lerobot(
                        args.lerobot_out,
                        trajectories,
                        fps=args.control_freq,
                        task_description=f"{args.env}",
                        success_flags=[m["summary"]["final_success"] for m in episode_metadata],
                    )
                except Exception as e:
                    import traceback
                    print(f"[lerobot] ERROR during save: {e}")
                    traceback.print_exc()
        env.close()

    write_results_csv(args.results_out, rows)
    stats = aggregate_rows(rows)
    print("saved_csv:", Path(args.results_out).expanduser())
    print("final_success_rate:", stats["final_success_rate"])
    print("ever_success_rate:", stats["ever_success_rate"])
    print("successful_episodes:", len(trajectories))
    print("total_attempts:", episode)
    if min_success is not None and len(trajectories) < min_success:
        print(f"WARNING: target {min_success} successes not reached after {episode} attempts ({len(trajectories)} collected)")

    failures_path = Path(args.failures_out).expanduser()
    failures_path.parent.mkdir(parents=True, exist_ok=True)
    failures_path.write_text(json.dumps(failures, indent=2))
    print("saved_failures:", failures_path)

    if not trajectories:
        raise RuntimeError("No final-success scripted episodes were collected.")

    dataset_metadata = base_dataset_metadata
    quality_thresholds = None
    if (
        args.grasp_mode == "physical"
        and args.robot == "Panda"
        and args.cable_model in {"flex", "flex_cable", "flexcomp"}
    ):
        quality_thresholds = {"phase_target_jump_fail": 0.2}
    quality_report = quality_report_from_trajectories(
        trajectories,
        metadata=dataset_metadata,
        table_z=getattr(env, "table_top_z", None),
        thresholds=quality_thresholds,
    )
    if args.quality_out:
        write_quality_json(args.quality_out, quality_report)
        print("saved_quality:", Path(args.quality_out).expanduser())
    print("trajectory_quality_passed:", bool(quality_report["passed"]))
    print("trajectory_quality_failures:", int(quality_report["failure_count"]))
    dataset_metadata = {**dataset_metadata, "trajectory_quality": quality_report}
    save_dataset(
        args.out,
        trajectories=trajectories,
        obs_keys=DEFAULT_OBS_KEYS,
        metadata=dataset_metadata,
        episode_metadata=episode_metadata,
    )
    print("saved_dataset:", Path(args.out).expanduser())


if __name__ == "__main__":
    main()
