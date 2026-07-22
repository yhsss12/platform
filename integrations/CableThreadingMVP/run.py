#!/usr/bin/env python3
"""
CableThreadingMVP — 统一入口脚本

用法：
    python run.py env-test                          # 环境冒烟测试
    python run.py expert --episodes 10              # 采集专家数据集
    python run.py eval --episodes 20                # 评估专家策略
    python run.py video --episodes 1                # 录制视频

    # Panda + flex 线缆
    python run.py expert --episodes 10 --robot Panda --cable-model flex

    # 带图像的 HDF5 输出
    python run.py expert --episodes 10 --hdf5-out datasets/threading.hdf5
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# 确保 examples/ 可导入
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))


def _live_env_kwargs(live_enabled, live_config):
    from examples.cable_threading.utils import live_camera_kwargs

    if not live_enabled or live_config is None:
        return {}
    return live_camera_kwargs(live_config)


def _mark_live_failure(live_config, exc):
    from examples.cable_threading.utils import write_live_status

    if live_config.get("failedStage") == "obs_validation":
        write_live_status(live_config)
        return

    live_config["status"] = "failed"
    live_config["error"] = str(exc)
    msg = str(exc).lower()
    if "missing observation keys" in msg:
        live_config["failedStage"] = "rollout"
        live_config["failureReason"] = "obs_key_mismatch"
        live_config["errorMessage"] = (
            "策略评测环境与模型观测不匹配：评测环境未提供模型所需的观测项"
            "（例如 robot0_eye_in_hand_image）。"
        )
    else:
        live_config["failedStage"] = live_config.get("failedStage") or "rollout"
        live_config["failureReason"] = live_config.get("failureReason") or "runner_exception"
        live_config["errorMessage"] = live_config.get("errorMessage") or (
            "评测 runner 执行异常，请查看日志中的 Traceback。"
        )
    live_config["logPaths"] = {
        "stdout": "logs/run.log",
        "stderr": "logs/run.log",
        "run": "logs/run.log",
    }
    write_live_status(live_config)


def _live_viz_settings(args):
    from examples.cable_threading.utils import (
        DEFAULT_LIVE_FRAME_HEIGHT,
        DEFAULT_LIVE_FRAME_WIDTH,
        DEFAULT_LIVE_JPEG_QUALITY,
    )

    return {
        "frame_width": int(getattr(args, "live_frame_width", DEFAULT_LIVE_FRAME_WIDTH)),
        "frame_height": int(getattr(args, "live_frame_height", DEFAULT_LIVE_FRAME_HEIGHT)),
        "jpeg_quality": int(getattr(args, "live_jpeg_quality", DEFAULT_LIVE_JPEG_QUALITY)),
    }


def _add_live_viz_args(p):
    from examples.cable_threading.utils import (
        DEFAULT_LIVE_FRAME_HEIGHT,
        DEFAULT_LIVE_FRAME_WIDTH,
        DEFAULT_LIVE_JPEG_QUALITY,
    )

    p.add_argument(
        "--live-frame-width",
        type=int,
        default=DEFAULT_LIVE_FRAME_WIDTH,
        help=f"live 可视化帧宽度（默认 {DEFAULT_LIVE_FRAME_WIDTH}，不影响训练 obs）",
    )
    p.add_argument(
        "--live-frame-height",
        type=int,
        default=DEFAULT_LIVE_FRAME_HEIGHT,
        help=f"live 可视化帧高度（默认 {DEFAULT_LIVE_FRAME_HEIGHT}，不影响训练 obs）",
    )
    p.add_argument(
        "--live-jpeg-quality",
        type=int,
        default=DEFAULT_LIVE_JPEG_QUALITY,
        help=f"live JPEG 质量（默认 {DEFAULT_LIVE_JPEG_QUALITY}）",
    )


def cmd_env_test(args):
    """冒烟测试：创建环境 → reset → step → 检查观测/奖励/成功判定。"""
    from examples.cable_threading.utils import DEFAULT_OBS_KEYS, make_env

    env = make_env(
        robot=args.robot,
        cable_model=args.cable_model,
        grasp_mode=args.grasp_mode,
        difficulty=args.difficulty,
        horizon=args.horizon,
        seed=args.seed,
    )
    try:
        obs = env.reset()
        print(f"[env-test] obs keys: {list(obs.keys())}")
        print(f"[env-test] obs dim: {sum(np.asarray(v).size for v in obs.values())}")

        low, high = env.action_spec
        action = (low + high) / 2  # 中位动作
        obs2, reward, done, info = env.step(action)
        print(f"[env-test] step reward: {reward:.4f}")
        print(f"[env-test] step done: {done}")
        print(f"[env-test] info keys: {list(info.keys())}")
        print(f"[env-test] final_success: {info.get('final_success', 'N/A')}")
        print("[env-test] PASSED")
    finally:
        env.close()


def cmd_expert(args):
    """运行专家策略，采集数据集。"""
    from examples.cable_threading.utils import (
        DEFAULT_OBS_KEYS,
        SUPPORTED_CABLE_MODELS,
        aggregate_rows,
        build_expert_env_make_kwargs,
        make_env,
        rollout_expert_episode,
        save_dataset,
        write_live_status,
        write_live_timeline,
        write_results_csv,
        synthesize_live_video,
    )
    from robosuite.utils.dlo.episode_schema import formal_metadata
    from robosuite.utils.dlo.trajectory_quality import quality_report_from_trajectories

    live_frame_dir = getattr(args, "live_frame_dir", None)
    live_enabled = bool(live_frame_dir)
    lerobot_out = getattr(args, "lerobot_out", None)
    need_image_data = live_enabled or bool(args.hdf5_out) or bool(lerobot_out)
    live_config = None
    if live_enabled:
        live_dir = Path(live_frame_dir)
        live_dir.mkdir(parents=True, exist_ok=True)
        save_frames = bool(getattr(args, "live_save_frames", False))
        video_out = getattr(args, "live_video_out", None)
        status_path = (
            Path(args.live_status_out)
            if getattr(args, "live_status_out", None)
            else live_dir / "status.json"
        )
        live_config = {
            "status": "running",
            "jobType": "generate",
            "frame_dir": str(live_dir),
            "frames_dir": str(live_dir / "frames") if save_frames else None,
            "save_frames": save_frames,
            "video_out": video_out,
            "video_fps": int(getattr(args, "live_video_fps", 20)),
            "generate_video_status": "pending" if video_out else None,
            "generate_video_exists": False,
            "generate_video_size_bytes": 0,
            "saved_frame_count": 0,
            "frame_every": int(getattr(args, "live_frame_every", 5)),
            "camera": getattr(args, "live_camera", "agentview"),
            "status_path": str(status_path),
            "episode": 0,
            "episodes": args.episodes,
            "horizon": args.horizon,
            "step": 0,
            "frame_count": 0,
            "phase": "",
            "successful_episodes": 0,
            "success_so_far": None,
            "final_success_rate": None,
            "error": None,
            "timeline_path": (
                str(Path(args.live_timeline_out))
                if getattr(args, "live_timeline_out", None)
                else None
            ),
            "timeline_events": [],
            "timeline_seen": set(),
            "frame_status": "warming_up",
            "skipped_invalid_frame": 0,
            "has_valid_frame": False,
            "live_warmup_steps": 10,
            "live_required_consecutive_valid": 3,
            "live_sim_forward_warmup": 5,
            "live_obs_resized": False,
            **_live_viz_settings(args),
        }
        if save_frames:
            from examples.cable_threading.obs_schema import DEFAULT_EVAL_DISPLAY_CAMERA

            record_camera = getattr(args, "live_display_camera", DEFAULT_EVAL_DISPLAY_CAMERA)
            live_config.update(
                {
                    "display_camera": record_camera,
                    "record_camera": record_camera,
                    "frame_source": "obs_image",
                    "allow_camera_fallback": False,
                    "camera_fallback_used": False,
                }
            )
            (live_dir / "frames").mkdir(parents=True, exist_ok=True)
        write_live_status(live_config)

    env = make_env(
        robot=args.robot,
        cable_model=args.cable_model,
        grasp_mode=args.grasp_mode,
        difficulty=args.difficulty,
        horizon=args.horizon,
        seed=args.seed,
        has_offscreen_renderer=need_image_data,
        use_camera_obs=need_image_data,
        **build_expert_env_make_kwargs(
            live_enabled=live_enabled,
            live_config=live_config,
            hdf5_out=need_image_data,
        ),
    )

    trajectories = []
    episode_metadata = []
    rows = []
    failures = []

    try:
        for episode in range(args.episodes):
            episode_seed = args.seed + episode
            if live_config is not None:
                live_config["episode"] = episode
                live_config["step"] = 0
                live_config["phase"] = ""
                live_config.pop("_last_display_frame", None)
                required_valid = int(live_config.get("live_required_consecutive_valid", 3))
                if live_config.get("has_valid_frame"):
                    live_config["_consecutive_valid"] = required_valid
                    live_config["frame_status"] = "ready"
                else:
                    live_config["_consecutive_valid"] = 0
                    live_config["frame_status"] = "warming_up"
                write_live_status(live_config)

            summary, phase_log, transitions = rollout_expert_episode(
                env,
                obs_keys=DEFAULT_OBS_KEYS,
                record_trajectory=True,
                record_raw_obs=need_image_data,
                episode_index=episode,
                seed=episode_seed,
                live_config=live_config,
            )
            rows.append(summary)
            print(episode, summary.get("final_success", False), summary)

            meta_entry = {
                "episode": episode,
                "seed": episode_seed,
                "phase_log": phase_log,
                "summary": summary,
            }
            if summary.get("final_success"):
                trajectories.append(transitions)
                episode_metadata.append(meta_entry)
            else:
                failures.append(meta_entry)

            if live_config is not None:
                live_config["successful_episodes"] = len(trajectories)
                live_config["episodeSuccess"] = bool(summary.get("final_success"))
                if rows:
                    stats_so_far = aggregate_rows(rows)
                    live_config["success_so_far"] = stats_so_far.get("final_success_rate")
                write_live_status(live_config)
    except Exception as exc:
        if live_config is not None:
            _mark_live_failure(live_config, exc)
        raise
    finally:
        env.close()

    # 保存 CSV
    csv_path = Path(args.results_out)
    write_results_csv(csv_path, rows)
    stats = aggregate_rows(rows)
    print(f"saved_csv: {csv_path}")
    print(f"final_success_rate: {stats.get('final_success_rate', 0.0):.4f}")
    print(f"successful_episodes: {len(trajectories)}")

    # 保存失败案例 JSON
    failures_path = Path(args.failures_out)
    failures_path.parent.mkdir(parents=True, exist_ok=True)
    from examples.cable_threading.failure_reason import build_cable_threading_failure_reason

    enriched_failures = []
    for entry in failures:
        summary = entry.get("summary") if isinstance(entry.get("summary"), dict) else {}
        episode = entry.get("episode", summary.get("episode"))
        reason = str(summary.get("failure_reason") or entry.get("failure_reason") or "").strip()
        if not reason:
            reason = build_cable_threading_failure_reason(summary)
            summary["failure_reason"] = reason
        entry["summary"] = summary
        entry["failure_reason"] = reason
        entry["failureReason"] = reason
        entry["writtenToDataset"] = False
        if episode is not None:
            entry["episode"] = episode
        enriched_failures.append(entry)
    failures_path.write_text(json.dumps(enriched_failures, indent=2, default=str))
    print(f"saved_failures: {failures_path}")

    if live_config is not None and live_config.get("video_out"):
        synthesize_live_video(live_config)
        if live_config.get("generate_video_exists"):
            print(f"saved_generate_video: {live_config.get('generate_video')}")

    if live_config is not None and live_config.get("timeline_path"):
        write_live_timeline(live_config)
        print(f"saved_timeline: {live_config.get('timeline_path')}")

    if not trajectories:
        print("WARNING: No successful episodes collected. Skipping dataset save.")
        if live_config is not None:
            live_config["status"] = "failed"
            live_config["error"] = "No successful episodes collected"
            live_config["final_success_rate"] = stats.get("final_success_rate", 0.0)
            live_config["successful_episodes"] = 0
            live_config["savedCsv"] = str(csv_path)
            live_config["savedFailures"] = str(failures_path)
            write_live_status(live_config)
        return

    # 保存 NPZ 数据集
    dataset_metadata = formal_metadata(
        env_name="CableThreading",
        robot=args.robot,
        controller="default",
        seed=args.seed,
        horizon=args.horizon,
        scene_randomization="random",
        policy="robot_endpoint_oracle",
        cable_model=args.cable_model,
        difficulty=args.difficulty,
        grasp_mode=args.grasp_mode,
        attachment_side_channel=args.grasp_mode == "attachment",
        attachment_field="attachment_enabled",
        attachment_policy="recorded_or_controller" if args.grasp_mode == "attachment" else None,
        attachment_input_mode="not_used_by_policy",
        attachment_control_mode="eval_controller",
        include_attachment_obs=False,
    )
    quality = quality_report_from_trajectories(trajectories, metadata=dataset_metadata)
    dataset_metadata["trajectory_quality"] = quality

    npz_path = Path(args.out)
    save_dataset(npz_path, trajectories, obs_keys=DEFAULT_OBS_KEYS, metadata=dataset_metadata, episode_metadata=episode_metadata)
    print(f"saved_dataset: {npz_path}")

    # HDF5 输出（含图像）：直接写入采集轨迹，禁止 replay 重跑 expert
    if args.hdf5_out:
        from robosuite.utils.dlo.hdf5_dataset import (
            HDF5_IMAGE_KEYS,
            HDF5_LOW_DIM_KEYS,
            HDF5_TASK_OBS_KEYS,
            CONTROLLER_TYPE,
            CURRENT_ACTION_MODE,
            build_hdf5_manifest_fields,
            save_dataset_hdf5,
            validate_hdf5_trajectory_actions,
        )

        hdf5_path = Path(args.hdf5_out)
        dataset_metadata["current_action_mode"] = CURRENT_ACTION_MODE
        dataset_metadata["controller_type"] = CONTROLLER_TYPE
        dataset_metadata["taskTemplateId"] = "cable_threading_single_arm"
        dataset_metadata["taskType"] = "cable_threading"
        dataset_metadata["simulatorBackend"] = "mujoco"
        if args.grasp_mode == "attachment":
            dataset_metadata["side_channel_keys"] = ["attachment_enabled"]
        hdf5_save_info = save_dataset_hdf5(
            hdf5_path,
            trajectories,
            image_keys=list(HDF5_IMAGE_KEYS),
            low_dim_keys=list(HDF5_LOW_DIM_KEYS),
            task_obs_keys=list(HDF5_TASK_OBS_KEYS),
            metadata=dataset_metadata,
            episode_metadata=episode_metadata,
        )
        validation = validate_hdf5_trajectory_actions(hdf5_path, trajectories)
        if not validation["ok"]:
            raise RuntimeError(
                "HDF5 action/attachment validation failed: " + "; ".join(validation["issues"])
            )
        print(
            f"[hdf5] validated actions against source trajectories "
            f"(max_diff={validation['max_action_diff']})"
        )
        print(f"saved_hdf5: {hdf5_path}")
    else:
        hdf5_save_info = None

    lerobot_save_info = None
    if lerobot_out:
        from robosuite.utils.dlo.lerobot_platform_export import save_cable_threading_lerobot_dataset

        lerobot_path = Path(lerobot_out)
        lerobot_save_info = save_cable_threading_lerobot_dataset(
            lerobot_path,
            trajectories,
            robot=getattr(args, "lerobot_robot", None) or args.robot,
            task_instruction=getattr(args, "lerobot_task_instruction", ""),
            fps=int(getattr(args, "lerobot_fps", 20)),
            episode_metadata=episode_metadata,
        )
        print(f"saved_lerobot: {lerobot_path}")
        if lerobot_save_info.get("pi0Ready"):
            print("lerobot_pi0_ready: true")
        else:
            print(f"lerobot_pi0_ready: false ({lerobot_save_info.get('pi0ReadyReason')})")

    # manifest.json
    primary_format = "lerobot" if lerobot_out else ("hdf5" if args.hdf5_out else "npz")
    manifest = {
        "dataset": str(npz_path),
        "num_successful": len(trajectories),
        "num_failed": len(failures),
        "obs_keys": list(DEFAULT_OBS_KEYS),
        "action_dim": int(lerobot_save_info.get("action_dim") if lerobot_save_info else 7),
        "control_freq": int(getattr(args, "lerobot_fps", 20) if lerobot_out else 20),
        "horizon": args.horizon,
        "cable_model": args.cable_model,
        "difficulty": args.difficulty,
        "robot": args.robot,
        "seed": args.seed,
        "grasp_mode": args.grasp_mode,
        "attachment_side_channel": args.grasp_mode == "attachment",
        "attachment_field": "attachment_enabled",
        "attachment_policy": "recorded_or_controller" if args.grasp_mode == "attachment" else None,
        "attachmentInputMode": "not_used_by_policy",
        "attachmentControlMode": "eval_controller",
        "includeAttachmentObs": False,
        "taskTemplateId": "cable_threading_single_arm",
        "taskType": "cable_threading",
        "simulatorBackend": "mujoco",
        "side_channel_keys": ["attachment_enabled"] if args.grasp_mode == "attachment" else [],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "primaryFormat": primary_format,
        "datasetFormats": [primary_format],
        "availableFormats": [primary_format],
        "format": primary_format,
        "datasetFormat": primary_format,
    }
    if args.hdf5_out and Path(args.hdf5_out).exists():
        manifest["hdf5"] = str(Path(args.hdf5_out))
        if hdf5_save_info:
            manifest.update(build_hdf5_manifest_fields(hdf5_save_info))
        if lerobot_out:
            manifest["datasetFormats"] = ["lerobot", "hdf5"]
            manifest["availableFormats"] = manifest["datasetFormats"]
    if lerobot_save_info:
        lerobot_path = Path(lerobot_out)
        manifest["lerobot"] = str(lerobot_path)
        manifest["lerobotMetadata"] = {
            "status": "ready",
            "path": str(lerobot_path),
            "metadataPath": str(lerobot_path / "metadata.json"),
            "statsPath": str(lerobot_path / "stats.json"),
            "reportPath": str(lerobot_path / "generation_report.json"),
            "taskInstruction": lerobot_save_info.get("task_instruction"),
            "robot": lerobot_save_info.get("robot"),
            "stateDim": lerobot_save_info.get("state_dim"),
            "actionDim": lerobot_save_info.get("action_dim"),
            "pi0Ready": bool(lerobot_save_info.get("pi0Ready")),
            "pi0ReadyReason": lerobot_save_info.get("pi0ReadyReason") or "",
        }
        manifest["taskDescription"] = lerobot_save_info.get("task_instruction")
        manifest["state_dim"] = lerobot_save_info.get("state_dim")
        manifest["action_dim"] = lerobot_save_info.get("action_dim")
        manifest["controller_type"] = lerobot_save_info.get("controller_type")
        manifest["action_mode"] = lerobot_save_info.get("action_mode")
        manifest["pi0Ready"] = bool(lerobot_save_info.get("pi0Ready"))
        manifest["pi0ReadyReason"] = lerobot_save_info.get("pi0ReadyReason") or ""
    manifest_path = (
        Path(args.manifest_out)
        if getattr(args, "manifest_out", None)
        else npz_path.with_suffix(".manifest.json")
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"saved_manifest: {manifest_path}")

    if live_config is not None:
        live_config["status"] = "completed"
        live_config["final_success_rate"] = stats.get("final_success_rate", 0.0)
        live_config["successful_episodes"] = len(trajectories)
        live_config["savedDataset"] = str(npz_path)
        live_config["savedManifest"] = str(manifest_path)
        live_config["savedCsv"] = str(csv_path)
        live_config["savedFailures"] = str(failures_path)
        if args.hdf5_out and Path(args.hdf5_out).exists():
            live_config["savedHdf5"] = str(Path(args.hdf5_out))
        if lerobot_out and Path(lerobot_out).exists():
            live_config["savedLerobot"] = str(Path(lerobot_out))
        write_live_status(live_config)


def cmd_eval(args):
    """评估策略并输出结构化结果。"""
    from examples.cable_threading.obs_schema import DEFAULT_EVAL_DISPLAY_CAMERA
    from examples.cable_threading.utils import (
        RandomPolicy,
        RobomimicPolicyAdapter,
        aggregate_rows,
        apply_obs_validation_failure,
        make_env,
        policy_eval_camera_kwargs,
        rollout_expert_episode,
        rollout_policy_episode,
        synthesize_live_video,
        clear_live_saved_frames,
        validate_policy_obs_schema,
        warmup_live_env,
        write_live_status,
        write_live_timeline,
        write_results_csv,
    )
    from examples.cable_threading.utils import DEFAULT_OBS_KEYS

    live_frame_dir = getattr(args, "live_frame_dir", None)
    live_enabled = bool(live_frame_dir)
    live_config = None
    if live_enabled:
        live_dir = Path(live_frame_dir)
        live_dir.mkdir(parents=True, exist_ok=True)
        save_frames = bool(getattr(args, "live_save_frames", False)) or bool(
            getattr(args, "episode_video_dir", None)
        )
        video_out = getattr(args, "live_video_out", None)
        status_path = (
            Path(args.live_status_out)
            if getattr(args, "live_status_out", None)
            else live_dir / "status.json"
        )
        record_camera = getattr(args, "live_display_camera", DEFAULT_EVAL_DISPLAY_CAMERA)
        allow_camera_fallback = bool(getattr(args, "allow_camera_fallback", False))
        live_config = {
            "status": "running",
            "jobType": "evaluate",
            "job_id": getattr(args, "job_id", None),
            "frame_dir": str(live_dir),
            "frames_dir": str(live_dir / "frames") if save_frames else None,
            "save_frames": save_frames,
            "video_out": video_out,
            "video_fps": int(getattr(args, "live_video_fps", 20)),
            "eval_video_status": "pending" if video_out else None,
            "eval_video_exists": False,
            "eval_video_size_bytes": 0,
            "saved_frame_count": 0,
            "frame_every": int(getattr(args, "live_frame_every", 5)),
            "camera": getattr(args, "live_camera", "agentview"),
            "display_camera": record_camera,
            "record_camera": record_camera,
            "allow_camera_fallback": allow_camera_fallback,
            "camera_fallback_used": False,
            "frame_source": "sim_render",
            "status_path": str(status_path),
            "episode": 0,
            "episodes": args.episodes,
            "horizon": args.horizon,
            "step": 0,
            "frame_count": 0,
            "phase": "",
            "successful_episodes": 0,
            "success_so_far": None,
            "final_success_rate": None,
            "error": None,
            "timeline_path": (
                str(Path(args.live_timeline_out))
                if getattr(args, "live_timeline_out", None)
                else None
            ),
            "timeline_events": [],
            "timeline_seen": set(),
            "frame_status": "warming_up",
            "skipped_invalid_frame": 0,
            "has_valid_frame": False,
            "live_warmup_steps": 10,
            "live_required_consecutive_valid": 3,
            "live_sim_forward_warmup": 5,
            "live_obs_resized": False,
            **_live_viz_settings(args),
        }
        if getattr(args, "record_step_metrics", True):
            results_parent = Path(args.out).resolve().parent
            live_config["record_step_metrics"] = True
            live_config["step_metrics_output_dir"] = str(results_parent / "step_metrics")
            live_config["record_step_metrics_full_arrays"] = bool(
                getattr(args, "record_step_metrics_full_arrays", False)
            )
            live_config["step_metrics_downsample"] = int(
                getattr(args, "step_metrics_downsample", 1) or 1
            )
        if save_frames:
            (live_dir / "frames").mkdir(parents=True, exist_ok=True)
        print(
            f"record_camera: {record_camera} allow_camera_fallback: {allow_camera_fallback}"
        )
        write_live_status(live_config)

    policy = None
    policy_name = args.policy
    eval_runtime: dict[str, str] = {}
    if args.policy == "robomimic":
        if not args.checkpoint:
            print("ERROR: --checkpoint required for robomimic policy")
            return
        policy = RobomimicPolicyAdapter(args.checkpoint, device=args.device)
    elif args.policy == "diffusion_policy":
        if not args.checkpoint:
            print("ERROR: --checkpoint required for diffusion_policy")
            return
        from examples.cable_threading.dp_lab.policy_runtime import DiffusionPolicyAdapter
        from examples.cable_threading.dp_eval_runtime import resolve_dp_eval_runtime

        policy = DiffusionPolicyAdapter(args.checkpoint, device=args.device)
        eval_runtime = resolve_dp_eval_runtime(
            policy="diffusion_policy",
            checkpoint_path=args.checkpoint,
            eval_executor=getattr(args, "eval_executor", None),
            controller_type=getattr(args, "controller_type", None),
            action_mode=getattr(args, "action_mode", None),
        )
        print(
            "dp_eval_runtime:",
            f"evalExecutor={eval_runtime['evalExecutor']}",
            f"controllerType={eval_runtime['controllerType']}",
            f"actionMode={eval_runtime['actionMode']}",
        )
    elif args.policy == "act":
        if not args.checkpoint:
            print("ERROR: --checkpoint required for act policy")
            return
        from examples.cable_threading.act_lab.policy_runtime import ACTPolicyAdapter
        from examples.cable_threading.act_eval_runtime import resolve_act_eval_runtime

        policy = ACTPolicyAdapter(args.checkpoint, device=args.device)
        eval_runtime = resolve_act_eval_runtime(
            policy="act",
            checkpoint_path=args.checkpoint,
            eval_executor=getattr(args, "eval_executor", None),
            controller_type=getattr(args, "controller_type", None),
            action_mode=getattr(args, "action_mode", None),
        )
        print(
            "act_eval_runtime:",
            f"evalExecutor={eval_runtime['evalExecutor']}",
            f"controllerType={eval_runtime['controllerType']}",
            f"actionMode={eval_runtime['actionMode']}",
        )
    elif args.policy == "pi0":
        if not args.checkpoint:
            print("ERROR: --checkpoint required for pi0 policy")
            return
        from examples.cable_threading.pi0_eval_runtime import resolve_pi0_eval_runtime
        from examples.cable_threading.pi0_lab.policy_runtime import Pi0PolicyAdapter

        eval_runtime = resolve_pi0_eval_runtime(
            policy="pi0",
            checkpoint_path=args.checkpoint,
            train_config_path=getattr(args, "train_config", None),
            eval_executor=getattr(args, "eval_executor", None),
            controller_type=getattr(args, "controller_type", None),
            action_mode=getattr(args, "action_mode", None),
            robot=getattr(args, "robot", None),
            task_instruction=getattr(args, "task_instruction", None),
        )
        policy = Pi0PolicyAdapter(
            args.checkpoint,
            device=args.device,
            train_config_path=getattr(args, "train_config", None),
            task_instruction=eval_runtime.get("taskInstruction") or getattr(args, "task_instruction", None),
        )
        print(
            "pi0_eval_runtime:",
            f"evalExecutor={eval_runtime['evalExecutor']}",
            f"controllerType={eval_runtime['controllerType']}",
            f"actionMode={eval_runtime['actionMode']}",
            f"robot={eval_runtime.get('robot')}",
            f"stateDim={eval_runtime.get('stateDim')}",
            f"actionDim={eval_runtime.get('actionDim')}",
        )

    if args.policy in {"robomimic", "diffusion_policy", "act", "pi0"}:
        env_camera_kwargs = policy_eval_camera_kwargs()
        use_camera_obs = True
        has_offscreen = True
    elif live_enabled:
        env_camera_kwargs = _live_env_kwargs(live_enabled, live_config)
        use_camera_obs = live_enabled
        has_offscreen = live_enabled
    else:
        env_camera_kwargs = {}
        use_camera_obs = False
        has_offscreen = False

    use_joint_executor = eval_runtime.get("evalExecutor") == "joint_position"
    eval_robot = str(eval_runtime.get("robot") or args.robot)
    if args.policy == "pi0" and use_joint_executor:
        eval_robot = "Panda"
    if use_joint_executor:
        from examples.cable_threading.joint_controller import make_joint_position_env

        env = make_joint_position_env(
            robot=eval_robot,
            cable_model=args.cable_model,
            grasp_mode=args.grasp_mode,
            difficulty=args.difficulty,
            horizon=args.horizon,
            seed=args.seed,
            use_camera_obs=use_camera_obs,
            has_offscreen_renderer=has_offscreen,
            camera_names=env_camera_kwargs.get("camera_names"),
        )
    else:
        env = make_env(
            robot=args.robot,
            cable_model=args.cable_model,
            grasp_mode=args.grasp_mode,
            difficulty=args.difficulty,
            horizon=args.horizon,
            seed=args.seed,
            has_offscreen_renderer=has_offscreen,
            use_camera_obs=use_camera_obs,
            **env_camera_kwargs,
        )

    if live_config is not None and eval_runtime:
        live_config.update(eval_runtime)

    if args.policy == "random":
        policy = RandomPolicy(env, seed=args.seed)

    if args.policy == "robomimic":
        probe_obs = env.reset()
        if live_config is not None:
            warmup_live_env(env, live_config)
        validation = validate_policy_obs_schema(policy, probe_obs)
        if not validation["valid"]:
            if live_config is not None:
                apply_obs_validation_failure(live_config, validation)
            env.close()
            print(f"OBS_VALIDATION_FAILED: {validation.get('errorMessage')}")
            return
        if hasattr(policy, "reset"):
            policy.reset()
    elif args.policy == "diffusion_policy":
        from examples.cable_threading.obs_schema import validate_diffusion_policy_obs_schema

        probe_obs = env.reset()
        if live_config is not None:
            warmup_live_env(env, live_config)
        validation = validate_diffusion_policy_obs_schema(policy, probe_obs)
        if not validation["valid"]:
            if live_config is not None:
                apply_obs_validation_failure(live_config, validation)
            env.close()
            print(f"OBS_VALIDATION_FAILED: {validation.get('errorMessage')}")
            return
        if hasattr(policy, "reset"):
            policy.reset()
    elif args.policy == "act":
        from examples.cable_threading.obs_schema import validate_act_obs_schema

        probe_obs = env.reset()
        if live_config is not None:
            warmup_live_env(env, live_config)
        validation = validate_act_obs_schema(policy, probe_obs)
        if not validation["valid"]:
            if live_config is not None:
                apply_obs_validation_failure(live_config, validation)
            env.close()
            print(f"OBS_VALIDATION_FAILED: {validation.get('errorMessage')}")
            return
        if hasattr(policy, "reset"):
            policy.reset()
    elif args.policy == "pi0":
        from examples.cable_threading.obs_schema import validate_pi0_obs_schema

        probe_obs = env.reset()
        if live_config is not None:
            warmup_live_env(env, live_config)
        validation = validate_pi0_obs_schema(policy, probe_obs)
        if not validation["valid"]:
            if live_config is not None:
                apply_obs_validation_failure(live_config, validation)
            env.close()
            print(f"OBS_VALIDATION_FAILED: {validation.get('errorMessage')}")
            return
        if hasattr(policy, "reset"):
            policy.reset()

    rows = []
    episode_video_dir = getattr(args, "episode_video_dir", None)
    per_episode_videos: list[str] = []
    if live_config is not None and episode_video_dir:
        from examples.cable_threading.utils import _warmup_live_after_reset

        prime_obs = env.reset()
        _warmup_live_after_reset(env, prime_obs, live_config)
        clear_live_saved_frames(live_config)
    try:
        for episode in range(args.episodes):
            episode_seed = args.seed + episode
            if live_config is not None:
                live_config["episode"] = episode
                live_config["step"] = 0
                live_config["phase"] = ""
                live_config.pop("_last_display_frame", None)
                required_valid = int(live_config.get("live_required_consecutive_valid", 3))
                if live_config.get("has_valid_frame") and episode > 0:
                    live_config["_consecutive_valid"] = required_valid
                    live_config["frame_status"] = "ready"
                else:
                    live_config["_consecutive_valid"] = 0
                    if not live_config.get("has_valid_frame"):
                        live_config["frame_status"] = "warming_up"
                if episode_video_dir:
                    ep_video_path = Path(episode_video_dir) / f"episode_{episode + 1:03d}.mp4"
                    live_config["video_out"] = str(ep_video_path)
                    clear_live_saved_frames(live_config)
                write_live_status(live_config)

            if args.policy == "scripted":
                summary, _, _ = rollout_expert_episode(
                    env,
                    episode_index=episode,
                    seed=episode_seed,
                    live_config=live_config,
                )
            else:
                summary, _ = rollout_policy_episode(
                    env,
                    policy,
                    obs_keys=DEFAULT_OBS_KEYS,
                    episode_index=episode,
                    seed=episode_seed,
                    policy_name=policy_name,
                    live_config=live_config,
                    attachment_mode=getattr(args, "attachment_mode", "policy"),
                )
            rows.append(summary)
            print(episode, summary.get("final_success", False))

            if live_config is not None and episode_video_dir and live_config.get("video_out"):
                synthesize_live_video(live_config)
                video_path = live_config.get("eval_video") or live_config.get("videoPath")
                if video_path:
                    video_file = Path(video_path)
                    summary["videoPath"] = str(video_file)
                    summary["videoUri"] = f"videos/{video_file.name}"
                    summary["episodeIndex"] = episode + 1
                    summary["recordCamera"] = live_config.get("record_camera")
                    summary["cameraFallbackUsed"] = bool(live_config.get("camera_fallback_used"))
                    if live_config.get("actual_record_camera"):
                        summary["actualRecordCamera"] = live_config.get("actual_record_camera")
                    per_episode_videos.append(str(video_path))
                else:
                    print(
                        f"WARNING: episode {episode} video not generated "
                        f"(status={live_config.get('eval_video_status')})"
                    )
                clear_live_saved_frames(live_config)

            if live_config is not None:
                if summary.get("final_success"):
                    live_config["successful_episodes"] = int(
                        live_config.get("successful_episodes", 0)
                    ) + 1
                live_config["episodeSuccess"] = bool(summary.get("final_success"))
                if rows:
                    stats_so_far = aggregate_rows(rows)
                    live_config["success_so_far"] = stats_so_far.get("final_success_rate")
                write_live_status(live_config)
    except Exception as exc:
        if live_config is not None:
            _mark_live_failure(live_config, exc)
        raise
    finally:
        env.close()

    # CSV
    csv_path = Path(args.out)
    write_results_csv(csv_path, rows)
    stats = aggregate_rows(rows)
    print(f"saved_csv: {csv_path}")

    # 结构化 JSON
    results_path = csv_path.with_suffix(".results.json")
    results_payload = {
        "success_rate": stats.get("final_success_rate", 0.0),
        "ever_success_rate": stats.get("ever_success_rate", 0.0),
        "num_episodes": len(rows),
        "aggregate": stats,
        "episodes": rows,
    }
    results_path.write_text(json.dumps(results_payload, indent=2, default=str))
    print(f"saved_results: {results_path}")

    results_dir = csv_path.parent
    aggregate_path = results_dir / "aggregate_result.json"
    per_episode_path = results_dir / "per_episode_results.json"
    aggregate_doc = {
        "task_name": "线缆穿杆",
        "requested_episodes": int(args.episodes),
        "total_episodes": len(rows),
        "completed_episodes": len(rows),
        "success_episodes": sum(1 for row in rows if row.get("final_success")),
        "final_success_rate": stats.get("final_success_rate", 0.0),
        "ever_success_rate": stats.get("ever_success_rate", 0.0),
        "success_rate": stats.get("final_success_rate", 0.0),
        "mean_thread_completion_max": stats.get("mean_thread_completion_max"),
        "mean_endpoint_goal_error_final": stats.get("mean_endpoint_goal_error_final"),
        "mean_straightness_error_final": stats.get("mean_straightness_error_final"),
        "mean_anchor_error_final": stats.get("mean_anchor_error_final"),
        "mean_tabletop_spread_final": stats.get("mean_tabletop_spread_final"),
        "failure_count": sum(1 for row in rows if not row.get("final_success")),
        "failed_episodes": sum(1 for row in rows if not row.get("final_success")),
        "recorded_video_count": len(per_episode_videos),
        "recordCamera": live_config.get("record_camera") if live_config else getattr(args, "live_display_camera", DEFAULT_EVAL_DISPLAY_CAMERA),
        "cameraFallbackUsed": bool(live_config.get("camera_fallback_used")) if live_config else False,
        "attachment_mode": getattr(args, "attachment_mode", "policy"),
        **stats,
    }
    if eval_runtime:
        aggregate_doc.update(eval_runtime)
        aggregate_doc["sideChannelMode"] = eval_runtime.get("sideChannelMode")
    if args.policy == "pi0" and eval_runtime.get("evalExecutor") == "joint_position":
        aggregate_doc.update(
            {
                "policyType": "pi0",
                "modelType": "pi0",
                "evalExecutor": "joint_position",
                "robot": eval_runtime.get("robot") or "Panda",
                "controllerType": "JOINT_POSITION",
                "stateDim": eval_runtime.get("stateDim") or 9,
                "actionDim": eval_runtime.get("actionDim") or 8,
                "taskInstruction": eval_runtime.get("taskInstruction"),
                "episodes": int(args.episodes),
                "horizon": int(args.horizon),
                "rollout_ok": True,
            }
        )
    if episode_video_dir and per_episode_videos:
        import shutil

        first_video = None
        for item in per_episode_videos:
            path = Path(item)
            if path.is_file():
                first_video = path
                break
        if first_video is None:
            for candidate in sorted(Path(episode_video_dir).glob("episode_*.mp4")):
                if candidate.is_file() and not candidate.name.endswith(".browser.mp4"):
                    first_video = candidate
                    break
        compat_video = Path(episode_video_dir) / "eval.mp4"
        if first_video and first_video.is_file():
            shutil.copy2(first_video, compat_video)
            aggregate_doc["videoPath"] = str(compat_video)
            browser_video = live_config.get("browserVideoPath") if live_config else None
            if browser_video:
                aggregate_doc["browserVideoPath"] = browser_video
            aggregate_doc["videoStatus"] = "available"
        if live_config is not None:
            live_config["status"] = "completed"
            live_config["completedEpisodes"] = len(rows)
            live_config["requestedEpisodes"] = int(args.episodes)
            live_config["successfulEpisodes"] = aggregate_doc["success_episodes"]
            live_config["failedEpisodes"] = aggregate_doc["failed_episodes"]
            live_config["recordedVideoCount"] = len(per_episode_videos)
            live_config["progressPercent"] = 100
            write_live_status(live_config)
    elif live_config is not None and live_config.get("video_out"):
        synthesize_live_video(live_config)
        if live_config.get("eval_video_exists"):
            print(f"saved_eval_video: {live_config.get('eval_video')}")
            browser_video = live_config.get("browserVideoPath") or live_config.get("eval_browser_video")
            if browser_video:
                print(f"saved_eval_browser_video: {browser_video}")
            aggregate_doc["videoPath"] = live_config.get("videoPath") or live_config.get("eval_video")
            aggregate_doc["browserVideoPath"] = browser_video
            aggregate_doc["videoStatus"] = live_config.get("videoStatus") or "available"
            aggregate_doc["videoResolution"] = live_config.get("videoResolution")
            aggregate_doc["videoFps"] = live_config.get("videoFps")
            aggregate_doc["displayCamera"] = live_config.get("display_camera")
    try:
        from step_metrics import attach_run_metrics_to_aggregate

        aggregate_doc = attach_run_metrics_to_aggregate(aggregate_doc, results_dir)
    except Exception as exc:
        print(f"WARNING: runMetrics aggregation failed: {exc}")
    aggregate_path.write_text(json.dumps(aggregate_doc, indent=2, default=str), encoding="utf-8")
    per_episode_path.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    print(f"saved_aggregate: {aggregate_path}")
    print(f"saved_per_episode: {per_episode_path}")

    # 失败案例
    failures = [r for r in rows if not r.get("final_success")]
    failures_path = csv_path.with_suffix(".failures.json")
    failures_path.write_text(json.dumps(failures, indent=2, default=str))
    print(f"saved_failures: {failures_path}")

    if live_config is not None:
        live_config["status"] = "completed"
        live_config["final_success_rate"] = stats.get("final_success_rate", 0.0)
        live_config["savedCsv"] = str(csv_path)
        live_config["savedResults"] = str(results_path)
        live_config["savedFailures"] = str(failures_path)
        if live_config.get("timeline_path"):
            write_live_timeline(live_config)
            print(f"saved_timeline: {live_config.get('timeline_path')}")
        write_live_status(live_config)

    print(f"success_rate: {stats.get('final_success_rate', 0.0):.4f}")


def cmd_video(args):
    """录制专家策略视频。"""
    from examples.cable_threading.utils import make_env, rollout_expert_episode

    try:
        import imageio.v2 as iio
    except ImportError:
        print("ERROR: imageio not installed. pip install imageio[ffmpeg]")
        return

    env = make_env(
        robot=args.robot,
        cable_model=args.cable_model,
        difficulty=args.difficulty,
        horizon=args.horizon,
        seed=args.seed,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["agentview"],
    )

    video_path = Path(args.video_out)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    writer = iio.get_writer(str(video_path), fps=20, format="FFMPEG", codec="libx264", pixelformat="yuv420p", quality=7)

    try:
        for episode in range(args.episodes):
            obs = env.reset()
            from examples.cable_threading.utils import EXPERT_PHASES, expert_phase_action
            for phase_cfg in EXPERT_PHASES:
                for local_step in range(phase_cfg["max_steps"]):
                    if local_step == phase_cfg.get("attach_on_step", -1):
                        env.set_attachment_enabled(True)
                    if local_step == phase_cfg.get("detach_on_step", -1):
                        env.set_attachment_enabled(False)
                    action, _ = expert_phase_action(env, phase_cfg["name"])
                    obs, reward, done, info = env.step(action)
                    if "agentview_image" in obs:
                        writer.append_data(obs["agentview_image"][::-1])
                    if info.get("final_success") or done:
                        break
                if info.get("final_success") or done:
                    break
            print(f"episode {episode}: success={info.get('final_success', False)}")
    finally:
        writer.close()
        env.close()

    print(f"saved_video: {video_path}")


def main():
    from examples.cable_threading.obs_schema import DEFAULT_EVAL_DISPLAY_CAMERA

    parser = argparse.ArgumentParser(description="CableThreadingMVP — 统一入口")
    subparsers = parser.add_subparsers(dest="command")

    # 共享参数
    def add_common_args(p):
        p.add_argument("--robot", default="UR5e", help="机器人类型 (UR5e/Panda)")
        p.add_argument("--cable-model", default="rmb", help="线缆模型 (rmb/flex/composite_cable/composite_soft)")
        p.add_argument("--grasp-mode", default="attachment", help="抓取模式 (attachment/physical)")
        p.add_argument("--difficulty", default="easy", help="难度 (easy/medium/hard)")
        p.add_argument("--horizon", type=int, default=600, help="最大步数（默认 600，Panda+flex 需要较长 horizon）")
        p.add_argument("--seed", type=int, default=0, help="随机种子")

    # env-test
    p_env = subparsers.add_parser("env-test", help="环境冒烟测试")
    add_common_args(p_env)

    # expert
    p_expert = subparsers.add_parser("expert", help="采集专家数据集")
    add_common_args(p_expert)
    p_expert.add_argument("--episodes", type=int, default=10, help="采集 episode 数")
    p_expert.add_argument("--out", default="datasets/cable_threading_expert.npz", help="NPZ 输出路径")
    p_expert.add_argument("--hdf5-out", default=None, help="HDF5 输出路径（含图像）")
    p_expert.add_argument("--lerobot-out", default=None, help="LeRobot v3.0 数据集输出目录")
    p_expert.add_argument(
        "--lerobot-task-instruction",
        default="thread the cable through the pole",
        help="LeRobot 任务自然语言描述",
    )
    p_expert.add_argument("--lerobot-robot", default="Panda", help="LeRobot metadata 中的 robot 字段")
    p_expert.add_argument("--lerobot-fps", type=int, default=20, help="LeRobot 控制频率")
    p_expert.add_argument(
        "--manifest-out",
        default=None,
        help="dataset.manifest.json 输出路径（默认与 NPZ 同目录）",
    )
    p_expert.add_argument("--results-out", default="results/expert_collect.csv", help="CSV 结果路径")
    p_expert.add_argument("--failures-out", default="results/expert_failures.json", help="失败案例路径")
    p_expert.add_argument(
        "--live-frame-dir",
        default=None,
        help="实时画面输出目录（expert rollout 同一步循环内渲染）",
    )
    p_expert.add_argument(
        "--live-frame-every",
        type=int,
        default=5,
        help="每隔多少 step 保存一帧 latest.jpg",
    )
    p_expert.add_argument(
        "--live-status-out",
        default=None,
        help="实时状态 JSON 输出路径（默认 live-frame-dir/status.json）",
    )
    p_expert.add_argument(
        "--live-camera",
        default="agentview",
        help="实时渲染相机（默认 agentview）",
    )
    p_expert.add_argument(
        "--live-save-frames",
        action="store_true",
        help="在 expert rollout 中保存完整帧序列到 live/frames/",
    )
    p_expert.add_argument(
        "--live-video-out",
        default=None,
        help="任务完成后将帧序列合成为 MP4 的输出路径",
    )
    p_expert.add_argument(
        "--live-video-fps",
        type=int,
        default=20,
        help="合成视频帧率（默认 20）",
    )
    p_expert.add_argument(
        "--live-timeline-out",
        default=None,
        help="保存 generate 阶段同步事件 JSON 路径",
    )
    _add_live_viz_args(p_expert)

    # eval
    p_eval = subparsers.add_parser("eval", help="评估策略")
    add_common_args(p_eval)
    p_eval.add_argument("--episodes", type=int, default=20, help="评估 episode 数")
    p_eval.add_argument("--policy", choices=["scripted", "random", "robomimic", "diffusion_policy", "act", "pi0"], default="scripted")
    p_eval.add_argument("--checkpoint", default=None, help="robomimic / diffusion_policy / act / pi0 checkpoint 路径")
    p_eval.add_argument(
        "--train-config",
        default=None,
        help="pi0 / ACT 评测 train_config.json 路径（pi0 joint-space 必需）",
    )
    p_eval.add_argument(
        "--task-instruction",
        default=None,
        help="pi0 自然语言任务描述（joint-space 必需）",
    )
    p_eval.add_argument("--device", default="cpu", help="推理设备")
    p_eval.add_argument("--out", default="results/eval.csv", help="CSV 输出路径")
    p_eval.add_argument(
        "--live-frame-dir",
        default=None,
        help="实时画面输出目录（eval rollout 同一步循环内渲染）",
    )
    p_eval.add_argument(
        "--live-frame-every",
        type=int,
        default=5,
        help="每隔多少 step 保存一帧 latest.jpg",
    )
    p_eval.add_argument(
        "--live-status-out",
        default=None,
        help="实时状态 JSON 输出路径（默认 live-frame-dir/status.json）",
    )
    p_eval.add_argument(
        "--live-camera",
        default="agentview",
        help="（已弃用于 eval 展示）策略 obs 相机名；展示视频请用 --live-display-camera",
    )
    p_eval.add_argument(
        "--live-display-camera",
        default=DEFAULT_EVAL_DISPLAY_CAMERA,
        help="评测展示相机（sim.render 原生分辨率，默认 agentview，与数据生成展示视角一致）",
    )
    p_eval.add_argument(
        "--live-save-frames",
        action="store_true",
        help="在 eval rollout 中保存完整帧序列到 live/frames/",
    )
    p_eval.add_argument(
        "--live-video-out",
        default=None,
        help="评测完成后将帧序列合成为 MP4 的输出路径（单文件模式）",
    )
    p_eval.add_argument(
        "--episode-video-dir",
        default=None,
        help="按 episode 输出独立 MP4 的目录（episode_001.mp4 ...）",
    )
    p_eval.add_argument(
        "--allow-camera-fallback",
        action="store_true",
        help="允许在指定 record camera 无效时 fallback 到 obs 相机（默认关闭）",
    )
    p_eval.add_argument(
        "--attachment-mode",
        choices=["policy", "recorded", "none"],
        default="policy",
        help="policy=自动推断 attach/detach；recorded=调试回放 HDF5 侧信道；none=不控制 attachment",
    )
    p_eval.add_argument(
        "--eval-executor",
        default=None,
        help="DP 评测执行器：joint_position 或 osc_pose（默认从 checkpoint train_config 推断）",
    )
    p_eval.add_argument(
        "--controller-type",
        default=None,
        help="覆盖评测控制器类型（JOINT_POSITION / OSC_POSE）",
    )
    p_eval.add_argument(
        "--action-mode",
        default=None,
        help="覆盖评测 action mode（joint_delta / osc_pose_delta_eef）",
    )
    p_eval.add_argument(
        "--live-video-fps",
        type=int,
        default=20,
        help="合成视频帧率（默认 20）",
    )
    p_eval.add_argument(
        "--live-timeline-out",
        default=None,
        help="保存 eval 阶段同步事件 JSON 路径",
    )
    p_eval.add_argument(
        "--job-id",
        default=None,
        help="平台 job id（用于 step_metrics 落盘）",
    )
    p_eval.add_argument(
        "--record-step-metrics",
        dest="record_step_metrics",
        action="store_true",
        default=True,
        help="记录 step 级运行指标 summary（默认开启）",
    )
    p_eval.add_argument(
        "--no-record-step-metrics",
        dest="record_step_metrics",
        action="store_false",
        help="关闭 step 级运行指标记录",
    )
    p_eval.add_argument(
        "--record-step-metrics-full-arrays",
        action="store_true",
        default=False,
        help="额外落盘完整 action 数组（npz）",
    )
    p_eval.add_argument(
        "--step-metrics-downsample",
        type=int,
        default=1,
        help="step metrics 降采样间隔（默认每步记录）",
    )
    _add_live_viz_args(p_eval)

    # video
    p_video = subparsers.add_parser("video", help="录制视频")
    add_common_args(p_video)
    p_video.add_argument("--episodes", type=int, default=1, help="录制 episode 数")
    p_video.add_argument("--video-out", default="results/expert_video.mp4", help="视频输出路径")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    {"env-test": cmd_env_test, "expert": cmd_expert, "eval": cmd_eval, "video": cmd_video}[args.command](args)


if __name__ == "__main__":
    main()
