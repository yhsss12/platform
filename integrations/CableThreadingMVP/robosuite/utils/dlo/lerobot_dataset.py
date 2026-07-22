"""
lerobot_dataset.py -- LeRobot v3.0 数据集保存工具

提供将专家轨迹保存为 LeRobot v3.0 格式（Parquet + MP4）的功能，
可直接用于 LeRobot 的 LeRobotDataset 训练管线。

目录结构：
  dataset_dir/
    meta/
      info.json                    # schema, features, fps, path templates
      stats.json                   # per-feature min/max/mean/std/count
      episodes/chunk-000/file-000.parquet  # episode metadata
    data/chunk-000/file-000.parquet        # all frames concatenated
    videos/{camera_key}/chunk-000/file-000.mp4  # per-camera video
"""

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


# 默认低维状态键（拼接为 observation.state）
LEROBOT_STATE_KEYS = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")

# 默认图像键：{lerobot_key: raw_obs_key}
LEROBOT_IMAGE_KEYS = {
    "observation.images.agentview": "agentview_image",
    "observation.images.eye_in_hand": "robot0_eye_in_hand_image",
}


def _extract_state(raw_obs, state_keys):
    """从 raw_obs 中提取并拼接状态向量。"""
    parts = []
    for key in state_keys:
        val = raw_obs.get(key)
        if val is not None:
            parts.append(np.asarray(val, dtype=np.float32).ravel())
    if not parts:
        return None
    return np.concatenate(parts)


def _compute_stats(values):
    """计算一维数值数组的统计量。"""
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": [float(np.nanmin(arr))],
        "max": [float(np.nanmax(arr))],
        "mean": [float(np.nanmean(arr))],
        "std": [float(np.nanstd(arr))],
        "count": [int(arr.shape[0])],
    }


def _compute_stats_multidim(values):
    """计算多维数值数组的统计量（逐列）。"""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return {
        "min": np.nanmin(arr, axis=0).tolist(),
        "max": np.nanmax(arr, axis=0).tolist(),
        "mean": np.nanmean(arr, axis=0).tolist(),
        "std": np.nanstd(arr, axis=0).tolist(),
        "count": [int(arr.shape[0])],
    }


def save_dataset_lerobot(
    path,
    trajectories,
    *,
    state_keys=LEROBOT_STATE_KEYS,
    image_keys=LEROBOT_IMAGE_KEYS,
    fps=20,
    task_description="",
    success_flags=None,
    chunks_size=1000,
):
    """将专家轨迹保存为 LeRobot v3.0 格式数据集。

    Args:
        path: 输出目录路径
        trajectories: list of list of transition dicts (每个 dict 需含 raw_obs)
        state_keys: 拼接为 observation.state 的 raw_obs 键名
        image_keys: {lerobot_key: raw_obs_key} 映射
        fps: 控制频率
        task_description: 任务自然语言描述
        success_flags: list[bool], 每个 episode 是否成功
        chunks_size: 每个 chunk 的最大 episode 数
    """
    path = Path(path).expanduser()
    num_episodes = len(trajectories)
    if num_episodes == 0:
        print("warning: no trajectories to save")
        return

    # 推断维度
    first_obs = trajectories[0][0].get("raw_obs", {})
    state_dim = len(_extract_state(first_obs, state_keys))
    action_dim = int(np.asarray(trajectories[0][0]["action"]).ravel().shape[0])
    total_frames = sum(len(traj) for traj in trajectories)

    # --- 目录结构 ---
    data_dir = path / "data" / "chunk-000"
    meta_dir = path / "meta"
    episodes_dir = meta_dir / "episodes" / "chunk-000"
    video_dirs = {}
    for lerobot_key in image_keys:
        vdir = path / "videos" / lerobot_key / "chunk-000"
        vdir.mkdir(parents=True, exist_ok=True)
        video_dirs[lerobot_key] = vdir
    data_dir.mkdir(parents=True, exist_ok=True)
    episodes_dir.mkdir(parents=True, exist_ok=True)

    # --- 遍历轨迹，收集数据 ---
    all_state = []
    all_action = []
    all_episode_idx = []
    all_frame_idx = []
    all_timestamp = []
    all_reward = []
    all_done = []
    all_success = []
    all_index = []
    all_task_index = []

    episode_records = []  # for episodes parquet
    video_frames = {k: [] for k in image_keys}  # {lerobot_key: [frames]}
    ep_stats_state = []
    ep_stats_action = []

    global_idx = 0
    for ep_idx, traj in enumerate(trajectories):
        ep_len = len(traj)
        ep_state = []
        ep_action = []
        ep_success = bool(success_flags[ep_idx]) if success_flags else True

        for frame_idx, step in enumerate(traj):
            raw_obs = step.get("raw_obs", {})
            state = _extract_state(raw_obs, state_keys)
            action = np.asarray(step["action"], dtype=np.float32).ravel()

            all_state.append(state.tolist())
            all_action.append(action.tolist())
            all_episode_idx.append(ep_idx)
            all_frame_idx.append(frame_idx)
            all_timestamp.append(frame_idx / fps)
            all_reward.append(float(step.get("reward", 0.0)))
            all_done.append(bool(step.get("done", False)))
            all_success.append(ep_success)
            all_index.append(global_idx)
            all_task_index.append(0)

            ep_state.append(state)
            ep_action.append(action)

            # 图像
            for lerobot_key, raw_key in image_keys.items():
                img = raw_obs.get(raw_key)
                if img is not None:
                    video_frames[lerobot_key].append(np.asarray(img, dtype=np.uint8))

            global_idx += 1

        # per-episode stats
        ep_state_arr = np.stack(ep_state)
        ep_action_arr = np.stack(ep_action)
        ep_stats_state.append(ep_state_arr)
        ep_stats_action.append(ep_action_arr)

        episode_records.append({
            "episode_index": ep_idx,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": global_idx - ep_len,
            "dataset_to_index": global_idx,
            "length": ep_len,
            "tasks": [task_description],
        })

    # --- 写 data/ parquet ---
    state_type = pa.list_(pa.float32(), list_size=state_dim)
    action_type = pa.list_(pa.float32(), list_size=action_dim)

    data_table = pa.table({
        "observation.state": pa.array(all_state, type=state_type),
        "action": pa.array(all_action, type=action_type),
        "episode_index": pa.array(all_episode_idx, type=pa.int64()),
        "frame_index": pa.array(all_frame_idx, type=pa.int64()),
        "timestamp": pa.array(all_timestamp, type=pa.float32()),
        "next.reward": pa.array(all_reward, type=pa.float32()),
        "next.done": pa.array(all_done, type=pa.bool_()),
        "next.success": pa.array(all_success, type=pa.bool_()),
        "index": pa.array(all_index, type=pa.int64()),
        "task_index": pa.array(all_task_index, type=pa.int64()),
    })
    data_path = data_dir / "file-000.parquet"
    pq.write_table(data_table, data_path)

    # --- 写视频 ---
    import imageio.v2 as iio
    for lerobot_key, frames in video_frames.items():
        if not frames:
            continue
        video_path = video_dirs[lerobot_key] / "file-000.mp4"
        writer = iio.get_writer(
            str(video_path), fps=fps, codec="libx264", quality=8, pixelformat="yuv420p",
            macro_block_size=16,
        )
        for frame in frames:
            writer.append_data(frame[::-1])  # OpenGL bottom-to-top → video top-to-bottom
        writer.close()

    # --- 写 meta/episodes/ parquet ---
    episodes_table = pa.table({
        "episode_index": pa.array([r["episode_index"] for r in episode_records], type=pa.int64()),
        "data/chunk_index": pa.array([r["data/chunk_index"] for r in episode_records], type=pa.int64()),
        "data/file_index": pa.array([r["data/file_index"] for r in episode_records], type=pa.int64()),
        "dataset_from_index": pa.array([r["dataset_from_index"] for r in episode_records], type=pa.int64()),
        "dataset_to_index": pa.array([r["dataset_to_index"] for r in episode_records], type=pa.int64()),
        "tasks": pa.array([r["tasks"] for r in episode_records]),
        "length": pa.array([r["length"] for r in episode_records], type=pa.int64()),
    })
    pq.write_table(episodes_table, episodes_dir / "file-000.parquet")

    # --- 写 meta/stats.json ---
    all_state_arr = np.array(all_state, dtype=np.float64)
    all_action_arr = np.array(all_action, dtype=np.float64)
    stats = {
        "observation.state": _compute_stats_multidim(all_state_arr),
        "action": _compute_stats_multidim(all_action_arr),
        "episode_index": _compute_stats(all_episode_idx),
        "frame_index": _compute_stats(all_frame_idx),
        "timestamp": _compute_stats(all_timestamp),
        "next.reward": _compute_stats(all_reward),
        "index": _compute_stats(all_index),
        "task_index": _compute_stats(all_task_index),
    }
    (meta_dir / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))

    # --- 写 meta/info.json ---
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": [state_dim],
            "names": None,
            "fps": float(fps),
        },
        "action": {
            "dtype": "float32",
            "shape": [action_dim],
            "names": None,
            "fps": float(fps),
        },
        "episode_index": {"dtype": "int64", "shape": [1], "names": None, "fps": float(fps)},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None, "fps": float(fps)},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None, "fps": float(fps)},
        "next.reward": {"dtype": "float32", "shape": [1], "names": None, "fps": float(fps)},
        "next.done": {"dtype": "bool", "shape": [1], "names": None, "fps": float(fps)},
        "next.success": {"dtype": "bool", "shape": [1], "names": None, "fps": float(fps)},
        "index": {"dtype": "int64", "shape": [1], "names": None, "fps": float(fps)},
        "task_index": {"dtype": "int64", "shape": [1], "names": None, "fps": float(fps)},
    }
    # 图像 feature
    for lerobot_key in image_keys:
        sample_frame = video_frames[lerobot_key]
        if sample_frame:
            h, w = sample_frame[0].shape[:2]
        else:
            h, w = 720, 1280
        features[lerobot_key] = {
            "dtype": "video",
            "shape": [h, w, 3],
            "names": ["height", "width", "channel"],
            "video_info": {
                "video.fps": float(fps),
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }

    info = {
        "codebase_version": "v3.0",
        "robot_type": "unknown",
        "total_episodes": num_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "chunks_size": chunks_size,
        "fps": float(fps),
        "splits": {"train": f"0:{num_episodes}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": features,
    }
    (meta_dir / "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))

    print(f"saved_lerobot: {path}")
    print(f"episodes: {num_episodes}")
    print(f"total_frames: {total_frames}")
    print(f"state_dim: {state_dim}")
    print(f"action_dim: {action_dim}")
    print(f"image_keys: {list(image_keys.keys())}")
