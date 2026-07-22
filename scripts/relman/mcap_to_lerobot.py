#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
MCAP 直接转 LeRobot 数据集
方案 B：复用 flexible_mcap_to_hdf5 的 MCAP 解析与对齐逻辑，
      直接写入 LeRobotDataset，无中间 HDF5。
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import torch
import yaml

# 添加当前目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flexible_mcap_to_hdf5 import (
    FlexibleMcapReader,
    load_config,
    TopicConfig,
    AlignmentConfig,
)
from flexible_mcap_to_hdf5 import log_info, log_warning

# LeRobot 依赖（可选）
try:
    from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False

import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class LeRobotConfig:
    """LeRobot 转换配置"""
    repo_id: str = "my_org/my_lerobot_dataset"
    robot_type: str = "aloha"
    fps: float = 20.0
    state_sources: List[Dict[str, str]] = field(default_factory=list)
    action_mode: str = "next_state"
    camera_mapping: Dict[str, str] = field(default_factory=dict)
    has_velocity: bool = False
    has_effort: bool = False
    default_instruction: str = "pick and place the object"
    instructions_path: Optional[str] = "instructions.json"
    mode: str = "image"  # "video" | "image"
    use_videos: bool = False
    image_writer_processes: int = 1
    image_writer_threads: int = 1
    video_backend: Optional[str] = None
    batch_encoding_size: int = 1
    vcodec: str = "h264"
    streaming_encoding: bool = False
    encoder_queue_maxsize: int = 10
    encoder_threads: Optional[int] = 1

def _limit_cpu_threads(n: int) -> None:
    n = max(1, int(n or 1))
    os.environ.setdefault("OMP_NUM_THREADS", str(n))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(n))
    os.environ.setdefault("MKL_NUM_THREADS", str(n))
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(n))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(n))
    try:
        torch.set_num_threads(n)
        torch.set_num_interop_threads(1)
    except Exception:
        pass


def load_lerobot_config(config_path: str) -> LeRobotConfig:
    """加载 LeRobot 专用配置"""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    lr = cfg.get("lerobot", cfg)
    raw_vcodec = str(lr.get("vcodec", "h264") or "h264")
    vcodec_aliases = {
        "libx264": "h264",
        "libx265": "hevc",
    }
    vcodec = vcodec_aliases.get(raw_vcodec, raw_vcodec)
    return LeRobotConfig(
        repo_id=lr.get("repo_id", "my_org/my_lerobot_dataset"),
        robot_type=lr.get("robot_type", "aloha"),
        fps=float(lr.get("fps", 20.0)),
        state_sources=lr.get("state_sources", []),
        action_mode=lr.get("action_mode", "next_state"),
        camera_mapping=lr.get("camera_mapping", {}),
        has_velocity=lr.get("has_velocity", False),
        has_effort=lr.get("has_effort", False),
        default_instruction=lr.get("default_instruction", "pick and place the object"),
        instructions_path=lr.get("instructions_path"),
        mode=lr.get("mode", "image"),
        use_videos=lr.get("use_videos", False),
        image_writer_processes=int(lr.get("image_writer_processes", 1)),
        image_writer_threads=int(lr.get("image_writer_threads", 1)),
        video_backend=lr.get("video_backend"),
        batch_encoding_size=int(lr.get("batch_encoding_size", 1)),
        vcodec=vcodec,
        streaming_encoding=bool(lr.get("streaming_encoding", False)),
        encoder_queue_maxsize=int(lr.get("encoder_queue_maxsize", 10)),
        encoder_threads=(None if lr.get("encoder_threads", None) is None else int(lr.get("encoder_threads"))),
    )


def _extract_state_vector(
    aligned_data: Dict[str, List[Dict[str, Any]]],
    frame_idx: int,
    state_sources: List[Dict[str, str]],
) -> np.ndarray:
    """从对齐数据中提取 state 向量（按 state_sources 顺序拼接）

    说明：本文件支持把末端六维力（Sixforce / processed force）作为“额外观测”
    （写入 observation.force.*），因此在拼接 state 向量时会自动跳过力话题。
    """

    def _force_side(topic: str) -> Optional[str]:
        tl = (topic or "").lower()
        if "get_force_data_result" in tl:
            if "/left/" in tl or "left" in tl:
                return "left"
            if "/right/" in tl or "right" in tl:
                return "right"
            return "unknown"
        if "processed" in tl and "force" in tl:
            if "left" in tl:
                return "left"
            if "right" in tl:
                return "right"
        return None

    parts = []
    for src in state_sources:
        topic = src["topic"]
        field = src.get("field", "data")
        if _force_side(topic) is not None:
            continue
        if topic not in aligned_data or frame_idx >= len(aligned_data[topic]):
            raise ValueError(f"缺少话题 {topic} 在帧 {frame_idx} 的数据")
        msg = aligned_data[topic][frame_idx]
        val = msg.get(field)
        if val is None:
            raise ValueError(f"话题 {topic} 帧 {frame_idx} 无字段 {field}")
        arr = np.asarray(val, dtype=np.float32).flatten()
        parts.append(arr)
    if not parts:
        raise ValueError("state_sources 中未检测到可用的关节/夹爪状态字段（可能仅包含力话题）")
    return np.concatenate(parts)


def _get_force_sources(
    state_sources: List[Dict[str, str]],
    topic_configs: Optional[List["TopicConfig"]] = None,
) -> Dict[str, Dict[str, str]]:
    """从 state_sources 中抽取“末端六维力”话题，并按左右归类。

    与 HDF5 路径保持一致：除话题名关键字外，还会参考 topic_configs 中的
    message_type/custom_processor（如 sixforce）作为兜底识别依据。
    """
    force_by_side: Dict[str, Dict[str, str]] = {}
    cfg_by_topic: Dict[str, Any] = {}
    if topic_configs:
        for cfg in topic_configs:
            tname = str(getattr(cfg, "topic_name", "") or "").strip()
            if tname:
                cfg_by_topic[tname] = cfg

    def _guess_side(topic: str, cfg: Optional[Any]) -> Optional[str]:
        tl = (topic or "").lower()
        hdf5_path = str(getattr(cfg, "hdf5_path", "") or "").lower()
        if "get_force_data_result" in tl:
            if "/left/" in tl or "left" in tl:
                return "left"
            if "/right/" in tl or "right" in tl:
                return "right"
            return None
        if "processed" in tl and "force" in tl:
            if "left" in tl:
                return "left"
            if "right" in tl:
                return "right"
        # 兼容 HDF5 约定路径：/observations/force_left_state、force_right_state
        if "force_left" in hdf5_path or "/left/" in hdf5_path:
            return "left"
        if "force_right" in hdf5_path or "/right/" in hdf5_path:
            return "right"
        return None

    def _is_force_source(topic: str, cfg: Optional[Any]) -> bool:
        tl = (topic or "").lower()
        if "get_force_data_result" in tl:
            return True
        if "processed" in tl and "force" in tl:
            return True
        msg_type = str(getattr(cfg, "message_type", "") or "").lower()
        custom_proc = str(getattr(cfg, "custom_processor", "") or "").lower()
        # 与 HDF5 一致：显式 sixforce 视为六维力矩源
        if "sixforce" in msg_type or "sixforce" in custom_proc:
            return True
        return False

    for src in state_sources:
        topic = str(src.get("topic", "") or "")
        cfg = cfg_by_topic.get(topic)
        if not _is_force_source(topic, cfg):
            continue
        side = _guess_side(topic, cfg)
        if side:
            # 发生重复时取第一个
            if side not in force_by_side:
                force_by_side[side] = src
    return force_by_side


def _extract_force_vector(
    aligned_data: Dict[str, List[Dict[str, Any]]],
    frame_idx: int,
    src: Dict[str, str],
) -> np.ndarray:
    topic = src["topic"]
    field = src.get("field", "data")
    # 与 HDF5 口径保持一致：力话题缺失/缺帧时不阻断流程，回退零向量。
    if topic not in aligned_data or frame_idx >= len(aligned_data[topic]):
        return np.zeros((6,), dtype=np.float32)
    msg = aligned_data[topic][frame_idx] or {}
    val = msg.get(field)
    if val is None:
        return np.zeros((6,), dtype=np.float32)
    arr = np.asarray(val, dtype=np.float32).flatten()
    # 固定输出维度 (6,)
    if arr.shape[0] < 6:
        arr = np.pad(arr, (0, 6 - arr.shape[0]), mode="constant", constant_values=0.0)
    elif arr.shape[0] > 6:
        arr = arr[:6]
    return arr


def _extract_images_for_frame(
    aligned_data: Dict[str, List[Dict[str, Any]]],
    frame_idx: int,
    camera_mapping: Dict[str, str],
) -> Dict[str, np.ndarray]:
    """提取当前帧各相机的图像（BGR->RGB 由 processor 处理）"""
    imgs = {}
    for cam_name, topic in camera_mapping.items():
        if topic not in aligned_data or frame_idx >= len(aligned_data[topic]):
            continue
        msg = aligned_data[topic][frame_idx]
        data = msg.get("data")
        if data is not None:
            imgs[cam_name] = np.asarray(data, dtype=np.uint8)
    return imgs


def _get_instruction(
    mcap_dir: Path,
    lerobot_config: LeRobotConfig,
) -> str:
    """获取任务指令：优先 instructions.json，否则用默认"""
    path = lerobot_config.instructions_path
    if path:
        json_path = mcap_dir / path
        if json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    d = json.load(f)
                instrs = d.get("instructions", [d.get("instruction", lerobot_config.default_instruction)])
                return np.random.choice(instrs) if isinstance(instrs, list) else instrs
            except Exception as e:
                log_warning(f"读取 instructions.json 失败: {e}")
    return lerobot_config.default_instruction


def _create_lerobot_dataset(
    lerobot_config: LeRobotConfig,
    topic_configs: List["TopicConfig"],
    joint_dim: int,
    camera_shapes: Dict[str, Tuple[int, ...]],
) -> "LeRobotDataset":
    """创建空的 LeRobot 数据集"""
    if not LEROBOT_AVAILABLE:
        raise RuntimeError("未安装 lerobot，请执行: pip install lerobot")

    motors = [f"joint_{i}" for i in range(joint_dim)]
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (joint_dim,),
            "names": [motors],
        },
        "action": {
            "dtype": "float32",
            "shape": (joint_dim,),
            "names": [motors],
        },
    }
    if lerobot_config.has_velocity:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (joint_dim,),
            "names": [motors],
        }
    if lerobot_config.has_effort:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": (joint_dim,),
            "names": [motors],
        }
    for cam_name, shape in camera_shapes.items():
        features[f"observation.images.{cam_name}"] = {
            "dtype": "image" if lerobot_config.mode == "image" else "video",
            "shape": shape,
            "names": ["height", "width", "channels"],
        }

    # 末端六维力矩作为额外观测：
    # - observation.force.{side}: 6 维 [Fx,Fy,Fz,Mx,My,Mz]（兼容历史命名）
    # - observation.force_xyz.{side}: 3 维 [Fx,Fy,Fz]
    # - observation.torque.{side}: 3 维 [Mx,My,Mz]
    force_by_side = _get_force_sources(lerobot_config.state_sources, topic_configs)
    if force_by_side:
        for side in sorted(force_by_side.keys()):
            features[f"observation.force.{side}"] = {
                "dtype": "float32",
                "shape": (6,),
                "names": ["Fx", "Fy", "Fz", "Mx", "My", "Mz"],
            }
            features[f"observation.force_xyz.{side}"] = {
                "dtype": "float32",
                "shape": (3,),
                "names": ["Fx", "Fy", "Fz"],
            }
            features[f"observation.torque.{side}"] = {
                "dtype": "float32",
                "shape": (3,),
                "names": ["Mx", "My", "Mz"],
            }

    if Path(HF_LEROBOT_HOME / lerobot_config.repo_id).exists():
        import shutil
        shutil.rmtree(HF_LEROBOT_HOME / lerobot_config.repo_id)

    # Convert fps to integer if it's a float, to avoid 'float' object has no attribute 'numerator' error
    # in av library when encoding video
    fps = lerobot_config.fps
    if isinstance(fps, float) and fps.is_integer():
        fps = int(fps)

    return LeRobotDataset.create(
        repo_id=lerobot_config.repo_id,
        fps=fps,
        robot_type=lerobot_config.robot_type,
        features=features,
        use_videos=lerobot_config.use_videos,
        tolerance_s=0.0001,
        image_writer_processes=max(1, int(getattr(lerobot_config, "image_writer_processes", 1) or 1)),
        image_writer_threads=max(1, int(getattr(lerobot_config, "image_writer_threads", 1) or 1)),
        video_backend=getattr(lerobot_config, "video_backend", None),
        batch_encoding_size=max(1, int(getattr(lerobot_config, "batch_encoding_size", 1) or 1)),
        vcodec=str(getattr(lerobot_config, "vcodec", "libx264") or "libx264"),
        streaming_encoding=bool(getattr(lerobot_config, "streaming_encoding", False)),
        encoder_queue_maxsize=max(1, int(getattr(lerobot_config, "encoder_queue_maxsize", 10) or 10)),
        encoder_threads=getattr(lerobot_config, "encoder_threads", None),
    )


def _align_mcap_data(
    reader: FlexibleMcapReader,
    data: Dict[str, List[Dict[str, Any]]],
    alignment_config: AlignmentConfig,
) -> Dict[str, List[Dict[str, Any]]]:
    """执行时间对齐，统一各策略的返回值"""
    if alignment_config.strategy == "backfill_on_grid":
        aligned_data, *_ = reader.align_data_backfill_on_grid(data, alignment_config)
    elif alignment_config.strategy == "hybrid_alignment":
        aligned_data, *_ = reader.align_data_hybrid(data, alignment_config)
    else:
        aligned_data, *_ = reader.align_data_with_window(data, alignment_config)
    return aligned_data


def _process_single_mcap(
    mcap_path: str,
    reader: FlexibleMcapReader,
    topic_configs: List[TopicConfig],
    alignment_config: AlignmentConfig,
    lerobot_config: LeRobotConfig,
) -> Tuple[
    Optional[Dict],
    Optional[int],
    Optional[int],
    Optional[Dict[str, Tuple[int, ...]]],
    Optional[str],
]:
    """
    处理单个 MCAP：读取、对齐，成功返回 (aligned_data, num_frames, joint_dim, camera_shapes, None)。
    失败最后一项为可读错误说明（便于后台任务展示，而非仅 “returned false”）。
    """
    data = reader.process_mcap(mcap_path)
    if not data:
        msg = (
            "MCAP 解析无数据：请检查 TopicConfig 是否与文件内话题一致，"
            "或先在「分析」页确认话题列表后再转换。"
        )
        logger.error(msg)
        return None, None, None, None, msg
    aligned_data = _align_mcap_data(reader, data, alignment_config)
    if not aligned_data:
        msg = "时间对齐后无可用数据（alignment 策略或主时间线话题可能不匹配）。"
        logger.error(msg)
        return None, None, None, None, msg
    non_empty = [len(aligned_data[t]) for t in aligned_data if len(aligned_data[t]) > 0]
    if not non_empty:
        msg = "对齐结果中所有话题长度为 0。"
        logger.error(msg)
        return None, None, None, None, msg
    num_frames = min(non_empty)
    if num_frames == 0:
        msg = "对齐后有效帧数为 0。"
        logger.error(msg)
        return None, None, None, None, msg
    force_by_side = _get_force_sources(lerobot_config.state_sources, topic_configs)
    force_topics = {str(v.get("topic", "") or "") for v in force_by_side.values()}
    for src in lerobot_config.state_sources:
        topic = src["topic"]
        # 与 HDF5 保持一致：六维力话题可缺省，不作为 state 主向量硬约束。
        if topic in force_topics:
            continue
        if topic not in aligned_data or len(aligned_data[topic]) < num_frames:
            actual = len(aligned_data[topic]) if topic in aligned_data else 0
            msg = (
                f"state_sources 中话题「{topic}」对齐后数据不足（需要 {num_frames} 帧，实际 {actual}）。"
            )
            logger.error(msg)
            return None, None, None, None, msg
    try:
        state_0 = _extract_state_vector(aligned_data, 0, lerobot_config.state_sources)
        joint_dim = int(state_0.size)
    except Exception as e:
        msg = f"从对齐数据提取 state 向量失败: {e}"
        logger.error(msg)
        return None, None, None, None, msg
    camera_shapes = {}
    for cam_name, topic in lerobot_config.camera_mapping.items():
        if topic in aligned_data and len(aligned_data[topic]) > 0:
            msg = aligned_data[topic][0]
            d = msg.get("data")
            if d is not None:
                arr = np.asarray(d)
                camera_shapes[cam_name] = tuple(arr.shape)
            else:
                camera_shapes[cam_name] = (480, 640, 3)
        else:
            camera_shapes[cam_name] = (480, 640, 3)
    if not camera_shapes:
        logger.warning("无相机数据，使用默认 (480,640,3)")
        for cam_name in lerobot_config.camera_mapping:
            camera_shapes[cam_name] = (480, 640, 3)
    return aligned_data, num_frames, joint_dim, camera_shapes, None


def _write_episode_to_dataset(
    dataset: LeRobotDataset,
    aligned_data: Dict[str, List[Dict[str, Any]]],
    num_frames: int,
    lerobot_config: LeRobotConfig,
    topic_configs: List["TopicConfig"],
    camera_shapes: Dict[str, Tuple[int, ...]],
    mcap_path: str,
) -> None:
    """将对齐后的数据写入 LeRobot 数据集的单个 episode"""
    states = np.stack([
        _extract_state_vector(aligned_data, i, lerobot_config.state_sources)
        for i in range(num_frames)
    ]).astype(np.float32)
    if lerobot_config.action_mode == "next_state":
        actions = np.roll(states, -1, axis=0)
        actions[-1] = states[-1]
    else:
        actions = states.copy()
    mcap_dir = Path(mcap_path).parent
    instruction = _get_instruction(mcap_dir, lerobot_config)
    last_imgs: Dict[str, np.ndarray] = {}
    force_by_side = _get_force_sources(lerobot_config.state_sources, topic_configs)
    for i in range(num_frames):
        force_obs: Dict[str, torch.Tensor] = {}
        for side, src in force_by_side.items():
            topic = src["topic"]
            msg = aligned_data[topic][i] if topic in aligned_data and i < len(aligned_data[topic]) else {}
            fvec6 = _extract_force_vector(aligned_data, i, src)
            force_obs[f"observation.force.{side}"] = torch.from_numpy(fvec6).float()
            f3 = msg.get("force")
            t3 = msg.get("torque")
            if f3 is None or t3 is None:
                f3 = fvec6[:3]
                t3 = fvec6[3:]
            force_obs[f"observation.force_xyz.{side}"] = torch.from_numpy(np.asarray(f3, dtype=np.float32).flatten()[:3]).float()
            force_obs[f"observation.torque.{side}"] = torch.from_numpy(np.asarray(t3, dtype=np.float32).flatten()[:3]).float()

        frame = {
            "observation.state": torch.from_numpy(states[i]).float(),
            "action": torch.from_numpy(actions[i]).float(),
            "task": instruction,
        }
        frame.update(force_obs)
        imgs = _extract_images_for_frame(
            aligned_data, i, lerobot_config.camera_mapping
        )
        for cam_name in lerobot_config.camera_mapping.keys():
            img = imgs.get(cam_name)
            if img is not None:
                last_imgs[cam_name] = img
            else:
                img = last_imgs.get(cam_name)
            if img is None:
                shape = camera_shapes.get(cam_name) or (480, 640, 3)
                img = np.zeros(shape, dtype=np.uint8)
            frame[f"observation.images.{cam_name}"] = img
        dataset.add_frame(frame)
    dataset.save_episode()


def convert_mcap_to_lerobot(
    mcap_path: str,
    output_repo_id: str,
    topic_configs: List[TopicConfig],
    alignment_config: AlignmentConfig,
    lerobot_config: LeRobotConfig,
    dataset: Optional[LeRobotDataset] = None,
) -> Tuple[bool, Optional[LeRobotDataset], Optional[str]]:
    """
    将单个 MCAP 转换为 LeRobot 的一个 episode。
    dataset 为 None 时创建新数据集，否则追加到现有 dataset。
    返回 (success, dataset, error_message)；成功时 error_message 为 None。
    """
    if not LEROBOT_AVAILABLE:
        msg = (
            "当前 Python 环境无法 import lerobot（LEROBOT_AVAILABLE=False）。"
            "请确认后台服务使用的解释器与已安装 lerobot 的虚拟环境一致。"
        )
        logger.error("%s 原始提示: pip install lerobot", msg)
        return False, None, msg

    _limit_cpu_threads(int(getattr(lerobot_config, "encoder_threads", 1) or 1))
    reader = FlexibleMcapReader(topic_configs)
    aligned_data, num_frames, joint_dim, camera_shapes, proc_err = _process_single_mcap(
        mcap_path, reader, topic_configs, alignment_config, lerobot_config
    )
    if proc_err:
        return False, None, proc_err

    cfg = LeRobotConfig(
        repo_id=output_repo_id,
        robot_type=lerobot_config.robot_type,
        fps=lerobot_config.fps,
        state_sources=lerobot_config.state_sources,
        action_mode=lerobot_config.action_mode,
        camera_mapping=lerobot_config.camera_mapping,
        has_velocity=lerobot_config.has_velocity,
        has_effort=lerobot_config.has_effort,
        default_instruction=lerobot_config.default_instruction,
        instructions_path=lerobot_config.instructions_path,
        mode=lerobot_config.mode,
        use_videos=lerobot_config.use_videos,
        image_writer_processes=int(getattr(lerobot_config, "image_writer_processes", 1) or 1),
        image_writer_threads=int(getattr(lerobot_config, "image_writer_threads", 1) or 1),
        video_backend=getattr(lerobot_config, "video_backend", None),
        batch_encoding_size=int(getattr(lerobot_config, "batch_encoding_size", 1) or 1),
        vcodec=str(getattr(lerobot_config, "vcodec", "libx264") or "libx264"),
        streaming_encoding=bool(getattr(lerobot_config, "streaming_encoding", False)),
        encoder_queue_maxsize=int(getattr(lerobot_config, "encoder_queue_maxsize", 10) or 10),
        encoder_threads=getattr(lerobot_config, "encoder_threads", 1),
    )

    if dataset is None:
        try:
            dataset = _create_lerobot_dataset(cfg, topic_configs, joint_dim, camera_shapes)
        except Exception as e:
            msg = f"创建 LeRobot 数据集失败: {e}"
            logger.error(msg)
            return False, None, msg

    try:
        _write_episode_to_dataset(
            dataset, aligned_data, num_frames, lerobot_config, topic_configs, camera_shapes, mcap_path
        )
    except Exception as e:
        msg = f"写入 LeRobot episode 失败: {e}"
        logger.error(msg)
        return False, None, msg
    log_info(f"成功转换: {mcap_path} -> LeRobot episode")
    return True, dataset, None


def main():
    parser = argparse.ArgumentParser(
        description="MCAP 直接转 LeRobot 数据集（方案 B）"
    )
    parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="MCAP 话题与对齐配置 (YAML)，需包含 topics 和 alignment",
    )
    parser.add_argument(
        "--lerobot-config",
        "-l",
        default=None,
        help="LeRobot 映射配置 (YAML)。若省略且 --config 含 lerobot 节则从 --config 读取",
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="MCAP 文件或目录",
    )
    parser.add_argument(
        "--output-repo",
        "-o",
        required=True,
        help="LeRobot 数据集 repo_id，如 my_org/my_dataset",
    )
    parser.add_argument(
        "--output-dir",
        "-d",
        default="lerobot_output",
        help="输出目录（与 lerobot_output 格式一致：data/, videos/, meta/），默认 lerobot_output",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="详细日志",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # 加载 MCAP 配置
    try:
        topic_configs, alignment_config, _ = load_config(args.config)
        log_info(f"MCAP 配置加载成功，话题数: {len(topic_configs)}")
    except Exception as e:
        logger.error(f"加载 MCAP 配置失败: {e}")
        return 1

    # 加载 LeRobot 配置
    lerobot_config_path = args.lerobot_config
    if not lerobot_config_path:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if cfg and "lerobot" in cfg:
            lerobot_config_path = args.config
            log_info("从 MCAP 配置中读取 lerobot 节")
        else:
            logger.error("未指定 --lerobot-config 且 --config 中无 lerobot 节")
            return 1
    try:
        lerobot_config = load_lerobot_config(lerobot_config_path)
        lerobot_config.repo_id = args.output_repo
        log_info(f"LeRobot 配置加载成功, repo_id={lerobot_config.repo_id}")
    except Exception as e:
        logger.error(f"加载 LeRobot 配置失败: {e}")
        return 1

    input_path = Path(args.input)
    success_count = 0
    dataset = None

    if input_path.is_file() and input_path.suffix.lower() == ".mcap":
        ok, dataset, conv_err = convert_mcap_to_lerobot(
            str(input_path),
            args.output_repo,
            topic_configs,
            alignment_config,
            lerobot_config,
            dataset=None,
        )
        if conv_err and not ok:
            logger.error(conv_err)
        success_count = 1 if ok else 0
    elif input_path.is_dir():
        mcap_files = sorted(input_path.glob("*.mcap"))
        if not mcap_files:
            logger.error(f"目录中未找到 MCAP 文件: {input_path}")
            return 1
        log_info(f"找到 {len(mcap_files)} 个 MCAP 文件")
        for mcap_file in tqdm.tqdm(mcap_files, desc="转换"):
            ok, dataset, conv_err = convert_mcap_to_lerobot(
                str(mcap_file),
                args.output_repo,
                topic_configs,
                alignment_config,
                lerobot_config,
                dataset=dataset,
            )
            if conv_err and not ok:
                logger.error("%s: %s", mcap_file, conv_err)
            if ok:
                success_count += 1
    else:
        logger.error("--input 需为 .mcap 文件或包含 .mcap 的目录")
        return 1

    if success_count > 0:
        log_info(f"转换完成，成功 {success_count} 个 episode")
        if LEROBOT_AVAILABLE and args.output_dir:
            import shutil
            output_dir = Path(args.output_dir).resolve()
            dataset_source = Path(HF_LEROBOT_HOME) / args.output_repo
            if dataset_source.exists():
                if output_dir.exists():
                    shutil.rmtree(output_dir)
                shutil.copytree(dataset_source, output_dir)
                log_info(f"输出已复制到: {output_dir}（与 lerobot_output 格式一致）")
            else:
                log_info(f"LeRobot 数据集路径: {dataset_source}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
