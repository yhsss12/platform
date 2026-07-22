"""
数据资产轻量元数据解析：HDF5 / MCAP / LeRobot
"""
import json
import os
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def parse_hdf5_meta(file_path: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    轻量解析 HDF5：相机数、是否有 actions/joint_states、episode 维度等。
    返回 (meta_dict, error_msg)，失败时 meta_dict 为 None。
    """
    try:
        import h5py
    except ImportError:
        return None, "h5py 未安装"
    meta: Dict[str, Any] = {}
    try:
        with h5py.File(file_path, "r") as f:
            keys = list(f.keys())
            # 相机数量：observations/images 或 顶层 images 的 group key 数
            if "observations" in keys:
                obs = f["observations"]
                if "images" in obs:
                    meta["cameras"] = len(list(obs["images"].keys()))
                else:
                    meta["cameras"] = 0
            elif "images" in keys:
                images_grp = f["images"]
                meta["cameras"] = len(list(images_grp.keys())) if hasattr(images_grp, "keys") else 1
            else:
                meta["cameras"] = 0
            meta["hasActions"] = "actions" in keys
            obs_keys = list(f.get("observations", {}).keys()) if "observations" in keys else []
            meta["hasJoint"] = "joint_states" in obs_keys
            if "actions" in keys:
                try:
                    actions = f["actions"]
                    if hasattr(actions, "shape") and len(actions.shape) > 0:
                        meta["episodeLength"] = int(actions.shape[0])
                except Exception:
                    pass
    except Exception as e:
        return None, str(e)[:200]
    return meta, None


def parse_mcap_meta(file_path: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    轻量解析 MCAP：topic 数量、起止时间、时长、消息总数。
    返回 (meta_dict, error_msg)。
    """
    try:
        from mcap.reader import make_reader
    except ImportError:
        return (
            None,
            "mcap 库未安装（当前 Python 无法 import mcap）。请在运行后端的同一环境中执行："
            "python3 -m pip install 'mcap>=1.0.0' 'mcap-ros2-support>=0.0.10' "
            "或 cd backend && python3 -m pip install -r requirements.txt；Docker 请重新 build 镜像。",
        )
    meta: Dict[str, Any] = {}
    try:
        with open(file_path, "rb") as f:
            # mcap.make_reader 的参数在不同版本存在差异：
            # - 新版：decoder_factories=[...]
            # - 旧版：无该参数
            # 这里做兼容：不需要解码，仅迭代消息即可。
            try:
                reader = make_reader(f, decoder_factories=[])
            except TypeError:
                reader = make_reader(f)
            topics = set()
            start_ts = None
            end_ts = None
            count = 0
            for item in reader.iter_messages():
                # 兼容：不同版本/调用方式下返回结构可能是：
                # - (schema, channel, message)
                # - message 对象（极少数封装）
                channel = None
                message = None
                if isinstance(item, tuple) and len(item) >= 3:
                    _, channel, message = item[0], item[1], item[2]
                else:
                    message = item
                    channel = getattr(item, "channel", None)

                topic = getattr(channel, "topic", None) if channel is not None else None
                if isinstance(topic, str) and topic:
                    topics.add(topic)
                ts = getattr(message, "log_time", None) or getattr(message, "publish_time", None)
                if ts is not None:
                    if start_ts is None:
                        start_ts = ts
                    end_ts = ts
                count += 1
                if count > 50000:
                    break
            meta["topics"] = len(topics)
            meta["messages"] = count
            if start_ts is not None and end_ts is not None:
                duration_ns = end_ts - start_ts
                meta["durationSec"] = round(duration_ns / 1e9, 2)
            else:
                meta["durationSec"] = 0
    except Exception as e:
        return None, str(e)[:200]
    return meta, None


def parse_lerobot_meta(file_path: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    LeRobot：支持 .zip 或目录。
    zip：解压到临时目录检查 meta.json / dataset_info.json / episodes/。
    目录：直接检查。
    返回 (meta_dict, error_msg)。
    """
    meta: Dict[str, Any] = {"episodes": 0, "hasVideos": False, "hasStates": False}
    dir_path: Optional[str] = None
    is_zip = file_path.lower().endswith(".zip")
    if is_zip:
        try:
            import tempfile
            with zipfile.ZipFile(file_path, "r") as zf:
                names = zf.namelist()
                # 找 meta.json / dataset_info.json
                has_meta = any("meta.json" in n or "dataset_info.json" in n for n in names)
                episodes = [n for n in names if "episodes" in n and (n.endswith("/") or "episode" in n)]
                meta["episodes"] = len(set(n.split("/")[1] if "/" in n else n for n in episodes if "episode" in n.lower()))
                if meta["episodes"] == 0 and any("episodes" in n for n in names):
                    meta["episodes"] = len([n for n in names if n.startswith("episodes/") and "/" in n])
                meta["hasVideos"] = any("video" in n.lower() or "videos" in n.lower() for n in names)
                meta["hasStates"] = any("state" in n.lower() or "states" in n.lower() for n in names)
                if not meta["episodes"] and has_meta:
                    meta["episodes"] = 1
            return meta, None
        except Exception as e:
            return None, str(e)[:200]
    else:
        p = Path(file_path)
        if not p.is_dir():
            return None, "LeRobot 需要目录或 .zip 文件"
        dir_path = file_path
    if dir_path:
        p = Path(dir_path)
        if (p / "meta.json").exists() or (p / "dataset_info.json").exists():
            meta["hasMetadata"] = True
        episodes_dir = p / "episodes"
        if episodes_dir.is_dir():
            meta["episodes"] = len(list(episodes_dir.iterdir()))
        if (p / "videos").exists() or any((p / "episodes").iterdir() if (p / "episodes").exists() else []):
            meta["hasVideos"] = True
        if (p / "states").exists():
            meta["hasStates"] = True
    return meta, None


def parse_meta_for_asset(file_path: str, format_kind: str) -> Tuple[Optional[str], str, Optional[str]]:
    """
    根据格式解析元数据，返回 (meta_json, parse_status, error_msg)。
    format_kind: "hdf5" | "mcap" | "lerobot"
    """
    if format_kind == "hdf5":
        meta_dict, err = parse_hdf5_meta(file_path)
    elif format_kind == "mcap":
        meta_dict, err = parse_mcap_meta(file_path)
    elif format_kind == "lerobot":
        meta_dict, err = parse_lerobot_meta(file_path)
    else:
        return None, "未解析", "未知格式"
    if err:
        return None, "失败", err
    return json.dumps(meta_dict, ensure_ascii=False), "成功", None
