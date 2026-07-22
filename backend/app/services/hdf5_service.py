"""
HDF5 文件操作服务
封装 label_task_description.py 的核心功能（标注任务描述生成）
"""
import os
import sys
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import h5py
import numpy as np
import cv2
import base64
from io import BytesIO
from app.core.config import settings

# 添加项目根目录到路径，以便导入 label_task_description.py
# __file__ 是 backend/app/services/hdf5_service.py
# 需要往上4层到达项目根目录 eai-ide/
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 在导入 label_task_description 之前，确保环境变量已设置（从 .env 文件加载）
# 使用 python-dotenv 加载 .env 文件（如果存在）
try:
    from dotenv import load_dotenv
    # 尝试从 backend 目录和项目根目录加载 .env 文件
    env_paths = [
        Path(__file__).parent.parent / ".env",  # backend/.env
        PROJECT_ROOT / ".env",  # eai-ide/.env
    ]
    _env_loaded = False
    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(env_path, override=False)  # override=False 表示不覆盖已存在的环境变量
            print(f"✓ 已加载环境变量文件: {env_path}")
            _env_loaded = True
            break
    if not _env_loaded and not any(
        os.getenv(k) for k in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL")
    ):
        print("ℹ 未找到 .env；若已用 Docker env_file 注入 OpenAI/数据库变量可忽略")
except ImportError:
    # 如果没有安装 python-dotenv，跳过
    print("⚠ 警告: python-dotenv 未安装，无法自动加载 .env 文件")
    pass

try:
    from label_task_description import (
        find_image_group,
        list_cameras_in_group,
        get_camera_data,
        extract_episode_index,
        get_episode_length,
        gen_task_description,
    )
except (ImportError, RuntimeError) as e:
    # 如果 label_task_description.py 不存在或环境变量未设置，定义 fallback 函数
    print(f"Warning: Could not import label_task_description: {e}")
    print("提示: 请设置环境变量 OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL")
    find_image_group = None
    list_cameras_in_group = None
    get_camera_data = None
    extract_episode_index = None
    get_episode_length = None
    gen_task_description = None


class HDF5Service:
    """HDF5 文件操作服务"""

    # HDF5 文件魔数（superblock signature）
    _HDF5_SIGNATURE = b'\x89HDF\r\n\x1a\n'

    def __init__(self, data_dir: Optional[str] = None):
        """
        初始化服务
        
        Args:
            data_dir: HDF5 文件存储目录，如果为 None 则从环境变量读取
        """
        self.data_dir = data_dir or os.getenv("HDF5_DATA_DIR", "/tmp/hdf5_data")
        os.makedirs(self.data_dir, exist_ok=True)

    def _is_hdf5_file(self, path: str) -> bool:
        """检查文件是否为有效 HDF5（避免对非 HDF5 二进制误调用 h5py）"""
        try:
            if not path or not os.path.isfile(path):
                return False
            with open(path, "rb") as f:
                sig = f.read(len(self._HDF5_SIGNATURE))
            return sig == self._HDF5_SIGNATURE
        except Exception:
            return False

    def find_image_group(self, hdf: h5py.File) -> Tuple[Optional[str], Optional[h5py.Group]]:
        """
        查找图像组：先尝试固定路径，再从根全树扫描（兼容任意 HDF5 结构）。
        """
        if find_image_group:
            return find_image_group(hdf)

        def _group_has_image_children(grp):
            for k in grp.keys():
                if "timestamp" in k.lower():
                    continue
                obj = grp[k]
                if isinstance(obj, h5py.Dataset) and obj.ndim >= 3 and obj.shape[-1] in (1, 3, 4):
                    return True
                if isinstance(obj, h5py.Group):
                    for kk in obj.keys():
                        sub = obj[kk]
                        if isinstance(sub, h5py.Dataset) and sub.ndim >= 3 and sub.shape[-1] in (1, 3, 4):
                            return True
            return False

        def _scan_tree_for_image_group(file_handle, grp):
            if grp is None or not isinstance(grp, h5py.Group):
                return None, None
            if _group_has_image_children(grp):
                path = grp.name if grp.name else "/"
                return path, grp
            for k in grp.keys():
                child = grp[k]
                if isinstance(child, h5py.Group):
                    found_path, found_grp = _scan_tree_for_image_group(file_handle, child)
                    if found_grp is not None:
                        return found_path, found_grp
            return None, None

        # 1) 固定路径
        candidate_groups = ["observations/images", "images", "observations"]
        for g in candidate_groups:
            if g in hdf:
                grp = hdf[g]
                if isinstance(grp, h5py.Group):
                    for k in grp.keys():
                        obj = grp[k]
                        if isinstance(obj, h5py.Dataset) and obj.ndim >= 3 and obj.shape[-1] in (1, 3, 4):
                            return g, grp
                        if isinstance(obj, h5py.Group):
                            for kk in obj.keys():
                                sub = obj[kk]
                                if isinstance(sub, h5py.Dataset) and sub.ndim >= 3 and sub.shape[-1] in (1, 3, 4):
                                    return g, grp
        # 2) 全树扫描
        return _scan_tree_for_image_group(hdf, hdf)

    def list_cameras(self, hdf: h5py.File) -> List[str]:
        """
        列出相机列表，返回从根到相机节点的完整路径（如 observations/images/camera1），便于展示。
        """
        group_path, grp = self.find_image_group(hdf)
        if grp is None:
            return []

        if list_cameras_in_group:
            keys = list_cameras_in_group(grp)
        else:
            keys = []
            for k in grp.keys():
                obj = grp[k]
                if "timestamp" in k.lower():
                    continue
                if isinstance(obj, h5py.Dataset) and obj.ndim >= 3 and obj.shape[-1] in (1, 3, 4):
                    keys.append(k)
                elif isinstance(obj, h5py.Group):
                    for kk in obj.keys():
                        sub = obj[kk]
                        if isinstance(sub, h5py.Dataset) and sub.ndim >= 3 and sub.shape[-1] in (1, 3, 4):
                            keys.append(k)
                            break

        # 拼接为从根到相机的完整路径（不含首位的 /）
        if not group_path or group_path == "/":
            return list(keys)
        base = group_path.rstrip("/")
        return [f"{base}/{k}" for k in keys]

    def _parse_camera_ref(self, hdf: h5py.File, camera_name: str) -> Tuple[str, str]:
        """
        将相机引用解析为 (group_path, camera_key)。
        camera_name 可为完整路径（如 observations/images/camera1）或短名（如 camera1）。
        """
        if "/" in camera_name:
            parts = camera_name.rsplit("/", 1)
            group_path = parts[0] or "/"
            camera_key = parts[1]
            return group_path, camera_key
        group_path, grp = self.find_image_group(hdf)
        return (group_path or "/"), camera_name

    def _is_image_like(self, node) -> bool:
        """判断节点是否为图像状 Dataset（ndim>=3, 最后一维 1/3/4）。"""
        if not isinstance(node, h5py.Dataset):
            return False
        if node.ndim < 3:
            return False
        try:
            return int(node.shape[-1]) in (1, 3, 4)
        except Exception:
            return False

    def _first_image_key_in_group(self, grp: h5py.Group) -> Optional[str]:
        """返回 Group 中第一个图像状子节点的 key，若无则返回 None。"""
        for k in grp.keys():
            if "timestamp" in k.lower():
                continue
            obj = grp[k]
            if self._is_image_like(obj):
                return k
            if isinstance(obj, h5py.Group):
                for kk in obj.keys():
                    if self._is_image_like(obj[kk]):
                        return k
                break
        return None

    def _resolve_path_to_frame_source(
        self, f: h5py.File, path: str
    ) -> Tuple[Optional[h5py.Group], Optional[str]]:
        """
        将任意路径解析为可读帧的 (group, key)，即 grp[key] 为图像 Dataset。
        若路径无效或不是图像，返回 (None, None)，调用方应返回黑帧。
        """
        path_norm = (path or "").strip().lstrip("/")
        if not path_norm:
            group_path, grp = self.find_image_group(f)
            if grp is None:
                return None, None
            key = self._first_image_key_in_group(grp)
            return (grp, key) if key else (None, None)
        try:
            node = f[path_norm]
        except (KeyError, ValueError):
            return None, None
        if self._is_image_like(node):
            if "/" in path_norm:
                parent_path, key = path_norm.rsplit("/", 1)
                grp = f[parent_path] if parent_path else f
            else:
                grp, key = f, path_norm
            return grp, key
        if isinstance(node, h5py.Group):
            key = self._first_image_key_in_group(node)
            return (node, key) if key else (None, None)
        return None, None

    def _black_jpeg(self, width: int = 64, height: int = 64, quality: int = 85) -> bytes:
        """返回一张小尺寸黑图 JPEG，用于“非相机”话题的占位。"""
        black = np.zeros((height, width, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", black, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes()

    def list_camera_candidate_paths(
        self, hdf: h5py.File, image_only: bool = True
    ) -> List[str]:
        """
        列出路径名中包含 "camera" 的节点供下拉选择。
        image_only=True（默认）：只返回能解析为图像源的路径，列表更短、可播放。
        image_only=False：返回所有含 "camera" 的路径（含 depth/pointcloud 等），非相机会得到黑帧。
        """
        collected: List[str] = []

        def visit(name: str, obj) -> None:
            path = name.lstrip("/")
            if not path:
                return
            if "camera" in path.lower():
                collected.append(path)
            if isinstance(obj, h5py.Group):
                for k in obj.keys():
                    visit(f"{path}/{k}" if path else k, obj[k])

        for key in hdf.keys():
            visit(key, hdf[key])
        paths = sorted(set(collected))
        if not image_only:
            return paths
        # 只保留能解析为图像源的路径（排除 depth/origin/pointcloud 等中间节点）
        return [p for p in paths if self._resolve_path_to_frame_source(hdf, p)[0] is not None]

    def get_episodes(self, task_id: Optional[str] = None) -> List[Dict]:
        """
        获取所有 episode 列表（可选按 taskId 过滤）
        
        Args:
            task_id: 任务 ID，如果提供则只返回该任务下的 episodes
            
        Returns:
            List[Dict]: episode 列表，每个包含 id, name, path
        """
        episodes = []
        
        # 如果指定了 task_id，只扫描该任务的目录
        if task_id:
            task_dir = os.path.join(self.data_dir, task_id)
            if not os.path.exists(task_dir):
                return episodes
            scan_dir = task_dir
        else:
            scan_dir = self.data_dir
        
        if not os.path.exists(scan_dir):
            return episodes
        
        # 递归扫描目录（支持子目录）
        for root, dirs, files in os.walk(scan_dir):
            for filename in files:
                if filename.lower().endswith((".hdf5", ".h5")):
                    filepath = os.path.join(root, filename)
                    if os.path.isfile(filepath):
                        episode_id = self._extract_episode_id(filename)
                        # 如果文件名重复，使用相对路径生成唯一 ID
                        if any(ep["id"] == episode_id for ep in episodes):
                            relative_path = os.path.relpath(filepath, scan_dir)
                            episode_id = f"{episode_id}_{hash(relative_path) % 10000}"
                        
                        episodes.append({
                            "id": episode_id,
                            "name": filename,
                            "path": filepath,
                        })
        
        # 按 episode_id 排序
        episodes.sort(key=lambda x: int(x["id"]) if x["id"].isdigit() else 0)
        return episodes

    def get_episode_info(
        self, episode_id: str, camera_candidates: bool = True
    ) -> Optional[Dict]:
        """
        获取 episode 详细信息。

        Args:
            episode_id: episode ID
            camera_candidates: 默认 True，返回所有路径名含 "camera" 的节点；False 时仅返回校验过的相机列表

        Returns:
            Dict: 含 cameras, frameCount 等
        """
        episode = self._find_episode_by_id(episode_id)
        if not episode:
            return None

        try:
            with h5py.File(episode["path"], "r") as f:
                if camera_candidates:
                    cameras = self.list_camera_candidate_paths(f, image_only=True)
                    if not cameras:
                        cameras = self.list_camera_candidate_paths(f, image_only=False)
                    if not cameras:
                        cameras = self.list_cameras(f)
                else:
                    cameras = self.list_cameras(f)
                # 用第一个有帧数的路径计算 frameCount，保证进度条有总帧数
                frame_count = 0
                for cam in cameras:
                    n = self.get_frame_count(episode["path"], cam)
                    if n > frame_count:
                        frame_count = n
                        if n > 1:
                            break
                return {
                    "id": episode_id,
                    "name": episode["name"],
                    "path": episode["path"],
                    "cameras": cameras,
                    "frameCount": frame_count,
                }
        except Exception as e:
            print(f"Error reading episode {episode_id}: {e}")
            return None

    def get_frame_count(self, hdf5_path: str, camera_name: str) -> int:
        """
        获取指定相机/路径的总帧数。
        camera_name 可为完整路径（含 'camera' 的任意路径）；若非相机话题则返回 1（便于前端显示一帧黑屏）。
        """
        if not self._is_hdf5_file(hdf5_path):
            return 0
        try:
            with h5py.File(hdf5_path, "r") as f:
                grp, key = self._resolve_path_to_frame_source(f, camera_name)
                if grp is not None and key is not None:
                    node = grp[key]
                    if isinstance(node, h5py.Dataset) and node.ndim >= 1:
                        return int(node.shape[0])
                    if isinstance(node, h5py.Group):
                        for ds in ["data", "images", "frames"] + list(node.keys()):
                            if ds in node and isinstance(node[ds], h5py.Dataset):
                                return int(node[ds].shape[0])
        except Exception as e:
            print(f"Error getting frame count: {e}")
        return 1  # 非相机话题返回 1，便于前端显示一帧黑屏

    def get_frame_image(
        self,
        hdf5_path: str,
        camera_name: str,
        frame_index: int,
        quality: int = 85
    ) -> Optional[bytes]:
        """
        获取指定帧的图像（JPEG bytes）。
        camera_name 可为完整路径；若路径不是相机或无效，返回一张黑图（表示该话题不是相机信息）。
        """
        if not self._is_hdf5_file(hdf5_path):
            return self._black_jpeg(quality=quality)
        try:
            with h5py.File(hdf5_path, "r") as f:
                grp, key = self._resolve_path_to_frame_source(f, camera_name)
                if grp is None or key is None:
                    return self._black_jpeg(quality=quality)
                node = grp[key]
                if isinstance(node, h5py.Dataset):
                    if frame_index >= node.shape[0]:
                        return self._black_jpeg(quality=quality)
                    frame = node[frame_index]
                elif isinstance(node, h5py.Group):
                    frame = None
                    for ds in ["data", "images", "frames"] + list(node.keys()):
                        if ds in node and isinstance(node[ds], h5py.Dataset):
                            d = node[ds]
                            if frame_index < d.shape[0]:
                                frame = d[frame_index]
                            break
                    if frame is None:
                        return self._black_jpeg(quality=quality)
                else:
                    return self._black_jpeg(quality=quality)

                frame = np.asarray(frame)
                if frame.dtype != np.uint8:
                    frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
                if frame.ndim == 2:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                elif frame.ndim >= 3 and frame.shape[-1] == 4:
                    frame_bgr = cv2.cvtColor(frame[..., :3], cv2.COLOR_RGB2BGR)
                elif frame.ndim >= 3 and frame.shape[-1] == 3:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                elif frame.ndim >= 3 and frame.shape[-1] == 1:
                    frame_bgr = cv2.cvtColor(frame.squeeze(-1), cv2.COLOR_GRAY2BGR)
                else:
                    frame_bgr = frame if frame.ndim == 3 else np.zeros((64, 64, 3), dtype=np.uint8)
                _, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
                return buf.tobytes()
        except Exception as e:
            print(f"Error getting frame: {e}")
            return self._black_jpeg(quality=quality)

    def generate_task_description(
        self,
        file_path: str,
        camera_name: Optional[str] = None,
        image_topic: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> Optional[str]:
        """
        生成任务描述（调用 label_task_description.gen_task_description）
        支持 HDF5（camera_name）和 MCAP（image_topic）。
        model / api_key / base_url 可选，为 None 时使用环境变量。
        """
        # 为了在 dev 模式下更稳（以及避免启动时 label_task_description 导入失败导致功能不可用），这里做一次惰性导入
        local_gen = gen_task_description
        if local_gen is None:
            try:
                from label_task_description import gen_task_description as _gen  # type: ignore
                local_gen = _gen
            except Exception as e:
                api_key_check = api_key or settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
                if not api_key_check:
                    print("错误: OPENAI_API_KEY 未设置。请设置环境变量或在 API 配置中填写。")
                    print("提示: 在 backend/.env 文件中添加:")
                    print("  OPENAI_BASE_URL=https://kfc-api.sxxe.net")
                    print("  OPENAI_API_KEY=your-api-key")
                    print("  OPENAI_MODEL=gemini-3-pro-preview")
                else:
                    print(f"错误: 导入 label_task_description 失败: {e}")
                return None
        
        try:
            # 完整路径时只把最后一节作为 camera_name（label_task_description 内部会 find_image_group）
            camera_key = camera_name.rsplit("/", 1)[-1] if (camera_name and "/" in camera_name) else camera_name
            return local_gen(
                file_path,
                camera_name=camera_key,
                image_topic=image_topic,
                model=model,
                api_key=api_key,
                base_url=base_url,
            )
        except Exception as e:
            print(f"Error generating task description: {e}")
            import traceback
            traceback.print_exc()
            # Let the caller (worker) mark the job as failed with error details.
            raise

    def _extract_episode_id(self, filename: str) -> str:
        """从文件名提取 episode ID"""
        if extract_episode_index:
            try:
                return str(extract_episode_index(filename))
            except:
                pass
        
        # Fallback: 从文件名提取数字
        import re
        match = re.search(r'episode[_\s]*(\d+)', filename, re.IGNORECASE)
        if match:
            return match.group(1)
        match = re.search(r'(\d+)', filename)
        if match:
            return match.group(1)
        return "0"

    def _find_episode_by_id(self, episode_id: str) -> Optional[Dict]:
        """根据 ID 查找 episode"""
        episodes = self.get_episodes()
        for ep in episodes:
            if ep["id"] == episode_id:
                return ep
        return None
    
    def _open_hdf5(self, hdf5_path: str):
        """打开 HDF5 文件（用于 with 语句）"""
        return h5py.File(hdf5_path, "r")

