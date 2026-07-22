"""
MCAP 文件操作服务
支持从 MCAP 录制文件中读取 CompressedImage 帧，供标注界面使用。
优先使用 /camera*/camera/color/image_raw/compressed 话题。
"""
import logging
import os
import threading
from typing import Optional, List, Tuple, Dict

logger = logging.getLogger(__name__)

# 全局锁：用于 get_frame_image、get_frames_batch 等单次调用
_mcap_file_lock = threading.Lock()

# 帧数缓存：避免每次请求都完整遍历 MCAP 文件导致卡顿
_frame_count_cache: Dict[Tuple[str, str], int] = {}

# 帧图像 LRU 缓存：键为 (abs_path, camera_name, frame_index)，值为 JPEG bytes
_frame_image_cache: Dict[Tuple[str, str, int], bytes] = {}
_FRAME_IMAGE_CACHE_MAX_ITEMS = 1024

# MCAP 颜色图像 topic 模式（优先用于标注）
# 支持 /camera1/camera/color/image_raw/compressed 或 /camera1/camera1/color/image_raw
_COLOR_IMAGE_TOPIC_SUBSTR = "/color/image_raw"


def _is_mcap_file(path: str) -> bool:
    """检查是否为 MCAP 文件"""
    if not path or not os.path.isfile(path):
        return False
    return path.lower().endswith(".mcap")


def _topic_to_camera_name(topic: str) -> str:
    """
    从 topic 提取相机短名，如 /camera1/camera/color/image_raw/compressed -> camera1
    """
    parts = topic.strip("/").split("/")
    if len(parts) >= 1:
        return parts[0]
    return topic


def _is_color_image_topic(topic: str) -> bool:
    """是否为颜色图像 topic。支持 /camera*/.../color/image_raw 或 .../color/image_raw/compressed"""
    if _COLOR_IMAGE_TOPIC_SUBSTR not in topic:
        return False
    # 排除 depth
    if "/depth/" in topic or "/depth_image" in topic:
        return False
    return True


def _ros_msg_to_jpeg_bytes(ros_msg) -> Optional[bytes]:
    """
    将 ROS 图像消息转为 JPEG 字节。
    支持 CompressedImage（直接返回 data）、sensor_msgs/Image（含 color/depth/ mono）。
    """
    # sensor_msgs/Image: data 为 raw 像素，需按 encoding 解码后转 JPEG
    if hasattr(ros_msg, "height") and hasattr(ros_msg, "width") and hasattr(ros_msg, "data") and ros_msg.data:
        try:
            import cv2
            import numpy as np
            h, w = int(ros_msg.height), int(ros_msg.width)
            enc = (getattr(ros_msg, "encoding") or "rgb8").lower()
            step = getattr(ros_msg, "step", 0) or 0
            raw = np.frombuffer(ros_msg.data, dtype=np.uint8)
            # 深度图常见 16UC1 / 32FC1，需按 step 或 dtype 解析
            if "16uc1" in enc or enc == "16uc1":
                raw = np.frombuffer(ros_msg.data, dtype=np.uint16)
                raw = raw[: h * w].reshape((h, w))
                raw = np.nan_to_num(raw, nan=0, posinf=0, neginf=0)
                if raw.max() > raw.min():
                    raw = (raw - raw.min()) / (raw.max() - raw.min()) * 255
                else:
                    raw = np.zeros_like(raw) if raw.dtype != np.uint8 else raw
                img = cv2.cvtColor(raw.astype(np.uint8), cv2.COLOR_GRAY2BGR)
                _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                return buf.tobytes()
            if "32fc1" in enc or enc == "32fc1":
                raw = np.frombuffer(ros_msg.data, dtype=np.float32)
                raw = raw[: h * w].reshape((h, w))
                raw = np.nan_to_num(raw, nan=0, posinf=0, neginf=0)
                if raw.max() > raw.min():
                    raw = (raw - raw.min()) / (raw.max() - raw.min()) * 255
                else:
                    raw = np.zeros_like(raw, dtype=np.uint8)
                img = cv2.cvtColor(raw.astype(np.uint8), cv2.COLOR_GRAY2BGR)
                _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                return buf.tobytes()
            step = step or (w * 3)
            if step != w * 3 and "bgra" in enc:
                step = w * 4
            raw = raw[: int(h) * int(step)].reshape((h, step))
            if "bgra" in enc or "rgba" in enc:
                raw = raw[:, : w * 4].reshape((h, w, 4))
                img = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
            elif "rgb" in enc:
                raw = raw[:, : w * 3].reshape((h, w, 3))
                img = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
            elif "bgr" in enc:
                img = raw[:, : w * 3].reshape((h, w, 3))
            elif "mono" in enc or enc == "8uc1":
                img = cv2.cvtColor(raw[:, :w], cv2.COLOR_GRAY2BGR)
            else:
                img = raw[:, : w * 3].reshape((h, w, 3))
            _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return buf.tobytes()
        except Exception as e:
            logger.warning("_ros_msg_to_jpeg_bytes Image decode error: %s", e)
    # CompressedImage: data 已是 JPEG
    if hasattr(ros_msg, "data") and ros_msg.data:
        return bytes(ros_msg.data)
    return None


class MCAPService:
    """MCAP 文件操作服务"""

    def __init__(self):
        pass

    def list_cameras(self, mcap_path: str) -> List[str]:
        """
        列出 MCAP 中的相机（颜色图像 topic 的完整话题名）。
        返回 /camera*/camera/color/image_raw/compressed 类型的完整 topic 字符串。
        """
        if not _is_mcap_file(mcap_path):
            return []
        try:
            from mcap.reader import make_reader

            cameras: List[str] = []
            with open(mcap_path, "rb") as f:
                reader = make_reader(f)
                summary = reader.get_summary()
                if summary is None or not getattr(summary, "channels", None):
                    return []
                for ch in summary.channels.values():
                    if _is_color_image_topic(ch.topic) and ch.topic not in cameras:
                        cameras.append(ch.topic)
            return sorted(cameras)
        except Exception as e:
            logger.warning("list_cameras error: %s", e)
            return []

    def list_camera_candidate_topics(self, mcap_path: str) -> List[str]:
        """
        列出所有相机相关 topic（与 HDF5 的 camera_candidates 一致），含 color / depth / ir 等图像流，
        供下拉框展示；从 MCAP 文件的 channels 动态扫描，非写死。
        """
        if not _is_mcap_file(mcap_path):
            return []
        try:
            from mcap.reader import make_reader
            candidates: List[str] = []
            with open(mcap_path, "rb") as f:
                reader = make_reader(f)
                summary = reader.get_summary()
                if summary is None or not getattr(summary, "channels", None):
                    return []
                for ch in summary.channels.values():
                    topic = (ch.topic or "").strip()
                    if "camera" not in topic.lower():
                        continue
                    schema_name = ""
                    try:
                        schema_id = getattr(ch, "schema_id", None)
                        if schema_id is not None and getattr(summary, "schemas", None):
                            schema = summary.schemas.get(schema_id)
                            if schema:
                                schema_name = (getattr(schema, "name", None) or getattr(schema, "full_name", None) or "").lower()
                    except Exception:
                        pass
                    topic_lower = topic.lower()
                    is_image_schema = "image" in schema_name or "compressedimage" in schema_name
                    is_image_topic = "image" in topic_lower or "depth" in topic_lower or "ir" in topic_lower or "infrared" in topic_lower
                    if is_image_schema or is_image_topic:
                        if topic not in candidates:
                            candidates.append(topic)
            return sorted(candidates) if candidates else self.list_cameras(mcap_path)
        except Exception as e:
            logger.warning("list_camera_candidate_topics error: %s", e)
            return []

    def get_time_range(self, mcap_path: str, camera_name: str) -> Tuple[int, int]:
        """
        获取指定相机的首尾时间戳（纳秒）。
        返回 (start_time_ns, end_time_ns)，失败时返回 (0, 0)。
        """
        if not _is_mcap_file(mcap_path):
            return (0, 0)
        try:
            from mcap.reader import make_reader

            topic = self._camera_to_topic(mcap_path, camera_name)
            if not topic:
                return (0, 0)
            with open(mcap_path, "rb") as f:
                reader = make_reader(f)
                first_ts: Optional[int] = None
                last_ts: Optional[int] = None
                for _, _, msg in reader.iter_messages(topics=[topic]):
                    ts = getattr(msg, "log_time", None) or getattr(msg, "publish_time", None)
                    if ts is not None:
                        if first_ts is None:
                            first_ts = ts
                        last_ts = ts
                if first_ts is not None and last_ts is not None:
                    return (first_ts, last_ts)
            return (0, 0)
        except Exception as e:
            logger.warning("get_time_range error: %s", e)
            return (0, 0)

    def get_frame_count(self, mcap_path: str, camera_name: str) -> int:
        """获取指定相机的帧数"""
        if not _is_mcap_file(mcap_path):
            return 0
        try:
            key = (os.path.abspath(mcap_path), camera_name)
            if key in _frame_count_cache:
                return _frame_count_cache[key]

            from mcap.reader import make_reader

            topic = self._camera_to_topic(mcap_path, camera_name)
            if not topic:
                return 0
            with open(mcap_path, "rb") as f:
                reader = make_reader(f)
                count = 0
                for _ in reader.iter_messages(topics=[topic]):
                    count += 1
            _frame_count_cache[key] = count
            return count
        except Exception as e:
            logger.warning("get_frame_count error: %s", e)
            return 0

    def _camera_to_topic(self, mcap_path: str, camera_name: str) -> Optional[str]:
        """将相机标识解析为完整 topic。支持完整 topic 或短名（camera1 等）。"""
        cameras = self.list_cameras(mcap_path)
        if not cameras:
            cameras = self.list_camera_candidate_topics(mcap_path)
        if camera_name in cameras:
            return camera_name
        for t in cameras:
            if _topic_to_camera_name(t) == camera_name:
                return t
        return None

    def get_frame_image(
        self,
        mcap_path: str,
        camera_name: str,
        frame_index: int,
        quality: int = 85,
    ) -> Optional[bytes]:
        """
        获取指定帧的图像（JPEG bytes）。
        CompressedImage 已是 JPEG 数据，直接返回；quality 参数对 MCAP 无效。

        为提升拖动与多视窗体验，这里增加一个简单的 LRU 缓存：
        - 以 (abs_path, camera_name, frame_index) 为 key
        - 全局最多保留 _FRAME_IMAGE_CACHE_MAX_ITEMS 条记录，超出则 FIFO 淘汰最旧的一条
        """
        if not _is_mcap_file(mcap_path):
            return None
        try:
            from mcap.reader import make_reader
            from mcap_ros2.decoder import DecoderFactory

            topic = self._camera_to_topic(mcap_path, camera_name)
            if not topic:
                return None

            key = (os.path.abspath(mcap_path), topic, int(frame_index))
            with _mcap_file_lock:
                if key in _frame_image_cache:
                    return _frame_image_cache[key]

                jpg = self._get_frame_image_locked(
                    mcap_path, topic, frame_index, make_reader, DecoderFactory()
                )
                if jpg:
                    if len(_frame_image_cache) >= _FRAME_IMAGE_CACHE_MAX_ITEMS:
                        # 简单 FIFO：弹出任意一条最旧记录
                        try:
                            oldest_key = next(iter(_frame_image_cache.keys()))
                            _frame_image_cache.pop(oldest_key, None)
                        except StopIteration:
                            pass
                    _frame_image_cache[key] = jpg
                return jpg
        except Exception as e:
            logger.warning("get_frame_image error: %s", e)
            import traceback
            logger.debug("%s", traceback.format_exc())
            return None

    def _get_frame_image_locked(
        self, mcap_path: str, topic: str, frame_index: int, make_reader, decoder_factory_cls
    ) -> Optional[bytes]:
        """在锁内读取帧（避免与 list_cameras 等混用锁导致死锁，此处由调用方持锁）"""
        with open(mcap_path, "rb") as f:
            reader = make_reader(f, decoder_factories=[decoder_factory_cls])
            for i, (_, _, _, ros_msg) in enumerate(
                reader.iter_decoded_messages(topics=[topic])
            ):
                if i == frame_index:
                    jpg = _ros_msg_to_jpeg_bytes(ros_msg)
                    return jpg
        return None

    def iter_frames(self, mcap_path: str, camera_name: str, start_frame: int = 0):
        """
        迭代器：从 start_frame 开始逐帧 yield JPEG bytes，顺序读 O(1)/帧。
        每个连接独立打开文件句柄，同一相机多视口可并发读取。
        """
        if not _is_mcap_file(mcap_path):
            return
        try:
            from mcap.reader import make_reader
            from mcap_ros2.decoder import DecoderFactory

            topic = self._camera_to_topic(mcap_path, camera_name)
            if not topic:
                return
            with open(mcap_path, "rb") as f:
                reader = make_reader(f, decoder_factories=[DecoderFactory()])
                for i, (_, _, _, ros_msg) in enumerate(
                    reader.iter_decoded_messages(topics=[topic])
                ):
                    if i < start_frame:
                        continue
                    jpg = _ros_msg_to_jpeg_bytes(ros_msg)
                    if jpg:
                        yield jpg
        except Exception as e:
            logger.warning("iter_frames error: %s", e)

    def get_frames_batch(
        self, mcap_path: str, camera_name: str, start_frame: int, count: int
    ) -> List[bytes]:
        """
        一次性读取多帧，只打开文件一次，用于预加载缓存。
        返回 [frame_start, frame_start+1, ..., frame_start+count-1] 的 JPEG bytes 列表。
        """
        if not _is_mcap_file(mcap_path) or count <= 0:
            return []
        try:
            from mcap.reader import make_reader
            from mcap_ros2.decoder import DecoderFactory

            topic = self._camera_to_topic(mcap_path, camera_name)
            if not topic:
                return []
            result: List[bytes] = []
            with _mcap_file_lock:
                with open(mcap_path, "rb") as f:
                    reader = make_reader(f, decoder_factories=[DecoderFactory()])
                    for i, (_, _, _, ros_msg) in enumerate(
                        reader.iter_decoded_messages(topics=[topic])
                    ):
                        if i < start_frame:
                            continue
                        if len(result) >= count:
                            break
                        jpg = _ros_msg_to_jpeg_bytes(ros_msg)
                        if jpg:
                            result.append(jpg)
            return result
        except Exception as e:
            logger.warning("get_frames_batch error: %s", e)
            return []
