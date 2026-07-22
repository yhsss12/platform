import threading
import time
import re
import os
import cv2
import numpy as np
import logging
from typing import Dict, Optional, Generator, Tuple
from fastapi import HTTPException

# Configure logging
logger = logging.getLogger(__name__)

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from sensor_msgs.msg import Image, CompressedImage
    from cv_bridge import CvBridge, CvBridgeError
    RCLPY_AVAILABLE = True
except (ImportError, AttributeError, Exception):
    logger.warning("rclpy or cv_bridge not found (or incompatible). Camera streaming will be disabled.")
    RCLPY_AVAILABLE = False
    Node = object  # Mock base class
    Image = object  # 占位，避免类型注解在类定义时 NameError
    CompressedImage = object

class CameraStreamManager:
    """
    Manages ROS2 camera subscriptions and provides MJPEG streams.
    """
    def __init__(self):
        self.node: Optional[Node] = None
        self.bridge = None
        self.executor = None
        self.spin_thread = None
        self.active_topics: Dict[str, dict] = {}
        self.lock = threading.Lock()
        
        # Topic mapping will be populated dynamically
        self.topic_mapping = {}
        self.topic_types = {}

    def _raw_ros_image_to_bgr(self, msg) -> Optional[np.ndarray]:
        """sensor_msgs/Image -> BGR，按 step 行宽解码（兼容行填充）。"""
        enc = (getattr(msg, "encoding", "") or "").lower()
        h = int(getattr(msg, "height", 0) or 0)
        w = int(getattr(msg, "width", 0) or 0)
        if h <= 0 or w <= 0:
            return None
        step = int(getattr(msg, "step", 0) or 0)
        data = memoryview(msg.data)
        dtype = np.uint8

        if enc in ("16uc1", "32fc1"):
            return None

        if enc in ("mono8", "8uc1"):
            row_b = step if step > 0 else w
            need = h * row_b
            if len(data) < need:
                raise ValueError(f"mono8: buffer {len(data)} < need {need}")
            arr = np.frombuffer(data, dtype=dtype, count=need)
            plane = arr.reshape(h, row_b)[:, :w]
            return cv2.cvtColor(plane, cv2.COLOR_GRAY2BGR)

        if enc in ("bgra8", "rgba8"):
            n_c = 4
            row_b = step if step > 0 else w * n_c
            need = h * row_b
            if len(data) < need:
                raise ValueError(f"{enc}: buffer too small")
            arr = np.frombuffer(data, dtype=dtype, count=need)
            plane = arr.reshape(h, row_b)[:, : w * n_c].reshape(h, w, n_c)
            code = cv2.COLOR_RGBA2BGR if "rgba" in enc else cv2.COLOR_BGRA2BGR
            return cv2.cvtColor(plane, code)

        n_c = 3
        row_b = step if step > 0 else w * n_c
        need = h * row_b
        if len(data) < need:
            raise ValueError(f"{enc}: buffer {len(data)} < need {need}")
        arr = np.frombuffer(data, dtype=dtype, count=need)
        plane = arr.reshape(h, row_b)[:, : w * n_c].reshape(h, w, n_c)
        if enc == "rgb8":
            return cv2.cvtColor(plane, cv2.COLOR_RGB2BGR)
        return plane

    def start(self) -> bool:
        """Initialize and start the ROS2 node. Returns True if started, False if skipped/failed."""
        if not RCLPY_AVAILABLE:
            logger.warning("ROS2 相机流已跳过: rclpy 未安装（非 ROS 环境下可忽略）")
            return False

        if self.node is not None:
            return True  # Already running

        try:
            if not rclpy.ok():
                rclpy.init()
            
            self.node = rclpy.create_node("camera_stream_manager")
            self.bridge = CvBridge()

            # Discover topics before starting the spin loop
            self.refresh_topics()
            
            # Start spinning in a background thread
            self.spin_thread = threading.Thread(target=self._spin_node, daemon=True)
            self.spin_thread.start()
            
            logger.info("CameraStreamManager started.")
            
            # Subscribe to discovered topics
            for cam_id, topic in self.topic_mapping.items():
                self.subscribe(cam_id, topic)
            return True
        except Exception as e:
            logger.error(f"Failed to start CameraStreamManager: {e}")
            return False

    def refresh_topics(self):
        """Discover available image topics and map them to generic IDs."""
        try:
            # Get all topics and types
            topic_names_and_types = self.node.get_topic_names_and_types()
            
            topic_type_map: Dict[str, str] = {}

            def _has_msg_type(types: list, sub: str) -> bool:
                for t in types or []:
                    t = str(t)
                    if sub in t:
                        return True
                return False

            for name, types in topic_names_and_types:
                if _has_msg_type(types, "sensor_msgs/msg/Image"):
                    topic_type_map[name] = "raw"
                elif _has_msg_type(types, "sensor_msgs/msg/CompressedImage"):
                    topic_type_map[name] = "compressed"

            def color_image_stream_base(topic: str) -> Optional[str]:
                t = (topic or "").rstrip("/")
                lt = t.lower()
                if "/color/image_raw" not in lt:
                    return None
                if lt.endswith("/compressed") or "/image_raw/compressed" in lt:
                    try:
                        t = t[: t.rindex("/compressed")]
                    except ValueError:
                        return None
                tlt = t.lower()
                if tlt.endswith("/image_raw"):
                    return t[: -len("/image_raw")].rstrip("/") or None
                return None

            def is_orbbec_style_stream_base(base: str) -> bool:
                parts = [p for p in (base or "").strip("/").split("/") if p]
                if len(parts) < 2:
                    return False
                return str(parts[1]) == "camera" and re.match(r"^camera\d+$", str(parts[0]), re.I) is not None

            def extract_camera_id(topic: str) -> Optional[str]:
                s = (topic or "").strip()
                if not s.startswith("/"):
                    return None
                low = s.lower()
                m = re.search(r"^(/camera\d+)/color/", low) or re.search(
                    r"/(camera\d+)/color/", low
                )
                if m:
                    g = m.group(1) if m.lastindex else None
                    if g:
                        return g.lstrip("/")
                parts = s.strip("/").split("/")
                if parts and str(parts[0]).startswith("camera"):
                    return str(parts[0])
                return None

            def extract_camera_id_from_stream_base(base: str) -> Optional[str]:
                s = (base or "").strip()
                if not s.startswith("/"):
                    s = "/" + s
                return extract_camera_id(f"{s}/image_raw")

            def is_color_image_topic(topic: str) -> bool:
                t = (topic or "").lower()
                if "/color/image_raw" not in t:
                    return False
                if "compresseddepth" in t:
                    return False
                if "/depth/" in t:
                    return False
                if "/infra" in t or "/infra1" in t or "/infra2" in t:
                    return False
                return True

            by_stream: Dict[str, Dict[str, str]] = {}
            for topic, t_type in topic_type_map.items():
                if not is_color_image_topic(topic):
                    continue
                base = color_image_stream_base(topic)
                if not base:
                    continue
                st = by_stream.setdefault(base, {})
                low_topic = (topic or "").lower()
                is_compressed_topic = low_topic.endswith("/compressed") or "/color/image_raw/compressed" in low_topic
                if is_compressed_topic and t_type == "compressed":
                    st["compressed"] = topic
                elif t_type == "raw":
                    st.setdefault("raw", topic)

            streams_by_cam: Dict[str, list] = {}
            for sbase, st in by_stream.items():
                cid = extract_camera_id_from_stream_base(sbase)
                if not cid:
                    tpc = (st.get("raw") or st.get("compressed") or "").strip()
                    if not tpc:
                        continue
                    cid = extract_camera_id(tpc)
                if not cid:
                    continue
                streams_by_cam.setdefault(cid, []).append((sbase, st))

            # 与采集端一致：多基路径时 ORBBEC(/cameraN/camera/color) 优先于仅压缩的 RealSense 式路径
            new_mapping: Dict[str, str] = {}
            new_types: Dict[str, str] = {}
            for cam_id in sorted(streams_by_cam.keys()):
                cands = streams_by_cam[cam_id]
                with_raw = [(bs, s) for bs, s in cands if "raw" in s]
                only_c = [(bs, s) for bs, s in cands if "raw" not in s]
                sort_key = lambda it: (not is_orbbec_style_stream_base(it[0]), len(it[0]))
                chosen: Optional[Tuple[str, dict]] = None
                if with_raw:
                    with_raw.sort(key=sort_key)
                    chosen = with_raw[0]
                elif only_c:
                    only_c.sort(key=sort_key)
                    chosen = only_c[0]
                if not chosen:
                    continue
                _b, st0 = chosen
                if "raw" in st0:
                    new_mapping[cam_id] = st0["raw"]
                    new_types[cam_id] = "raw"
                else:
                    new_mapping[cam_id] = st0["compressed"]
                    new_types[cam_id] = "compressed"
            
            # Identify removed or changed cam_ids to unsubscribe
            with self.lock:
                for cam_id, old_topic in list(self.topic_mapping.items()):
                    if cam_id not in new_mapping or new_mapping[cam_id] != old_topic:
                        self.unsubscribe(cam_id)

                self.topic_mapping = new_mapping
                self.topic_types = new_types
                
            if self.topic_mapping:
                for cam_id, topic in self.topic_mapping.items():
                    t_type = self.topic_types.get(cam_id, 'unknown')
                    logger.info(f"Mapped {cam_id} -> {topic} ({t_type})")
            else:
                logger.warning("No camera topics found!")
                
        except Exception as e:
            logger.error(f"Failed to discover topics: {e}")

    def unsubscribe(self, cam_id: str):
        """Unsubscribe from a camera topic."""
        # Lock is expected to be held by caller or not needed if atomic enough, 
        # but safely we should check lock. 
        # However, this method might be called from within refresh_topics which holds lock.
        # Let's check if we need RLock or just be careful.
        # simpler to assume caller handles lock if calling from internal, 
        # but here we can check if we have it? No.
        # Let's make unsubscribe safe to call.
        if cam_id in self.active_topics:
            sub_info = self.active_topics.pop(cam_id)
            subscription = sub_info["subscription"]
            if self.node and subscription:
                self.node.destroy_subscription(subscription)
            logger.info(f"Unsubscribed from {cam_id}")

    def _discover_topics(self):
        """Deprecated: use refresh_topics instead."""
        self.refresh_topics()


    def _spin_node(self):
        """Background thread to process ROS2 callbacks."""
        try:
            rclpy.spin(self.node)
        except Exception as e:
            logger.error(f"Error in ROS2 spin loop: {e}")
        finally:
            if rclpy.ok():
                rclpy.shutdown()

    def subscribe(self, cam_id: str, topic_name: str):
        """Subscribe to a camera topic."""
        if not RCLPY_AVAILABLE or not self.node:
            return

        with self.lock:
            if cam_id in self.active_topics:
                logger.info(f"Already subscribed to {cam_id} ({topic_name})")
                return

            use_rel = str(os.environ.get("EAI_CAMERA_SUB_RELIABLE", "") or "").strip().lower() in (
                "1", "true", "yes", "on",
            )
            qos_profile = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE if use_rel else ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=10 if use_rel else 5,
            )
            
            # Determine message type
            msg_type = Image
            if getattr(self, 'topic_types', {}).get(cam_id) == 'compressed':
                msg_type = CompressedImage

            subscription = self.node.create_subscription(
                msg_type,
                topic_name,
                lambda msg, cid=cam_id: self._image_callback(msg, cid),
                qos_profile
            )
            
            self.active_topics[cam_id] = {
                "subscription": subscription,
                "latest_frame": None,
                "last_update": 0,
                "topic": topic_name
            }
            logger.info(f"Subscribed to {topic_name} as {cam_id} ({msg_type.__name__})")

    def _image_callback(self, msg, cam_id: str):
        """Process incoming image messages."""
        try:
            cv_image = None
            
            if getattr(msg, "encoding", None) is None and getattr(msg, "format", None) is not None:
                fmt = (getattr(msg, "format", "") or "").lower()
                if "compresseddepth" in fmt:
                    return
                try:
                    np_arr = np.frombuffer(msg.data, np.uint8)
                    cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                except Exception as e:
                    logger.warning(f"Failed to decode compressed image for {cam_id}: {e}")
                    return
            else:
                cv_image = self._raw_ros_image_to_bgr(msg)

            if cv_image is None:
                return

            # Encode as JPEG
            ret, buffer = cv2.imencode('.jpg', cv_image)
            if ret:
                with self.lock:
                    if cam_id in self.active_topics:
                        self.active_topics[cam_id]["latest_frame"] = buffer.tobytes()
                        self.active_topics[cam_id]["last_update"] = time.time()
        except Exception as e:
            logger.error(f"Error processing image for {cam_id}: {e}")

    def get_frame_generator(self, cam_id: str) -> Generator[bytes, None, None]:
        """Yield MJPEG frames for a specific camera."""
        if not RCLPY_AVAILABLE:
            yield self._get_placeholder_frame("ROS2 Not Available")
            return

        if cam_id not in self.active_topics:
            # Try to subscribe if it's a known camera
            if cam_id in self.topic_mapping:
                self.subscribe(cam_id, self.topic_mapping[cam_id])
            else:
                yield self._get_placeholder_frame(f"Unknown Camera: {cam_id}")
                return

        while True:
            frame_data = None
            with self.lock:
                if cam_id in self.active_topics:
                    frame_data = self.active_topics[cam_id].get("latest_frame")
            
            if frame_data:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
            else:
                # Send a waiting placeholder or just wait
                # To avoid spamming, sleep a bit
                pass
            
            time.sleep(0.05)  # ~20 FPS cap

    def _get_placeholder_frame(self, text: str) -> bytes:
        """Generate a placeholder image with text."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(img, text, (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (255, 255, 255), 2, cv2.LINE_AA)
        ret, buffer = cv2.imencode('.jpg', img)
        if ret:
            return (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        return b''

# Global instance
stream_manager = CameraStreamManager()
