from __future__ import annotations

import queue
import threading
import time
import logging
import os
import re
from typing import Dict, Optional, Generator, Any, Tuple

logger = logging.getLogger(__name__)

try:
    import cv2
    import numpy as np
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from sensor_msgs.msg import Image, CompressedImage
    RCLPY_AVAILABLE = True
except Exception:
    logger.warning("ROS2/cv2 依赖缺失或不兼容，采集端视频流将不可用。")
    RCLPY_AVAILABLE = False
    Node = object  # type: ignore
    Image = object  # type: ignore
    CompressedImage = object  # type: ignore


class CameraStreamManager:
    """
    采集端 ROS2 相机订阅与 MJPEG 输出。

    逻辑基本与平台端的 CameraStreamManager 一致，但作为 Agent 的本机能力运行。
    """

    def __init__(self):
        self.node: Optional[Node] = None
        self.spin_thread: Optional[threading.Thread] = None
        self.watchdog_thread: Optional[threading.Thread] = None
        self._watchdog_stop = threading.Event()
        self.active_topics: Dict[str, dict] = {}
        self.lock = threading.Lock()
        self.topic_mapping: Dict[str, str] = {}
        self.topic_types: Dict[str, str] = {}
        # 关节 / 末端力矩 DDS 订阅（与相机共用同一 Node + spin）
        self.robot_state_bridge = None
        self._placeholder_cache: Dict[str, bytes] = {}
        # DDS 回调仅入队；解码/JPEG 在工作线程执行，减轻 spin 线程占用（缓解与 bag 录制争用）
        self._frame_queue: queue.Queue = queue.Queue(
            maxsize=max(1, int(os.environ.get("EAI_AGENT_PREVIEW_QUEUE_MAX", "2") or "2"))
        )
        self._frame_worker_stop = threading.Event()
        self._frame_worker_thread: Optional[threading.Thread] = None
        self._last_preview_encode_ts: Dict[str, float] = {}
        fluid = str(os.environ.get("EAI_AGENT_PREVIEW_FLUID", "") or "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self._preview_fluid = fluid
        min_iv_env = (os.environ.get("EAI_AGENT_PREVIEW_MIN_INTERVAL_SEC") or "").strip()
        max_fps_env = (os.environ.get("EAI_AGENT_PREVIEW_MAX_FPS") or "").strip()
        if min_iv_env:
            self._preview_min_interval_sec = max(0.0, float(min_iv_env))
        else:
            if fluid and not max_fps_env:
                mx = 30.0
            else:
                mx = float(max_fps_env or "12")
            if mx <= 0:
                self._preview_min_interval_sec = 0.0
            else:
                mx = max(1.0, min(mx, 60.0))
                self._preview_min_interval_sec = 1.0 / mx

        jq_raw = (os.environ.get("EAI_AGENT_PREVIEW_JPEG_QUALITY") or "").strip()
        if jq_raw:
            q = int(float(jq_raw))
            self._preview_jpeg_quality: Optional[int] = None if q <= 0 else max(1, min(q, 100))
        else:
            self._preview_jpeg_quality = 55 if fluid else None

        me_raw = (os.environ.get("EAI_AGENT_PREVIEW_MAX_EDGE") or "").strip()
        if me_raw:
            self._preview_max_edge = max(0, int(float(me_raw)))
        else:
            self._preview_max_edge = 640 if fluid else 0

    def _raw_bytes_to_bgr(self, enc: str, h: int, w: int, step: int, data: bytes) -> Optional[np.ndarray]:
        """将原始 Image 缓冲区转为 BGR（与 _raw_ros_image_to_bgr 算法一致，供工作线程使用）。"""
        enc = (enc or "").lower()
        if h <= 0 or w <= 0:
            return None
        step = int(step or 0)
        dtype = np.uint8

        if enc in ("16uc1", "32fc1"):
            return None

        if enc in ("mono8", "8uc1"):
            row_b = step if step > 0 else w
            need = h * row_b
            if len(data) < need:
                raise ValueError(f"mono8: buffer {len(data)} < need {need} (step={row_b})")
            arr = np.frombuffer(data, dtype=dtype, count=need)
            plane = arr.reshape(h, row_b)[:, :w]
            return cv2.cvtColor(plane, cv2.COLOR_GRAY2BGR)

        if enc in ("bgra8", "rgba8"):
            n_c = 4
            row_b = step if step > 0 else w * n_c
            need = h * row_b
            if len(data) < need:
                raise ValueError(f"{enc}: buffer {len(data)} < need {need}")
            arr = np.frombuffer(data, dtype=dtype, count=need)
            plane = arr.reshape(h, row_b)[:, : w * n_c].reshape(h, w, n_c)
            code = cv2.COLOR_RGBA2BGR if "rgba" in enc else cv2.COLOR_BGRA2BGR
            return cv2.cvtColor(plane, code)

        # rgb8 / bgr8 / 8uc3 等常见彩色
        n_c = 3
        row_b = step if step > 0 else w * n_c
        need = h * row_b
        if len(data) < need:
            raise ValueError(f"{enc}: buffer {len(data)} < need {need} (step={row_b})")
        arr = np.frombuffer(data, dtype=dtype, count=need)
        plane = arr.reshape(h, row_b)[:, : w * n_c].reshape(h, w, n_c)
        if enc == "rgb8":
            return cv2.cvtColor(plane, cv2.COLOR_RGB2BGR)
        # bgr8 / yuv422 等未覆盖格式：尽量按 BGR 解释
        return plane

    def _downscale_for_preview(self, img: Any) -> Any:
        """限制长边像素，降低编码与带宽开销（牺牲清晰度换流畅度）。"""
        max_edge = int(getattr(self, "_preview_max_edge", 0) or 0)
        if max_edge <= 0 or not RCLPY_AVAILABLE:
            return img
        h, w = int(img.shape[0]), int(img.shape[1])
        if h <= 0 or w <= 0:
            return img
        long_side = max(h, w)
        if long_side <= max_edge:
            return img
        scale = max_edge / float(long_side)
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    def start(self) -> bool:
        if not RCLPY_AVAILABLE:
            return False
        if self.node is not None:
            return True

        try:
            if not rclpy.ok():
                rclpy.init()
            self.node = rclpy.create_node("agent_camera_stream_manager")
            self.refresh_topics()

            self._frame_worker_stop.clear()
            self._frame_worker_thread = threading.Thread(target=self._frame_worker_loop, daemon=True)
            self._frame_worker_thread.start()

            self.spin_thread = threading.Thread(target=self._spin_node, daemon=True)
            self.spin_thread.start()

            for cam_id, topic in self.topic_mapping.items():
                self.subscribe(cam_id, topic)

            self._watchdog_stop.clear()
            self.watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
            self.watchdog_thread.start()

            try:
                from ros2_robot_state_bridge import RCLPY_STATE_AVAILABLE, RobotStateBridge

                if RCLPY_STATE_AVAILABLE:
                    self.robot_state_bridge = RobotStateBridge(self.node)
                    self.robot_state_bridge.start()
            except Exception as e:
                logger.warning("RobotStateBridge 未启动（将回退 ros2 topic echo）: %s", e)
                self.robot_state_bridge = None

            return True
        except Exception as e:
            logger.error("Failed to start CameraStreamManager: %s", e)
            return False

    def _spin_node(self):
        try:
            assert self.node is not None
            rclpy.spin(self.node)
        except Exception as e:
            logger.error("Error in ROS2 spin loop: %s", e)
        finally:
            try:
                if rclpy.ok():
                    rclpy.shutdown()
            except Exception:
                pass

    def refresh_topics(self):
        if not RCLPY_AVAILABLE or not self.node:
            print("[agent][stream] refresh_topics: RCLPY_AVAILABLE=", RCLPY_AVAILABLE, " node=", self.node)
            return

        try:
            topic_names_and_types = self.node.get_topic_names_and_types()
            print("[agent][stream] refresh_topics: topics from ROS2:")
            topic_type_map: Dict[str, str] = {}
            for name, types in topic_names_and_types:
                print(f"  - {name}: {types}")
                if any("sensor_msgs/msg/Image" in t for t in types):
                    topic_type_map[name] = "raw"
                elif any("sensor_msgs/msg/CompressedImage" in t for t in types):
                    topic_type_map[name] = "compressed"

            def extract_camera_id(topic: str) -> Optional[str]:
                """
                从 topic 推断 camera_id（短名，如 camera1），兼容 RealSense 常见「双段」命名：
                /camera1/camera1/color/image_raw
                /camera1/camera1/color/image_raw/compressed
                """
                s = (topic or "").strip()
                if not s.startswith("/"):
                    return None
                low = s.lower()
                m = re.search(r"/(camera\d+)/color/", low)
                if m:
                    return m.group(1)
                parts = s.strip("/").split("/")
                if parts and str(parts[0]).startswith("camera"):
                    return str(parts[0])
                return None

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

            by_camera: Dict[str, Dict[str, str]] = {}
            for topic, t_type in topic_type_map.items():
                if not is_color_image_topic(topic):
                    continue
                cam_id = extract_camera_id(topic)
                if not cam_id:
                    continue
                streams = by_camera.setdefault(cam_id, {})
                low_topic = (topic or "").lower()
                is_compressed_topic = low_topic.endswith("/compressed") or "/color/image_raw/compressed" in low_topic
                if is_compressed_topic and t_type == "compressed":
                    streams["compressed"] = topic
                elif t_type == "raw":
                    streams.setdefault("raw", topic)

            new_mapping: Dict[str, str] = {}
            new_types: Dict[str, str] = {}
            for cam_id in sorted(by_camera.keys()):
                streams = by_camera[cam_id]
                if "compressed" in streams:
                    new_mapping[cam_id] = streams["compressed"]
                    new_types[cam_id] = "compressed"
                elif "raw" in streams:
                    new_mapping[cam_id] = streams["raw"]
                    new_types[cam_id] = "raw"

            with self.lock:
                # 简化：不做复杂 unsubscribe，只更新映射；订阅时按需创建
                self.topic_mapping = new_mapping
                self.topic_types = new_types

            print("[agent][stream] refresh_topics: new_mapping =", self.topic_mapping)
            print("[agent][stream] refresh_topics: new_types   =", self.topic_types)
            # 发现新相机或话题切换后，立即对齐订阅（否则仅 start() 时订过一次，后续图变化永远不订）
            self._ensure_subscriptions_match_mapping()
        except Exception as e:
            logger.error("Failed to discover topics: %s", e)

    def _ensure_subscriptions_match_mapping(self) -> None:
        """按当前 topic_mapping 创建缺失订阅；已订阅但话题名变化则重建订阅。"""
        if not RCLPY_AVAILABLE or not self.node:
            return
        try:
            with self.lock:
                mapping = dict(self.topic_mapping)
            for cam_id, topic in mapping.items():
                if not topic:
                    continue
                with self.lock:
                    st = self.active_topics.get(cam_id)
                    cur = str(st.get("topic") or "") if isinstance(st, dict) else ""
                if cam_id not in self.active_topics:
                    self.subscribe(cam_id, topic)
                elif cur and cur != topic:
                    self._resubscribe(cam_id)
        except Exception as e:
            logger.warning("_ensure_subscriptions_match_mapping: %s", e)

    def subscribe(self, cam_id: str, topic_name: str):
        if not RCLPY_AVAILABLE or not self.node:
            return

        with self.lock:
            if cam_id in self.active_topics:
                return

            # 默认 BEST_EFFORT 与多数相机节点兼容；若仍无帧可设 EAI_CAMERA_SUB_RELIABLE=1 尝试 RELIABLE（与部分工业相机匹配）
            use_rel = str(os.environ.get("EAI_CAMERA_SUB_RELIABLE", "") or "").strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
            qos_profile = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE if use_rel else ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=10 if use_rel else 5,
            )

            msg_type = Image
            if self.topic_types.get(cam_id) == "compressed":
                msg_type = CompressedImage

            subscription = self.node.create_subscription(
                msg_type,
                topic_name,
                lambda msg, cid=cam_id: self._image_callback(msg, cid),
                qos_profile,
            )

            self.active_topics[cam_id] = {
                "subscription": subscription,
                "latest_frame": None,
                "last_update": 0,
                "subscribed_at": time.time(),
                "topic": topic_name,
                "frames": 0,
                "decode_errors": 0,
                "last_error": "",
                "last_error_ts": 0.0,
                "restart_count": 0,
            }

    def _raw_ros_image_to_bgr(self, msg: Any) -> Optional[np.ndarray]:
        """将 sensor_msgs/Image 转为 BGR（兼容外部调用）。"""
        enc = (getattr(msg, "encoding", "") or "").lower()
        h = int(getattr(msg, "height", 0) or 0)
        w = int(getattr(msg, "width", 0) or 0)
        step = int(getattr(msg, "step", 0) or 0)
        try:
            blob = bytes(memoryview(msg.data))
        except Exception:
            blob = bytes(msg.data)
        return self._raw_bytes_to_bgr(enc, h, w, step, blob)

    def _enqueue_preview_frame(self, cam_id: str, kind: str, payload: Tuple[Any, ...]) -> None:
        """非阻塞入队；队列满时丢弃最旧一条再放入，优先保留较新帧。"""
        item = (cam_id, kind, payload)
        try:
            self._frame_queue.put_nowait(item)
        except queue.Full:
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._frame_queue.put_nowait(item)
            except queue.Full:
                pass

    def _image_callback(self, msg, cam_id: str):
        """仅拷贝消息负载并入队，避免在 DDS 回调内解码/JPEG（降低 spin 线程占用）。"""
        try:
            if not RCLPY_AVAILABLE:
                return

            if getattr(msg, "encoding", None) is None and getattr(msg, "format", None) is not None:
                fmt = (getattr(msg, "format", "") or "").lower()
                if "compresseddepth" in fmt:
                    return
                try:
                    blob = bytes(memoryview(msg.data))
                except Exception:
                    blob = bytes(msg.data)
                self._enqueue_preview_frame(cam_id, "compressed", (fmt, blob))
            else:
                enc = (getattr(msg, "encoding", "") or "").lower()
                h = int(getattr(msg, "height", 0) or 0)
                w = int(getattr(msg, "width", 0) or 0)
                step = int(getattr(msg, "step", 0) or 0)
                try:
                    blob = bytes(memoryview(msg.data))
                except Exception:
                    blob = bytes(msg.data)
                self._enqueue_preview_frame(cam_id, "raw", (enc, h, w, step, blob))
        except Exception as e:
            with self.lock:
                if cam_id in self.active_topics:
                    self.active_topics[cam_id]["decode_errors"] = int(self.active_topics[cam_id].get("decode_errors") or 0) + 1
                    self.active_topics[cam_id]["last_error"] = str(e)[:200]
                    self.active_topics[cam_id]["last_error_ts"] = time.time()
            logger.error("Error enqueueing image for %s: %s", cam_id, e)

    def _frame_worker_loop(self) -> None:
        while not self._frame_worker_stop.is_set():
            try:
                item = self._frame_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            cam_id, kind, payload = item
            now = time.time()
            if self._preview_min_interval_sec > 0:
                last = self._last_preview_encode_ts.get(cam_id, 0.0)
                if now - last < self._preview_min_interval_sec:
                    continue
            try:
                cv_image = None
                if kind == "compressed":
                    fmt, blob = payload
                    if "compresseddepth" in (fmt or "").lower():
                        continue
                    np_arr = np.frombuffer(blob, np.uint8)
                    cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                else:
                    enc, h, w, step, blob = payload
                    cv_image = self._raw_bytes_to_bgr(enc, int(h), int(w), int(step), blob)

                if cv_image is None:
                    continue

                cv_image = self._downscale_for_preview(cv_image)
                enc_params: list[int] = []
                q = getattr(self, "_preview_jpeg_quality", None)
                if q is not None:
                    enc_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(q)]
                if enc_params:
                    ret, buffer = cv2.imencode(".jpg", cv_image, enc_params)
                else:
                    ret, buffer = cv2.imencode(".jpg", cv_image)
                if not ret:
                    continue
                ts = time.time()
                self._last_preview_encode_ts[cam_id] = ts
                with self.lock:
                    if cam_id in self.active_topics:
                        self.active_topics[cam_id]["latest_frame"] = buffer.tobytes()
                        self.active_topics[cam_id]["last_update"] = ts
                        self.active_topics[cam_id]["frames"] = int(self.active_topics[cam_id].get("frames") or 0) + 1
            except Exception as e:
                with self.lock:
                    if cam_id in self.active_topics:
                        self.active_topics[cam_id]["decode_errors"] = int(self.active_topics[cam_id].get("decode_errors") or 0) + 1
                        self.active_topics[cam_id]["last_error"] = str(e)[:200]
                        self.active_topics[cam_id]["last_error_ts"] = time.time()
                logger.error("Error processing preview frame for %s: %s", cam_id, e)

    def _make_placeholder_jpeg(self, text: str) -> bytes:
        key = (text or "").strip()[:120] or "NO FRAME"
        with self.lock:
            hit = self._placeholder_cache.get(key)
        if hit:
            return hit
        fallback = (
            b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00"
            b"\xff\xdb\x00C\x00"
            b"\x10\x0b\x0c\x0e\x0c\n\x10\x0e\r\x0e\x12\x11\x10\x13\x18(\x1a\x18\x16\x16\x181#%\x1d(=3?=:3"
            b"7;8@H\\N@DWE78PmQW_bghg>Mqypdx\\egc"
            b"\xff\xc0\x00\x11\x08\x00\x01\x00\x01\x03\x01\"\x00\x02\x11\x01\x03\x11\x01"
            b"\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b"
            b"\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05\x04\x04\x00\x00\x01}\x01\x02\x03\x00\x04\x11\x05\x12!1A\x06\x13Qa\x07\"q\x142\x81\x91\xa1\x08#B\xb1\xc1\x15R\xd1\xf0$3br\x82\t\n\x16\x17\x18\x19\x1a%&'()*456789:CDEFGHIJSTUVWXYZcdefghijstuvwxyz\x83\x84\x85\x86\x87\x88\x89\x8a\x92\x93\x94\x95\x96\x97\x98\x99\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa"
            b"\xff\xda\x00\x0c\x03\x01\x00\x02\x11\x03\x11\x00?\x00\xd2\xcf \xff\xd9"
        )
        try:
            img = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.putText(img, key, (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            ok, buf = cv2.imencode(".jpg", img)
            out = buf.tobytes() if ok and buf is not None else fallback
        except Exception:
            out = fallback
        with self.lock:
            if len(self._placeholder_cache) > 32:
                self._placeholder_cache.clear()
            self._placeholder_cache[key] = out
        return out

    def _resubscribe(self, cam_id: str) -> None:
        if not RCLPY_AVAILABLE or not self.node:
            return
        with self.lock:
            st = self.active_topics.get(cam_id)
            topic = self.topic_mapping.get(cam_id)
            if not topic and isinstance(st, dict):
                topic = st.get("topic")
            sub = st.get("subscription") if isinstance(st, dict) else None
        if not topic:
            return
        prev_restarts = 0
        try:
            if sub is not None:
                try:
                    self.node.destroy_subscription(sub)
                except Exception:
                    pass
            with self.lock:
                if cam_id in self.active_topics:
                    prev_restarts = int(self.active_topics[cam_id].get("restart_count") or 0)
                    # 必须移除条目，否则 subscribe() 因「已在 active_topics」直接返回，订阅不会被重建
                    del self.active_topics[cam_id]
        except Exception:
            pass
        try:
            self.subscribe(cam_id, topic)
            with self.lock:
                if cam_id in self.active_topics:
                    self.active_topics[cam_id]["restart_count"] = prev_restarts + 1
        except Exception:
            pass

    def _watchdog_loop(self) -> None:
        stall_sec = float(os.environ.get("EAI_STREAM_STALL_SEC", "4") or "4")
        interval_sec = float(os.environ.get("EAI_STREAM_WATCHDOG_INTERVAL_SEC", "2") or "2")
        while not self._watchdog_stop.is_set():
            try:
                now = time.time()
                with self.lock:
                    items = [(cid, dict(st)) for cid, st in self.active_topics.items()]
                # 不在此线程调用 refresh_topics()/get_topic_names_and_types：与 rclpy spin 线程并发会搞坏 Node，
                # 曾导致多路相机订阅异常、平台侧全部黑屏。卡死时仅用当前 mapping 重建订阅即可。
                stalled: list[str] = []
                for cid, st in items:
                    last = float(st.get("last_update") or 0)
                    if last > 0 and (now - last) >= stall_sec:
                        stalled.append(cid)
                for cid in stalled:
                    self._resubscribe(cid)
            except Exception:
                pass
            self._watchdog_stop.wait(interval_sec)

    def get_status(self) -> dict:
        now = time.time()
        with self.lock:
            mapping = dict(self.topic_mapping)
            types = dict(self.topic_types)
            active = {cid: dict(st) for cid, st in self.active_topics.items()}
        cams = []
        for cid in sorted(set(mapping.keys()) | set(active.keys())):
            st = active.get(cid, {})
            last = float(st.get("last_update") or 0)
            age = now - last if last > 0 else None
            cams.append(
                {
                    "camera_id": cid,
                    "topic": mapping.get(cid) or st.get("topic") or "",
                    "topic_type": types.get(cid) or "",
                    "last_update_ts": last,
                    "age_sec": age,
                    "frames": int(st.get("frames") or 0),
                    "decode_errors": int(st.get("decode_errors") or 0),
                    "restart_count": int(st.get("restart_count") or 0),
                    "last_error": str(st.get("last_error") or ""),
                    "last_error_ts": float(st.get("last_error_ts") or 0),
                }
            )
        return {
            "ok": True,
            "rclpy_available": bool(RCLPY_AVAILABLE),
            "node_ready": self.node is not None,
            "camera_count": len(cams),
            "cameras": cams,
        }

    def get_frame_generator(self, cam_id: str) -> Generator[bytes, None, None]:
        if not RCLPY_AVAILABLE:
            return self._placeholder_generator("ROS2 Not Available")

        # 按需 refresh + subscribe
        if cam_id not in self.active_topics:
            self.refresh_topics()
            topic = self.topic_mapping.get(cam_id)
            if topic:
                self.subscribe(cam_id, topic)
            else:
                return self._placeholder_generator(f"Unknown Camera: {cam_id}")

        while True:
            frame_data = None
            last_update = 0.0
            with self.lock:
                if cam_id in self.active_topics:
                    frame_data = self.active_topics[cam_id].get("latest_frame")
                    last_update = float(self.active_topics[cam_id].get("last_update") or 0)

            if frame_data:
                payload = frame_data
            else:
                payload = self._make_placeholder_jpeg(f"NO FRAME | {cam_id}")
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                + f"Content-Length: {len(payload)}\r\n".encode("ascii")
                + b"\r\n"
                + payload
                + b"\r\n"
            )

            if frame_data and last_update > 0 and time.time() - last_update < 1.0:
                time.sleep(0.05)
            else:
                time.sleep(0.2)

    def _placeholder_generator(self, text: str) -> Generator[bytes, None, None]:
        if not RCLPY_AVAILABLE:
            # 生成一个最简空帧
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n\r\n"
            return
        import cv2  # local import
        import numpy as np  # local import

        while True:
            img = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.putText(img, text, (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            ret, buffer = cv2.imencode(".jpg", img)
            if ret:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
                )
            time.sleep(0.2)


stream_manager = CameraStreamManager()
