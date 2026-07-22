from __future__ import annotations

import asyncio
import json
import logging
import re
import struct
import time
from typing import Any, Dict, Optional, Tuple, List

# 隧道出站命令优先级（数值越小越先发送）；WebRTC 信令默认走 HTTP，隧道内仍作兜底
_TUNNEL_CMD_PRIORITY: Dict[str, int] = {
    "COLLECT_START": 0,
    "COLLECT_STOP": 0,
    "COLLECT_PAUSE": 0,
    "COLLECT_RESUME": 0,
    "SCRIPT_DELETE_DATA": 1,
    "SCRIPT_START": 2,
    "SCRIPT_STOP": 2,
    "FS_LIST": 3,
    "FS_STAT": 3,
    "SCAN_COLLECT_SCRIPT": 3,
    "FS_READ": 4,
    "WEBRTC_OFFER": 9,
}
_DEFAULT_TUNNEL_CMD_PRIORITY = 5
from uuid import uuid4

from fastapi import WebSocket

from app.core.config import settings

from .experiment_logger import log_experiment_event
from .script_runner import script_runner
from .telemetry_file_logger import log_telemetry_event

logger = logging.getLogger(__name__)

CHUNK_SIZE = 32 * 1024  # 32768
MJPEG_CHUNK_KIND = 1
MJPEG_CHUNK_MAGIC = b"EAI1"
MJPEG_CHUNK_VER = 1
FRAME_ASSEMBLY_TIMEOUT_MS = 1000
MAX_MJPEG_CHUNKS_PER_FRAME = 512
# 全局同时在组帧的 (agent,cam) 上限（文档 §7.6B）；超出则丢弃新开帧的 chunk
MAX_MJPEG_CONCURRENT_ASSEMBLIES = 512

# 统一 MAC 写法，避免 sysfs「小写冒号」与配置「大写/无分隔符」导致隧道字典键不一致
_MAC_6_GROUPS = re.compile(
    r"^([0-9a-fA-F]{2})[:-]([0-9a-fA-F]{2})[:-]([0-9a-fA-F]{2})[:-]"
    r"([0-9a-fA-F]{2})[:-]([0-9a-fA-F]{2})[:-]([0-9a-fA-F]{2})$"
)


def normalize_mac_like_agent_id(agent_id: str) -> str:
    """将常见 MAC 写法统一为小写冒号；非 MAC 字符串原样返回（仅 strip）。"""
    s = (agent_id or "").strip()
    if not s:
        return s
    if re.fullmatch(r"[0-9a-fA-F]{12}", s):
        h = s.lower()
        return ":".join(h[i : i + 2] for i in range(0, 12, 2))
    m = _MAC_6_GROUPS.match(s)
    if m:
        return ":".join(x.lower() for x in m.groups())
    return s


def is_colon_mac_normalized(s: str) -> bool:
    return bool(re.fullmatch(r"([0-9a-f]{2}:){5}[0-9a-f]{2}", (s or "").strip()))


def _make_mjpeg_placeholder_jpeg_bytes(camera_id: str) -> bytes:
    """
    Generate a small placeholder JPEG for MJPEG multipart stream.
    Avoids depending on ROS; relies on OpenCV if available.
    """
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
        import cv2  # type: ignore
        import numpy as np  # type: ignore

        img = np.zeros((360, 640, 3), dtype=np.uint8)
        text = f"NO FRAME: {camera_id}"
        cv2.putText(img, text, (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        ok, buf = cv2.imencode(".jpg", img)
        if not ok:
            return fallback
        jpeg = buf.tobytes()
        return jpeg if jpeg else fallback
    except Exception:
        return fallback


class AgentTunnelManager:
    """
    Agent -> Platform WebSocket tunnel:
    - Control: CMD_REQUEST / CMD_ACK / CMD_RESULT
    - Log: LOG -> script_runner.broadcast
    """

    def __init__(self) -> None:
        # agent_id -> websocket
        self._connections: Dict[str, WebSocket] = {}
        self._connections_lock = asyncio.Lock()
        self._send_locks: Dict[str, asyncio.Lock] = {}
        # 每 Agent 出站优先级队列（控制命令优先于 WebRTC 等低优先级信令）
        self._send_queues: Dict[str, asyncio.PriorityQueue] = {}
        self._send_queue_seq: Dict[str, int] = {}
        self._send_workers: Dict[str, asyncio.Task] = {}

        # command_id -> future(result_payload)
        self._pending: Dict[str, asyncio.Future] = {}
        self._pending_lock = asyncio.Lock()
        self._command_states: Dict[str, Dict[str, Any]] = {}
        self._command_state_order: List[str] = []
        self._max_command_state = 500

        # Best-effort last seen
        self._last_seen_ts: Dict[str, float] = {}
        self._last_heartbeat_payload: Dict[str, Dict[str, Any]] = {}
        # 网络/心跳质量指标（按 agent_id 维护）
        self._last_heartbeat_recv_ts_ms: Dict[str, int] = {}
        self._heartbeat_expected_interval_ms: Dict[str, float] = {}
        self._heartbeat_missed_intervals_total: Dict[str, int] = {}

        # MJPEG tunnel streaming state:
        # - per agent_id: camera_id <-> cam_idx mapping
        self._camera_id_to_cam_idx: Dict[str, Dict[str, int]] = {}
        self._cam_idx_to_camera_id: Dict[str, Dict[int, str]] = {}

        # (agent_id, cam_idx) -> latest jpeg bytes
        self._latest_jpeg: Dict[Tuple[str, int], bytes] = {}
        # (agent_id, cam_idx) -> seq counter
        self._stream_seq: Dict[Tuple[str, int], int] = {}
        # (agent_id, cam_idx) -> condition for generator waiting
        self._stream_conditions: Dict[Tuple[str, int], asyncio.Condition] = {}

        # (agent_id, cam_idx) -> frame assembly buffer state
        self._frame_states: Dict[Tuple[str, int], Dict[str, Any]] = {}

        # 观测指标（文档 §11）
        self._metrics: Dict[str, int] = {
            "mjpeg_chunks_received": 0,
            "mjpeg_frames_completed": 0,
            "mjpeg_frames_timeout": 0,
            "mjpeg_frames_stale": 0,
            "mjpeg_chunks_malformed": 0,
            "mjpeg_assembly_budget_drops": 0,
            "webrtc_offer_ok": 0,
            "webrtc_offer_fail": 0,
            "cmd_requests_sent": 0,
            "cmd_result_ok": 0,
            "cmd_result_error": 0,
            "cmd_timeout": 0,
            "cmd_retries": 0,
        }

    def get_metrics(self) -> Dict[str, Any]:
        m = dict(self._metrics)
        recv = max(1, int(m.get("mjpeg_chunks_received", 0) or 0))
        stale = int(m.get("mjpeg_frames_stale", 0) or 0)
        to = int(m.get("mjpeg_frames_timeout", 0) or 0)
        bd = int(m.get("mjpeg_assembly_budget_drops", 0) or 0)
        m["mjpeg_drop_hints_per_1k_chunks"] = round((stale + to + bd) * 1000.0 / recv, 3)
        return {
            **m,
            "connected_agents": len(self._connections),
            "pending_commands": len(self._pending),
            "mjpeg_concurrent_assemblies": self._mjpeg_assembling_count(),
        }

    def record_webrtc_offer(
        self,
        *,
        ok: bool,
        route: Optional[str] = None,
        duration_ms: Optional[float] = None,
        agent_id: Optional[str] = None,
        device_id: Optional[int] = None,
        error: Optional[str] = None,
    ) -> None:
        if ok:
            self._metrics["webrtc_offer_ok"] = self._metrics.get("webrtc_offer_ok", 0) + 1
        else:
            self._metrics["webrtc_offer_fail"] = self._metrics.get("webrtc_offer_fail", 0) + 1
        try:
            log_telemetry_event(
                category="preview",
                event="webrtc_offer",
                ok=ok,
                route=route,
                duration_ms=round(duration_ms, 2) if duration_ms is not None else None,
                agent_id=agent_id,
                device_id=device_id,
                error=(error[:400] if error else None),
            )
        except Exception:
            pass

    def _tunnel_token_ok(self, *, agent_id: str, tunnel_token: Optional[str]) -> bool:
        raw = (getattr(settings, "AGENT_TUNNEL_TOKEN_BY_AGENT_JSON", None) or "").strip()
        if raw:
            try:
                mapping = json.loads(raw)
            except Exception:
                return False
            if not isinstance(mapping, dict):
                return False
            expected = mapping.get(agent_id)
            if expected is None:
                return False
            return (tunnel_token or "").strip() == str(expected).strip()
        expected = (settings.AGENT_TUNNEL_TOKEN or "").strip()
        if expected:
            return (tunnel_token or "").strip() == expected
        return True

    def _mjpeg_assembling_count(self) -> int:
        n = 0
        for st in self._frame_states.values():
            if st.get("assembling_frame_id") is None:
                continue
            chunks = st.get("chunks")
            if not isinstance(chunks, list):
                continue
            cc = int(st.get("chunk_count") or 0)
            if cc <= 0:
                continue
            got = sum(1 for c in chunks if c is not None)
            if got < cc:
                n += 1
        return n

    def get_command_states(
        self,
        *,
        limit: int = 100,
        agent_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        max_items = max(1, min(int(limit), self._max_command_state))
        ordered = list(reversed(self._command_state_order))
        out: List[Dict[str, Any]] = []
        for cid in ordered:
            st = self._command_states.get(cid)
            if not isinstance(st, dict):
                continue
            if agent_id and st.get("agent_id") != agent_id:
                continue
            out.append(dict(st))
            if len(out) >= max_items:
                break
        return out

    def _upsert_command_state(self, command_id: str, patch: Dict[str, Any]) -> None:
        now_ms = int(time.time() * 1000)
        st = self._command_states.get(command_id)
        if not isinstance(st, dict):
            st = {
                "command_id": command_id,
                "status": "pending",
                "created_ts_ms": now_ms,
                "updated_ts_ms": now_ms,
            }
            self._command_states[command_id] = st
            self._command_state_order.append(command_id)
            if len(self._command_state_order) > self._max_command_state:
                old_id = self._command_state_order.pop(0)
                self._command_states.pop(old_id, None)
        st.update(patch)
        st["updated_ts_ms"] = now_ms

    def _telemetry_log_control_command(
        self,
        command_id: str,
        *,
        outcome: str,
        result_payload: Optional[Dict[str, Any]] = None,
        timeout_sec: Optional[float] = None,
        send_error: Optional[str] = None,
    ) -> None:
        """控制命令闭环埋点（独立 JSONL，与实验埋点解耦）。"""
        try:
            st = self._command_states.get(str(command_id), {}) or {}
            if not isinstance(st, dict):
                st = {}
            first_send = st.get("first_send_ts_ms") or st.get("last_send_ts_ms")
            ack_ts = st.get("ack_ts_ms")
            now_ms = int(time.time() * 1000)
            req_to_result: Optional[int] = None
            if first_send:
                req_to_result = now_ms - int(first_send)
            req_to_ack: Optional[int] = None
            if first_send and ack_ts:
                req_to_ack = int(ack_ts) - int(first_send)
            ack_to_result: Optional[int] = None
            if ack_ts:
                ack_to_result = now_ms - int(ack_ts)
            ok: Optional[bool] = None
            msg: Optional[str] = None
            dup: Optional[bool] = None
            if isinstance(result_payload, dict):
                ok = bool(result_payload.get("success", True))
                raw_msg = result_payload.get("msg") or result_payload.get("message")
                msg = str(raw_msg)[:300] if raw_msg is not None else None
                if "duplicate" in result_payload:
                    dup = bool(result_payload.get("duplicate"))
            log_telemetry_event(
                category="tunnel",
                event="control_command_complete",
                command_id=str(command_id),
                agent_id=st.get("agent_id"),
                cmd=st.get("cmd"),
                outcome=outcome,
                attempts=st.get("attempts"),
                duration_ms=st.get("duration_ms"),
                ack_latency_ms=st.get("ack_latency_ms"),
                request_to_result_ms=req_to_result,
                request_to_ack_ms=req_to_ack,
                ack_to_result_ms=ack_to_result,
                result_success=ok,
                result_msg=msg,
                duplicate=dup,
                timeout_sec=timeout_sec,
                send_error=(send_error[:300] if send_error else None),
                job_id=st.get("job_id"),
                device_id=st.get("device_id"),
            )
        except Exception:
            pass

    async def accept(
        self,
        agent_id: str,
        websocket: WebSocket,
        tunnel_token: Optional[str] = None,
    ) -> bool:
        if not self._tunnel_token_ok(agent_id=agent_id, tunnel_token=tunnel_token):
            logger.warning("agent_tunnel: reject handshake agent_id=%s reason=invalid_or_missing_tunnel_token", agent_id)
            await websocket.accept()
            await websocket.close(code=4401, reason="invalid tunnel token")
            return False

        await websocket.accept()
        replaced_previous = False
        async with self._connections_lock:
            old = self._connections.get(agent_id)
            # Only allow one active tunnel per agent_id (replace old if needed).
            if old is not None and old is not websocket:
                replaced_previous = True
                try:
                    await old.close(code=4409, reason="Replaced by new tunnel")
                except Exception:
                    pass
            self._connections[agent_id] = websocket
            self._send_locks.setdefault(agent_id, asyncio.Lock())
            self._last_seen_ts[agent_id] = time.time()
        self._ensure_send_worker(agent_id)
        logger.info(
            "agent_tunnel: connected agent_id=%s replaced_previous=%s",
            agent_id,
            replaced_previous,
        )
        return True

    async def disconnect(self, agent_id: str, websocket: WebSocket, *, reason: Optional[str] = None) -> None:
        removed = False
        async with self._connections_lock:
            cur = self._connections.get(agent_id)
            if cur is websocket:
                self._connections.pop(agent_id, None)
                removed = True

        if removed:
            self._stop_send_worker(agent_id)
            async with self._pending_lock:
                for cid, fut in list(self._pending.items()):
                    if fut.done():
                        continue
                    st = self._command_states.get(str(cid))
                    if isinstance(st, dict) and st.get("agent_id") == agent_id:
                        logger.warning(
                            "agent_tunnel: active tunnel dropped while command pending agent_id=%s command_id=%s cmd=%s reason=%s",
                            agent_id,
                            cid,
                            st.get("cmd"),
                            reason or "",
                        )
                        try:
                            log_telemetry_event(
                                category="tunnel",
                                event="disconnect_pending_command",
                                agent_id=agent_id,
                                command_id=str(cid),
                                cmd=st.get("cmd"),
                                reason=(reason or "")[:200] or None,
                            )
                        except Exception:
                            pass
            logger.warning(
                "agent_tunnel: disconnected agent_id=%s reason=%s",
                agent_id,
                reason or "unknown",
            )
        else:
            logger.info(
                "agent_tunnel: disconnect ignored (socket not current tunnel) agent_id=%s reason=%s",
                agent_id,
                reason or "unknown",
            )

        removed_cameras = list((self._camera_id_to_cam_idx.get(agent_id) or {}).keys())
        await self._reset_agent_stream_state(agent_id, drop_mappings=True)
        if removed_cameras:
            log_experiment_event(
                role="platform",
                event="preview_mapping_removed",
                agent_id=agent_id,
                camera_ids=removed_cameras,
                camera_count=len(removed_cameras),
            )

    async def _match_connection_key(self, agent_id: str) -> Optional[str]:
        """在活跃隧道表中查找与 agent_id 等价的连接键（含 MAC 形式归一）。"""
        async with self._connections_lock:
            aid = (agent_id or "").strip()
            if aid and aid in self._connections:
                return aid
            norm = normalize_mac_like_agent_id(aid)
            if norm and norm != aid and norm in self._connections:
                return norm
            if is_colon_mac_normalized(norm):
                for connected_id in list(self._connections.keys()):
                    if normalize_mac_like_agent_id(connected_id) == norm:
                        return connected_id
        return None

    async def resolve_connected_socket_key(
        self,
        agent_id: str,
        *,
        platform_device_id: Optional[int] = None,
    ) -> Optional[str]:
        """
        解析 WebSocket 隧道在 _connections 中实际使用的 agent_id 键。
        - 兼容 MAC 大小写/分隔符差异；
        - 若同步侧解析出的是 devices.hardware_uuid（常为 MAC），而隧道使用配置里的 AGENT_ID（别名），
          可通过 platform_device_id + AgentRegistry 严格映射到在线隧道键。
        """
        key = await self._match_connection_key(agent_id)
        if key:
            return key
        if platform_device_id is None:
            return None
        try:
            from app.services.agent_registry import agent_registry

            ag = agent_registry.get_by_device_id_strict(int(platform_device_id))
        except Exception:
            ag = None
        if not ag:
            return None
        rid = str(getattr(ag, "agent_id", "") or "").strip()
        if not rid or rid == (agent_id or "").strip():
            return None
        return await self._match_connection_key(rid)

    async def has_connection(self, agent_id: str, platform_device_id: Optional[int] = None) -> bool:
        return (
            await self.resolve_connected_socket_key(agent_id, platform_device_id=platform_device_id)
        ) is not None

    def get_connected_agent_ids(self) -> List[str]:
        # Best-effort: avoid locking for lightweight iteration.
        try:
            return list(self._connections.keys())
        except Exception:
            return []

    async def send_cmd_and_wait(
        self,
        *,
        agent_id: str,
        cmd: str,
        payload: Optional[Dict[str, Any]] = None,
        timeout_sec: float = 15.0,
        command_id: Optional[str] = None,
        retry_times: int = 0,
    ) -> Dict[str, Any]:
        """
        Send CMD_REQUEST and await CMD_RESULT by command_id.
        Returns CMD_RESULT.payload.
        若传入 command_id，则用于超时重发同一命令（文档 §9）。
        """
        if not agent_id:
            raise ValueError("agent_id is required")

        socket_agent_id = await self._match_connection_key(agent_id) or (agent_id or "").strip()

        payload = dict(payload or {})
        command_id = (command_id or "").strip() or uuid4().hex
        self._upsert_command_state(
            command_id,
            {
                "agent_id": socket_agent_id,
                "cmd": cmd,
                "run_id": payload.get("run_id"),
                "scenario_id": payload.get("scenario_id"),
                "task_id": payload.get("task_id"),
                "job_id": payload.get("job_id"),
                "device_id": payload.get("device_id"),
                "status": "pending",
                "attempts": 0,
                "last_error": None,
                "duration_ms": None,
            },
        )
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        async with self._pending_lock:
            self._pending[command_id] = fut

        try:
            envelope: Dict[str, Any] = {
                "ver": 1,
                "type": "CMD_REQUEST",
                "agent_id": socket_agent_id,
                "command_id": command_id,
                "ts_ms": int(time.time() * 1000),
                "payload": {
                    "cmd": cmd,
                    "data": payload,
                },
            }
            self._metrics["cmd_requests_sent"] = self._metrics.get("cmd_requests_sent", 0) + 1
            attempts = max(0, int(retry_times)) + 1
            for idx in range(attempts):
                self._upsert_command_state(
                    command_id,
                    {
                        "status": "pending",
                        "attempts": idx + 1,
                    },
                )
                send_ts = int(time.time() * 1000)
                envelope["ts_ms"] = send_ts
                send_patch: Dict[str, Any] = {"last_send_ts_ms": send_ts, "status": "pending"}
                if idx == 0:
                    send_patch["first_send_ts_ms"] = send_ts
                    log_experiment_event(
                        role="platform",
                        event="command_sent",
                        ts_ms=send_ts,
                        command_id=command_id,
                        cmd=cmd,
                        agent_id=socket_agent_id,
                        run_id=payload.get("run_id"),
                        scenario_id=payload.get("scenario_id"),
                        task_id=payload.get("task_id"),
                        job_id=payload.get("job_id"),
                        device_id=payload.get("device_id"),
                    )
                self._upsert_command_state(command_id, send_patch)
                try:
                    await self._send_json(socket_agent_id, envelope)
                except Exception as exc:
                    self._telemetry_log_control_command(
                        command_id,
                        outcome="send_failed",
                        send_error=str(exc),
                    )
                    raise
                try:
                    result_payload = await asyncio.wait_for(fut, timeout=timeout_sec)
                    if not isinstance(result_payload, dict):
                        self._metrics["cmd_result_error"] = self._metrics.get("cmd_result_error", 0) + 1
                        self._upsert_command_state(
                            command_id,
                            {
                                "status": "failed",
                                "last_error": "Invalid CMD_RESULT payload",
                            },
                        )
                        self._telemetry_log_control_command(
                            command_id,
                            outcome="invalid_result_payload",
                            result_payload=None,
                        )
                        return {"success": False, "msg": "Invalid CMD_RESULT payload"}
                    created = int(self._command_states.get(command_id, {}).get("created_ts_ms") or 0)
                    duration_ms = int(time.time() * 1000) - created if created > 0 else None
                    if bool(result_payload.get("success", True)):
                        self._metrics["cmd_result_ok"] = self._metrics.get("cmd_result_ok", 0) + 1
                        self._upsert_command_state(
                            command_id,
                            {
                                "status": "succeeded",
                                "duration_ms": duration_ms,
                                "last_error": None,
                            },
                        )
                        self._telemetry_log_control_command(
                            command_id,
                            outcome="succeeded",
                            result_payload=result_payload,
                        )
                    else:
                        self._metrics["cmd_result_error"] = self._metrics.get("cmd_result_error", 0) + 1
                        self._upsert_command_state(
                            command_id,
                            {
                                "status": "failed",
                                "duration_ms": duration_ms,
                                "last_error": result_payload.get("msg") or result_payload.get("message"),
                            },
                        )
                        self._telemetry_log_control_command(
                            command_id,
                            outcome="failed",
                            result_payload=result_payload,
                        )
                    return result_payload
                except asyncio.TimeoutError:
                    self._metrics["cmd_timeout"] = self._metrics.get("cmd_timeout", 0) + 1
                    logger.warning(
                        "agent_tunnel: command_timeout agent_id=%s cmd=%s command_id=%s timeout_sec=%s attempt=%s/%s",
                        socket_agent_id,
                        cmd,
                        command_id,
                        timeout_sec,
                        idx + 1,
                        attempts,
                    )
                    log_experiment_event(
                        role="platform",
                        event="command_timeout",
                        ts_ms=int(time.time() * 1000),
                        command_id=command_id,
                        cmd=cmd,
                        agent_id=socket_agent_id,
                        run_id=payload.get("run_id"),
                        scenario_id=payload.get("scenario_id"),
                        task_id=payload.get("task_id"),
                        job_id=payload.get("job_id"),
                        device_id=payload.get("device_id"),
                    )
                    if idx + 1 >= attempts:
                        self._upsert_command_state(
                            command_id,
                            {
                                "status": "timeout",
                                "last_error": f"timeout>{timeout_sec}s",
                            },
                        )
                        self._telemetry_log_control_command(
                            command_id,
                            outcome="timeout",
                            timeout_sec=float(timeout_sec),
                        )
                        raise
                    self._metrics["cmd_retries"] = self._metrics.get("cmd_retries", 0) + 1
                    self._upsert_command_state(
                        command_id,
                        {
                            "status": "pending",
                            "last_error": f"retry after timeout>{timeout_sec}s",
                        },
                    )
                    log_telemetry_event(
                        category="tunnel",
                        event="control_command_retry",
                        command_id=command_id,
                        agent_id=socket_agent_id,
                        cmd=cmd,
                        attempt=idx + 1,
                        attempts_total=attempts,
                        timeout_sec=float(timeout_sec),
                    )
                    continue
        finally:
            async with self._pending_lock:
                self._pending.pop(command_id, None)

    def _cmd_priority_from_envelope(self, envelope: Dict[str, Any]) -> int:
        payload = envelope.get("payload") if isinstance(envelope.get("payload"), dict) else {}
        cmd = str(payload.get("cmd") or "").strip().upper()
        return _TUNNEL_CMD_PRIORITY.get(cmd, _DEFAULT_TUNNEL_CMD_PRIORITY)

    def _ensure_send_worker(self, agent_id: str) -> None:
        task = self._send_workers.get(agent_id)
        if task is not None and not task.done():
            return
        self._send_queues.setdefault(agent_id, asyncio.PriorityQueue())
        self._send_queue_seq.setdefault(agent_id, 0)
        self._send_workers[agent_id] = asyncio.create_task(self._send_worker_loop(agent_id))

    def _stop_send_worker(self, agent_id: str) -> None:
        task = self._send_workers.pop(agent_id, None)
        if task is not None and not task.done():
            task.cancel()
        q = self._send_queues.pop(agent_id, None)
        self._send_queue_seq.pop(agent_id, None)
        if q is None:
            return
        while not q.empty():
            try:
                _prio, _seq, _env, done = q.get_nowait()
            except asyncio.QueueEmpty:
                break
            if isinstance(done, asyncio.Future) and not done.done():
                done.set_exception(RuntimeError(f"agent tunnel disconnected: agent_id={agent_id}"))

    async def _send_worker_loop(self, agent_id: str) -> None:
        q = self._send_queues.get(agent_id)
        if q is None:
            return
        try:
            while True:
                priority, _seq, envelope, done = await q.get()
                try:
                    async with self._connections_lock:
                        ws = self._connections.get(agent_id)
                        lock = self._send_locks.get(agent_id)
                    if ws is None or lock is None:
                        if isinstance(done, asyncio.Future) and not done.done():
                            done.set_exception(
                                RuntimeError(f"agent tunnel not connected: agent_id={agent_id}")
                            )
                        continue
                    async with lock:
                        await ws.send_text(json.dumps(envelope, ensure_ascii=False))
                    if isinstance(done, asyncio.Future) and not done.done():
                        done.set_result(None)
                except Exception as exc:
                    logger.warning(
                        "agent_tunnel: send_failed agent_id=%s cmd=%s command_id=%s err=%s",
                        agent_id,
                        (envelope.get("payload") or {}).get("cmd"),
                        envelope.get("command_id"),
                        exc,
                    )
                    if isinstance(done, asyncio.Future) and not done.done():
                        done.set_exception(exc)
                finally:
                    q.task_done()
        except asyncio.CancelledError:
            return

    async def _send_json(self, agent_id: str, envelope: Dict[str, Any]) -> None:
        async with self._connections_lock:
            ws = self._connections.get(agent_id)
        if ws is None:
            raise RuntimeError(f"agent tunnel not connected: agent_id={agent_id}")

        self._ensure_send_worker(agent_id)
        q = self._send_queues.get(agent_id)
        if q is None:
            raise RuntimeError(f"agent tunnel send queue missing: agent_id={agent_id}")

        loop = asyncio.get_running_loop()
        done: asyncio.Future = loop.create_future()
        seq = self._send_queue_seq.get(agent_id, 0)
        self._send_queue_seq[agent_id] = seq + 1
        priority = self._cmd_priority_from_envelope(envelope)
        await q.put((priority, seq, envelope, done))
        await done

    async def handle_text_message(self, *, agent_id: str, message_text: str) -> None:
        """
        Parse incoming envelope and dispatch:
        - LOG -> script_runner.broadcast
        - CMD_RESULT -> resolve pending command future
        """
        try:
            data = json.loads(message_text)
        except Exception:
            # Ignore invalid payloads.
            return

        # 任意合法文本包均刷新 last_seen（文档 §10）
        self._last_seen_ts[agent_id] = time.time()

        msg_type = data.get("type")
        if msg_type == "LOG":
            payload = data.get("payload") or {}
            if isinstance(payload, dict):
                message = payload.get("message")
            else:
                message = str(payload)
            if not message:
                message = ""
            # Keep same semantics as existing HTTP /api/script/agent-log.
            await script_runner.broadcast(str(message))
            return

        if msg_type == "STREAM_CAPS":
            # Agent says which cameras it can provide; platform allocates cam_idx and replies STREAM_MAPPING.
            payload = data.get("payload") or {}
            if not isinstance(payload, dict):
                return
            camera_ids = payload.get("camera_ids") or []
            if not isinstance(camera_ids, list):
                camera_ids = []

            # Assign cam_idx deterministically (sorted) for stability across short reconnects.
            # Note: this is MVP behavior; in production you may want a persistent mapping.
            uniq = [str(x) for x in camera_ids if str(x).strip()]
            uniq_sorted = sorted(set(uniq))
            camera_id_to_cam_idx = {cid: i for i, cid in enumerate(uniq_sorted)}
            cam_idx_to_camera_id = {i: cid for cid, i in camera_id_to_cam_idx.items()}

            self._camera_id_to_cam_idx[agent_id] = camera_id_to_cam_idx
            self._cam_idx_to_camera_id[agent_id] = cam_idx_to_camera_id

            # Clear old reassembly caches for this agent, but keep current mapping.
            await self._reset_agent_stream_state(agent_id, drop_mappings=False)

            logger.info(
                "agent_tunnel: STREAM_CAPS agent_id=%s cameras=%s cam_idx=%s",
                agent_id,
                len(camera_ids),
                list(camera_id_to_cam_idx.values())[:8],
            )
            log_experiment_event(
                role="platform",
                event="preview_mapping_created",
                agent_id=agent_id,
                camera_ids=uniq_sorted,
                camera_count=len(uniq_sorted),
            )
            try:
                log_telemetry_event(
                    category="preview",
                    event="stream_caps",
                    agent_id=agent_id,
                    camera_count=len(uniq_sorted),
                    camera_ids=uniq_sorted[:32],
                )
            except Exception:
                pass

            try:
                await self._send_json(
                    agent_id,
                    {
                        "ver": 1,
                        "type": "STREAM_MAPPING",
                        "agent_id": agent_id,
                        "ts_ms": int(time.time() * 1000),
                        "payload": {
                            "camera_id_to_cam_idx": camera_id_to_cam_idx,
                            "cam_idx_to_camera_id": {str(k): v for k, v in cam_idx_to_camera_id.items()},
                            "chunk_size": CHUNK_SIZE,
                        },
                    },
                )
            except Exception:
                pass
            return

        if msg_type == "CMD_RESULT":
            command_id = data.get("command_id")
            result_payload = data.get("payload") or {}
            if not command_id:
                return
            ok = bool(result_payload.get("success", True)) if isinstance(result_payload, dict) else False
            dup = False
            if isinstance(result_payload, dict):
                dup = bool(result_payload.get("duplicate"))
            self._upsert_command_state(
                str(command_id),
                {
                    "status": "succeeded" if ok else "failed",
                    "last_error": None if ok else (
                        result_payload.get("msg") if isinstance(result_payload, dict) else "unknown cmd result payload"
                    ),
                    "agent_duplicate_reply": dup,
                },
            )
            st = self._command_states.get(str(command_id), {})
            cmd_name = st.get("cmd")
            if not ok:
                logger.warning(
                    "agent_tunnel: CMD_RESULT failed agent_id=%s command_id=%s cmd=%s msg=%s duplicate=%s",
                    agent_id,
                    command_id,
                    cmd_name,
                    result_payload.get("msg") if isinstance(result_payload, dict) else None,
                    dup,
                )
            elif cmd_name in ("DEVICE_LAUNCH", "DEVICE_STOP", "DEVICE_TEST_CONNECTION"):
                logger.info(
                    "agent_tunnel: CMD_RESULT ok agent_id=%s command_id=%s cmd=%s msg=%s",
                    agent_id,
                    command_id,
                    cmd_name,
                    result_payload.get("msg") if isinstance(result_payload, dict) else None,
                )
            log_experiment_event(
                role="platform",
                event="result_received",
                ts_ms=data.get("ts_ms"),
                command_id=str(command_id),
                cmd=st.get("cmd"),
                agent_id=agent_id,
                run_id=st.get("run_id"),
                scenario_id=st.get("scenario_id"),
                task_id=st.get("task_id"),
                job_id=st.get("job_id"),
                device_id=st.get("device_id"),
                success=ok,
                agent_duplicate_reply=dup,
                message=result_payload.get("msg") if isinstance(result_payload, dict) else None,
            )
            async with self._pending_lock:
                fut = self._pending.get(command_id)
            if fut is not None and not fut.done():
                fut.set_result(result_payload)
            return

        if msg_type == "CMD_ACK":
            command_id = data.get("command_id")
            if command_id:
                now_ms = int(time.time() * 1000)
                st0 = self._command_states.get(str(command_id), {})
                if not isinstance(st0, dict):
                    st0 = {}
                first = st0.get("first_send_ts_ms") or st0.get("last_send_ts_ms") or st0.get("created_ts_ms")
                ack_lat: Optional[int] = None
                if first is not None:
                    try:
                        ack_lat = now_ms - int(first)
                    except Exception:
                        ack_lat = None
                self._upsert_command_state(
                    str(command_id),
                    {
                        "status": "running",
                        "ack_ts_ms": now_ms,
                        "ack_latency_ms": ack_lat,
                    },
                )
                st = self._command_states.get(str(command_id), {})
                log_experiment_event(
                    role="platform",
                    event="ack_received",
                    ts_ms=data.get("ts_ms"),
                    command_id=str(command_id),
                    cmd=st.get("cmd"),
                    agent_id=agent_id,
                    run_id=st.get("run_id"),
                    scenario_id=st.get("scenario_id"),
                    task_id=st.get("task_id"),
                    job_id=st.get("job_id"),
                    device_id=st.get("device_id"),
                )
                try:
                    log_telemetry_event(
                        category="tunnel",
                        event="control_command_ack",
                        command_id=str(command_id),
                        agent_id=agent_id,
                        cmd=st.get("cmd"),
                        ack_latency_ms=ack_lat,
                        attempts=st.get("attempts"),
                    )
                except Exception:
                    pass
            return

        if msg_type == "HEARTBEAT":
            payload = data.get("payload") or {}
            if isinstance(payload, dict):
                now_ms = int(time.time() * 1000)
                sent_ts_ms = data.get("ts_ms")
                sent_ts_ms_int: Optional[int] = None
                try:
                    if sent_ts_ms is not None:
                        sent_ts_ms_int = int(sent_ts_ms)
                except Exception:
                    sent_ts_ms_int = None

                # latency: one-way-ish (agent sends ts_ms at send time; platform recv approximates delay)
                if sent_ts_ms_int is not None:
                    payload["tunnel_latency_ms"] = max(0, now_ms - sent_ts_ms_int)

                # heartbeat interval + missed estimation
                prev_recv = self._last_heartbeat_recv_ts_ms.get(agent_id)
                interval_ms: Optional[int] = None
                if isinstance(prev_recv, int) and prev_recv > 0:
                    interval_ms = max(0, now_ms - prev_recv)
                    payload["heartbeat_interval_ms"] = interval_ms

                    exp = self._heartbeat_expected_interval_ms.get(agent_id)
                    if exp is None:
                        exp = float(interval_ms) if interval_ms else 0.0
                    if exp and exp > 0 and interval_ms:
                        # EMA 更新“期望心跳间隔”（自适应 agent 配置不同步的问题）
                        payload["heartbeat_expected_interval_ms"] = round(exp, 1)
                        exp_new = (exp * 0.8) + (float(interval_ms) * 0.2)
                        self._heartbeat_expected_interval_ms[agent_id] = exp_new
                        # 若当前间隔显著变大，估计丢了若干个期望心跳
                        missed = 0
                        if float(interval_ms) > exp * 1.5:
                            missed = max(0, int(round(float(interval_ms) / exp)) - 1)
                        if missed:
                            self._heartbeat_missed_intervals_total[agent_id] = self._heartbeat_missed_intervals_total.get(agent_id, 0) + missed
                        payload["heartbeat_missed_intervals"] = missed
                    else:
                        self._heartbeat_expected_interval_ms[agent_id] = float(interval_ms) if interval_ms else exp or 0.0

                self._last_heartbeat_recv_ts_ms[agent_id] = now_ms
                self._last_heartbeat_payload[agent_id] = payload
            return

    def get_last_seen_ts(self, agent_id: str) -> Optional[float]:
        return self._last_seen_ts.get(agent_id)

    def get_last_heartbeat_payload(self, agent_id: str) -> Optional[Dict[str, Any]]:
        payload = self._last_heartbeat_payload.get(agent_id)
        if not isinstance(payload, dict):
            return None
        return dict(payload)

    async def _reset_agent_stream_state(self, agent_id: str, *, drop_mappings: bool = False) -> None:
        # Remove per-cam buffers for this agent.
        # Keep conditions; they will be recreated lazily by generator if missing.
        to_del_latest: List[Tuple[str, int]] = [k for k in self._latest_jpeg.keys() if k[0] == agent_id]
        for k in to_del_latest:
            self._latest_jpeg.pop(k, None)

        to_del_seq: List[Tuple[str, int]] = [k for k in self._stream_seq.keys() if k[0] == agent_id]
        for k in to_del_seq:
            self._stream_seq.pop(k, None)

        to_del_frames: List[Tuple[str, int]] = [k for k in self._frame_states.keys() if k[0] == agent_id]
        for k in to_del_frames:
            self._frame_states.pop(k, None)

        # Only disconnect / teardown should remove mapping.
        if drop_mappings:
            self._camera_id_to_cam_idx.pop(agent_id, None)
            self._cam_idx_to_camera_id.pop(agent_id, None)
        return

    def get_cam_idx(self, *, agent_id: str, camera_id: str) -> Optional[int]:
        m = self._camera_id_to_cam_idx.get(agent_id) or {}
        cid = (camera_id or "").strip()
        if not cid:
            return None
        v = m.get(cid)
        if v is None:
            return None
        return int(v)

    def get_camera_ids(self, agent_id: str) -> List[str]:
        m = self._camera_id_to_cam_idx.get(agent_id) or {}
        # Keep stable ordering by cam_idx.
        try:
            items = sorted([(cid, idx) for cid, idx in m.items()], key=lambda x: x[1])
            return [cid for cid, _ in items]
        except Exception:
            return list(m.keys())

    async def wait_for_new_jpeg(
        self,
        *,
        agent_id: str,
        cam_idx: int,
        last_seq: int,
        timeout_sec: float = 2.0,
    ) -> Tuple[int, Optional[bytes]]:
        key = (agent_id, int(cam_idx))
        cond = self._stream_conditions.get(key)
        if cond is None:
            self._stream_conditions[key] = asyncio.Condition()
            cond = self._stream_conditions[key]

        async with cond:
            seq0 = self._stream_seq.get(key, 0)
            if seq0 != last_seq and seq0 > 0:
                return seq0, self._latest_jpeg.get(key)

            try:
                await asyncio.wait_for(cond.wait(), timeout=timeout_sec)
            except TimeoutError:
                pass

            seq1 = self._stream_seq.get(key, 0)
            if seq1 == last_seq:
                return seq1, self._latest_jpeg.get(key)
            return seq1, self._latest_jpeg.get(key)

    async def handle_mjpeg_chunk_binary(
        self,
        *,
        agent_id: str,
        message_bytes: bytes,
    ) -> None:
        """
        Parse MJPEG_CHUNK binary and update per-cam latest_jpeg buffer.
        """
        if not message_bytes or len(message_bytes) < 32:
            self._metrics["mjpeg_chunks_malformed"] = self._metrics.get("mjpeg_chunks_malformed", 0) + 1
            return
        header = message_bytes[:32]
        try:
            (
                magic,
                ver,
                kind,
                reserved,
                frame_id,
                cam_idx,
                chunk_index,
                chunk_count,
                chunk_len,
            ) = struct.unpack(">4sBBH Q I I I I", header)
        except Exception:
            self._metrics["mjpeg_chunks_malformed"] = self._metrics.get("mjpeg_chunks_malformed", 0) + 1
            return

        if magic != MJPEG_CHUNK_MAGIC:
            self._metrics["mjpeg_chunks_malformed"] = self._metrics.get("mjpeg_chunks_malformed", 0) + 1
            return
        if ver != MJPEG_CHUNK_VER or kind != MJPEG_CHUNK_KIND:
            self._metrics["mjpeg_chunks_malformed"] = self._metrics.get("mjpeg_chunks_malformed", 0) + 1
            return
        if chunk_len < 0:
            self._metrics["mjpeg_chunks_malformed"] = self._metrics.get("mjpeg_chunks_malformed", 0) + 1
            return
        if int(chunk_count) <= 0 or int(chunk_count) > MAX_MJPEG_CHUNKS_PER_FRAME:
            self._metrics["mjpeg_chunks_malformed"] = self._metrics.get("mjpeg_chunks_malformed", 0) + 1
            return

        expected_payload_len = 32 + int(chunk_len)
        if len(message_bytes) < expected_payload_len:
            self._metrics["mjpeg_chunks_malformed"] = self._metrics.get("mjpeg_chunks_malformed", 0) + 1
            return
        chunk = message_bytes[32:expected_payload_len]

        self._metrics["mjpeg_chunks_received"] = self._metrics.get("mjpeg_chunks_received", 0) + 1

        await self._handle_mjpeg_chunk(
            agent_id=agent_id,
            cam_idx=int(cam_idx),
            frame_id=int(frame_id),
            chunk_index=int(chunk_index),
            chunk_count=int(chunk_count),
            chunk=chunk,
        )

    async def _handle_mjpeg_chunk(
        self,
        *,
        agent_id: str,
        cam_idx: int,
        frame_id: int,
        chunk_index: int,
        chunk_count: int,
        chunk: bytes,
    ) -> None:
        key = (agent_id, cam_idx)
        now_ms = int(time.time() * 1000)

        state = self._frame_states.get(key)
        if state is None:
            state = {
                "latest_frame_id": -1,
                "assembling_frame_id": None,
                "chunk_count": 0,
                "chunks": None,  # type: Optional[List[Optional[bytes]]]
                "last_chunk_ts_ms": 0,
                "assembly_start_ts_ms": 0,
            }
            self._frame_states[key] = state

        latest_frame_id = int(state.get("latest_frame_id", -1))

        # Timeout: 自 assembly_start_ts_ms 起超过阈值未完成则丢弃（避免误用 last_chunk_ts=0 导致误判）
        asm_id = state.get("assembling_frame_id")
        asm_start = int(state.get("assembly_start_ts_ms", 0) or 0)
        if asm_id is not None and asm_start > 0 and now_ms - asm_start > FRAME_ASSEMBLY_TIMEOUT_MS:
            self._metrics["mjpeg_frames_timeout"] = self._metrics.get("mjpeg_frames_timeout", 0) + 1
            state["assembling_frame_id"] = None
            state["chunk_count"] = 0
            state["chunks"] = None
            state["assembly_start_ts_ms"] = 0

        if frame_id < latest_frame_id:
            self._metrics["mjpeg_frames_stale"] = self._metrics.get("mjpeg_frames_stale", 0) + 1
            return

        def _budget_allows_new_assembly() -> bool:
            """新开帧组缓存前做全局预算检查（文档 §7.6B）。"""
            cnt = self._mjpeg_assembling_count()
            chunks0 = state.get("chunks")
            prev_inc = False
            if state.get("assembling_frame_id") is not None and isinstance(chunks0, list):
                cc0 = int(state.get("chunk_count") or 0)
                if cc0 > 0:
                    got0 = sum(1 for c in chunks0 if c is not None)
                    prev_inc = got0 < cc0
            if prev_inc:
                cnt -= 1
            return cnt < MAX_MJPEG_CONCURRENT_ASSEMBLIES

        if state.get("assembling_frame_id") != frame_id:
            if not _budget_allows_new_assembly():
                self._metrics["mjpeg_assembly_budget_drops"] = self._metrics.get("mjpeg_assembly_budget_drops", 0) + 1
                return
            # New frame: reset buffer.
            state["assembling_frame_id"] = frame_id
            state["chunk_count"] = int(chunk_count)
            state["chunks"] = [None] * int(chunk_count)
            state["assembly_start_ts_ms"] = now_ms

        # If chunk_count changed unexpectedly for the same frame, reset.
        if int(state.get("chunk_count", 0)) != int(chunk_count):
            if not _budget_allows_new_assembly():
                self._metrics["mjpeg_assembly_budget_drops"] = self._metrics.get("mjpeg_assembly_budget_drops", 0) + 1
                return
            state["assembling_frame_id"] = frame_id
            state["chunk_count"] = int(chunk_count)
            state["chunks"] = [None] * int(chunk_count)
            state["assembly_start_ts_ms"] = now_ms

        chunks = state.get("chunks")
        if not isinstance(chunks, list):
            return
        if chunk_index < 0 or chunk_index >= chunk_count:
            return
        if chunks[chunk_index] is None:
            chunks[chunk_index] = chunk
            state["last_chunk_ts_ms"] = now_ms

        # Check completeness
        received = 0
        for c in chunks:
            if c is not None:
                received += 1
        if received != int(chunk_count):
            return

        # Assemble complete JPEG（bytearray 减少中间拷贝）
        try:
            buf = bytearray()
            for c in chunks:
                if c:
                    buf.extend(c)
            jpeg_bytes = bytes(buf)
        except Exception:
            return

        self._metrics["mjpeg_frames_completed"] = self._metrics.get("mjpeg_frames_completed", 0) + 1

        state["latest_frame_id"] = frame_id
        state["assembling_frame_id"] = frame_id

        # Publish latest JPEG
        self._latest_jpeg[key] = jpeg_bytes
        seq = self._stream_seq.get(key, 0) + 1
        self._stream_seq[key] = seq

        cond = self._stream_conditions.get(key)
        if cond is None:
            return
        async with cond:
            cond.notify_all()


agent_tunnel_manager = AgentTunnelManager()
