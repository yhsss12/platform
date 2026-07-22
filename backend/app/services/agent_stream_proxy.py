"""
采集端 MJPEG 流代理：统一通过 Agent Tunnel 获取流列表与重组后的 MJPEG 帧。
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from app.crud.device import get_device_by_id
from app.db.session import AsyncSessionLocal
from app.services.agent_registry import agent_registry
from app.services.agent_tunnel_manager import agent_tunnel_manager

logger = logging.getLogger(__name__)

_STREAM_DEBUG = str(os.environ.get("EAI_STREAM_DEBUG", "") or "").strip().lower() in ("1", "true", "yes", "on")

_PLACEHOLDER_CACHE: Dict[Tuple[str, str], bytes] = {}


def _stream_dbg(msg: str, *args: Any) -> None:
    if _STREAM_DEBUG:
        logger.info("stream_debug: " + msg, *args)


def _wrap_mjpeg_frame(jpeg_bytes: bytes) -> bytes:
    bts = jpeg_bytes or b""
    return (
        b"--frame\r\n"
        + b"Content-Type: image/jpeg\r\n"
        + f"Content-Length: {len(bts)}\r\n".encode("ascii")
        + b"\r\n"
        + bts
        + b"\r\n"
    )


def _make_placeholder_bytes(camera_id: str, reason: str = "NO FRAME") -> bytes:
    """
    占位 JPEG（文档 §7.7）：在无映射/无帧时仍输出 multipart，便于前端稳定显示。
    """
    r = (reason or "NO FRAME").strip()[:120] or "NO FRAME"
    key = (camera_id, r)
    if key in _PLACEHOLDER_CACHE:
        return _PLACEHOLDER_CACHE[key]
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
        text = f"{r} | {camera_id}"
        cv2.putText(
            img,
            text[:90],
            (20, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        ok, buf = cv2.imencode(".jpg", img)
        jpeg = buf.tobytes() if ok and buf is not None else b""
    except Exception:
        jpeg = b""
    if len(_PLACEHOLDER_CACHE) > 64:
        _PLACEHOLDER_CACHE.clear()
    out = jpeg if jpeg else fallback
    _PLACEHOLDER_CACHE[key] = out
    return out


async def _resolve_device_bound_agent_id(device_id: Optional[int]) -> Optional[str]:
    """
    strict + 自动回填：
    - 先查 device_id -> agent_id 严格映射
    - 缺失时按 device.hardware_uuid(agent_id) 回填
    """
    if device_id is None:
        return None
    info = agent_registry.get_by_device_id_strict(int(device_id))
    if info and info.agent_id:
        return info.agent_id
    async with AsyncSessionLocal() as db:
        dev = await get_device_by_id(db, int(device_id))
    if dev is None:
        return None
    hw = str(getattr(dev, "hardware_uuid", "") or "").strip()
    if not hw:
        return None
    agent = agent_registry.get_by_id(hw)
    if agent and agent.agent_id:
        try:
            agent_registry.bind_device_to_agent(device_id=int(device_id), agent_id=hw)
        except Exception:
            pass
        return agent.agent_id
    return None


async def _resolve_stream_agent_id(
    *,
    device_id: Optional[int] = None,
    agent_id: Optional[str] = None,
) -> Optional[str]:
    if agent_id:
        return agent_id
    if device_id is not None:
        # 显式 device_id 时必须严格绑定该设备，禁止回退到「任意在线 Agent」，
        # 否则多采集端并存时会出现列表/MJPEG 串台、看起来像「缓存了别的设备画面」。
        return await _resolve_device_bound_agent_id(int(device_id))

    # 未带 device_id 的旧调用方：保留开发态回退（单机单 Agent）
    for aid in agent_tunnel_manager.get_connected_agent_ids():
        if aid and aid != "local-agent":
            return aid
    return None


async def _resolve_stream_agent_id_for_camera(
    *,
    camera_id: str,
    device_id: Optional[int] = None,
    agent_id: Optional[str] = None,
) -> Optional[str]:
    """
    Resolve which connected tunnel agent can provide `camera_id`.
    Avoid relying solely on device_id->agent_id mapping.
    """
    if agent_id:
        if agent_tunnel_manager.get_cam_idx(agent_id=agent_id, camera_id=camera_id) is not None:
            return agent_id
        return None

    # Try device_id mapping first.
    if device_id is not None:
        aid = await _resolve_device_bound_agent_id(int(device_id))
        if aid and agent_tunnel_manager.get_cam_idx(agent_id=aid, camera_id=camera_id) is not None:
            return aid
        # 已指定设备但尚无 cam_idx（STREAM_MAPPING 未到）：不得串到其他 Agent
        return None

    # 未带 device_id：才允许按 camera_id 在在线 Agent 间兜底（多机场景应避免不传 device_id）
    for aid in agent_tunnel_manager.get_connected_agent_ids():
        if aid and aid != "local-agent":
            if agent_tunnel_manager.get_cam_idx(agent_id=aid, camera_id=camera_id) is not None:
                return aid
    return None


async def get_stream_generator_via_tunnel(
    stream_id: str,
    *,
    device_id: Optional[int] = None,
    agent_id: Optional[str] = None,
) -> AsyncIterator[bytes]:
    """
    Serve MJPEG from tunnel reassembly cache (phase-2).
    """
    resolved_agent_id = await _resolve_stream_agent_id_for_camera(
        camera_id=stream_id,
        device_id=device_id,
        agent_id=agent_id,
    )

    if not resolved_agent_id:
        _stream_dbg(
            "mjpeg_gen no_agent stream_id=%s device_id=%s agent_id=%s",
            stream_id,
            device_id,
            agent_id,
        )
        while True:
            yield _wrap_mjpeg_frame(
                _make_placeholder_bytes(stream_id, "no agent for camera / tunnel")
            )
            await asyncio.sleep(1.0)

    last_seq = 0
    last_good: bytes = b""
    last_new_wall_ts = 0.0
    stall_warned = False
    last_diag_wall = 0.0
    cam_idx = agent_tunnel_manager.get_cam_idx(agent_id=resolved_agent_id, camera_id=stream_id)
    _stream_dbg(
        "mjpeg_gen start stream_id=%s device_id=%s agent_id=%s resolved=%s cam_idx=%s",
        stream_id,
        device_id,
        agent_id,
        resolved_agent_id,
        cam_idx,
    )
    while True:
        try:
            # 客户端断开时，Starlette 会取消该协程；这里静默退出，避免刷 traceback。
            await asyncio.sleep(0)
        except asyncio.CancelledError:
            return

        if not await agent_tunnel_manager.has_connection(resolved_agent_id):
            yield _wrap_mjpeg_frame(
                _make_placeholder_bytes(stream_id, "tunnel disconnected")
            )
            await asyncio.sleep(0.6)
            continue

        cam_idx = agent_tunnel_manager.get_cam_idx(agent_id=resolved_agent_id, camera_id=stream_id)
        if cam_idx is None:
            nowd = time.time()
            if _STREAM_DEBUG and nowd - last_diag_wall >= 5.0:
                last_diag_wall = nowd
                _stream_dbg(
                    "mjpeg_gen cam_idx_missing stream_id=%s resolved=%s (caps/mapping not ready or wrong camera_id)",
                    stream_id,
                    resolved_agent_id,
                )
            yield _wrap_mjpeg_frame(
                _make_placeholder_bytes(stream_id, "STREAM_MAPPING missing / cam_idx")
            )
            await asyncio.sleep(0.6)
            continue

        try:
            seq, jpeg_bytes = await agent_tunnel_manager.wait_for_new_jpeg(
                agent_id=resolved_agent_id,
                cam_idx=cam_idx,
                last_seq=last_seq,
                timeout_sec=2.0,
            )
        except asyncio.TimeoutError:
            seq, jpeg_bytes = last_seq, b""
        except asyncio.CancelledError:
            return

        if seq != last_seq and jpeg_bytes:
            last_seq = seq
            last_good = jpeg_bytes
            last_new_wall_ts = time.time()
            stall_warned = False
            yield _wrap_mjpeg_frame(jpeg_bytes)
        else:
            # 无新 seq 时：短时重复上一帧保持画面连贯；长时间无新帧则改发占位，避免「像缓存了一张静态旧图」
            now = time.time()
            if last_good and last_new_wall_ts > 0 and (now - last_new_wall_ts) < 12.0:
                yield _wrap_mjpeg_frame(last_good)
            else:
                if last_good and not stall_warned:
                    logger.warning(
                        "mjpeg_tunnel: stalled stream_id=%s agent_id=%s cam_idx=%s idle_sec=%.1f",
                        stream_id,
                        resolved_agent_id,
                        cam_idx,
                        (now - last_new_wall_ts) if last_new_wall_ts > 0 else -1.0,
                    )
                    stall_warned = True
                yield _wrap_mjpeg_frame(
                    _make_placeholder_bytes(
                        stream_id,
                        "no new frame" if last_good else "waiting frame",
                    )
                )


async def list_streams_via_agent(
    *,
    device_id: Optional[int] = None,
    agent_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    通过 tunnel 缓存返回相机列表（不再依赖 Agent HTTP）。
    """
    # Phase-2 first: if tunnel mapping exists, return from cache (avoid HTTP dependency).
    resolved_agent_id = await _resolve_stream_agent_id(device_id=device_id, agent_id=agent_id)
    _stream_dbg(
        "list_streams start device_id=%s agent_id=%s resolved_agent_id=%s",
        device_id,
        agent_id,
        resolved_agent_id,
    )
    if resolved_agent_id:
        deadline = time.time() + 2.0
        while time.time() < deadline:
            ids = agent_tunnel_manager.get_camera_ids(resolved_agent_id)
            if ids:
                _stream_dbg("list_streams ok agent_id=%s camera_ids=%s", resolved_agent_id, ids)
                return [{"id": cid, "name": cid, "url": ""} for cid in ids]
            await asyncio.sleep(0.25)
        _stream_dbg(
            "list_streams empty_mapping agent_id=%s (waited 2s, STREAM_MAPPING may be missing)",
            resolved_agent_id,
        )
        return []

    # 未指定 device_id 时，才扫描任意在线 Agent（兼容旧前端/脚本）
    deadline = time.time() + 2.0
    connected = [aid for aid in agent_tunnel_manager.get_connected_agent_ids() if aid and aid != "local-agent"]
    if connected:
        while time.time() < deadline:
            for aid in connected:
                ids = agent_tunnel_manager.get_camera_ids(aid)
                if ids:
                    _stream_dbg("list_streams legacy_fallback agent_id=%s camera_ids=%s", aid, ids)
                    return [{"id": cid, "name": cid, "url": ""} for cid in ids]
            await asyncio.sleep(0.25)

    _stream_dbg("list_streams no_cameras device_id=%s agent_id=%s", device_id, agent_id)
    return []


async def get_stream_status_via_tunnel(
    *,
    device_id: Optional[int] = None,
    agent_id: Optional[str] = None,
) -> Dict[str, Any]:
    resolved_agent_id = await _resolve_stream_agent_id(device_id=device_id, agent_id=agent_id)
    if not resolved_agent_id:
        return {"ok": False, "error": "未找到可用的采集端 Agent"}
    if not await agent_tunnel_manager.has_connection(resolved_agent_id):
        return {"ok": False, "error": "采集端隧道未连接"}
    try:
        result = await agent_tunnel_manager.send_cmd_and_wait(
            agent_id=resolved_agent_id,
            cmd="STREAM_STATUS",
            payload={},
            timeout_sec=8.0,
            retry_times=1,
        )
        if not bool(result.get("success", False)):
            return {"ok": False, "error": str(result.get("msg") or "隧道命令失败")[:200]}
        data = result.get("data")
        if isinstance(data, dict):
            return {"ok": True, "data": data, "agent_id": resolved_agent_id}
        return {"ok": False, "error": "采集端返回格式不正确"}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}


async def get_stream_generator_via_tunnel_or_agent(
    stream_id: str,
    *,
    device_id: Optional[int] = None,
    agent_id: Optional[str] = None,
) -> AsyncIterator[bytes]:
    """
    Tunnel-only generator. Function name kept for compatibility.
    """
    resolved_agent_id = await _resolve_stream_agent_id_for_camera(
        camera_id=stream_id,
        device_id=device_id,
        agent_id=agent_id,
    )
    if resolved_agent_id and await agent_tunnel_manager.has_connection(resolved_agent_id):
        async for frame in get_stream_generator_via_tunnel(
            stream_id,
            device_id=device_id,
            agent_id=agent_id,
        ):
            yield frame
        return

    # No HTTP fallback: keep yielding placeholder through tunnel generator path.
    async for frame in get_stream_generator_via_tunnel(
        stream_id,
        device_id=device_id,
        agent_id=agent_id,
    ):
        yield frame
