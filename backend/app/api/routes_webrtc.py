import logging
import time
from typing import Any, Dict, Optional

import httpx
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.crud.device import get_device_by_id
from app.db.session import AsyncSessionLocal
from app.services.agent_http_base_url import resolve_agent_http_base_url
from app.services.agent_registry import agent_registry
from app.core.config import settings
from app.services.agent_tunnel_manager import agent_tunnel_manager

logger = logging.getLogger(__name__)

router = APIRouter()


class WebRtcOfferRequest(BaseModel):
    """前端发送到平台的 WebRTC Offer 信令。"""

    sdp: str = Field(description="浏览器生成的 SDP offer")
    type: str = Field(description="SDP 类型，一般为 'offer'")
    device_id: Optional[int] = Field(
        default=None, description="设备 ID，用于从注册表解析 Agent"
    )
    agent_id: Optional[str] = Field(
        default=None, description="Agent ID，优先级高于 device_id"
    )
    camera_id: Optional[str] = Field(
        default=None, description="摄像头 ID，用于在 Agent 端选择具体相机流"
    )
    run_id: Optional[str] = Field(
        default=None, description="实验运行 ID"
    )
    scenario_id: Optional[str] = Field(
        default=None, description="实验场景 ID"
    )


class WebRtcAnswerResponse(BaseModel):
    """平台转发 Agent 返回的 Answer。"""

    sdp: str
    type: str


async def _webrtc_offer_via_http(*, base_url: str, payload: WebRtcOfferRequest) -> JSONResponse:
    url = f"{base_url.rstrip('/')}/api/agent/webrtc/offer"
    body: Dict[str, Any] = {"sdp": payload.sdp, "type": payload.type}
    if payload.camera_id:
        body["camera_id"] = payload.camera_id
    async with httpx.AsyncClient(timeout=25.0) as client:
        r = await client.post(url, json=body)
        if r.status_code >= 400:
            raise HTTPException(
                status_code=502,
                detail=(r.text or "")[:500] or f"Agent HTTP WebRTC 失败: {r.status_code}",
            )
        data = r.json()
    if not data.get("sdp") or not data.get("type"):
        raise HTTPException(status_code=502, detail="Agent 返回的 WebRTC answer 格式不正确（HTTP）")
    return JSONResponse(content={"sdp": str(data["sdp"]), "type": str(data["type"])})


async def _assert_device_exists_or_raise(device_id: int) -> None:
    """防呆：作业引用的 device_id 已被删除时，给出明确错误而非隧道错误。"""
    async with AsyncSessionLocal() as db:
        dev = await get_device_by_id(db, int(device_id))
    if dev is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"设备不存在（device_id={device_id}），可能已删除或重建。"
                "请在任务中心重新领取作业并选择当前设备。"
            ),
        )


async def _resolve_agent_id_with_autobind(*, device_id: int) -> Optional[str]:
    """
    自动回填 device_id -> agent_id：
    - 先查严格映射
    - 若缺失，尝试用 devices.hardware_uuid（即 agent_id）对在线 Agent 回填
    """
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
    if agent is None:
        return None
    try:
        agent_registry.bind_device_to_agent(device_id=int(device_id), agent_id=hw)
    except Exception:
        pass
    return hw


@router.post("/offer")
async def forward_webrtc_offer(payload: WebRtcOfferRequest) -> JSONResponse:
    """
    将浏览器的 WebRTC offer 转发到采集端 Agent，由 Agent 生成 answer 并返回。

    平台只做信令转发，不参与媒体通道建立。
    """
    t0 = time.perf_counter()

    def _elapsed_ms() -> float:
        return (time.perf_counter() - t0) * 1000.0

    resolved_agent_id: Optional[str] = None
    if payload.device_id is not None:
        await _assert_device_exists_or_raise(int(payload.device_id))

    resolved_agent_id = (payload.agent_id or "").strip() or None
    if not resolved_agent_id and payload.device_id is not None:
        resolved_agent_id = await _resolve_agent_id_with_autobind(device_id=int(payload.device_id))
    if not resolved_agent_id:
        agent_tunnel_manager.record_webrtc_offer(
            ok=False,
            route="resolve_agent",
            duration_ms=_elapsed_ms(),
            device_id=payload.device_id,
            error="no_agent_mapping",
        )
        raise HTTPException(
            status_code=404,
            detail="未找到可用的采集端 Agent：请确认作业已绑定设备且设备页已完成 Agent 绑定（内存中缺少 device→agent 映射，采集端需重新注册/心跳或重新 connect-agent）",
        )

    http_base = await resolve_agent_http_base_url(
        device_id=payload.device_id,
        agent_id=resolved_agent_id if resolved_agent_id != "local-agent" else None,
    )
    if (
        settings.WEBRTC_OFFER_PREFER_HTTP
        and http_base
        and resolved_agent_id != "local-agent"
    ):
        try:
            out = await _webrtc_offer_via_http(base_url=http_base, payload=payload)
            agent_tunnel_manager.record_webrtc_offer(
                ok=True,
                route="http_primary",
                duration_ms=_elapsed_ms(),
                agent_id=resolved_agent_id,
                device_id=payload.device_id,
            )
            return out
        except HTTPException:
            raise
        except Exception as exc:
            logger.warning("webrtc http primary failed, try tunnel: %s", exc)

    if not await agent_tunnel_manager.has_connection(resolved_agent_id):
        agent_tunnel_manager.record_webrtc_offer(
            ok=False,
            route="tunnel_disconnected",
            duration_ms=_elapsed_ms(),
            agent_id=resolved_agent_id,
            device_id=payload.device_id,
        )
        base = await resolve_agent_http_base_url(
            device_id=payload.device_id,
            agent_id=resolved_agent_id if resolved_agent_id != "local-agent" else None,
        )
        if base and resolved_agent_id != "local-agent":
            try:
                out = await _webrtc_offer_via_http(base_url=base, payload=payload)
                agent_tunnel_manager.record_webrtc_offer(
                    ok=True,
                    route="http_fallback",
                    duration_ms=_elapsed_ms(),
                    agent_id=resolved_agent_id,
                    device_id=payload.device_id,
                )
                return out
            except HTTPException:
                raise
            except Exception as exc:
                logger.warning("webrtc http fallback failed: %s", exc)
                agent_tunnel_manager.record_webrtc_offer(
                    ok=False,
                    route="http_fallback",
                    duration_ms=_elapsed_ms(),
                    agent_id=resolved_agent_id,
                    device_id=payload.device_id,
                    error=str(exc)[:400],
                )
        raise HTTPException(
            status_code=503,
            detail=(
                f"采集端隧道未连接（agent_id={resolved_agent_id}）。"
                "请确认边缘 Agent 已启动并成功连接 "
                "`WebSocket /api/agent/tunnel`（若配置了 AGENT_TUNNEL_TOKEN，URL 需带相同 token）。"
                "内网可达时平台可尝试 HTTP 回退，当前未成功。"
            ),
        )

    try:
        cmd_payload: Dict[str, Any] = {"sdp": payload.sdp, "type": payload.type}
        if payload.camera_id:
            cmd_payload["camera_id"] = payload.camera_id
        if payload.run_id:
            cmd_payload["run_id"] = payload.run_id
        if payload.scenario_id:
            cmd_payload["scenario_id"] = payload.scenario_id
        result = await agent_tunnel_manager.send_cmd_and_wait(
            agent_id=resolved_agent_id,
            cmd="WEBRTC_OFFER",
            payload=cmd_payload,
            timeout_sec=20.0,
            retry_times=1,
        )
        if not bool(result.get("success", False)):
            agent_tunnel_manager.record_webrtc_offer(
                ok=False,
                route="tunnel_cmd",
                duration_ms=_elapsed_ms(),
                agent_id=resolved_agent_id,
                device_id=payload.device_id,
                error=str(result.get("msg") or "")[:400] or None,
            )
            raise HTTPException(status_code=502, detail=result.get("msg") or "Agent 处理 WebRTC offer 失败")
        sdp = result.get("sdp")
        typ = result.get("type")
        if not sdp or not typ:
            agent_tunnel_manager.record_webrtc_offer(
                ok=False,
                route="tunnel_cmd",
                duration_ms=_elapsed_ms(),
                agent_id=resolved_agent_id,
                device_id=payload.device_id,
                error="invalid_answer_shape",
            )
            raise HTTPException(status_code=502, detail="Agent 返回的 WebRTC answer 格式不正确")
        agent_tunnel_manager.record_webrtc_offer(
            ok=True,
            route="tunnel_cmd",
            duration_ms=_elapsed_ms(),
            agent_id=resolved_agent_id,
            device_id=payload.device_id,
        )
        return JSONResponse(content={"sdp": str(sdp), "type": str(typ)})
    except HTTPException:
        raise
    except Exception as e:
        agent_tunnel_manager.record_webrtc_offer(
            ok=False,
            route="tunnel_cmd",
            duration_ms=_elapsed_ms(),
            agent_id=resolved_agent_id,
            device_id=payload.device_id,
            error=str(e)[:400],
        )
        logger.warning("webrtc tunnel forward failed: %s", e)
        raise HTTPException(status_code=502, detail=f"转发 WebRTC offer 失败: {e}")
