from __future__ import annotations

import json
import logging
import os
import re
import asyncio
from typing import Optional, Dict, Any

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.agent_registry import agent_registry, AgentInfo
from app.services.agent_tunnel_manager import (
    agent_tunnel_manager,
    is_colon_mac_normalized,
    normalize_mac_like_agent_id,
)
from app.services.minio_service import project_bucket_name
from app.core.config import settings
from app.crud.device import get_device_by_hardware_uuid, get_device_by_id, get_devices
from app.services.task_job_store import is_cancelled

logger = logging.getLogger(__name__)

# 响应体/元数据截断长度，避免日志刷屏
_SYNC_LOG_BODY_MAX = 2000
_SYNC_META_SNIPPET_MAX = 240


def _snippet(s: Optional[str], max_len: int = _SYNC_META_SNIPPET_MAX) -> str:
    if not s:
        return ""
    t = (s or "").replace("\n", " ").strip()
    if len(t) <= max_len:
        return t
    return t[: max_len - 3] + "..."


def _normalize_minio_endpoint(endpoint: str) -> str:
    """
    MinIO Python SDK 需要 host:port，不接受带 scheme 的 URL。
    错误配置为 http://host:9000 时会导致连接失败。
    """
    s = (endpoint or "").strip()
    if not s:
        return ""
    s = re.sub(r"^https?://", "", s, flags=re.IGNORECASE)
    return s.split("/")[0].strip()


def _minio_endpoint_loopback_host(host: str) -> bool:
    h = (host or "").strip().lower()
    return h in {"127.0.0.1", "localhost", "::1", "0.0.0.0"} or h.startswith("127.")


def _minio_endpoint_for_agent_payload(resolved_agent_id: str) -> str:
    """
    下发给采集端进程连接 S3/MinIO 的 host:port。
    远程机器上 127.0.0.1 指向采集机自身，故对非 local-agent 优先 MINIO_PUBLIC_ENDPOINT
    （浏览器/局域网可达的平台侧地址）；同机 local-agent 仍优先 MINIO_ENDPOINT。
    """
    pub_raw = (settings.MINIO_PUBLIC_ENDPOINT or "").strip()
    loc_raw = (settings.MINIO_ENDPOINT or "").strip()
    aid = (resolved_agent_id or "").strip().lower()
    if aid == "local-agent":
        raw = loc_raw or pub_raw
        mode = "local-agent: MINIO_ENDPOINT 优先"
    else:
        raw = pub_raw or loc_raw
        mode = "远程采集: MINIO_PUBLIC_ENDPOINT 优先"
    ep = _normalize_minio_endpoint(raw)
    if not ep:
        ep = _normalize_minio_endpoint(loc_raw or pub_raw)
    if ep:
        host_only = ep.split(":")[0].strip()
        if aid != "local-agent" and _minio_endpoint_loopback_host(host_only):
            logger.warning(
                "data_sync: 采集端 agent_id=%s 将使用回环 MinIO 地址 %r（%s）。"
                "远程设备会连接失败，请在 .env 设置 MINIO_PUBLIC_ENDPOINT=平台局域网IP:9000（或与 MINIO_API_PORT 一致）",
                resolved_agent_id,
                ep,
                mode,
            )
    return ep


def _pick_agent(agent_id: Optional[str] = None) -> Optional[AgentInfo]:
    if agent_id:
        return agent_registry.get_by_id(agent_id)
    agents = agent_registry.list_agents()
    for a in agents:
        if a.agent_id == "local-agent":
            continue
        # 不要把回环地址当作“远端采集端”
        host = str(getattr(a, "host", "") or "").strip().lower()
        if host in {"127.0.0.1", "localhost", "0.0.0.0"} or host.startswith("127."):
            continue
        if a.online and a.base_url:
            return a
    return agent_registry.get_by_id("local-agent")


def _parse_meta_dict(meta_json: Optional[str]) -> Optional[dict]:
    """解析资产 meta 为 dict；无效时只打一次 warning。"""
    if not meta_json:
        return None
    try:
        parsed = json.loads(meta_json)
    except Exception as e:
        logger.warning(
            "data_sync: 解析资产 meta JSON 失败: %s meta_snippet=%r",
            e,
            _snippet(meta_json),
        )
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def _device_id_from_meta_dict(parsed: Optional[dict]) -> Optional[str]:
    """meta.collect.device_id：平台 devices 表主键。"""
    if not parsed:
        return None
    c = parsed.get("collect")
    if not isinstance(c, dict):
        return None
    v = c.get("device_id")
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def _hardware_from_meta_dict(parsed: Optional[dict]) -> Optional[str]:
    """meta.collect 中 hardware_uuid / agent_id（与设备表 hardware_uuid 一致）。"""
    if not parsed:
        return None
    c = parsed.get("collect")
    if not isinstance(c, dict):
        return None
    for key in ("hardware_uuid", "agent_id"):
        v = c.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _collect_device_id_from_meta(meta_json: Optional[str]) -> Optional[str]:
    return _device_id_from_meta_dict(_parse_meta_dict(meta_json))


def _collect_agent_hardware_from_meta(meta_json: Optional[str]) -> Optional[str]:
    return _hardware_from_meta_dict(_parse_meta_dict(meta_json))


def effective_collect_device_id_for_sync(
    meta_json: Optional[str],
    asset_device_id: Optional[str],
) -> Optional[str]:
    """资产表 device_id 与 meta.collect.device_id 合并（与单条同步入参一致）。"""
    cid = (asset_device_id or "").strip() or (_collect_device_id_from_meta(meta_json) or "").strip()
    return cid or None


async def resolve_sync_agent_id(
    device_db: AsyncSession,
    *,
    agent_id: Optional[str],
    meta_json: Optional[str],
    collect_device_id: Optional[str] = None,
) -> str:
    """
    隧道同步使用的采集端标识（等于设备 hardware_uuid / Agent agent_id）。
    与 sync_asset_via_agent 解析规则一致，供批量任务按 Agent 分桶并发，无需再解析 HTTP base_url。
    """
    cid = effective_collect_device_id_for_sync(meta_json, collect_device_id)
    resolved: Optional[str] = (agent_id or "").strip() or _collect_agent_hardware_from_meta(meta_json)
    if not resolved and cid:
        try:
            did = int(str(cid).strip())
        except Exception:
            did = None
        if did is not None:
            dev = await get_device_by_id(device_db, did)
            hw = (str(getattr(dev, "hardware_uuid", "") or "").strip() if dev else "") or ""
            resolved = hw or None
    resolved = (resolved or "").strip()
    if not resolved:
        raise RuntimeError(
            "无法确定采集端 agent_id（请保证数据资产已写入设备ID（采集保存时自动带入），"
            "或传入 query 参数 agent_id（等于目标设备的 hardware_uuid），"
            "或在 meta.collect 中写入 hardware_uuid / device_id。"
        )
    nid = normalize_mac_like_agent_id(resolved)
    if is_colon_mac_normalized(nid):
        return nid
    return resolved


def _registry_base_url_for_hint(hint: Optional[str]) -> Optional[str]:
    if hint:
        info = agent_registry.get_by_id(hint)
        if info and info.base_url:
            host = str(getattr(info, "host", "") or "").strip().lower()
            if host in {"127.0.0.1", "localhost", "0.0.0.0"} or host.startswith("127."):
                return None
            return info.base_url
    picked = _pick_agent(hint)
    if picked and picked.base_url:
        host = str(getattr(picked, "host", "") or "").strip().lower()
        if host in {"127.0.0.1", "localhost", "0.0.0.0"} or host.startswith("127."):
            return None
        return picked.base_url
    return None


async def resolve_agent_base_url_for_sync(
    device_db: AsyncSession,
    *,
    agent_id: Optional[str],
    meta_json: Optional[str],
    collect_device_id: Optional[str] = None,
) -> str:
    """
    解析采集端 HTTP 根地址：
    0) 数据资产 device_id 或 meta.collect.device_id → devices 表
    1) 内存 AgentRegistry（设备连接或 Agent 自注册）
    2) 设备表 agent_ip / agent_port（设备连接成功时由平台落库，后端重启后仍可用）
    """
    meta_parsed = _parse_meta_dict(meta_json)
    cid = (collect_device_id or "").strip() or (_device_id_from_meta_dict(meta_parsed) or "").strip()
    if cid:
        try:
            did = int(cid)
        except ValueError:
            did = None
        if did is not None:
            dev = await get_device_by_id(device_db, did)
            if dev and dev.agent_ip and dev.agent_port:
                hip = str(dev.agent_ip).strip().lower()
                # devices 表里若误写了 127.0.0.1/localhost（常见于 Agent 自报），
                # 对平台是不可达地址，应跳过继续走 registry / 其他解析路径
                if hip in {"127.0.0.1", "localhost", "0.0.0.0"} or hip.startswith("127."):
                    logger.warning(
                        "data_sync: devices 表 agent_ip 为本地回环地址，已跳过 device_id=%s agent_ip=%r agent_port=%r",
                        did,
                        str(dev.agent_ip),
                        dev.agent_port,
                    )
                else:
                    u = f"http://{str(dev.agent_ip).strip()}:{int(dev.agent_port)}"
                    logger.info("data_sync: 使用 devices 表记录解析采集端 url=%s device_id=%s", u, did)
                    return u

    hints: list[str] = []
    q = (agent_id or "").strip()
    if q:
        hints.append(q)
    from_meta = _hardware_from_meta_dict(meta_parsed)
    if from_meta and from_meta not in hints:
        hints.append(from_meta)

    for h in hints:
        url = _registry_base_url_for_hint(h or None)
        if url:
            return url

    for h in hints:
        dev = await get_device_by_hardware_uuid(device_db, h)
        if dev and dev.agent_ip and dev.agent_port:
            hip = str(dev.agent_ip).strip().lower()
            if hip in {"127.0.0.1", "localhost", "0.0.0.0"} or hip.startswith("127."):
                continue
            return f"http://{str(dev.agent_ip).strip()}:{int(dev.agent_port)}"

    url = _registry_base_url_for_hint(None)
    if url:
        return url

    devices, _ = await get_devices(device_db, skip=0, limit=500)
    candidates = []
    for d in devices:
        if not getattr(d, "agent_ip", None) or not getattr(d, "agent_port", None):
            continue
        hip = str(getattr(d, "agent_ip", "") or "").strip().lower()
        if hip in {"127.0.0.1", "localhost", "0.0.0.0"} or hip.startswith("127."):
            continue
        candidates.append(d)
    if len(candidates) == 1:
        d = candidates[0]
        return f"http://{str(d.agent_ip).strip()}:{int(d.agent_port)}"
    if len(candidates) > 1:
        logger.error(
            "data_sync: 多台设备均含 agent_ip/agent_port，无法自动选择。hints=%s collect_device_id=%r",
            hints,
            (collect_device_id or "").strip() or None,
        )
        raise RuntimeError(
            "存在多台已记录采集端 IP 的设备，无法自动选择用于同步。"
            "请保证数据资产已写入设备ID（采集保存时自动带入），"
            "或传入 query 参数 agent_id（等于目标设备的 hardware_uuid），"
            "或在 meta.collect 中写入 hardware_uuid / device_id。"
        )

    logger.error(
        "data_sync: 无法解析采集端 base_url。hints=%s collect_device_id=%r",
        hints,
        (collect_device_id or "").strip() or None,
    )
    raise RuntimeError(
        "未找到可用的采集端地址。请先在平台「设备」中完成采集端连接（成功后平台会保存 IP 与端口），"
        "并确保该记录未被删除。"
    )


async def sync_asset_via_agent(
    device_db: AsyncSession,
    *,
    asset_id: int,
    source_path: str,
    project_id: str,
    project_name: str,
    agent_id: Optional[str] = None,
    meta_json: Optional[str] = None,
    collect_device_id: Optional[str] = None,
    cancel_task_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    由采集端 Agent 读取本机 source_path，按平台下发的 MinIO 配置写入服务器 MinIO。
    平台不直接读采集盘，仅转发凭证与对象前缀。
    """
    # 隧道转发模式：按 agent_id（hardware_uuid）路由，不依赖采集端 IP。
    cid_merged = effective_collect_device_id_for_sync(meta_json, collect_device_id)
    resolved_agent_id = await resolve_sync_agent_id(
        device_db,
        agent_id=agent_id,
        meta_json=meta_json,
        collect_device_id=cid_merged,
    )

    bucket = project_bucket_name(project_name or project_id)
    object_prefix = f"projects/{project_id}/collect/{asset_id}"
    ep_norm = _minio_endpoint_for_agent_payload(resolved_agent_id)
    logger.debug(
        "data_sync: 下发 MinIO 至采集端 asset_id=%s agent_id=%s minio_endpoint=%r",
        asset_id,
        resolved_agent_id,
        ep_norm,
    )
    payload: Dict[str, Any] = {
        "asset_id": asset_id,
        "source_path": source_path,
        "bucket_name": bucket,
        "object_prefix": object_prefix,
        "minio_endpoint": ep_norm,
        "minio_access_key": (settings.MINIO_ACCESS_KEY or "").strip(),
        "minio_secret_key": (settings.MINIO_SECRET_KEY or "").strip(),
        "minio_secure": bool(settings.MINIO_SECURE),
    }
    if not payload["minio_endpoint"] or not payload["minio_access_key"] or not payload["minio_secret_key"]:
        logger.error("data_sync: MinIO 未配置 asset_id=%s", asset_id)
        raise RuntimeError("MinIO 未配置，请设置 MINIO_ENDPOINT/MINIO_ACCESS_KEY/MINIO_SECRET_KEY")

    logger.info(
        "data_sync: (tunnel) 请求采集端同步 asset_id=%s project_id=%s agent_id=%s bucket=%s prefix=%s source_path=%r",
        asset_id,
        project_id,
        resolved_agent_id,
        bucket,
        object_prefix,
        source_path,
    )

    try:
        if cancel_task_id and is_cancelled(cancel_task_id):
            raise RuntimeError("已取消")
        result: Dict[str, Any] | None = None
        did_int: Optional[int] = None
        if cid_merged:
            try:
                did_int = int(str(cid_merged).strip())
            except Exception:
                did_int = None
        socket_key = await agent_tunnel_manager.resolve_connected_socket_key(
            resolved_agent_id,
            platform_device_id=did_int,
        )
        if socket_key:
            # 通过隧道发送命令，让采集端执行与 HTTP /api/agent/data/sync 同样的同步逻辑
            result = await agent_tunnel_manager.send_cmd_and_wait(
                agent_id=socket_key,
                cmd="DATA_SYNC",
                payload=payload,
                timeout_sec=1800.0,
                retry_times=0,
            )
        else:
            # 关键修复（方案 A）：
            # RQ worker 进程不共享 Web 进程的隧道内存态连接；本进程查不到连接时，
            # 回环调用 Web 进程内部接口 /api/agent/tunnel/cmd 代发命令。
            internal_base = str(os.getenv("EAI_INTERNAL_API_BASE", "") or "").strip() or "http://127.0.0.1:8000"
            internal_base = internal_base.rstrip("/")
            url = f"{internal_base}/api/agent/tunnel/cmd"
            try:
                async with httpx.AsyncClient(timeout=1810.0) as client:
                    resp = await client.post(
                        url,
                        json={
                            "agent_id": resolved_agent_id,
                            "cmd": "DATA_SYNC",
                            "payload": payload,
                            "timeout_sec": 1800.0,
                            "retry_times": 0,
                            "platform_device_id": did_int,
                        },
                    )
                resp.raise_for_status()
                data = resp.json() if resp.content else {}
                if isinstance(data, dict):
                    result = data
                else:
                    result = {"success": False, "msg": "采集端返回格式错误（非 dict）"}
            except Exception as exc:
                result = {
                    "success": False,
                    "msg": f"采集端隧道未连接（agent_id={resolved_agent_id}）; internal_call_failed: {exc}",
                }
    except asyncio.CancelledError as e:
        raise RuntimeError("已取消") from e
    except asyncio.TimeoutError as e:
        raise RuntimeError("连接采集端超时（30 分钟内未完成）：隧道命令执行超时") from e
    except Exception as e:
        raise

    if not isinstance(result, dict):
        raise RuntimeError("采集端返回格式错误（非 dict）")
    if not bool(result.get("success", False)):
        msg = str(result.get("msg") or result.get("message") or "Agent 同步失败")
        # 兼容旧采集端未实现 DATA_SYNC：可选回退到 HTTP（默认关闭，避免“又依赖 IP”）
        allow_fallback = str(os.getenv("EAI_SYNC_HTTP_FALLBACK", "false") or "").strip().lower() in {"1","true","yes","on"}
        if allow_fallback and ("Unsupported cmd" in msg or "隧道未连接" in msg):
            base_url = (await resolve_agent_base_url_for_sync(
                device_db,
                agent_id=resolved_agent_id,
                meta_json=meta_json,
                collect_device_id=cid_merged,
            )).rstrip("/")
            sync_url = f"{base_url}/api/agent/data/sync"
            logger.warning("data_sync: tunnel 失败，回退 HTTP asset_id=%s agent_id=%s url=%s err=%r", asset_id, resolved_agent_id, sync_url, msg[:200])
            try:
                async with httpx.AsyncClient(timeout=1800.0) as client:
                    resp = await client.post(sync_url, json=payload)
                resp.raise_for_status()
                data = resp.json() if resp.content else {}
                if isinstance(data, dict) and data.get("ok") and str(data.get("minio_path") or "").startswith("minio://"):
                    return {"agent_id": resolved_agent_id or "", "minio_path": str(data.get("minio_path") or "").strip()}
                raise RuntimeError(str((data.get("message") if isinstance(data, dict) else "") or "HTTP 同步失败"))
            except Exception as exc:
                raise RuntimeError(f"{msg}；HTTP 回退也失败：{exc}") from exc
        raise RuntimeError(msg)
    minio_path = str(result.get("minio_path") or "").strip()
    if not minio_path.startswith("minio://"):
        logger.error(
            "data_sync: Agent 返回的 minio_path 无效 asset_id=%r path=%r",
            asset_id,
            _snippet(minio_path, 200),
        )
        raise RuntimeError("Agent 未返回有效 minio_path")
    logger.info("data_sync: 同步成功 asset_id=%s minio_path=%r", asset_id, _snippet(minio_path, 300))
    return {"agent_id": resolved_agent_id or "", "minio_path": minio_path}
