"""
设备管理 API 路由
"""
from fastapi import APIRouter, Body, Depends, HTTPException, Path as PathParam, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from typing import List, Optional
from pathlib import Path
from datetime import datetime
import json
import logging
import asyncio
import traceback
import time
import signal
import subprocess
import os
import re
import tempfile

import httpx

from app.db.session import get_db
from app.core.deps import require_admin_async, get_current_user
from app.models import User
from app.crud.device import (
    get_devices,
    get_device_by_id,
    get_device_by_hardware_uuid,
    create_device,
    update_device,
    delete_device,
    create_test_result,
    get_latest_test_result,
)
from app.schemas.device import (
    DeviceCreate,
    DeviceUpdate,
    DeviceResponse,
    TestConnectionRequest,
    TestConnectionResponse,
    DeviceTestResultBase,
    DeviceLaunchConfigCreate,
    DeviceLaunchConfigUpdate,
    DeviceConnectRequest,
    DeviceConnectByAgentRequest,
    ScanCollectScriptRequest,
)
from app.schemas.common import ApiResponse
from app.core.config import settings
from app.services.ros2_connection_tester import test_ros2_connection
from app.services.agent_registry import agent_registry, AgentInfo
from app.services.agent_tunnel_manager import agent_tunnel_manager
from app.services.audit_service import log_audit_safe
from app.constants import audit_actions as AA
from app.constants import audit_resources as AR

logger = logging.getLogger(__name__)

router = APIRouter()

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _is_local_only_host(host: str | None) -> bool:
    h = (host or "").strip().lower()
    if not h:
        return True
    if h in ("0.0.0.0", "localhost", "127.0.0.1"):
        return True
    if h.startswith("127."):
        return True
    return False


def _resolve_agent_with_autobind(*, device: DeviceResponse | object, device_id: int) -> AgentInfo | None:
    """
    自动回填映射：
    - 先按严格映射找 device_id -> agent
    - 若缺失，尝试用 devices.hardware_uuid（这里存 agent_id）回填到 AgentRegistry
    """
    try:
        a = agent_registry.get_by_device_id_strict(int(device_id))
        if a is not None:
            return a
    except Exception:
        pass

    hw = str(getattr(device, "hardware_uuid", "") or "").strip()
    if not hw:
        return None
    candidate = agent_registry.get_by_id(hw)
    if candidate is None:
        return None
    try:
        agent_registry.bind_device_to_agent(device_id=int(device_id), agent_id=hw)
    except Exception:
        pass
    return candidate


async def _agent_tunnel_connected_for_device(device) -> Optional[bool]:
    """
    能解析出采集端 agent_id 时返回隧道是否连通；否则 None（纯 ROS 配置设备，连接态仍看测试结果）。
    """
    try:
        did = int(getattr(device, "id", 0) or 0)
    except (TypeError, ValueError):
        return None
    if not did:
        return None
    ag = _resolve_agent_with_autobind(device=device, device_id=did)
    hw = str(getattr(device, "hardware_uuid", "") or "").strip()
    aid = ""
    if ag and getattr(ag, "agent_id", None):
        aid = str(ag.agent_id).strip()
    elif hw:
        aid = hw
    else:
        return None
    try:
        return bool(await agent_tunnel_manager.has_connection(aid, platform_device_id=did))
    except Exception:
        return False


async def _geoip_lookup(ip: str) -> dict:
    """
    IP 地理位置查询（最佳努力）。

    当前实现使用公开的 ip-api.com（无密钥，有限速率）。
    若查询失败则返回 {}，不影响设备连接流程。
    """
    ip = (ip or "").strip()
    if not ip:
        return {}

    # 避免解析掉内网 IP 时请求外部服务（按需放开）
    private_prefixes = ("10.", "172.16.", "172.17.", "172.18.", "172.19.", "172.20.", "172.21.", "172.22.", "172.23.", "172.24.", "172.25.", "172.26.", "172.27.", "172.28.", "172.29.", "172.30.", "172.31.", "192.168.")
    if ip.startswith(private_prefixes) or ip == "127.0.0.1":
        return {"note": "private ip"}

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            # 仅请求我们需要的字段
            resp = await client.get(
                f"http://ip-api.com/json/{ip}?fields=status,message,country,regionName,city,lat,lon,timezone,isp",
            )
            resp.raise_for_status()
            data = resp.json() if resp.content else {}
            if not isinstance(data, dict) or data.get("status") != "success":
                return {"note": "geoip failed"}
            return {
                "country": data.get("country"),
                "region": data.get("regionName"),
                "city": data.get("city"),
                "lat": data.get("lat"),
                "lon": data.get("lon"),
                "timezone": data.get("timezone"),
                "isp": data.get("isp"),
            }
    except Exception:
        return {}



@router.get("/health")
async def health_check():
    """设备API健康检查（PostgreSQL）"""
    try:
        from app.db.session import engine
        async with engine.connect() as conn:
            result = await conn.execute(
                text(
                    "SELECT 1 FROM information_schema.tables "
                    "WHERE table_schema = 'public' AND table_name = 'devices'"
                )
            )
            table_exists = result.fetchone() is not None
        return {
            "ok": True,
            "table_exists": table_exists,
            "message": "设备API正常" if table_exists else "设备表未创建，请执行迁移或重启后端"
        }
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def safe_json_loads(json_str: str | None, default=None):
    """安全地解析 JSON 字符串"""
    if json_str is None:
        return default if default is not None else []
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return default if default is not None else []



@router.get("", response_model=ApiResponse)
async def list_devices(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """获取设备列表"""
    try:
        devices, total = await get_devices(db, skip=skip, limit=limit)

        # 设备归属团队可见性：
        # - 超级管理员：可见全库
        # - 其他用户：仅可见自己可访问的 team_id 所绑定设备
        from app.core.roles import is_super_admin
        from app.crud import team as team_crud

        if not is_super_admin(getattr(current_user, "role", None)):
            accessible_team_ids = await team_crud.list_team_ids_accessible_by_user(
                db, str(getattr(current_user, "id", "") or "")
            )
            if not accessible_team_ids:
                return ApiResponse(ok=True, data=[])

            devices = [
                d
                for d in devices
                if getattr(d, "team_id", None) is not None
                and str(getattr(d, "team_id", "")) in accessible_team_ids
            ]

        # 团队名称映射（用于列表页展示）
        team_map: dict[str, str] = {}
        try:
            team_ids: set[str] = set()
            for d in devices:
                tid = getattr(d, "team_id", None)
                if tid is None:
                    continue
                tid_s = str(tid).strip()
                if tid_s:
                    team_ids.add(tid_s)
            if team_ids:
                from sqlalchemy import select as sql_select
                from app.models.team import Team
                res = await db.execute(sql_select(Team).where(Team.id.in_(list(team_ids))))
                for t in res.scalars().all():
                    if t and t.id:
                        team_map[str(t.id)] = t.name
        except Exception:
            team_map = {}
        
        # 转换为响应格式，包含最新测试结果
        # GeoIP 最佳努力：同一请求内做简单缓存，避免重复请求同一 IP
        location_cache: dict[str, dict] = {}
        device_responses = []
        for device in devices:
            try:
                # 获取最新测试结果
                latest_test = None
                if device.test_results:
                    latest_test = device.test_results[0] if device.test_results else None
                
                # 构建响应
                device_dict = {
                    "id": device.id,
                    "name": device.name,
                    "vendor": device.vendor,
                    "model": device.model,
                    "device_type": device.device_type,
                    "created_at": device.created_at,
                    "updated_at": device.updated_at,
                    "hardware_uuid": device.hardware_uuid,
                    "team_id": getattr(device, "team_id", None),
                    "team_name": team_map.get(str(getattr(device, "team_id", "") or "").strip()) if getattr(device, "team_id", None) else None,
                    "hostname": device.hostname,
                    "agent_ip": device.agent_ip,
                    "agent_port": device.agent_port,
                    "agent_status": device.agent_status,
                    "camera_list": safe_json_loads(device.camera_list_json, []),
                    "collect_script_compress": getattr(device, "collect_script_compress", None),
                    "collect_script_raw": getattr(device, "collect_script_raw", None),
                }
                # 运行状态：从 agent_registry 中推断
                runtime_status = None
                agent_for_device: AgentInfo | None = None
                try:
                    agent = agent_registry.get_by_device_id(device.id)
                    if agent:
                        agent_for_device = agent
                        runtime_status = agent.runtime_status
                except Exception:
                    runtime_status = None
                if runtime_status:
                    device_dict["runtime_status"] = runtime_status

                # 兜底：若数据库里没有保存 agent_ip/port（例如旧数据），则从内存 AgentRegistry 回填
                if agent_for_device and not device_dict.get("agent_ip"):
                    device_dict["agent_ip"] = agent_for_device.host
                    device_dict["agent_port"] = agent_for_device.port
                    # hostname 只有做显示用途，优先保持数据库值
                    if not device_dict.get("hostname"):
                        device_dict["hostname"] = agent_for_device.name

                # 返回“所在地”：若当前响应里没有 location，则基于 agent_ip 做 GeoIP（最佳努力）
                agent_ip = device_dict.get("agent_ip")
                if agent_ip and not device_dict.get("location"):
                    if agent_ip not in location_cache:
                        location_cache[agent_ip] = await _geoip_lookup(str(agent_ip))
                    device_dict["location"] = location_cache.get(agent_ip)
                
                # ROS2 配置
                if device.ros2_config:
                    peer_ips = safe_json_loads(device.ros2_config.peer_ips_json, [])
                    device_dict["ros2_config"] = {
                        "mode": device.ros2_config.mode,
                        "local_bind_ip": device.ros2_config.local_bind_ip,
                        "domain_id": device.ros2_config.domain_id,
                        "discovery_protocol": device.ros2_config.discovery_protocol,
                        "initial_announcements_count": device.ros2_config.initial_announcements_count,
                        "initial_announcements_period_sec": device.ros2_config.initial_announcements_period_sec,
                        "profile_path": device.ros2_config.profile_path,
                        "peer_ips": peer_ips
                    }

                # 启动配置
                if device.launch_config:
                    env_vars = safe_json_loads(device.launch_config.env_vars_json, {})
                    device_dict["launch_config"] = {
                        "id": device.launch_config.id,
                        "script_path": device.launch_config.script_path,
                        "script_args": device.launch_config.script_args,
                        "stop_script_path": getattr(device.launch_config, "stop_script_path", None),
                        "stop_script_args": getattr(device.launch_config, "stop_script_args", None),
                        "env_vars": env_vars
                    }
                
                # 最新测试结果
                if latest_test:
                    nodes_sample = safe_json_loads(latest_test.nodes_sample_json, [])
                    topics_sample = safe_json_loads(latest_test.topics_sample_json, [])
                    device_dict["last_test_result"] = {
                        "status": latest_test.status,
                        "node_count": latest_test.node_count,
                        "nodes_sample": nodes_sample,
                        "topic_count": latest_test.topic_count,
                        "topics_sample": topics_sample,
                        "error_type": latest_test.error_type,
                        "error_message": latest_test.error_message,
                        "tested_at": latest_test.tested_at
                    }

                device_dict["agent_tunnel_connected"] = await _agent_tunnel_connected_for_device(device)

                device_responses.append(DeviceResponse(**device_dict))
            except Exception as e:
                # 记录单个设备处理错误，但继续处理其他设备
                import traceback
                print(f"处理设备 {device.id} 时出错: {e}")
                print(traceback.format_exc())
                continue
        
        return ApiResponse(ok=True, data=device_responses)
    except Exception as e:
        import traceback
        error_msg = f"获取设备列表失败: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_msg)


@router.get("/{device_id}", response_model=ApiResponse)
async def get_device(
    device_id: int = PathParam(..., description="设备ID"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """获取设备详情"""
    device = await get_device_by_id(db, device_id)
    if not device:
        raise HTTPException(status_code=404, detail="设备不存在")

    # 设备可见性：仅可见归属团队
    from app.core.roles import is_super_admin
    from app.crud import team as team_crud

    if not is_super_admin(getattr(current_user, "role", None)):
        accessible_team_ids = await team_crud.list_team_ids_accessible_by_user(
            db, str(getattr(current_user, "id", "") or "")
        )
        if not accessible_team_ids or not device.team_id or str(device.team_id) not in accessible_team_ids:
            # 按“不可见等同不存在”返回 404，避免信息泄露
            raise HTTPException(status_code=404, detail="设备不存在")
    
    # 构建响应（与列表接口相同逻辑）
    team_name: str | None = None
    try:
        if getattr(device, "team_id", None):
            from app.crud.team import get_team_by_id
            team_obj = await get_team_by_id(db, str(getattr(device, "team_id", "") or "").strip())
            team_name = getattr(team_obj, "name", None)
    except Exception:
        team_name = None

    device_dict = {
        "id": device.id,
        "name": device.name,
        "vendor": device.vendor,
        "model": device.model,
        "device_type": device.device_type,
        "created_at": device.created_at,
        "updated_at": device.updated_at,
        "hardware_uuid": device.hardware_uuid,
        "team_id": getattr(device, "team_id", None),
        "team_name": team_name,
        "hostname": device.hostname,
        "agent_ip": device.agent_ip,
        "agent_port": device.agent_port,
        "agent_status": device.agent_status,
        "camera_list": safe_json_loads(device.camera_list_json, []),
        "collect_script_compress": getattr(device, "collect_script_compress", None),
        "collect_script_raw": getattr(device, "collect_script_raw", None),
    }
    # 运行状态：从 agent_registry 中推断
    runtime_status = None
    agent_for_device: AgentInfo | None = None
    try:
        agent = agent_registry.get_by_device_id(device.id)
        if agent:
            agent_for_device = agent
            runtime_status = agent.runtime_status
    except Exception:
        runtime_status = None
    if runtime_status:
        device_dict["runtime_status"] = runtime_status

    # 兜底：若数据库里没有保存 agent_ip/port，则从内存 AgentRegistry 回填
    if agent_for_device and not device_dict.get("agent_ip"):
        device_dict["agent_ip"] = agent_for_device.host
        device_dict["agent_port"] = agent_for_device.port
        if not device_dict.get("hostname"):
            device_dict["hostname"] = agent_for_device.name

    # 返回“所在地”：若当前响应里没有 location，则基于 agent_ip 做 GeoIP（最佳努力）
    agent_ip = device_dict.get("agent_ip")
    if agent_ip and not device_dict.get("location"):
        device_dict["location"] = await _geoip_lookup(str(agent_ip))
    
    if device.ros2_config:
        peer_ips = safe_json_loads(device.ros2_config.peer_ips_json, [])
        device_dict["ros2_config"] = {
            "mode": device.ros2_config.mode,
            "local_bind_ip": device.ros2_config.local_bind_ip,
            "domain_id": device.ros2_config.domain_id,
            "discovery_protocol": device.ros2_config.discovery_protocol,
            "initial_announcements_count": device.ros2_config.initial_announcements_count,
            "initial_announcements_period_sec": device.ros2_config.initial_announcements_period_sec,
            "profile_path": device.ros2_config.profile_path,
            "peer_ips": peer_ips
        }

    if device.launch_config:
        env_vars = safe_json_loads(device.launch_config.env_vars_json, {})
        device_dict["launch_config"] = {
            "id": device.launch_config.id,
            "script_path": device.launch_config.script_path,
            "script_args": device.launch_config.script_args,
            "stop_script_path": getattr(device.launch_config, "stop_script_path", None),
            "stop_script_args": getattr(device.launch_config, "stop_script_args", None),
            "env_vars": env_vars
        }
    
    latest_test = device.test_results[0] if device.test_results else None
    if latest_test:
        nodes_sample = safe_json_loads(latest_test.nodes_sample_json, [])
        topics_sample = safe_json_loads(latest_test.topics_sample_json, [])
        device_dict["last_test_result"] = {
            "status": latest_test.status,
            "node_count": latest_test.node_count,
            "nodes_sample": nodes_sample,
            "topic_count": latest_test.topic_count,
            "topics_sample": topics_sample,
            "error_type": latest_test.error_type,
            "error_message": latest_test.error_message,
            "tested_at": latest_test.tested_at
        }

    device_dict["agent_tunnel_connected"] = await _agent_tunnel_connected_for_device(device)

    return ApiResponse(ok=True, data=DeviceResponse(**device_dict))


@router.post("", response_model=ApiResponse)
async def create_new_device(
    request: Request,
    device: DeviceCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin_async),
):
    """创建设备（普通成员只读，禁止创建）"""
    device.name = (device.name or "").strip()
    device.vendor = (device.vendor or "").strip() or None
    device.model = (device.model or "").strip() or None
    device.hardware_uuid = (getattr(device, "hardware_uuid", None) or "").strip() or None
    device.hostname = (getattr(device, "hostname", None) or "").strip() or None
    device.agent_ip = (getattr(device, "agent_ip", None) or "").strip() or None
    if device.agent_port is not None:
        try:
            device.agent_port = int(device.agent_port)
        except Exception:
            device.agent_port = None
    if not device.hardware_uuid:
        raise HTTPException(
            status_code=400,
            detail="必须先完成 Agent 安装并建立通信连接，然后使用在线 Agent 的 agent_id 作为设备唯一标识（hardware_uuid）",
        )

    aid = str(device.hardware_uuid).strip()
    agent = agent_registry.get_by_id(aid)
    if agent is None:
        raise HTTPException(status_code=409, detail="Agent 未在线：请先完成安装并确保已与平台建立连接")
    stale_after = float(getattr(settings, "AGENT_TUNNEL_OFFLINE_AFTER_SEC", 45.0) or 45.0)
    last_ts = agent_tunnel_manager.get_last_seen_ts(aid)
    if last_ts is None or (time.time() - float(last_ts)) > stale_after:
        raise HTTPException(status_code=409, detail="Agent 通信连接不可用（隧道未连接或已超时）：请检查网络并重试")
    if not await agent_tunnel_manager.has_connection(aid):
        raise HTTPException(status_code=409, detail="Agent 通信连接不可用（隧道未连接）：请检查网络并重试")
    if not device.agent_ip:
        # 仅在 AgentRegistry 给出“可被平台访问”的地址时才自动回填；
        # 避免将 127.0.0.1/localhost 这种“对采集端本机成立，对平台无意义”的值写入 devices 表。
        cand = (getattr(agent, "host", None) or "").strip() or None
        if cand and not _is_local_only_host(cand):
            device.agent_ip = cand
    if not device.agent_port:
        try:
            device.agent_port = int(getattr(agent, "port", None) or 0) or None
        except Exception:
            device.agent_port = None

    if device.agent_ip and _is_local_only_host(device.agent_ip):
        raise HTTPException(
            status_code=400,
            detail="采集端地址填写无效：agent_ip 不能是 127.0.0.1/localhost。请在设备里填写采集端所在机器的可达 IP（平台能访问到的内网 IP）并重试。",
        )

    # 针对“一湃智能”设备自动配置启动脚本
    if device.name == "一湃智能":
        if not device.launch_config:
            device.launch_config = DeviceLaunchConfigCreate(
                script_path="/home/sia/workspace/test/start_all_ros_topics.sh",
                stop_script_path="/home/sia/workspace/test/stop_all_ros_topics.sh",
            )
            print(f"✅ 已自动为设备 {device.name} 配置启动脚本 (compressed)")

    # 验证 ROS2 配置
    # (已移除 fastdds_tailscale_peer 强制验证逻辑)

    # 归属团队绑定：按“发起添加该设备的账号”所在团队来绑定
    from app.core.roles import is_super_admin
    from app.crud import team as team_crud

    binding_team_id: str | None = None
    if not is_super_admin(getattr(current_user, "role", None)):
        team_ids = await team_crud.list_team_ids_accessible_by_user(
            db, str(getattr(current_user, "id", "") or "")
        )
        if not team_ids:
            raise HTTPException(status_code=403, detail="当前账号未关联团队，无法添加设备")
        binding_team_id = team_ids[0]
    
    # 去重：若硬件 UUID 已存在，则复用该设备（并在首次被团队添加时补齐 team_id）
    existing_device = None
    if getattr(device, "hardware_uuid", None):
        existing_device = await get_device_by_hardware_uuid(db, device.hardware_uuid)

    if existing_device:
        if not is_super_admin(getattr(current_user, "role", None)):
            if existing_device.team_id and str(existing_device.team_id) != str(binding_team_id):
                raise HTTPException(status_code=403, detail="该设备已被其他团队添加，无法重复添加")
            if not existing_device.team_id and binding_team_id:
                existing_device.team_id = binding_team_id
                db.add(existing_device)
                await db.commit()
                await db.refresh(existing_device)
        db_device = existing_device
    else:
        # 创建设备
        if not is_super_admin(getattr(current_user, "role", None)) and binding_team_id:
            device.team_id = binding_team_id
        db_device = await create_device(db, device)
    log_audit_safe(
        user=current_user,
        action_type=AA.CREATE_DEVICE,
        resource_type=AR.DEVICE,
        resource_id=str(getattr(db_device, "id", "") or ""),
        resource_name=str(getattr(db_device, "name", "") or ""),
        team_id=str(getattr(db_device, "team_id", "") or "") or None,
        result="SUCCESS",
        detail_json={
            "hardware_uuid": getattr(db_device, "hardware_uuid", None),
            "agent_ip": getattr(db_device, "agent_ip", None),
            "agent_port": getattr(db_device, "agent_port", None),
        },
        request=request,
    )
    
    # 构建响应
    try:
        team_name = None
        try:
            if getattr(db_device, "team_id", None):
                team_obj = await team_crud.get_team_by_id(db, str(getattr(db_device, "team_id", "") or "").strip())
                team_name = getattr(team_obj, "name", None) if team_obj else None
        except Exception:
            team_name = None

        device_dict = {
            "id": db_device.id,
            "name": db_device.name,
            "vendor": db_device.vendor,
            "model": db_device.model,
            "device_type": db_device.device_type,
            "created_at": db_device.created_at.isoformat() if hasattr(db_device.created_at, 'isoformat') else str(db_device.created_at),
            "updated_at": db_device.updated_at.isoformat() if hasattr(db_device.updated_at, 'isoformat') else str(db_device.updated_at),
            "hardware_uuid": db_device.hardware_uuid,
            "hostname": db_device.hostname,
            "agent_ip": db_device.agent_ip,
            "agent_port": db_device.agent_port,
            "agent_status": db_device.agent_status,
            "camera_list": safe_json_loads(db_device.camera_list_json, []),
            "collect_script_compress": getattr(db_device, "collect_script_compress", None),
            "collect_script_raw": getattr(db_device, "collect_script_raw", None),
            "team_id": getattr(db_device, "team_id", None),
            "team_name": team_name,
        }
        
        # 使用 hasattr 检查，避免触发懒加载
        ros2_config = None
        try:
            ros2_config = db_device.ros2_config
        except:
            pass

        if ros2_config:
            peer_ips = safe_json_loads(ros2_config.peer_ips_json, [])
            device_dict["ros2_config"] = {
                "mode": ros2_config.mode,
                "local_bind_ip": ros2_config.local_bind_ip,
                "domain_id": ros2_config.domain_id,
                "discovery_protocol": ros2_config.discovery_protocol,
                "initial_announcements_count": ros2_config.initial_announcements_count,
                "initial_announcements_period_sec": ros2_config.initial_announcements_period_sec,
                "profile_path": ros2_config.profile_path,
                "peer_ips": peer_ips
            }

        if db_device.launch_config:
            env_vars = safe_json_loads(db_device.launch_config.env_vars_json, {})
            device_dict["launch_config"] = {
                "id": db_device.launch_config.id,
                "script_path": db_device.launch_config.script_path,
                "script_args": db_device.launch_config.script_args,
                "stop_script_path": getattr(db_device.launch_config, "stop_script_path", None),
                "stop_script_args": getattr(db_device.launch_config, "stop_script_args", None),
                "env_vars": env_vars
            }
        
        # 直接返回字典，不使用Pydantic模型，避免序列化问题
        return ApiResponse(ok=True, data=device_dict)
    except Exception as e:
        import traceback
        error_msg = f"构建响应失败: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_msg)


@router.put("/{device_id}", response_model=ApiResponse)
async def update_existing_device(
    device_id: int = PathParam(..., description="设备ID"),
    device_update: DeviceUpdate = ...,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin_async),
):
    """更新设备（普通成员只读，禁止修改）"""
    db_device = await get_device_by_id(db, device_id)
    if not db_device:
        raise HTTPException(status_code=404, detail="设备不存在")

    # 设备可见性/可编辑范围：非超管仅可操作归属团队设备
    from app.core.roles import is_super_admin
    from app.crud import team as team_crud

    if not is_super_admin(getattr(current_user, "role", None)):
        accessible_team_ids = await team_crud.list_team_ids_accessible_by_user(
            db, str(getattr(current_user, "id", "") or "")
        )
        if not accessible_team_ids or not db_device.team_id or str(db_device.team_id) not in accessible_team_ids:
            raise HTTPException(status_code=404, detail="设备不存在")
    
    # 验证 ROS2 配置
    # (已移除 fastdds_tailscale_peer 强制验证逻辑)
    # if device_update.ros2_config:
    #     if device_update.ros2_config.mode == "fastdds_tailscale_peer":
    #         ...
    
    # 更新设备
    updated_device = await update_device(db, device_id, device_update)
    if not updated_device:
        raise HTTPException(status_code=404, detail="设备不存在")
    
    # 构建响应
    team_name = None
    try:
        if getattr(updated_device, "team_id", None):
            team_obj = await team_crud.get_team_by_id(db, str(getattr(updated_device, "team_id", "") or "").strip())
            team_name = getattr(team_obj, "name", None) if team_obj else None
    except Exception:
        team_name = None

    device_dict = {
        "id": updated_device.id,
        "name": updated_device.name,
        "vendor": updated_device.vendor,
        "model": updated_device.model,
        "device_type": updated_device.device_type,
        "created_at": updated_device.created_at,
        "updated_at": updated_device.updated_at,
        "hardware_uuid": updated_device.hardware_uuid,
        "hostname": updated_device.hostname,
        "agent_ip": updated_device.agent_ip,
        "agent_port": updated_device.agent_port,
        "agent_status": updated_device.agent_status,
        "camera_list": safe_json_loads(updated_device.camera_list_json, []),
        "collect_script_compress": getattr(updated_device, "collect_script_compress", None),
        "collect_script_raw": getattr(updated_device, "collect_script_raw", None),
        "team_id": getattr(updated_device, "team_id", None),
        "team_name": team_name,
    }
    
    if updated_device.ros2_config:
        peer_ips = safe_json_loads(updated_device.ros2_config.peer_ips_json, [])
        device_dict["ros2_config"] = {
            "mode": updated_device.ros2_config.mode,
            "local_bind_ip": updated_device.ros2_config.local_bind_ip,
            "domain_id": updated_device.ros2_config.domain_id,
            "discovery_protocol": updated_device.ros2_config.discovery_protocol,
            "initial_announcements_count": updated_device.ros2_config.initial_announcements_count,
            "initial_announcements_period_sec": updated_device.ros2_config.initial_announcements_period_sec,
            "profile_path": updated_device.ros2_config.profile_path,
            "peer_ips": peer_ips
        }

    if updated_device.launch_config:
        env_vars = safe_json_loads(updated_device.launch_config.env_vars_json, {})
        device_dict["launch_config"] = {
            "id": updated_device.launch_config.id,
            "script_path": updated_device.launch_config.script_path,
            "script_args": updated_device.launch_config.script_args,
            "stop_script_path": getattr(updated_device.launch_config, "stop_script_path", None),
            "stop_script_args": getattr(updated_device.launch_config, "stop_script_args", None),
            "env_vars": env_vars
        }
    
    latest_test = updated_device.test_results[0] if updated_device.test_results else None
    if latest_test:
        nodes_sample = json.loads(latest_test.nodes_sample_json) if latest_test.nodes_sample_json else []
        topics_sample = json.loads(latest_test.topics_sample_json) if latest_test.topics_sample_json else []
        device_dict["last_test_result"] = {
            "status": latest_test.status,
            "node_count": latest_test.node_count,
            "nodes_sample": nodes_sample,
            "topic_count": latest_test.topic_count,
            "topics_sample": topics_sample,
            "error_type": latest_test.error_type,
            "error_message": latest_test.error_message,
            "tested_at": latest_test.tested_at
        }

    device_dict["agent_tunnel_connected"] = await _agent_tunnel_connected_for_device(updated_device)

    return ApiResponse(ok=True, data=DeviceResponse(**device_dict))


@router.delete("/{device_id}", response_model=ApiResponse)
async def delete_existing_device(
    device_id: int = PathParam(..., description="设备ID"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin_async),
):
    """删除设备（普通成员只读，禁止删除）"""
    # 设备可见性/可删除范围：非超管仅可操作归属团队设备
    from app.core.roles import is_super_admin
    from app.crud import team as team_crud

    db_device = await get_device_by_id(db, device_id)
    if not db_device:
        raise HTTPException(status_code=404, detail="设备不存在")

    if not is_super_admin(getattr(current_user, "role", None)):
        accessible_team_ids = await team_crud.list_team_ids_accessible_by_user(
            db, str(getattr(current_user, "id", "") or "")
        )
        if not accessible_team_ids or not db_device.team_id or str(db_device.team_id) not in accessible_team_ids:
            raise HTTPException(status_code=404, detail="设备不存在")

    success = await delete_device(db, device_id)
    if not success:
        raise HTTPException(status_code=404, detail="设备不存在")

    try:
        agent_registry.unregister_device(device_id)
    except Exception:
        pass

    # 配置文件目录清理逻辑已移除（不再生成配置文件）

    return ApiResponse(ok=True, data={"message": "设备已删除"})


@router.post("/{device_id}/test-connection", response_model=ApiResponse)
async def test_device_connection(
    device_id: int = PathParam(..., description="设备ID"),
    request: TestConnectionRequest = ...,
    db: AsyncSession = Depends(get_db)
):
    """测试设备连接（仅通过采集端 Agent，在采集端本机执行 ROS2 检测）"""
    device = await get_device_by_id(db, device_id)
    if not device:
        raise HTTPException(status_code=404, detail="设备不存在")
    
    test_result_dict = None

    # 1. 通过采集端 Agent 隧道执行“只读”测试（不在服务器本机执行 ROS2 命令）
    # 严格映射：避免未绑定时回退 local-agent，导致误报“隧道未连接”。
    agent: AgentInfo | None = _resolve_agent_with_autobind(device=device, device_id=device_id)

    resolved_aid = agent.agent_id if agent and agent.agent_id else None
    if not resolved_aid:
        raise HTTPException(
            status_code=400,
            detail="未找到在线采集端 Agent，请先在采集端启动 Agent 并绑定该设备ID",
        )
    if not await agent_tunnel_manager.has_connection(resolved_aid, platform_device_id=device_id):
        raise HTTPException(
            status_code=503,
            detail=(
                f"采集端隧道未连接（agent_id={resolved_aid}）。"
                "请确认边缘 Agent 已启动并成功连接 /api/agent/tunnel，"
                "且设备已通过 /api/devices/connect-agent 绑定到该 Agent。"
            ),
        )

    try:
        sk = await agent_tunnel_manager.resolve_connected_socket_key(
            resolved_aid, platform_device_id=device_id
        ) or resolved_aid
        result = await agent_tunnel_manager.send_cmd_and_wait(
            agent_id=sk,
            cmd="DEVICE_TEST_CONNECTION",
            payload={"device_id": device_id},
            timeout_sec=35.0,
            retry_times=1,
        )
        ok = bool(result.get("success", False))
        msg = result.get("msg")
        node_count = int(result.get("node_count") or 0)
        topic_count = int(result.get("topic_count") or 0)
        test_result_dict = {
            "status": "success" if ok else "fail",
            "node_count": node_count,
            "nodes_sample": [],
            "topic_count": topic_count,
            "topics_sample": [],
            "error_type": None if ok else "AGENT_TEST_ERROR",
            "error_message": msg,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"通过隧道调用采集端测试失败: {e}")
    
    # 保存测试结果
    test_result = DeviceTestResultBase(**test_result_dict)
    db_test_result = await create_test_result(db, device_id, test_result)
    
    # 构建响应
    nodes_sample = safe_json_loads(db_test_result.nodes_sample_json, [])
    topics_sample = safe_json_loads(db_test_result.topics_sample_json, [])
    
    from app.schemas.device import DeviceTestResultResponse
    result_response = DeviceTestResultResponse(
        status=db_test_result.status,
        node_count=db_test_result.node_count,
        nodes_sample=nodes_sample,
        topic_count=db_test_result.topic_count,
        topics_sample=topics_sample,
        error_type=db_test_result.error_type,
        error_message=db_test_result.error_message,
        tested_at=db_test_result.tested_at
    )
    
    return ApiResponse(
        ok=test_result_dict["status"] == "success",
        data=TestConnectionResponse(
            success=test_result_dict["status"] == "success",
            result=result_response,
            message=test_result_dict.get("error_message")
        )
    )


@router.post("/{device_id}/scan-collect-script", response_model=ApiResponse)
async def scan_device_collect_script(
    device_id: int = PathParam(..., description="设备ID"),
    body: ScanCollectScriptRequest = Body(...),
    db: AsyncSession = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    """
    在采集端读取设备配置的采集脚本并解析频率检测话题与脚本默认阈值。
    供创建/编辑任务时展示每话题频率标准输入框。
    """
    device = await get_device_by_id(db, device_id)
    if not device:
        raise HTTPException(status_code=404, detail="设备不存在")

    fmt = (body.camera_data_format or "压缩").strip()
    is_raw = fmt == "原始"
    script_path = (
        (device.collect_script_raw or "").strip()
        if is_raw
        else (device.collect_script_compress or "").strip()
    )
    if not script_path:
        raise HTTPException(
            status_code=400,
            detail="设备未配置采集脚本路径，请在采集端 Client 连接设备时填写压缩/原始脚本路径",
        )

    agent: AgentInfo | None = _resolve_agent_with_autobind(device=device, device_id=device_id)
    resolved_aid = agent.agent_id if agent and agent.agent_id else None
    if not resolved_aid:
        raise HTTPException(
            status_code=400,
            detail="未找到在线采集端 Agent，请先在采集端启动 Agent 并绑定该设备",
        )
    if not await agent_tunnel_manager.has_connection(resolved_aid, platform_device_id=device_id):
        raise HTTPException(
            status_code=503,
            detail=f"采集端隧道未连接（agent_id={resolved_aid}）",
        )

    try:
        sk = await agent_tunnel_manager.resolve_connected_socket_key(
            resolved_aid, platform_device_id=device_id
        ) or resolved_aid
        result = await agent_tunnel_manager.send_cmd_and_wait(
            agent_id=sk,
            cmd="SCAN_COLLECT_SCRIPT",
            payload={"script_path": script_path},
            timeout_sec=30.0,
            retry_times=1,
        )
        if not bool(result.get("success", False)):
            raise HTTPException(
                status_code=502,
                detail=result.get("msg") or "采集端扫描脚本失败",
            )
        data = result.get("data")
        if not isinstance(data, dict):
            raise HTTPException(status_code=502, detail="采集端返回格式不正确")
        return ApiResponse(ok=True, data=data)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"扫描采集脚本失败: {e}") from e


@router.post("/{device_id}/launch", response_model=ApiResponse)
async def launch_device(
    device_id: int = PathParam(..., description="设备ID"),
    db: AsyncSession = Depends(get_db)
):
    """启动设备（必须通过采集端 Agent 执行启动脚本）"""
    device = await get_device_by_id(db, device_id)
    if not device:
        raise HTTPException(status_code=404, detail="设备不存在")
    
    if not device.launch_config:
        raise HTTPException(status_code=400, detail="设备未配置启动脚本")
    
    raw_script_path = device.launch_config.script_path
    if not raw_script_path:
        raise HTTPException(status_code=400, detail="设备未配置启动脚本")

    # 启动脚本环境由采集端自身负责，此处不再传递服务器本机 env

    # 优先通过 Agent 调用设备启动（严格映射，不回退 local-agent）
    agent: AgentInfo | None = _resolve_agent_with_autobind(device=device, device_id=device_id)

    resolved_aid = agent.agent_id if agent and agent.agent_id else None
    if not resolved_aid:
        raise HTTPException(
            status_code=400,
            detail="未找到在线采集端 Agent，请先在采集端启动 Agent 并绑定该设备ID",
        )
    if not await agent_tunnel_manager.has_connection(resolved_aid, platform_device_id=device_id):
        raise HTTPException(
            status_code=503,
            detail=(
                f"采集端隧道未连接（agent_id={resolved_aid}）。"
                "请确认边缘 Agent 已启动并成功连接 /api/agent/tunnel，"
                "且设备已通过 /api/devices/connect-agent 绑定到该 Agent。"
            ),
        )

    try:
        launch_env: dict = {}
        try:
            raw_env = safe_json_loads(getattr(device.launch_config, "env_vars_json", None), {}) or {}
            if isinstance(raw_env, dict):
                launch_env = {str(k): str(v) for k, v in raw_env.items() if str(k).strip()}
        except Exception:
            launch_env = {}

        payload = {
            "script_path": raw_script_path,
            "script_args": device.launch_config.script_args or "",
            "env": launch_env,
        }
        sk_launch = await agent_tunnel_manager.resolve_connected_socket_key(
            resolved_aid, platform_device_id=device_id
        ) or resolved_aid
        logger.info(
            "device_launch: request device_id=%s agent_id=%s script=%s script_args_len=%s env_keys=%s",
            device_id,
            resolved_aid,
            raw_script_path,
            len(device.launch_config.script_args or ""),
            sorted(launch_env.keys()),
        )
        result = await agent_tunnel_manager.send_cmd_and_wait(
            agent_id=sk_launch,
            cmd="DEVICE_LAUNCH",
            payload=payload,
            timeout_sec=30.0,
            retry_times=1,
        )
        ok_launch = bool(result.get("success", False))
        logger.info(
            "device_launch: tunnel_result device_id=%s agent_id=%s success=%s msg=%s raw=%s",
            device_id,
            resolved_aid,
            ok_launch,
            result.get("msg"),
            result,
        )
        if not ok_launch:
            logger.warning(
                "device_launch: failed device_id=%s agent_id=%s detail=%s",
                device_id,
                resolved_aid,
                result.get("msg") or result,
            )
            raise HTTPException(
                status_code=500,
                detail=f"采集端启动失败: {result.get('msg') or '未知错误'}",
            )
        return ApiResponse(ok=True, data={"message": result.get("msg") or "连接成功：已通过 Agent 启动设备"})
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(
            "device_launch: exception device_id=%s agent_id=%s err=%s",
            device_id,
            resolved_aid,
            e,
        )
        raise HTTPException(status_code=500, detail=f"通过隧道调用采集端启动设备失败: {e}")


@router.post("/{device_id}/stop", response_model=ApiResponse)
async def stop_device(
    device_id: int = PathParam(..., description="设备ID"),
    db: AsyncSession = Depends(get_db)
):
    """停止设备（必须通过采集端 Agent 执行停止脚本）"""
    device = await get_device_by_id(db, device_id)
    if not device:
        raise HTTPException(status_code=404, detail="设备不存在")
    
    stop_script_path = None
    stop_script_args = ""
    if getattr(device, "launch_config", None):
        stop_script_path = (getattr(device.launch_config, "stop_script_path", None) or "").strip() or None
        stop_script_args = (getattr(device.launch_config, "stop_script_args", None) or "").strip()
    if not stop_script_path:
        stop_script_path = "/home/sia/workspace/test/stop_all_ros_topics.sh"

    # 严格映射：避免未绑定时回退 local-agent，导致误报“隧道未连接”。
    agent: AgentInfo | None = _resolve_agent_with_autobind(device=device, device_id=device_id)

    resolved_aid = agent.agent_id if agent and agent.agent_id else None
    if not resolved_aid:
        raise HTTPException(
            status_code=400,
            detail="未找到在线采集端 Agent，请先在采集端启动 Agent 并绑定该设备ID",
        )
    if not await agent_tunnel_manager.has_connection(resolved_aid, platform_device_id=device_id):
        raise HTTPException(
            status_code=503,
            detail=(
                f"采集端隧道未连接（agent_id={resolved_aid}）。"
                "请确认边缘 Agent 已启动并成功连接 /api/agent/tunnel，"
                "且设备已通过 /api/devices/connect-agent 绑定到该 Agent。"
            ),
        )

    try:
        stop_env: dict = {}
        try:
            raw_env = safe_json_loads(getattr(device.launch_config, "env_vars_json", None), {}) or {}
            if isinstance(raw_env, dict):
                stop_env = {str(k): str(v) for k, v in raw_env.items() if str(k).strip()}
        except Exception:
            stop_env = {}

        payload = {
            "script_path": stop_script_path,
            "script_args": stop_script_args,
            "env": stop_env,
        }
        sk_stop = await agent_tunnel_manager.resolve_connected_socket_key(
            resolved_aid, platform_device_id=device_id
        ) or resolved_aid
        logger.info(
            "device_stop: request device_id=%s agent_id=%s script=%s script_args_len=%s env_keys=%s",
            device_id,
            resolved_aid,
            stop_script_path,
            len(stop_script_args or ""),
            sorted(stop_env.keys()),
        )
        result = await agent_tunnel_manager.send_cmd_and_wait(
            agent_id=sk_stop,
            cmd="DEVICE_STOP",
            payload=payload,
            timeout_sec=25.0,
            retry_times=1,
        )
        ok_stop = bool(result.get("success", False))
        logger.info(
            "device_stop: tunnel_result device_id=%s agent_id=%s success=%s msg=%s raw=%s",
            device_id,
            resolved_aid,
            ok_stop,
            result.get("msg"),
            result,
        )
        if not ok_stop:
            logger.warning(
                "device_stop: failed device_id=%s agent_id=%s detail=%s",
                device_id,
                resolved_aid,
                result.get("msg") or result,
            )
            raise HTTPException(
                status_code=500,
                detail=f"采集端停止失败: {result.get('msg') or '未知错误'}",
            )

        test_result = DeviceTestResultBase(
            status="fail",
            error_type="STOPPED",
            error_message="设备已停止",
            node_count=0,
            nodes_sample=[],
            topic_count=0,
            topics_sample=[],
        )
        await create_test_result(db, device_id, test_result)
        return ApiResponse(ok=True, data={"message": result.get("msg") or "设备已停止（通过 Agent）"})
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(
            "device_stop: exception device_id=%s agent_id=%s err=%s",
            device_id,
            resolved_aid,
            e,
        )
        raise HTTPException(status_code=500, detail=f"通过隧道调用采集端停止设备失败: {e}")


@router.post("/connect", response_model=ApiResponse)
async def connect_device(payload: DeviceConnectRequest, db: AsyncSession = Depends(get_db)):
    """
    已弃用：请使用 POST /api/devices/connect-agent。
    """
    raise HTTPException(
        status_code=410,
        detail="接口已弃用：请使用 /api/devices/connect-agent 通过在线隧道 Agent 绑定设备",
    )


@router.get("/agents/online", response_model=ApiResponse)
async def list_online_agents():
    """
    在线 Agent 列表（用于“添加设备”下拉选择，避免手工输入 IP/Port）。
    """
    items = []
    connected_ids = set(agent_tunnel_manager.get_connected_agent_ids())
    stale_after = float(getattr(settings, "AGENT_TUNNEL_OFFLINE_AFTER_SEC", 45.0) or 45.0)
    now = time.time()
    registry_agents = {
        a.agent_id: a for a in agent_registry.list_agents() if getattr(a, "agent_id", None)
    }
    # 关键：重启后若 register/heartbeat 尚未补到 registry，也要把 tunnel 已连接的 agent 暴露给前端。
    candidate_ids = set(registry_agents.keys()) | connected_ids

    for aid in sorted(candidate_ids):
        if aid == "local-agent":
            continue
        a = registry_agents.get(aid)
        # 仅返回在线或已连接隧道的 agent
        is_online = bool(getattr(a, "online", False)) or (aid in connected_ids)
        if not is_online:
            continue
        last_ts = agent_tunnel_manager.get_last_seen_ts(aid)
        sec_since_tunnel = None
        tunnel_stale = False
        if last_ts is not None:
            sec_since_tunnel = round(now - float(last_ts), 3)
            tunnel_stale = (now - float(last_ts)) > stale_after
        heartbeat_payload = agent_tunnel_manager.get_last_heartbeat_payload(aid)
        # 没有 registry 条目时，尽量从 heartbeat 中补齐展示字段
        hb_host = heartbeat_payload.get("host") if isinstance(heartbeat_payload, dict) else None
        hb_port = heartbeat_payload.get("port") if isinstance(heartbeat_payload, dict) else None
        if tunnel_stale:
            # stale 后置空：不再向前端下发 CPU/内存/磁盘等“最后心跳统计”，避免误读为当前在线。
            heartbeat_payload = {}
        items.append(
            {
                "agent_id": aid,
                "name": getattr(a, "name", None) or aid,
                "host": getattr(a, "host", None) or hb_host or "",
                "port": getattr(a, "port", None) or hb_port or 0,
                "online": is_online,
                "runtime_status": getattr(a, "runtime_status", None) or "ONLINE_IDLE",
                "camera_list": agent_tunnel_manager.get_camera_ids(aid),
                "tunnel_last_seen_ts": last_ts,
                "seconds_since_tunnel_seen": sec_since_tunnel,
                "tunnel_stale": tunnel_stale,
                "heartbeat": heartbeat_payload,
            }
        )
    return ApiResponse(ok=True, data=items)


@router.post("/connect-agent", response_model=ApiResponse)
async def connect_device_by_agent(
    request: Request,
    payload: DeviceConnectByAgentRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin_async),
):
    """
    通过在线 agent_id 直接绑定设备（不再要求前端输入 IP/Port）。
    """
    agent_id = (payload.agent_id or "").strip()
    if not agent_id:
        raise HTTPException(status_code=400, detail="agent_id 不能为空")

    agent = agent_registry.get_by_id(agent_id)
    if agent is None:
        raise HTTPException(status_code=404, detail="未找到在线 Agent")

    stale_after = float(getattr(settings, "AGENT_TUNNEL_OFFLINE_AFTER_SEC", 45.0) or 45.0)
    last_ts = agent_tunnel_manager.get_last_seen_ts(agent_id)
    if last_ts is None or (time.time() - float(last_ts)) > stale_after:
        raise HTTPException(status_code=409, detail="Agent 通信连接不可用（隧道未连接或已超时）：请先完成安装并确保在线")
    if not await agent_tunnel_manager.has_connection(agent_id):
        raise HTTPException(status_code=409, detail="Agent 通信连接不可用（隧道未连接）：请先完成安装并确保在线")

    camera_list = agent_tunnel_manager.get_camera_ids(agent_id)
    agent_status_str = agent.runtime_status or "ONLINE_IDLE"
    resolved_name = (getattr(payload, "name", None) or "").strip() or str(agent.name or agent.agent_id)
    resolved_vendor = getattr(payload, "vendor", None)
    resolved_model = getattr(payload, "model", None)
    resolved_device_type = ((getattr(payload, "device_type", None) or "") or "ROS2").strip() or "ROS2"
    resolved_hostname = (getattr(payload, "hostname", None) or "").strip() or None
    resolved_collect_script_compress = (getattr(payload, "collect_script_compress", None) or "").strip() or None
    resolved_collect_script_raw = (getattr(payload, "collect_script_raw", None) or "").strip() or None
    default_launch_config_create = DeviceLaunchConfigCreate(
        script_path="/home/sia/workspace/test/start_all_ros_topics.sh",
        script_args="",
        stop_script_path="/home/sia/workspace/test/stop_all_ros_topics.sh",
        stop_script_args="",
        env_vars={},
    )

    # 归属团队绑定：按“发起添加该设备的账号”所在团队来绑定
    from app.core.roles import is_super_admin
    from app.crud import team as team_crud

    binding_team_id: str | None = None
    if is_super_admin(getattr(current_user, "role", None)):
        # 超管：允许设备不绑定团队（列表过滤逻辑会隐藏/不隐藏由 UI 角色决定）
        binding_team_id = getattr(payload, "team_id", None)  # payload 目前不含该字段，兜底保留
    else:
        team_ids = await team_crud.list_team_ids_accessible_by_user(db, str(getattr(current_user, "id", "") or ""))
        if not team_ids:
            raise HTTPException(status_code=403, detail="当前账号未关联团队，无法绑定设备")
        # 若同一用户属于多个团队：取最先的一个作为绑定目标
        binding_team_id = team_ids[0]

    try:
        existing_device = await get_device_by_hardware_uuid(db, agent_id)
        if existing_device:
            # 同一设备只能被“加入一次”：若已存在且归属团队不一致，拒绝覆盖
            if not is_super_admin(getattr(current_user, "role", None)):
                if existing_device.team_id and str(existing_device.team_id) != str(binding_team_id):
                    raise HTTPException(status_code=403, detail="该设备已被其他团队添加，无法再次添加")
                # 兼容历史数据：若旧设备未绑定团队，则在首次由团队添加时自动绑定
                if not existing_device.team_id and binding_team_id:
                    existing_device.team_id = binding_team_id
                    db.add(existing_device)
                    await db.commit()
                    await db.refresh(existing_device)

            launch_config_update = None
            try:
                if getattr(existing_device, "launch_config", None) is None:
                    launch_config_update = DeviceLaunchConfigUpdate(
                        script_path=default_launch_config_create.script_path,
                        script_args=default_launch_config_create.script_args,
                        env_vars=default_launch_config_create.env_vars,
                    )
            except Exception:
                launch_config_update = None

            if payload.launch_config is not None:
                launch_config_update = DeviceLaunchConfigUpdate(
                    script_path=payload.launch_config.script_path,
                    script_args=payload.launch_config.script_args,
                    env_vars=payload.launch_config.env_vars,
                )

            update_kwargs = {
                "device_type": resolved_device_type,
                "hardware_uuid": agent_id,
                "hostname": resolved_hostname or agent.name,
                "agent_ip": agent.host,
                "agent_port": agent.port,
                "agent_status": agent_status_str,
                "camera_list": camera_list,
            }
            if getattr(payload, "name", None) is not None and str(payload.name).strip():
                update_kwargs["name"] = resolved_name
            if getattr(payload, "vendor", None) is not None:
                update_kwargs["vendor"] = resolved_vendor
            if getattr(payload, "model", None) is not None:
                update_kwargs["model"] = resolved_model
            if launch_config_update is not None:
                update_kwargs["launch_config"] = launch_config_update
            if getattr(payload, "ros2_config", None) is not None:
                update_kwargs["ros2_config"] = payload.ros2_config
            if getattr(payload, "collect_script_compress", None) is not None:
                update_kwargs["collect_script_compress"] = resolved_collect_script_compress
            if getattr(payload, "collect_script_raw", None) is not None:
                update_kwargs["collect_script_raw"] = resolved_collect_script_raw

            device = await update_device(
                db,
                existing_device.id,
                DeviceUpdate(**update_kwargs),
            )
        else:
            device_launch_config = payload.launch_config or default_launch_config_create
            device = await create_device(
                db,
                DeviceCreate(
                    name=resolved_name,
                    vendor=resolved_vendor,
                    model=resolved_model,
                    device_type=resolved_device_type,
                    hardware_uuid=agent_id,
                    team_id=binding_team_id,
                    hostname=resolved_hostname or agent.name,
                    agent_ip=agent.host,
                    agent_port=agent.port,
                    agent_status=agent_status_str,
                    camera_list=camera_list,
                    launch_config=device_launch_config,
                    ros2_config=getattr(payload, "ros2_config", None),
                    collect_script_compress=resolved_collect_script_compress,
                    collect_script_raw=resolved_collect_script_raw,
                ),
            )

        if not device:
            raise HTTPException(status_code=500, detail="设备落库失败")
        device = await get_device_by_id(db, device.id)
        if not device:
            raise HTTPException(status_code=500, detail="设备重新查询失败")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"设备落库失败: {e}")

    # 刷新 registry 中的设备映射
    try:
        existing_agent = agent_registry.get_by_id(agent_id)
        if existing_agent:
            devices_union = sorted(set(existing_agent.devices + [device.id]))
            agent_registry.register_agent(
                agent_id=agent_id,
                name=existing_agent.name or agent_id,
                host=existing_agent.host,
                port=existing_agent.port,
                devices=devices_union,
            )
    except Exception:
        pass

    # 不再在“添加设备”时自动启动/测试设备，仅完成绑定
    runtime_status = "ONLINE_IDLE"
    db_test_result = None

    try:
        updated_agent = agent_registry.get_by_id(agent_id)
        if updated_agent:
            updated_agent.runtime_status = runtime_status
            updated_agent.online = True
    except Exception:
        pass

    tunnel_ok = False
    try:
        tunnel_ok = bool(await agent_tunnel_manager.has_connection(agent_id))
    except Exception:
        tunnel_ok = False

    log_audit_safe(
        user=current_user,
        action_type=AA.CONNECT_DEVICE,
        resource_type=AR.DEVICE,
        resource_id=str(getattr(device, "id", "") or ""),
        resource_name=str(getattr(device, "name", "") or ""),
        team_id=str(getattr(device, "team_id", "") or "") or None,
        result="SUCCESS",
        detail_json={"agent_id": agent_id},
        request=request,
    )
    return ApiResponse(
        ok=True,
        data={
            "id": device.id,
            "name": device.name,
            "vendor": device.vendor,
            "model": device.model,
            "device_type": device.device_type,
            "created_at": device.created_at.isoformat() if hasattr(device.created_at, "isoformat") else str(device.created_at),
            "updated_at": device.updated_at.isoformat() if hasattr(device.updated_at, "isoformat") else str(device.updated_at),
            "hardware_uuid": device.hardware_uuid,
            "hostname": device.hostname,
            "agent_ip": device.agent_ip,
            "agent_port": device.agent_port,
            "agent_status": device.agent_status,
            "camera_list": safe_json_loads(device.camera_list_json, []),
            "collect_script_compress": getattr(device, "collect_script_compress", None),
            "collect_script_raw": getattr(device, "collect_script_raw", None),
            "runtime_status": runtime_status,
            "agent_tunnel_connected": tunnel_ok,
            "last_test_result": (
                {
                    "status": db_test_result.status,
                    "node_count": db_test_result.node_count,
                    "nodes_sample": safe_json_loads(db_test_result.nodes_sample_json, []),
                    "topic_count": db_test_result.topic_count,
                    "topics_sample": safe_json_loads(db_test_result.topics_sample_json, []),
                    "error_type": db_test_result.error_type,
                    "error_message": db_test_result.error_message,
                    "tested_at": db_test_result.tested_at,
                }
                if db_test_result
                else None
            ),
        },
    )
