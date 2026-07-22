"""
文件系统相关 API 路由
用于浏览服务器目录和列出 HDF5 文件；平台内置资源浏览器 list / inspect。
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional, List, Any
from app.schemas.common import ApiResponse
from app.core.deps import get_current_user, require_admin
from app.models.user import User
import os
from pathlib import Path
from datetime import datetime, timezone
from app.crud.device import get_device_by_id
from app.db.session import AsyncSessionLocal
from app.services.agent_registry import agent_registry
from app.services.agent_tunnel_manager import agent_tunnel_manager

router = APIRouter()


def _format_mtime_utc_iso(st_mtime: float) -> str:
    return datetime.fromtimestamp(st_mtime, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


async def _resolve_agent_id_for_tunnel(
    device_id: Optional[int],
    agent_id: Optional[str],
) -> Optional[str]:
    aid = (agent_id or "").strip() or None
    if aid:
        return aid
    if device_id is not None:
        # 严格映射优先，避免回退 local-agent 误报“采集端未连接”
        info = agent_registry.get_by_device_id_strict(int(device_id))
        if info and info.agent_id:
            return info.agent_id
        # 自动回填：device.hardware_uuid 约定为 agent_id
        async with AsyncSessionLocal() as db:
            dev = await get_device_by_id(db, int(device_id))
        if dev is not None:
            hw = str(getattr(dev, "hardware_uuid", "") or "").strip()
            if hw:
                agent = agent_registry.get_by_id(hw)
                if agent and agent.agent_id:
                    try:
                        agent_registry.bind_device_to_agent(device_id=int(device_id), agent_id=hw)
                    except Exception:
                        pass
                    return agent.agent_id
    return None

# 安全根目录
ROOT_DIR = os.path.expanduser("~")

# 平台资源浏览器白名单：只允许访问以下根目录及其子路径
FS_LIST_WHITELIST = [
    "/home/ubuntu",
    os.path.expanduser("~"),
]
# 去重并规范化
FS_LIST_WHITELIST = list({os.path.realpath(p) for p in FS_LIST_WHITELIST if p})


def _path_under_whitelist(resolved: str) -> bool:
    resolved = os.path.realpath(resolved)
    for root in FS_LIST_WHITELIST:
        if resolved == root or resolved.startswith(root + os.sep):
            return True
    return False


def validate_path_whitelist(path: str) -> str:
    """校验 path 在白名单根目录下，返回规范化绝对路径；否则抛 HTTPException。"""
    if not path or not path.strip():
        raise HTTPException(status_code=400, detail="path 不能为空")
    path = path.strip()
    try:
        resolved = os.path.realpath(path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"无效路径: {str(e)}")
    if not _path_under_whitelist(resolved):
        raise HTTPException(status_code=403, detail="路径不在允许访问的根目录下")
    return resolved


def validate_path(path: str) -> str:
    """
    验证路径是否在 ROOT_DIR 下，返回规范化路径
    如果路径越界，抛出 HTTPException
    """
    if not path:
        return ROOT_DIR
    
    # 规范化路径（解析 .. 和符号链接）
    try:
        real_path = os.path.realpath(path)
        real_root = os.path.realpath(ROOT_DIR)
        
        # 确保路径在 ROOT_DIR 下
        if not real_path.startswith(real_root):
            raise HTTPException(
                status_code=403,
                detail=f"路径必须在 {ROOT_DIR} 目录下"
            )
        
        return real_path
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"无效的路径: {str(e)}")


@router.get("/list-dirs")
async def list_dirs(
    base: Optional[str] = Query(None, description="基础路径，默认为 ROOT_DIR"),
    _user: User = Depends(require_admin),
):
    """
    列出指定目录下的所有子目录
    
    Query Parameters:
        base: 可选，基础路径。如果不提供，默认为 ROOT_DIR
    
    Returns:
        {
            "ok": true,
            "data": {
                "base": string,  # 当前路径
                "dirs": string[]  # 目录名列表（仅名称，不包含路径）
            }
        }
    """
    try:
        # 如果没有提供 base，使用 ROOT_DIR
        if base is None or base == "":
            target_path = ROOT_DIR
        else:
            target_path = base
        
        # 验证路径
        # 允许访问系统中的任何路径，只要它存在且有权限
        # 但我们仍然可以限制在某个根目录下，如果需要的话
        # 目前放宽限制，允许用户浏览
        
        # 检查路径是否存在且为目录
        if not os.path.exists(target_path):
            return {
                "ok": False,
                "error": f"路径不存在: {target_path}"
            }
        
        if not os.path.isdir(target_path):
            return {
                "ok": False,
                "error": f"路径不是目录: {target_path}"
            }
        
        # 列出所有子目录
        dirs = []
        try:
            # 使用 os.scandir 替代 os.listdir，性能更好且直接提供 entry 对象
            with os.scandir(target_path) as entries:
                for entry in entries:
                    if entry.is_dir():
                        # 获取文件名，如果包含无效字节，Python 会用 surrogateescape 处理
                        name = entry.name
                        # 为了 JSON 安全，我们需要去除 surrogate 字符
                        # encode('utf-8', 'replace') 会将无效字节替换为 �
                        # decode('utf-8') 再转回字符串
                        safe_name = name.encode('utf-8', 'replace').decode('utf-8')
                        dirs.append(safe_name)
        except PermissionError as e:
            return {
                "ok": False,
                "error": f"没有权限访问该目录: {str(e)}"
            }
        except Exception as e:
            return {
                "ok": False,
                "error": f"读取目录失败: {str(e)}"
            }
        
        # 排序
        dirs.sort()
        
        return {
            "ok": True,
            "data": {
                "base": target_path,
                "dirs": dirs
            }
        }
    except Exception as e:
        return {
            "ok": False,
            "error": f"服务器错误: {str(e)}"
        }


@router.get("/list-hdf5", response_model=ApiResponse)
async def list_hdf5(
    dir: str = Query(..., description="目录路径"),
    _user: User = Depends(require_admin),
):
    """
    列出指定目录下的所有 .hdf5 文件（仅一级，不递归）
    
    Query Parameters:
        dir: 必需，目录路径
    
    Returns:
        {
            "ok": true,
            "data": {
                "dir": string,    # 目录路径
                "files": string[] # 文件名列表（仅名称，不包含路径）
            }
        }
    """
    try:
        # 验证路径
        validated_dir = validate_path(dir)
        
        # 检查路径是否存在且为目录
        if not os.path.exists(validated_dir):
            raise HTTPException(status_code=404, detail=f"路径不存在: {validated_dir}")
        
        if not os.path.isdir(validated_dir):
            raise HTTPException(status_code=400, detail=f"路径不是目录: {validated_dir}")
        
        # 列出所有 .hdf5 文件
        hdf5_files = []
        try:
            with os.scandir(validated_dir) as entries:
                for entry in entries:
                    if entry.is_file():
                        name = entry.name
                        # 为了 JSON 安全，我们需要去除 surrogate 字符
                        safe_name = name.encode('utf-8', 'replace').decode('utf-8')
                        
                        # 检查扩展名
                        if safe_name.lower().endswith(('.hdf5', '.h5')):
                            hdf5_files.append(safe_name)
        except PermissionError:
            raise HTTPException(status_code=403, detail="没有权限访问该目录")
        
        # 排序
        hdf5_files.sort()
        
        return ApiResponse(
            ok=True,
            data={
                "dir": validated_dir,
                "files": hdf5_files
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"列出 HDF5 文件失败: {str(e)}")


@router.get("/list")
async def fs_list(
    path: str = Query("", description="要列出的目录绝对路径"),
    _user: User = Depends(require_admin),
):
    """
    平台内置资源浏览器：列出目录内容。
    返回 { "path": "...", "items": [ { "name", "type": "dir"|"file", "size"?, "mtime" } ] }
    只允许访问白名单根目录下的路径。
    """
    try:
        if not path or not path.strip():
            # 默认第一个白名单根
            path = FS_LIST_WHITELIST[0] if FS_LIST_WHITELIST else ROOT_DIR
        resolved = validate_path_whitelist(path)
        if not os.path.isdir(resolved):
            return ApiResponse(ok=False, error="路径不是目录")
        items: List[dict] = []
        with os.scandir(resolved) as entries:
            for entry in entries:
                try:
                    name = entry.name.encode("utf-8", "replace").decode("utf-8")
                    stat = entry.stat()
                    mtime = _format_mtime_utc_iso(stat.st_mtime) if stat else None
                    if entry.is_dir():
                        items.append({"name": name, "type": "dir", "mtime": mtime})
                    else:
                        items.append({
                            "name": name,
                            "type": "file",
                            "size": stat.st_size,
                            "mtime": mtime,
                        })
                except (OSError, PermissionError):
                    continue
        items.sort(key=lambda x: (x["type"] != "dir", x["name"].lower()))
        return ApiResponse(ok=True, data={"path": resolved, "items": items})
    except HTTPException:
        raise
    except Exception as e:
        return ApiResponse(ok=False, error=str(e)[:300])


@router.get("/agent-list-dirs")
async def agent_list_dirs(
    base: Optional[str] = Query(None, description="采集端基础路径，可选，由 Agent 自行解析"),
    device_id: Optional[int] = Query(None, description="设备 ID，用于从注册表解析 Agent"),
    agent_id: Optional[str] = Query(None, description="Agent ID，优先级高于 device_id"),
    _user: User = Depends(get_current_user),
):
    """
    通过采集端 Agent 列出子目录。

    - 平台解析采集端 HTTP 根地址（注册表或 devices 表等，与数据同步一致）
    - 转发到 Agent 的 `/api/agent/fs/list-dirs` 接口
    - 返回结构与 `/api/fs/list-dirs` 相同：
      { "ok": true, "data": { "base": "...", "dirs": [...] } }
    """
    resolved_aid = await _resolve_agent_id_for_tunnel(device_id, agent_id)
    if not resolved_aid:
        return {"ok": False, "error": "未找到可用的采集端 Agent，或 Agent 未上报地址"}

    if not await agent_tunnel_manager.has_connection(resolved_aid):
        return {"ok": False, "error": "采集端隧道未连接"}

    try:
        result = await agent_tunnel_manager.send_cmd_and_wait(
            agent_id=resolved_aid,
            cmd="FS_LIST_DIRS",
            payload={"base": base},
            timeout_sec=15.0,
            retry_times=1,
        )
        if not bool(result.get("success", False)):
            return {"ok": False, "error": result.get("msg") or "隧道命令失败"}
        data = result.get("data")
        if isinstance(data, dict):
            return data
        return {"ok": False, "error": "采集端返回格式不正确"}
    except Exception as e:
        return {"ok": False, "error": f"服务器错误: {str(e)}"}


@router.get("/agent-list", response_model=ApiResponse)
async def agent_fs_list(
    path: str = Query("", description="采集端要列出的目录绝对路径或相对路径（相对采集端根目录）"),
    device_id: Optional[int] = Query(None, description="设备 ID，用于从注册表解析 Agent"),
    agent_id: Optional[str] = Query(None, description="Agent ID，优先级高于 device_id"),
    _user: User = Depends(get_current_user),
):
    """
    通过采集端 Agent 列出目录内容（用于“设备脚本路径选择”等需要浏览采集端文件系统的场景）。
    返回结构与 /api/fs/list 对齐：{ path, items: [{name,type,size?,mtime?}] }。
    """
    resolved_aid = await _resolve_agent_id_for_tunnel(device_id, agent_id)
    if not resolved_aid:
        return ApiResponse(ok=False, error="未找到可用的采集端 Agent，或 Agent 未绑定/未上线")
    if not await agent_tunnel_manager.has_connection(resolved_aid):
        return ApiResponse(ok=False, error="采集端隧道未连接")

    try:
        result = await agent_tunnel_manager.send_cmd_and_wait(
            agent_id=resolved_aid,
            cmd="FS_LIST",
            payload={"path": path},
            timeout_sec=15.0,
            retry_times=1,
        )
        if not bool(result.get("success", False)):
            return ApiResponse(ok=False, error=result.get("msg") or "隧道命令失败")
        data = result.get("data")
        if isinstance(data, dict) and data.get("ok") and isinstance(data.get("data"), dict):
            return ApiResponse(ok=True, data=data.get("data"))
        err = None
        if isinstance(data, dict):
            err = data.get("error")
        return ApiResponse(ok=False, error=str(err or "采集端返回格式不正确")[:300])
    except Exception as e:
        return ApiResponse(ok=False, error=f"服务器错误: {str(e)[:200]}")


@router.get("/inspect")
async def fs_inspect(
    path: str = Query(..., description="要检测的目录绝对路径"),
    _user: User = Depends(require_admin),
):
    """
    检测目录是否为有效的 LeRobot 数据集。
    规则：存在 meta/、data/、videos/ 中至少 2 个，或存在 meta.json / dataset_info.json / meta/info.json 任一。
    返回 { "isLeRobot": bool, "reason": "...", "metaHint": { "hasMeta", "hasData", "hasVideos" } }
    """
    try:
        resolved = validate_path_whitelist(path)
        if not os.path.isdir(resolved):
            return ApiResponse(
                ok=True,
                data={
                    "isLeRobot": False,
                    "reason": "路径不是目录",
                    "metaHint": {"hasMeta": False, "hasData": False, "hasVideos": False},
                },
            )
        p = Path(resolved)
        has_meta_dir = (p / "meta").is_dir()
        has_data_dir = (p / "data").is_dir()
        has_videos_dir = (p / "videos").is_dir()
        count_dirs = sum([has_meta_dir, has_data_dir, has_videos_dir])
        has_meta_json = (p / "meta.json").is_file()
        has_dataset_info = (p / "dataset_info.json").is_file()
        has_meta_info = (p / "meta" / "info.json").is_file() if has_meta_dir else False
        if count_dirs >= 2:
            reason = "found " + ", ".join(
                [x for x in ["meta/", "data/", "videos/"] if (p / x.replace("/", "")).is_dir()]
            )
            return ApiResponse(
                ok=True,
                data={
                    "isLeRobot": True,
                    "reason": reason,
                    "metaHint": {"hasMeta": has_meta_dir, "hasData": has_data_dir, "hasVideos": has_videos_dir},
                },
            )
        if has_meta_json or has_dataset_info or has_meta_info:
            reason = "found meta file (meta.json / dataset_info.json / meta/info.json)"
            return ApiResponse(
                ok=True,
                data={
                    "isLeRobot": True,
                    "reason": reason,
                    "metaHint": {"hasMeta": has_meta_dir, "hasData": has_data_dir, "hasVideos": has_videos_dir},
                },
            )
        reason = "current directory is not a valid LeRobot dataset (need 2+ of meta/, data/, videos/ or a meta file)"
        return ApiResponse(
            ok=True,
            data={
                "isLeRobot": False,
                "reason": reason,
                "metaHint": {"hasMeta": has_meta_dir, "hasData": has_data_dir, "hasVideos": has_videos_dir},
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        return ApiResponse(ok=False, error=str(e)[:300])
