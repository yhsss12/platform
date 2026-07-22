from __future__ import annotations

import hashlib
import time
import uuid
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlencode, urlparse

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from app.core.config import settings
from app.core.deps import get_current_user
from app.models.user import User
from app.schemas.common import ApiResponse
from app.services.agent_deb_builder import build_deb_from_agent_tarball
from app.services.agent_installer_linux_script import apply_linux_installer_replacements, resolve_linux_installer_script
from app.services.agent_package_manager import AgentPackageManager, default_package_manager
from app.services.public_endpoint_resolver import get_public_endpoint_resolver
from app.services.task_queue import redis_conn

router = APIRouter()


class ZeroInstallStartResponse(BaseModel):
    token: str
    script_url: str
    status_url: str
    command: str


@router.post("/installer/start")
async def zero_install_start(request: Request, current_user: User = Depends(get_current_user)) -> ApiResponse:
    token = str(uuid.uuid4())
    now = int(time.time())
    redis_conn.hset(
        f"agent_zero_install:{token}",
        mapping={"status": "pending", "progress": "0", "stage": "init", "logs": "", "ts": str(now)},
    )
    redis_conn.expire(f"agent_zero_install:{token}", 60 * 30)
    resolver = get_public_endpoint_resolver(default_base_url=str(getattr(settings, "PUBLIC_BASE_URL", "") or ""))
    base = resolver.resolve(request).base_url.rstrip("/")
    parsed = urlparse(base)
    ip = parsed.hostname or "172.18.0.114"
    if parsed.port:
        port = parsed.port
    else:
        port = 443 if (parsed.scheme or "").lower() == "https" else 80
    params: dict[str, str] = {"token": token, "ip": ip, "port": str(port)}
    if getattr(settings, "AGENT_INSTALL_USE_CONDA", False):
        params["use_conda"] = "1"
        cs = str(getattr(settings, "AGENT_INSTALL_CONDA_SH", "") or "").strip()
        ce = str(getattr(settings, "AGENT_INSTALL_CONDA_ENV", "") or "").strip()
        if cs:
            params["conda_sh"] = cs
        if ce:
            params["conda_env"] = ce
    rs = str(getattr(settings, "AGENT_INSTALL_ROS_SETUP", "") or "").strip()
    if rs and rs != "/opt/ros/humble/setup.bash":
        params["ros_setup"] = rs
    rws = str(getattr(settings, "AGENT_INSTALL_ROS_WS_SETUP", "") or "").strip()
    if rws:
        params["ros_ws_setup"] = rws
    su = str(getattr(settings, "AGENT_INSTALL_SERVICE_USER", "") or "").strip()
    if su and su != "root":
        params["service_user"] = su
    script_url = f"{base}/api/agent/installer/linux.sh?{urlencode(params)}"
    status_url = f"/api/agent/installer/status/{token}"
    command = f"curl -fsSL '{script_url}' | sudo bash"
    return ApiResponse(
        ok=True,
        data=ZeroInstallStartResponse(
            token=token,
            script_url=script_url,
            status_url=status_url,
            command=command,
        ).model_dump(),
    )


@router.get("/installer/status/{token}")
async def zero_install_status(token: str):
    key = f"agent_zero_install:{token}"
    if not redis_conn.exists(key):
        raise HTTPException(status_code=404, detail="not found")
    data = redis_conn.hgetall(key)
    out: Dict[str, Any] = {}
    for k, v in data.items():
        kk = k.decode("utf-8") if isinstance(k, (bytes, bytearray)) else str(k)
        vv = v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else str(v)
        out[kk] = vv
    logs = out.get("logs") or ""
    lines = []
    if logs:
        for line in logs.split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                ts, lvl, msg = line.split("|", 2)
                lines.append({"ts": ts, "level": lvl, "message": msg})
            except Exception:
                lines.append({"ts": "", "level": "info", "message": line})
    return {
        "ok": True,
        "data": {
            "status": out.get("status", "pending"),
            "progress": int(out.get("progress") or 0),
            "stage": out.get("stage", ""),
            "logs": lines,
        },
    }


@router.api_route("/installer/progress", methods=["GET", "POST"])
async def zero_install_progress(
    token: str = Query(...),
    stage: str = Query(""),
    progress: int = Query(0),
    level: str = Query("info"),
    message: str = Query(""),
    status: str = Query("running"),
):
    key = f"agent_zero_install:{token}"
    if not redis_conn.exists(key):
        raise HTTPException(status_code=404, detail="not found")
    now = int(time.time())
    logs_raw = redis_conn.hget(key, "logs") or b""
    prev = logs_raw.decode("utf-8", "replace") if isinstance(logs_raw, (bytes, bytearray)) else str(logs_raw)
    line = f"{now}|{level}|{message}".strip()
    new_logs = f"{prev}\n{line}" if prev else line
    if len(new_logs) > 40000:
        new_logs = new_logs[-30000:]
    redis_conn.hset(
        key,
        mapping={
            "status": status,
            "progress": str(max(0, min(100, progress))),
            "stage": stage,
            "logs": new_logs,
            "ts": str(now),
        },
    )
    return {"ok": True}


@router.get("/installer/config")
async def zero_install_config(request: Request, token: str):
    key = f"agent_zero_install:{token}"
    if not redis_conn.exists(key):
        raise HTTPException(status_code=404, detail="not found")
    resolver = get_public_endpoint_resolver(default_base_url=str(getattr(settings, "PUBLIC_BASE_URL", "") or ""))
    base = resolver.resolve(request).base_url.rstrip("/")
    return {
        "ok": True,
        "data": {
            "server_base": base,
            "tunnel_token": (getattr(settings, "AGENT_TUNNEL_TOKEN", None) or None),
            "agent_name_suggest": "eai-agent",
        },
    }


@router.get("/installer/resolve")
async def zero_install_resolve(
    token: str = Query(...),
    os_name: str = Query("linux"),
    arch: str = Query("x86_64"),
    version: str | None = None,
):
    key = f"agent_zero_install:{token}"
    if not redis_conn.exists(key):
        raise HTTPException(status_code=404, detail="not found")
    mgr: AgentPackageManager = default_package_manager()
    try:
        ref = mgr.resolve(os_name=os_name, arch=arch, version=version)
    except ValueError as exc:
        return {"ok": False, "error": str(exc), "data": None}
    backend_dir = Path(__file__).resolve().parents[2]
    gen_dir = backend_dir / "agent_packages" / "generated"
    out_name = f"eai-agent_{ref.version}_{arch}.deb"
    out_path = gen_dir / out_name
    if not out_path.exists():
        build_deb_from_agent_tarball(
            tar_gz_path=ref.file_path,
            output_deb_path=str(out_path),
            version=ref.version,
            arch=arch,
        )
    sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
    url = f"/api/agent/installer/pkg/{out_name}"
    return {"ok": True, "data": {"url": url, "sha256": sha, "version": ref.version, "format": "deb"}}


@router.get("/installer/pkg/{filename}")
async def zero_install_pkg(filename: str):
    base = Path(__file__).resolve().parents[2] / "agent_packages"
    path = (base / filename).resolve()
    if not str(path).startswith(str(base.resolve())):
        raise HTTPException(status_code=403, detail="forbidden")
    if not path.exists():
        generated = (base / "generated" / filename).resolve()
        if str(generated).startswith(str(base.resolve())) and generated.exists():
            path = generated
        else:
            raise HTTPException(status_code=404, detail="not found")
    return Response(content=path.read_bytes(), media_type="application/octet-stream")


@router.get("/installer/linux.sh")
async def zero_install_linux_script(
    token: str,
    ip: str = "172.18.0.114",
    port: int = 8000,
    os_name: str = "linux",
    arch: str = "x86_64",
    version: str | None = None,
    use_conda: str | None = None,
    conda_sh: str | None = None,
    conda_env: str | None = None,
    ros_setup: str | None = None,
    ros_ws_setup: str | None = None,
    service_user: str | None = None,
):
    try:
        key = f"agent_zero_install:{token}"
        if not redis_conn.exists(key):
            return PlainTextResponse("not found", status_code=404)

        try:
            repl = resolve_linux_installer_script(
                settings=settings,
                query_use_conda=use_conda,
                query_conda_sh=conda_sh,
                query_conda_env=conda_env,
                query_ros_setup=ros_setup,
                query_ros_ws_setup=ros_ws_setup,
                query_service_user=service_user,
            )
        except ValueError as exc:
            return PlainTextResponse(str(exc), status_code=400)

        mgr: AgentPackageManager = default_package_manager()
        ref = mgr.resolve(os_name=os_name, arch=arch, version=version)

        template_path = Path(__file__).resolve().parents[2] / "static" / "scripts" / "installer_template.sh"
        if not template_path.exists():
            return PlainTextResponse("installer template not found", status_code=404)

        script = template_path.read_text(encoding="utf-8")
        script = apply_linux_installer_replacements(script, repl)
        script = script.replace("{{TOKEN}}", token)
        script = script.replace("{{SERVER_IP}}", ip)
        script = script.replace("{{SERVER_PORT}}", str(port))
        script = script.replace("{{AGENT_TUNNEL_TOKEN}}", getattr(settings, "AGENT_TUNNEL_TOKEN", "") or "")
        script = script.replace("{{AGENT_BUNDLE_NAME}}", Path(ref.file_path).name)
        return Response(content=script, media_type="text/x-sh")
    except Exception as exc:
        return PlainTextResponse(f"internal error: {exc}", status_code=500)
