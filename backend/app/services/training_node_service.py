"""训练节点注册与健康探测（L20 远程 SSH + 本地 L20 节点）。"""

from __future__ import annotations

import logging
import os
import re
import shlex
import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any, Literal, Optional

from app.core.env_loader import ensure_dotenv_loaded

logger = logging.getLogger(__name__)

TrainingNodeStatus = Literal["available", "busy", "unreachable", "misconfigured", "placeholder"]

_NODE_ID_IP_PATTERN = re.compile(r"^l20-(\d+-\d+-\d+-\d+)$", re.IGNORECASE)

_GPU_MEMORY_PATTERN = re.compile(
    r"^([^,]+),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)",
    re.MULTILINE,
)

_NODE_PROBE_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_NODE_PROBE_CACHE_TTL_SEC = 30.0
_NODE_PROBE_LOCK = threading.Lock()

BUSY_GPU_MEMORY_RATIO = 0.85
BUSY_GPU_MEMORY_USED_MB = 4096


@dataclass(frozen=True)
class TrainingNodeConfig:
    node_id: str
    label: str
    device_label: str
    execution_mode: Literal["local", "remote_ssh"]
    description: str = ""
    host: str = ""
    ssh_user: str = ""
    ssh_password: str = ""
    ssh_key_path: str = ""
    ssh_port: int = 22
    workdir: str = ""
    data_root: str = ""
    conda_bin: str = ""
    conda_env: str = ""
    python_bin: str = ""
    gpu_model: str = "NVIDIA L20"
    gpu_memory_gb: float = 46.0

    @property
    def ssh_target(self) -> str:
        return f"{self.ssh_user}@{self.host}" if self.ssh_user and self.host else ""


def _env(name: str, default: str = "") -> str:
    ensure_dotenv_loaded()
    return (os.environ.get(name) or default).strip()


def format_training_node_display_name(gpu_short_name: str, host_or_ip: str) -> str:
    """统一训练节点展示名：L20 · 172.18.0.73"""
    gpu = (gpu_short_name or "L20").strip() or "L20"
    ip = (host_or_ip or "").strip()
    if ip:
        return f"{gpu} · {ip}"
    return gpu


def _resolve_local_host_ip() -> str:
    override = _env("TRAIN_NODE_LOCAL_HOST")
    if override:
        return override
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        pass
    try:
        return socket.gethostbyname(socket.gethostname())
    except OSError:
        return "127.0.0.1"


def _short_gpu_name(gpu_model: str) -> str:
    text = (gpu_model or "").strip()
    if not text:
        return "L20"
    upper = text.upper()
    for token in ("L20", "H20", "A100", "H100", "V100", "RTX"):
        if token in upper:
            return token
    parts = text.split()
    return parts[-1] if parts else "L20"


def _probe_local_gpu_info() -> dict[str, Any]:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        smi_text = (proc.stdout or proc.stderr or "").strip()
        if proc.returncode == 0 and smi_text:
            return _parse_gpu_info(smi_text, fallback_model="NVIDIA L20")
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("local nvidia-smi probe failed: %s", exc)
    return {"name": "NVIDIA L20"}


def display_name_from_node_id(node_id: str) -> Optional[str]:
    match = _NODE_ID_IP_PATTERN.match((node_id or "").strip())
    if not match:
        return None
    ip = match.group(1).replace("-", ".")
    return format_training_node_display_name("L20", ip)


def resolve_training_node_display_name(
    *,
    training_node_id: Optional[str] = None,
    device_label: Optional[str] = None,
    execution_mode: Optional[str] = None,
) -> str:
    cfg = resolve_training_node(training_node_id=training_node_id, device_label=device_label)
    if cfg is not None:
        return cfg.device_label

    parsed = display_name_from_node_id(training_node_id or "")
    if parsed:
        return parsed

    label = (device_label or "").strip()
    if label and "·" in label:
        return label

    mode = (execution_mode or "").strip().lower()
    upper = label.upper()
    if upper == "H20" or (mode == "local" and upper in {"", "L20"}):
        return _build_local_l20_node_config().device_label
    if upper == "L20":
        return _build_l20_node_config().device_label
    if label:
        return label
    return _build_l20_node_config().device_label


def enrich_training_node_display_fields(
    data: dict[str, Any],
    *,
    train_config: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    cfg_source = train_config or {}
    node_id = str(data.get("trainingNodeId") or cfg_source.get("trainingNodeId") or "").strip()
    execution_mode = str(data.get("executionMode") or cfg_source.get("executionMode") or "").strip()
    display = resolve_training_node_display_name(
        training_node_id=node_id or None,
        device_label=str(data.get("deviceLabel") or cfg_source.get("deviceLabel") or ""),
        execution_mode=execution_mode or None,
    )
    enriched = dict(data)
    enriched["trainingNodeDisplayName"] = display
    enriched["deviceLabel"] = display
    if node_id:
        enriched["trainingNodeId"] = node_id
    return enriched


def _missing_workdir_message(workdir: str) -> str:
    return f"远端平台工作目录不存在，请先同步项目代码到 {workdir}"


def _build_l20_node_config() -> TrainingNodeConfig:
    host = _env("TRAIN_NODE_L20_HOST", "172.18.0.73")
    user = _env("TRAIN_NODE_L20_USER", "zyf")
    display = format_training_node_display_name("L20", host)
    return TrainingNodeConfig(
        node_id="l20-172-18-0-73",
        label=display,
        device_label=display,
        execution_mode="remote_ssh",
        description=f"NVIDIA L20 远程 GPU 训练节点（{host}）",
        host=host,
        ssh_user=user,
        ssh_password=_env("TRAIN_NODE_L20_PASSWORD"),
        ssh_key_path=_env("TRAIN_NODE_L20_SSH_KEY"),
        ssh_port=int(_env("TRAIN_NODE_L20_PORT", "22") or "22"),
        workdir=_env("TRAIN_NODE_L20_WORKDIR", f"/home/{user}/eai-idev2.1"),
        data_root=_env("TRAIN_NODE_L20_DATA_ROOT"),
        conda_bin=_env("TRAIN_NODE_L20_CONDA", f"/home/{user}/anaconda3/bin/conda"),
        conda_env=_env("TRAIN_NODE_L20_CONDA_ENV"),
        python_bin=_env("TRAIN_NODE_L20_PYTHON_BIN"),
        gpu_model="NVIDIA L20",
        gpu_memory_gb=46.0,
    )


def _build_local_l20_node_config() -> TrainingNodeConfig:
    host = _resolve_local_host_ip()
    gpu_info = _probe_local_gpu_info()
    gpu_model = str(gpu_info.get("name") or "NVIDIA L20")
    short = _short_gpu_name(gpu_model)
    display = format_training_node_display_name(short, host)
    return TrainingNodeConfig(
        node_id="h20-local-placeholder",
        label=display,
        device_label=display,
        execution_mode="local",
        description=f"本地 {short} GPU 训练节点（{host}）",
        host=host,
        gpu_model=gpu_model,
        gpu_memory_gb=46.0,
    )


def _build_h20_placeholder_config() -> TrainingNodeConfig:
    """保留历史 node_id；实际为本机 L20。"""
    return _build_local_l20_node_config()


def get_training_node_registry() -> dict[str, TrainingNodeConfig]:
    return {
        "l20-172-18-0-73": _build_l20_node_config(),
        "l20": _build_l20_node_config(),
        "h20-local-placeholder": _build_h20_placeholder_config(),
        "h20": _build_h20_placeholder_config(),
    }


def resolve_training_node(
    *,
    training_node_id: Optional[str] = None,
    device_label: Optional[str] = None,
    device: Optional[str] = None,
) -> Optional[TrainingNodeConfig]:
    registry = get_training_node_registry()
    node_id = (training_node_id or "").strip().lower()
    if node_id:
        if node_id in registry:
            return registry[node_id]
        for key, cfg in registry.items():
            if cfg.node_id == node_id:
                return cfg

    label = (device_label or "").strip().upper()
    if label == "L20":
        return registry["l20-172-18-0-73"]
    if label == "H20":
        return registry["h20-local-placeholder"]
    if "·" in (device_label or ""):
        for cfg in registry.values():
            if cfg.device_label == device_label:
                return cfg

    dev = (device or "").strip().lower()
    if dev == "l20":
        return registry["l20-172-18-0-73"]
    if dev == "h20":
        return registry["h20-local-placeholder"]
    return None


def list_training_nodes(*, refresh: bool = False) -> list[dict[str, Any]]:
    seen: set[str] = set()
    nodes: list[dict[str, Any]] = []
    registry = get_training_node_registry()
    for cfg in registry.values():
        if cfg.node_id in seen:
            continue
        seen.add(cfg.node_id)
        nodes.append(probe_training_node(cfg.node_id, refresh=refresh))
    nodes.sort(key=lambda item: (item.get("host") or "", item.get("nodeId") or ""))
    return nodes


def probe_training_node(node_id: str, *, refresh: bool = False) -> dict[str, Any]:
    registry = get_training_node_registry()
    cfg = registry.get(node_id) or resolve_training_node(training_node_id=node_id)
    if cfg is None:
        return {
            "nodeId": node_id,
            "label": node_id,
            "status": "misconfigured",
            "statusLabel": "环境未配置",
            "message": f"未知训练节点: {node_id}",
            "selectable": False,
        }

    cache_key = cfg.node_id
    if not refresh:
        with _NODE_PROBE_LOCK:
            cached = _NODE_PROBE_CACHE.get(cache_key)
            if cached and (time.time() - cached[0]) < _NODE_PROBE_CACHE_TTL_SEC:
                return dict(cached[1])

    payload = _probe_node_config(cfg)
    with _NODE_PROBE_LOCK:
        _NODE_PROBE_CACHE[cache_key] = (time.time(), payload)
    return dict(payload)


def invalidate_training_node_probe_cache(node_id: Optional[str] = None) -> None:
    with _NODE_PROBE_LOCK:
        if node_id:
            _NODE_PROBE_CACHE.pop(node_id, None)
        else:
            _NODE_PROBE_CACHE.clear()


def _probe_node_config(cfg: TrainingNodeConfig) -> dict[str, Any]:
    display_name = cfg.device_label or cfg.label
    base = {
        "nodeId": cfg.node_id,
        "label": cfg.label,
        "deviceLabel": cfg.device_label,
        "trainingNodeDisplayName": display_name,
        "executionMode": cfg.execution_mode,
        "description": cfg.description,
        "gpuModel": cfg.gpu_model,
        "gpuMemoryGb": cfg.gpu_memory_gb,
        "host": cfg.host if cfg.execution_mode == "remote_ssh" else (cfg.host or _resolve_local_host_ip()),
        "sshTarget": cfg.ssh_target if cfg.execution_mode == "remote_ssh" else None,
        "workdir": cfg.workdir or None,
    }

    if cfg.execution_mode == "local":
        gpu_info = _probe_local_gpu_info()
        checks: dict[str, dict[str, Any]] = {
            "ssh": {"ok": True, "detail": "local"},
            "nvidiaSmi": {"ok": bool(gpu_info.get("name")), "detail": str(gpu_info.get("name") or "未探测")},
            "conda": {"ok": None, "detail": "未探测"},
            "workdir": {"ok": True, "detail": "local"},
        }
        status: TrainingNodeStatus = "available" if checks["nvidiaSmi"]["ok"] else "placeholder"
        message = "本地 L20 节点可用" if status == "available" else "本地 GPU 未探测到"
        return _finalize_probe(
            base,
            checks,
            gpu_info,
            status=status,
            message=message,
        )

    checks: dict[str, dict[str, Any]] = {
        "ssh": {"ok": False, "detail": ""},
        "nvidiaSmi": {"ok": False, "detail": ""},
        "conda": {"ok": False, "detail": ""},
        "workdir": {"ok": False, "detail": ""},
    }
    gpu_info: dict[str, Any] = {}

    from app.services.training_node_ssh import TrainingNodeSSHClient, _resolve_ssh_auth, ssh_credentials_missing_message

    auth_mode, _auth = _resolve_ssh_auth(cfg)
    if auth_mode == "missing":
        detail = ssh_credentials_missing_message()
        checks["ssh"]["detail"] = detail
        return _finalize_probe(base, checks, gpu_info, status="unreachable", message=detail)

    try:
        client = TrainingNodeSSHClient(cfg)
        rc, out, err = client.run("echo ok", timeout=15)
        checks["ssh"]["ok"] = rc == 0 and "ok" in out
        checks["ssh"]["detail"] = out.strip() or err.strip() or ("connected" if checks["ssh"]["ok"] else "connection failed")
    except Exception as exc:
        checks["ssh"]["detail"] = str(exc)
        return _finalize_probe(base, checks, gpu_info, status="unreachable", message=str(exc))

    if not checks["ssh"]["ok"]:
        return _finalize_probe(base, checks, gpu_info, status="unreachable", message="SSH 不可连接")

    try:
        client = TrainingNodeSSHClient(cfg)
        rc, out, err = client.run(
            "nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader,nounits 2>/dev/null || nvidia-smi -L",
            timeout=20,
        )
        smi_text = (out or err).strip()
        checks["nvidiaSmi"]["ok"] = rc == 0 and bool(smi_text)
        checks["nvidiaSmi"]["detail"] = smi_text[:500]
        gpu_info = _parse_gpu_info(smi_text, fallback_model=cfg.gpu_model)
    except Exception as exc:
        checks["nvidiaSmi"]["detail"] = str(exc)

    if cfg.conda_bin:
        try:
            client = TrainingNodeSSHClient(cfg)
            rc, out, err = client.run(f"{shlex.quote(cfg.conda_bin)} --version 2>&1", timeout=15)
            detail = (out or err).strip()
            checks["conda"]["ok"] = rc == 0 and "conda" in detail.lower()
            checks["conda"]["detail"] = detail[:200]
        except Exception as exc:
            checks["conda"]["detail"] = str(exc)
    else:
        checks["conda"]["ok"] = True
        checks["conda"]["detail"] = "未配置 conda 路径"

    if cfg.workdir:
        try:
            client = TrainingNodeSSHClient(cfg)
            rc, out, err = client.run(f"test -d {shlex.quote(cfg.workdir)} && echo exists", timeout=15)
            exists = rc == 0 and "exists" in out
            checks["workdir"]["ok"] = exists
            missing_msg = _missing_workdir_message(cfg.workdir)
            checks["workdir"]["detail"] = cfg.workdir if exists else missing_msg
        except Exception as exc:
            checks["workdir"]["detail"] = str(exc)
    else:
        checks["workdir"]["detail"] = "未配置 TRAIN_NODE_L20_WORKDIR"

    if not checks["workdir"]["ok"]:
        missing_msg = _missing_workdir_message(cfg.workdir) if cfg.workdir else "未配置 TRAIN_NODE_L20_WORKDIR"
        return _finalize_probe(
            base,
            checks,
            gpu_info,
            status="misconfigured",
            message=missing_msg,
        )

    if not checks["nvidiaSmi"]["ok"]:
        return _finalize_probe(
            base,
            checks,
            gpu_info,
            status="misconfigured",
            message="nvidia-smi 不可用",
        )

    if cfg.conda_bin and not checks["conda"]["ok"]:
        return _finalize_probe(
            base,
            checks,
            gpu_info,
            status="misconfigured",
            message="conda 不可用",
        )

    busy_reason = _gpu_busy_reason(gpu_info)
    if busy_reason:
        return _finalize_probe(
            base,
            checks,
            gpu_info,
            status="busy",
            message=busy_reason,
        )

    return _finalize_probe(
        base,
        checks,
        gpu_info,
        status="available",
        message="节点可用",
    )


def _parse_gpu_info(smi_text: str, *, fallback_model: str) -> dict[str, Any]:
    match = _GPU_MEMORY_PATTERN.search(smi_text)
    if match:
        name, total, used, free = match.groups()
        total_mb = float(total)
        used_mb = float(used)
        free_mb = float(free)
        return {
            "name": name.strip(),
            "memoryTotalMb": total_mb,
            "memoryUsedMb": used_mb,
            "memoryFreeMb": free_mb,
            "memoryUsedRatio": used_mb / total_mb if total_mb > 0 else 0.0,
        }
    first_line = smi_text.splitlines()[0].strip() if smi_text else fallback_model
    return {"name": first_line or fallback_model}


def _gpu_busy_reason(gpu_info: dict[str, Any]) -> str:
    used_mb = float(gpu_info.get("memoryUsedMb") or 0)
    ratio = float(gpu_info.get("memoryUsedRatio") or 0)
    if ratio >= BUSY_GPU_MEMORY_RATIO:
        return f"GPU 显存占用 {ratio * 100:.0f}%（{used_mb:.0f} MB）"
    if used_mb >= BUSY_GPU_MEMORY_USED_MB and ratio >= 0.2:
        return f"GPU 显存已占用 {used_mb:.0f} MB"
    return ""


def _status_label(status: TrainingNodeStatus) -> str:
    return {
        "available": "可用",
        "busy": "忙碌",
        "unreachable": "不可连接",
        "misconfigured": "环境未配置",
        "placeholder": "本机 GPU 节点",
    }.get(status, status)


def _finalize_probe(
    base: dict[str, Any],
    checks: dict[str, dict[str, Any]],
    gpu_info: dict[str, Any],
    *,
    status: TrainingNodeStatus,
    message: str,
) -> dict[str, Any]:
    selectable = status in {"available", "busy", "placeholder"}
    return {
        **base,
        "status": status,
        "statusLabel": _status_label(status),
        "message": message,
        "selectable": selectable,
        "checks": checks,
        "gpu": gpu_info or None,
    }


def validate_remote_node_for_job(cfg: TrainingNodeConfig, *, allow_busy: bool = True) -> None:
    """创建训练任务前校验远程节点；不可用时抛出 ValueError。"""
    if cfg.execution_mode != "remote_ssh":
        return

    probe = probe_training_node(cfg.node_id, refresh=True)
    status = probe.get("status")
    if status == "unreachable":
        raise ValueError(probe.get("message") or "训练节点不可连接")
    if status == "misconfigured":
        raise ValueError(probe.get("message") or "训练节点环境未配置")
    if status == "busy" and not allow_busy:
        raise ValueError(probe.get("message") or "训练节点 GPU 忙碌")


def format_node_probe_report(node_id: str) -> str:
    probe = probe_training_node(node_id, refresh=True)
    lines = [
        f"节点: {probe.get('label')} ({probe.get('nodeId')})",
        f"状态: {probe.get('statusLabel')} ({probe.get('status')})",
        f"说明: {probe.get('message')}",
    ]
    if probe.get("sshTarget"):
        lines.append(f"SSH: {probe.get('sshTarget')}")
    if probe.get("workdir"):
        lines.append(f"工作目录: {probe.get('workdir')}")
    checks = probe.get("checks") or {}
    for key, label in (
        ("ssh", "SSH"),
        ("nvidiaSmi", "nvidia-smi"),
        ("conda", "conda"),
        ("workdir", "workdir"),
    ):
        item = checks.get(key) or {}
        ok = item.get("ok")
        detail = item.get("detail") or ""
        status = "OK" if ok else ("FAIL" if ok is False else "SKIP")
        lines.append(f"  [{status}] {label}: {detail[:300]}")
    gpu = probe.get("gpu") or {}
    if gpu:
        lines.append(
            f"GPU: {gpu.get('name')} | "
            f"显存 {gpu.get('memoryUsedMb', '?')}/{gpu.get('memoryTotalMb', '?')} MB"
        )
    return "\n".join(lines)
