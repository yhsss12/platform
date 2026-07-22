"""
平台隧道命令辅助：供 agent_main 在处理 CMD_REQUEST 时调用。

需处理：
- SCRIPT_GET_REPORT：内存中调用 validate_bag.validate，不落盘 json
- SCRIPT_DELETE_DATA：删除采集输出文件或目录

与 backend `routes_script.py` 约定：`path` 经平台归一化为 episode 目录（含 *.mcap 所在目录）。
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Any, Dict


def tunnel_cmd_script_delete_data(data: Dict[str, Any]) -> Dict[str, Any]:
    p = str(data.get("path") or "").strip()
    allow_job_workspace = bool(data.get("allow_job_workspace"))
    allow_incomplete = bool(data.get("allow_incomplete_episode"))
    ws_root = str(data.get("workspace_root") or "").strip()
    if not p:
        return {"success": False, "msg": "path required"}

    def _is_under_workspace_root(root: str, target: str) -> bool:
        if not (root or "").strip():
            return False
        try:
            rr = os.path.realpath(root.strip())
            rt = os.path.realpath(target)
            return rt == rr or rt.startswith(rr + os.sep)
        except OSError:
            return False

    def _is_job_workspace_segment_dir(path: str) -> bool:
        try:
            bn = os.path.basename(os.path.normpath(path).rstrip(os.sep))
        except Exception:
            return False
        return len(bn) == 4 and bn.isdigit()
    # 防误删：仅允许删除 bag 文件或“episode 目录”（目录下含 .mcap/.db3 或 validation_report.json）
    try:
        rp = os.path.realpath(p)
        # 永久保护：禁止删除 agent 安装目录
        if rp == "/opt/eai-agent" or rp.startswith("/opt/eai-agent/"):
            return {"success": False, "msg": f"refuse to delete protected path: {p}"}
    except Exception:
        pass

    def _looks_like_bag_file(x: str) -> bool:
        low = (x or "").strip().lower()
        return low.endswith(".mcap") or low.endswith(".mca") or low.endswith(".db3") or low.endswith(".bag")

    def _dir_has_bag_files(d: str) -> bool:
        try:
            if not os.path.isdir(d):
                return False
            for name in os.listdir(d):
                low = name.lower()
                if low.endswith(".mcap") or low.endswith(".mca") or low.endswith(".db3"):
                    full = os.path.join(d, name)
                    if os.path.isfile(full):
                        return True
            if os.path.isfile(os.path.join(d, "validation_report.json")):
                return True
        except OSError:
            return False
        return False

    try:
        if os.path.isdir(p):
            if allow_job_workspace and _is_job_workspace_segment_dir(p):
                pass
            elif (
                allow_incomplete
                and ws_root
                and _is_under_workspace_root(ws_root, p)
            ):
                try:
                    bn = os.path.basename(os.path.normpath(p).rstrip(os.sep))
                except Exception:
                    bn = ""
                if not bn.lower().startswith("episode_"):
                    return {"success": False, "msg": f"refuse incomplete delete: not episode dir: {p}"}
            elif not _dir_has_bag_files(p):
                return {"success": False, "msg": f"refuse to delete non-bag dir: {p}"}
        else:
            if not _looks_like_bag_file(p):
                return {"success": False, "msg": f"refuse to delete non-bag file: {p}"}
    except Exception:
        return {"success": False, "msg": f"refuse to delete unsafe path: {p}"}
    try:
        if not os.path.lexists(p):
            return {"success": True, "msg": "ok"}
        if os.path.isdir(p) and not os.path.islink(p):
            shutil.rmtree(p)
        else:
            os.remove(p)
        return {"success": True, "msg": "ok"}
    except Exception as e:
        return {"success": False, "msg": str(e)}


def tunnel_cmd_script_get_report(data: Dict[str, Any]) -> Dict[str, Any]:
    raw = str(data.get("path") or "").strip()
    if not raw:
        return {"success": False, "msg": "path required"}

    from validate_bag import bag_container_dir, validate

    bag_target = raw if os.path.exists(raw) else bag_container_dir(raw)
    if not os.path.exists(bag_target):
        return {"success": False, "msg": f"bag path not found on client: {bag_target}"}

    # 先做一次最小自检：很多“Validation report not found”其实是 ros2 bag info 失败导致 validate 返回 None。
    # 这里把关键 stderr/stdout 带回平台，便于定位（ROS 环境、bag 损坏、权限等）。
    try:
        p = subprocess.run(
            ["ros2", "bag", "info", bag_target],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if p.returncode != 0:
            out = (p.stdout or "").strip()
            err = (p.stderr or "").strip()
            preview = (err or out)[:800] if (err or out) else ""
            return {
                "success": False,
                "msg": f"ros2 bag info failed (rc={p.returncode}): {preview or 'no output'}",
            }
    except FileNotFoundError:
        return {"success": False, "msg": "ros2 command not found on client (PATH missing?)"}
    except Exception as e:
        return {"success": False, "msg": f"ros2 bag info probe failed: {e}"}
    try:
        report = validate(bag_target, 30.0, "raw")
    except Exception as e:
        return {"success": False, "msg": f"validate_bag failed: {e}"}
    if not report:
        return {"success": False, "msg": "validate_bag returned empty report"}
    return {"success": True, "data": report}


def dispatch_tunnel_script_command(cmd: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """返回结构需符合平台 CMD_RESULT.payload：含 success / msg / data。"""
    if cmd == "SCRIPT_GET_REPORT":
        return tunnel_cmd_script_get_report(data)
    if cmd == "SCRIPT_DELETE_DATA":
        return tunnel_cmd_script_delete_data(data)
    return {"success": False, "msg": f"unsupported cmd: {cmd}"}

