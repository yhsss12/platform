"""
一键安装脚本（linux.sh）占位符：生成 run-agent.sh 与 Conda/ROS 相关片段。

与交互终端中「conda activate → source ROS → uvicorn/python agent_main」对齐；
systemd 通过 ExecStart 调用 /opt/eai-agent/run-agent.sh，避免 ExecStart 内嵌引号难以转义。
"""

from __future__ import annotations

import base64
import re
import shlex
from dataclasses import dataclass
from typing import Any, Optional

_PATH_RE = re.compile(r"^[/a-zA-Z0-9._\-]+$", re.ASCII)
_COND_ENV_RE = re.compile(r"^[\w.\-]+$", re.ASCII)
_USER_RE = re.compile(r"^[a-z_][a-z0-9_-]{0,31}$", re.IGNORECASE)


def _b64_utf8(s: str) -> str:
    return base64.b64encode(s.encode("utf-8")).decode("ascii")


def validate_abs_path(value: str, *, allow_empty: bool = False) -> str:
    v = (value or "").strip()
    if not v:
        if allow_empty:
            return ""
        raise ValueError("路径不能为空")
    if len(v) > 512:
        raise ValueError("路径过长")
    if not v.startswith("/"):
        raise ValueError("路径必须为绝对路径")
    if not _PATH_RE.match(v):
        raise ValueError("路径包含非法字符（仅允许字母数字及 /._-）")
    return v


def validate_conda_env(value: str) -> str:
    v = (value or "").strip()
    if not v:
        raise ValueError("Conda 环境名不能为空")
    if len(v) > 128:
        raise ValueError("Conda 环境名过长")
    if not _COND_ENV_RE.match(v):
        raise ValueError("Conda 环境名包含非法字符")
    return v


def validate_service_user(value: str) -> str:
    v = (value or "").strip()
    if not v:
        v = "__SUDO_USER__"
    if v == "root":
        return "root"
    if v == "__SUDO_USER__":
        return "__SUDO_USER__"
    if not _USER_RE.match(v):
        raise ValueError("systemd User 非法（仅小写用户名风格，或为 root / __SUDO_USER__）")
    return v


def _parse_optional_bool(raw: Optional[str]) -> Optional[bool]:
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if s == "":
        return None
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off"):
        return False
    raise ValueError("use_conda 取值须为 1/0/true/false")


def build_run_agent_sh(
    *,
    use_conda: bool,
    agent_dir: str,
    ros_setup: str,
    ros_ws_setup: str,
    conda_sh: str,
    conda_env: str,
) -> str:
    """生成将安装到 ${AGENT_DIR}/run-agent.sh 的完整脚本（含 shebang）。"""
    ad = validate_abs_path(agent_dir)
    ros = validate_abs_path(ros_setup)
    ros_ws = validate_abs_path(ros_ws_setup, allow_empty=True)

    # 显式处理 cd 失败；ERR trap 可捕获 set -e 下其它静默失败
    # 首行即 echo：避免旧脚本「cd 遇权限/不存在」在 set -e 时无任何可见输出
    _qad = shlex.quote(ad)
    lines: list[str] = [
        "#!/usr/bin/env bash",
        # 必须在 set -e 之前，否则旧脚本在 cd 等失败时终端可能整段无输出
        r'echo "eai-agent: run-agent.sh begin" "$(id 2>&1 || true)" >&2',
        "set -Eeuo pipefail",
        "trap 'echo eai-agent: err line $LINENO status=$? cmd=$BASH_COMMAND >&2' ERR",
        f"cd {_qad} || {{ echo 'eai-agent: cannot cd, check User=/permissions/chown' >&2; echo {_qad} >&2; exit 1; }}",
    ]
    # /opt/ros/*/setup.bash 与部分 conda 脚本会引用未定义变量，在 set -u 下整段 source 失败
    #（用 bash -c 无 -u 时则正常，易误判为「ROS 坏了」）—— 仅在 source 期关闭 nounset
    lines.append("set +u")
    if use_conda:
        cs = validate_abs_path(conda_sh)
        ce = validate_conda_env(conda_env)
        lines.append(f"source {shlex.quote(cs)} >/dev/null 2>&1")
        lines.append(f"conda activate {shlex.quote(ce)}")
    lines.append("export PYTHONUNBUFFERED=1")
    # 非 0 仍不阻启动：有的机器 humble 在部分用户下会告警退出
    lines.append(
        f"if ! source {shlex.quote(ros)} 2>/dev/null; then "
        f"echo 'eai-agent: optional: ROS setup failed, continuing' >&2; fi",
    )
    if ros_ws:
        lines.append(
            f"if [ -f {shlex.quote(ros_ws)} ]; then "
            f"if ! source {shlex.quote(ros_ws)} 2>/dev/null; then :; fi; "
            f"fi",
        )
    lines.append("set -u  # 恢复自写 if/exec 的 nounset 防护")
    if use_conda:
        # -u: unbuffered logs under systemd
        lines.append('echo "eai-agent: exec conda python -u agent_main.py (user=$(id -un 2>/dev/null) cwd=$(pwd))" >&2')
        lines.append("exec python -u agent_main.py")
    else:
        py = f"{ad}/venv/bin/python"
        main_py = f"{ad}/agent_main.py"
        q_py = shlex.quote(py)
        q_main = shlex.quote(main_py)
        lines.append(f"if [ ! -x {q_py} ]; then")
        lines.append(
            f'  echo "eai-agent: missing venv at {q_py} (re-run linux.sh installer)" >&2',
        )
        lines.append("  exit 1")
        lines.append("fi")
        # journalctl 中至少能见到一行，便于与「仅 systemd 行」区分；python 的 stderr 也会进 journal
        lines.append(f'echo "eai-agent: exec {q_py} -u {q_main} (user=$(id -un 2>/dev/null) cwd=$(pwd))" >&2')
        lines.append(f"exec {q_py} -u {q_main}")
    return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class LinuxInstallerScriptReplacements:
    agent_use_conda: str
    run_agent_sh_b64: str
    conda_sh_b64: str
    conda_env_b64: str
    service_user: str


def _merge_opt_str(query_val: Optional[str], setting_val: str) -> str:
    if query_val is not None:
        return str(query_val).strip()
    return str(setting_val or "").strip()


def resolve_linux_installer_script(
    *,
    settings: Any,
    query_use_conda: Optional[str] = None,
    query_conda_sh: Optional[str] = None,
    query_conda_env: Optional[str] = None,
    query_ros_setup: Optional[str] = None,
    query_ros_ws_setup: Optional[str] = None,
    query_service_user: Optional[str] = None,
) -> LinuxInstallerScriptReplacements:
    """
    合并 settings 与 URL 查询参数（查询优先），并校验 Conda 模式下必填项。
    """
    q_bool = _parse_optional_bool(query_use_conda)
    use_conda = q_bool if q_bool is not None else bool(getattr(settings, "AGENT_INSTALL_USE_CONDA", False))

    conda_sh = _merge_opt_str(query_conda_sh, str(getattr(settings, "AGENT_INSTALL_CONDA_SH", "") or ""))
    conda_env = _merge_opt_str(query_conda_env, str(getattr(settings, "AGENT_INSTALL_CONDA_ENV", "") or ""))

    ros_setup = _merge_opt_str(query_ros_setup, str(getattr(settings, "AGENT_INSTALL_ROS_SETUP", "") or ""))
    if not ros_setup:
        ros_setup = "/opt/ros/humble/setup.bash"
    ros_ws_setup = _merge_opt_str(query_ros_ws_setup, str(getattr(settings, "AGENT_INSTALL_ROS_WS_SETUP", "") or ""))

    service_user_raw = _merge_opt_str(query_service_user, str(getattr(settings, "AGENT_INSTALL_SERVICE_USER", "") or ""))
    if not service_user_raw:
        service_user_raw = "__SUDO_USER__"
    service_user = validate_service_user(service_user_raw)

    ros_setup = validate_abs_path(ros_setup)
    ros_ws_norm = validate_abs_path(ros_ws_setup, allow_empty=True)

    if use_conda:
        conda_sh = validate_abs_path(conda_sh)
        conda_env = validate_conda_env(conda_env)
    else:
        conda_sh = ""
        conda_env = ""

    agent_dir = "/opt/eai-agent"
    run_sh = build_run_agent_sh(
        use_conda=use_conda,
        agent_dir=agent_dir,
        ros_setup=ros_setup,
        ros_ws_setup=ros_ws_norm,
        conda_sh=conda_sh,
        conda_env=conda_env,
    )

    return LinuxInstallerScriptReplacements(
        agent_use_conda="1" if use_conda else "0",
        run_agent_sh_b64=_b64_utf8(run_sh),
        conda_sh_b64=_b64_utf8(conda_sh),
        conda_env_b64=_b64_utf8(conda_env),
        service_user=service_user,
    )


def apply_linux_installer_replacements(script: str, repl: LinuxInstallerScriptReplacements) -> str:
    out = script
    out = out.replace("{{AGENT_USE_CONDA}}", repl.agent_use_conda)
    out = out.replace("{{RUN_AGENT_SH_B64}}", repl.run_agent_sh_b64)
    out = out.replace("{{CONDA_SH_B64}}", repl.conda_sh_b64)
    out = out.replace("{{CONDA_ENV_B64}}", repl.conda_env_b64)
    out = out.replace("{{SERVICE_USER}}", repl.service_user)
    return out
