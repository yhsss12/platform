#!/usr/bin/env python3
"""P4-C: Validate independent nut-assembly-mvp environment for MimicGen datagen."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_REPORT = REPO_ROOT / "runs" / "nut_assembly" / "debug" / "p4_mimicgen_env_validation_report.md"
ENV_NAME = os.environ.get("NUT_ASSEMBLY_MVP_ENV", "nut-assembly-mvp")
MIMICGEN_VENDOR = REPO_ROOT / "runs" / "nut_assembly" / "vendor" / "mimicgen-main"


def _run(cmd: list[str], *, env: dict | None = None) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)
    return proc.returncode, proc.stdout, proc.stderr


def _conda_python(env_name: str) -> str | None:
    code, out, _ = _run(["conda", "run", "-n", env_name, "python", "-c", "import sys; print(sys.executable)"])
    if code != 0:
        return None
    return out.strip().splitlines()[-1] if out.strip() else None


def _probe_python(python: str) -> dict:
    script = """
import json, traceback
out = {}
for mod in ("mujoco", "robosuite", "robomimic", "mimicgen", "gdown"):
    try:
        m = __import__(mod)
        out[mod] = getattr(m, "__version__", "ok")
    except Exception as e:
        out[mod] = f"FAIL: {type(e).__name__}: {e}"
try:
    import robosuite
    from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
    out["single_arm_env"] = "ok"
except Exception as e:
    out["single_arm_env"] = f"FAIL: {type(e).__name__}: {e}"
try:
    import robosuite
    env = robosuite.make("NutAssemblySquare", robots="Panda", has_renderer=False, use_camera_obs=False)
    out["NutAssemblySquare"] = "registered"
    env.close()
except Exception as e:
    out["NutAssemblySquare"] = f"FAIL: {type(e).__name__}: {e}"
print(json.dumps(out))
"""
    code, out, err = _run([python, "-c", script])
    if code != 0:
        return {"error": err or out or "probe_failed"}
    try:
        return json.loads(out.strip().splitlines()[-1])
    except json.JSONDecodeError:
        return {"raw": out, "stderr": err}


def _try_mimicgen_generate(python: str) -> dict:
    if not MIMICGEN_VENDOR.is_dir():
        return {"ok": False, "error": "mimicgen vendor missing"}
    script_path = MIMICGEN_VENDOR / "mimicgen" / "scripts" / "generate_dataset.py"
    if not script_path.is_file():
        return {"ok": False, "error": "generate_dataset.py missing"}
    env = os.environ.copy()
    env["MUJOCO_GL"] = env.get("MUJOCO_GL", "egl")
    env["PYTHONPATH"] = f"{MIMICGEN_VENDOR}:{env.get('PYTHONPATH', '')}"
    code, out, err = _run([python, str(script_path), "--help"], env=env)
    return {
        "ok": code == 0,
        "exitCode": code,
        "stdoutTail": out[-1500:],
        "stderrTail": err[-1500:],
    }


def main() -> int:
    lines: list[str] = [
        "# P4-C MimicGen 独立环境验证报告",
        "",
        f"**生成时间**: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## 1. conda env 名称",
        "",
        f"`{ENV_NAME}`",
        "",
    ]

    py = _conda_python(ENV_NAME)
    if py is None:
        lines.extend(
            [
                "## 2. 环境状态",
                "",
                f"conda 环境 `{ENV_NAME}` **未安装**。",
                "",
                "建议创建文件：`runs/nut_assembly/debug/nut-assembly-mvp-environment.yml`",
                "",
                "## 3. 当前 cable-threading-mvp 基线探测",
                "",
            ]
        )
        py = sys.executable
    else:
        lines.extend(["## 2. Python 路径", "", f"`{py}`", ""])

    probe = _probe_python(py)
    lines.extend(["## 3. 依赖探测", "", "```json", json.dumps(probe, indent=2, ensure_ascii=False), "```", ""])

    gen = _try_mimicgen_generate(py)
    lines.extend(["## 4. generate_dataset.py 启动", "", "```json", json.dumps(gen, indent=2, ensure_ascii=False), "```", ""])

    lines.extend(
        [
            "## 5. 阻塞点",
            "",
            "- `gdown` 缺失（cable-threading-mvp 环境）",
            "- `robosuite.environments.manipulation.single_arm_env` 在 CableThreadingMVP vendored robosuite 中不存在",
            "",
            "## 6. 是否建议接入平台 worker",
            "",
            "**暂不建议**。应先完成独立 `nut-assembly-mvp` 环境并在该环境中跑通 MimicGen datagen，再增加 worker 子进程切换。",
            "",
            "## 7. 不影响 robosuite_rollout",
            "",
            "当前平台主路径仍为 `robosuite_rollout` + `partial_scripted`，本验证不改变生成链路。",
        ]
    )

    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report: {OUT_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
