from __future__ import annotations

import json
import subprocess
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from utils.runtime_env import (
    build_rollout_subprocess_env,
    resolve_rollout_python,
)


def run_repair_rollout_subprocess(
    *,
    env_name: str,
    seed: int,
    horizon: int,
    extra_xy_bias: np.ndarray | None = None,
    timeout_s: int = 900,
) -> dict[str, Any]:
    """Run one repair rollout in cable-threading-mvp where NutAssemblySquare scripted policy is validated."""
    python_bin = resolve_rollout_python()
    if not python_bin.is_file():
        return {"ok": False, "error": f"cable-threading-mvp python not found: {python_bin}"}

    script = Path(__file__).resolve().parents[1] / "scripts" / "run_pinn_repair_rollout.py"
    output_json = Path("/tmp") / f"pinn_repair_rollout_{seed}_{abs(hash(str(extra_xy_bias))) & 0xFFFF:x}.json"
    cmd = [
        str(python_bin),
        str(script),
        "--env-name",
        env_name,
        "--seed",
        str(seed),
        "--horizon",
        str(horizon),
        "--output-json",
        str(output_json),
    ]
    if extra_xy_bias is not None:
        cmd.extend(["--extra-xy-bias", json.dumps(np.asarray(extra_xy_bias, dtype=float).tolist())])

    env = build_rollout_subprocess_env()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(Path(__file__).resolve().parents[1]),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {"ok": False, "error": "repair_rollout_timeout", "traceback": (exc.stdout or "") + (exc.stderr or "")}
    except Exception:
        return {"ok": False, "error": "repair_rollout_launch_failed", "traceback": traceback.format_exc()}

    if not output_json.is_file():
        return {
            "ok": False,
            "error": f"repair rollout exited {proc.returncode} without result json",
            "traceback": (proc.stdout or "") + (proc.stderr or ""),
        }
    try:
        payload = json.loads(output_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"ok": False, "error": f"invalid repair rollout result: {exc}"}
    if proc.returncode != 0 and payload.get("ok"):
        payload["returnCode"] = proc.returncode
    return payload


def run_rollout_fallback_subprocess(
    *,
    job_root: Path,
    env_name: str,
    episodes: int,
    seed: int,
    horizon: int,
    render_video: bool,
) -> dict[str, Any]:
    """
    Run robosuite rollout in cable-threading-mvp subprocess.
    Keeps MimicGen worker (nut-assembly-mvp) free of CableThreadingMVP robosuite imports.
    """
    python_bin = resolve_rollout_python()
    if not python_bin.is_file():
        return {
            "ok": False,
            "error": f"cable-threading-mvp python not found: {python_bin}",
            "traceback": "",
        }

    output_json = job_root / "intermediate" / "rollout_fallback_result.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_rollout_fallback.py"
    cmd = [
        str(python_bin),
        str(script),
        "--job-root",
        str(job_root),
        "--episodes",
        str(episodes),
        "--seed",
        str(seed),
        "--horizon",
        str(horizon),
        "--env-name",
        env_name,
        "--output-json",
        str(output_json),
    ]
    if render_video:
        cmd.append("--render-video")

    env = build_rollout_subprocess_env()
    log_path = job_root / "logs" / "fallback_rollout.log"
    header = [
        "=== launching rollout fallback subprocess ===",
        f"fallback_python={python_bin}",
        f"command={' '.join(cmd)}",
    ]
    log_path.write_text("\n".join(header) + "\n", encoding="utf-8")

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(Path(__file__).resolve().parents[1]),
            env=env,
            capture_output=True,
            text=True,
            timeout=1800,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        combined = (exc.stdout or "") + "\n" + (exc.stderr or "")
        return {
            "ok": False,
            "error": "rollout_fallback_timeout",
            "traceback": combined,
            "returnCode": -1,
        }
    except Exception:
        return {
            "ok": False,
            "error": "rollout_fallback_launch_failed",
            "traceback": traceback.format_exc(),
            "returnCode": -1,
        }

    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(combined)
        fh.write(f"\nreturn_code={proc.returncode}\n")

    if not output_json.is_file():
        return {
            "ok": False,
            "error": f"rollout fallback exited {proc.returncode} without result json",
            "traceback": combined,
            "returnCode": proc.returncode,
        }

    try:
        payload = json.loads(output_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "ok": False,
            "error": f"invalid rollout fallback result: {exc}",
            "traceback": combined,
            "returnCode": proc.returncode,
        }

    if proc.returncode != 0 or not payload.get("ok"):
        return {
            "ok": False,
            "error": str(payload.get("error") or f"rollout fallback exit {proc.returncode}"),
            "traceback": str(payload.get("traceback") or combined),
            "returnCode": proc.returncode,
            "rolloutResult": payload.get("rolloutResult"),
        }

    rollout_result = payload.get("rolloutResult") or {}
    return {
        "ok": True,
        "rolloutResult": rollout_result,
        "hdf5Info": payload.get("hdf5Info") or {},
        "returnCode": proc.returncode,
    }
