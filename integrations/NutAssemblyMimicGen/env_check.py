#!/usr/bin/env python3
"""NutAssembly MimicGen environment validation (P5)."""

from __future__ import annotations

import json
import platform
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

_INTEGRATION_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _INTEGRATION_ROOT.parents[1]
if str(_INTEGRATION_ROOT) not in sys.path:
    sys.path.insert(0, str(_INTEGRATION_ROOT))

from utils.runtime_env import (  # noqa: E402
    ENV_CHECK_OUTPUT,
    MIMICGEN_VENDOR,
    NUT_ASSEMBLY_MVP_PYTHON,
    resolve_mimicgen_root,
)

OUTPUT_PATH = ENV_CHECK_OUTPUT


def _check_import(name: str, import_fn) -> dict[str, Any]:
    try:
        mod = import_fn()
        version = getattr(mod, "__version__", None)
        return {"ok": True, "version": version}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "traceback": traceback.format_exc()}


def _check_env_registered(env_name: str) -> dict[str, Any]:
    try:
        import robosuite as suite

        if hasattr(suite, "ALL_ENVIRONMENTS"):
            raw = suite.ALL_ENVIRONMENTS
            registered = list(raw.keys()) if hasattr(raw, "keys") else list(raw)
        else:
            registered = []
        ok = env_name in registered
        return {
            "ok": ok,
            "registered": ok,
            "availableSample": registered[:20],
            "error": None if ok else f"{env_name} not in robosuite.ALL_ENVIRONMENTS",
        }
    except Exception as exc:
        return {"ok": False, "registered": False, "error": str(exc), "traceback": traceback.format_exc()}


def _check_mg_nut_assembly() -> dict[str, Any]:
    try:
        from mimicgen.env_interfaces.robosuite import MG_NutAssembly  # noqa: F401

        return {"ok": True, "class": "MG_NutAssembly"}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "traceback": traceback.format_exc()}


def _check_reset_env(env_name: str) -> dict[str, Any]:
    try:
        import robosuite as suite

        env = suite.make(
            env_name,
            robots="Panda",
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=False,
            control_freq=20,
        )
        env.reset()
        env.close()
        return {"ok": True, "envName": env_name}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "traceback": traceback.format_exc()}


def _check_object_poses(env_name: str) -> dict[str, Any]:
    try:
        import numpy as np
        import robosuite as suite
        from mimicgen.env_interfaces.robosuite import MG_NutAssembly

        env = suite.make(
            env_name,
            robots="Panda",
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            control_freq=20,
        )
        env.reset()
        interface = MG_NutAssembly(env=env)
        action = np.zeros(env.action_dim)
        info = interface.get_datagen_info(action=action)
        poses = info.object_poses if hasattr(info, "object_poses") else {}
        keys = list(poses.keys()) if isinstance(poses, dict) else []
        env.close()
        return {"ok": True, "objectPoseKeys": keys}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "traceback": traceback.format_exc()}


def _check_offscreen_render(env_name: str) -> dict[str, Any]:
    try:
        import robosuite as suite

        env = suite.make(
            env_name,
            robots="Panda",
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            camera_names="agentview",
            camera_heights=84,
            camera_widths=84,
            control_freq=20,
        )
        env.reset()
        obs = env.sim.render(width=84, height=84, camera_name="agentview")
        env.close()
        shape = getattr(obs, "shape", None)
        return {"ok": obs is not None, "renderShape": list(shape) if shape is not None else None}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "traceback": traceback.format_exc()}


def run_env_check() -> dict[str, Any]:
    mimicgen_root = resolve_mimicgen_root()
    if mimicgen_root and str(mimicgen_root) not in sys.path:
        sys.path.insert(0, str(mimicgen_root))

    # Register MimicGen robosuite envs (Square_D0, NutAssembly_D0, etc.)
    try:
        import mimicgen.envs.robosuite.nut_assembly  # noqa: F401
    except Exception as exc:
        return {
            "checkedAt": datetime.now().isoformat(timespec="seconds"),
            "overallOk": False,
            "fatalError": traceback.format_exc(),
            "mimicgenEnvImportError": str(exc),
        }

    report: dict[str, Any] = {
        "checkedAt": datetime.now().isoformat(timespec="seconds"),
        "pythonExecutable": sys.executable,
        "pythonVersion": platform.python_version(),
        "platform": platform.platform(),
        "nutAssemblyMvpPython": str(NUT_ASSEMBLY_MVP_PYTHON),
        "nutAssemblyMvpExists": NUT_ASSEMBLY_MVP_PYTHON.is_file(),
        "mimicgenRoot": str(mimicgen_root) if mimicgen_root else None,
        "mimicgenVendorPath": str(MIMICGEN_VENDOR),
        "checks": {},
        "overallOk": False,
    }

    report["checks"]["python"] = {"ok": True, "version": platform.python_version()}
    report["checks"]["mujoco"] = _check_import("mujoco", lambda: __import__("mujoco"))
    report["checks"]["robosuite"] = _check_import("robosuite", lambda: __import__("robosuite"))
    try:
        import robosuite

        robosuite_file = str(getattr(robosuite, "__file__", ""))
        report["checks"]["robosuite"]["file"] = robosuite_file
        if "CableThreadingMVP" in robosuite_file.replace("\\", "/"):
            report["checks"]["robosuite"]["ok"] = False
            report["checks"]["robosuite"]["error"] = f"wrong_robosuite_source: {robosuite_file}"
    except Exception:
        pass
    report["checks"]["robosuite_macros_private"] = _check_import(
        "robosuite.macros_private", lambda: __import__("robosuite.macros_private")
    )
    report["checks"]["termcolor"] = _check_import("termcolor", lambda: __import__("termcolor"))
    report["checks"]["robomimic"] = _check_import("robomimic", lambda: __import__("robomimic"))
    report["checks"]["mimicgen"] = _check_import("mimicgen", lambda: __import__("mimicgen"))
    report["checks"]["gdown"] = _check_import("gdown", lambda: __import__("gdown"))

    square_env = "Square_D0"
    nut_env = "NutAssembly_D0"
    report["checks"]["Square_D0_registered"] = _check_env_registered(square_env)
    report["checks"]["NutAssembly_D0_registered"] = _check_env_registered(nut_env)
    report["checks"]["MG_NutAssembly"] = _check_mg_nut_assembly()

    reset_env = square_env
    if report["checks"]["Square_D0_registered"].get("ok"):
        reset_env = square_env
    elif report["checks"]["NutAssembly_D0_registered"].get("ok"):
        reset_env = nut_env

    report["checks"]["reset_env"] = _check_reset_env(reset_env)
    report["checks"]["object_poses"] = _check_object_poses(reset_env)
    report["checks"]["offscreen_render"] = _check_offscreen_render(reset_env)

    critical = [
        report["checks"]["mujoco"]["ok"],
        report["checks"]["robosuite"]["ok"],
        report["checks"]["robosuite_macros_private"]["ok"],
        report["checks"]["termcolor"]["ok"],
        report["checks"]["robomimic"]["ok"],
        report["checks"]["mimicgen"]["ok"],
        report["checks"]["MG_NutAssembly"]["ok"],
        report["checks"]["reset_env"]["ok"],
    ]
    report["overallOk"] = all(critical)
    return report


def main() -> int:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = run_env_check()
    except Exception:
        report = {
            "checkedAt": datetime.now().isoformat(timespec="seconds"),
            "overallOk": False,
            "fatalError": traceback.format_exc(),
        }
    OUTPUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nWrote: {OUTPUT_PATH}")
    return 0 if report.get("overallOk") else 1


if __name__ == "__main__":
    raise SystemExit(main())
