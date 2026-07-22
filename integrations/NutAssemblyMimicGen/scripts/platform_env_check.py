#!/usr/bin/env python3
"""P0: NutAssembly 环境启动验证（只读 smoke，不修改 MimicGen / CableThreadingMVP 源码）。"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_INTEGRATION_ROOT = _SCRIPT_DIR.parent
_REPO_ROOT = _INTEGRATION_ROOT.parents[1]
_DATA_ROOT = Path(os.environ.get("EAI_DATA_ROOT") or (_REPO_ROOT / "eai-data")).expanduser()
_CABLE_MVP = _REPO_ROOT / "integrations" / "CableThreadingMVP"
_MIMICGEN_ROOT = _REPO_ROOT / "third_party" / "mimicgen" / "mimicgen"
_DEFAULT_OUT = _DATA_ROOT / "runs" / "nut_assembly" / "debug" / "platform_env_check.json"

# robosuite vendored（CableThreadingMVP）可直接加载的环境
_ROBOSUITE_ENV_CANDIDATES = [
    "NutAssemblySquare",
    "NutAssembly",
    "NutAssemblyRound",
]

# MimicGen 扩展环境（需临时 prepend third_party/mimicgen 到 sys.path）
_MIMICGEN_ENV_CANDIDATES = [
    "Square_D0",
    "NutAssembly_D0",
]


def _ensure_robosuite_path() -> None:
    cable_str = str(_CABLE_MVP)
    if cable_str not in sys.path:
        sys.path.insert(0, cable_str)


def _resolve_mimicgen_parent() -> Path | None:
    for candidate in (_MIMICGEN_ROOT.parent,):
        if (candidate / "mimicgen" / "__init__.py").is_file():
            return candidate
    return None


def _maybe_prepend_mimicgen() -> tuple[bool, str | None]:
    mimicgen_parent_path = _resolve_mimicgen_parent()
    if mimicgen_parent_path is None:
        return False, f"mimicgen package not found under {_MIMICGEN_ROOT.parent}"
    mimicgen_parent = str(mimicgen_parent_path)
    if mimicgen_parent not in sys.path:
        sys.path.insert(0, mimicgen_parent)
    try:
        import mimicgen  # noqa: F401

        return True, None
    except Exception as exc:
        return False, str(exc)


def _make_env(env_name: str):
    import robosuite

    return robosuite.make(
        env_name=env_name,
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
        horizon=200,
    )


def _check_success(env: Any) -> tuple[bool, Any]:
    if hasattr(env, "_check_success"):
        try:
            return True, bool(env._check_success())
        except Exception as exc:
            return False, str(exc)
    return False, "no _check_success on env"


def _read_object_poses_via_mg_interface(env: Any) -> tuple[list[str], str | None]:
    try:
        from mimicgen.env_interfaces.base import make_interface

        iface = make_interface(name="MG_NutAssembly", interface_type="robosuite", env=env)
        poses = iface.get_object_poses()
        return sorted(list(poses.keys())), None
    except Exception as exc:
        return [], str(exc)


def _read_object_poses_fallback(env: Any) -> list[str]:
    keys: list[str] = []
    if hasattr(env, "nuts") and hasattr(env, "nut_to_id"):
        for nut_name in getattr(env, "nut_to_id", {}):
            keys.append(f"{nut_name}_nut")
    for peg in ("peg1", "peg2"):
        try:
            env.sim.model.body_name2id(peg)
            keys.append(f"{peg}_peg")
        except Exception:
            pass
    return sorted(keys)


def _probe_env(env_name: str, *, source: str, try_mg_interface: bool) -> dict[str, Any]:
    result: dict[str, Any] = {
        "ok": False,
        "env_name": env_name,
        "env_source": source,
        "reset_ok": False,
        "step_ok": False,
        "success_check_ok": False,
        "success_value": None,
        "object_pose_keys": [],
        "object_pose_source": None,
        "mujoco_version": "",
        "robosuite_source": str(_CABLE_MVP),
        "error": None,
    }
    env = None
    try:
        import mujoco

        result["mujoco_version"] = str(getattr(mujoco, "__version__", ""))
        env = _make_env(env_name)
        env.reset()
        result["reset_ok"] = True

        low, high = env.action_spec
        action = np.zeros_like((low + high) / 2.0)
        for _ in range(5):
            env.step(action)
        result["step_ok"] = True

        sc_ok, sc_val = _check_success(env)
        result["success_check_ok"] = sc_ok
        result["success_value"] = sc_val

        if try_mg_interface:
            keys, err = _read_object_poses_via_mg_interface(env)
            if keys:
                result["object_pose_keys"] = keys
                result["object_pose_source"] = "MG_NutAssembly"
            elif err:
                result["object_pose_source"] = f"MG_NutAssembly_failed: {err}"
                result["object_pose_keys"] = _read_object_poses_fallback(env)
            else:
                result["object_pose_keys"] = _read_object_poses_fallback(env)
                result["object_pose_source"] = "sim_fallback"
        else:
            result["object_pose_keys"] = _read_object_poses_fallback(env)
            result["object_pose_source"] = "sim_fallback"

        result["ok"] = result["reset_ok"] and result["step_ok"] and result["success_check_ok"]
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    return result


def run_checks(*, include_mimicgen: bool) -> dict[str, Any]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    _ensure_robosuite_path()
    # CableThreadingMVP/robosuite 已在 controller.py 内适配 MuJoCo 3.7 mj_fullM，勿再 monkeypatch。

    summary: dict[str, Any] = {
        "ok": False,
        "robosuite_source": str(_CABLE_MVP),
        "mimicgen_root": str(_resolve_mimicgen_parent() or _MIMICGEN_ROOT.parent),
        "mimicgen_import_ok": False,
        "mimicgen_import_error": None,
        "environments": [],
        "recommended_env": None,
    }

    for env_name in _ROBOSUITE_ENV_CANDIDATES:
        summary["environments"].append(
            _probe_env(env_name, source="CableThreadingMVP/robosuite", try_mg_interface=False)
        )

    if include_mimicgen:
        mg_ok, mg_err = _maybe_prepend_mimicgen()
        summary["mimicgen_import_ok"] = mg_ok
        summary["mimicgen_import_error"] = mg_err
        if mg_ok:
            for env_name in _MIMICGEN_ENV_CANDIDATES:
                summary["environments"].append(
                    _probe_env(env_name, source="third_party/mimicgen", try_mg_interface=True)
                )

    for row in summary["environments"]:
        if row.get("ok"):
            summary["recommended_env"] = row.get("env_name")
            summary["ok"] = True
            break

    if summary["recommended_env"] is None:
        for row in summary["environments"]:
            if row.get("reset_ok") and row.get("step_ok"):
                summary["recommended_env"] = row.get("env_name")
                summary["ok"] = True
                break

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="NutAssembly platform env check (P0)")
    parser.add_argument(
        "--output",
        type=str,
        default=str(_DEFAULT_OUT),
        help="JSON output path",
    )
    parser.add_argument(
        "--skip-mimicgen",
        action="store_true",
        help="Only test CableThreadingMVP robosuite envs",
    )
    args = parser.parse_args()

    payload = run_checks(include_mimicgen=not args.skip_mimicgen)
    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"wrote: {out_path}")
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
