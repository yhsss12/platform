#!/usr/bin/env python3
"""Probe Isaac runtime imports without starting SimulationApp."""

from __future__ import annotations

import json
import sys

CHECKS = [
    "isaacsim",
    "omni.isaac.kit",
    "omni.isaac.core",
    "isaaclab",
    "isaaclab_tasks",
]


def main() -> int:
    payload = {
        "python_executable": sys.executable,
        "can_import_isaacsim": False,
        "can_import_omni_isaac_kit": False,
        "can_import_omni_isaac_core": False,
        "can_import_isaaclab": False,
        "can_import_isaaclab_tasks": False,
        "can_import_simulation_app": False,
        "detected_isaac_version": None,
        "errors": {},
    }

    for module_name in CHECKS:
        try:
            __import__(module_name)
            payload[f"can_import_{module_name.replace('.', '_')}"] = True
        except Exception as exc:
            payload["errors"][module_name] = repr(exc)

    try:
        import isaacsim  # noqa: F401

        payload["can_import_isaacsim"] = True
        payload["detected_isaac_version"] = getattr(isaacsim, "__version__", None)
    except Exception as exc:
        payload["errors"]["isaacsim"] = repr(exc)

    try:
        from isaacsim import SimulationApp  # noqa: F401

        payload["can_import_simulation_app"] = True
    except Exception as exc:
        payload["errors"]["SimulationApp"] = repr(exc)

    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
