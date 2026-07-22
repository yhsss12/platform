#!/usr/bin/env python3
"""Probe the FrankaPickPlace controller in an Isaac Sim SimulationApp context."""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

CANDIDATE_IMPORTS = [
    {
        "label": "isaacsim.robot.experimental.manipulators.examples.franka",
        "module": "isaacsim.robot.experimental.manipulators.examples.franka",
        "attr": "FrankaPickPlace",
        "source": "NVIDIA Isaac Sim experimental FrankaPickPlace",
        "extensions": ["isaacsim.robot.experimental.manipulators.examples"],
    },
    {
        "label": "isaacsim.robot.manipulators.examples.franka",
        "module": "isaacsim.robot.manipulators.examples.franka",
        "attr": "FrankaPickPlace",
        "source": "NVIDIA Isaac Sim manipulators.examples FrankaPickPlace",
        "extensions": ["isaacsim.robot.manipulators.examples"],
    },
    {
        "label": "omni.isaac.examples.franka",
        "module": "omni.isaac.examples.franka",
        "attr": "FrankaPickPlace",
        "source": "omni.isaac.examples FrankaPickPlace",
        "extensions": [],
    },
]


def emit(payload: dict[str, object], code: int, output_path: Path | None) -> int:
    text = json.dumps(payload, ensure_ascii=False)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    print(text, flush=True)
    return code


def main() -> int:
    output_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    result: dict[str, object] = {
        "simulation_app_imported": False,
        "simulation_app_error": None,
        "controller_available": False,
        "controller_import_path": None,
        "controller_source": None,
        "candidates": [],
        "traceback": None,
    }

    simulation_app = None
    try:
        from isaacsim import SimulationApp

        result["simulation_app_imported"] = True
        simulation_app = SimulationApp({"headless": True})

        try:
            from isaacsim.core.utils import extensions as extensions_utils

            enable_extension = extensions_utils.enable_extension
        except Exception:
            enable_extension = None

        enabled_extensions: set[str] = set()
        for candidate in CANDIDATE_IMPORTS:
            for ext in candidate.get("extensions", []):
                if ext and ext not in enabled_extensions and enable_extension is not None:
                    try:
                        enable_extension(ext)
                        enabled_extensions.add(ext)
                    except Exception as exc:
                        result.setdefault("extension_errors", {})[ext] = repr(exc)

        for candidate in CANDIDATE_IMPORTS:
            entry = {
                "label": candidate["label"],
                "ok": False,
                "error": None,
                "source": candidate["source"],
            }
            try:
                module = __import__(candidate["module"], fromlist=[candidate["attr"]])
                controller_cls = getattr(module, candidate["attr"])
                entry["ok"] = True
                entry["class_module"] = getattr(controller_cls, "__module__", None)
                if not result["controller_available"]:
                    result["controller_available"] = True
                    result["controller_import_path"] = candidate["label"]
                    result["controller_source"] = candidate["source"]
            except Exception as exc:
                entry["error"] = repr(exc)
            result["candidates"].append(entry)

        emit(result, 0 if result["controller_available"] else 2, output_path)
    except Exception as exc:
        if not result.get("simulation_app_error"):
            result["simulation_app_error"] = repr(exc)
        result["traceback"] = traceback.format_exc()
        emit(result, 1, output_path)
    finally:
        if simulation_app is not None:
            try:
                simulation_app.close()
            except Exception:
                pass

    return 0 if result.get("controller_available") else 2


if __name__ == "__main__":
    raise SystemExit(main())
