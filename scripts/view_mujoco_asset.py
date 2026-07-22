#!/usr/bin/env python3
"""Open a MuJoCo XML asset in the interactive viewer (CLI only)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="View a MuJoCo XML asset with mujoco.viewer")
    parser.add_argument("--xml", required=True, help="Path to model_preview.xml or model.xml")
    args = parser.parse_args()

    xml_path = Path(args.xml).expanduser().resolve()
    if not xml_path.is_file():
        print(f"error: XML not found: {xml_path}", file=sys.stderr)
        return 1

    try:
        import mujoco
        import mujoco.viewer
    except ImportError as exc:
        print(
            "error: mujoco package is not installed in this Python environment.\n"
            "Install mujoco>=3.0 in a desktop/conda env, then rerun this script.",
            file=sys.stderr,
        )
        print(f"detail: {exc}", file=sys.stderr)
        return 1

    try:
        model = mujoco.MjModel.from_xml_path(str(xml_path))
        data = mujoco.MjData(model)
    except Exception as exc:
        print(f"error: failed to load MuJoCo model from {xml_path}", file=sys.stderr)
        print(f"detail: {exc}", file=sys.stderr)
        return 1

    print(
        f"loaded: nbody={model.nbody} ngeom={model.ngeom} nmesh={model.nmesh} "
        f"nq={model.nq} nv={model.nv}"
    )
    print("launching viewer (requires display / GLFW)...")

    try:
        mujoco.viewer.launch(model, data)
    except Exception as exc:
        print(f"error: viewer launch failed: {exc}", file=sys.stderr)
        print("hint: run on a machine with a graphical display, or use X11 forwarding.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
