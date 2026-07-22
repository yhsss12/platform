from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_diagnose_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "diagnose_isaac_runtime.py"
    spec = importlib.util.spec_from_file_location("diagnose_isaac_runtime", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_diagnose_loads_dotenv_isaaclab_root(monkeypatch, tmp_path):
    mod = _load_diagnose_module()
    env_file = tmp_path / ".env"
    env_file.write_text(
        f"ISAACLAB_ROOT={tmp_path / 'IsaacLab'}\nISAACLAB_PYTHON=/tmp/isaac/bin/python\n",
        encoding="utf-8",
    )
    isaaclab = tmp_path / "IsaacLab"
    isaaclab.mkdir()
    (isaaclab / "isaaclab.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    (isaaclab / "isaaclab.sh").chmod(0o755)

    monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(mod, "probe_runtime_imports", lambda runner: {"can_import_isaaclab": True, "can_import_isaacsim": True, "can_import_simulation_app": True})
    monkeypatch.setattr(
        mod,
        "probe_franka_controller",
        lambda runner: {"controller_available": False, "controller_import_path": None},
    )
    monkeypatch.setattr(mod, "search_franka_pick_place_files", lambda: [])

    report = mod.diagnose()
    assert report["isaac_lab_runtime_available"] is True
    assert report["diagnosis"] == "controller_unavailable"
    assert "Isaac Lab runtime detected" in str(report["skip_reason"])


def test_diagnose_reports_ready_when_controller_available(monkeypatch, tmp_path):
    mod = _load_diagnose_module()
    monkeypatch.setattr(mod, "_load_dotenv", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        mod,
        "discover_runners",
        lambda: [
            mod.Runner(kind="python", command=["/tmp/isaac/bin/python"], label="ISAACLAB_PYTHON")
        ],
    )
    monkeypatch.setattr(
        mod,
        "probe_runtime_imports",
        lambda runner: {
            "can_import_isaaclab": True,
            "can_import_isaacsim": True,
            "can_import_simulation_app": True,
            "detected_isaac_version": "5.1.0",
        },
    )
    monkeypatch.setattr(
        mod,
        "probe_franka_controller",
        lambda runner: {
            "controller_available": True,
            "controller_import_path": "isaacsim.robot.manipulators.examples.franka",
        },
    )
    monkeypatch.setattr(mod, "search_franka_pick_place_files", lambda: ["/tmp/pick_place.py"])

    report = mod.diagnose()
    assert report["diagnosis"] == "ready"
    assert report["can_import_franka_pick_place"] is True
    assert report["franka_pick_place_import_path"] == "isaacsim.robot.manipulators.examples.franka"
