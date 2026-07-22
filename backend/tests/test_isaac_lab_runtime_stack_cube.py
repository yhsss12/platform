from __future__ import annotations

from pathlib import Path

import pytest

from app.services.isaac_lab.generate_cli import DEFAULT_MIMIC_TASK_ID
from app.services.isaac_lab.isaac_runtime_service import (
    STACK_CUBE_ISSUE_MISSING_DEFAULT_SEED,
    STACK_CUBE_ISSUE_MISSING_ROOT,
    STACK_CUBE_ISSUE_RUNTIME_DISABLED,
    get_runtime_status,
)


def _configure_isaac_root(tmp_path: Path) -> Path:
    root = tmp_path / "IsaacLab"
    root.mkdir()
    sh = root / "isaaclab.sh"
    sh.write_text("#!/bin/bash\n", encoding="utf-8")
    sh.chmod(0o755)
    tasks_dir = root / "source" / "isaaclab_tasks"
    tasks_dir.mkdir(parents=True)
    (root / "VERSION").write_text("2.3.2\n", encoding="utf-8")
    (tasks_dir / "stack_mimic.py").write_text(
        f'ENV_ID = "{DEFAULT_MIMIC_TASK_ID}"\n',
        encoding="utf-8",
    )
    (tasks_dir / "stack_replay.py").write_text(
        'ENV_ID = "Isaac-Stack-Cube-Franka-IK-Rel-v0"\n',
        encoding="utf-8",
    )
    return root


def test_runtime_status_reports_default_seed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = _configure_isaac_root(tmp_path)
    seed = tmp_path / "stack_cube_seed.hdf5"
    seed.write_bytes(b"hdf5")

    from app.core.config import settings

    settings.ISAACLAB_ROOT = str(root)
    settings.ISAACLAB_RUNTIME_ENABLED = True
    settings.ISAACLAB_STACK_CUBE_DEFAULT_SEED = str(seed)

    monkeypatch.setattr(
        "app.services.isaac_lab.isaac_runtime_service.check_gpu_available",
        lambda: (True, []),
    )

    status = get_runtime_status()
    assert status["defaultSeedAvailable"] is True
    assert status["defaultSeedFile"] == str(seed)
    assert status["stackCubeGenerationReady"] is True
    assert STACK_CUBE_ISSUE_MISSING_DEFAULT_SEED not in status["stackCubeIssueCodes"]


def test_runtime_status_missing_default_seed(tmp_path: Path):
    root = _configure_isaac_root(tmp_path)
    from app.core.config import settings

    settings.ISAACLAB_ROOT = str(root)
    settings.ISAACLAB_RUNTIME_ENABLED = True
    settings.ISAACLAB_STACK_CUBE_DEFAULT_SEED = str(tmp_path / "missing.hdf5")

    status = get_runtime_status()
    assert status["defaultSeedAvailable"] is False
    assert status["stackCubeGenerationReady"] is False
    assert STACK_CUBE_ISSUE_MISSING_DEFAULT_SEED in status["stackCubeIssueCodes"]


def test_runtime_status_runtime_disabled():
    from app.core.config import settings

    settings.ISAACLAB_ROOT = None
    settings.ISAACLAB_RUNTIME_ENABLED = False

    status = get_runtime_status()
    assert status["stackCubeGenerationReady"] is False
    assert STACK_CUBE_ISSUE_RUNTIME_DISABLED in status["stackCubeIssueCodes"]
    assert STACK_CUBE_ISSUE_MISSING_ROOT in status["stackCubeIssueCodes"]
