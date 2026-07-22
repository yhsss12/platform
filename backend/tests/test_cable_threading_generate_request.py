from __future__ import annotations

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from app.schemas.cable_threading import CableThreadingGenerateRequest
from app.services import cable_threading_service as svc


def test_schema_accepts_lerobot_output_format():
    req = CableThreadingGenerateRequest(
        episodes=1,
        outputFormat="lerobot",
        saveHdf5=False,
        lerobotTaskInstruction="thread the cable through the pole",
        lerobotRobot="Panda",
        lerobotFps=20,
    )
    assert req.outputFormat == "lerobot"
    assert req.saveHdf5 is False
    assert req.lerobotFps == 20


def test_schema_rejects_unknown_output_format():
    with pytest.raises(ValidationError):
        CableThreadingGenerateRequest(outputFormat="LeRobot")  # type: ignore[arg-type]


def test_validate_output_format_accepts_lerobot():
    assert svc._validate_output_format("lerobot") == "lerobot"


def test_validate_output_format_rejects_unknown():
    with pytest.raises(HTTPException) as exc:
        svc._validate_output_format("mcap")
    assert exc.value.status_code == 400


def test_build_generate_command_includes_lerobot_flags(tmp_path):
    job_root = tmp_path / "ct_gen_test"
    job_root.mkdir(parents=True)
    cmd = svc._build_generate_command(
        job_root,
        episodes=1,
        robot="Panda",
        cable_model="composite_cable",
        difficulty="easy",
        horizon=200,
        seed=0,
        save_hdf5=False,
        output_format="lerobot",
        include_live=False,
        lerobot_task_instruction="thread the cable through the pole",
        lerobot_robot="Panda",
        lerobot_fps=20,
    )
    assert "--lerobot-out" in cmd
    assert str(job_root / "datasets" / "lerobot_dataset") in cmd
    assert "--lerobot-task-instruction" in cmd
    assert "--hdf5-out" not in cmd
    assert cmd[cmd.index("--out") + 1].endswith("datasets/debug/dataset.npz")


def test_build_generate_command_hdf5_unchanged(tmp_path):
    job_root = tmp_path / "ct_gen_hdf5"
    job_root.mkdir(parents=True)
    cmd = svc._build_generate_command(
        job_root,
        episodes=1,
        robot="Panda",
        cable_model="composite_cable",
        difficulty="easy",
        horizon=600,
        seed=0,
        save_hdf5=True,
        output_format="hdf5",
        include_live=False,
    )
    assert "--hdf5-out" in cmd
    assert "--lerobot-out" not in cmd
