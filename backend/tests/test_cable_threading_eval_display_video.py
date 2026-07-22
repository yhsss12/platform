from __future__ import annotations

import sys
from pathlib import Path

import pytest

_CABLE_MVP = Path(__file__).resolve().parents[2] / "integrations" / "CableThreadingMVP"
if str(_CABLE_MVP) not in sys.path:
    sys.path.insert(0, str(_CABLE_MVP))

from examples.cable_threading.obs_schema import (  # noqa: E402
    EVAL_DISPLAY_CAMERA,
    EVAL_DISPLAY_FRAME_HEIGHT,
    EVAL_DISPLAY_FRAME_WIDTH,
    policy_eval_camera_kwargs,
)


def test_policy_eval_cameras_unaffected_by_display_camera():
    kwargs = policy_eval_camera_kwargs()
    assert kwargs["camera_names"] == ["agentview", "robot0_eye_in_hand"]
    assert kwargs["camera_heights"] == 256
    assert kwargs["camera_widths"] == 256
    assert "eval_display_camera" not in kwargs["camera_names"]
    assert "robot0_eye_in_hand" in kwargs["camera_names"]


def test_display_camera_constants():
    assert EVAL_DISPLAY_CAMERA == "agentview"
    assert EVAL_DISPLAY_FRAME_WIDTH == 1280
    assert EVAL_DISPLAY_FRAME_HEIGHT == 720


def test_resolve_job_video_prefers_browser(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from app.services import cable_threading_service as svc

    job_id = "ct_eval_20260617_999999_aaaa"
    job_root = tmp_path / "jobs" / job_id
    videos = job_root / "videos"
    videos.mkdir(parents=True)
    (videos / "eval.mp4").write_bytes(b"raw")
    (videos / "eval.browser.mp4").write_bytes(b"browser")

    monkeypatch.setattr(svc, "OUTPUT_ROOT", tmp_path)

    resolved = svc.resolve_job_video_path(job_id)
    assert resolved is not None
    assert resolved.name == "eval.browser.mp4"


def test_build_eval_command_uses_display_camera(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from app.services import cable_threading_service as svc

    monkeypatch.setattr(svc, "PYTHON_BIN", Path("/usr/bin/python3"))
    monkeypatch.setattr(svc, "RUN_PY", Path("/tmp/run.py"))

    cmd = svc._build_eval_command(
        tmp_path / "jobs" / "ct_eval_20260617_999999_aaaa",
        episodes=2,
        robot="Panda",
        cable_model="composite_cable",
        difficulty="easy",
        horizon=200,
        seed=0,
        policy="robomimic",
        checkpoint="/tmp/model.pth",
    )
    assert "--live-display-camera" in cmd
    assert "agentview" in cmd
    assert "eval_display_camera" not in cmd
    assert "--live-camera" not in cmd
