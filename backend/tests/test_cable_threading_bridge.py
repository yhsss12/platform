from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from app.services import cable_threading_service as svc


def test_new_job_dirs_use_current_output_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    current = tmp_path / "data" / "runs" / "cable_threading"
    legacy = tmp_path / "code" / "runs" / "cable_threading"
    monkeypatch.setattr(svc, "OUTPUT_ROOT", current)
    monkeypatch.setattr(svc, "LEGACY_OUTPUT_ROOT", legacy)

    job_id = "ct_gen_20260720_120000_abcd"
    job_root = svc._prepare_job_dirs(job_id, include_datasets=True, include_videos=True)

    assert job_root == current / "jobs" / job_id
    assert (job_root / "datasets").is_dir()
    assert not legacy.exists()


def test_existing_legacy_job_remains_readable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    current = tmp_path / "data" / "runs" / "cable_threading"
    legacy = tmp_path / "code" / "runs" / "cable_threading"
    monkeypatch.setattr(svc, "OUTPUT_ROOT", current)
    monkeypatch.setattr(svc, "LEGACY_OUTPUT_ROOT", legacy)

    job_id = "ct_eval_20260720_120000_abcd"
    log_path = legacy / "jobs" / job_id / "logs" / "run.log"
    log_path.parent.mkdir(parents=True)
    log_path.write_text("legacy evaluation log\n", encoding="utf-8")

    assert svc._job_dir(job_id) == legacy / "jobs" / job_id
    assert svc.resolve_job_log_path(job_id) == log_path.resolve()
    assert svc.read_job_log_tail(job_id) == "legacy evaluation log"


def test_validate_robot_rejects_unknown():
    with pytest.raises(HTTPException) as exc:
        svc._validate_robot("Kuka")
    assert exc.value.status_code == 400


def test_validate_cable_model_rejects_unknown():
    with pytest.raises(HTTPException) as exc:
        svc._validate_cable_model("unknown")
    assert exc.value.status_code == 400


def test_parse_stdout_metrics():
    stdout = "\n".join(
        [
            "saved_csv: /tmp/collect.csv",
            "final_success_rate: 0.9000",
            "successful_episodes: 9",
            "saved_dataset: /tmp/dataset.npz",
            "saved_hdf5: /tmp/dataset.hdf5",
            "saved_manifest: /tmp/dataset.manifest.json",
            "saved_failures: /tmp/failures.json",
        ]
    )
    assert svc._parse_stdout_float(stdout, "final_success_rate") == 0.9
    assert svc._parse_stdout_int(stdout, "successful_episodes") == 9
    assert svc._parse_stdout_value(stdout, "saved_dataset") == "/tmp/dataset.npz"


def test_path_info_missing_file(tmp_path: Path):
    missing = tmp_path / "missing.npz"
    info = svc._path_info(missing)
    assert info["exists"] is False
    assert info["sizeBytes"] is None


def test_run_video_invokes_subprocess_with_safe_args(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    job_root = tmp_path / "jobs"
    monkeypatch.setattr(svc, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(svc, "WORKING_DIR", tmp_path / "CableThreadingMVP")
    monkeypatch.setattr(svc, "RUN_PY", tmp_path / "CableThreadingMVP" / "run.py")
    monkeypatch.setattr(svc, "PYTHON_BIN", tmp_path / "python")

    (tmp_path / "CableThreadingMVP").mkdir(parents=True)
    (tmp_path / "CableThreadingMVP" / "run.py").write_text("# stub", encoding="utf-8")
    (tmp_path / "python").write_text("", encoding="utf-8")
    (tmp_path / "python").chmod(0o755)

    captured: dict[str, object] = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["cwd"] = kwargs.get("cwd")
        captured["env"] = kwargs.get("env")
        video_out = cmd[cmd.index("--video-out") + 1]
        Path(video_out).parent.mkdir(parents=True, exist_ok=True)
        Path(video_out).write_bytes(b"\x00\x00\x00\x18ftypmp42")
        class Proc:
            returncode = 0
            stdout = "saved_video: demo.mp4\n"
            stderr = ""

        return Proc()

    monkeypatch.setattr(svc.subprocess, "run", fake_run)

    result = svc.run_video(
        episodes=1,
        robot="Panda",
        cable_model="composite_cable",
        difficulty="easy",
        horizon=600,
        seed=0,
    )

    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert "video" in cmd
    assert "--cable-model" in cmd
    assert cmd[cmd.index("--cable-model") + 1] == "composite_cable"
    assert captured["env"]["PYTHONNOUSERSITE"] == "1"
    assert captured["env"]["MUJOCO_GL"] == "egl"
    assert result["status"] == "completed"
    assert result["videoExists"] is True


def test_load_eval_results_json(tmp_path: Path):
    path = tmp_path / "eval.results.json"
    path.write_text(
        json.dumps(
            {
                "success_rate": 0.8,
                "ever_success_rate": 0.9,
                "num_episodes": 10,
                "aggregate": {"final_success_rate": 0.8},
            }
        ),
        encoding="utf-8",
    )
    data = svc._load_eval_results_json(path)
    assert data["success_rate"] == 0.8
    assert data["aggregate"]["final_success_rate"] == 0.8


def test_validate_job_id_accepts_known_formats():
    assert svc.validate_job_id("ct_eval_20260609_152138_65cb") == "ct_eval_20260609_152138_65cb"
    assert svc.validate_job_id("ct_vid_20260609_152401_f0fd") == "ct_vid_20260609_152401_f0fd"
    assert svc.validate_job_id("ct_eval_20260624_dp_smoke") == "ct_eval_20260624_dp_smoke"


def test_validate_job_id_rejects_invalid():
    with pytest.raises(HTTPException) as exc:
        svc.validate_job_id("../etc/passwd")
    assert exc.value.status_code == 400


def test_resolve_job_frame_path_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(svc, "OUTPUT_ROOT", tmp_path)
    job_id = "ct_gen_20260609_152436_cd03"
    assert svc.resolve_job_frame_path(job_id) is None


def test_resolve_job_frame_path_exists(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(svc, "OUTPUT_ROOT", tmp_path)
    job_id = "ct_gen_20260609_152436_cd03"
    frame = tmp_path / "jobs" / job_id / "live" / "latest.jpg"
    frame.parent.mkdir(parents=True, exist_ok=True)
    frame.write_bytes(b"\xff\xd8\xff")
    (frame.parent / "status.json").write_text(
        json.dumps({"hasValidFrame": True, "frameStatus": "ready", "frameCount": 1}),
        encoding="utf-8",
    )
    assert svc.resolve_job_frame_path(job_id) == frame.resolve()


def test_resolve_job_frame_path_rejects_invalid_only_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(svc, "OUTPUT_ROOT", tmp_path)
    job_id = "ct_eval_20260609_152436_cd03"
    frame = tmp_path / "jobs" / job_id / "live" / "latest.jpg"
    frame.parent.mkdir(parents=True, exist_ok=True)
    frame.write_bytes(b"\xff\xd8\xff")
    (frame.parent / "status.json").write_text(
        json.dumps({"hasValidFrame": False, "frameStatus": "warming_up", "frameCount": 0}),
        encoding="utf-8",
    )
    assert svc.resolve_job_frame_path(job_id) is None


def test_get_generate_job_status_reads_live_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(svc, "OUTPUT_ROOT", tmp_path)
    job_id = "ct_gen_20260609_152436_cd03"
    job_root = tmp_path / "jobs" / job_id
    live_dir = job_root / "live"
    live_dir.mkdir(parents=True, exist_ok=True)
    (live_dir / "status.json").write_text(
        json.dumps(
            {
                "status": "running",
                "episode": 0,
                "episodes": 1,
                "step": 12,
                "frameCount": 2,
                "phase": "pull_through",
            }
        ),
        encoding="utf-8",
    )
    result = svc.get_generate_job_status(job_id)
    assert result["status"] == "running"
    assert result["live"]["step"] == 12
    assert result["live"]["phase"] == "pull_through"


def test_start_generate_async_registers_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(svc, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(svc, "WORKING_DIR", tmp_path / "CableThreadingMVP")
    monkeypatch.setattr(svc, "RUN_PY", tmp_path / "CableThreadingMVP" / "run.py")
    monkeypatch.setattr(svc, "PYTHON_BIN", tmp_path / "python")
    (tmp_path / "CableThreadingMVP").mkdir(parents=True)
    (tmp_path / "CableThreadingMVP" / "run.py").write_text("# stub", encoding="utf-8")
    (tmp_path / "python").write_text("", encoding="utf-8")
    (tmp_path / "python").chmod(0o755)

    class FakeProc:
        def __init__(self):
            self.returncode = None

        def poll(self):
            return None

    monkeypatch.setattr(
        svc.subprocess,
        "Popen",
        lambda *args, **kwargs: FakeProc(),
    )

    result = svc.start_generate_async(
        episodes=1,
        robot="Panda",
        cable_model="composite_cable",
        difficulty="easy",
        horizon=600,
        seed=0,
        save_hdf5=False,
        output_format="npz",
        save_process_video=True,
    )
    assert result["status"] == "running"
    assert "generate-async" not in result["frameUrl"]
    assert result["jobId"] in result["frameUrl"]
    assert result["jobId"] in svc.ASYNC_JOBS
    record = svc.ASYNC_JOBS[result["jobId"]]
    assert "--live-save-frames" in record.command
    assert "--live-video-out" in record.command
    assert "--live-timeline-out" in record.command


def test_resolve_job_video_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(svc, "OUTPUT_ROOT", tmp_path)
    job_id = "ct_vid_20260609_152401_f0fd"
    video_path = tmp_path / "jobs" / job_id / "videos" / "demo.mp4"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.write_bytes(b"fake-mp4")

    resolved = svc.resolve_job_video_path(job_id)
    assert resolved == video_path.resolve()
    assert resolved.is_file()


def test_resolve_job_video_path_prefers_generate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(svc, "OUTPUT_ROOT", tmp_path)
    job_id = "ct_gen_20260609_152436_cd03"
    job_root = tmp_path / "jobs" / job_id / "videos"
    job_root.mkdir(parents=True, exist_ok=True)
    generate_path = job_root / "generate.mp4"
    demo_path = job_root / "demo.mp4"
    generate_path.write_bytes(b"generate-mp4")
    demo_path.write_bytes(b"demo-mp4")

    resolved = svc.resolve_job_video_path(job_id)
    assert resolved == generate_path.resolve()


def test_resolve_job_video_path_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(svc, "OUTPUT_ROOT", tmp_path)
    job_id = "ct_gen_20260609_152436_cd03"
    assert svc.resolve_job_video_path(job_id) is None


def test_build_eval_command_includes_attachment_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(svc, "PYTHON_BIN", tmp_path / "python")
    monkeypatch.setattr(svc, "RUN_PY", tmp_path / "run.py")
    (tmp_path / "python").write_text("", encoding="utf-8")
    (tmp_path / "run.py").write_text("# stub", encoding="utf-8")

    job_root = tmp_path / "ct_eval_test"
    cmd = svc._build_eval_command(
        job_root,
        episodes=3,
        robot="Panda",
        cable_model="composite_cable",
        difficulty="easy",
        horizon=600,
        seed=0,
        policy="diffusion_policy",
        checkpoint="/tmp/model.pt",
        device="cuda",
        grasp_mode="attachment",
        attachment_mode="policy",
    )
    assert "--attachment-mode" in cmd
    assert cmd[cmd.index("--attachment-mode") + 1] == "policy"
    assert "--grasp-mode" in cmd
    assert cmd[cmd.index("--grasp-mode") + 1] == "attachment"
