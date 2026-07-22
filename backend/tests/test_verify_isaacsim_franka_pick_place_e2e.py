from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_collect_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "verification" / "verify_isaacsim_franka_pick_place_collect.py"
    spec = importlib.util.spec_from_file_location("verify_collect", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_collect_organizes_adapter_outputs_and_writes_acceptance(tmp_path: Path):
    mod = _load_collect_module()
    job_dir = tmp_path / "verify_isaacsim_franka_pick_place"
    job_dir.mkdir(parents=True)
    (job_dir / "videos").mkdir()
    video = job_dir / "videos" / "ep_000001.mp4"
    video.write_bytes(b"\x00\x00\x00\x18ftypmp42")

    (job_dir / "run_result.json").write_text(
        json.dumps(
            {
                "task_id": "isaacsim_franka_pick_place",
                "episode_id": "ep_000001",
                "success": True,
                "controller_done": True,
                "pick_success": True,
                "place_success": True,
                "video_available": True,
                "video_status": "available",
                "video_path": str(video),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (job_dir / "metrics.json").write_text(
        json.dumps(
            {
                "success": True,
                "pick_success": True,
                "place_success": True,
                "controller_done": True,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (job_dir / "episode_manifest.json").write_text(
        json.dumps(
            {
                "episode_id": "ep_000001",
                "task_id": "isaacsim_franka_pick_place",
                "video_available": True,
                "video_status": "available",
                "video_path": "videos/ep_000001.mp4",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (job_dir / "trajectory.json").write_text(
        json.dumps({"episode_id": "ep_000001", "steps": 10}, ensure_ascii=False),
        encoding="utf-8",
    )

    mod.organize_job_dir(job_dir)
    assert (job_dir / "status.json").is_file()
    assert (job_dir / "dataset_manifest.json").is_file()
    assert (job_dir / "results" / "aggregate_metrics.json").is_file()
    assert (job_dir / "episodes" / "ep_000001" / "episode_manifest.json").is_file()

    dataset_manifest = json.loads((job_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    episode_manifest = json.loads(
        (job_dir / "episodes" / "ep_000001" / "episode_manifest.json").read_text(encoding="utf-8")
    )
    assert dataset_manifest["task_id"] == "isaacsim_franka_pick_place"
    assert episode_manifest["task_id"] == "isaacsim_franka_pick_place"
    assert dataset_manifest["video_status"] == "available"


def test_write_skipped_acceptance(tmp_path: Path):
    mod = _load_collect_module()
    job_dir = tmp_path / "verify_job"
    mod.write_acceptance_md(
        job_dir,
        summary={},
        preview_path=None,
        checks=[],
        conclusion="SKIPPED: Isaac Lab runtime detected, but NVIDIA Isaac Sim official FrankaPickPlace controller is not importable in this environment.",
        skipped=True,
        skip_reason="Isaac Lab runtime detected, but NVIDIA Isaac Sim official FrankaPickPlace controller is not importable in this environment.",
        diagnosis={
            "isaac_lab_runtime_available": True,
            "isaac_sim_runtime_available": True,
            "can_import_franka_pick_place": False,
            "diagnosis": "controller_unavailable",
        },
    )
    text = (job_dir / "ACCEPTANCE.md").read_text(encoding="utf-8")
    assert "isaac_lab_runtime_available: `True`" in text
    assert "can_import_franka_pick_place: `False`" in text
    assert "controller is not importable" in text
