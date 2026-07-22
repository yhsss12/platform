from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json

import pytest
from fastapi import HTTPException

from app.services.isaac_lab import isaac_dataset_service as dataset_svc


def test_import_demo_rejects_missing_file(tmp_path: Path):
    with pytest.raises(HTTPException) as exc:
        dataset_svc.import_demo_hdf5(
            dataset_file=str(tmp_path / "missing.hdf5"),
            display_name="Stack Demo",
        )
    assert exc.value.status_code == 400
    assert "not found" in str(exc.value.detail).lower()


def test_import_demo_registers_available_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    registry = tmp_path / "registry.json"
    monkeypatch.setattr(dataset_svc, "ISAAC_DATASET_REGISTRY_PATH", registry)

    hdf5 = tmp_path / "stack_demo.hdf5"
    hdf5.write_bytes(b"hdf5")

    row = dataset_svc.import_demo_hdf5(
        dataset_file=str(hdf5),
        display_name="Stack Cube Demo",
        task_id="Isaac-Stack-Cube-Franka-IK-Rel-v0",
    )

    assert row["status"] == "available"
    assert row["format"] == "hdf5"
    assert row["sourceType"] == "imported_demo"
    assert row["simulatorBackend"] == "isaac_lab"
    assert row["replayAvailable"] is True
    assert row["replayBackend"] == "isaac_lab"
    assert row["taskTemplateId"] == "isaac_block_stacking"
    assert row["trainable"] is True
    assert row["trainingBackends"] == ["isaac_robomimic_bc"]
    assert row["datasetFile"] == str(hdf5.resolve())
    assert row["name"] == "Stack Cube Demo"

    listed = dataset_svc.list_isaac_datasets()
    assert len(listed) == 1
    assert listed[0]["id"] == row["id"]


def test_list_skips_missing_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    registry = tmp_path / "registry.json"
    monkeypatch.setattr(dataset_svc, "ISAAC_DATASET_REGISTRY_PATH", registry)

    hdf5 = tmp_path / "demo.hdf5"
    hdf5.write_bytes(b"hdf5")
    imported = dataset_svc.import_demo_hdf5(
        dataset_file=str(hdf5),
        display_name="Demo",
    )

    hdf5.unlink()
    assert dataset_svc.list_isaac_datasets() == []

    with pytest.raises(HTTPException) as exc:
        dataset_svc.get_isaac_dataset(imported["id"])
    assert exc.value.status_code == 400


def test_get_isaac_dataset_not_found():
    with pytest.raises(HTTPException) as exc:
        dataset_svc.get_isaac_dataset("isaac_ds_missing")
    assert exc.value.status_code == 404


def test_delete_isaac_dataset_removes_registry_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    registry = tmp_path / "registry.json"
    monkeypatch.setattr(dataset_svc, "ISAAC_DATASET_REGISTRY_PATH", registry)

    hdf5 = tmp_path / "demo.hdf5"
    hdf5.write_bytes(b"hdf5")
    imported = dataset_svc.import_demo_hdf5(dataset_file=str(hdf5), display_name="Demo")

    dataset_svc.delete_isaac_dataset(imported["id"])
    assert dataset_svc.list_isaac_datasets() == []
    assert hdf5.is_file()
    dataset_svc.delete_isaac_dataset(imported["id"])


def test_delete_generated_isaac_dataset_removes_new_runs_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    registry = tmp_path / "assets" / "registry.json"
    runs_root = tmp_path / "runs"
    job_id = "isaac_gen_20260721_150332_30ce"
    job_root = runs_root / "isaac_lab" / "jobs" / job_id
    job_root.mkdir(parents=True)
    (job_root / "dataset.hdf5").write_bytes(b"hdf5")
    registry.parent.mkdir(parents=True)
    registry.write_text(
        json.dumps({"datasets": [{"id": "isaac_ds_1", "sourceJobId": job_id}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(dataset_svc, "ISAAC_DATASET_REGISTRY_PATH", registry)
    monkeypatch.setattr(dataset_svc, "platform_paths", SimpleNamespace(runs_root=runs_root))

    dataset_svc.delete_isaac_dataset("isaac_ds_1")

    assert not job_root.exists()
    assert dataset_svc._read_registry_file() == []


def test_register_generated_dataset_sets_preview_video_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    registry = tmp_path / "registry.json"
    monkeypatch.setattr(dataset_svc, "ISAAC_DATASET_REGISTRY_PATH", registry)

    job_id = "isaac_gen_20260615_120000_abcd"
    job_root = tmp_path / "jobs" / job_id
    artifacts = job_root / "artifacts"
    artifacts.mkdir(parents=True)
    dataset_file = artifacts / "dataset.hdf5"
    dataset_file.write_bytes(b"hdf5")
    preview = artifacts / "preview.mp4"
    preview.write_bytes(b"mp4")

    def fake_preview(job: str) -> Path:
        assert job == job_id
        return preview

    monkeypatch.setattr(dataset_svc, "isaac_job_preview_video_path", fake_preview)

    row = dataset_svc.register_generated_dataset(
        job_id=job_id,
        dataset_name="Stack Generated",
        dataset_file=dataset_file,
        episode_count=2,
    )

    assert row["sourceJobId"] == job_id
    assert row["previewVideoPath"] == str(preview.resolve())
    assert row["videoAvailable"] is True


def test_register_generated_dataset_resolves_generation_mode_from_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    registry = tmp_path / "registry.json"
    monkeypatch.setattr(dataset_svc, "ISAAC_DATASET_REGISTRY_PATH", registry)

    job_id = "isaac_gen_20260617_120000_abcd"
    job_root = tmp_path / "jobs" / job_id
    artifacts = job_root / "artifacts"
    artifacts.mkdir(parents=True)
    dataset_file = artifacts / "dataset.hdf5"
    dataset_file.write_bytes(b"hdf5")
    meta_dir = job_root / "metadata"
    meta_dir.mkdir(parents=True)
    (meta_dir / "request.json").write_text(
        '{"generationMode": "scripted_expert"}',
        encoding="utf-8",
    )
    (job_root / "generation_manifest.json").write_text(
        '{"metrics": {"generationMode": "scripted_expert"}}',
        encoding="utf-8",
    )

    def fake_preview(job: str) -> Path:
        preview = artifacts / "preview.mp4"
        preview.write_bytes(b"mp4")
        return preview

    monkeypatch.setattr(dataset_svc, "isaac_job_preview_video_path", fake_preview)

    row = dataset_svc.register_generated_dataset(
        job_id=job_id,
        dataset_name="Stack Generated",
        dataset_file=dataset_file,
        episode_count=1,
    )

    assert row["generationMode"] == "scripted_expert"


def test_import_demo_idempotent_for_same_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    registry = tmp_path / "registry.json"
    monkeypatch.setattr(dataset_svc, "ISAAC_DATASET_REGISTRY_PATH", registry)

    hdf5 = tmp_path / "demo.hdf5"
    hdf5.write_bytes(b"hdf5")

    first = dataset_svc.import_demo_hdf5(dataset_file=str(hdf5), display_name="First")
    second = dataset_svc.import_demo_hdf5(dataset_file=str(hdf5), display_name="Second")
    assert first["id"] == second["id"]
    assert len(dataset_svc.list_isaac_datasets()) == 1
