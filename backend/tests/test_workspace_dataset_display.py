from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.schemas.workspace_benchmark import DatasetResponse
from app.services import workspace_dataset_service as ws_dataset_svc
from app.services.dataset_naming import is_canonical_dataset_display_name
from app.services.isaac_lab import isaac_dataset_service as isaac_dataset_svc


def test_dataset_response_includes_trajectory_fields():
    row = DatasetResponse(
        id="ds_test",
        name="test",
        sourceJobId="ct_gen_20260617_104326_e1e8",
        manifestPath="/tmp/manifest.json",
        successfulEpisodes=9,
        totalEpisodes=10,
        validTrajectories=9,
        generationRounds=10,
    )
    assert row.successfulEpisodes == 9
    assert row.totalEpisodes == 10


def test_cable_threading_list_enriches_trajectory_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    jobs_root = tmp_path / "cable" / "jobs"
    job_id = "ct_gen_20260617_120000_abcd"
    job_dir = jobs_root / job_id
    datasets_dir = job_dir / "datasets"
    datasets_dir.mkdir(parents=True)
    (datasets_dir / "dataset.hdf5").write_bytes(b"x" * 64)
    manifest = {
        "num_successful": 7,
        "num_failed": 3,
        "created_at": "2026-06-17T12:00:00",
    }
    (datasets_dir / "dataset.manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    monkeypatch.setattr(ws_dataset_svc, "CABLE_THREADING_ROOT", jobs_root)
    monkeypatch.setattr(ws_dataset_svc, "DUAL_ARM_ROOT", tmp_path / "dual" / "jobs")
    monkeypatch.setattr(isaac_dataset_svc, "ISAAC_DATASET_REGISTRY_PATH", tmp_path / "registry.json")

    rows = ws_dataset_svc.scan_datasets_for_api()
    row = next(item for item in rows if item["sourceJobId"] == job_id)

    assert row["simulatorBackend"] == "mujoco"
    assert row["successfulEpisodes"] == 7
    assert row["totalEpisodes"] == 10
    assert row["validTrajectories"] == 7
    assert row["generationRounds"] == 10
    assert row["dataCount"] == 10
    assert row["taskDisplayName"] == "线缆穿杆"
    assert row["name"] == "线缆穿杆数据_20260617_1200"
    assert row["datasetName"] == "线缆穿杆数据_20260617_1200"
    assert is_canonical_dataset_display_name(row["displayName"])


def test_nut_assembly_list_uses_real_trainable_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    jobs_root = tmp_path / "nut" / "jobs"
    job_id = "na_gen_20260722_141154_8b55"
    job_dir = jobs_root / job_id
    (job_dir / "datasets").mkdir(parents=True)
    (job_dir / "datasets" / "nut_assembly_generated.hdf5").write_bytes(b"x" * 64)
    (job_dir / "manifest.json").write_text(
        json.dumps(
            {
                "taskType": "nut_assembly",
                "demoCount": 5,
                "successEpisodes": 0,
                "validForTrainingEpisodes": 0,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(ws_dataset_svc, "NUT_ASSEMBLY_ROOT", jobs_root)
    monkeypatch.setattr(ws_dataset_svc, "CABLE_THREADING_ROOT", tmp_path / "cable" / "jobs")
    monkeypatch.setattr(ws_dataset_svc, "DUAL_ARM_ROOT", tmp_path / "dual" / "jobs")
    monkeypatch.setattr(isaac_dataset_svc, "ISAAC_DATASET_REGISTRY_PATH", tmp_path / "registry.json")

    rows = ws_dataset_svc.scan_datasets_for_api()
    row = next(item for item in rows if item["sourceJobId"] == job_id)

    assert row["successfulEpisodes"] == 0
    assert row["validTrajectories"] == 0
    assert row["totalEpisodes"] == 5
    assert row["trainable"] is False
    assert row["datasetName"] == "螺母装配数据_20260722_1411"


def test_isaac_list_enriches_demo_count_as_valid_trajectories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    registry = tmp_path / "registry.json"
    monkeypatch.setattr(isaac_dataset_svc, "ISAAC_DATASET_REGISTRY_PATH", registry)
    monkeypatch.setattr(ws_dataset_svc, "CABLE_THREADING_ROOT", tmp_path / "cable" / "jobs")
    monkeypatch.setattr(ws_dataset_svc, "DUAL_ARM_ROOT", tmp_path / "dual" / "jobs")

    hdf5 = tmp_path / "stack.hdf5"
    hdf5.write_bytes(b"hdf5")
    imported = isaac_dataset_svc.import_demo_hdf5(
        dataset_file=str(hdf5),
        display_name="Stack Demo",
    )

    rows = ws_dataset_svc.scan_datasets_for_api()
    row = next(item for item in rows if item["id"] == imported["id"])

    assert row["simulatorBackend"] == "isaac_lab"
    assert row["successfulEpisodes"] == row["totalEpisodes"]
    assert row["validTrajectories"] == row["generationRounds"]
    assert row["fileSizeBytes"] == 4
    assert row["dataCount"] == row["totalEpisodes"]
