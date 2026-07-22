from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services import workspace_dataset_service as ws_dataset_svc
from app.services.workspace_dataset_stats_service import (
    enrich_dataset_list_stats,
    resolve_dataset_data_count,
    resolve_dataset_size_bytes,
)


def test_resolve_data_count_from_total_episodes() -> None:
    row = {
        "id": "ds_ct",
        "sourceJobId": "ct_gen_20260101_120000_abcd",
        "totalEpisodes": 10,
        "successfulEpisodes": 9,
        "episodeCount": 10,
        "format": "hdf5",
    }
    assert resolve_dataset_data_count(row) == 10


def test_resolve_data_count_import_parsed() -> None:
    row = {
        "id": "ds_import_abc",
        "sourceJobId": "import_ds_import_abc",
        "episodeCount": 120,
        "episodeParsed": True,
        "status": "ready",
        "format": "hdf5",
    }
    assert resolve_dataset_data_count(row) == 120


def test_resolve_data_count_import_needs_build_shows_one_file() -> None:
    row = {
        "id": "ds_import_abc",
        "sourceJobId": "import_ds_import_abc",
        "episodeCount": 0,
        "episodeParsed": False,
        "needsBuild": True,
        "status": "needs_build",
        "format": "hdf5",
        "datasetFile": "/tmp/imports/ds_import_abc/source.hdf5",
    }
    # 测试不依赖真实文件：storagePath 分支在单测中由 datasetFile 覆盖；此处用 monkeypath 思路改为内联存在性
    assert resolve_dataset_data_count(row) is None


def test_resolve_data_count_import_unparsed_with_file_is_one(tmp_path: Path) -> None:
    hdf5 = tmp_path / "source.hdf5"
    hdf5.write_bytes(b"x")
    row = {
        "id": "ds_import_abc",
        "sourceJobId": "import_ds_import_abc",
        "episodeCount": 0,
        "episodeParsed": False,
        "needsBuild": True,
        "status": "needs_build",
        "format": "hdf5",
        "datasetFile": str(hdf5),
    }
    assert resolve_dataset_data_count(row) == 1
    assert row["episodeCount"] == 0


def test_resolve_data_count_import_needs_mapping_unparsed_is_one(tmp_path: Path) -> None:
    hdf5 = tmp_path / "source.hdf5"
    hdf5.write_bytes(b"x")
    row = {
        "id": "ds_import_map",
        "sourceJobId": "import_ds_import_map",
        "episodeCount": 0,
        "episodeParsed": False,
        "needsMapping": True,
        "status": "needs_mapping",
        "format": "hdf5",
        "datasetFile": str(hdf5),
    }
    assert resolve_dataset_data_count(row) == 1


def test_resolve_data_count_manifest_only_is_null() -> None:
    row = {
        "id": "ds_manifest",
        "sourceJobId": "dac_gen_abc",
        "format": "manifest",
        "totalEpisodes": 5,
    }
    assert resolve_dataset_data_count(row) is None


def test_resolve_size_hdf5_from_dataset_file(tmp_path: Path) -> None:
    hdf5 = tmp_path / "dataset.hdf5"
    hdf5.write_bytes(b"x" * 2048)
    row = {
        "id": "ds_hdf5",
        "sourceJobId": "ct_gen_test",
        "format": "hdf5",
        "datasetFile": str(hdf5),
    }
    assert resolve_dataset_size_bytes(row) == 2048


def test_resolve_size_lerobot_directory(tmp_path: Path) -> None:
    lerobot_dir = tmp_path / "lerobot_dataset"
    lerobot_dir.mkdir()
    (lerobot_dir / "a.bin").write_bytes(b"a" * 100)
    (lerobot_dir / "nested").mkdir()
    (lerobot_dir / "nested" / "b.bin").write_bytes(b"b" * 50)
    row = {
        "id": "ds_lerobot",
        "sourceJobId": "ct_gen_test",
        "format": "lerobot",
        "lerobotPath": str(lerobot_dir),
    }
    assert resolve_dataset_size_bytes(row) == 150


def test_resolve_size_uses_existing_file_size_bytes() -> None:
    row = {
        "id": "ds_import",
        "sourceJobId": "import_ds_import_x",
        "fileSizeBytes": 4096,
        "format": "hdf5",
    }
    assert resolve_dataset_size_bytes(row) == 4096


def test_coerce_enriches_data_count_and_size(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    jobs_root = tmp_path / "cable" / "jobs"
    job_id = "ct_gen_20260617_120000_abcd"
    job_dir = jobs_root / job_id
    datasets_dir = job_dir / "datasets"
    datasets_dir.mkdir(parents=True)
    hdf5 = datasets_dir / "dataset.hdf5"
    hdf5.write_bytes(b"h" * 1024)
    manifest = {
        "num_successful": 7,
        "num_failed": 3,
        "created_at": "2026-06-17T12:00:00",
    }
    (datasets_dir / "dataset.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    monkeypatch.setattr(ws_dataset_svc, "CABLE_THREADING_ROOT", jobs_root)
    monkeypatch.setattr(ws_dataset_svc, "DUAL_ARM_ROOT", tmp_path / "dual" / "jobs")
    monkeypatch.setattr(ws_dataset_svc, "DATA_GENERATION_ROOT", tmp_path / "data_generation" / "jobs")
    from app.services.isaac_lab import isaac_dataset_service as isaac_dataset_svc

    monkeypatch.setattr(isaac_dataset_svc, "ISAAC_DATASET_REGISTRY_PATH", tmp_path / "registry.json")
    monkeypatch.setattr(
        "app.services.workspace_dataset_import_service.IMPORT_ROOT",
        tmp_path / "imports",
    )

    rows = ws_dataset_svc.scan_datasets_for_api()
    row = next(item for item in rows if item["sourceJobId"] == job_id)

    assert row["dataCount"] == 10
    assert row["fileSizeBytes"] == 1024


def test_enrich_persists_file_size_to_manifest(tmp_path: Path) -> None:
    job_dir = tmp_path / "ct_gen_persist"
    datasets_dir = job_dir / "datasets"
    datasets_dir.mkdir(parents=True)
    hdf5 = datasets_dir / "dataset.hdf5"
    hdf5.write_bytes(b"z" * 512)
    manifest_path = datasets_dir / "dataset.manifest.json"
    manifest_path.write_text(json.dumps({"num_successful": 1, "num_failed": 0}), encoding="utf-8")

    row = {
        "id": "ds_persist",
        "sourceJobId": "ct_gen_persist",
        "manifestPath": str(manifest_path),
        "storagePath": str(datasets_dir),
        "format": "hdf5",
        "datasetFile": str(hdf5),
        "totalEpisodes": 1,
    }
    enrich_dataset_list_stats(row, persist_size=True)

    saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert saved["fileSizeBytes"] == 512
    assert row["fileSizeBytes"] == 512
    assert row["dataCount"] == 1
