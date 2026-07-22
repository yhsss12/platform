from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services import workspace_dataset_import_service as svc
from app.services import workspace_dataset_service as dataset_svc


@pytest.fixture
def sample_hdf5(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    import_root = tmp_path / "imports"
    monkeypatch.setattr(svc, "IMPORT_ROOT", import_root)

    h5py = pytest.importorskip("h5py")
    import numpy as np

    file_path = tmp_path / "sample.hdf5"
    with h5py.File(file_path, "w") as handle:
        data = handle.create_group("data")
        demo = data.create_group("demo_0")
        obs = demo.create_group("obs")
        obs.create_dataset("qpos", data=np.zeros((10, 7), dtype=np.float32))
        obs.create_dataset("agentview_image", data=np.zeros((10, 64, 64, 3), dtype=np.uint8))
        demo.create_dataset("actions", data=np.zeros((10, 7), dtype=np.float32))
    return file_path


def test_parse_hdf5_detects_training_fields(sample_hdf5: Path) -> None:
    parsed = svc._parse_hdf5_file(sample_hdf5)
    assert parsed["episodeCount"] == 1
    assert parsed["recognizedFields"]["action"]
    assert parsed["recognizedFields"]["qpos"]
    assert parsed["recognizedFields"]["image"]


def test_import_hdf5_dataset_upload_registers_row(
    sample_hdf5: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    import_root = sample_hdf5.parent / "imports"
    monkeypatch.setattr(svc, "IMPORT_ROOT", import_root)

    class _Upload:
        filename = "sample.hdf5"

        async def read(self) -> bytes:
            return sample_hdf5.read_bytes()

    result = asyncio.run(
        svc.import_hdf5_dataset_upload(
            name="测试导入集",
            data_source="real_collection",
            task_type="cable_threading",
            robot_type="fr3",
            file=_Upload(),  # type: ignore[arg-type]
        )
    )

    dataset_id = result["datasetId"]
    import_dir = import_root / dataset_id
    assert import_dir.is_dir()
    assert (import_dir / "source.hdf5").is_file()
    assert (import_dir / "metadata.json").is_file()
    assert (import_dir / "schema.json").is_file()
    assert (import_dir / "validation_report.json").is_file()

    metadata = json.loads((import_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["dataSourceLabel"] == "真实导入"
    assert metadata["format"] == "hdf5"
    assert metadata["status"] == svc.IMPORT_STATUS_AVAILABLE
    assert metadata["episodeCount"] == 1

    row = svc.import_record_to_dataset_row(import_dir)
    assert row is not None
    assert row["id"] == dataset_id
    assert row["trainable"] is True
    assert "episodes" in row["dataScaleLabel"]


def test_list_imported_datasets_in_workspace_index(
    sample_hdf5: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import_root = sample_hdf5.parent / "imports"
    monkeypatch.setattr(svc, "IMPORT_ROOT", import_root)

    dataset_id = svc.make_dataset_import_id()
    import_dir = import_root / dataset_id
    import_dir.mkdir(parents=True)
    (import_dir / "source.hdf5").write_bytes(sample_hdf5.read_bytes())
    svc._write_json(
        import_dir / "metadata.json",
        {
            "datasetId": dataset_id,
            "name": "索引测试",
            "displayName": "索引测试",
            "sourceJobId": svc.make_import_source_job_id(dataset_id),
            "sourceType": "real_robot_imported",
            "dataSourceLabel": "真实导入",
            "taskType": "custom",
            "format": "hdf5",
            "status": svc.IMPORT_STATUS_AVAILABLE,
            "episodeCount": 1,
            "fileSizeBytes": 128,
            "dataScaleLabel": "128.00 B / 1 episodes",
            "trainable": True,
            "createdAt": "2026-06-26T00:00:00+00:00",
            "updatedAt": "2026-06-26T00:00:00+00:00",
        },
    )

    rows = svc.list_imported_datasets()
    assert any(row["id"] == dataset_id for row in rows)

    monkeypatch.setattr(dataset_svc, "list_datasets", dataset_svc.list_datasets)
    monkeypatch.setattr(
        "app.services.workspace_dataset_import_service.list_imported_datasets",
        svc.list_imported_datasets,
    )
