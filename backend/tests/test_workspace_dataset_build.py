from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services import workspace_dataset_build_service as build_svc
from app.services import workspace_dataset_import_service as import_svc
from app.services import workspace_dataset_service as dataset_svc


@pytest.fixture
def nonstandard_hdf5(tmp_path: Path) -> Path:
    h5py = pytest.importorskip("h5py")
    import numpy as np

    file_path = tmp_path / "custom.hdf5"
    with h5py.File(file_path, "w") as handle:
        handle.create_dataset("action", data=np.zeros((12, 7), dtype=np.float32))
        obs = handle.create_group("observations")
        obs.create_dataset("qpos", data=np.zeros((12, 7), dtype=np.float32))
    return file_path


@pytest.fixture
def import_env(
    tmp_path: Path,
    nonstandard_hdf5: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[str, Path, Path]:
    import_root = tmp_path / "imports"
    built_root = tmp_path / "built"
    monkeypatch.setattr(import_svc, "IMPORT_ROOT", import_root)
    monkeypatch.setattr(build_svc, "IMPORT_ROOT", import_root)
    monkeypatch.setattr(build_svc, "BUILT_ROOT", built_root)

    dataset_id = import_svc.make_dataset_import_id()
    import_dir = import_root / dataset_id
    import_dir.mkdir(parents=True)
    (import_dir / "source.hdf5").write_bytes(nonstandard_hdf5.read_bytes())
    import_svc._write_json(
        import_dir / "metadata.json",
        {
            "datasetId": dataset_id,
            "name": "待构建导入",
            "displayName": "待构建导入",
            "sourceJobId": import_svc.make_import_source_job_id(dataset_id),
            "sourceType": "real_robot_imported",
            "dataSourceLabel": "真实导入",
            "taskType": "custom",
            "format": "hdf5",
            "status": import_svc.IMPORT_STATUS_NEEDS_MAPPING,
            "episodeCount": 0,
            "episodeParsed": False,
            "trainable": False,
            "needsMapping": True,
            "fileSizeBytes": nonstandard_hdf5.stat().st_size,
            "createdAt": "2026-06-26T00:00:00+00:00",
            "updatedAt": "2026-06-26T00:00:00+00:00",
        },
    )
    parsed = import_svc._parse_hdf5_file(import_dir / "source.hdf5")
    import_svc._write_json(import_dir / "schema.json", {"fields": parsed["tree"]})
    return dataset_id, import_dir, built_root


def test_get_import_dataset_schema(import_env: tuple[str, Path, Path]) -> None:
    dataset_id, _, _ = import_env
    schema = build_svc.get_import_dataset_schema(dataset_id)
    assert schema["datasetId"] == dataset_id
    assert any(field["path"] == "action" for field in schema["fields"])


def test_build_auto_detect_without_manual_mapping(
    import_env: tuple[str, Path, Path],
) -> None:
    dataset_id, import_dir, built_root = import_env
    result = build_svc.build_dataset_from_import(
        {
            "sourceDatasetId": dataset_id,
            "outputName": "自动识别_built",
            "taskType": "custom",
            "targetFormat": "standard_hdf5",
            "auto": True,
        }
    )
    built_id = result["builtDatasetId"]
    assert (built_root / built_id / "dataset.hdf5").is_file()
    assert import_dir.is_dir()


def test_build_auto_detect_failure_message(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    h5py = pytest.importorskip("h5py")
    import numpy as np
    from fastapi import HTTPException

    import_root = tmp_path / "imports"
    built_root = tmp_path / "built"
    monkeypatch.setattr(import_svc, "IMPORT_ROOT", import_root)
    monkeypatch.setattr(build_svc, "IMPORT_ROOT", import_root)
    monkeypatch.setattr(build_svc, "BUILT_ROOT", built_root)

    file_path = tmp_path / "empty.hdf5"
    with h5py.File(file_path, "w") as handle:
        handle.create_dataset("unknown_field", data=np.zeros((4, 3), dtype=np.float32))

    dataset_id = import_svc.make_dataset_import_id()
    import_dir = import_root / dataset_id
    import_dir.mkdir(parents=True)
    (import_dir / "source.hdf5").write_bytes(file_path.read_bytes())
    import_svc._write_json(
        import_dir / "metadata.json",
        {
            "datasetId": dataset_id,
            "name": "无法识别",
            "displayName": "无法识别",
            "sourceJobId": import_svc.make_import_source_job_id(dataset_id),
            "format": "hdf5",
            "status": import_svc.IMPORT_STATUS_NEEDS_MAPPING,
        },
    )

    with pytest.raises(HTTPException) as exc:
        build_svc.build_dataset_from_import(
            {
                "sourceDatasetId": dataset_id,
                "outputName": "fail_built",
                "taskType": "custom",
                "targetFormat": "standard_hdf5",
                "auto": True,
            }
        )
    assert str(exc.value.detail) == build_svc.AUTO_FIELD_DETECT_FAILED_MSG


def test_build_dataset_from_import_creates_built_artifacts(
    import_env: tuple[str, Path, Path],
) -> None:
    dataset_id, import_dir, built_root = import_env
    result = build_svc.build_dataset_from_import(
        {
            "sourceDatasetId": dataset_id,
            "outputName": "待构建导入_built",
            "taskType": "custom",
            "targetFormat": "standard_hdf5",
            "fieldMapping": {
                "action": "action",
                "qpos": "observations/qpos",
                "image": None,
                "qvel": None,
                "done": None,
            },
            "episodeRule": {"type": "single_episode"},
        }
    )

    built_id = result["builtDatasetId"]
    built_dir = built_root / built_id
    assert built_dir.is_dir()
    assert (built_dir / "dataset.hdf5").is_file()
    assert (built_dir / "manifest.json").is_file()
    assert (built_dir / "schema.json").is_file()
    assert (built_dir / "validation_report.json").is_file()
    assert import_dir.is_dir()
    assert (import_dir / "source.hdf5").is_file()

    manifest = json.loads((built_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["sourceType"] == "real_robot_built"
    assert manifest["dataSourceLabel"] == "真实数据构建"
    assert manifest["dataCount"] == 1
    assert manifest["trainable"] is True
    assert manifest["directTrainable"] is True
    assert manifest["sourceDatasetId"] == dataset_id

    row = result["dataset"]
    assert row is not None
    assert row["id"] == built_id
    assert row["dataCount"] == 1
    assert row["fileSizeBytes"] > 0


def test_build_rejects_mismatched_time_dimension(
    import_env: tuple[str, Path, Path],
) -> None:
    dataset_id, import_dir, _ = import_env
    h5py = pytest.importorskip("h5py")
    import numpy as np

    with h5py.File(import_dir / "source.hdf5", "a") as handle:
        obs = handle["observations"]
        del obs["qpos"]
        obs.create_dataset("qpos", data=np.zeros((8, 7), dtype=np.float32))

    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        build_svc.build_dataset_from_import(
            {
                "sourceDatasetId": dataset_id,
                "outputName": "bad_built",
                "taskType": "custom",
                "targetFormat": "standard_hdf5",
                "fieldMapping": {
                    "action": "action",
                    "qpos": "observations/qpos",
                },
                "episodeRule": {"type": "single_episode"},
            }
        )
    assert "不一致" in str(exc.value.detail)


def test_list_built_datasets_in_workspace_index(
    import_env: tuple[str, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_id, _, _ = import_env
    result = build_svc.build_dataset_from_import(
        {
            "sourceDatasetId": dataset_id,
            "outputName": "索引测试_built",
            "taskType": "custom",
            "targetFormat": "standard_hdf5",
            "fieldMapping": {"action": "action", "qpos": "observations/qpos"},
            "episodeRule": {"type": "single_episode"},
        }
    )
    built_id = result["builtDatasetId"]

    monkeypatch.setattr(
        "app.services.workspace_dataset_build_service.list_built_datasets",
        build_svc.list_built_datasets,
    )
    monkeypatch.setattr(
        "app.services.workspace_dataset_import_service.list_imported_datasets",
        import_svc.list_imported_datasets,
    )

    rows = build_svc.list_built_datasets()
    assert any(row["id"] == built_id for row in rows)

    combined = dataset_svc.list_datasets()
    assert any(row.get("id") == built_id for row in combined)
