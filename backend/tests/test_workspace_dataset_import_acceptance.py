"""验收测试：数据中心 HDF5 导入边界与稳定性。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from app.services import workspace_dataset_import_service as svc


@pytest.fixture
def import_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "imports"
    monkeypatch.setattr(svc, "IMPORT_ROOT", root)
    monkeypatch.setattr(svc, "TRAINING_JOBS_ROOT", tmp_path / "training_jobs")
    return root


def _make_hdf5(path: Path, builder) -> Path:
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, "w") as handle:
        builder(handle)
    return path


def _upload_file(path: Path, filename: str | None = None):
    class _Upload:
        def __init__(self, file_path: Path, upload_name: str) -> None:
            self.filename = upload_name
            self._path = file_path

        async def read(self) -> bytes:
            return self._path.read_bytes()

    return _Upload(path, filename or path.name)


def _import(path: Path, import_root: Path, **kwargs) -> dict:
    defaults = {
        "name": "验收数据集",
        "data_source": "real_collection",
        "task_type": "cable_threading",
        "robot_type": "fr3",
    }
    defaults.update(kwargs)
    return asyncio.run(
        svc.import_hdf5_dataset_upload(
            file=_upload_file(path),
            **defaults,
        )
    )


def test_action_qpos_dataset_available(import_root: Path, tmp_path: Path) -> None:
    import numpy as np

    path = _make_hdf5(
        tmp_path / "action_qpos.hdf5",
        lambda h: (
            (data := h.create_group("data")).create_group("demo_0")
            .create_group("obs")
            .create_dataset("qpos", data=np.zeros((8, 7), dtype=np.float32)),
            data["demo_0"].create_dataset("actions", data=np.zeros((8, 7), dtype=np.float32)),
        ),
    )
    result = _import(path, import_root)
    assert result["status"] == svc.IMPORT_STATUS_AVAILABLE
    report = json.loads((import_root / result["datasetId"] / "validation_report.json").read_text(encoding="utf-8"))
    assert report["trainable"] is True
    assert report["errors"] == []


def test_action_image_dataset_available(import_root: Path, tmp_path: Path) -> None:
    import numpy as np

    path = _make_hdf5(
        tmp_path / "action_image.hdf5",
        lambda h: (
            (data := h.create_group("data")).create_group("demo_0")
            .create_group("obs")
            .create_dataset("agentview_image", data=np.zeros((8, 64, 64, 3), dtype=np.uint8)),
            data["demo_0"].create_dataset("actions", data=np.zeros((8, 7), dtype=np.float32)),
        ),
    )
    result = _import(path, import_root)
    assert result["status"] == svc.IMPORT_STATUS_AVAILABLE


def test_missing_action_pending_mapping(import_root: Path, tmp_path: Path) -> None:
    import numpy as np

    path = _make_hdf5(
        tmp_path / "no_action.hdf5",
        lambda h: (
            (data := h.create_group("data")).create_group("demo_0")
            .create_group("obs")
            .create_dataset("qpos", data=np.zeros((8, 7), dtype=np.float32))
        ),
    )
    result = _import(path, import_root)
    assert result["status"] == svc.IMPORT_STATUS_PENDING_MAPPING
    assert result["dataset"]["trainable"] is False


def test_missing_obs_pending_mapping(import_root: Path, tmp_path: Path) -> None:
    import numpy as np

    path = _make_hdf5(
        tmp_path / "no_obs.hdf5",
        lambda h: (
            (data := h.create_group("data")).create_group("demo_0").create_dataset(
                "actions", data=np.zeros((8, 7), dtype=np.float32)
            )
        ),
    )
    result = _import(path, import_root)
    assert result["status"] == svc.IMPORT_STATUS_PENDING_MAPPING
    assert result["dataset"]["needsMapping"] is True
    assert result["dataset"]["directTrainable"] is False


def test_time_dim_mismatch_needs_mapping(import_root: Path, tmp_path: Path) -> None:
    import numpy as np

    path = _make_hdf5(
        tmp_path / "time_mismatch.hdf5",
        lambda h: (
            (data := h.create_group("data")).create_group("demo_0")
            .create_group("obs")
            .create_dataset("qpos", data=np.zeros((10, 7), dtype=np.float32)),
            data["demo_0"].create_dataset("actions", data=np.zeros((8, 7), dtype=np.float32)),
        ),
    )
    result = _import(path, import_root)
    assert result["status"] == svc.IMPORT_STATUS_NEEDS_MAPPING
    assert result["dataset"]["trainable"] is False
    assert result["dataset"]["needsMapping"] is True
    report = json.loads((import_root / result["datasetId"] / "validation_report.json").read_text(encoding="utf-8"))
    assert report["trainable"] is False
    assert any("不一致" in err for err in report.get("errors") or [])


def test_no_episode_needs_build(import_root: Path, tmp_path: Path) -> None:
    import numpy as np

    path = _make_hdf5(
        tmp_path / "flat.hdf5",
        lambda h: (
            h.create_dataset("actions", data=np.zeros((8, 7), dtype=np.float32)),
            h.create_dataset("qpos", data=np.zeros((8, 7), dtype=np.float32)),
        ),
    )
    result = _import(path, import_root)
    assert result["status"] == svc.IMPORT_STATUS_NEEDS_BUILD
    assert result["dataset"]["needsBuild"] is True
    assert result["dataset"]["episodeParsed"] is False
    assert "待构建" in (result["dataset"].get("dataScaleLabel") or "")


def test_ready_dataset_direct_trainable(import_root: Path, tmp_path: Path) -> None:
    import numpy as np

    path = _make_hdf5(
        tmp_path / "ready.hdf5",
        lambda h: (
            (data := h.create_group("data")).create_group("demo_0")
            .create_group("obs")
            .create_dataset("state", data=np.zeros((6, 7), dtype=np.float32)),
            data["demo_0"].create_dataset("action", data=np.zeros((6, 7), dtype=np.float32)),
        ),
    )
    result = _import(path, import_root)
    assert result["status"] == svc.IMPORT_STATUS_READY
    assert result["dataset"]["directTrainable"] is True
    assert result["dataset"]["needsBuild"] is False
    assert result["dataset"]["needsMapping"] is False
    report = json.loads((import_root / result["datasetId"] / "validation_report.json").read_text(encoding="utf-8"))
    assert report["trainable"] is True


def test_corrupt_hdf5_import_failed(import_root: Path, tmp_path: Path) -> None:
    path = tmp_path / "corrupt.hdf5"
    path.write_bytes(b"not-a-valid-hdf5-content")
    result = _import(path, import_root)
    assert result["status"] == svc.IMPORT_STATUS_FAILED
    report = json.loads((import_root / result["datasetId"] / "validation_report.json").read_text(encoding="utf-8"))
    assert report["trainable"] is False
    assert report["errors"]


def test_empty_hdf5_not_available(import_root: Path, tmp_path: Path) -> None:
    path = _make_hdf5(tmp_path / "empty.hdf5", lambda h: None)
    result = _import(path, import_root)
    assert result["status"] != svc.IMPORT_STATUS_AVAILABLE


def test_schema_contains_path_shape_dtype(import_root: Path, tmp_path: Path) -> None:
    import numpy as np

    path = _make_hdf5(
        tmp_path / "schema.hdf5",
        lambda h: (
            (data := h.create_group("data")).create_group("demo_0")
            .create_group("obs")
            .create_dataset("qpos", data=np.zeros((4, 7), dtype=np.float32)),
            data["demo_0"].create_dataset("actions", data=np.zeros((4, 7), dtype=np.float32)),
        ),
    )
    result = _import(path, import_root)
    schema = json.loads((import_root / result["datasetId"] / "schema.json").read_text(encoding="utf-8"))
    fields = schema.get("fields") or []
    assert fields
    sample = fields[0]
    assert "path" in sample and "shape" in sample and "dtype" in sample


def test_saved_path_fixed_and_filename_sanitized(import_root: Path, tmp_path: Path) -> None:
    import numpy as np

    path = _make_hdf5(
        tmp_path / "real.hdf5",
        lambda h: (
            (data := h.create_group("data")).create_group("demo_0")
            .create_group("obs")
            .create_dataset("qpos", data=np.zeros((4, 7), dtype=np.float32)),
            data["demo_0"].create_dataset("actions", data=np.zeros((4, 7), dtype=np.float32)),
        ),
    )
    result = asyncio.run(
        svc.import_hdf5_dataset_upload(
            name="穿越测试",
            data_source="simulation_export",
            task_type="custom",
            robot_type="fr3",
            file=_upload_file(path, "../../evil/name.hdf5"),
        )
    )
    dataset_id = result["datasetId"]
    hdf5_path = import_root / dataset_id / "source.hdf5"
    assert hdf5_path.is_file()
    assert "evil" not in str(hdf5_path)
    metadata = json.loads((import_root / dataset_id / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["dataSourceLabel"] == "仿真导出"
    assert metadata["originalFileName"] == "name.hdf5"


def test_dataset_ids_are_unique(import_root: Path, tmp_path: Path) -> None:
    ids = {svc.make_dataset_import_id() for _ in range(200)}
    assert len(ids) == 200


def test_delete_removes_registry_and_directory(import_root: Path, tmp_path: Path) -> None:
    import numpy as np

    path = _make_hdf5(
        tmp_path / "delete_me.hdf5",
        lambda h: (
            (data := h.create_group("data")).create_group("demo_0")
            .create_group("obs")
            .create_dataset("qpos", data=np.zeros((2, 7), dtype=np.float32)),
            data["demo_0"].create_dataset("actions", data=np.zeros((2, 7), dtype=np.float32)),
        ),
    )
    result = _import(path, import_root)
    dataset_id = result["datasetId"]
    assert (import_root / dataset_id).is_dir()
    svc.delete_imported_dataset(dataset_id)
    assert not (import_root / dataset_id).exists()
    assert not any(row["id"] == dataset_id for row in svc.list_imported_datasets())


def test_delete_blocked_when_referenced_by_training(import_root: Path, tmp_path: Path) -> None:
    import numpy as np

    path = _make_hdf5(
        tmp_path / "referenced.hdf5",
        lambda h: (
            (data := h.create_group("data")).create_group("demo_0")
            .create_group("obs")
            .create_dataset("qpos", data=np.zeros((2, 7), dtype=np.float32)),
            data["demo_0"].create_dataset("actions", data=np.zeros((2, 7), dtype=np.float32)),
        ),
    )
    result = _import(path, import_root)
    dataset_id = result["datasetId"]

    train_root = tmp_path / "training_jobs" / "train_job_1" / "artifacts"
    train_root.mkdir(parents=True)
    (train_root / "dataset_manifest.json").write_text(
        json.dumps({"datasetId": dataset_id}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="训练任务引用"):
        svc.delete_imported_dataset(dataset_id)
