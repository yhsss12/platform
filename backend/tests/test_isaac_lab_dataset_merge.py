from __future__ import annotations

from pathlib import Path

import pytest

from app.services import workspace_dataset_service as ws_dataset_svc
from app.services.isaac_lab import isaac_dataset_service as isaac_dataset_svc


def test_list_datasets_merges_isaac_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    registry = tmp_path / "registry.json"
    monkeypatch.setattr(isaac_dataset_svc, "ISAAC_DATASET_REGISTRY_PATH", registry)
    monkeypatch.setattr(ws_dataset_svc, "CABLE_THREADING_ROOT", tmp_path / "cable" / "jobs")
    monkeypatch.setattr(ws_dataset_svc, "DUAL_ARM_ROOT", tmp_path / "dual" / "jobs")

    hdf5 = tmp_path / "stack.hdf5"
    hdf5.write_bytes(b"hdf5")
    row = isaac_dataset_svc.import_demo_hdf5(
        dataset_file=str(hdf5),
        display_name="Merged Demo",
    )

    rows = ws_dataset_svc.list_datasets()
    ids = {item["id"] for item in rows}
    assert row["id"] in ids
