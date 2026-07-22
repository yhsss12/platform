from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services import dual_arm_cable_dataset_service as dac_dataset_svc
from app.services import workspace_dataset_service as ws_dataset_svc

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SAMPLE_OLD = PROJECT_ROOT / "runs" / "dual_arm_cable" / "jobs" / "dac_gen_20260612_154646_3c5b"
SAMPLE_NEW = PROJECT_ROOT / "runs" / "dual_arm_cable" / "jobs" / "dac_gen_20260614_222430_e1f6"


@pytest.mark.skipif(not SAMPLE_OLD.is_dir(), reason="old dual-arm sample job missing")
def test_old_dual_arm_job_lists_manifest_with_failure_reason():
    rows = ws_dataset_svc.list_datasets()
    row = next((r for r in rows if r.get("sourceJobId") == SAMPLE_OLD.name), None)
    assert row is not None
    assert row.get("format") == "manifest"
    assert row.get("trainable") is False
    assert row.get("ilExportFailureReason")


@pytest.mark.skipif(not SAMPLE_NEW.is_dir(), reason="new dual-arm sample job missing")
def test_new_dual_arm_job_lists_hdf5_when_built():
    hdf5 = SAMPLE_NEW / "datasets" / "dataset.hdf5"
    manifest = SAMPLE_NEW / "datasets" / "dataset.manifest.json"
    if not (hdf5.is_file() and manifest.is_file()):
        dac_dataset_svc.auto_build_il_dataset_after_generate(SAMPLE_NEW.name)

    rows = ws_dataset_svc.list_datasets()
    row = next((r for r in rows if r.get("sourceJobId") == SAMPLE_NEW.name), None)
    assert row is not None
    assert row.get("format") == "hdf5"
    assert row.get("trainable") is True
    assert row.get("episodeCount", 0) >= 1
    assert row.get("builtDatasetPath")
    assert "torch_bc" in (row.get("trainingBackends") or [])


def test_auto_build_skips_non_completed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    job_id = "dac_gen_20260615_120000_abcd"
    job_dir = tmp_path / job_id
    job_dir.mkdir(parents=True)
    (job_dir / "status.json").write_text(
        json.dumps({"status": "running"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(dac_dataset_svc, "DUAL_ARM_ROOT", tmp_path)
    result = dac_dataset_svc.auto_build_il_dataset_after_generate(job_id)
    assert result["status"] == "skipped"
