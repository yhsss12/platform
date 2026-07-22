from __future__ import annotations

from pathlib import Path

from app.services import nut_assembly_service as service


def test_nut_assembly_job_dir_prefers_current_then_legacy(
    tmp_path: Path, monkeypatch
):
    current = tmp_path / "current"
    legacy = tmp_path / "legacy"
    monkeypatch.setattr(service, "OUTPUT_ROOT", current)
    monkeypatch.setattr(service, "LEGACY_OUTPUT_ROOT", legacy)

    job_id = "na_gen_20990101_000000_abcd"
    assert service._job_dir(job_id) == current / "jobs" / job_id

    legacy_job = legacy / "jobs" / job_id
    legacy_job.mkdir(parents=True)
    assert service._job_dir(job_id) == legacy_job

    current_job = current / "jobs" / job_id
    current_job.mkdir(parents=True)
    assert service._job_dir(job_id) == current_job
