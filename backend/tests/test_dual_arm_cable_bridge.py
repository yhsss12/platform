from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.services import dual_arm_cable_service as svc


def test_make_job_id_format():
    job_id = svc.make_job_id("dac_gen")
    assert re.match(r"^dac_gen_\d{8}_\d{6}_[a-f0-9]{4}$", job_id)


def test_validate_job_id_rejects_traversal():
    from fastapi import HTTPException

    with pytest.raises(HTTPException):
        svc.validate_job_id("../etc/passwd")

    with pytest.raises(HTTPException):
        svc.validate_job_id("dac_p0_test_001")


def test_new_job_directory_uses_current_output_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    current = tmp_path / "data" / "dual_arm_cable"
    legacy = tmp_path / "code" / "runs" / "dual_arm_cable"
    monkeypatch.setattr(svc, "OUTPUT_ROOT", current)
    monkeypatch.setattr(svc, "LEGACY_OUTPUT_ROOT", legacy)

    job_id = "dac_gen_20990101_000000_abcd"
    assert svc._job_dir(job_id) == current / "jobs" / job_id


def test_existing_legacy_job_remains_readable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    current = tmp_path / "data" / "dual_arm_cable"
    legacy = tmp_path / "code" / "runs" / "dual_arm_cable"
    job_id = "dac_gen_20990101_000000_abcd"
    legacy_job = legacy / "jobs" / job_id
    legacy_job.mkdir(parents=True)
    monkeypatch.setattr(svc, "OUTPUT_ROOT", current)
    monkeypatch.setattr(svc, "LEGACY_OUTPUT_ROOT", legacy)

    assert svc._job_dir(job_id) == legacy_job
