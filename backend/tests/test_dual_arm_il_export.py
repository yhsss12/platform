from __future__ import annotations

from pathlib import Path

import pytest

from integrations.dual_arm_cable.export_il_dataset import IlExportError, export_job, inspect_job

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SAMPLE_JOB = PROJECT_ROOT / "runs" / "dual_arm_cable" / "jobs" / "dac_gen_20260612_154646_3c5b"


@pytest.mark.skipif(not SAMPLE_JOB.is_dir(), reason="sample dual-arm job missing")
def test_inspect_sample_job_reports_missing_actions():
    report = inspect_job(SAMPLE_JOB)
    assert report["actionAvailable"] is False
    assert report["exportReady"] is False
    assert "step_level_actions" in report["missingFields"]
    assert report["failureReason"] == "missing step-level actions; cannot export IL dataset"


@pytest.mark.skipif(not SAMPLE_JOB.is_dir(), reason="sample dual-arm job missing")
def test_export_sample_job_fails_without_fake_hdf5():
    with pytest.raises(IlExportError) as exc:
        export_job(SAMPLE_JOB)
    assert exc.value.report["hdf5Created"] is False
    export_report_path = SAMPLE_JOB / "datasets" / "export_report.json"
    assert export_report_path.is_file()
