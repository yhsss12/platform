from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_evaluation_path_closed_loop_with_external_data_root(tmp_path: Path) -> None:
    data_root = tmp_path / "eai-data"
    backend_root = Path(__file__).resolve().parents[1]
    script = r'''
from pathlib import Path

from app.services import artifact_upload_service
from app.services import training_job_sync_service
from app.services import workspace_runtime_paths
from app.services.evaluation import job_paths
from app.services.evaluation.report_export import report_data

expected_root = Path(__import__("os").environ["EAI_DATA_ROOT"]).resolve()
job_id = "eval_20990101_000000_abcd"
job_dir = job_paths.prepare_eval_job_root(job_id)
(job_dir / "status.json").write_text('{"status":"completed"}', encoding="utf-8")

assert job_paths.EVAL_OUTPUT_ROOT == expected_root / "runs" / "evaluations"
assert job_dir == expected_root / "runs" / "evaluations" / "jobs" / job_id
assert workspace_runtime_paths.resolve_eval_job_root(job_id) == job_dir
assert training_job_sync_service._resolve_eval_job_dir(job_id) == job_dir
assert artifact_upload_service._resolve_eval_root(job_id) == job_dir
assert report_data.REPORT_OUTPUT_ROOT == expected_root / "runs" / "evaluation_reports"
'''
    env = dict(os.environ)
    env["EAI_DATA_ROOT"] = str(data_root)
    env["PYTHONPATH"] = str(backend_root)
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=backend_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    legacy = backend_root.parent / "runs" / "evaluations" / "jobs" / "eval_20990101_000000_abcd"
    assert not legacy.exists()
