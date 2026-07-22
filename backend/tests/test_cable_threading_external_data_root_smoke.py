from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_cable_threading_jobs_use_external_data_root(tmp_path: Path) -> None:
    data_root = tmp_path / "eai-data"
    backend_root = Path(__file__).resolve().parents[1]
    script = r'''
from pathlib import Path

from app.services import cable_threading_service as service

expected_root = Path(__import__("os").environ["EAI_DATA_ROOT"]).resolve()
job_id = "ct_gen_20990101_000000_abcd"
job_dir = service._prepare_job_dirs(job_id, include_datasets=True, include_videos=True)

assert service.OUTPUT_ROOT == expected_root / "runs" / "cable_threading"
assert job_dir == expected_root / "runs" / "cable_threading" / "jobs" / job_id
assert service._job_dir(job_id) == job_dir
assert (job_dir / "logs").is_dir()
assert (job_dir / "results").is_dir()
assert (job_dir / "datasets").is_dir()
assert (job_dir / "videos").is_dir()
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
    legacy_job = backend_root.parent / "runs" / "cable_threading" / "jobs" / "ct_gen_20990101_000000_abcd"
    assert not legacy_job.exists()
