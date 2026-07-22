from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_nut_assembly_uses_external_data_root(tmp_path: Path):
    project_root = Path(__file__).resolve().parents[2]
    data_root = tmp_path / "eai-data"
    code = r'''
from pathlib import Path
from app.services import nut_assembly_service as service

expected = Path(__import__('os').environ['EAI_DATA_ROOT']).resolve() / 'runs' / 'nut_assembly'
assert service.OUTPUT_ROOT == expected

new_id = 'na_gen_20990101_000000_abcd'
job_dir = service._prepare_job_dirs(new_id, include_videos=True)
assert job_dir == expected / 'jobs' / new_id
assert service._job_dir(new_id) == job_dir
assert (job_dir / 'logs').is_dir()
assert (job_dir / 'datasets').is_dir()
assert (job_dir / 'videos').is_dir()
'''
    env = os.environ.copy()
    env["EAI_DATA_ROOT"] = str(data_root)
    env["PYTHONPATH"] = str(project_root / "backend")
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=project_root / "backend",
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    legacy = project_root / "runs" / "nut_assembly" / "jobs" / "na_gen_20990101_000000_abcd"
    assert not legacy.exists()
