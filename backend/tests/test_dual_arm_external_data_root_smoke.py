from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_dual_arm_paths_use_external_data_root(tmp_path: Path) -> None:
    data_root = tmp_path / "eai-data"
    backend_root = Path(__file__).resolve().parents[1]
    script = r'''
from pathlib import Path

from app.services import dual_arm_cable_dataset_service as dataset_service
from app.services import dual_arm_cable_service as service

expected_root = Path(__import__("os").environ["EAI_DATA_ROOT"]).resolve()
job_id = "dac_gen_20990101_000000_abcd"
job_dir = service.OUTPUT_ROOT / "jobs" / job_id
job_dir.mkdir(parents=True)

assert service.OUTPUT_ROOT == expected_root / "runs" / "dual_arm_cable"
assert service._job_dir(job_id) == job_dir
assert dataset_service.DUAL_ARM_ROOT == expected_root / "runs" / "dual_arm_cable" / "jobs"
assert dataset_service.resolve_job_dir(job_id) == job_dir
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
    legacy = backend_root.parent / "runs" / "dual_arm_cable" / "jobs" / "dac_gen_20990101_000000_abcd"
    assert not legacy.exists()
