from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_training_path_closed_loop_with_external_data_root(tmp_path: Path) -> None:
    data_root = tmp_path / "eai-data"
    backend_root = Path(__file__).resolve().parents[1]
    script = r'''
import json
from pathlib import Path

from app.services import artifact_upload_service
from app.services import model_asset_checkpoint_resolver
from app.services import training_job_sync_service
from app.services import training_service
from app.services import workspace_job_service
from app.services import workspace_model_asset_service

expected_root = Path(__import__("os").environ["EAI_DATA_ROOT"]).resolve()
job_id = "train_20990101_000000_abcd"
job_dir = expected_root / "runs" / "training" / "jobs" / job_id
checkpoint = job_dir / "checkpoints" / "model_final.pt"
checkpoint.parent.mkdir(parents=True)
checkpoint.write_bytes(b"checkpoint")
(job_dir / "status.json").write_text(
    json.dumps({"status": "completed", "trainJobId": job_id}),
    encoding="utf-8",
)

assert training_service.TRAINING_ROOT == expected_root / "runs" / "training"
assert training_job_sync_service._resolve_train_job_dir(job_id) == job_dir
assert workspace_model_asset_service._find_training_job_dir(job_id) == job_dir
assert checkpoint in model_asset_checkpoint_resolver.iter_local_checkpoint_candidates(
    train_job_id=job_id,
)
assert job_id in artifact_upload_service.discover_terminal_job_ids(limit=100)
assert workspace_job_service._validate_runtime_delete_path(str(job_dir)) == job_dir
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
    assert not (backend_root.parent / "runs" / "training" / "jobs" / "train_20990101_000000_abcd").exists()

