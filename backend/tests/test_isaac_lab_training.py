"""Update isaac training tests for popen_to_log polling."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.schemas.training import CreateTrainingJobRequest
from app.services import training_service as svc
from app.services.isaac_lab import training_service as isaac_train


@pytest.fixture
def isaac_hdf5(tmp_path: Path) -> Path:
    hdf5 = tmp_path / "dataset.hdf5"
    hdf5.write_bytes(b"hdf5")
    return hdf5


@pytest.fixture
def isaac_manifest(isaac_hdf5: Path) -> dict:
    return {
        "datasetId": "isaac_ds_test",
        "datasetName": "stack cube test",
        "taskType": "isaac_block_stacking",
        "taskTemplateId": "isaac_block_stacking",
        "simulatorBackend": "isaac_lab",
        "sourceJobId": "isaac_gen_20260616_132738_feec",
        "datasetFile": str(isaac_hdf5),
        "episodeCount": 10,
        "successfulEpisodes": 10,
        "artifacts": {"hdf5": str(isaac_hdf5)},
    }


def _mock_popen_runner(isaac_root: Path, *, returncode: int = 0):
    runner = MagicMock()
    runner.root = isaac_root
    runner.is_ready.return_value = True
    runner.build_command.return_value = ["isaaclab.sh", "-p", "train.py"]

    proc = MagicMock()
    proc.pid = 4242
    proc.poll.side_effect = [None, 0]
    proc.returncode = returncode

    def popen_side_effect(_script, *args, log_path: Path):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            "command: isaaclab.sh\n\nTrain Epoch 1\n    \"Loss\": -5.0,\nTrain Epoch 2\n    \"Loss\": -14.0,\n",
            encoding="utf-8",
        )
        return proc

    runner.popen_to_log.side_effect = popen_side_effect
    return runner, proc


def test_create_training_job_request_accepts_isaac_robomimic_bc():
    payload = CreateTrainingJobRequest(
        datasetId="isaac_ds_test",
        trainingBackend="isaac_robomimic_bc",
    )
    assert payload.trainingBackend == "isaac_robomimic_bc"


def test_isaac_dataset_trainable_validation(isaac_manifest: dict):
    ok, reason = svc._validate_dataset_trainable(isaac_manifest)
    assert ok is True
    assert reason == ""


def test_isaac_resolve_hdf5_from_dataset_file(isaac_manifest: dict, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(svc, "ALLOWED_PATH_ROOTS", [Path(isaac_manifest["datasetFile"]).parent.parent.resolve()])
    resolved = svc._resolve_hdf5_path(isaac_manifest)
    assert resolved == Path(isaac_manifest["datasetFile"]).resolve()


def test_isaac_backend_routes_to_isaac_robomimic_bc(isaac_manifest: dict):
    capabilities = {"supportedTrainingBackends": ["robomimic_bc", "isaac_robomimic_bc"]}
    backend, message = svc._resolve_training_backend(
        downstream_model_type="Robomimic",
        training_backend="auto",
        has_hdf5=True,
        capabilities=capabilities,
        manifest=isaac_manifest,
    )
    assert backend == "isaac_robomimic_bc"
    assert message == ""


def test_isaac_dataset_rejects_mujoco_robomimic_bc(isaac_manifest: dict):
    capabilities = {"supportedTrainingBackends": ["robomimic_bc", "isaac_robomimic_bc"]}
    backend, message = svc._resolve_training_backend(
        downstream_model_type="Robomimic",
        training_backend="robomimic_bc",
        has_hdf5=True,
        capabilities=capabilities,
        manifest=isaac_manifest,
    )
    assert backend is None
    assert "isaac_robomimic_bc" in message


def test_cable_threading_still_routes_robomimic_bc():
    capabilities = {"supportedTrainingBackends": ["robomimic_bc", "isaac_robomimic_bc"]}
    manifest = {"taskType": "cable_threading", "sourceJobId": "ct_gen_20260101_120000_abcd"}
    backend, message = svc._resolve_training_backend(
        downstream_model_type="Robomimic",
        training_backend="auto",
        has_hdf5=True,
        capabilities=capabilities,
        manifest=manifest,
    )
    assert backend == "robomimic_bc"
    assert message == ""


def test_parse_training_log_isaac_format(tmp_path: Path):
    log_path = tmp_path / "train.log"
    log_path.write_text(
        'Train Epoch 3\n    "Loss": -15.742894287109374,\n',
        encoding="utf-8",
    )
    epoch, loss = svc._parse_training_log(log_path, 10)
    assert epoch == 3
    assert loss == pytest.approx(-15.742894287109374)


def test_execute_isaac_training_registers_model_on_success(
    tmp_path: Path, isaac_manifest: dict, isaac_hdf5: Path, monkeypatch: pytest.MonkeyPatch
):
    train_job_id = "train_isaac_test_001"
    train_job_dir = tmp_path / "jobs" / train_job_id
    logs_dir = train_job_dir / "logs"
    checkpoints_dir = train_job_dir / "checkpoints"
    artifacts_dir = train_job_dir / "artifacts"
    for folder in (logs_dir, checkpoints_dir, artifacts_dir):
        folder.mkdir(parents=True)

    isaac_root = tmp_path / "IsaacLab"
    isaac_log_root = isaac_root / "logs" / "platform_train" / train_job_id / isaac_train.ISAAC_STACK_TASK_ENV
    source_ckpt = isaac_log_root / "2026-06-16_12-00-00" / "models" / "model_epoch_2.pth"
    source_ckpt.parent.mkdir(parents=True)
    source_ckpt.write_bytes(b"checkpoint-bytes")

    runner, _proc = _mock_popen_runner(isaac_root, returncode=0)
    statuses: list[dict] = []

    def update_status(job_dir: Path, payload: dict) -> None:
        statuses.append(payload)

    registered: dict = {}

    def register_model_manifest(**kwargs):
        manifest = {
            "modelAssetId": "model_isaac_test",
            "checkpointPath": str(kwargs["checkpoint_path"]),
            "taskType": "isaac_block_stacking",
            "backendType": "isaac_robomimic_bc",
        }
        registered.update(manifest)
        (artifacts_dir / "model_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return manifest

    monkeypatch.setattr(isaac_train, "IsaacLabCliRunner", MagicMock(from_settings=lambda: runner))
    monkeypatch.setattr(
        isaac_train,
        "probe_isaac_robomimic_training_capability",
        lambda: {"ready": True, "issues": [], "evidence": []},
    )
    monkeypatch.setattr(isaac_train.time, "sleep", lambda _s: None)

    isaac_train.execute_isaac_robomimic_training(
        train_job_id=train_job_id,
        train_job_dir=train_job_dir,
        manifest=isaac_manifest,
        train_config={"epochs": 2, "taskName": "stack test"},
        hdf5_path=isaac_hdf5,
        update_status=update_status,
        register_model_manifest=register_model_manifest,
        sync_workspace_job=lambda _job_id: None,
    )

    final_ckpt = checkpoints_dir / "model_final.pth"
    assert final_ckpt.is_file()
    assert registered["modelAssetId"] == "model_isaac_test"
    assert statuses[-1]["status"] == "completed"
    assert (logs_dir / "train.log").is_file()
    assert any(item.get("epoch", 0) >= 1 for item in statuses)
    assert (artifacts_dir / "model_manifest.json").is_file()


def test_execute_isaac_training_failed_without_checkpoint(
    tmp_path: Path, isaac_manifest: dict, isaac_hdf5: Path, monkeypatch: pytest.MonkeyPatch
):
    train_job_id = "train_isaac_test_002"
    train_job_dir = tmp_path / "jobs" / train_job_id
    logs_dir = train_job_dir / "logs"
    logs_dir.mkdir(parents=True)

    isaac_root = tmp_path / "IsaacLab"
    runner, _proc = _mock_popen_runner(isaac_root, returncode=0)

    statuses: list[dict] = []
    register_called = {"value": False}

    def register_model_manifest(**kwargs):
        register_called["value"] = True
        return {}

    monkeypatch.setattr(isaac_train, "IsaacLabCliRunner", MagicMock(from_settings=lambda: runner))
    monkeypatch.setattr(
        isaac_train,
        "probe_isaac_robomimic_training_capability",
        lambda: {"ready": True, "issues": [], "evidence": []},
    )
    monkeypatch.setattr(isaac_train.time, "sleep", lambda _s: None)

    isaac_train.execute_isaac_robomimic_training(
        train_job_id=train_job_id,
        train_job_dir=train_job_dir,
        manifest=isaac_manifest,
        train_config={"epochs": 2},
        hdf5_path=isaac_hdf5,
        update_status=lambda _job_dir, payload: statuses.append(payload),
        register_model_manifest=register_model_manifest,
        sync_workspace_job=lambda _job_id: None,
    )

    assert register_called["value"] is False
    assert statuses[-1]["status"] == "failed"
    assert "model_epoch" in statuses[-1]["message"]


def test_execute_isaac_training_failed_on_nonzero_exit(
    tmp_path: Path, isaac_manifest: dict, isaac_hdf5: Path, monkeypatch: pytest.MonkeyPatch
):
    train_job_id = "train_isaac_test_003"
    train_job_dir = tmp_path / "jobs" / train_job_id
    logs_dir = train_job_dir / "logs"
    logs_dir.mkdir(parents=True)

    isaac_root = tmp_path / "IsaacLab"
    runner, _proc = _mock_popen_runner(isaac_root, returncode=1)

    statuses: list[dict] = []

    monkeypatch.setattr(isaac_train, "IsaacLabCliRunner", MagicMock(from_settings=lambda: runner))
    monkeypatch.setattr(
        isaac_train,
        "probe_isaac_robomimic_training_capability",
        lambda: {"ready": True, "issues": [], "evidence": []},
    )
    monkeypatch.setattr(isaac_train.time, "sleep", lambda _s: None)

    isaac_train.execute_isaac_robomimic_training(
        train_job_id=train_job_id,
        train_job_dir=train_job_dir,
        manifest=isaac_manifest,
        train_config={"epochs": 2},
        hdf5_path=isaac_hdf5,
        update_status=lambda _job_dir, payload: statuses.append(payload),
        register_model_manifest=lambda **kwargs: kwargs,
        sync_workspace_job=lambda _job_id: None,
    )

    assert statuses[-1]["status"] == "failed"
    assert statuses[-1]["checkpointExists"] is False
