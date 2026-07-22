from pathlib import Path

from app.services import model_asset_checkpoint_resolver as checkpoint_resolver
from app.services import training_job_sync_service as sync_service
from app.services import training_service
from app.services.training_node_service import TrainingNodeConfig
from app.services.training_remote_runner import remote_training_job_dir


def test_training_write_root_uses_configured_runs_root() -> None:
    assert training_service.TRAINING_ROOT == training_service.RUNTIME_ROOT / "training"


def test_training_job_sync_uses_configured_job_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    current_jobs = tmp_path / "data" / "runs" / "training" / "jobs"
    current_job = current_jobs / "train_old"
    current_job.mkdir(parents=True)
    monkeypatch.setattr(sync_service, "TRAINING_JOBS_ROOT", current_jobs)

    assert sync_service._resolve_train_job_dir("train_old") == current_job.resolve()


def test_checkpoint_resolver_uses_configured_training_job(
    tmp_path: Path,
    monkeypatch,
) -> None:
    current_jobs = tmp_path / "data" / "runs" / "training" / "jobs"
    checkpoint = current_jobs / "train_old" / "checkpoints" / "model_final.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(checkpoint_resolver, "TRAINING_JOBS_ROOT", current_jobs)

    candidates = checkpoint_resolver.iter_local_checkpoint_candidates(
        train_job_id="train_old",
    )

    assert checkpoint in candidates


def test_safe_training_path_rejects_prefix_collision(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runs_root = tmp_path / "data" / "runs"
    monkeypatch.setattr(training_service, "ALLOWED_PATH_ROOTS", [runs_root])

    try:
        training_service._resolve_safe_path(str(tmp_path / "data" / "runs-escape" / "file.hdf5"))
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 403
    else:
        raise AssertionError("prefix-collision path must be rejected")


def test_remote_training_uses_separate_data_root_when_configured() -> None:
    node = TrainingNodeConfig(
        node_id="remote",
        label="remote",
        device_label="L20",
        execution_mode="remote_ssh",
        workdir="/srv/eai-code",
        data_root="/srv/eai-data",
    )

    assert remote_training_job_dir(node, "train_1") == "/srv/eai-data/runs/training/jobs/train_1"


def test_remote_training_uses_runs_under_workdir_without_data_root() -> None:
    node = TrainingNodeConfig(
        node_id="remote",
        label="remote",
        device_label="L20",
        execution_mode="remote_ssh",
        workdir="/srv/eai-code",
    )

    assert remote_training_job_dir(node, "train_old") == (
        "/srv/eai-code/runs/training/jobs/train_old"
    )
