from pathlib import Path

from app.core.platform_paths import (
    build_platform_paths,
    is_path_within,
    resolve_runtime_reference,
    runtime_reference_for_storage,
)


def test_unconfigured_paths_use_repository_local_new_layout(tmp_path: Path) -> None:
    paths = build_platform_paths(project_root=tmp_path, data_root="")

    assert paths.legacy_layout is False
    assert paths.runs_root == tmp_path / "runs"
    assert paths.assets_root == tmp_path / "assets"
    assert paths.logs_root == tmp_path / "logs"
    assert paths.state_root == tmp_path / "state"


def test_configured_paths_use_external_data_layout(tmp_path: Path) -> None:
    project_root = tmp_path / "code" / "eai-idev2.1"
    data_root = tmp_path / "data"
    paths = build_platform_paths(project_root=project_root, data_root=data_root)

    assert paths.legacy_layout is False
    assert paths.data_root == data_root
    assert paths.runs_root == data_root / "runs"
    assert paths.training_jobs == data_root / "runs" / "training" / "jobs"
    assert paths.evaluation_jobs == data_root / "runs" / "evaluations" / "jobs"
    assert paths.datasets == data_root / "assets" / "datasets"
    assert paths.models == data_root / "assets" / "models"
    assert paths.state_root == data_root / "state"


def test_relative_data_root_is_resolved_from_project_root(tmp_path: Path) -> None:
    paths = build_platform_paths(project_root=tmp_path, data_root="../data")

    assert paths.data_root == (tmp_path / "../data").resolve()


def test_path_boundary_check_does_not_accept_prefix_collision(tmp_path: Path) -> None:
    root = tmp_path / "runs"

    assert is_path_within(root / "training" / "job-1", root)
    assert is_path_within(root, root)
    assert not is_path_within(tmp_path / "runs-escape" / "job-1", root)


def test_import_has_no_directory_creation_side_effect(tmp_path: Path) -> None:
    data_root = tmp_path / "not-created"

    build_platform_paths(project_root=tmp_path, data_root=data_root)

    assert not data_root.exists()


def test_new_runtime_reference_round_trip(tmp_path: Path) -> None:
    project_root = tmp_path / "code"
    paths = build_platform_paths(project_root=project_root, data_root=tmp_path / "data")
    job_path = paths.training_jobs / "train_1"

    stored = runtime_reference_for_storage(job_path, paths)

    assert stored == "runs/training/jobs/train_1"
    assert resolve_runtime_reference(stored, paths) == job_path.resolve()


def test_runs_reference_uses_configured_data_root(tmp_path: Path) -> None:
    project_root = tmp_path / "code"
    paths = build_platform_paths(project_root=project_root, data_root=tmp_path / "data")

    resolved = resolve_runtime_reference(
        "runs/training/jobs/train_old",
        paths,
    )

    assert resolved == (tmp_path / "data/runs/training/jobs/train_old").resolve()


def test_external_path_is_stored_absolute(tmp_path: Path) -> None:
    paths = build_platform_paths(project_root=tmp_path / "code", data_root=tmp_path / "data")
    external = tmp_path / "agent-data" / "job-1"

    assert runtime_reference_for_storage(external, paths) == str(external.resolve())


def test_environment_data_root_is_used(monkeypatch, tmp_path: Path) -> None:
    configured = tmp_path / "from-environment"
    monkeypatch.setenv("EAI_DATA_ROOT", str(configured))

    paths = build_platform_paths(project_root=tmp_path)

    assert paths.data_root == configured.resolve()
    assert paths.runs_root == configured.resolve() / "runs"
