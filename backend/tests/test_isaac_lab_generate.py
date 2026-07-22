from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException

from app.core.platform_paths import build_platform_paths
from app.services.isaac_lab import generate_service as generate_svc
from app.services.isaac_lab import paths as paths_svc
from app.services.isaac_lab.generate_cli import (
    DEFAULT_MIMIC_TASK_ID,
    DEFAULT_SCRIPTED_EXPERT_TASK_ID,
    EXPERT_POLICY_SCRIPT_BASENAME,
    SCRIPTED_EXPERT_SCRIPT_BASENAME,
    build_annotate_demos_cli_args,
    build_expert_policy_cli_args,
    build_generate_dataset_cli_args,
    build_scripted_expert_cli_args,
    MimicGenerateCliParams,
    ScriptedExpertCliParams,
)
from app.services.isaac_lab.job_paths import (
    is_isaac_gen_job_id,
    isaac_job_generation_manifest_path,
    isaac_job_metrics_path,
    isaac_job_status_path,
)
from app.services.isaac_lab.cli_runner import IsaacLabCliRunner, IsaacLabCliRunResult
from app.services.isaac_lab.paths import resolve_stack_cube_default_seed


def _configure_isaac_root(tmp_path: Path, *, include_mimic_task: bool = True) -> Path:
    root = tmp_path / "IsaacLab"
    root.mkdir()
    sh = root / "isaaclab.sh"
    sh.write_text("#!/bin/bash\n", encoding="utf-8")
    sh.chmod(0o755)
    tasks_dir = root / "source" / "isaaclab_tasks"
    tasks_dir.mkdir(parents=True)
    (root / "VERSION").write_text("2.3.2\n", encoding="utf-8")
    if include_mimic_task:
        (tasks_dir / "stack_mimic.py").write_text(
            f'ENV_ID = "{DEFAULT_MIMIC_TASK_ID}"\n',
            encoding="utf-8",
        )
        (tasks_dir / "stack_replay.py").write_text(
            'ENV_ID = "Isaac-Stack-Cube-Franka-IK-Rel-v0"\n',
            encoding="utf-8",
        )
    return root


def test_isaac_gen_job_id_pattern():
    job_id = generate_svc.make_isaac_gen_job_id()
    assert is_isaac_gen_job_id(job_id)


def test_build_mimic_cli_args(tmp_path: Path):
    seed = tmp_path / "seed.hdf5"
    seed.write_bytes(b"hdf5")
    annotated = tmp_path / "annotated.hdf5"
    output = tmp_path / "out.hdf5"
    params = MimicGenerateCliParams(
        mimic_task_id=DEFAULT_MIMIC_TASK_ID,
        seed_dataset_file=seed,
        annotated_dataset_file=annotated,
        output_dataset_file=output,
        num_demos=5,
        num_envs=2,
        headless=True,
        enable_cameras=True,
    )
    args = build_generate_dataset_cli_args(params)
    assert "--generation_num_trials" in args
    assert "5" in args
    assert "--headless" in args

    annotate_args = build_annotate_demos_cli_args(
        mimic_task_id=DEFAULT_MIMIC_TASK_ID,
        input_file=seed,
        output_file=annotated,
        headless=True,
        enable_cameras=False,
    )
    assert "--auto" in annotate_args
    assert DEFAULT_MIMIC_TASK_ID in annotate_args


def test_start_generate_503_when_unconfigured(monkeypatch: pytest.MonkeyPatch):
    from app.core.config import settings

    settings.ISAACLAB_ROOT = None
    settings.ISAACLAB_RUNTIME_ENABLED = False

    with pytest.raises(HTTPException) as exc:
        generate_svc.start_generate_dataset(
            dataset_name="test",
            num_demos=1,
            seed_dataset_file="/tmp/seed.hdf5",
        )
    assert exc.value.status_code == 503


def test_start_generate_400_mimic_without_seed_or_default(tmp_path: Path):
    root = _configure_isaac_root(tmp_path)
    from app.core.config import settings

    settings.ISAACLAB_ROOT = str(root)
    settings.ISAACLAB_RUNTIME_ENABLED = True
    settings.ISAACLAB_STACK_CUBE_DEFAULT_SEED = str(tmp_path / "missing_default.hdf5")

    with pytest.raises(HTTPException) as exc:
        generate_svc.start_generate_dataset(
            dataset_name="test",
            num_demos=1,
            generation_mode="mimic_auto",
        )
    assert exc.value.status_code == 400
    assert "Default 物块堆叠 seed HDF5" in str(exc.value.detail)


def test_resolve_generation_seed_uses_default(tmp_path: Path):
    seed = tmp_path / "stack_cube_seed.hdf5"
    seed.write_bytes(b"hdf5")
    from app.core.config import settings

    settings.ISAACLAB_STACK_CUBE_DEFAULT_SEED = str(seed)

    path, source, dataset_id = generate_svc.resolve_generation_seed()
    assert path.resolve() == seed.resolve()
    assert source == "default_seed"
    assert dataset_id is None


def test_resolve_stack_cube_default_seed_relative_to_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    seed_rel = tmp_path / "runs" / "isaac_lab" / "seeds"
    seed_rel.mkdir(parents=True)
    seed_file = seed_rel / "stack_cube_seed.hdf5"
    seed_file.write_bytes(b"hdf5")

    monkeypatch.setattr("app.services.isaac_lab.paths.PROJECT_ROOT", tmp_path)
    from app.core.config import settings

    settings.ISAACLAB_STACK_CUBE_DEFAULT_SEED = "runs/isaac_lab/seeds/stack_cube_seed.hdf5"
    settings.ISAACLAB_ROOT = None

    resolved, exists = resolve_stack_cube_default_seed()
    assert exists is True
    assert resolved.resolve() == seed_file.resolve()


def test_resolve_stack_cube_default_seed_uses_platform_data_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    seed_file = tmp_path / "runs" / "isaac_lab" / "seeds" / "stack_cube_seed.hdf5"
    seed_file.parent.mkdir(parents=True)
    seed_file.write_bytes(b"seed")
    monkeypatch.setattr(paths_svc, "platform_paths", build_platform_paths(project_root=tmp_path / "code", data_root=tmp_path))
    monkeypatch.setattr(paths_svc.settings, "ISAACLAB_STACK_CUBE_DEFAULT_SEED", "")

    resolved, exists = paths_svc.resolve_stack_cube_default_seed()

    assert exists is True
    assert resolved == seed_file


def test_start_generate_400_missing_seed_file(tmp_path: Path):
    root = _configure_isaac_root(tmp_path)
    from app.core.config import settings

    settings.ISAACLAB_ROOT = str(root)
    settings.ISAACLAB_RUNTIME_ENABLED = True

    with pytest.raises(HTTPException) as exc:
        generate_svc.start_generate_dataset(
            dataset_name="test",
            num_demos=1,
            generation_mode="mimic_auto",
            seed_dataset_file=str(tmp_path / "missing.hdf5"),
        )
    assert exc.value.status_code == 400


def _configure_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = _configure_isaac_root(tmp_path)
    jobs_root = tmp_path / "jobs"
    monkeypatch.setenv("ISAACLAB_ROOT", str(root))
    monkeypatch.setenv("ISAACLAB_RUNTIME_ENABLED", "true")
    monkeypatch.setenv("ISAACLAB_OUTPUT_ROOT", str(jobs_root))
    from app.core.config import settings

    settings.ISAACLAB_ROOT = str(root)
    settings.ISAACLAB_RUNTIME_ENABLED = True
    settings.ISAACLAB_OUTPUT_ROOT = str(jobs_root)
    settings.ISAACLAB_GENERATE_TIMEOUT = 30
    return root


def test_build_expert_policy_cli_args(tmp_path: Path):
    output = tmp_path / "dataset.hdf5"
    params = ScriptedExpertCliParams(
        task_id=DEFAULT_SCRIPTED_EXPERT_TASK_ID,
        dataset_file=output,
        num_demos=3,
        seed=42,
        max_attempts=15,
        headless=True,
        enable_cameras=False,
    )
    args = build_expert_policy_cli_args(params)
    assert DEFAULT_SCRIPTED_EXPERT_TASK_ID in args
    assert str(output) in args
    assert "--num_demos" in args
    assert "3" in args
    assert "--seed" in args
    assert "42" in args
    assert "--max_attempts" in args
    assert "--headless" in args
    assert "--enable_cameras" not in args
    assert "--no-record_camera_obs" not in args
    assert "--image_resolution" in args


def test_build_expert_policy_cli_args_with_camera_options(tmp_path: Path):
    output = tmp_path / "dataset.hdf5"
    params = ScriptedExpertCliParams(
        task_id=DEFAULT_SCRIPTED_EXPERT_TASK_ID,
        dataset_file=output,
        num_demos=2,
        seed=1,
        max_attempts=0,
        headless=True,
        enable_cameras=True,
        record_camera_obs=False,
        image_resolution=256,
        include_wrist_camera=True,
    )
    args = build_expert_policy_cli_args(params)
    assert "--enable_cameras" in args
    assert "--no-record_camera_obs" in args
    assert "--image_resolution" in args
    assert "256" in args
    assert "--include_wrist_camera" in args


def test_build_scripted_expert_cli_args(tmp_path: Path):
    output = tmp_path / "dataset.hdf5"
    params = ScriptedExpertCliParams(
        task_id=DEFAULT_SCRIPTED_EXPERT_TASK_ID,
        dataset_file=output,
        num_demos=3,
        seed=42,
        max_attempts=15,
        headless=True,
        enable_cameras=False,
    )
    args = build_scripted_expert_cli_args(params)
    assert DEFAULT_SCRIPTED_EXPERT_TASK_ID in args
    assert str(output) in args
    assert "--num_demos" in args
    assert "3" in args
    assert "--seed" in args
    assert "42" in args
    assert "--max_attempts" in args
    assert "--headless" in args
    assert "--enable_cameras" not in args


def test_start_scripted_expert_does_not_require_seed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _configure_runtime(tmp_path, monkeypatch)
    from app.core.config import settings

    settings.ISAACLAB_STACK_CUBE_DEFAULT_SEED = str(tmp_path / "missing_default.hdf5")

    def _fake_run(self, script_relative, *args, stdout_path, stderr_path, timeout):
        stdout_path.write_text("ERROR: isaaclab is not available\n", encoding="utf-8")
        stderr_path.write_text("ModuleNotFoundError: No module named 'isaaclab'\n", encoding="utf-8")
        return IsaacLabCliRunResult(
            returncode=1,
            command=self.build_command(script_relative, *args),
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )

    monkeypatch.setattr(IsaacLabCliRunner, "run_to_files", _fake_run)

    started = generate_svc.start_generate_dataset(
        dataset_name="scripted-test",
        num_demos=2,
        generation_mode="scripted_expert",
        seed=7,
    )
    assert is_isaac_gen_job_id(started["jobId"])


def test_scripted_expert_job_failed_on_cli_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _configure_runtime(tmp_path, monkeypatch)

    def _fake_run(self, script_relative, *args, stdout_path, stderr_path, timeout):
        assert EXPERT_POLICY_SCRIPT_BASENAME in script_relative
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("ModuleNotFoundError: No module named 'isaaclab'\n", encoding="utf-8")
        return IsaacLabCliRunResult(
            returncode=1,
            command=self.build_command(script_relative, *args),
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )

    monkeypatch.setattr(IsaacLabCliRunner, "run_to_files", _fake_run)

    started = generate_svc.start_generate_dataset(
        dataset_name="scripted-fail",
        num_demos=1,
        generation_mode="scripted_expert",
    )
    job_id = started["jobId"]
    import json
    import time

    status = {}
    for _ in range(50):
        status = generate_svc.get_generate_job_status(job_id)
        if status["status"] in {"completed", "failed"}:
            break
        time.sleep(0.05)

    assert status["status"] == "failed"
    assert status.get("datasetId") is None
    manifest = json.loads(isaac_job_generation_manifest_path(job_id).read_text(encoding="utf-8"))
    assert manifest["generationMode"] == "expert_policy"
    assert manifest["seedSource"] is None
    assert manifest["expertScript"] == EXPERT_POLICY_SCRIPT_BASENAME


def test_scripted_expert_job_failed_when_hdf5_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _configure_runtime(tmp_path, monkeypatch)

    def _fake_run(self, script_relative, *args, stdout_path, stderr_path, timeout):
        stdout_path.write_text("SUCCESS\n", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return IsaacLabCliRunResult(
            returncode=0,
            command=self.build_command(script_relative, *args),
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )

    monkeypatch.setattr(IsaacLabCliRunner, "run_to_files", _fake_run)

    started = generate_svc.start_generate_dataset(
        dataset_name="scripted-no-hdf5",
        num_demos=1,
        generation_mode="scripted_expert",
    )
    job_id = started["jobId"]
    import json
    import time

    status = {}
    for _ in range(50):
        status = generate_svc.get_generate_job_status(job_id)
        if status["status"] in {"completed", "failed"}:
            break
        time.sleep(0.05)

    assert status["status"] == "failed"
    assert status.get("datasetAvailable") is False
    metrics = json.loads(isaac_job_metrics_path(job_id).read_text(encoding="utf-8"))
    assert metrics["generationMode"] == "expert_policy"
    assert metrics["episodeCount"] == 0


def test_runtime_status_includes_scripted_expert_flags(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = _configure_isaac_root(tmp_path)
    from app.core.config import settings

    settings.ISAACLAB_ROOT = str(root)
    settings.ISAACLAB_RUNTIME_ENABLED = True
    from app.services.isaac_lab.isaac_runtime_service import get_runtime_status

    status = get_runtime_status()
    assert status["scriptedExpertAvailable"] is True
    assert "scriptedExpertReady" in status
    assert "scriptedExpertIssueCodes" in status


def test_get_generate_job_status_enriches_episode_fields_from_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _configure_runtime(tmp_path, monkeypatch)
    job_id = "isaac_gen_20260617_120000_ab12"
    job_root = tmp_path / "jobs" / job_id
    job_root.mkdir(parents=True)
    meta_dir = job_root / "metadata"
    meta_dir.mkdir()
    (meta_dir / "request.json").write_text(
        '{"numDemos": 10, "seed": 0, "enableCameras": true, "generationMode": "mimic_auto"}',
        encoding="utf-8",
    )
    (job_root / "status.json").write_text(
        '{"jobId": "isaac_gen_20260617_120000_ab12", "status": "running", "phase": "generate"}',
        encoding="utf-8",
    )
    artifacts = job_root / "artifacts"
    artifacts.mkdir()
    (artifacts / "generate.stdout.log").write_text(
        "7/23 (30.4%) successful demos generated by mimic\n",
        encoding="utf-8",
    )

    status = generate_svc.get_generate_job_status(job_id)

    assert status["numDemos"] == 10
    assert status["totalEpisodes"] == 10
    assert status["successfulEpisodes"] == 7
    assert status["completedEpisodes"] == 7
    assert status["currentEpisode"] == 7
    assert status["seed"] == 0
    assert isinstance(status["progress"], int)
    assert 25 <= status["progress"] <= 80


def test_finalize_status_merges_existing_fields(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _configure_runtime(tmp_path, monkeypatch)
    from app.services.isaac_lab.isaac_job_utils import finalize_status

    job_id = "isaac_gen_20260617_120001_ab12"
    job_root = tmp_path / "jobs" / job_id
    job_root.mkdir(parents=True)
    finalize_status(job_id, {"jobId": job_id, "status": "queued", "numDemos": 10, "datasetName": "demo"})
    finalize_status(job_id, {"status": "running", "phase": "annotate", "message": "Running annotate_demos.py…"})

    payload = isaac_job_status_path(job_id).read_text(encoding="utf-8")
    assert '"numDemos": 10' in payload
    assert '"phase": "annotate"' in payload
    assert '"datasetName": "demo"' in payload
