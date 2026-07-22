from __future__ import annotations

from pathlib import Path

import pytest

from app.services.isaac_lab import isaac_replay_context_service as ctx_svc
from app.services.isaac_lab import isaac_dataset_service as dataset_svc


def test_video_source_label():
    assert "replay.mp4" in ctx_svc.video_source_label("replay")
    assert "preview.mp4" in ctx_svc.video_source_label("preview")
    assert "转码" in ctx_svc.video_source_label("converted", transcoded=True)


def test_resolve_dataset_playback_prefers_preview_when_no_replay_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from app.services.isaac_lab import job_paths as job_paths_mod

    registry = tmp_path / "registry.json"
    jobs_root = tmp_path / "jobs"
    monkeypatch.setattr(dataset_svc, "ISAAC_DATASET_REGISTRY_PATH", registry)
    monkeypatch.setattr(job_paths_mod, "_output_root", lambda: jobs_root)

    gen_id = "isaac_gen_20260615_120000_abcd"
    gen_root = jobs_root / gen_id / "artifacts"
    gen_root.mkdir(parents=True)
    hdf5 = gen_root / "dataset.hdf5"
    hdf5.write_bytes(b"hdf5")
    preview = gen_root / "preview.mp4"
    preview.write_bytes(b"ftyp" + b"\x00" * 20 + b"moov" + b"avc1" + b"mdat")
    (jobs_root / gen_id / "status.json").write_text(
        f'{{"jobId":"{gen_id}","status":"completed"}}',
        encoding="utf-8",
    )

    row = dataset_svc.register_generated_dataset(
        job_id=gen_id,
        dataset_name="Stack",
        dataset_file=hdf5,
        episode_count=1,
    )

    payload = ctx_svc.resolve_dataset_playback(row["id"])
    assert payload["playback"] is not None
    assert payload["playback"]["videoSourceKind"] == "preview"
    assert payload["usingPreviewFallback"] is True
    assert payload["playback"]["videoJobId"] == gen_id


def test_find_reusable_replay_job_matches_dataset_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from app.services.isaac_lab import job_paths as job_paths_mod

    jobs_root = tmp_path / "jobs"
    monkeypatch.setattr(job_paths_mod, "_output_root", lambda: jobs_root)

    dataset_file = tmp_path / "dataset.hdf5"
    dataset_file.write_bytes(b"hdf5")
    replay_id = "isaac_replay_20260615_120000_ab12"
    replay_root = jobs_root / replay_id
    (replay_root / "metadata").mkdir(parents=True)
    (replay_root / "artifacts").mkdir(parents=True)
    (replay_root / "artifacts" / "replay.mp4").write_bytes(
        b"ftyp" + b"\x00" * 20 + b"moov" + b"avc1" + b"mdat"
    )
    (replay_root / "metadata" / "request.json").write_text(
        f'{{"datasetFile":"{dataset_file}","datasetId":"isaac_ds_test"}}',
        encoding="utf-8",
    )
    (replay_root / "status.json").write_text(
        '{"jobId":"%s","status":"completed"}' % replay_id,
        encoding="utf-8",
    )

    found = ctx_svc.find_reusable_replay_job(dataset_id="isaac_ds_test", dataset_file=dataset_file)
    assert found is not None
    assert found["jobId"] == replay_id
    assert found["videoAvailable"] is True
