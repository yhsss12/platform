from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.services import training_node_service as node_svc


def test_resolve_training_node_l20():
    cfg = node_svc.resolve_training_node(training_node_id="l20-172-18-0-73")
    assert cfg is not None
    assert cfg.node_id == "l20-172-18-0-73"
    assert cfg.execution_mode == "remote_ssh"
    assert "·" in cfg.device_label
    assert "172.18.0.73" in cfg.device_label


def test_resolve_training_node_h20_placeholder():
    cfg = node_svc.resolve_training_node(training_node_id="h20-local-placeholder")
    assert cfg is not None
    assert cfg.execution_mode == "local"
    assert "L20" in cfg.device_label
    assert "H20" not in cfg.device_label


def test_list_training_nodes_deduplicates():
    nodes = node_svc.list_training_nodes(refresh=True)
    node_ids = [item["nodeId"] for item in nodes]
    assert len(node_ids) == len(set(node_ids))
    assert "l20-172-18-0-73" in node_ids
    assert "h20-local-placeholder" in node_ids


def test_probe_l20_unreachable_without_credentials(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TRAIN_NODE_L20_PASSWORD", "")
    monkeypatch.setenv("TRAIN_NODE_L20_SSH_KEY", "")
    node_svc.invalidate_training_node_probe_cache()
    probe = node_svc.probe_training_node("l20-172-18-0-73", refresh=True)
    assert probe["status"] == "unreachable"
    assert "TRAIN_NODE_L20_PASSWORD" in (probe.get("message") or "")
    assert "·" in probe["deviceLabel"]
    assert "172.18.0.73" in probe["deviceLabel"]


def test_probe_h20_placeholder_local():
    probe = node_svc.probe_training_node("h20-local-placeholder", refresh=True)
    assert probe["status"] in {"available", "placeholder"}
    assert probe["selectable"] is True
    assert "L20" in probe["deviceLabel"]
    assert "H20" not in probe["deviceLabel"]


def test_validate_remote_node_misconfigured_workdir(monkeypatch: pytest.MonkeyPatch):
    cfg = node_svc.resolve_training_node(training_node_id="l20-172-18-0-73")
    assert cfg is not None

    fake_probe = {
        "status": "misconfigured",
        "message": "远端平台工作目录不存在，请先同步项目代码到 /missing/workdir",
    }
    with patch.object(node_svc, "probe_training_node", return_value=fake_probe):
        with pytest.raises(ValueError, match="远端平台工作目录不存在"):
            node_svc.validate_remote_node_for_job(cfg)


def test_format_training_node_display_name():
    assert node_svc.format_training_node_display_name("L20", "172.18.0.73") == "L20 · 172.18.0.73"


def test_resolve_training_node_display_name_legacy_h20():
    display = node_svc.resolve_training_node_display_name(device_label="H20")
    assert "L20" in display
    assert "H20" not in display
    assert "·" in display


def test_gpu_busy_reason():
    assert node_svc._gpu_busy_reason({"memoryUsedMb": 40000, "memoryUsedRatio": 0.9}).startswith("GPU 显存占用")
    assert node_svc._gpu_busy_reason({"memoryUsedMb": 100, "memoryUsedRatio": 0.01}) == ""


def test_create_training_job_rejects_misconfigured_remote_node(tmp_path, monkeypatch: pytest.MonkeyPatch):
    from app.services import training_service as svc

    monkeypatch.setattr(svc, "TRAINING_ROOT", tmp_path / "training")
    monkeypatch.setattr(svc, "ALLOWED_PATH_ROOTS", [tmp_path.resolve()])

    h5py = pytest.importorskip("h5py")
    hdf5 = tmp_path / "sample.hdf5"
    with h5py.File(hdf5, "w") as handle:
        data = handle.create_group("data")
        demo = data.create_group("demo_0")
        obs = demo.create_group("obs")
        obs.create_dataset("agentview_image", data=[[[[0]]]] * 4)
        obs.create_dataset("robot0_eef_pos", data=[[0.0, 0.0, 0.0]] * 4)
        demo.create_dataset("actions", data=[[0.0] * 7] * 4)

    manifest = {
        "datasetId": "ds1",
        "datasetName": "sample",
        "successfulEpisodes": 1,
        "artifacts": {"hdf5": str(hdf5)},
        "taskType": "cable_threading",
    }

    fake_cfg = node_svc.TrainingNodeConfig(
        node_id="l20-172-18-0-73",
        label="L20",
        device_label="L20",
        execution_mode="remote_ssh",
        host="127.0.0.1",
        ssh_user="zyf",
        workdir="/missing/workdir",
    )

    with patch.object(node_svc, "resolve_training_node", return_value=fake_cfg):
        with patch.object(node_svc, "validate_remote_node_for_job", side_effect=ValueError("训练节点未配置平台工作目录")):
            with pytest.raises(Exception) as exc:
                svc.create_training_job(
                    {
                        "datasetId": "ds1",
                        "datasetManifest": manifest,
                        "downstreamModelType": "DiffusionPolicy",
                        "trainingBackend": "diffusion_policy",
                        "trainingNodeId": "l20-172-18-0-73",
                        "epochs": 1,
                    }
                )
    assert "训练节点未配置平台工作目录" in str(exc.value)


def test_create_training_job_local_without_training_node_id(tmp_path, monkeypatch: pytest.MonkeyPatch):
    from app.services import training_service as svc

    monkeypatch.setattr(svc, "TRAINING_ROOT", tmp_path / "training")
    monkeypatch.setattr(svc, "ALLOWED_PATH_ROOTS", [tmp_path.resolve()])

    h5py = pytest.importorskip("h5py")
    hdf5 = tmp_path / "sample.hdf5"
    with h5py.File(hdf5, "w") as handle:
        data = handle.create_group("data")
        demo = data.create_group("demo_0")
        obs = demo.create_group("obs")
        obs.create_dataset("agentview_image", data=[[[[0]]]] * 4)
        obs.create_dataset("robot0_eef_pos", data=[[0.0, 0.0, 0.0]] * 4)
        demo.create_dataset("actions", data=[[0.0] * 7] * 4)

    manifest = {
        "datasetId": "ds1",
        "datasetName": "sample",
        "successfulEpisodes": 1,
        "artifacts": {"hdf5": str(hdf5)},
        "taskType": "cable_threading",
    }

    with patch.object(svc.threading.Thread, "start", MagicMock()):
        result = svc.create_training_job(
            {
                "datasetId": "ds1",
                "datasetManifest": manifest,
                "downstreamModelType": "DiffusionPolicy",
                "trainingBackend": "diffusion_policy",
                "deviceLabel": "L20",
                "epochs": 1,
            }
        )
    assert result["trainJobId"]
    train_config = svc._read_json(svc._train_job_dir(result["trainJobId"]) / "config" / "train_config.json")
    assert train_config.get("executionMode") == "local"
    assert train_config.get("trainingNodeId") is None
