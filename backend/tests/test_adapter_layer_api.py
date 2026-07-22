from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import routes_adapter_layer
from app.core.deps import get_current_user


def _build_test_app() -> FastAPI:
    app = FastAPI()
    app.include_router(routes_adapter_layer.router, prefix="/api/adapter")

    async def _fake_user():
        return SimpleNamespace(id="test-user", role="SUPER_ADMIN", is_active=True)

    app.dependency_overrides[get_current_user] = _fake_user
    return app


@pytest.fixture
def client() -> TestClient:
    return TestClient(_build_test_app())


MUJOCO_MANIFEST = {
    "datasetId": "ds_api_test",
    "datasetName": "API 测试数据集",
    "taskName": "线缆穿杆",
    "taskType": "cable_threading",
    "backend": "mujoco",
    "robotType": "Panda",
    "dataFormat": "HDF5",
    "observationSpace": {
        "type": "low_dim",
        "keys": ["robot0_eef_pos", "robot0_eef_quat"],
    },
    "actionSpace": {"type": "continuous", "dim": 7},
    "episodes": 8,
    "successfulEpisodes": 8,
    "artifacts": {"hdf5": "/tmp/api_dataset.hdf5"},
}


def test_api_analyze_compatibility(client: TestClient):
    response = client.post("/api/adapter/compatibility/analyze", json={"datasetManifest": MUJOCO_MANIFEST})
    assert response.status_code == 200
    body = response.json()
    assert body["datasetId"] == "ds_api_test"
    assert "robomimic_bc" in body["recommendedModels"]
    robomimic = next(item for item in body["results"] if item["modelType"] == "robomimic_bc")
    assert robomimic["compatible"] is True


def test_api_training_plan(client: TestClient):
    response = client.post(
        "/api/adapter/training-plan",
        json={"datasetManifest": MUJOCO_MANIFEST, "modelType": "robomimic_bc"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["epochs"] == 5
    assert body["batchSize"] == 16
    assert body["learningRate"] == 0.0001
    assert body["advancedConfig"]


def test_api_training_plan_incompatible_returns_422(client: TestClient):
    response = client.post(
        "/api/adapter/training-plan",
        json={"datasetManifest": MUJOCO_MANIFEST, "modelType": "torch_bc"},
    )
    assert response.status_code == 422
    assert "不兼容" in response.json()["detail"]


def test_api_evaluation_plan(client: TestClient):
    response = client.post(
        "/api/adapter/evaluation-plan",
        json={
            "modelAssetOrTrainingPlan": {
                "datasetId": "ds_api_test",
                "modelType": "robomimic_bc",
                "trainingBackend": "robomimic_bc",
                "epochs": 5,
                "taskName": "线缆穿杆",
                "simulator": "mujoco",
                "robotType": "Panda",
            }
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["taskTemplateId"] == "cable_threading_single_arm"
    assert body["numEpisodes"] == 10
    assert body["metrics"]


def test_api_get_dataset_compatibility_not_found(client: TestClient):
    with patch(
        "app.services.adapter_layer.adapter_service.resolve_manifest_by_dataset_id",
        return_value=None,
    ):
        response = client.get("/api/adapter/datasets/ds_missing/compatibility")
    assert response.status_code == 404


def test_api_get_dataset_compatibility_found(client: TestClient):
    with patch(
        "app.services.adapter_layer.adapter_service.resolve_manifest_by_dataset_id",
        return_value=MUJOCO_MANIFEST,
    ):
        response = client.get("/api/adapter/datasets/ds_api_test/compatibility")
    assert response.status_code == 200
    assert response.json()["datasetId"] == "ds_api_test"


def test_api_training_adaptation_plan(client: TestClient, tmp_path):
    h5py = pytest.importorskip("h5py")
    hdf5 = tmp_path / "api_dataset.hdf5"
    with h5py.File(hdf5, "w") as handle:
        data = handle.create_group("data")
        demo = data.create_group("demo_0")
        obs = demo.create_group("obs")
        obs.create_dataset("robot0_eef_pos", data=[[0.0] * 3])
        demo.create_dataset("actions", data=[[0.0] * 7] * 10)

    manifest = {
        **MUJOCO_MANIFEST,
        "artifacts": {"hdf5": str(hdf5)},
    }
    with patch(
        "app.services.adapter_layer.adapter_service.resolve_manifest_by_dataset_id",
        return_value=manifest,
    ):
        response = client.post(
            "/api/adapter/training-adaptation-plan",
            json={
                "datasetId": "ds_api_test",
                "modelType": "robomimic_bc",
                "overrides": {"trainingConfig": {"epochs": 12}},
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert body["datasetProfile"]["actionDim"] == 7
    assert body["modelAdaptation"]["modelType"] == "robomimic_bc"
    assert body["validation"]["adaptable"] is True
    assert body["modelAdaptation"]["trainingConfig"]["epochs"] == 12
    assert body["configPatch"]["epochs"] == 12
    assert body["adapterLayerVersion"] == "2.0"


def test_api_training_adaptation_plan_act_not_adaptable(client: TestClient, tmp_path):
    h5py = pytest.importorskip("h5py")
    hdf5 = tmp_path / "api_lowdim.hdf5"
    with h5py.File(hdf5, "w") as handle:
        data = handle.create_group("data")
        demo = data.create_group("demo_0")
        obs = demo.create_group("obs")
        obs.create_dataset("robot0_eef_pos", data=[[0.0] * 3])
        demo.create_dataset("actions", data=[[0.0] * 7] * 5)

    manifest = {**MUJOCO_MANIFEST, "artifacts": {"hdf5": str(hdf5)}}
    with patch(
        "app.services.adapter_layer.adapter_service.resolve_manifest_by_dataset_id",
        return_value=manifest,
    ):
        response = client.post(
            "/api/adapter/training-adaptation-plan",
            json={"datasetId": "ds_api_test", "modelType": "act"},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["validation"]["adaptable"] is False
    assert body["configPatch"]["_adaptationBlocked"] is True

