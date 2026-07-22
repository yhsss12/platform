from __future__ import annotations

from app.services.training_service import _manifest_has_pi0_ready_lerobot, _resolve_training_backend


def _lerobot_manifest() -> dict:
    return {
        "sourceJobId": "ct_gen_20260630_120927_1153",
        "successfulEpisodes": 1,
        "availableFormats": ["lerobot"],
        "primaryFormat": "lerobot",
        "lerobot": {
            "status": "ready",
            "path": "datasets/lerobot_dataset",
            "taskInstruction": "thread the cable through the pole",
            "robot": "Panda",
            "stateDim": 9,
            "actionDim": 8,
            "pi0Ready": True,
        },
    }


def test_manifest_has_pi0_ready_lerobot():
    assert _manifest_has_pi0_ready_lerobot(_lerobot_manifest()) is True


def test_pi0_backend_allows_lerobot_without_hdf5():
    backend, message = _resolve_training_backend(
        downstream_model_type="pi0",
        training_backend="pi0",
        has_hdf5=False,
        capabilities={"supportedTrainingBackends": ["pi0"]},
        manifest=_lerobot_manifest(),
    )
    assert backend == "pi0"
    assert message == ""


def test_dp_backend_still_requires_hdf5():
    backend, message = _resolve_training_backend(
        downstream_model_type="Diffusion Policy",
        training_backend="diffusion_policy",
        has_hdf5=False,
        capabilities={"supportedTrainingBackends": ["diffusion_policy"]},
        manifest=_lerobot_manifest(),
    )
    assert backend is None
    assert "HDF5" in message
