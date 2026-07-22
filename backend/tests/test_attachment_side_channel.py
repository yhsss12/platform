"""Attachment side-channel pipeline: HDF5, replay, training config, eval defaults."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

_CABLE_MVP = Path(__file__).resolve().parents[2] / "integrations" / "CableThreadingMVP"
_BACKEND = Path(__file__).resolve().parents[1]
if str(_CABLE_MVP) not in sys.path:
    sys.path.insert(0, str(_CABLE_MVP))
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from examples.cable_threading.hdf5_replay import replay_hdf5_demo  # noqa: E402
from robosuite.utils.dlo.hdf5_dataset import (  # noqa: E402
    save_dataset_hdf5,
    validate_hdf5_trajectory_actions,
)

_DEFAULT_DATASET_DIR = (
    Path(__file__).resolve().parents[2]
    / "runs"
    / "cable_threading"
    / "jobs"
    / "ct_gen_20260618_095819_8aa6"
    / "datasets"
)


def _synthetic_trajectory(steps: int = 6, *, attach_pattern: list[bool] | None = None) -> list[dict]:
    pattern = attach_pattern or [False, True, True, True, False, False][:steps]
    traj = []
    for t in range(steps):
        traj.append(
            {
                "action": np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32),
                "reward": 0.0,
                "done": t == steps - 1,
                "attachment_enabled": bool(pattern[t] if t < len(pattern) else pattern[-1]),
                "raw_obs": {
                    "robot0_eef_pos": np.zeros(3, dtype=np.float32),
                    "robot0_gripper_qpos": np.array([0.04, 0.04], dtype=np.float32),
                },
            }
        )
    return traj


def test_hdf5_writes_attachment_enabled():
    traj = _synthetic_trajectory()
    metadata = {
        "env_name": "CableThreading",
        "grasp_mode": "attachment",
        "attachment_side_channel": True,
        "attachment_field": "attachment_enabled",
        "attachment_policy": "recorded_or_controller",
        "attachment_input_mode": "not_used_by_policy",
        "attachment_control_mode": "eval_controller",
        "include_attachment_obs": False,
        "side_channel_keys": ["attachment_enabled"],
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "attach_test.hdf5"
        save_dataset_hdf5(path, [traj], metadata=metadata)
        validation = validate_hdf5_trajectory_actions(path, [traj])
        assert validation["ok"], validation["issues"]
        with h5py.File(path, "r") as handle:
            data = handle["data"]
            assert data.attrs.get("attachment_side_channel") in (True, np.bool_(True))
            assert str(data.attrs.get("grasp_mode")) == "attachment"
            attach = np.asarray(data["demo_0"]["attachment_enabled"], dtype=bool)
            actions = np.asarray(data["demo_0"]["actions"])
            assert attach.shape[0] == actions.shape[0]
            assert attach.tolist() == [step["attachment_enabled"] for step in traj]


def test_training_config_excludes_attachment_from_action_dim():
    from app.services.adapter_layer.dataset_profiler import DatasetProfile
    from app.services.adapter_layer.model_adaptation_builder import build_model_adaptation_plan

    profile = DatasetProfile(
        datasetId="ds_attach",
        observationKeys=["robot0_eef_pos", "robot0_gripper_qpos", "attachment_enabled"],
        cameraKeys=[],
        observationType="low_dim",
        actionDim=7,
        actionSpace="delta_pose",
        attachmentSideChannel=True,
    )
    for model_type in ("robomimic_bc", "diffusion_policy", "act"):
        plan = build_model_adaptation_plan(profile, model_type)
        assert plan.outputConfig["action_dim"] == 7
        low_dim = plan.inputConfig.get("low_dim_keys") or []
        assert "attachment_enabled" not in low_dim
        assert plan.trainingConfig.get("attachmentInputMode") == "not_used_by_policy"
        assert plan.trainingConfig.get("attachmentSideChannel") is True


def test_eval_default_attachment_mode_is_policy():
    with open(_CABLE_MVP / "run.py", encoding="utf-8") as fh:
        source = fh.read()
    assert '"--attachment-mode"' in source
    assert 'default="policy"' in source
    assert 'choices=["policy", "recorded", "none"]' in source


@pytest.mark.integration
def test_replay_with_attachment_reproduces_success():
    hdf5_path = _DEFAULT_DATASET_DIR / "dataset.hdf5"
    if not hdf5_path.is_file():
        pytest.skip("cable threading HDF5 not present")

    with h5py.File(hdf5_path, "r") as f:
        success_demo = None
        seed = 0
        for demo_key in sorted(k for k in f["data"].keys() if str(k).startswith("demo_")):
            meta_raw = f["data"][demo_key].attrs.get("benchmark_episode_metadata", "{}")
            if isinstance(meta_raw, bytes):
                meta_raw = meta_raw.decode("utf-8")
            meta = json.loads(meta_raw) if meta_raw else {}
            summary = meta.get("summary", meta)
            if bool(summary.get("final_success")):
                success_demo = demo_key
                seed = int(meta.get("seed", 0))
                break
    if success_demo is None:
        pytest.skip("no successful demo in HDF5")

    final_info, _ = replay_hdf5_demo(
        hdf5_path,
        success_demo,
        seed=seed,
        use_recorded_attachment=True,
    )
    assert bool(final_info.get("final_success")) is True


@pytest.mark.integration
def test_replay_without_attachment_degrades_or_fails():
    hdf5_path = _DEFAULT_DATASET_DIR / "dataset.hdf5"
    if not hdf5_path.is_file():
        pytest.skip("cable threading HDF5 not present")

    with h5py.File(hdf5_path, "r") as f:
        demo_key = next(k for k in f["data"].keys() if str(k).startswith("demo_"))
        meta_raw = f["data"][demo_key].attrs.get("benchmark_episode_metadata", "{}")
        if isinstance(meta_raw, bytes):
            meta_raw = meta_raw.decode("utf-8")
        meta = json.loads(meta_raw) if meta_raw else {}
        seed = int(meta.get("seed", 0))
        attach = np.asarray(f["data"][demo_key]["attachment_enabled"], dtype=bool)
    if not np.any(attach):
        pytest.skip("demo has no attachment transitions")

    with_attach, _ = replay_hdf5_demo(
        hdf5_path, demo_key, seed=seed, use_recorded_attachment=True
    )
    without_attach, _ = replay_hdf5_demo(
        hdf5_path, demo_key, seed=seed, use_recorded_attachment=False
    )
    with_score = float(with_attach.get("thread_completion_max", with_attach.get("thread_completion", 0.0)))
    without_score = float(
        without_attach.get("thread_completion_max", without_attach.get("thread_completion", 0.0))
    )
    degraded = (
        bool(with_attach.get("final_success")) and not bool(without_attach.get("final_success"))
    ) or (with_score - without_score > 0.05)
    assert degraded, (
        f"expected degradation without attachment: "
        f"with={with_score:.3f}/{with_attach.get('final_success')} "
        f"without={without_score:.3f}/{without_attach.get('final_success')}"
    )
