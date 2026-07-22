import sys
from pathlib import Path

import numpy as np


def test_lerobot_force_torque_features_and_frames():
    relman_dir = Path(__file__).resolve().parents[2] / "scripts" / "relman"
    class _DummyTensor:
        def __init__(self, arr):
            self._arr = np.asarray(arr, dtype=np.float32)

        def float(self):
            return self

        def numpy(self):
            return self._arr

    class _DummyTorchModule:
        @staticmethod
        def from_numpy(arr):
            return _DummyTensor(arr)

        @staticmethod
        def set_num_threads(_n):
            return None

        @staticmethod
        def set_num_interop_threads(_n):
            return None

    sys.modules.setdefault("torch", _DummyTorchModule())
    sys.path.insert(0, str(relman_dir))
    import mcap_to_lerobot  # type: ignore

    mcap_to_lerobot.LEROBOT_AVAILABLE = True

    created = {}

    class DummyDataset:
        def __init__(self, features):
            self.features = features
            self.frames = []

        @classmethod
        def create(cls, **kwargs):
            created["kwargs"] = kwargs
            return cls(kwargs["features"])

        def add_frame(self, frame):
            self.frames.append(frame)

        def save_episode(self):
            return None

    mcap_to_lerobot.LeRobotDataset = DummyDataset
    mcap_to_lerobot.HF_LEROBOT_HOME = Path("/tmp")

    cfg = mcap_to_lerobot.LeRobotConfig(
        repo_id="dummy/repo",
        robot_type="aloha",
        fps=20.0,
        state_sources=[
            {"topic": "/left/joint_states", "field": "data"},
            {"topic": "/left/rm_driver/get_force_data_result", "field": "data"},
        ],
        camera_mapping={},
        action_mode="next_state",
        mode="image",
        use_videos=False,
    )

    ds = mcap_to_lerobot._create_lerobot_dataset(cfg, joint_dim=2, camera_shapes={})
    feats = ds.features

    assert "observation.force.left" in feats and feats["observation.force.left"]["shape"] == (6,)
    assert "observation.force_xyz.left" in feats and feats["observation.force_xyz.left"]["shape"] == (3,)
    assert "observation.torque.left" in feats and feats["observation.torque.left"]["shape"] == (3,)

    aligned = {
        "/left/joint_states": [{"data": [0.1, 0.2]}, {"data": [0.2, 0.3]}],
        "/left/rm_driver/get_force_data_result": [
            {"data": [1, 2, 3, 4, 5, 6], "force": [1, 2, 3], "torque": [4, 5, 6]},
            {"data": [7, 8, 9, 10, 11, 12]},
        ],
    }

    mcap_to_lerobot._write_episode_to_dataset(ds, aligned, num_frames=2, lerobot_config=cfg, camera_shapes={}, mcap_path="/tmp/a.mcap")

    assert len(ds.frames) == 2
    f0 = ds.frames[0]
    assert np.allclose(f0["observation.force.left"].numpy(), np.array([1, 2, 3, 4, 5, 6], dtype=np.float32))
    assert np.allclose(f0["observation.force_xyz.left"].numpy(), np.array([1, 2, 3], dtype=np.float32))
    assert np.allclose(f0["observation.torque.left"].numpy(), np.array([4, 5, 6], dtype=np.float32))

    f1 = ds.frames[1]
    assert np.allclose(f1["observation.force.left"].numpy(), np.array([7, 8, 9, 10, 11, 12], dtype=np.float32))
    assert np.allclose(f1["observation.force_xyz.left"].numpy(), np.array([7, 8, 9], dtype=np.float32))
    assert np.allclose(f1["observation.torque.left"].numpy(), np.array([10, 11, 12], dtype=np.float32))
