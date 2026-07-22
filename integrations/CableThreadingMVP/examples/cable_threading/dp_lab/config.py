from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DpLabConfig:
    task_name: str = "cable_threading"
    action_dim: int = 7
    action_key: str = "actions"
    action_mode: str = "osc_pose_delta_eef"
    controller_type: str = "OSC_POSE"
    eval_executor: str = "osc_pose"
    trained_action_mode: str | None = None
    observation_schema: str | None = None
    action_schema: str | None = None
    controller_schema: str | None = None
    side_channel_schema: str | None = None
    preferred_policy_schema_id: str | None = None
    gripper_action_key: str | None = None
    low_dim_keys: list[str] = field(
        default_factory=lambda: [
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
        ]
    )
    image_keys: list[str] = field(
        default_factory=lambda: [
            "agentview_image",
            "robot0_eye_in_hand_image",
        ]
    )

    n_obs_steps: int = 2
    horizon: int = 16
    n_action_steps: int = 8

    num_diffusion_steps: int = 20
    num_inference_steps: int = 20

    batch_size: int = 16
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    seed: int = 42

    image_size: int = 128
    vision_encoder: str = "resnet18"  # resnet18 | tiny_cnn
    use_ema: bool = True
    ema_decay: float = 0.999

    # debug / smoke 限制（None 表示不限制）
    max_train_windows: int | None = None
    max_batches_per_epoch: int | None = None
    low_dim_dim: int | None = None

    @property
    def resolved_low_dim_dim(self) -> int:
        if self.low_dim_dim is not None and self.low_dim_dim > 0:
            return int(self.low_dim_dim)
        return 3 + 4 + 2

    @property
    def num_cameras(self) -> int:
        return len(self.image_keys)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_checkpoint_dict(self) -> dict[str, Any]:
        """Serialize config for checkpoint/train_config with resolved low_dim_dim."""
        data = self.to_dict()
        data["low_dim_dim"] = self.resolved_low_dim_dim
        return data

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DpLabConfig":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"invalid config yaml: {path}")
        fields = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in fields}
        return cls(**filtered)
