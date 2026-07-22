from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ActLabConfig:
    task_name: str = "adapted_task"
    action_dim: int = 7
    action_key: str = "actions"
    action_mode: str = "osc_pose_delta_eef"
    controller_type: str = "OSC_POSE"
    eval_executor: str = "osc_pose"
    trained_action_mode: str | None = None
    gripper_action_key: str | None = None
    preferred_policy_schema_id: str | None = None
    observation_schema: str | None = None
    action_schema: str | None = None
    controller_schema: str | None = None
    side_channel_schema: str | None = None
    state_dim: int = 0
    low_dim_dim: int | None = None
    chunk_size: int = 20

    image_keys: list[str] = field(default_factory=lambda: ["agentview_image"])
    low_dim_keys: list[str] = field(
        default_factory=lambda: [
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
        ]
    )

    hidden_dim: int = 512
    dim_feedforward: int = 2048
    enc_layers: int = 4
    dec_layers: int = 4
    nheads: int = 8
    dropout: float = 0.1
    kl_weight: float = 10.0
    latent_dim: int = 32
    backbone: str = "tiny_cnn"  # tiny_cnn | resnet18

    batch_size: int = 8
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    seed: int = 1
    image_size: int = 128
    val_ratio: float = 0.1

    max_train_samples: int | None = None
    max_batches_per_epoch: int | None = None
    act_variant: str = "image_proprio"

    @property
    def num_cameras(self) -> int:
        return len(self.image_keys)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ActLabConfig":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"invalid config yaml: {path}")
        fields = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in fields}
        return cls(**filtered)
