"""Historical dual-arm torch-BC checkpoint loader snapshot."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

DEFAULT_OBS_KEYS = [
    "left_arm_joint_pos",
    "right_arm_joint_pos",
    "left_arm_joint_vel",
    "right_arm_joint_vel",
    "cable_state",
]

REQUIRED_MANIFEST_FIELDS = (
    "observationSchema",
    "actionSchema",
    "actionDim",
)


@dataclass
class TorchBcPolicySpec:
    checkpoint_path: Path
    obs_dim: int
    action_dim: int
    obs_keys: list[str]
    action_semantics: str
    backend_type: str
    low_dim_only: bool
    task_type: str
    task_template_id: str
    model_asset_id: Optional[str] = None


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _find_sidecar_manifest(checkpoint_path: Path) -> dict[str, Any]:
    candidates = [
        checkpoint_path.parent / "training_manifest.json",
        checkpoint_path.parent.parent / "training_manifest.json",
        checkpoint_path.parent / "metrics.json",
    ]
    train_job_root = checkpoint_path
    for _ in range(6):
        train_job_root = train_job_root.parent
        manifest = train_job_root / "artifacts" / "model_manifest.json"
        if manifest.is_file():
            return _read_json(manifest)
    for path in candidates:
        if path.is_file():
            return _read_json(path)
    return {}


def _validate_dual_arm_manifest(manifest: dict[str, Any], checkpoint_path: Path) -> None:
    task_type = str(manifest.get("taskType") or "")
    template_id = str(manifest.get("taskTemplateId") or "")
    if task_type and task_type != "dual_arm_cable_manipulation":
        raise ValueError(
            f"checkpoint taskType={task_type!r} is not dual_arm_cable_manipulation: {checkpoint_path}"
        )
    if template_id and template_id not in {
        "dual_arm_cable_manipulation",
        "task_dual_arm_cable_manipulation_v1",
    }:
        raise ValueError(
            f"checkpoint taskTemplateId={template_id!r} is not a dual-arm cable model: {checkpoint_path}"
        )
    backend = str(manifest.get("backendType") or manifest.get("trainingBackend") or manifest.get("backend") or "")
    if backend and backend not in {"torch_bc"}:
        raise ValueError(f"unsupported backendType={backend!r} for dual-arm torch_bc loader")
    obs_schema = str(manifest.get("observationSchema") or "")
    if obs_schema and obs_schema != "dual_arm_cable_il_v1":
        raise ValueError(f"unsupported observationSchema={obs_schema!r}")
    action_dim = manifest.get("actionDim")
    if action_dim is not None and int(action_dim) != 14:
        raise ValueError(f"expected actionDim=14, got {action_dim}")


def load_torch_bc_policy_spec(
    checkpoint_path: str | Path,
    *,
    model_manifest: Optional[dict[str, Any]] = None,
) -> TorchBcPolicySpec:
    ckpt = Path(checkpoint_path).expanduser().resolve()
    if not ckpt.is_file() or ckpt.stat().st_size <= 0:
        raise FileNotFoundError(f"checkpoint not found or empty: {ckpt}")

    sidecar = _find_sidecar_manifest(ckpt)
    manifest = dict(sidecar)
    if model_manifest:
        manifest.update(model_manifest)

    import torch

    payload = torch.load(ckpt, map_location="cpu")
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError(f"invalid torch_bc checkpoint format: {ckpt}")

    backend = str(payload.get("backend") or manifest.get("backendType") or "torch_bc")
    if backend != "torch_bc":
        raise ValueError(f"checkpoint backend={backend!r} is not torch_bc")

    obs_dim = int(payload.get("obs_dim") or manifest.get("obsDim") or 0)
    action_dim = int(payload.get("action_dim") or manifest.get("actionDim") or 0)
    if obs_dim <= 0 or action_dim <= 0:
        raise ValueError(f"checkpoint missing obs_dim/action_dim: {ckpt}")

    obs_keys = list(payload.get("obs_keys") or DEFAULT_OBS_KEYS)
    if obs_keys != DEFAULT_OBS_KEYS:
        raise ValueError(f"unsupported obs_keys in checkpoint: {obs_keys}")

    _validate_dual_arm_manifest(manifest, ckpt)

    return TorchBcPolicySpec(
        checkpoint_path=ckpt,
        obs_dim=obs_dim,
        action_dim=action_dim,
        obs_keys=obs_keys,
        action_semantics=str(
            manifest.get("actionSemantics") or "recorded_joint_position_targets"
        ),
        backend_type="torch_bc",
        low_dim_only=True,
        task_type=str(manifest.get("taskType") or "dual_arm_cable_manipulation"),
        task_template_id=str(manifest.get("taskTemplateId") or "dual_arm_cable_manipulation"),
        model_asset_id=manifest.get("modelAssetId"),
    )


class TorchBcPolicy:
    def __init__(self, spec: TorchBcPolicySpec, *, device: str = "cpu") -> None:
        import torch
        import torch.nn as nn

        self.spec = spec
        self.device = torch.device(
            "cuda" if device != "cpu" and torch.cuda.is_available() else "cpu"
        )
        self.model = nn.Sequential(
            nn.Linear(spec.obs_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, spec.action_dim),
        ).to(self.device)
        payload = torch.load(spec.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()

    @staticmethod
    def vectorize_observation(qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
        qpos = np.asarray(qpos, dtype=np.float32).reshape(-1)
        qvel = np.asarray(qvel, dtype=np.float32).reshape(-1)
        left_pos = qpos[:7] if qpos.shape[0] >= 7 else np.zeros(7, dtype=np.float32)
        right_pos = qpos[7:14] if qpos.shape[0] >= 14 else np.zeros(7, dtype=np.float32)
        left_vel = qvel[:7] if qvel.shape[0] >= 7 else np.zeros(7, dtype=np.float32)
        right_vel = qvel[7:14] if qvel.shape[0] >= 14 else np.zeros(7, dtype=np.float32)
        cable = qpos[14:].copy() if qpos.shape[0] > 14 else np.zeros(0, dtype=np.float32)
        return np.concatenate([left_pos, right_pos, left_vel, right_vel, cable], axis=0)

    def predict(self, obs_vector: np.ndarray) -> np.ndarray:
        import torch

        obs = np.asarray(obs_vector, dtype=np.float32).reshape(-1)
        if obs.shape[0] != self.spec.obs_dim:
            raise ValueError(
                f"obs dim mismatch: expected {self.spec.obs_dim}, got {obs.shape[0]}"
            )
        with torch.no_grad():
            tensor = torch.from_numpy(obs).unsqueeze(0).to(self.device)
            action = self.model(tensor).cpu().numpy()[0]
        return np.asarray(action, dtype=np.float32)
