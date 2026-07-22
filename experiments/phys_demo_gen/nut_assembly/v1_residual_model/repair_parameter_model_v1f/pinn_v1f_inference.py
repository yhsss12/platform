"""V1-F：Uncertainty-aware repair-parameter field 推理辅助。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

import sys

_V1F_DIR = Path(__file__).resolve().parent
_V1_DIR = _V1F_DIR.parent
_EXPERIMENT_DIR = _V1_DIR.parent
if str(_V1F_DIR) not in sys.path:
    sys.path.insert(0, str(_V1F_DIR))

from pinn_v1f_repair_model import PINNV1FRepairModel, explicit_v1f_repair_energy  # noqa: E402
from v1f_repair_dataset import (  # noqa: E402
    build_input_vector_v1f,
    build_param_mask_v1f,
    build_theta_vector_v1f,
)

DEFAULT_MODEL = _EXPERIMENT_DIR / "outputs" / "v1f_repair_parameter_model" / "model_v1f.pt"

_MODEL: PINNV1FRepairModel | None = None
_DEVICE: torch.device | None = None


def clear_v1f_model_cache() -> None:
    global _MODEL, _DEVICE
    _MODEL = None
    _DEVICE = None


def _infer_v1f_architecture(ckpt: dict[str, Any]) -> tuple[int, int, float]:
    hidden_dim = ckpt.get("hidden_dim")
    num_layers = ckpt.get("num_layers")
    dropout = float(ckpt.get("dropout", 0.1))
    if hidden_dim is None or num_layers is None:
        backbone_weights = sorted(
            k for k in ckpt["state_dict"] if k.startswith("backbone.") and k.endswith(".weight")
        )
        if backbone_weights:
            hidden_dim = int(ckpt["state_dict"][backbone_weights[0]].shape[0])
            num_layers = len(backbone_weights) + 1
    return int(hidden_dim or 192), int(num_layers or 5), dropout


def load_v1f_repair_model(model_path: Path | None = None) -> PINNV1FRepairModel:
    global _MODEL, _DEVICE
    if _MODEL is not None:
        return _MODEL
    path = Path(model_path or DEFAULT_MODEL)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    hidden_dim, num_layers, dropout = _infer_v1f_architecture(ckpt)
    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _MODEL = PINNV1FRepairModel(
        input_dim=ckpt["input_dim"],
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )
    _MODEL.load_state_dict(ckpt["state_dict"])
    _MODEL.to(_DEVICE)
    _MODEL.eval()
    return _MODEL


def build_v1f_features_from_repair_spec(
    *,
    context: dict[str, Any],
    insertion: dict[str, float] | None = None,
    transport: dict[str, float] | None = None,
    grasp_lift: dict[str, float] | None = None,
    lift_extra: dict[str, float] | None = None,
    active: str,
) -> np.ndarray:
    theta = build_theta_vector_v1f(
        insertion=insertion,
        transport=transport,
        grasp_lift=grasp_lift,
        lift_extra=lift_extra,
    )
    mask = build_param_mask_v1f(active=active)
    return build_input_vector_v1f(context, theta, mask)


@torch.no_grad()
def score_v1f_repair_candidate(
    features: np.ndarray,
    *,
    model_path: Path | None = None,
    uncertainty_penalty: float = 0.0,
) -> dict[str, float]:
    model = load_v1f_repair_model(model_path)
    x = torch.from_numpy(features.astype(np.float32)).unsqueeze(0).to(_DEVICE)
    out = model(x)
    explicit = explicit_v1f_repair_energy(x)
    e_total = float(out["E_total"].item())
    uncertainty = float(out["uncertainty"].item())
    return {
        "v1f_E_total": e_total,
        "v1f_E_total_uncertainty_aware": e_total + uncertainty_penalty * uncertainty,
        "v1f_uncertainty": uncertainty,
        "explicit_E_total": float(explicit.item()),
        "v1f_success_prob": float(torch.sigmoid(out["success_logit"]).item()),
        "v1f_grasp_success_prob": float(torch.sigmoid(out["grasp_success_logit"]).item()),
        "v1f_lift_success_prob": float(torch.sigmoid(out["lift_success_logit"]).item()),
    }
