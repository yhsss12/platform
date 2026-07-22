"""V1-E：Repair-parameter field 推理辅助。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

import sys

_V1E_DIR = Path(__file__).resolve().parent
_V1_DIR = _V1E_DIR.parent
_EXPERIMENT_DIR = _V1_DIR.parent
if str(_V1E_DIR) not in sys.path:
    sys.path.insert(0, str(_V1E_DIR))

from pinn_repair_parameter_model import PINNRepairParameterModel, explicit_repair_energy  # noqa: E402
from repair_dataset import (  # noqa: E402
    build_input_vector,
    build_param_mask,
    build_theta_vector,
    extract_failed_context,
)

DEFAULT_MODEL = _EXPERIMENT_DIR / "outputs" / "v1_repair_parameter_model" / "model.pt"

_MODEL: PINNRepairParameterModel | None = None
_DEVICE: torch.device | None = None


def clear_repair_model_cache() -> None:
    global _MODEL, _DEVICE
    _MODEL = None
    _DEVICE = None


def load_repair_model(model_path: Path | None = None) -> PINNRepairParameterModel:
    global _MODEL, _DEVICE
    if _MODEL is not None:
        return _MODEL
    path = Path(model_path or DEFAULT_MODEL)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _MODEL = PINNRepairParameterModel(
        input_dim=ckpt["input_dim"],
        hidden_dim=ckpt["hidden_dim"],
        num_layers=ckpt["num_layers"],
        dropout=ckpt.get("dropout", 0.1),
    )
    _MODEL.load_state_dict(ckpt["state_dict"])
    _MODEL.to(_DEVICE)
    _MODEL.eval()
    return _MODEL


def build_features_from_repair_spec(
    *,
    context: dict[str, Any],
    insertion: dict[str, float] | None = None,
    transport: dict[str, float] | None = None,
    grasp_lift: dict[str, float] | None = None,
    active: str,
) -> np.ndarray:
    theta = build_theta_vector(insertion=insertion, transport=transport, grasp_lift=grasp_lift)
    mask = build_param_mask(active=active)
    return build_input_vector(context, theta, mask)


@torch.no_grad()
def score_repair_candidate(
    features: np.ndarray,
    *,
    model_path: Path | None = None,
) -> dict[str, float]:
    model = load_repair_model(model_path)
    x = torch.from_numpy(features.astype(np.float32)).unsqueeze(0).to(_DEVICE)
    out = model(x)
    explicit = explicit_repair_energy(x)
    return {
        "pinn_E_total": float(out["E_total"].item()),
        "explicit_E_total": float(explicit.item()),
        "pinn_success_prob": float(torch.sigmoid(out["success_logit"]).item()),
        "pinn_grasp_success_prob": float(torch.sigmoid(out["grasp_success_logit"]).item()),
        "pinn_lift_success_prob": float(torch.sigmoid(out["lift_success_logit"]).item()),
    }


def score_repair_spec(
    *,
    context: dict[str, Any],
    insertion: dict[str, float] | None = None,
    transport: dict[str, float] | None = None,
    grasp_lift: dict[str, float] | None = None,
    active: str,
    model_path: Path | None = None,
) -> dict[str, float]:
    features = build_features_from_repair_spec(
        context=context,
        insertion=insertion,
        transport=transport,
        grasp_lift=grasp_lift,
        active=active,
    )
    return score_repair_candidate(features, model_path=model_path)


def context_from_original(original: dict[str, Any], *, demo_key: str, failure_type: str) -> dict[str, Any]:
    return extract_failed_context(original, demo_key=demo_key, failure_type=failure_type)
