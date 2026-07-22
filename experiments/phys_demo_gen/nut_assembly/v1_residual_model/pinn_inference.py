"""V1-D：PINN 推理与 sim-in-loop 打分辅助。"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

_V1_DIR = Path(__file__).resolve().parent
_EXPERIMENT_DIR = _V1_DIR.parent
if str(_EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENT_DIR))
if str(_V1_DIR) not in sys.path:
    sys.path.insert(0, str(_V1_DIR))

from pinn_residual_energy_model import PINNResidualEnergyModel

DEFAULT_MODEL = _EXPERIMENT_DIR / "outputs" / "v1_residual_model_pinn" / "model.pt"
DEFAULT_FAILED_HDF5 = _EXPERIMENT_DIR.parents[2] / "mnt" / "data" / "demo_failed.hdf5"

_SCORING_CTX: dict[str, Any] = {}
_PINN_MODEL: PINNResidualEnergyModel | None = None
_PINN_DEVICE: torch.device | None = None
_EXT_CACHE: dict[str, dict[str, float]] = {}


def set_pinn_scoring_context(
    *,
    demo_key: str,
    hdf5_path: str | Path,
    stage: str = "insertion",
    model_path: str | Path | None = None,
) -> None:
    global _SCORING_CTX
    _SCORING_CTX = {
        "demo_key": demo_key,
        "hdf5_path": str(hdf5_path),
        "stage": stage,
        "model_path": str(model_path or DEFAULT_MODEL),
    }


def clear_pinn_scoring_context() -> None:
    global _SCORING_CTX
    _SCORING_CTX = {}


def clear_pinn_model_cache() -> None:
    global _PINN_MODEL, _PINN_DEVICE
    _PINN_MODEL = None
    _PINN_DEVICE = None


def load_pinn_model(model_path: Path | None = None) -> PINNResidualEnergyModel:
    global _PINN_MODEL, _PINN_DEVICE
    path = Path(model_path or _SCORING_CTX.get("model_path") or DEFAULT_MODEL)
    if _PINN_MODEL is not None:
        return _PINN_MODEL
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    _PINN_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _PINN_MODEL = PINNResidualEnergyModel(
        input_dim=ckpt["input_dim"],
        hidden_dim=ckpt["hidden_dim"],
        num_layers=ckpt["num_layers"],
        dropout=ckpt.get("dropout", 0.1),
    )
    _PINN_MODEL.load_state_dict(ckpt["state_dict"])
    _PINN_MODEL.to(_PINN_DEVICE)
    _PINN_MODEL.eval()
    return _PINN_MODEL


def _extended_features(hdf5_path: str, demo_key: str) -> dict[str, float]:
    key = f"{hdf5_path}:{demo_key}"
    if key not in _EXT_CACHE:
        from build_training_dataset import _extended_hdf5_features

        _EXT_CACHE[key] = _extended_hdf5_features(hdf5_path, demo_key)
    return _EXT_CACHE[key]


def rollout_row_to_features(
    row: dict[str, Any],
    *,
    demo_key: str,
    hdf5_path: str,
    stage: str = "insertion",
) -> np.ndarray:
    from build_training_dataset import (
        _sample_from_grasp_rollout,
        _sample_from_sim_rollout,
    )

    ext = _extended_features(hdf5_path, demo_key)
    payload = dict(row)
    if stage == "grasp":
        if "grasp_params" not in payload and any(k.startswith("grasp_") for k in payload):
            pass
        sample = _sample_from_grasp_rollout(
            payload,
            dataset_version="v1c",
            source="pinn_inference",
            demo_key=demo_key,
            hdf5_path=hdf5_path,
            ext_base=ext,
        )
    else:
        if stage == "transport" and "transport_params" not in payload:
            tp = {}
            for k in (
                "transport_xy_gain",
                "transport_xy_offset_scale",
                "pre_align_height",
                "lift_height",
                "approach_steps",
                "transport_steps",
                "transport_hold_steps",
                "gripper_close_shift",
                "speed_scale",
            ):
                csv_k = f"transport_{k}"
                if csv_k in payload:
                    tp[k] = payload[csv_k]
            if tp:
                payload["transport_params"] = tp
        sample = _sample_from_sim_rollout(
            payload,
            dataset_version="v1c",
            source="pinn_inference",
            demo_key=demo_key,
            hdf5_path=hdf5_path,
            ext_base=ext,
        )
    return sample["features"]


@torch.no_grad()
def predict_pinn_outputs(
    features: np.ndarray,
    *,
    model_path: Path | None = None,
) -> dict[str, float]:
    model = load_pinn_model(model_path)
    x = torch.from_numpy(features.astype(np.float32)).unsqueeze(0).to(_PINN_DEVICE)
    out = model(x)
    return {
        "pinn_E_total": float(out["E_total"].item()),
        "pinn_success_prob": float(torch.sigmoid(out["success_logit"]).item()),
        "pinn_grasp_success_prob": float(torch.sigmoid(out["grasp_success_logit"]).item()),
        "pinn_lift_success_prob": float(torch.sigmoid(out["lift_success_logit"]).item()),
    }


def score_rollout_with_pinn(
    result: dict[str, Any],
    *,
    demo_key: str | None = None,
    hdf5_path: str | None = None,
    stage: str | None = None,
    model_path: Path | None = None,
) -> float:
    ctx = _SCORING_CTX
    demo_key = demo_key or result.get("demo_name") or ctx.get("demo_key")
    hdf5_path = hdf5_path or result.get("source_file") or ctx.get("hdf5_path")
    stage = stage or ctx.get("stage", "insertion")
    if not demo_key or not hdf5_path:
        raise ValueError("demo_key and hdf5_path required for PINN scoring")

    features = rollout_row_to_features(result, demo_key=demo_key, hdf5_path=hdf5_path, stage=stage)
    preds = predict_pinn_outputs(features, model_path=model_path)
    result["pinn_E_total_pred"] = preds["pinn_E_total"]
    result["pinn_success_prob"] = preds["pinn_success_prob"]
    result["pinn_grasp_success_prob"] = preds["pinn_grasp_success_prob"]
    result["pinn_lift_success_prob"] = preds["pinn_lift_success_prob"]
    return preds["pinn_E_total"]


def explicit_energy_score(result: dict[str, Any]) -> float:
    """Explicit weighted E_total_norm (same as energy_full search)."""
    xy = float(result.get("E_xy_norm", 0.0))
    transport = float(result.get("E_transport_norm", 0.0))
    yaw = float(result.get("E_yaw_norm", 0.0))
    z = float(result.get("E_z_norm", 0.0))
    smooth = float(result.get("E_smooth_norm", 0.0))
    return 3 * xy + 3 * transport + 2 * yaw + 2 * z + 0.2 * smooth
