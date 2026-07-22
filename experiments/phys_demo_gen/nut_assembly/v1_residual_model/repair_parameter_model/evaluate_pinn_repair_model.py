#!/usr/bin/env python3
"""V1-E：评估 PINN Repair Parameter Model（random split）。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

import sys

_V1E_DIR = Path(__file__).resolve().parent
_V1_DIR = _V1E_DIR.parent
if str(_V1_DIR) not in sys.path:
    sys.path.insert(0, str(_V1_DIR))
if str(_V1E_DIR) not in sys.path:
    sys.path.insert(0, str(_V1E_DIR))

from group_split_utils import top_k_refined_success_rate, top_k_success_rate  # noqa: E402
from pinn_repair_parameter_model import PINNRepairParameterModel, explicit_repair_energy  # noqa: E402
from repair_dataset import load_repair_npz  # noqa: E402
from residual_dataset import FAILURE_TYPES, OUTCOME_TYPES  # noqa: E402

_EXPERIMENT_DIR = _V1_DIR.parent
DEFAULT_DATASET = _EXPERIMENT_DIR / "outputs" / "v1_repair_parameter_model" / "repair_parameter_dataset.npz"
DEFAULT_MODEL = _EXPERIMENT_DIR / "outputs" / "v1_repair_parameter_model" / "model.pt"
DEFAULT_OUTPUT = DEFAULT_MODEL.parent


def _safe_corr(fn, x, y):
    if len(x) < 2 or np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return None
    v, _ = fn(x, y)
    return float(v)


def load_model(model_path: Path, bundle: dict) -> PINNRepairParameterModel:
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    model = PINNRepairParameterModel(
        input_dim=ckpt["input_dim"],
        hidden_dim=ckpt["hidden_dim"],
        num_layers=ckpt["num_layers"],
        dropout=ckpt.get("dropout", 0.1),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def predict_all(model, bundle):
    x = torch.from_numpy(bundle["features"])
    out = model(x)
    explicit = explicit_repair_energy(x)
    return (
        out["E_total"].numpy(),
        explicit.numpy(),
        torch.sigmoid(out["success_logit"]).numpy(),
        out["failure_type_logits"].argmax(dim=-1).numpy(),
        out["outcome_logits"].argmax(dim=-1).numpy(),
        torch.sigmoid(out["grasp_success_logit"]).numpy(),
        torch.sigmoid(out["lift_success_logit"]).numpy(),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate V1-E repair parameter model")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    bundle = load_repair_npz(args.dataset)
    model = load_model(args.model, bundle)
    pred_total, explicit_total, pred_success, pred_fail, pred_outcome, pred_grasp, pred_lift = predict_all(
        model, bundle
    )

    target = bundle["target_E_total"]
    success = bundle["success_flag"]
    refined = bundle["refined_success_flag"]

    report = {
        "model_version": "V1-E_PINNRepairParameterModel",
        "dataset": str(args.dataset),
        "model": str(args.model),
        "num_samples": int(len(target)),
        "pinn_metrics": {
            "E_total_mae": float(np.mean(np.abs(pred_total - target))),
            "E_total_rmse": float(np.sqrt(np.mean((pred_total - target) ** 2))),
            "pearson_E_total": _safe_corr(pearsonr, pred_total, target),
            "spearman_E_total": _safe_corr(spearmanr, pred_total, target),
            "success_accuracy": float(np.mean((pred_success >= 0.5) == (success >= 0.5))),
            "failure_type_accuracy": float(np.mean(pred_fail == bundle["failure_type_idx"])),
            "outcome_accuracy": float(np.mean(pred_outcome == bundle["outcome_idx"])),
            "grasp_success_accuracy": float(np.mean((pred_grasp >= 0.5) == (bundle["grasp_success_flag"] >= 0.5))),
            "lift_success_accuracy": float(np.mean((pred_lift >= 0.5) == (bundle["lift_success_flag"] >= 0.5))),
            "top_k_refined_success": {
                "top_1": top_k_refined_success_rate(pred_total, refined, 1),
                "top_3": top_k_refined_success_rate(pred_total, refined, 3),
                "top_5": top_k_refined_success_rate(pred_total, refined, 5),
            },
        },
        "explicit_baseline_metrics": {
            "pearson_E_total": _safe_corr(pearsonr, explicit_total, target),
            "top_k_refined_success": {
                "top_1": top_k_refined_success_rate(explicit_total, refined, 1),
                "top_3": top_k_refined_success_rate(explicit_total, refined, 3),
                "top_5": top_k_refined_success_rate(explicit_total, refined, 5),
            },
        },
        "random_baseline_top_k_refined_success": {
            "top_1": top_k_refined_success_rate(np.random.default_rng(0).random(len(target)), refined, 1),
            "top_3": top_k_refined_success_rate(np.random.default_rng(1).random(len(target)), refined, 3),
            "top_5": top_k_refined_success_rate(np.random.default_rng(2).random(len(target)), refined, 5),
        },
        "notes": [
            "V1-E evaluates repair-parameter field, not trajectory residual predictor (V1-D).",
            "Explicit energy is baseline/supervision only.",
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "evaluation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
