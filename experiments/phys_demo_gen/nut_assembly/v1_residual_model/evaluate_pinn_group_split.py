"""V1-D：PINN random split + group split 评估。"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

from group_split_utils import (
    build_leave_one_demo_out_splits,
    build_prediction_rows,
    evaluate_predictions_on_indices,
    load_enriched_meta,
    split_train_val,
    summarize_generalization_risk,
    top_k_refined_success_rate,
    top_k_success_rate,
)
from pinn_residual_energy_model import PINNResidualEnergyModel
from residual_dataset import FAILURE_TYPES, OUTCOME_TYPES, load_npz_dataset
from train_pinn_residual_model import train_pinn_model
from train_residual_model import split_indices

_V1_DIR = Path(__file__).resolve().parent
_EXPERIMENT_DIR = _V1_DIR.parent
DEFAULT_DATASET = _EXPERIMENT_DIR / "outputs" / "v1_residual_model_v1c" / "training_dataset.npz"
DEFAULT_OUTPUT = _EXPERIMENT_DIR / "outputs" / "v1_residual_model_pinn"
DEFAULT_MODEL = DEFAULT_OUTPUT / "model.pt"


def _safe_corr(fn, x, y):
    if len(x) < 2 or np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return None
    v, _ = fn(x, y)
    return float(v)


def load_pinn_model(model_path: Path, bundle: dict) -> PINNResidualEnergyModel:
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    model = PINNResidualEnergyModel(
        input_dim=ckpt["input_dim"],
        hidden_dim=ckpt["hidden_dim"],
        num_layers=ckpt["num_layers"],
        dropout=ckpt.get("dropout", 0.1),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def predict_all(model: PINNResidualEnergyModel, bundle: dict):
    features = torch.from_numpy(bundle["features"])
    out = model(features)
    return (
        out["E_total"].numpy(),
        torch.sigmoid(out["success_logit"]).numpy(),
        out["failure_type_logits"].argmax(dim=-1).numpy(),
        out["outcome_logits"].argmax(dim=-1).numpy(),
        torch.sigmoid(out["grasp_success_logit"]).numpy(),
        torch.sigmoid(out["lift_success_logit"]).numpy(),
        out["E_components"].numpy(),
        out["E_total_consistent"].numpy(),
    )


def evaluate_indices(
    *,
    bundle,
    meta_records,
    indices: np.ndarray,
    pred_total,
    pred_success_prob,
    pred_failure_idx,
    pred_outcome_idx,
    pred_grasp_prob,
    pred_lift_prob,
    pred_components,
    pred_total_consistent,
    label: str,
) -> dict:
    target_total = bundle["target_E_total"][indices]
    success_flag = bundle["success_flag"][indices]
    failure_type_idx = bundle["failure_type_idx"][indices]
    outcome_idx = bundle["outcome_idx"][indices]
    refined_success = bundle.get("refined_success_flag", bundle["success_flag"])[indices]
    grasp_success = bundle.get("grasp_success_flag")
    lift_success = bundle.get("lift_success_flag")
    if grasp_success is not None:
        grasp_success = grasp_success[indices]
    if lift_success is not None:
        lift_success = lift_success[indices]

    pt = pred_total[indices]
    mae = float(np.mean(np.abs(pt - target_total)))
    rmse = float(np.sqrt(np.mean((pt - target_total) ** 2)))

    lift_mask = np.array([FAILURE_TYPES[int(i)] == "lift_failed" for i in failure_type_idx])
    demo_3_lift_acc = None
    if np.any(lift_mask):
        demo_3_lift_acc = float(np.mean(pred_failure_idx[indices][lift_mask] == failure_type_idx[lift_mask]))

    refined_mask = np.array([OUTCOME_TYPES[int(outcome_idx[i])] == "refined_success" for i in range(len(indices))])
    refined_recall = float(np.mean(pred_outcome_idx[indices][refined_mask] == outcome_idx[refined_mask])) if np.any(refined_mask) else None

    components_nonneg = bool(np.all(pred_components[indices] >= -1e-6))
    consistency_mae = float(np.mean(np.abs(pt - pred_total_consistent[indices])))

    return {
        "eval_scope": label,
        "num_samples": int(len(indices)),
        "E_total_mae": mae,
        "E_total_rmse": rmse,
        "pearson_E_total": _safe_corr(pearsonr, pt, target_total),
        "spearman_E_total": _safe_corr(spearmanr, pt, target_total),
        "success_classification_accuracy": float(np.mean((pred_success_prob[indices] >= 0.5) == (success_flag >= 0.5))),
        "failure_type_accuracy": float(np.mean(pred_failure_idx[indices] == failure_type_idx)),
        "outcome_classification_accuracy": float(np.mean(pred_outcome_idx[indices] == outcome_idx)),
        "grasp_success_accuracy": float(np.mean((pred_grasp_prob[indices] >= 0.5) == (grasp_success >= 0.5)))
        if grasp_success is not None
        else None,
        "lift_success_accuracy": float(np.mean((pred_lift_prob[indices] >= 0.5) == (lift_success >= 0.5)))
        if lift_success is not None
        else None,
        "lift_failed_accuracy": demo_3_lift_acc,
        "refined_success_outcome_recall": refined_recall,
        "top_k_low_energy_contains_success": {
            "top_1": top_k_success_rate(pt, success_flag, 1),
            "top_3": top_k_success_rate(pt, success_flag, 3),
            "top_5": top_k_success_rate(pt, success_flag, 5),
        },
        "top_k_low_energy_contains_refined_success": {
            "top_1": top_k_refined_success_rate(pt, refined_success, 1),
            "top_3": top_k_refined_success_rate(pt, refined_success, 3),
            "top_5": top_k_refined_success_rate(pt, refined_success, 5),
        },
        "E_components_non_negative": components_nonneg,
        "E_total_component_consistency_mae": consistency_mae,
    }


def run_group_splits(
    *,
    dataset_path,
    output_dir,
    bundle,
    meta_records,
    epochs,
    batch_size,
    lr,
    hidden_dim,
    num_layers,
    dropout,
    val_frac,
    seed,
) -> tuple[list[dict], list[dict], dict]:
    splits = build_leave_one_demo_out_splits(meta_records)
    split_results = []
    all_predictions = []
    all_confusions = {}

    for split in splits:
        print(f"PINN group split: {split['split_id']}")
        train_idx, val_idx = split_train_val(split["train_idx"], val_frac, seed)
        model, _ = train_pinn_model(
            dataset_path=dataset_path,
            train_idx=train_idx,
            val_idx=val_idx,
            output_dir=None,
            save_model=False,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            seed=seed,
        )
        preds = predict_all(model, bundle)
        result = evaluate_predictions_on_indices(
            bundle=bundle,
            meta_records=meta_records,
            test_idx=split["test_idx"],
            pred_total=preds[0],
            pred_success_prob=preds[1],
            pred_failure_idx=preds[2],
            pred_outcome_idx=preds[3],
            pred_grasp_prob=preds[4],
            pred_lift_prob=preds[5],
            split_info=split,
        )
        split_results.append(result)
        all_predictions.extend(
            build_prediction_rows(
                split_id=split["split_id"],
                test_idx=split["test_idx"],
                meta_records=meta_records,
                bundle=bundle,
                pred_total=preds[0],
                pred_success_prob=preds[1],
                pred_failure_idx=preds[2],
                pred_outcome_idx=preds[3],
                pred_grasp_prob=preds[4],
                pred_lift_prob=preds[5],
            )
        )
        all_confusions[split["split_id"]] = {
            "failure_type": result.pop("confusion_matrix_failure_type"),
            "outcome": result.pop("confusion_matrix_outcome"),
        }

    generalization = summarize_generalization_risk(split_results, None)
    return split_results, all_predictions, {"generalization": generalization, "confusions": all_confusions}


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate V1-D PINN (random + group split)")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--retrain-random", action="store_true", help="Retrain random-split model before eval")
    parser.add_argument("--run-group-split", action="store_true")
    parser.add_argument("--skip-group-split", action="store_true")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--group-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    group_epochs = args.group_epochs or args.epochs

    args.output_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_npz_dataset(args.dataset)
    meta_records = load_enriched_meta(args.dataset)

    if args.retrain_random or not args.model.exists():
        train_idx, val_idx = split_indices(len(bundle["features"]), args.val_frac, args.seed)
        train_pinn_model(
            dataset_path=args.dataset,
            train_idx=train_idx,
            val_idx=val_idx,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
        )

    model = load_pinn_model(args.model, bundle)
    preds = predict_all(model, bundle)
    all_idx = np.arange(len(bundle["features"]))

    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    val_idx = np.array(ckpt.get("val_indices", []))
    train_idx = np.array(ckpt.get("train_indices", []))
    random_eval = evaluate_indices(
        bundle=bundle,
        meta_records=meta_records,
        indices=all_idx,
        pred_total=preds[0],
        pred_success_prob=preds[1],
        pred_failure_idx=preds[2],
        pred_outcome_idx=preds[3],
        pred_grasp_prob=preds[4],
        pred_lift_prob=preds[5],
        pred_components=preds[6],
        pred_total_consistent=preds[7],
        label="random_split_full_dataset",
    )
    random_eval["train_size"] = int(len(train_idx))
    random_eval["val_size"] = int(len(val_idx))

    report = {
        "model_version": "V1-D_PINN_style_residual_energy_mlp",
        "dataset": str(args.dataset),
        "model": str(args.model),
        "random_split": random_eval,
        "acceptance_checks": {
            "positive_correlation": (random_eval.get("pearson_E_total") or -1) > 0,
            "E_components_non_negative": random_eval.get("E_components_non_negative", False),
            "E_total_consistency": (random_eval.get("E_total_component_consistency_mae") or 999) < 1.0,
        },
    }

    group_report = None
    if not args.skip_group_split:
        split_results, pred_rows, group_extra = run_group_splits(
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            bundle=bundle,
            meta_records=meta_records,
            epochs=group_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            hidden_dim=128,
            num_layers=3,
            dropout=0.1,
            val_frac=0.15,
            seed=args.seed,
        )
        group_report = {
            "split_results": split_results,
            **group_extra["generalization"],
        }
        report["group_split"] = group_report
        report["primary_eval"] = "group_split"

        with (args.output_dir / "group_split_report.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "task": "V1-D_PINN_group_split",
                    "split_results": split_results,
                    "generalization_assessment": group_extra["generalization"],
                },
                handle,
                indent=2,
            )
        with (args.output_dir / "confusion_matrices.json").open("w", encoding="utf-8") as handle:
            json.dump(group_extra["confusions"], handle, indent=2)

        pred_fields = list(pred_rows[0].keys()) if pred_rows else []
        if pred_rows:
            with (args.output_dir / "per_split_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=pred_fields)
                writer.writeheader()
                writer.writerows(pred_rows)

    (args.output_dir / "evaluation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    csv_path = args.output_dir / "predictions.csv"
    meta_records_full = meta_records
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "sample_idx", "source_demo", "sample_source", "target_E_total", "pred_E_total",
            "target_success", "pred_success_prob", "target_failure_type", "pred_failure_type",
            "target_outcome", "pred_outcome", "target_grasp_success", "pred_grasp_success_prob",
            "target_lift_success", "pred_lift_success_prob",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i in range(len(bundle["features"])):
            m = meta_records_full[i]
            writer.writerow({
                "sample_idx": i,
                "source_demo": m.get("source_demo"),
                "sample_source": m.get("sample_source"),
                "target_E_total": float(bundle["target_E_total"][i]),
                "pred_E_total": float(preds[0][i]),
                "target_success": int(bundle["success_flag"][i]),
                "pred_success_prob": float(preds[1][i]),
                "target_failure_type": m.get("failure_type"),
                "pred_failure_type": FAILURE_TYPES[int(preds[2][i])],
                "target_outcome": m.get("outcome"),
                "pred_outcome": OUTCOME_TYPES[int(preds[3][i])],
                "target_grasp_success": int(bundle["grasp_success_flag"][i]) if "grasp_success_flag" in bundle else "",
                "pred_grasp_success_prob": float(preds[4][i]),
                "target_lift_success": int(bundle["lift_success_flag"][i]) if "lift_success_flag" in bundle else "",
                "pred_lift_success_prob": float(preds[5][i]),
            })

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
