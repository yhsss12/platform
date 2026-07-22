"""V1-C.5：Group-split / leave-one-demo-out 泛化评估。"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from group_split_utils import (
    build_failure_mode_holdout_splits,
    build_leave_one_demo_out_splits,
    build_prediction_rows,
    evaluate_predictions_on_indices,
    load_enriched_meta,
    split_train_val,
    summarize_generalization_risk,
)
from residual_dataset import load_npz_dataset
from train_residual_model import train_model

_V1_DIR = Path(__file__).resolve().parent
_EXPERIMENT_DIR = _V1_DIR.parent
DEFAULT_DATASET_V1C = _EXPERIMENT_DIR / "outputs" / "v1_residual_model_v1c" / "training_dataset.npz"
DEFAULT_OUTPUT = _EXPERIMENT_DIR / "outputs" / "v1_residual_model_v1c_group_split"
DEFAULT_RANDOM_EVAL = _EXPERIMENT_DIR / "outputs" / "v1_residual_model_v1c" / "evaluation_report.json"


@torch.no_grad()
def predict_all(model, bundle: dict) -> tuple[np.ndarray, ...]:
    model.eval()
    features = torch.from_numpy(bundle["features"])
    out = model(features)
    pred_total = out["E_total"].numpy()
    pred_success_prob = torch.sigmoid(out["success_logit"]).numpy()
    pred_failure_idx = out["failure_type_logits"].argmax(dim=-1).numpy()
    pred_outcome_idx = out["outcome_logits"].argmax(dim=-1).numpy() if "outcome_logits" in out else None
    pred_grasp_prob = (
        torch.sigmoid(out["grasp_success_logit"]).numpy() if "grasp_success_logit" in out else None
    )
    pred_lift_prob = (
        torch.sigmoid(out["lift_success_logit"]).numpy() if "lift_success_logit" in out else None
    )
    return pred_total, pred_success_prob, pred_failure_idx, pred_outcome_idx, pred_grasp_prob, pred_lift_prob


def load_random_baseline(path: Path) -> dict | None:
    if not path.exists():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    return {
        "num_samples": report.get("num_samples"),
        "feature_dim": report.get("feature_dim"),
        "E_total_mae": report.get("E_total_mae"),
        "E_total_rmse": report.get("E_total_rmse"),
        "pearson_E_total": report.get("pearson_E_total"),
        "spearman_E_total": report.get("spearman_E_total"),
        "success_classification_accuracy": report.get("success_classification_accuracy"),
        "failure_type_accuracy": report.get("failure_type_accuracy"),
        "outcome_classification_accuracy": report.get("outcome_classification_accuracy"),
        "grasp_success_accuracy": report.get("grasp_success_accuracy"),
        "lift_success_accuracy": report.get("lift_success_accuracy"),
    }


def run_split(
    *,
    split: dict,
    bundle: dict,
    meta_records: list,
    dataset_path: Path,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    val_frac: float,
    seed: int,
) -> tuple[dict, list[dict], dict]:
    split_id = split["split_id"]
    train_idx, val_idx = split_train_val(split["train_idx"], val_frac, seed)

    model, train_info = train_model(
        dataset_path=dataset_path,
        train_idx=train_idx,
        val_idx=val_idx,
        model_version="v1c",
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        seed=seed,
    )

    split_model_dir = output_dir / "models" / split_id
    split_model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": train_info["feature_dim"],
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "predict_outcome": True,
            "predict_grasp_lift": True,
            "model_version": "v1c",
            "split_id": split_id,
            "train_indices": train_idx.tolist(),
            "val_indices": val_idx.tolist(),
            "test_indices": split["test_idx"].tolist(),
        },
        split_model_dir / "model.pt",
    )

    pred_total, pred_success_prob, pred_failure_idx, pred_outcome_idx, pred_grasp_prob, pred_lift_prob = predict_all(
        model, bundle
    )
    split_result = evaluate_predictions_on_indices(
        bundle=bundle,
        meta_records=meta_records,
        test_idx=split["test_idx"],
        pred_total=pred_total,
        pred_success_prob=pred_success_prob,
        pred_failure_idx=pred_failure_idx,
        pred_outcome_idx=pred_outcome_idx,
        pred_grasp_prob=pred_grasp_prob,
        pred_lift_prob=pred_lift_prob,
        split_info=split,
    )
    split_result["training"] = train_info

    prediction_rows = build_prediction_rows(
        split_id=split_id,
        test_idx=split["test_idx"],
        meta_records=meta_records,
        bundle=bundle,
        pred_total=pred_total,
        pred_success_prob=pred_success_prob,
        pred_failure_idx=pred_failure_idx,
        pred_outcome_idx=pred_outcome_idx,
        pred_grasp_prob=pred_grasp_prob,
        pred_lift_prob=pred_lift_prob,
    )

    confusion = {
        "failure_type": split_result.pop("confusion_matrix_failure_type"),
        "outcome": split_result.pop("confusion_matrix_outcome"),
    }
    return split_result, prediction_rows, confusion


def main() -> int:
    parser = argparse.ArgumentParser(description="V1-C.5 group split evaluation")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_V1C)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--random-eval-report", type=Path, default=DEFAULT_RANDOM_EVAL)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-failure-mode-holdout", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_npz_dataset(args.dataset)
    meta_records = load_enriched_meta(args.dataset)

    splits = build_leave_one_demo_out_splits(meta_records)
    if not args.skip_failure_mode_holdout:
        splits.extend(build_failure_mode_holdout_splits(meta_records))

    split_results: list[dict] = []
    all_predictions: list[dict] = []
    all_confusions: dict[str, dict] = {}

    for split in splits:
        print(f"Running split: {split['split_id']} (train={len(split['train_idx'])}, test={len(split['test_idx'])})")
        result, rows, confusion = run_split(
            split=split,
            bundle=bundle,
            meta_records=meta_records,
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            val_frac=args.val_frac,
            seed=args.seed,
        )
        split_results.append(result)
        all_predictions.extend(rows)
        all_confusions[split["split_id"]] = confusion

    random_baseline = load_random_baseline(args.random_eval_report)
    generalization = summarize_generalization_risk(split_results, random_baseline)

    report = {
        "task": "V1-C.5_group_split_evaluation",
        "dataset": str(args.dataset),
        "num_samples": int(len(bundle["features"])),
        "feature_dim": int(bundle["features"].shape[1]),
        "num_splits": len(split_results),
        "split_ids": [s["split_id"] for s in split_results],
        "generalization_assessment": generalization,
        "random_split_baseline": random_baseline,
        "split_results": split_results,
        "notes": [
            "V1-C random split metrics can be optimistic due to demo-level leakage.",
            "Group split / leave-one-demo-out is the preferred generalization check.",
            "Use this report to decide V2-B5 (demo_3 lift_failed) vs PINA/PINN formalization.",
            "V1-C is still a PyTorch residual prototype, not a final PINN.",
        ],
    }

    (args.output_dir / "group_split_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output_dir / "confusion_matrices.json").write_text(json.dumps(all_confusions, indent=2), encoding="utf-8")

    summary_fields = [
        "split_id",
        "split_type",
        "test_demo",
        "holdout_failure_mode",
        "train_size",
        "test_size",
        "E_total_mae",
        "E_total_rmse",
        "pearson_E_total",
        "spearman_E_total",
        "success_classification_accuracy",
        "failure_type_accuracy",
        "outcome_classification_accuracy",
        "grasp_success_accuracy",
        "lift_success_accuracy",
    ]
    with (args.output_dir / "group_split_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for row in split_results:
            writer.writerow({k: row.get(k, "") for k in summary_fields})

    pred_fields = [
        "split_id",
        "sample_idx",
        "source_demo",
        "source_failure_mode",
        "sample_source",
        "target_E_total",
        "pred_E_total",
        "target_success",
        "pred_success_prob",
        "target_failure_type",
        "pred_failure_type",
        "target_outcome",
        "pred_outcome",
        "target_grasp_success",
        "pred_grasp_success_prob",
        "target_lift_success",
        "pred_lift_success_prob",
    ]
    with (args.output_dir / "per_split_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=pred_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_predictions)

    comparison = {
        "random_split": random_baseline,
        "group_split_aggregate": {
            "leave_one_demo_out_avg_pearson": generalization.get("leave_one_demo_out_avg_pearson"),
            "leave_one_demo_out_avg_success_accuracy": generalization.get(
                "leave_one_demo_out_avg_success_accuracy"
            ),
            "generalization_risk": generalization.get("generalization_risk"),
            "v2_b5_priority": generalization.get("v2_b5_priority"),
        },
        "per_split": [
            {
                "split_id": r["split_id"],
                "test_demo": r.get("test_demo"),
                "holdout_failure_mode": r.get("holdout_failure_mode"),
                "pearson_E_total": r.get("pearson_E_total"),
                "success_classification_accuracy": r.get("success_classification_accuracy"),
                "failure_type_accuracy": r.get("failure_type_accuracy"),
                "grasp_success_accuracy": r.get("grasp_success_accuracy"),
                "lift_success_accuracy": r.get("lift_success_accuracy"),
                "special_focus": r.get("special_focus"),
            }
            for r in split_results
        ],
        "notes": generalization.get("notes", []),
    }
    (args.output_dir / "v1c_random_vs_group_split_comparison.json").write_text(
        json.dumps(comparison, indent=2),
        encoding="utf-8",
    )

    print(json.dumps({"output_dir": str(args.output_dir), "generalization_assessment": generalization}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
