"""V1-D：V1-C vs PINN 对比 + physics loss ablation。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_pinn_group_split import evaluate_indices, load_pinn_model, predict_all, run_group_splits
from group_split_utils import build_leave_one_demo_out_splits, load_enriched_meta, summarize_generalization_risk
from pinn_residual_energy_model import PhysicsLossConfig
from residual_dataset import load_npz_dataset
from train_pinn_residual_model import ABLATION_PRESETS, train_pinn_model
from train_residual_model import split_indices

_V1_DIR = Path(__file__).resolve().parent
_EXPERIMENT_DIR = _V1_DIR.parent
DEFAULT_DATASET = _EXPERIMENT_DIR / "outputs" / "v1_residual_model_v1c" / "training_dataset.npz"
DEFAULT_OUTPUT = _EXPERIMENT_DIR / "outputs" / "v1_residual_model_pinn"
V1C_RANDOM = _EXPERIMENT_DIR / "outputs" / "v1_residual_model_v1c" / "evaluation_report.json"
V1C_GROUP = _EXPERIMENT_DIR / "outputs" / "v1_residual_model_v1c_group_split" / "group_split_report.json"
V1C_GROUP_COMPARE = _EXPERIMENT_DIR / "outputs" / "v1_residual_model_v1c_group_split" / "v1c_random_vs_group_split_comparison.json"


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _compact_random(report: dict | None) -> dict | None:
    if not report:
        return None
    if "random_split" in report:
        return report["random_split"]
    keys = [
        "pearson_E_total", "success_classification_accuracy", "failure_type_accuracy",
        "outcome_classification_accuracy", "grasp_success_accuracy", "lift_success_accuracy",
        "E_total_mae", "E_total_rmse",
    ]
    return {k: report.get(k) for k in keys}


def _lodo_aggregate(split_results: list[dict]) -> dict:
    if not split_results:
        return {}
    return {
        "avg_pearson": float(np.nanmean([r.get("pearson_E_total") or np.nan for r in split_results])),
        "avg_success_accuracy": float(np.nanmean([r.get("success_classification_accuracy") or np.nan for r in split_results])),
        "avg_failure_type_accuracy": float(np.nanmean([r.get("failure_type_accuracy") or np.nan for r in split_results])),
        "avg_grasp_success_accuracy": float(np.nanmean([r.get("grasp_success_accuracy") or np.nan for r in split_results if r.get("grasp_success_accuracy") is not None] or [np.nan])),
    }


def run_ablation(
    *,
    dataset_path: Path,
    bundle: dict,
    meta_records: list,
    ablation_name: str,
    physics: PhysicsLossConfig,
    epochs: int,
    seed: int,
) -> dict:
    from group_split_utils import evaluate_predictions_on_indices, split_train_val

    train_idx, val_idx = split_indices(len(bundle["features"]), 0.2, seed)
    model, train_info = train_pinn_model(
        dataset_path=dataset_path,
        train_idx=train_idx,
        val_idx=val_idx,
        output_dir=None,
        save_model=False,
        physics=physics,
        epochs=epochs,
        seed=seed,
        model_tag=f"ablation_{ablation_name}",
    )
    preds = predict_all(model, bundle)
    all_idx = np.arange(len(bundle["features"]))
    random_metrics = evaluate_indices(
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
        label=f"ablation_{ablation_name}_random",
    )

    lodo_epochs = max(epochs // 2, 100)
    lodo_results = []
    for split in build_leave_one_demo_out_splits(meta_records):
        inner_train, inner_val = split_train_val(split["train_idx"], 0.15, seed)
        m, _ = train_pinn_model(
            dataset_path=dataset_path,
            train_idx=inner_train,
            val_idx=inner_val,
            output_dir=None,
            save_model=False,
            physics=physics,
            epochs=lodo_epochs,
            seed=seed,
            model_tag=f"ablation_{ablation_name}_{split['split_id']}",
        )
        p = predict_all(m, bundle)
        result = evaluate_predictions_on_indices(
            bundle=bundle,
            meta_records=meta_records,
            test_idx=split["test_idx"],
            pred_total=p[0],
            pred_success_prob=p[1],
            pred_failure_idx=p[2],
            pred_outcome_idx=p[3],
            pred_grasp_prob=p[4],
            pred_lift_prob=p[5],
            split_info=split,
        )
        result.pop("confusion_matrix_failure_type", None)
        result.pop("confusion_matrix_outcome", None)
        lodo_results.append(result)

    return {
        "ablation": ablation_name,
        "physics_config": {
            "use_phys_components": physics.use_phys_components,
            "use_total_consistency": physics.use_total_consistency,
            "use_margin": physics.use_margin,
        },
        "epochs": epochs,
        "lodo_epochs": lodo_epochs,
        "best_val_loss": train_info["best_val_loss"],
        "random_split": random_metrics,
        "leave_one_demo_out_aggregate": _lodo_aggregate(lodo_results),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare V1-C vs V1-D PINN + physics ablation")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ablation-epochs", type=int, default=150)
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    v1c_random = _load_json(V1C_RANDOM)
    v1c_group = _load_json(V1C_GROUP)
    v1c_group_cmp = _load_json(V1C_GROUP_COMPARE)
    pinn_random = _load_json(args.output_dir / "evaluation_report.json")
    pinn_group = _load_json(args.output_dir / "group_split_report.json")

    v1c_lodo = (v1c_group or {}).get("split_results") or (v1c_group_cmp or {}).get("per_split")
    pinn_lodo = (pinn_group or {}).get("split_results")

    comparison = {
        "random_split": {
            "v1c": _compact_random(v1c_random),
            "pinn": _compact_random(pinn_random),
        },
        "group_split_leave_one_demo_out": {
            "v1c_aggregate": _lodo_aggregate(v1c_lodo or []),
            "pinn_aggregate": _lodo_aggregate(pinn_lodo or []),
            "v1c_per_split": v1c_lodo,
            "pinn_per_split": pinn_lodo,
        },
        "notes": [
            "Primary comparison should use group split / leave-one-demo-out, not random split alone.",
            "PINN-style model adds explicit Nut Assembly physics residuals to the loss.",
            "Not a final generalization model; demo count remains limited.",
        ],
    }

    if v1c_lodo and pinn_lodo:
        v1c_succ = comparison["group_split_leave_one_demo_out"]["v1c_aggregate"].get("avg_success_accuracy")
        pinn_succ = comparison["group_split_leave_one_demo_out"]["pinn_aggregate"].get("avg_success_accuracy")
        if v1c_succ is not None and pinn_succ is not None:
            comparison["acceptance"] = {
                "pinn_lodo_success_not_worse_than_v1c": pinn_succ >= v1c_succ - 0.05,
                "delta_success_accuracy": pinn_succ - v1c_succ,
            }

    ablation_results = []
    if not args.skip_ablation:
        bundle = load_npz_dataset(args.dataset)
        meta_records = load_enriched_meta(args.dataset)
        for name in ("no_phys_components", "no_total_consistency", "no_margin", "full"):
            print(f"Ablation: {name}")
            ablation_results.append(
                run_ablation(
                    dataset_path=args.dataset,
                    bundle=bundle,
                    meta_records=meta_records,
                    ablation_name=name,
                    physics=ABLATION_PRESETS[name],
                    epochs=args.ablation_epochs,
                    seed=args.seed,
                )
            )

        full = next(r for r in ablation_results if r["ablation"] == "full")
        ablation_acceptance = {
            "full_not_worse_than_no_phys_on_lodo_success": (
                full["leave_one_demo_out_aggregate"].get("avg_success_accuracy", 0)
                >= next(r for r in ablation_results if r["ablation"] == "no_phys_components")["leave_one_demo_out_aggregate"].get("avg_success_accuracy", 0) - 0.05
            ),
        }
        (args.output_dir / "physics_loss_ablation.json").write_text(
            json.dumps({"ablations": ablation_results, "acceptance": ablation_acceptance}, indent=2),
            encoding="utf-8",
        )
        comparison["physics_loss_ablation_summary"] = [
            {
                "ablation": r["ablation"],
                "random_pearson": r["random_split"].get("pearson_E_total"),
                "lodo_avg_success": r["leave_one_demo_out_aggregate"].get("avg_success_accuracy"),
                "lodo_avg_pearson": r["leave_one_demo_out_aggregate"].get("avg_pearson"),
            }
            for r in ablation_results
        ]

    (args.output_dir / "v1c_vs_pinn_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(json.dumps(comparison, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
