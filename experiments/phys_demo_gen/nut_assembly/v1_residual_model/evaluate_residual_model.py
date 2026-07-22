"""V1-A / V1-B / V1-C：评估 Residual Energy Model。"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

from residual_dataset import FAILURE_TYPES, OUTCOME_TYPES, load_npz_dataset
from residual_energy_model import ResidualEnergyModel

_V1_DIR = Path(__file__).resolve().parent
_EXPERIMENT_DIR = _V1_DIR.parent
DEFAULT_OUTPUT_V1A = _EXPERIMENT_DIR / "outputs" / "v1_residual_model"
DEFAULT_OUTPUT_V1B = _EXPERIMENT_DIR / "outputs" / "v1_residual_model_v1b"
DEFAULT_OUTPUT_V1C = _EXPERIMENT_DIR / "outputs" / "v1_residual_model_v1c"
DEFAULT_DATASET_V1A = DEFAULT_OUTPUT_V1A / "training_dataset.npz"
DEFAULT_DATASET_V1B = DEFAULT_OUTPUT_V1B / "training_dataset.npz"
DEFAULT_DATASET_V1C = DEFAULT_OUTPUT_V1C / "training_dataset.npz"
DEFAULT_MODEL_V1A = DEFAULT_OUTPUT_V1A / "model.pt"
DEFAULT_MODEL_V1B = DEFAULT_OUTPUT_V1B / "model.pt"
DEFAULT_MODEL_V1C = DEFAULT_OUTPUT_V1C / "model.pt"


def _safe_corr(fn, x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2 or np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return None
    value, _ = fn(x, y)
    return float(value)


def top_k_success_rate(pred_total: np.ndarray, success: np.ndarray, k: int) -> float:
    if len(pred_total) == 0:
        return 0.0
    k = min(k, len(pred_total))
    order = np.argsort(pred_total)
    picked = success[order[:k]]
    return float(np.any(picked > 0.5))


def top_k_refined_success_rate(pred_total: np.ndarray, refined_success: np.ndarray, k: int) -> float:
    if len(pred_total) == 0:
        return 0.0
    k = min(k, len(pred_total))
    order = np.argsort(pred_total)
    picked = refined_success[order[:k]]
    return float(np.any(picked > 0.5))


def energy_separation(pred_total: np.ndarray, success: np.ndarray) -> dict[str, float | None]:
    succ = pred_total[success > 0.5]
    fail = pred_total[success <= 0.5]
    if len(succ) == 0 or len(fail) == 0:
        return {
            "success_pred_mean": float(np.mean(succ)) if len(succ) else None,
            "failed_pred_mean": float(np.mean(fail)) if len(fail) else None,
            "mean_gap_success_minus_failed": None,
        }
    return {
        "success_pred_mean": float(np.mean(succ)),
        "failed_pred_mean": float(np.mean(fail)),
        "mean_gap_success_minus_failed": float(np.mean(succ) - np.mean(fail)),
    }


def _class_accuracy(
    pred_idx: np.ndarray,
    target_idx: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> float | None:
    if mask is None:
        mask = np.ones(len(pred_idx), dtype=bool)
    if not np.any(mask):
        return None
    return float(np.mean(pred_idx[mask] == target_idx[mask]))


def _load_model(model_path: Path, bundle: dict) -> ResidualEnergyModel:
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]
    num_failure_types = int(state["head_failure.weight"].shape[0])
    predict_outcome = ckpt.get("predict_outcome", "head_outcome.weight" in state)
    predict_grasp_lift = ckpt.get("predict_grasp_lift", "head_grasp_success.weight" in state)
    num_outcome_types = int(state["head_outcome.weight"].shape[0]) if predict_outcome else len(OUTCOME_TYPES)
    model = ResidualEnergyModel(
        input_dim=ckpt["input_dim"],
        hidden_dim=ckpt["hidden_dim"],
        num_layers=ckpt["num_layers"],
        dropout=ckpt.get("dropout", 0.1),
        num_failure_types=num_failure_types,
        num_outcome_types=num_outcome_types,
        predict_outcome=predict_outcome,
        predict_grasp_lift=predict_grasp_lift,
    )
    model.load_state_dict(state)
    model.eval()
    return model


def evaluate_model(
    *,
    dataset_path: Path,
    model_path: Path,
    model_version: str,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None, list[dict]]:
    bundle = load_npz_dataset(dataset_path)
    meta = bundle["meta"]
    meta_records = meta.get("meta_records", [])

    model = _load_model(model_path, bundle)
    features = torch.from_numpy(bundle["features"])
    with torch.no_grad():
        out = model(features)

    pred_total = out["E_total"].numpy()
    pred_success_prob = torch.sigmoid(out["success_logit"]).numpy()
    pred_failure_idx = out["failure_type_logits"].argmax(dim=-1).numpy()
    pred_outcome_idx = (
        out["outcome_logits"].argmax(dim=-1).numpy() if "outcome_logits" in out else None
    )
    pred_grasp_prob = (
        torch.sigmoid(out["grasp_success_logit"]).numpy() if "grasp_success_logit" in out else None
    )
    pred_lift_prob = (
        torch.sigmoid(out["lift_success_logit"]).numpy() if "lift_success_logit" in out else None
    )

    target_total = bundle["target_E_total"]
    success_flag = bundle["success_flag"]
    failure_type_idx = bundle["failure_type_idx"]
    outcome_idx = bundle.get("outcome_idx")
    refined_success_flag = bundle.get("refined_success_flag")
    grasp_success_flag = bundle.get("grasp_success_flag")
    lift_success_flag = bundle.get("lift_success_flag")

    mae = float(np.mean(np.abs(pred_total - target_total)))
    rmse = float(np.sqrt(np.mean((pred_total - target_total) ** 2)))
    success_acc = float(np.mean((pred_success_prob >= 0.5) == (success_flag >= 0.5)))
    failure_acc = float(np.mean(pred_failure_idx == failure_type_idx))

    failure_type_counts = Counter(FAILURE_TYPES[int(idx)] for idx in failure_type_idx)
    outcome_counts = Counter(
        OUTCOME_TYPES[int(outcome_idx[i])] for i in range(len(outcome_idx))
    ) if outcome_idx is not None else {}

    grasp_mask = np.array([FAILURE_TYPES[int(i)] == "grasp_failed" for i in failure_type_idx])
    lift_mask = np.array([FAILURE_TYPES[int(i)] == "lift_failed" for i in failure_type_idx])
    grasp_acc = _class_accuracy(pred_failure_idx, failure_type_idx, mask=grasp_mask)
    lift_failed_acc = _class_accuracy(pred_failure_idx, failure_type_idx, mask=lift_mask)

    grasp_success_acc = None
    lift_success_acc = None
    if pred_grasp_prob is not None and grasp_success_flag is not None:
        grasp_success_acc = float(np.mean((pred_grasp_prob >= 0.5) == (grasp_success_flag >= 0.5)))
    if pred_lift_prob is not None and lift_success_flag is not None:
        lift_success_acc = float(np.mean((pred_lift_prob >= 0.5) == (lift_success_flag >= 0.5)))

    outcome_acc = None
    outcome_separation = {}
    if pred_outcome_idx is not None and outcome_idx is not None:
        outcome_acc = float(np.mean(pred_outcome_idx == outcome_idx))
        for outcome_name in (
            "refined_success",
            "improved_but_failed",
            "no_improvement",
            "grasp_improved_but_failed",
            "grasp_no_improvement",
        ):
            mask = np.array([OUTCOME_TYPES[int(outcome_idx[i])] == outcome_name for i in range(len(outcome_idx))])
            if np.any(mask):
                outcome_separation[outcome_name] = {
                    "count": int(np.sum(mask)),
                    "pred_energy_mean": float(np.mean(pred_total[mask])),
                    "target_energy_mean": float(np.mean(target_total[mask])),
                    "pred_outcome_accuracy_on_class": float(
                        np.mean(pred_outcome_idx[mask] == outcome_idx[mask])
                    ),
                }

    demo_2_demo_3 = {}
    grasp_best_sources = {
        "demo_2": ["grasp_refined_success"],
        "demo_3": ["lift_failed_candidate", "grasp_improved_but_failed"],
    }
    for demo_key, sources in grasp_best_sources.items():
        idx = None
        for src in sources:
            matches = [
                i
                for i, m in enumerate(meta_records)
                if m.get("demo_key") == demo_key and m.get("source") == src
            ]
            if matches:
                idx = matches[0]
                break
        if idx is None:
            matches = [
                i
                for i, m in enumerate(meta_records)
                if m.get("demo_key") == demo_key
                and m.get("outcome") in ("refined_success", "grasp_improved_but_failed")
            ]
            if matches:
                idx = matches[0]
        if idx is not None:
            demo_2_demo_3[demo_key] = {
                "sample_idx": idx,
                "source": meta_records[idx].get("source"),
                "target_failure_type": meta_records[idx].get("failure_type"),
                "target_outcome": meta_records[idx].get("outcome"),
                "pred_failure_type": FAILURE_TYPES[int(pred_failure_idx[idx])],
                "pred_outcome": OUTCOME_TYPES[int(pred_outcome_idx[idx])] if pred_outcome_idx is not None else None,
                "target_E_total": float(target_total[idx]),
                "pred_E_total": float(pred_total[idx]),
                "pred_grasp_success_prob": float(pred_grasp_prob[idx]) if pred_grasp_prob is not None else None,
                "pred_lift_success_prob": float(pred_lift_prob[idx]) if pred_lift_prob is not None else None,
                "target_grasp_success": float(grasp_success_flag[idx]) if grasp_success_flag is not None else None,
                "target_lift_success": float(lift_success_flag[idx]) if lift_success_flag is not None else None,
            }

    demo_2_3_distinguishable = False
    if "demo_2" in demo_2_demo_3 and "demo_3" in demo_2_demo_3:
        d2 = demo_2_demo_3["demo_2"]
        d3 = demo_2_demo_3["demo_3"]
        d2["target_success"] = float(success_flag[d2["sample_idx"]])
        d3["target_success"] = float(success_flag[d3["sample_idx"]])
        demo_2_3_distinguishable = (
            d2["target_outcome"] == "refined_success"
            and d3["target_outcome"] == "grasp_improved_but_failed"
            and d3["target_failure_type"] == "lift_failed"
            and d2["target_success"] != d3["target_success"]
            and d2["target_E_total"] < d3["target_E_total"] - 1.0
        )

    report = {
        "model_version": model_version,
        "dataset": str(dataset_path),
        "model": str(model_path),
        "num_samples": int(len(target_total)),
        "feature_dim": int(bundle["features"].shape[1]),
        "failure_type_distribution": dict(failure_type_counts),
        "outcome_distribution": dict(outcome_counts),
        "E_total_mae": mae,
        "E_total_rmse": rmse,
        "pearson_E_total": _safe_corr(pearsonr, pred_total, target_total),
        "spearman_E_total": _safe_corr(spearmanr, pred_total, target_total),
        "success_classification_accuracy": success_acc,
        "failure_type_accuracy": failure_acc,
        "grasp_failed_accuracy": grasp_acc,
        "lift_failed_accuracy": lift_failed_acc,
        "grasp_success_accuracy": grasp_success_acc,
        "lift_success_accuracy": lift_success_acc,
        "outcome_classification_accuracy": outcome_acc,
        "outcome_separation": outcome_separation,
        "demo_2_vs_demo_3_grasp_best": demo_2_demo_3,
        "top_k_low_energy_contains_success": {
            "top_1": top_k_success_rate(pred_total, success_flag, 1),
            "top_3": top_k_success_rate(pred_total, success_flag, 3),
            "top_5": top_k_success_rate(pred_total, success_flag, 5),
        },
        "top_k_low_energy_contains_refined_success": {
            "top_1": top_k_refined_success_rate(
                pred_total, refined_success_flag if refined_success_flag is not None else success_flag, 1
            ),
            "top_3": top_k_refined_success_rate(
                pred_total, refined_success_flag if refined_success_flag is not None else success_flag, 3
            ),
            "top_5": top_k_refined_success_rate(
                pred_total, refined_success_flag if refined_success_flag is not None else success_flag, 5
            ),
        },
        "predicted_energy_success_separation": energy_separation(pred_total, success_flag),
        "target_energy_success_separation": energy_separation(target_total, success_flag),
        "acceptance_checks": {
            "positive_correlation": (_safe_corr(pearsonr, pred_total, target_total) or -1.0) > 0,
            "success_failed_separable": (
                energy_separation(pred_total, success_flag)["mean_gap_success_minus_failed"] or -1.0
            )
            < 0,
            "contains_grasp_failed_samples": bool(np.any(grasp_mask)),
            "contains_lift_failed_samples": bool(np.any(lift_mask)),
            "demo_2_demo_3_distinguishable": demo_2_3_distinguishable,
        },
    }
    return (
        report,
        pred_total,
        pred_success_prob,
        pred_failure_idx,
        pred_outcome_idx,
        pred_grasp_prob,
        pred_lift_prob,
        meta_records,
    )


def _compact_report(report: dict) -> dict:
    return {
        "num_samples": report["num_samples"],
        "feature_dim": report.get("feature_dim"),
        "E_total_mae": report["E_total_mae"],
        "E_total_rmse": report["E_total_rmse"],
        "pearson_E_total": report["pearson_E_total"],
        "spearman_E_total": report.get("spearman_E_total"),
        "success_classification_accuracy": report["success_classification_accuracy"],
        "failure_type_accuracy": report["failure_type_accuracy"],
        "outcome_classification_accuracy": report.get("outcome_classification_accuracy"),
        "grasp_failed_accuracy": report.get("grasp_failed_accuracy"),
        "lift_failed_accuracy": report.get("lift_failed_accuracy"),
        "grasp_success_accuracy": report.get("grasp_success_accuracy"),
        "lift_success_accuracy": report.get("lift_success_accuracy"),
        "failure_type_distribution": report["failure_type_distribution"],
        "outcome_distribution": report.get("outcome_distribution"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate V1-A / V1-B / V1-C residual energy model")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_V1C)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_V1C)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_V1C)
    parser.add_argument("--model-version", choices=["v1a", "v1b", "v1c"], default="v1c")
    parser.add_argument("--compare-v1a-dir", type=Path, default=DEFAULT_OUTPUT_V1A)
    parser.add_argument("--compare-v1b-dir", type=Path, default=DEFAULT_OUTPUT_V1B)
    args = parser.parse_args()

    defaults = {
        "v1a": (DEFAULT_DATASET_V1A, DEFAULT_MODEL_V1A, DEFAULT_OUTPUT_V1A),
        "v1b": (DEFAULT_DATASET_V1B, DEFAULT_MODEL_V1B, DEFAULT_OUTPUT_V1B),
        "v1c": (DEFAULT_DATASET_V1C, DEFAULT_MODEL_V1C, DEFAULT_OUTPUT_V1C),
    }
    default_dataset, default_model, default_output = defaults[args.model_version]
    if args.dataset == DEFAULT_DATASET_V1C and args.model_version != "v1c":
        args.dataset = default_dataset
    if args.model == DEFAULT_MODEL_V1C and args.model_version != "v1c":
        args.model = default_model
    if args.output_dir == DEFAULT_OUTPUT_V1C and args.model_version != "v1c":
        args.output_dir = default_output

    version_label = (
        f"{args.model_version.upper()}_grasp_aware_residual_energy_mlp"
        if args.model_version == "v1c"
        else f"{args.model_version.upper()}_residual_energy_mlp"
    )
    report, pred_total, pred_success_prob, pred_failure_idx, pred_outcome_idx, pred_grasp_prob, pred_lift_prob, meta_records = evaluate_model(
        dataset_path=args.dataset,
        model_path=args.model,
        model_version=version_label,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "evaluation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    bundle = load_npz_dataset(args.dataset)
    target_total = bundle["target_E_total"]
    success_flag = bundle["success_flag"]
    failure_type_idx = bundle["failure_type_idx"]
    outcome_idx = bundle.get("outcome_idx")
    grasp_success_flag = bundle.get("grasp_success_flag")
    lift_success_flag = bundle.get("lift_success_flag")

    csv_path = args.output_dir / "predictions.csv"
    fieldnames = [
        "sample_idx",
        "source",
        "demo_key",
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
        "nut_lift_delta",
        "nut_displacement_after_grasp",
        "improvement_ratio",
        "refined_success_flag",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(target_total)):
            meta_i = meta_records[i] if i < len(meta_records) else {}
            writer.writerow(
                {
                    "sample_idx": i,
                    "source": meta_i.get("source", ""),
                    "demo_key": meta_i.get("demo_key", ""),
                    "target_E_total": float(target_total[i]),
                    "pred_E_total": float(pred_total[i]),
                    "target_success": int(success_flag[i]),
                    "pred_success_prob": float(pred_success_prob[i]),
                    "target_failure_type": FAILURE_TYPES[int(failure_type_idx[i])],
                    "pred_failure_type": FAILURE_TYPES[int(pred_failure_idx[i])],
                    "target_outcome": OUTCOME_TYPES[int(outcome_idx[i])] if outcome_idx is not None else "",
                    "pred_outcome": OUTCOME_TYPES[int(pred_outcome_idx[i])]
                    if pred_outcome_idx is not None
                    else "",
                    "target_grasp_success": int(grasp_success_flag[i]) if grasp_success_flag is not None else "",
                    "pred_grasp_success_prob": float(pred_grasp_prob[i]) if pred_grasp_prob is not None else "",
                    "target_lift_success": int(lift_success_flag[i]) if lift_success_flag is not None else "",
                    "pred_lift_success_prob": float(pred_lift_prob[i]) if pred_lift_prob is not None else "",
                    "nut_lift_delta": float(bundle.get("nut_lift_delta", np.zeros(len(target_total)))[i])
                    if "nut_lift_delta" in bundle
                    else "",
                    "nut_displacement_after_grasp": float(
                        bundle.get("nut_displacement_after_grasp", np.zeros(len(target_total)))[i]
                    )
                    if "nut_displacement_after_grasp" in bundle
                    else "",
                    "improvement_ratio": float(bundle.get("improvement_ratio", np.zeros(len(target_total)))[i])
                    if "improvement_ratio" in bundle
                    else "",
                    "refined_success_flag": int(bundle.get("refined_success_flag", np.zeros(len(target_total)))[i])
                    if "refined_success_flag" in bundle
                    else "",
                }
            )

    comparison = {"v1b": None, "v1c": _compact_report(report), "dataset_size_v1b": 112}
    v1b_dataset = args.compare_v1b_dir / "training_dataset.npz"
    v1b_model = args.compare_v1b_dir / "model.pt"
    if v1b_dataset.exists() and v1b_model.exists():
        v1b_report, _, _, _, _, _, _, _ = evaluate_model(
            dataset_path=v1b_dataset,
            model_path=v1b_model,
            model_version="V1-B_residual_energy_mlp",
        )
        comparison["v1b"] = _compact_report(v1b_report)
        comparison["dataset_size_v1b"] = v1b_report["num_samples"]
    comparison["v1c"] = {
        **_compact_report(report),
        "dataset_size_increase": report["num_samples"] - comparison["dataset_size_v1b"],
        "demo_2_vs_demo_3_grasp_best": report.get("demo_2_vs_demo_3_grasp_best"),
        "acceptance_checks": report.get("acceptance_checks"),
    }

    if args.model_version == "v1b":
        comparison_v1a = {"v1a": None, "v1b": _compact_report(report), "dataset_size_v1a": 56}
        v1a_dataset = args.compare_v1a_dir / "training_dataset.npz"
        v1a_model = args.compare_v1a_dir / "model.pt"
        if v1a_dataset.exists() and v1a_model.exists():
            v1a_report, _, _, _, _, _, _, _ = evaluate_model(
                dataset_path=v1a_dataset,
                model_path=v1a_model,
                model_version="V1-A_residual_energy_mlp",
            )
            comparison_v1a["v1a"] = _compact_report(v1a_report)
            comparison_v1a["dataset_size_v1a"] = v1a_report["num_samples"]
        comparison_v1a["v1b"] = {
            **_compact_report(report),
            "dataset_size_increase": report["num_samples"] - comparison_v1a["dataset_size_v1a"],
        }
        (args.output_dir / "v1a_vs_v1b_comparison.json").write_text(
            json.dumps(comparison_v1a, indent=2),
            encoding="utf-8",
        )
    else:
        (args.output_dir / "v1b_vs_v1c_comparison.json").write_text(
            json.dumps(comparison, indent=2),
            encoding="utf-8",
        )

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
